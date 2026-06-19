"""src/strat/fleet_search.py -- Fleet-of-agents search: multi-TF features + TI x Chimera archetypes.

LANES (user mandate 2026-06-20):
  (a) MULTI-TIMEFRAME: load 4h bars, build finer-TF mom/breakout/vol features, test orthogonality to 1d agents.
  (b) TI x 3-CHIMERA archetype: enumerate TI x {vpin,ofi,dev,fdclose,dvol,...} cross products + test vs TI-only.

DATA WALL: DEV <= 2024-05-15 ONLY. All evals on random DEV slices.

Run: python -m strat.fleet_search
No emoji. No git commits.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strat.fleet_lab import (
    load_wide, invoke, fleet_invoke, slice_dates, DEV_END, COST, CHIM_DIR, _rsi
)

N_SLICES = 200   # enough for stable means
HOLD = 7
K = 5

# ============================================================
# LANE A: Load 4h chimera parquets, build finer-TF features
# ============================================================

CHIM_4H = CHIM_DIR.parent / "4h"
DEV_END_MS = int(pd.Timestamp(DEV_END).value // 10**6)


def load_4h_wide(n=50, start="2019-01-01"):
    """Load 4h chimera data, aggregate per DAY for cross-sectional alignment with 1d lab.
    Returns feature dict aligned to the SAME daily index as load_wide().
    Features are aggregates of intraday bars (mom of last 4h vs 6 bars ago etc.) -- causal.
    """
    s_ms = pd.Timestamp(start).value // 10**6
    rows = []
    for f in sorted(glob.glob(str(CHIM_4H / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            cols_want = ["timestamp", "close", "high", "low", "open",
                         "volume_usd", "buy_vol", "sell_vol",
                         "norm_vpin", "norm_deviation", "norm_fd_close",
                         "norm_flow_imbalance", "norm_funding"]
            avail = pl.read_parquet_schema(f)
            cols = [c for c in cols_want if c in avail]
            df = pl.read_parquet(f, columns=cols).sort("timestamp")
        except Exception:
            continue
        ms = df["timestamp"].to_numpy()
        mask = (ms >= s_ms) & (ms < DEV_END_MS)
        if mask.sum() < 400:
            continue
        d = df.filter(pl.Series(mask))
        # Convert to pandas with daily-floored dates for aggregation
        ts = pd.to_datetime(d["timestamp"].to_numpy(), unit="ms").normalize()
        pdf = pd.DataFrame({c: d[c].to_numpy() for c in cols if c != "timestamp"}, index=ts)
        rows.append((sym, pdf, int(mask.sum())))
    rows = sorted(rows, key=lambda r: -r[2])[:n]
    syms = [r[0] for r in rows]

    # Build daily-aligned features from intraday bars
    # Strategy: for each asset, resample 4h bars to daily, compute features on the intraday sequence
    # Features use only bars available by end of day (causal) -- we use last-bar-of-day close etc.

    F4h = {}

    def _build(col_fn, feat_name):
        """col_fn(pdf) -> daily Series for each asset."""
        frames = {}
        for sym, pdf, _ in rows:
            try:
                s = col_fn(pdf)
                frames[sym] = s
            except Exception:
                pass
        df = pd.DataFrame(frames).sort_index()
        F4h[feat_name] = df

    # Helper: last bar of day
    def _last(pdf, col):
        return pdf[col].resample("D").last() if col in pdf.columns else None

    # mom_4h_6bars: % change last 4h bar vs 6 bars ago (approx 24h for 4h data = 6 bars)
    # This is finer-grained than daily mom7
    def _mom_intra(pdf, n_bars):
        c = pdf["close"] if "close" in pdf.columns else None
        if c is None: return None
        # compute rolling n_bars momentum on the 4h series, take last of day
        m = c / c.shift(n_bars) - 1
        return m.resample("D").last()

    def _brk_intra(pdf, n_bars):
        """Breakout: close vs n_bars high max (4h resolution)."""
        c = pdf["close"] if "close" in pdf.columns else None
        h = pdf["high"] if "high" in pdf.columns else None
        if c is None or h is None: return None
        rolling_max = h.rolling(n_bars, min_periods=n_bars).max().shift(1)
        brk = c / (rolling_max + 1e-12) - 1
        return brk.resample("D").last()

    def _vol_intra(pdf, fast=12, slow=48):
        """Vol expansion: std of last `fast` 4h bars vs `slow`."""
        c = pdf["close"] if "close" in pdf.columns else None
        if c is None: return None
        r = c.pct_change()
        ve = r.rolling(fast).std() / (r.rolling(slow).std() + 1e-12)
        return ve.resample("D").last()

    def _rsi_intra(pdf, n=28):
        """RSI on 4h close (~7-day RSI at 4h resolution)."""
        c = pdf["close"] if "close" in pdf.columns else None
        if c is None: return None
        d = c.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
        rsi = 100 - 100 / (1 + up / (dn + 1e-12))
        return rsi.resample("D").last()

    def _accel_intra(pdf, n=6):
        """Momentum acceleration on 4h bars."""
        c = pdf["close"] if "close" in pdf.columns else None
        if c is None: return None
        m1 = c / c.shift(n) - 1
        m2 = c.shift(n) / c.shift(2 * n) - 1
        return (m1 - m2).resample("D").last()

    def _ofi_intra(pdf):
        """OFI from 4h buy/sell vol, last of day."""
        bv = pdf.get("buy_vol"); sv = pdf.get("sell_vol")
        if bv is None or sv is None: return None
        ofi = (bv - sv) / (bv + sv + 1e-9)
        return ofi.resample("D").last()

    def _chim_last(pdf, col):
        if col not in pdf.columns: return None
        return pdf[col].resample("D").last()

    # Build per-asset and assemble wide DataFrames
    feat_builders = {
        "4h_mom6":   lambda pdf: _mom_intra(pdf, 6),    # 24h momentum at 4h resolution
        "4h_mom12":  lambda pdf: _mom_intra(pdf, 12),   # 48h
        "4h_mom24":  lambda pdf: _mom_intra(pdf, 24),   # ~4d
        "4h_brk24":  lambda pdf: _brk_intra(pdf, 24),
        "4h_brk48":  lambda pdf: _brk_intra(pdf, 48),
        "4h_volexp": lambda pdf: _vol_intra(pdf, 12, 48),
        "4h_rsi28":  lambda pdf: _rsi_intra(pdf, 28),
        "4h_accel":  lambda pdf: _accel_intra(pdf, 6),
        "4h_ofi":    lambda pdf: _ofi_intra(pdf),
        "4h_vpin":   lambda pdf: _chim_last(pdf, "norm_vpin"),
        "4h_dev":    lambda pdf: _chim_last(pdf, "norm_deviation"),
        "4h_fdclose": lambda pdf: _chim_last(pdf, "norm_fd_close"),
        "4h_flow":   lambda pdf: _chim_last(pdf, "norm_flow_imbalance"),
        "4h_funding": lambda pdf: _chim_last(pdf, "norm_funding"),
    }

    built = {name: {} for name in feat_builders}
    for sym, pdf, _ in rows:
        for name, fn in feat_builders.items():
            try:
                s = fn(pdf)
                if s is not None:
                    built[name][sym] = s
            except Exception:
                pass

    F4 = {}
    for name, d in built.items():
        if d:
            F4[name] = pd.DataFrame(d).sort_index()

    return {"F4h": F4, "syms": syms}


def _align_feature(feat_df, ref_index, ref_cols):
    """Align a feature DataFrame to the reference daily index + columns of the 1d lab."""
    return feat_df.reindex(index=ref_index, columns=ref_cols)


def merge_lab_4h(lab, data4h):
    """Merge 4h features into the main lab F dict, aligned to lab's daily index."""
    ref_index = lab["C"].index
    ref_cols = lab["C"].columns
    for name, df in data4h["F4h"].items():
        aligned = _align_feature(df, ref_index, ref_cols)
        lab["F"][name] = aligned.shift(1)  # shift-1 for causality (same as 1d features)
    return lab


# ============================================================
# EVALUATION HARNESS
# ============================================================

def eval_agent(lab, feats, signs=None, n_slices=N_SLICES, hold=HOLD, K=K, seed=0):
    """Evaluate an agent over random DEV slices. Returns dict of stats."""
    ds = slice_dates(lab, n_slices, hold=hold, seed=seed)
    # EW reference
    C = lab["C"]
    ew_rois = []
    for d in ds:
        if d + hold >= len(C.index): continue
        vs = [C[s].iloc[d+hold]/C[s].iloc[d]-1 for s in C.columns
              if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+hold])]
        if vs: ew_rois.append(float(np.mean(vs)))
    ew_rois = np.array(ew_rois)

    rois = []
    for d in ds:
        r = invoke(lab, feats, d, hold, K, signs)
        if r is not None:
            rois.append(r)
    rois = np.array(rois)
    n = len(rois)
    if n < 10:
        return {"n": n, "mean": float("nan"), "profit_rate": float("nan"), "beat_ew": float("nan")}

    mean_roi = float(np.mean(rois)) * 100
    profit_rate = float(np.mean(rois > 0)) * 100
    beat_ew = float(np.mean(rois[:len(ew_rois)] > ew_rois[:n])) * 100 if len(ew_rois) > 0 else float("nan")
    return {
        "n": n,
        "mean": mean_roi,
        "profit_rate": profit_rate,
        "beat_ew": beat_ew,
    }


def eval_fleet(lab, fleet, n_slices=N_SLICES, hold=HOLD, seed=0):
    """Evaluate ensemble fleet."""
    ds = slice_dates(lab, n_slices, hold=hold, seed=seed)
    C = lab["C"]
    ew_rois = []
    for d in ds:
        if d + hold >= len(C.index): continue
        vs = [C[s].iloc[d+hold]/C[s].iloc[d]-1 for s in C.columns
              if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+hold])]
        if vs: ew_rois.append(float(np.mean(vs)))
    ew_rois = np.array(ew_rois)

    rois = []
    for d in ds:
        r = fleet_invoke(lab, fleet, d, hold)
        if r is not None:
            rois.append(r)
    rois = np.array(rois)
    n = len(rois)
    if n < 10:
        return {"n": n, "mean": float("nan"), "profit_rate": float("nan"), "beat_ew": float("nan")}
    mean_roi = float(np.mean(rois)) * 100
    profit_rate = float(np.mean(rois > 0)) * 100
    beat_ew = float(np.mean(rois[:min(n,len(ew_rois))] > ew_rois[:min(n,len(ew_rois))])) * 100
    return {"n": n, "mean": mean_roi, "profit_rate": profit_rate, "beat_ew": beat_ew}


def print_result(name, stats, width=48):
    m = stats["mean"]
    pr = stats["profit_rate"]
    bew = stats["beat_ew"]
    print(f"  {name:{width}}{pr:>7.0f}%{m:>8.2f}%{bew:>8.0f}%")


# ============================================================
# LANE A: Multi-TF orthogonality test
# ============================================================

def lane_a_multitf(lab):
    print("\n=== LANE A: Multi-Timeframe Features (4h features on daily grid) ===")
    print("  Building 4h feature library...")
    data4h = load_4h_wide(n=50)
    lab = merge_lab_4h(lab, data4h)
    avail_4h = [k for k in lab["F"] if k.startswith("4h_")]
    print(f"  Loaded {len(avail_4h)} 4h features: {avail_4h}")

    print(f"\n  SINGLE 4h FEATURE vs 1d equivalents ({N_SLICES} slices, K={K}, hold={HOLD}d):")
    print(f"  {'agent':{48}}{'profit%':>7}{'mean%':>8}{'beatEW%':>8}")

    # Baseline 1d features
    baselines = ["mom7", "mom14", "mom30", "rsi14", "brk14", "volexp", "accel", "ofi", "vpin"]
    for f in baselines:
        s = eval_agent(lab, [f])
        print_result(f"1d:{f}", s)

    # All 4h single features
    print()
    for f in avail_4h:
        s = eval_agent(lab, [f])
        print_result(f"4h:{f}", s)

    # Cross: best 1d + best 4h combinations
    print("\n  CROSS-TF COMBOS (1d TI + 4h TI):")
    combos = [
        ("mom14 + 4h_mom6",    ["mom14", "4h_mom6"]),
        ("mom14 + 4h_mom12",   ["mom14", "4h_mom12"]),
        ("mom14 + 4h_brk24",   ["mom14", "4h_brk24"]),
        ("mom14 + 4h_rsi28",   ["mom14", "4h_rsi28"]),
        ("mom14 + 4h_accel",   ["mom14", "4h_accel"]),
        ("brk14 + 4h_brk24",   ["brk14", "4h_brk24"]),
        ("mom14 + 4h_ofi",     ["mom14", "4h_ofi"]),
        ("mom14 + 4h_funding", ["mom14", "4h_funding"]),
        ("accel + 4h_accel",   ["accel", "4h_accel"]),
        ("rsi14 + 4h_rsi28",   ["rsi14", "4h_rsi28"]),
        ("volexp + 4h_volexp", ["volexp", "4h_volexp"]),
    ]
    for name, feats in combos:
        s = eval_agent(lab, feats)
        print_result(f"CROSS:{name}", s)

    # Pure 4h ensemble
    print("\n  4h-ONLY ENSEMBLE (all available 4h price features):")
    price_4h = [f for f in avail_4h if "vpin" not in f and "dev" not in f
                and "fdclose" not in f and "flow" not in f and "funding" not in f]
    s = eval_agent(lab, price_4h)
    print_result("4h price ensemble", s)

    # 4h chimera-only
    chim_4h = [f for f in avail_4h if any(x in f for x in ["vpin","dev","fdclose","flow","funding","ofi"])]
    if chim_4h:
        s = eval_agent(lab, chim_4h)
        print_result("4h chim ensemble", s)

    # Best cross-TF fleet
    print("\n  CROSS-TF FLEET (1d top agents + 4h agents, ensembled):")
    fleet_ab = [
        {"feats": ["mom14"], "K": K},
        {"feats": ["4h_mom6"], "K": K},
        {"feats": ["4h_brk24"], "K": K},
        {"feats": ["4h_accel"], "K": K},
    ]
    s = eval_fleet(lab, fleet_ab)
    print_result("fleet: 1d_mom14 + 4h_mom6 + brk24 + accel", s)

    fleet_full = [
        {"feats": ["mom14"], "K": K},
        {"feats": ["mom14", "4h_mom6"], "K": K},
        {"feats": ["brk14", "4h_brk24"], "K": K},
        {"feats": ["accel", "4h_accel"], "K": K},
        {"feats": ["rsi14", "4h_rsi28"], "K": K},
    ]
    s = eval_fleet(lab, fleet_full)
    print_result("fleet: cross-TF 5-agent ensemble", s)

    return lab


# ============================================================
# LANE B: TI x Chimera archetype search
# ============================================================

def lane_b_ti_x_chimera(lab):
    print("\n=== LANE B: TI x Chimera Archetype Search ===")
    print(f"  ({N_SLICES} slices, K={K}, hold={HOLD}d)")
    print(f"  {'agent':{48}}{'profit%':>7}{'mean%':>8}{'beatEW%':>8}")

    # Baseline: TI-only reference
    baselines_b = [
        ("1TI: mom14",               ["mom14"]),
        ("1TI: brk14",               ["brk14"]),
        ("1TI: accel",               ["accel"]),
        ("1TI: rsi14",               ["rsi14"]),
        ("1TI: rangepos",            ["rangepos"]),
        ("2TI: mom14+brk14",         ["mom14", "brk14"]),
        ("2TI: mom14+accel",         ["mom14", "accel"]),
        ("2TI: brk14+volexp",        ["brk14", "volexp"]),
    ]
    print("\n  --- TI-ONLY BASELINES ---")
    for name, feats in baselines_b:
        s = eval_agent(lab, feats)
        print_result(name, s)

    # 1d chimera features
    chim_feats_1d = ["vpin", "ofi", "dev", "fdclose", "dvol"]
    ti_feats = ["mom14", "brk14", "accel", "rsi14", "rangepos", "volexp"]

    print("\n  --- CHIMERA-ONLY (1d) ---")
    for c in chim_feats_1d:
        s = eval_agent(lab, [c])
        print_result(f"1d_chim: {c}", s)
    # all 1d chimera ensemble
    s = eval_agent(lab, chim_feats_1d)
    print_result("1d_chim: all ensemble", s)

    print("\n  --- TI x 1-CHIMERA (1 TI + 1 chimera feature) ---")
    for ti in ["mom14", "brk14", "accel"]:
        for c in chim_feats_1d:
            feats = [ti, c]
            s = eval_agent(lab, feats)
            print_result(f"TIxC1: {ti}+{c}", s)

    print("\n  --- TI x 2-CHIMERA ---")
    chim_pairs = [
        ("vpin", "ofi"),
        ("vpin", "dev"),
        ("ofi", "dev"),
        ("vpin", "fdclose"),
        ("ofi", "fdclose"),
        ("dev", "dvol"),
        ("vpin", "dvol"),
    ]
    for ti in ["mom14", "brk14"]:
        for c1, c2 in chim_pairs:
            feats = [ti, c1, c2]
            s = eval_agent(lab, feats)
            print_result(f"TIxC2: {ti}+{c1}+{c2}", s)

    print("\n  --- TI x 3-CHIMERA (user's explicit archetype) ---")
    chim_triples = [
        ("vpin", "ofi", "dev"),
        ("vpin", "ofi", "fdclose"),
        ("vpin", "dev", "dvol"),
        ("ofi", "dev", "fdclose"),
        ("vpin", "ofi", "dvol"),
        ("ofi", "fdclose", "dvol"),
    ]
    for ti in ["mom14", "brk14", "accel", "rsi14"]:
        for c1, c2, c3 in chim_triples:
            feats = [ti, c1, c2, c3]
            s = eval_agent(lab, feats)
            print_result(f"TIxC3: {ti}+{c1}+{c2}+{c3}", s)

    # 4h chimera features (added in lane A) -- TI x 4h chimera
    avail_4h = [k for k in lab["F"] if k.startswith("4h_")]
    chim_4h_names = [f for f in avail_4h if any(x in f for x in ["vpin","dev","fdclose","flow","funding","ofi"])]
    if chim_4h_names:
        print("\n  --- TI(1d) x 3-CHIMERA (mix 1d + 4h chimera) ---")
        for ti in ["mom14", "brk14"]:
            # top 4h chim combos
            for c3 in chim_4h_names[:4]:
                feats = [ti, "vpin", "ofi", c3]
                s = eval_agent(lab, feats)
                print_result(f"TIxC3_mix: {ti}+vpin+ofi+{c3}", s)

    # Extended: 4-5 feature combos (TI + full chimera)
    print("\n  --- TI + ALL CHIMERA (1d) ---")
    for ti in ["mom14", "brk14", "accel"]:
        feats = [ti] + chim_feats_1d
        s = eval_agent(lab, feats)
        print_result(f"TI+all_chim: {ti}", s)

    # FLEET: best TI agents + best TIxChimera agents
    print("\n  --- FLEET: TI-ONLY vs TIxCHIMERA ENSEMBLE ---")

    fleet_ti_only = [
        {"feats": ["mom14"], "K": K},
        {"feats": ["brk14"], "K": K},
        {"feats": ["accel"], "K": K},
        {"feats": ["mom14", "brk14"], "K": K},
        {"feats": ["mom14", "accel"], "K": K},
    ]
    s = eval_fleet(lab, fleet_ti_only)
    print_result("fleet: TI-only 5-agent", s)

    fleet_ti_chim = [
        {"feats": ["mom14"], "K": K},
        {"feats": ["mom14", "vpin", "ofi", "dev"], "K": K},
        {"feats": ["brk14", "vpin", "ofi", "dev"], "K": K},
        {"feats": ["accel", "vpin", "ofi"], "K": K},
        {"feats": ["mom14", "ofi", "fdclose"], "K": K},
    ]
    s = eval_fleet(lab, fleet_ti_chim)
    print_result("fleet: TIxChimera 5-agent", s)

    # If 4h available, add cross-TF x chimera super-fleet
    if avail_4h:
        fleet_super = [
            {"feats": ["mom14"], "K": K},
            {"feats": ["mom14", "vpin", "ofi", "dev"], "K": K},
            {"feats": ["brk14", "vpin", "ofi", "dev"], "K": K},
            {"feats": ["4h_mom6"], "K": K},
            {"feats": ["4h_brk24", "4h_vpin"], "K": K},
            {"feats": ["mom14", "4h_ofi"], "K": K},
            {"feats": ["accel", "4h_accel"], "K": K},
        ]
        s = eval_fleet(lab, fleet_super)
        print_result("fleet: cross-TF + chimera super-fleet (7)", s)


# ============================================================
# IMPORTANCE SCREEN (greedy-forward over all features)
# ============================================================

def importance_screen(lab, top_n=8):
    """Greedy-forward feature importance: start from best single feature, add one at a time."""
    print("\n=== IMPORTANCE SCREEN: Greedy-Forward Feature Selection ===")
    all_feats = list(lab["F"].keys())
    print(f"  Total features available: {len(all_feats)}")
    print(f"  Screening single features first ({len(all_feats)} evals)...")

    single_scores = []
    for f in all_feats:
        s = eval_agent(lab, [f], n_slices=150, seed=42)
        single_scores.append((f, s["mean"] if not np.isnan(s["mean"]) else -99.0))

    single_scores.sort(key=lambda x: -x[1])
    print(f"  {'feature':{30}}{'mean%':>8}")
    for f, m in single_scores[:15]:
        print(f"  {f:{30}}{m:>8.2f}%")

    # Greedy forward from best single
    selected = [single_scores[0][0]]
    remaining = [f for f, _ in single_scores[1:]]
    print(f"\n  GREEDY FORWARD (starting from '{selected[0]}'):")
    for step in range(min(top_n - 1, 6)):
        best_f = None; best_m = -99.0
        for f in remaining[:20]:  # limit to top-20 remaining for speed
            cands = selected + [f]
            s = eval_agent(lab, cands, n_slices=100, seed=42)
            m = s["mean"] if not np.isnan(s["mean"]) else -99.0
            if m > best_m:
                best_m = m; best_f = f
        if best_f is None: break
        selected.append(best_f)
        remaining.remove(best_f)
        print(f"    step {step+2}: add '{best_f}' -> set={selected} mean={best_m:.2f}%")

    print(f"\n  Final greedy set: {selected}")
    s = eval_agent(lab, selected, n_slices=N_SLICES, seed=0)
    print(f"  Full eval ({N_SLICES} slices):")
    print_result(f"greedy({len(selected)} feats)", s)
    return selected


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("[fleet_search] Loading 1d DEV lab (u50, <= 2024-05-15)...")
    lab = load_wide(n=50)
    C = lab["C"]
    print(f"  loaded {len(lab['syms'])} assets, {len(C.index)} dates ({C.index.min().date()} -> {C.index.max().date()})")
    print(f"  1d features: {list(lab['F'].keys())}")

    # EW baseline
    ds = slice_dates(lab, N_SLICES, hold=HOLD, seed=0)
    ew_rois = []
    for d in ds:
        if d + HOLD >= len(C.index): continue
        vs = [C[s].iloc[d+HOLD]/C[s].iloc[d]-1 for s in C.columns
              if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+HOLD])]
        if vs: ew_rois.append(float(np.mean(vs)))
    ew = np.array(ew_rois)
    print(f"  EW ref ({len(ew)} slices): mean={100*ew.mean():.2f}% profit={100*np.mean(ew>0):.0f}%")

    print(f"\n  {'AGENT':{48}}{'profit%':>7}{'mean%':>8}{'beatEW%':>8}")

    # --- LANE A: multi-TF ---
    lab = lane_a_multitf(lab)

    # --- LANE B: TI x Chimera ---
    lane_b_ti_x_chimera(lab)

    # --- IMPORTANCE SCREEN ---
    best_set = importance_screen(lab)

    # --- FINAL CROSS-SEED CHECK on top greedy set ---
    print("\n=== CROSS-SEED ROBUSTNESS (greedy set, 3 seeds) ===")
    print(f"  Greedy set: {best_set}")
    for seed in [0, 1, 2]:
        s = eval_agent(lab, best_set, n_slices=N_SLICES, seed=seed)
        print_result(f"greedy seed={seed}", s)

    print(f"\n[fleet_search] Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
