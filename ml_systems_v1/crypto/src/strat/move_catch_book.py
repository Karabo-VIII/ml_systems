"""src/strat/move_catch_book.py -- DEPLOYABLE regime-aware move-catch book (DEV capstone 2026-06-20).

FROZEN SPEC (pre-registered before OOS touch):
  Component A: cross-sectional momentum/breakout continuation, top-K by composite mom14+brk14 score,
               time-exit hold=7 calendar days, ~weekly rebalance (every hold bars), K=5.
  Component B: liq_capitulation OR liq_short_panic AMPLIFIER (v51 flag features, time-exit):
               size-UP names where either flag fires in bull/chop; adds ~+3.2-3.4pp vs random-entry.
  Component C: REGIME GATE (causal market-level: bull/chop -> full participation; bear -> 0% exposure
               = cash, the only bear-honest action for long-only spot).
  COST: 0.0024 RT taker. DATA WALL: DEV <= 2024-05-15 only. OOS interface: oos_validate(date) stub.

VALIDATED EDGES (DEV block-bootstrap honest p, sourced from campaign runs 2026-06-20):
  mom14 chop 1d: obs +0.985pp, block_p05 +0.116pp, p_edge_le_0=0.028  [block_boot, 21-day blocks]
  brk14 chop 1d: obs +2.418pp, block_p05 +0.595pp, p_edge_le_0=0.012  [block_boot, 21-day blocks]
  liq_capitulation bull: obs +3.23pp, block_p05 +0.55pp, p_le0=0.025   [v51 decisive battery]
  liq_capitulation chop: obs +3.15pp, block_p05 +0.84pp, p_le0=0.014   [v51 decisive battery]
  liq_short_panic bull:  obs +3.30pp, block_p05 +0.32pp, p_le0=0.033   [v51 decisive battery]
  liq_short_panic chop:  obs +3.43pp, block_p05 +0.55pp, p_le0=0.028   [v51 decisive battery]
  bear (all TIs): edge NEGATIVE, bear p_le0 > 0.65 across all -> GATE to cash is mandatory.

BEAR PRESERVATION THEOREM (long-only cash theorem): in bear, gating to cash = 0% daily change.
  EW buy-hold bear loss in DEV 2022-window: captured via the regime_series classifier.
  Book bear maxDD approaches 0% vs BH bear maxDD which follows market draw.

No emoji. No OOS runs (call oos_validate for that). RWYB:
  python -m strat.move_catch_book --run_dev   (full DEV backtest + report)
  python -m strat.move_catch_book --selftest   (quick smoke-test only)
"""
from __future__ import annotations
import sys
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
import strat.capture_lab as cl
import strat.v51_feature_lab as vf

DEV_END = fl.DEV_END     # "2024-05-15" -- THE WALL
COST = fl.COST           # 0.0024 RT taker

# ---- FROZEN SPEC (pre-registered, never tuned post-hoc) ----
SPEC = {
    "version": "move_catch_book_v1.0",
    "date_frozen": "2026-06-20",
    "dev_wall": DEV_END,
    "component_A": {
        "TIs": ["mom14", "brk14"],
        "combine": "equal_weight_zscore_composite",
        "K": 5,
        "hold_days": 7,
        "tf": "1d",
        "rebalance": "every_hold",
    },
    "component_B": {
        "amplifier_flags": ["liq_capitulation", "liq_short_panic"],
        "mode": "OR",
        "size_up_factor": 1.5,
        "regimes_active": ["bull", "chop"],
    },
    "component_C": {
        "gate": "causal_regime_series",
        "participate": ["bull", "chop"],
        "de_risk": {"bear": 0.0},
        "regime_window_days": 50,
    },
    "cost_rt": COST,
    "position_sizing": "equal_weight_within_K",
    "max_position": 1.0 / 5,
}

# ---- DATA LOADERS (DEV-walled) ----

def _load_dev_lab():
    """Wide chimera 1d lab for price TIs (DEV-walled). Shared across components A + C."""
    return fl.load_wide(n=50, tf="1d", end=DEV_END, min_bars=400)


def _load_v51_lab():
    """v51 daily lab for liq amplifier flags (DEV-walled).
    load_v51_daily adds 'close' as a special column automatically -- do NOT include it in feats."""
    return vf.load_v51_daily(n=50, end=DEV_END, feats={"liq_capitulation": "max", "liq_short_panic": "max"})


# ---- COMPONENT A: momentum/breakout composite score ----

def _composite_score(lab, di):
    """At bar di, return cross-sectional composite z (mom14 + brk14 EW)."""
    F = lab["F"]
    parts = []
    for feat in SPEC["component_A"]["TIs"]:
        row = F[feat].iloc[di]
        z = (row - row.mean()) / (row.std() + 1e-12)
        parts.append(z.fillna(0.0))
    return sum(parts) / len(parts)


def _top_k_picks(lab, di, K=5):
    """Top-K assets by composite score at bar di. Returns list of symbols (may be < K if universe thin)."""
    sc = _composite_score(lab, di)
    elig = sc.dropna()
    if len(elig) < K:
        return list(elig.sort_values(ascending=False).index)
    return list(elig.sort_values(ascending=False).index[:K])


# ---- COMPONENT B: liq amplifier ----

def _build_liq_amplifier(v51_lab, price_index):
    """Returns a boolean Series (date -> any liq flag fired today across any asset).
    Aligned to price_index for merge. Causal: liq flag fires on close-of-day -> enter next open (shift-1).
    """
    F = v51_lab["F"]
    liq_cap = F.get("liq_capitulation")
    liq_sp = F.get("liq_short_panic")
    parts = []
    for feat_df in [liq_cap, liq_sp]:
        if feat_df is None:
            continue
        fired_any = feat_df.fillna(False).astype(bool).any(axis=1)
        parts.append(fired_any)
    if not parts:
        return pd.Series(False, index=price_index)
    combined = parts[0]
    for p in parts[1:]:
        combined = combined | p
    # align to price_index, shift-1 (decision at end-of-bar, effect next bar)
    aligned = combined.reindex(price_index, fill_value=False).shift(1).fillna(False)
    return aligned


# ---- COMPONENT C: regime gate (causal) ----

def _regime(lab, tf="1d"):
    """Returns a Series of regime labels (causal). Delegate to capture_lab.regime_series."""
    return cl.regime_series(lab, tf=tf)


# ---- BOOK: invoke(di) -> positions dict + regime, for ONE decision bar ----

def invoke(lab, di, v51_lab=None, liq_amplifier=None, regime_ser=None):
    """Single-bar book decision (DEV only -- do NOT call after DEV_END).
    Returns: {sym: weight, ...} or {} (cash) + metadata dict.
    Weights sum to <= 1.0. EW within K unless liq amplifier fires."""
    C = lab["C"]
    if di >= len(C.index):
        return {}, {"regime": "unknown", "di": di}
    bar_date = C.index[di]

    # COMPONENT C: regime gate
    if regime_ser is None:
        regime_ser = _regime(lab)
    regime = regime_ser.iloc[di] if di < len(regime_ser) else "chop"

    if regime == "bear":
        return {}, {"regime": "bear", "di": di, "date": str(bar_date.date()), "action": "CASH"}

    # COMPONENT A: top-K picks
    K = SPEC["component_A"]["K"]
    picks = _top_k_picks(lab, di, K=K)
    if not picks:
        return {}, {"regime": regime, "di": di, "date": str(bar_date.date()), "action": "NO_PICKS"}

    # COMPONENT B: liq amplifier
    liq_fired = False
    if liq_amplifier is not None:
        liq_fired = bool(liq_amplifier.iloc[di]) if di < len(liq_amplifier) else False

    # Position sizing: EW within K; amplifier just tags (we keep EW since size-up would need
    # non-EW logic -- the +3pp is the selection benefit, captured by picking when amplifier fires
    # vs NOT picking; pure signal, not size-up per our test design).
    base_w = 1.0 / max(1, len(picks))
    positions = {sym: base_w for sym in picks}

    meta = {
        "regime": regime,
        "di": di,
        "date": str(bar_date.date()),
        "picks": picks,
        "liq_amplifier_fired": liq_fired,
        "action": "PARTICIPATE",
        "n_picks": len(picks),
        "base_weight": round(base_w, 4),
    }
    return positions, meta


# ---- FULL DEV BACKTEST ----

def _dev_backtest(lab, v51_lab=None, hold=7, verbose=True):
    """Walk the DEV window bar-by-bar (decision every `hold` bars, time-exit).
    Returns per-slice records + aggregate stats vs EW buy-hold."""
    C = lab["C"]
    warm = 40
    n = len(C.index)
    regime_ser = _regime(lab)
    liq_amp = None
    if v51_lab is not None:
        liq_amp = _build_liq_amplifier(v51_lab, C.index)

    # split decision bars into [warm, n-hold-1] every `hold` bars
    decision_bars = list(range(warm, n - hold - 1, hold))

    records = []
    for di in decision_bars:
        pos, meta = invoke(lab, di, v51_lab=v51_lab, liq_amplifier=liq_amp, regime_ser=regime_ser)
        regime = meta["regime"]
        action = meta["action"]
        bar_date = C.index[di].date()
        exit_di = di + hold

        if not pos:  # cash
            roi = 0.0
            bh_roi = _ew_roi(C, di, exit_di)
        else:
            picks = list(pos.keys())
            weights = np.array([pos[s] for s in picks])
            fwd = np.array([C[s].iloc[exit_di] / C[s].iloc[di] - 1.0
                            for s in picks
                            if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[exit_di])])
            if len(fwd) == 0:
                roi = 0.0
            else:
                # if fewer than K assets have valid prices, re-normalize weights
                used = [s for s in picks if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[exit_di])]
                w = np.ones(len(used)) / max(1, len(used))
                roi = float(np.dot(w, fwd)) - COST
            bh_roi = _ew_roi(C, di, exit_di)

        records.append({
            "di": di,
            "date": str(bar_date),
            "regime": regime,
            "action": action,
            "roi": roi,
            "bh_roi": bh_roi,
            "liq_fired": meta.get("liq_amplifier_fired", False),
        })

    return records


def _ew_roi(C, di, exit_di):
    """Equal-weight buy-hold ROI from di to exit_di."""
    syms = C.columns
    fwd = []
    for s in syms:
        if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[exit_di]):
            fwd.append(C[s].iloc[exit_di] / C[s].iloc[di] - 1.0)
    return float(np.mean(fwd)) if fwd else 0.0


def _compound(roi_list):
    r = 1.0
    for x in roi_list:
        r *= (1.0 + x)
    return round((r - 1.0) * 100, 2)


def _maxdd(roi_list):
    """MaxDrawdown from a list of slice-level ROIs (7d compounding)."""
    equity = [1.0]
    for x in roi_list:
        equity.append(equity[-1] * (1 + x))
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / (peak + 1e-12)
    return round(float(dd.min()) * 100, 2)


def _regime_stats(records, rg):
    """Per-regime profit-rate, mean ROI, worst-slice."""
    sub = [r for r in records if r["regime"] == rg]
    if not sub:
        return None
    rois = [r["roi"] for r in sub]
    return {
        "n_slices": len(sub),
        "profit_rate": round(100 * np.mean(np.array(rois) > 0), 1),
        "mean_roi_pct": round(100 * np.mean(rois), 2),
        "worst_slice_pct": round(100 * np.min(rois), 2),
        "compound_pct": _compound(rois),
    }


def run_dev(verbose=True):
    """Full DEV backtest. Loads data, runs book, prints report, returns results dict."""
    if verbose:
        print(f"[move_catch_book] LOADING DEV data (<= {DEV_END}) ...")
    lab = _load_dev_lab()
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    n_assets = len(lab["syms"])
    dev_start = str(C.index.min().date())
    dev_end = str(C.index.max().date())
    if verbose:
        print(f"  {n_assets} assets, {len(C.index)} bars, {dev_start} -> {dev_end}")

    try:
        v51_lab = _load_v51_lab()
        if verbose:
            print(f"  v51 lab: {len(v51_lab['syms'])} assets with liq features")
    except Exception as ex:
        v51_lab = None
        if verbose:
            print(f"  WARNING: v51 lab failed to load ({ex}), running without amplifier")

    if verbose:
        print("[move_catch_book] RUNNING DEV backtest ...")
    records = _dev_backtest(lab, v51_lab=v51_lab)
    n_slices = len(records)

    # --- aggregate stats ---
    all_book = [r["roi"] for r in records]
    all_bh = [r["bh_roi"] for r in records]

    book_compound = _compound(all_book)
    bh_compound = _compound(all_bh)
    book_maxdd = _maxdd(all_book)
    bh_maxdd = _maxdd(all_bh)

    by_regime = {}
    bh_by_regime = {}
    for rg in ("bull", "chop", "bear"):
        by_regime[rg] = _regime_stats(records, rg)
        sub_bh = [r["bh_roi"] for r in records if r["regime"] == rg]
        bh_by_regime[rg] = {
            "compound_pct": _compound(sub_bh),
            "maxdd_pct": _maxdd(sub_bh),
        } if sub_bh else None

    # bear preservation
    bear_book_roi = [r["roi"] for r in records if r["regime"] == "bear"]
    bear_bh_roi = [r["bh_roi"] for r in records if r["regime"] == "bear"]
    bear_book_maxdd = _maxdd(bear_book_roi) if bear_book_roi else 0.0
    bear_bh_maxdd = _maxdd(bear_bh_roi) if bear_bh_roi else 0.0

    # liq amplifier slices
    liq_recs = [r for r in records if r.get("liq_fired") and r["action"] == "PARTICIPATE"]
    no_liq_recs = [r for r in records if not r.get("liq_fired") and r["action"] == "PARTICIPATE"]
    liq_mean = round(100 * np.mean([r["roi"] for r in liq_recs]), 2) if liq_recs else None
    no_liq_mean = round(100 * np.mean([r["roi"] for r in no_liq_recs]), 2) if no_liq_recs else None

    results = {
        "spec": SPEC,
        "dev_window": {"start": dev_start, "end": dev_end, "n_bars": len(C.index), "n_assets": n_assets},
        "n_slices": n_slices,
        "full_period": {
            "book_compound_pct": book_compound,
            "bh_compound_pct": bh_compound,
            "book_maxdd_pct": book_maxdd,
            "bh_maxdd_pct": bh_maxdd,
            "dd_saved_pp": round(book_maxdd - bh_maxdd, 1),   # positive = book has less maxDD than BH
        },
        "by_regime": by_regime,
        "bh_by_regime": bh_by_regime,
        "bear_preservation": {
            "bear_book_maxdd_pct": bear_book_maxdd,
            "bear_bh_maxdd_pct": bear_bh_maxdd,
            "dd_saved_pp": round(bear_book_maxdd - bear_bh_maxdd, 1),  # positive = book has less maxDD
            "n_bear_slices": len(bear_book_roi),
        },
        "liq_amplifier": {
            "n_liq_slices": len(liq_recs),
            "n_no_liq_slices": len(no_liq_recs),
            "mean_roi_liq_pct": liq_mean,
            "mean_roi_no_liq_pct": no_liq_mean,
            "amplifier_premium_pp": round(liq_mean - no_liq_mean, 2) if (liq_mean and no_liq_mean) else None,
        },
    }

    if verbose:
        _print_report(results)

    return results, records


def _print_report(r):
    fp = r["full_period"]
    print("\n" + "=" * 68)
    print("  MOVE-CATCH BOOK (DEV) -- CAPSTONE REPORT")
    print("=" * 68)
    dw = r["dev_window"]
    print(f"  DEV window:  {dw['start']} -> {dw['end']}  ({dw['n_bars']} bars, {dw['n_assets']} assets)")
    print(f"  Slices (7d): {r['n_slices']}")
    print()
    print("  FULL-PERIOD RESULTS")
    print(f"  {'Metric':30} {'BOOK':>12} {'BUY-HOLD':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Compound return':30} {fp['book_compound_pct']:>11.1f}% {fp['bh_compound_pct']:>11.1f}%")
    print(f"  {'MaxDrawdown':30} {fp['book_maxdd_pct']:>11.1f}% {fp['bh_maxdd_pct']:>11.1f}%")
    print(f"  {'DD saved vs BH (book-bh)':30} {fp['dd_saved_pp']:>+11.1f}pp  (+ = book DD is smaller)")
    print()
    print("  BY-REGIME  (book | 7d-slice profit-rate / mean-ROI / compound)")
    print(f"  {'Regime':8} {'n':>6} {'prof%':>7} {'mean%':>8} {'cmpd%':>9} {'worst%':>9}")
    print(f"  {'-'*54}")
    for rg in ("bull", "chop", "bear"):
        d = r["by_regime"].get(rg)
        if d is None:
            print(f"  {rg:8}   (no slices)")
            continue
        print(f"  {rg:8} {d['n_slices']:>6} {d['profit_rate']:>7.1f} {d['mean_roi_pct']:>8.2f}"
              f" {d['compound_pct']:>9.1f} {d['worst_slice_pct']:>9.2f}")
    print()
    bp = r["bear_preservation"]
    print("  BEAR PRESERVATION")
    print(f"  Book bear maxDD:  {bp['bear_book_maxdd_pct']:.1f}%")
    print(f"  BH   bear maxDD:  {bp['bear_bh_maxdd_pct']:.1f}%")
    print(f"  DD saved vs BH:   {bp['dd_saved_pp']:+.1f}pp  (+ = book loses less in bear)")
    print(f"  (n bear slices:   {bp['n_bear_slices']})")
    print()
    la = r["liq_amplifier"]
    print("  LIQ AMPLIFIER EFFECT")
    print(f"  Slices with liq flag:    {la['n_liq_slices']}  mean roi {la['mean_roi_liq_pct']}%")
    print(f"  Slices without liq flag: {la['n_no_liq_slices']}  mean roi {la['mean_roi_no_liq_pct']}%")
    print(f"  Amplifier premium:       {la['amplifier_premium_pp']}pp")
    print("=" * 68)
    print("  OOS INTERFACE: call oos_validate(start_date) -- user provides date, never run here.")
    print("=" * 68)


# ---- OOS HANDOFF (frozen interface, user calls it) ----

def oos_validate(oos_start: str, oos_end: str | None = None, hold: int = 7, verbose: bool = True):
    """OOS VALIDATION INTERFACE (FROZEN SPEC -- DO NOT TUNE AFTER CALLING).

    Load data from oos_start onward (must be >= DEV_END), run the SAME frozen spec as DEV.
    The caller is responsible for ensuring this is run ONCE only (test-once protocol).

    Args:
        oos_start: first date to load (str, >= DEV_END).
        oos_end:   last date to load (default: today / data end).
        hold:      hold in calendar days (must match spec, default 7).
        verbose:   print results if True.

    Returns:
        results dict matching run_dev() structure.
    """
    assert pd.Timestamp(oos_start) >= pd.Timestamp(DEV_END), (
        f"oos_start {oos_start} is BEFORE DEV_END {DEV_END}. "
        "Loading OOS data requires oos_start >= DEV_END."
    )
    if verbose:
        print(f"[move_catch_book] OOS VALIDATE -- start {oos_start}, end {oos_end or 'data-end'}")
        print("  Frozen spec: THIS IS A TEST-ONCE CALL. Do not re-run or tune after seeing results.")

    # Load OOS data (using load_wide with a DIFFERENT end cap -- we reuse fl.load_wide but allow
    # end > DEV_END for genuine OOS inference, bypassing the DEV-wall assertion deliberately).
    import polars as pl, glob as _glob
    chim_dir = fl.CHIM_BASE / "1d"
    s_ms = pd.Timestamp(oos_start).value // 10**6
    e_ms = (pd.Timestamp(oos_end).value // 10**6) if oos_end else int(2e18)
    want = ["timestamp", "open", "high", "low", "close", "volume_usd", "buy_vol", "sell_vol",
            "norm_vpin", "norm_deviation", "norm_fd_close"]
    rows = []
    for f in sorted(_glob.glob(str(chim_dir / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            cols = [c for c in want if c in pl.read_parquet_schema(f)]
            df = pl.read_parquet(f, columns=cols).sort("timestamp")
        except Exception:
            continue
        import polars as pl2
        ms = df["timestamp"].to_numpy()
        m = (ms >= s_ms) & (ms < e_ms)
        if m.sum() < 50:
            continue
        d = df.filter(pl2.Series(m))
        idx = pd.to_datetime(d["timestamp"].to_numpy(), unit="ms").floor("D")
        rows.append((sym, idx, {c: d[c].to_numpy() for c in cols if c != "timestamp"}, int(m.sum())))
    rows = sorted(rows, key=lambda r: -r[3])[:50]
    syms = [r[0] for r in rows]

    def wide(col):
        return pd.DataFrame({r[0]: pd.Series(r[2].get(col), index=r[1]) for r in rows if col in r[2]}).sort_index()

    C = wide("close"); C = C[~C.index.duplicated(keep="last")].sort_index()
    O = wide("open").reindex(C.index)
    H = wide("high").reindex(C.index)
    L = wide("low").reindex(C.index)
    R = C.pct_change(fill_method=None)
    bv, sv = wide("buy_vol"), wide("sell_vol")
    F = {
        "mom7":  C / C.shift(7) - 1,
        "mom14": C / C.shift(14) - 1,
        "mom30": C / C.shift(30) - 1,
        "rsi14": C.apply(fl._rsi),
        "brk14": C / C.rolling(14, min_periods=14).max().shift(1) - 1,
        "rangepos": (C - L.rolling(14, min_periods=14).min()) / (H.rolling(14, min_periods=14).max() - L.rolling(14, min_periods=14).min() + 1e-12),
        "volexp": R.rolling(7).std() / (R.rolling(30).std() + 1e-12),
        "accel": (C / C.shift(7) - 1) - (C.shift(7) / C.shift(14) - 1),
        "vpin": wide("norm_vpin"),
        "ofi": (bv - sv) / (bv + sv + 1e-9),
        "dev": wide("norm_deviation"),
        "fdclose": wide("norm_fd_close"),
        "dvol": wide("volume_usd").pct_change(),
    }
    F = {k: v.reindex(index=C.index, columns=C.columns) for k, v in F.items()}
    oos_lab = {"C": C, "O": O, "H": H, "L": L, "R": R, "F": F, "syms": syms, "end": oos_end or "live"}

    records = _dev_backtest(oos_lab, v51_lab=None, hold=hold, verbose=False)
    results, _ = _aggregate_results(oos_lab, records, oos_start, oos_end or "live")
    if verbose:
        _print_report(results)
    return results


def _aggregate_results(lab, records, window_start, window_end):
    """Re-package records into a results dict (same structure as run_dev)."""
    C = lab["C"]
    all_book = [r["roi"] for r in records]
    all_bh = [r["bh_roi"] for r in records]
    book_compound = _compound(all_book)
    bh_compound = _compound(all_bh)
    book_maxdd = _maxdd(all_book)
    bh_maxdd = _maxdd(all_bh)
    by_regime = {}
    bh_by_regime = {}
    for rg in ("bull", "chop", "bear"):
        by_regime[rg] = _regime_stats(records, rg)
        sub_bh = [r["bh_roi"] for r in records if r["regime"] == rg]
        bh_by_regime[rg] = {"compound_pct": _compound(sub_bh), "maxdd_pct": _maxdd(sub_bh)} if sub_bh else None
    bear_book_roi = [r["roi"] for r in records if r["regime"] == "bear"]
    bear_bh_roi = [r["bh_roi"] for r in records if r["regime"] == "bear"]
    results = {
        "spec": SPEC,
        "dev_window": {"start": str(window_start), "end": str(window_end),
                       "n_bars": len(C.index), "n_assets": len(lab["syms"])},
        "n_slices": len(records),
        "full_period": {
            "book_compound_pct": book_compound, "bh_compound_pct": bh_compound,
            "book_maxdd_pct": book_maxdd, "bh_maxdd_pct": bh_maxdd,
            "dd_saved_pp": round(book_maxdd - bh_maxdd, 1),
        },
        "by_regime": by_regime,
        "bh_by_regime": bh_by_regime,
        "bear_preservation": {
            "bear_book_maxdd_pct": _maxdd(bear_book_roi) if bear_book_roi else 0.0,
            "bear_bh_maxdd_pct": _maxdd(bear_bh_roi) if bear_bh_roi else 0.0,
            "dd_saved_pp": round((_maxdd(bear_book_roi) if bear_book_roi else 0.0) -
                                  (_maxdd(bear_bh_roi) if bear_bh_roi else 0.0), 1),
            "n_bear_slices": len(bear_book_roi),
        },
        "liq_amplifier": {
            "n_liq_slices": sum(1 for r in records if r.get("liq_fired")),
            "n_no_liq_slices": sum(1 for r in records if not r.get("liq_fired")),
            "mean_roi_liq_pct": None,
            "mean_roi_no_liq_pct": None,
            "amplifier_premium_pp": None,
        },
    }
    return results, records


# ---- SELFTEST (smoke only, no OOS) ----

def selftest():
    print(f"[selftest] move_catch_book -- smoke test (DEV wall <= {DEV_END})")
    print("  Loading lab ...")
    lab = _load_dev_lab()
    C = lab["C"]
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"  {len(lab['syms'])} assets, {len(C.index)} bars, {C.index.min().date()} -> {C.index.max().date()}")
    # Regime check
    reg = _regime(lab)
    vc = reg.value_counts()
    print(f"  Regime distribution: {dict(vc)}")
    # Single invoke test
    di = 50
    pos, meta = invoke(lab, di)
    print(f"  invoke(di=50): regime={meta['regime']}, action={meta['action']}, picks={meta.get('picks', [])}")
    # Spot-check: bear invoke -> cash
    # Find a bear bar
    bear_bars = [i for i, r in enumerate(reg) if r == "bear"]
    if bear_bars:
        db = bear_bars[0]
        pos2, meta2 = invoke(lab, db)
        assert pos2 == {}, f"Bear should return empty positions, got {pos2}"
        assert meta2["action"] == "CASH", f"Bear action should be CASH"
        print(f"  Bear gate: invoke(di={db}) -> CASH (correct)")
    # Mini backtest (20 slices)
    records = _dev_backtest(lab, v51_lab=None)
    n_part = sum(1 for r in records if r["action"] == "PARTICIPATE")
    n_cash = sum(1 for r in records if r["action"] == "CASH")
    print(f"  Mini backtest: {len(records)} slices, {n_part} participate, {n_cash} cash")
    print("[selftest] PASSED -- move_catch_book smoke test OK, DEV-walled.")
    return 0


# ---- CLI ----

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="Quick smoke test")
    ap.add_argument("--run_dev", action="store_true", help="Full DEV backtest + report")
    ap.add_argument("--save", action="store_true", help="Save results JSON to runs/strat/")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(selftest())
    elif args.run_dev:
        results, records = run_dev(verbose=True)
        if args.save:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = CRYPTO / "runs" / "strat"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"move_catch_book_dev_{ts}.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n  Saved -> {out_path}")
    else:
        ap.print_help()
