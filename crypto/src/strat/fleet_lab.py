"""src/strat/fleet_lab.py -- FLEET-of-agents lab + slice-invocation harness (DEV-window-walled).

USER (orc 2026-06-20): a FLEET of trading agents, each with a DIFFERENT information set (1 TI, 2 TIs, overlapping,
TI x 3 chimera features, none, ...). The search space {feature-subset} x {universe} x {TF} x {combine} is huge ->
search INTELLIGENTLY (importance-guided + evolutionary), not brute force. And ACTUALLY test slice invocation.

DATA WALL (binding, user 2026-06-20): DEVELOP on TRAIN+VAL ONLY (<= 2024-05-15). NEVER touch OOS/UNSEEN (>= 2024-05-15).
load_wide() HARD-CAPS at DEV_END so no agent can peek held-out data. (Canonical split: val_end 2024-05-15.)

AN AGENT = a subset of FEATURES (TIs + chimera) + a combine rule. agent_score -> per-asset causal composite ->
top-K positions held `hold` bars. invoke(agent, slice) -> ROI on that slice. fleet = a population of agents.

RWYB: python -m strat.fleet_lab --selftest        (loads DEV u50, invokes sample agents on a DEV slice)
No emoji. No git commits.
"""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent
DEV_END = "2024-05-15"          # <<< THE WALL: TRAIN+VAL only. Never load/eval >= this. (calendar date -> TF-agnostic)
COST = 0.0024                   # taker RT
CHIM_BASE = CRYPTO / "data" / "processed" / "chimera"
CHIM_DIR = CHIM_BASE / "1d"     # default; load_wide(tf=...) repoints for sub-daily
BARS_PER_DAY = {"1d": 1, "4h": 6, "2h": 12, "1h": 24, "30m": 48, "15m": 96}   # for days<->bars at sub-daily
PANDAS_FREQ = {"1d": "D", "4h": "4h", "2h": "2h", "1h": "1h", "30m": "30min", "15m": "15min"}  # floor-to-bar-grid


def _rsi(s, n=14):
    d = s.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


# columns aggregated by SUM when down-sampling raw bars (volumes are extensive). OHLC is first/max/min/last.
_SUM_COLS = ("volume_usd", "buy_vol", "sell_vol")
# pre-normalized chimera features: rolling-z-scored against a NATIVE-cadence baseline (VPIN intra-bar flow
# toxicity, deviation-from-local-trend, fractal-dim of the path). They are NOT honestly reconstructible by
# aggregating two native-1h normalized values -> a synthesized-2h bar has no native-2h normalization baseline.
# We set them to NaN at synthesized-2h (UNAVAILABLE, not silently wrong) so a feature/agent that uses them is
# transparently absent at 2h rather than fed a meaningless number. ofi/dvol are recomputed from summed raw vols.
_CHIM_NORM_COLS = ("norm_vpin", "norm_deviation", "norm_fd_close")
_2H_MS = 7_200_000   # 2h bucket width in ms (matches ma_2020_breakdown._panel)


def _synth_2h_row(sym, idx_1h, data_1h, n_1h):
    """Aggregate ONE symbol's native-1h row -> a non-overlapping CAUSAL 2h row, REUSING the OHLC aggregation
    of ma_2020_breakdown._panel (O=first, H=max, L=min, C=last per 2h bucket). Volumes summed; pre-normalized
    chimera features set NaN (no native-2h baseline -> not reconstructible). idx_1h is the FLOORED 1h grid
    (so bucket boundaries land on the 2h grid); a 2h bar covering [t, t+2h) uses ONLY its two 1h bars (causal,
    no future bar leaks back -- the stamp is the bar grid-floor, value available at bar-close). Returns the same
    (sym, idx2h, {col: np.array}, n2h) tuple shape the native glob loop produces, or None if too few bars."""
    ms_1h = (idx_1h.asi8 // 10**6).astype(np.int64)        # floored 1h grid -> ms
    bucket = ms_1h // _2H_MS                                # 2h bucket id (same arithmetic as the canonical _panel)
    cols = list(data_1h.keys())
    df = pd.DataFrame({c: data_1h[c] for c in cols})
    df["__b"] = bucket
    agg = {}
    if "open" in cols:   agg["open"] = ("open", "first")
    if "high" in cols:   agg["high"] = ("high", "max")
    if "low" in cols:    agg["low"] = ("low", "min")
    if "close" in cols:  agg["close"] = ("close", "last")
    for c in _SUM_COLS:
        if c in cols:    agg[c] = (c, "sum")
    g = df.groupby("__b", sort=True).agg(**agg)
    idx2h = pd.to_datetime(g.index.to_numpy() * _2H_MS, unit="ms")   # bucket-id -> 2h grid stamp (bar-open/floor)
    out = {c: g[c].to_numpy() for c in g.columns}
    n2h = len(g)
    # pre-normalized chimera features are UNAVAILABLE at synthesized-2h -> NaN (documented; never silently wrong)
    for c in _CHIM_NORM_COLS:
        if c in cols:
            out[c] = np.full(n2h, np.nan)
    return (sym, idx2h, out, n2h)


def _native_2h_present():
    """True iff a native 2h chimera dir with at least one parquet exists (else synthesize from 1h)."""
    d = CHIM_BASE / "2h"
    return d.is_dir() and bool(glob.glob(str(d / "*.parquet")))


def load_wide(n=50, start="2019-01-01", end=DEV_END, min_bars=400, tf="1d"):
    """DEV-window wide universe at timeframe `tf`. HARD cap at `end` (<= DEV_END). Returns features dict (all causal).

    tf in {1d,4h,2h,1h,30m,15m}. The wall is a CALENDAR date so it caps any TF identically. For sub-daily the index is
    floored to the bar grid (NOT to the day) so intraday bars don't collapse. Feature windows are in BARS (caller rescales
    for calendar semantics via BARS_PER_DAY)."""
    assert pd.Timestamp(end) <= pd.Timestamp(DEV_END), f"WALL VIOLATION: end {end} >= DEV_END {DEV_END}"
    assert tf in BARS_PER_DAY, f"unknown tf {tf}"
    import polars as pl
    # 2h has NO native chimera dir (verified on disk: 2h ABSENT) -> SYNTHESIZE from native 1h by 2-bar OHLC
    # aggregation (the canonical ma_2020_breakdown._panel recipe). NEVER 1h relabeled: bars are HALVED.
    synth_2h = (tf == "2h" and not _native_2h_present())
    read_tf = "1h" if synth_2h else tf                     # which native dir we actually read parquet from
    floor_freq = PANDAS_FREQ[read_tf]                      # 1h grid for synth (so 2h buckets land on the 2h grid)
    # min_bars is a 2h-cadence floor; the native 1h source has ~2x as many bars per the same window
    read_min_bars = (min_bars * 2 if synth_2h else min_bars)
    chim_dir = CHIM_BASE / read_tf
    s_ms = pd.Timestamp(start).value // 10**6; e_ms = pd.Timestamp(end).value // 10**6
    want = ["timestamp", "open", "high", "low", "close", "volume_usd", "buy_vol", "sell_vol",
            "norm_vpin", "norm_deviation", "norm_fd_close"]
    rows = []
    for f in sorted(glob.glob(str(chim_dir / "*.parquet"))):
        sym = Path(f).stem.split("_")[0].upper()
        try:
            cols = [c for c in want if c in pl.read_parquet_schema(f)]
            df = pl.read_parquet(f, columns=cols).sort("timestamp")
        except Exception:
            continue
        ms = df["timestamp"].to_numpy()
        m = (ms >= s_ms) & (ms < e_ms)                     # WALL: strictly < end
        if m.sum() < read_min_bars:
            continue
        d = df.filter(pl.Series(m))
        # floor raw (jittered, end-of-bar) stamps to the bar-grid so assets ALIGN in the wide join.
        idx = pd.to_datetime(d["timestamp"].to_numpy(), unit="ms").floor(floor_freq)
        data = {c: d[c].to_numpy() for c in cols if c != "timestamp"}
        if synth_2h:                                       # native-1h row -> non-overlapping causal 2h row
            row = _synth_2h_row(sym, idx, data, int(m.sum()))
            if row is not None and row[3] >= min_bars:
                rows.append(row)
        else:
            rows.append((sym, idx, data, int(m.sum())))
    rows = sorted(rows, key=lambda r: -r[3])[:n]            # top-n by DEV-window coverage
    syms = [r[0] for r in rows]
    def wide(col):
        return pd.DataFrame({r[0]: pd.Series(r[2].get(col), index=r[1]) for r in rows if col in r[2]}).sort_index()
    C, O, H, L = wide("close"), wide("open"), wide("high"), wide("low")
    C = C[~C.index.duplicated(keep="last")].sort_index()
    O = O.reindex(C.index); H = H.reindex(C.index); L = L.reindex(C.index)
    R = C.pct_change(fill_method=None)
    bv, sv = wide("buy_vol"), wide("sell_vol")
    F = {  # FEATURE LIBRARY (causal, cross-sectional). agents subset these.
        # --- TIs (price) ---
        "mom7":  C / C.shift(7) - 1,
        "mom14": C / C.shift(14) - 1,
        "mom30": C / C.shift(30) - 1,
        "rsi14": C.apply(_rsi),
        "brk14": C / C.rolling(14, min_periods=14).max().shift(1) - 1,        # breakout vs prior 14d high
        "rangepos": (C - L.rolling(14, min_periods=14).min()) / (H.rolling(14, min_periods=14).max() - L.rolling(14, min_periods=14).min() + 1e-12),
        "volexp": R.rolling(7).std() / (R.rolling(30).std() + 1e-12),         # vol expansion
        "accel": (C / C.shift(7) - 1) - (C.shift(7) / C.shift(14) - 1),       # momentum acceleration
        # --- chimera (exogenous-ish) ---
        "vpin": wide("norm_vpin"),                                            # order-flow toxicity
        "ofi": (bv - sv) / (bv + sv + 1e-9),                                  # order-flow imbalance
        "dev": wide("norm_deviation"),
        "fdclose": wide("norm_fd_close"),
        "dvol": wide("volume_usd").pct_change(fill_method=None),              # $-volume change
    }
    # shift-1 every feature to guarantee causality at decision bar (value known at d used for d->d+h)
    F = {k: v.reindex(index=C.index, columns=C.columns) for k, v in F.items()}
    # feature availability: at synthesized-2h the pre-normalized chimera features are NaN (no native-2h baseline)
    unavailable = list(_CHIM_NORM_COLS) if synth_2h else []
    return {"C": C, "O": O, "H": H, "L": L, "R": R, "F": F, "syms": syms, "end": end,
            "tf": tf, "synth_2h": synth_2h, "n_admitted": len(syms),
            "feats_unavailable": ["vpin", "dev", "fdclose"] if synth_2h else [],
            "n_native_files": len(glob.glob(str((CHIM_BASE / read_tf) / "*.parquet")))}


# ---- AGENT: a feature-subset + combine -> per-asset score -> top-K positions ----
def agent_score(lab, feats, di, signs=None):
    """Cross-sectional composite z-score at bar di from the agent's feature subset (causal: uses row di)."""
    C = lab["C"]; F = lab["F"]
    parts = []
    for j, f in enumerate(feats):
        row = F[f].iloc[di]
        z = (row - row.mean()) / (row.std() + 1e-12)
        sgn = 1.0 if signs is None else signs[j]
        parts.append(sgn * z.fillna(0.0))
    return sum(parts) / max(1, len(parts))


def invoke(lab, feats, di, hold=7, K=5, signs=None):
    """SLICE INVOCATION: agent decides at di (using features <= di), holds top-K `hold` bars -> net ROI."""
    C = lab["C"]
    if di + hold >= len(C.index): return None
    sc = agent_score(lab, feats, di, signs)
    elig = sc.dropna()
    if len(elig) < K: return None
    picks = elig.sort_values(ascending=False).index[:K]
    fwd = np.mean([C[s].iloc[di + hold] / C[s].iloc[di] - 1 for s in picks
                   if pd.notna(C[s].iloc[di]) and pd.notna(C[s].iloc[di + hold])])
    return float(fwd) - COST


def fleet_invoke(lab, fleet, di, hold=7):
    """Ensemble of agents -> mean ROI (each agent = (feats, K, signs))."""
    rs = [invoke(lab, a["feats"], di, hold, a.get("K", 5), a.get("signs")) for a in fleet]
    rs = [r for r in rs if r is not None]
    return float(np.mean(rs)) if rs else None


def slice_dates(lab, n=200, hold=7, seed=0):
    C = lab["C"]; rng = np.random.default_rng(seed)
    valid = [i for i in range(40, len(C.index) - hold - 1)]
    return sorted(rng.choice(valid, min(n, len(valid)), replace=False))


def selftest(tf="1d"):
    bpd = BARS_PER_DAY[tf]
    hold = 7 * bpd                  # 7 CALENDAR days in bars (the slice horizon is calendar-fixed across TFs)
    print(f"[selftest] fleet_lab tf={tf} ({bpd} bars/day) -- DEV wall <= {DEV_END}, hold={hold} bars (7d)")
    lab = load_wide(n=50, min_bars=200 * bpd if tf != "1d" else 400, tf=tf)
    C = lab["C"]
    print(f"  loaded {len(lab['syms'])} assets; range {C.index.min()} -> {C.index.max()}  (must be < {DEV_END})")
    assert C.index.max() < pd.Timestamp(DEV_END), "WALL VIOLATION"
    print(f"  features: {list(lab['F'].keys())}")
    # invoke a few sample agents (different info sets) on random DEV slices
    agents = {
        "1TI(mom14)":        {"feats": ["mom14"]},
        "2TI(mom14,rsi14)":  {"feats": ["mom14", "rsi14"]},
        "TIxChimera(mom14,vpin,ofi)": {"feats": ["mom14", "vpin", "ofi"]},
        "breakout+volexp":   {"feats": ["brk14", "volexp"]},
        "chimera-only(ofi,vpin,dev)": {"feats": ["ofi", "vpin", "dev"]},
    }
    ds = slice_dates(lab, 150, hold=hold)
    print(f"\n  SLICE-INVOCATION TEST ({len(ds)} random DEV 7d slices = {hold} bars, top-5):")
    print(f"  {'agent':30}{'profit%':>9}{'mean':>8}{'beatEW':>8}")
    ew = [np.mean([C[s].iloc[d+hold]/C[s].iloc[d]-1 for s in C.columns if pd.notna(C[s].iloc[d]) and pd.notna(C[s].iloc[d+hold])]) for d in ds]
    ew = np.array(ew)
    for name, a in agents.items():
        rr = [invoke(lab, a["feats"], d, hold, 5) for d in ds]; rr = np.array([x for x in rr if x is not None])
        if len(rr) < 3: print(f"  {name:30}  (insufficient)"); continue
        print(f"  {name:30}{100*np.mean(rr>0):>8.0f}%{100*rr.mean():>8.2f}{100*np.mean(rr[:len(ew)]>ew[:len(rr)]):>7.0f}%")
    print(f"  {'EW buy-hold (ref)':30}{100*np.mean(ew>0):>8.0f}%{100*ew.mean():>8.2f}")
    print("\n[selftest] PASSED -- slice invocation works, DEV-walled.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--tf", default="1d")
    a = ap.parse_args()
    raise SystemExit(selftest(a.tf) if a.selftest else 0)
