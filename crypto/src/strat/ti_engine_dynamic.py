"""Phase 2 — DYNAMIC/ADAPTIVE ENGINES (DYN-3: MA-EA first).

Implements the PHASE2_DYNAMIC_ENGINES_DESIGN.md architecture:
  - adapt_fn(lookback_window) → adapted_params
  - static twin = adapt_fn returns base params (m=0)
  - One free param per engine
  - Full V1.1 rigor (systematic grid, TRAIN selection, confirmation, regime gate, cost)

DYN-3 MA-EA: Efficiency-Adaptive MA Period
  mechanism: period = base_period * (1 - m * eff)  clamped [min_period, max_period]
  eff = |close[t]-close[t-L]| / sum(|diff|[t-L:t])  (trend efficiency, 0..1)
  high eff (strong trend) → shorter period → faster MA → catch moves earlier
  low eff (chop) → longer period → slower MA → fewer false crosses
  Free param: m (adaptation strength). Static twin: m=0.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _MA, held_cross

BARS_PER_DAY = fl.BARS_PER_DAY
TRAIN_END = "2023-07-01"
VAL_END = "2024-05-15"
OOS_2024_END = "2024-12-31"
TRAIN_WALL_MS = int(pd.Timestamp(TRAIN_END).value // 10**6)
VAL_WALL_MS = int(pd.Timestamp(VAL_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp("2025-01-01").value // 10**6)
MA_TYPES = list(_MA.keys())
OUTDIR = CRYPTO / "runs" / "engines" / "V7_dynamic"

# ---- MA-cross config grid (fast,slow period pairs) ----
# Systematic: fast {3..20}, slow = fast*{2,2.5,3,4}. ~60 configs per type.
MA_CONFIGS = []
for fast in [3, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
    for mult in [2.0, 2.5, 3.0, 4.0]:
        slow = int(fast * mult)
        if slow <= 80:
            MA_CONFIGS.append((fast, slow))
# dedup
MA_CONFIGS = sorted(set(MA_CONFIGS))


# =============================================================================
# EFFICIENCY-ADAPTIVE MA HELD (the core DYN-3 mechanism)
# =============================================================================

def efficiency_ratio(close: np.ndarray, lookback: int) -> np.ndarray:
    """Causal efficiency ratio: |net move| / sum(|bar moves|) over trailing window."""
    c = np.asarray(close, float)
    n = len(c)
    s = pd.Series(c)
    net = (s - s.shift(lookback)).abs()
    abs_diff = s.diff().abs()
    sum_moves = abs_diff.rolling(lookback, min_periods=lookback).sum()
    eff = (net / sum_moves.replace(0, np.nan)).to_numpy()
    eff = np.nan_to_num(eff, nan=0.0)
    return eff


def adaptive_ma_held(close: np.ndarray, base_fast: int, base_slow: int,
                     ma_type: str, m: float, lookback_bars: int,
                     min_period: int = 3, max_period: int = 80) -> np.ndarray:
    """Efficiency-adaptive MA crossover. Adapts both fast and slow periods.

    adapted_fast = round(base_fast * (1 - m * eff))  clamped [min_period, base_fast*2]
    adapted_slow = round(base_slow * (1 - m * eff))  clamped [base_fast+1, max_period]

    When m=0: identical to static (frozen periods). The static twin.
    When m>0: trends shorten periods (faster), chop lengthens (slower).
    """
    c = np.asarray(close, float)
    n = len(c)
    if m == 0.0:
        # static twin: simple held_cross with fixed periods
        return held_cross(c, (base_fast, base_slow), ma_type)

    eff = efficiency_ratio(c, lookback_bars)
    ma_fn = _MA[ma_type]

    # We can't cheaply compute a rolling-period MA (each bar has different period).
    # Approach: discretize the adapted periods into a small set and blend.
    # For efficiency: compute MA at a few candidate periods and select per-bar.
    # Practical: compute 5 period variants and at each bar pick the one closest
    # to the adapted period.
    fast_candidates = sorted(set([
        max(min_period, int(base_fast * (1 - m * e)))
        for e in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ] + [base_fast]))
    slow_candidates = sorted(set([
        min(max_period, max(base_fast + 1, int(base_slow * (1 - m * e))))
        for e in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ] + [base_slow]))

    # Pre-compute all MAs
    ma_cache = {}
    for p in set(fast_candidates + slow_candidates):
        if p >= 2 and p <= len(c):
            ma_cache[p] = ma_fn(c, p)

    # Per-bar: determine adapted periods and the crossover state
    held = np.zeros(n, dtype=np.int8)
    warmup = max(base_slow, lookback_bars) + 5
    for i in range(warmup, n):
        e = eff[i]
        af = max(min_period, int(round(base_fast * (1 - m * e))))
        asl = max(af + 1, min(max_period, int(round(base_slow * (1 - m * e)))))

        # Find closest precomputed fast
        best_f = min(fast_candidates, key=lambda p: abs(p - af))
        best_s = min(slow_candidates, key=lambda p: abs(p - asl))

        if best_f in ma_cache and best_s in ma_cache:
            if ma_cache[best_f][i] > ma_cache[best_s][i]:
                held[i] = 1
    return held


# =============================================================================
# CONFIRMATION + ENSEMBLE (same as V6 rigorous)
# =============================================================================

def apply_confirmation(close, raw_held, lookback_bars, slice_bars):
    """signal AND close > EMA(lookback) AND EMA rising → hold for slice_bars."""
    from strat.ma_type_upgrade import _ema
    c = np.asarray(close, float)
    n = len(c); m = np.isfinite(c)
    out = np.zeros(n, dtype=np.int8)
    if m.sum() < max(5, lookback_bars):
        return out
    cseg = c[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[False], ema_lb[1:] > ema_lb[:-1]])
    raw = np.asarray(raw_held, dtype=bool)[m]
    confirmed = raw & above & rising
    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0; L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            held_seg[i:end] = 1
            i = end
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


def make_dynamic_ma_provider(ma_type: str, configs: list, m: float,
                              lookback_bars: int, slice_bars: int,
                              base_vote_threshold: float = 0.5):
    """Dynamic MA-EA hold provider with EFFICIENCY GATE.

    The core insight from V6: static MA ensembles fire ~100% because SOME configs
    always show fast > slow even in chop. The adaptation mechanism that WORKS is not
    tweaking thresholds — it's GATING the engine OFF during choppy periods using
    the efficiency ratio as a within-bull selectivity layer.

    mechanism: engine = CASH when eff < eff_threshold (chop → sit out)
               engine = ACTIVE (normal ensemble) when eff >= eff_threshold (trending)
    eff_threshold = base_eff + m * vov_z (adapts: raise bar in uncertain periods)

    m=0: fixed eff_threshold = base_eff (static-with-efficiency-gate)
    m>0: adapts the eff threshold based on vol-of-vol (the dynamic layer)

    This is mechanically the SAME as what ADX does (ADX IS an efficiency gate) but
    applied to the MA ensemble — making it the MA+ADX hybrid that the data says works.
    """
    base_eff_threshold = 0.25  # above this = trending, allow entries (tune via m)

    def provider(sym, o, h, l, c):
        close = np.asarray(c, float)
        n = len(close)
        if n < lookback_bars + 10:
            return np.zeros(n, np.int8)

        # Efficiency ratio (the within-bull selectivity signal)
        eff = efficiency_ratio(close, lookback_bars)

        # Vol-of-vol for adaptive threshold
        rets = np.zeros(n); rets[1:] = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)
        rv = pd.Series(rets).rolling(lookback_bars, min_periods=lookback_bars // 2).std().to_numpy()
        rv_mean = pd.Series(rv).rolling(lookback_bars * 4, min_periods=lookback_bars).mean().to_numpy()
        rv_std = pd.Series(rv).rolling(lookback_bars * 4, min_periods=lookback_bars).std().to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            vov_z = np.where(rv_std > 1e-10, (rv - rv_mean) / rv_std, 0.0)
        vov_z = np.nan_to_num(vov_z, nan=0.0)

        # Adapted efficiency threshold
        eff_thr = np.clip(base_eff_threshold + m * 0.08 * vov_z, 0.10, 0.70)

        # MA ensemble votes (with period adaptation)
        votes = np.zeros(n, float)
        nv = 0
        for fast, slow in configs:
            try:
                held = adaptive_ma_held(close, fast, slow, ma_type, m * 0.3, lookback_bars)
                votes += held.astype(float)
                nv += 1
            except Exception:
                continue
        if nv == 0:
            return np.zeros(n, np.int8)
        frac = votes / nv
        ma_signal = (frac > base_vote_threshold).astype(np.int8)

        # DYNAMIC HOLD: held = 1 when BOTH:
        #   (a) MA ensemble says long (ma_signal = 1)
        #   (b) Efficiency above adapted threshold (eff >= eff_thr)
        # Exit IMMEDIATELY when either condition fails.
        # This replaces the fixed 7d slice-hold — the hold duration is now
        # ADAPTIVE (stays on while trend is efficient, exits when chop starts).
        from strat.ma_type_upgrade import _ema
        cseg = close[np.isfinite(close)]
        if len(cseg) < lookback_bars:
            return np.zeros(n, np.int8)
        ema_lb = np.zeros(n)
        m_fin = np.isfinite(close)
        ema_lb[m_fin] = _ema(close[m_fin], lookback_bars)

        # Confirmation: MA signal + eff above threshold + close > EMA + EMA rising
        above_ema = close > ema_lb
        ema_rising = np.concatenate([[False], ema_lb[1:] > ema_lb[:-1]])
        eff_open = eff >= eff_thr

        held = (ma_signal.astype(bool) & eff_open & above_ema & ema_rising).astype(np.int8)
        held[:lookback_bars] = 0
        return held
    return provider


# =============================================================================
# REGIME GATE
# =============================================================================

def btc_regime(wall_ms):
    rec = _load_asset("BTCUSDT", "1d", wall_ms)
    if not rec:
        return None
    d, _, _, _, cl = rec
    sma = pd.Series(cl, index=d).rolling(100).mean()
    s = pd.Series(0, index=d, dtype=int); st = 0
    for i in range(100, len(d)):
        if np.isnan(sma.iloc[i]):
            continue
        if st == 0 and cl[i] > sma.iloc[i] * 1.02:
            st = 1
        elif st == 1 and cl[i] < sma.iloc[i] * 0.98:
            st = 0
        s.iloc[i] = st
    return s


def causal_regime(wall_ms):
    """De-leaked 1d BTC regime for GATING on a per-bar grid (see V1.1 align_htf_on bug class).

    btc_regime[D] is BTC's day-D 1d CLOSE regime (~23:59). Broadcasting it onto an
    intraday/1d bar grid (normalize-to-day + ffill, or per-bar prior[-1]) makes a day-D bar
    read regime[D] => SAME-DAY LOOK-AHEAD. Causal fix: a bar on day D reads the last CLOSED
    1d bar = regime[D-1]. We shift the daily series forward one full 1d bar before broadcast.
    Gating consumers route through here (see runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json).
    """
    s = btc_regime(wall_ms)
    if s is None:
        return None
    return s.shift(1).fillna(0).astype(int)


# =============================================================================
# CELL RUNNER (shared between dynamic and static-twin)
# =============================================================================

def run_cell(ma_type: str, tf: str, m: float, lookback_days: int = 7,
             slice_days: int = 7, vote_threshold: float = 0.5,
             window_start=None, window_end=None, wall_ms=None, n_null=50):
    """Run one DYN-3 MA-EA cell. m=0 is the static twin."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    if wall_ms is None:
        wall_ms = VAL_WALL_MS

    regime = causal_regime(wall_ms)   # de-leaked: bar on day D reads regime[D-1]
    provider = make_dynamic_ma_provider(ma_type, MA_CONFIGS, m, lookback_bars, slice_bars, vote_threshold)

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        held = provider(sym, o, h, l, c)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)

    if len(close_cols) < 10:
        return {"status": "INSUFFICIENT_ASSETS", "ma_type": ma_type, "tf": tf, "m": m}

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)

    # Vectorized replay
    rets = close_df.pct_change().to_numpy()[1:]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]

    reg_vals = np.ones(len(dates))
    if regime is not None:
        norm_days = pd.DatetimeIndex(dates).normalize()
        ra = regime.reindex(regime.index.union(norm_days)).ffill().reindex(norm_days).fillna(0)
        reg_vals = ra.to_numpy().astype(float)

    mask = np.ones(len(dates), dtype=bool)
    if window_start:
        mask &= dates >= pd.Timestamp(window_start)
    if window_end:
        mask &= dates < pd.Timestamp(window_end)

    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (MAKER_RT / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals
    n_all = valid.sum(axis=1)
    al = np.where(n_all > 0, (rets_f * valid).sum(axis=1) / np.maximum(n_all, 1), 0.0)

    eng_w = eng[mask]; al_w = al[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0

    if len(eng_w) == 0:
        return {"status": "NO_BARS", "ma_type": ma_type, "tf": tf, "m": m}

    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    al_lin_wk = float(np.mean(al_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0.0
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (float(np.std(eng_w)) * np.sqrt(bpd * 365)) if np.std(eng_w) > 1e-12 else 0.0

    # Random-null
    T, A = rets_f.shape
    randc = []
    for s in range(n_null):
        rng = np.random.default_rng(s)
        rh = (rng.random((T, A)) < deploy).astype(np.int8)
        nh = (rh * valid).sum(axis=1)
        r = np.where(nh > 0, (rets_f * rh * valid).sum(axis=1) / np.maximum(nh, 1), 0.0) * reg_vals
        r = r[mask]
        randc.append(float(np.prod(1 + r) - 1) * 100)
    null_pct = float(np.mean(np.array(randc) < comp)) * 100

    return {
        "status": "OK", "ma_type": ma_type, "tf": tf, "m": round(m, 3),
        "n_configs": len(MA_CONFIGS), "lookback_days": lookback_days,
        "n_assets": len(close_cols), "n_bars": int(mask.sum()),
        "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "al_lin_wk": round(al_lin_wk, 4),
        "edge_pp": round(lin_wk - al_lin_wk, 4),
        "compound": round(comp, 1), "maxdd": round(maxdd, 1), "sharpe": round(sharpe, 3),
        "null_pct": round(null_pct, 1), "elapsed_s": round(time.time() - t0, 1),
    }


# =============================================================================
# SWEEP: all MA types × TFs × m values (static-twin A/B built in)
# =============================================================================

def run_dyn3_sweep(ma_types=None, tfs=None, m_values=None, n_null=50):
    if ma_types is None:
        ma_types = MA_TYPES
    if tfs is None:
        tfs = ["1d", "4h", "2h", "1h", "30m"]
    if m_values is None:
        m_values = [0.0, 0.3, 0.5, 0.7, 0.9]  # 0.0 = static twin

    print("=" * 110)
    print("DYN-3 MA-EA — Efficiency-Adaptive MA Period (Phase 2)")
    print("=" * 110)
    print(f"m values: {m_values} (m=0 = static twin)")
    print(f"VAL window: {TRAIN_END}..{VAL_END} | gate=BTC-SMA100 | cost={MAKER_RT} RT")
    print(f"  {'type':<6} {'tf':<4} {'m':>4} | {'deploy':>6} {'edge':>8} {'null':>5} {'shrp':>6} "
          f"{'comp':>8} | vs_static")
    print("  " + "-" * 90)

    all_rows = []
    for ma_type in ma_types:
        for tf in tfs:
            static_edge = None
            for m in m_values:
                r = run_cell(ma_type, tf, m, window_start=TRAIN_END, window_end=VAL_END,
                             wall_ms=VAL_WALL_MS, n_null=n_null)
                all_rows.append(r)
                if r["status"] != "OK":
                    print(f"  {ma_type:<6} {tf:<4} {m:>4.1f} | {r['status']}")
                    continue
                if m == 0.0:
                    static_edge = r["edge_pp"]
                vs = (f"{r['edge_pp'] - static_edge:+.3f}" if static_edge is not None
                      else "—")
                print(f"  {ma_type:<6} {tf:<4} {m:>4.1f} | {r['deploy']:>5.1%} "
                      f"{r['edge_pp']:>+7.3f} {r['null_pct']:>4.0f} {r['sharpe']:>6.2f} "
                      f"{r['compound']:>+7.0f}% | {vs}")
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="all")
    ap.add_argument("--tfs", default="1d,4h,2h,1h,30m")
    ap.add_argument("--m", default="0.0,0.3,0.5,0.7,0.9")
    ap.add_argument("--null", type=int, default=50)
    a = ap.parse_args()

    types = MA_TYPES if a.types == "all" else [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]
    m_vals = [float(x) for x in a.m.split(",")]

    rows = run_dyn3_sweep(types, tfs, m_vals, a.null)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"dyn3_ma_ea_{ts}.json"
    with open(out, "w") as f:
        json.dump({"engine": "DYN-3 MA-EA", "rows": rows, "timestamp": ts}, f, indent=2, default=str)

    # Summary: which (type,tf) cells have dynamic > static?
    ok = [r for r in rows if r["status"] == "OK"]
    statics = {(r["ma_type"], r["tf"]): r for r in ok if r["m"] == 0.0}
    wins = []
    for r in ok:
        if r["m"] > 0:
            st = statics.get((r["ma_type"], r["tf"]))
            if st and r["edge_pp"] > st["edge_pp"] and r["deploy"] < 0.80:
                wins.append((r, st))

    print(f"\n{'='*110}")
    print(f"  STATIC-TWIN A/B: {len(wins)} cells where dynamic > static AND deploy<80%")
    print(f"{'='*110}")
    for r, st in sorted(wins, key=lambda x: -(x[0]["edge_pp"] - x[1]["edge_pp"])):
        print(f"  {r['ma_type']}|{r['tf']} m={r['m']}: edge {r['edge_pp']:+.3f} (static {st['edge_pp']:+.3f}) "
              f"deploy {r['deploy']:.1%} null {r['null_pct']:.0f}")
    if not wins:
        print("  NONE — adaptation does not add selectivity beyond static for MA family.")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
