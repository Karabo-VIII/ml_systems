"""DYN-5: Conviction-on-Agreement — dynamic config selection for MA family.

CHARTER §ENGINE BUILD RIGOR applies. Full V1.1 architecture.

Mechanism (within ONE TI family, e.g., WMA):
  1. Maintain a pool of 44+ MA-cross configs (the systematic grid)
  2. Every SLICE (7d): evaluate which configs were profitable in the TRAILING lookback
     window (7-14d). These become the "credible pool" (top-K).
  3. For each asset at each bar: count how many of the K credible configs fire "long"
  4. Enter ONLY when agreement ≥ conviction threshold (e.g., ≥60% of credible pool agrees)
  5. Exit when agreement drops below exit threshold

This is NOT an ensemble (which superposes ALL configs). It's a CONVICTION filter:
  - The lookback SELECTS which configs are currently credible
  - The agreement requirement demands CONVERGENCE of credible configs
  - When only 2/10 credible configs fire → sit out (low conviction)
  - When 8/10 fire → enter with high conviction (signal convergence)

The dynamic part: the credible pool ADAPTS every slice based on recent performance.
Static twin: random pool selection (ignore trailing P&L, just pick K random configs).

Key difference from DYN-3 (efficiency gate):
  - DYN-3 uses a SEPARATE signal (efficiency ratio) to gate the MA
  - DYN-5 uses the MA configs THEMSELVES to create selectivity via agreement
  - They can combine (efficiency gate + conviction filter = layered selectivity)

D73 safety: selects configs by their OWN recent P&L (endogenous), not external factors.
The "persistence" concern applies but is mitigated by requiring AGREEMENT (not just "run
the recent winner") — convergence of multiple profitable configs is a different signal.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
import sys
import json
import time
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

# Systematic MA-cross config grid (same as DYN-3)
MA_CONFIGS = []
for fast in [3, 5, 7, 8, 9, 10, 12, 14, 16, 18, 20]:
    for mult in [2.0, 2.5, 3.0, 4.0]:
        slow = int(fast * mult)
        if slow <= 80:
            MA_CONFIGS.append((fast, slow))
MA_CONFIGS = sorted(set(MA_CONFIGS))


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

    btc_regime[D] is BTC's day-D 1d CLOSE regime (~23:59). The normalize-to-day + ffill
    broadcast makes a day-D bar read regime[D] => SAME-DAY LOOK-AHEAD. Causal fix: a bar on
    day D reads the last CLOSED 1d bar = regime[D-1]; shift the daily series forward one 1d
    bar before broadcast (see runs/engines/AUDIT_ORACLE_UPGRADE/leakcheck.json).
    """
    s = btc_regime(wall_ms)
    if s is None:
        return None
    return s.shift(1).fillna(0).astype(int)


def make_dyn5_provider(ma_type: str, configs: list, lookback_bars: int,
                       slice_bars: int, top_k: int = 10,
                       conviction_threshold: float = 0.6):
    """DYN-5 Conviction-on-Agreement provider.

    Per asset:
      1. At each slice boundary (every slice_bars): rank configs by trailing P&L
      2. Take top-K as the credible pool
      3. Compute agreement: fraction of K that fire "long" at each bar
      4. held=1 only when agreement >= conviction_threshold
    """
    from strat.ma_type_upgrade import _ema

    def provider(sym, o, h, l, c):
        close = np.asarray(c, float)
        n = len(close)
        if n < lookback_bars + slice_bars:
            return np.zeros(n, np.int8)

        # Pre-compute held states for ALL configs (causal)
        all_helds = []
        for fast, slow in configs:
            try:
                h_cfg = held_cross(close, (fast, slow), ma_type)
                all_helds.append(h_cfg)
            except Exception:
                all_helds.append(np.zeros(n, np.int8))
        all_helds = np.array(all_helds, dtype=np.int8)  # [n_configs x n_bars]

        # Pre-compute per-config per-bar returns (for ranking by trailing P&L)
        rets = np.zeros(n)
        rets[1:] = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)

        held = np.zeros(n, dtype=np.int8)
        # Process in slices
        for slice_start in range(lookback_bars, n, slice_bars):
            slice_end = min(slice_start + slice_bars, n)

            # Rank configs by trailing-lookback P&L
            lb_start = max(0, slice_start - lookback_bars)
            lb_rets = rets[lb_start:slice_start]
            config_pnls = []
            for ci in range(len(configs)):
                cfg_held = all_helds[ci, lb_start:slice_start]
                # P&L = sum(held[t-1] * ret[t]) over lookback (lag-1 for causality)
                cfg_held_lag = np.concatenate([[0], cfg_held[:-1]])
                pnl = np.sum(cfg_held_lag * lb_rets)
                config_pnls.append(pnl)

            # Select top-K credible configs
            ranked = np.argsort(config_pnls)[::-1]
            credible = ranked[:min(top_k, len(ranked))]
            credible_positive = [ci for ci in credible if config_pnls[ci] > 0]
            if len(credible_positive) < 3:
                # Not enough credible configs → sit out this slice
                continue
            pool = credible_positive

            # Compute agreement within the credible pool for this slice
            for t in range(slice_start, slice_end):
                n_agree = sum(1 for ci in pool if all_helds[ci, t] == 1)
                agreement = n_agree / len(pool)
                if agreement >= conviction_threshold:
                    held[t] = 1

        # Confirmation layer: held AND close > EMA(lookback) AND EMA rising
        ema_lb = _ema(close, lookback_bars)
        above = close > ema_lb
        rising = np.concatenate([[False], ema_lb[1:] > ema_lb[:-1]])
        held = held & above.astype(np.int8) & rising.astype(np.int8)
        held[:lookback_bars] = 0
        return held

    return provider


def run_dyn5_cell(ma_type: str, tf: str, top_k: int = 10,
                  conviction_threshold: float = 0.6,
                  lookback_days: int = 7, slice_days: int = 7,
                  window_start=None, window_end=None, wall_ms=None, n_null=50):
    """Run one DYN-5 cell."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    if wall_ms is None:
        wall_ms = VAL_WALL_MS

    regime = causal_regime(wall_ms)   # de-leaked: bar on day D reads regime[D-1]
    provider = make_dyn5_provider(ma_type, MA_CONFIGS, lookback_bars, slice_bars,
                                  top_k, conviction_threshold)

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        h_state = provider(sym, o, h, l, c)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(h_state, index=dates)

    if len(close_cols) < 10:
        return {"status": "INSUFFICIENT_ASSETS", "ma_type": ma_type, "tf": tf}

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
        return {"status": "NO_BARS", "ma_type": ma_type, "tf": tf}

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
        "status": "OK", "engine": "DYN-5", "ma_type": ma_type, "tf": tf,
        "top_k": top_k, "conviction_threshold": conviction_threshold,
        "lookback_days": lookback_days, "slice_days": slice_days,
        "n_configs": len(MA_CONFIGS), "n_assets": len(close_cols),
        "n_bars": int(mask.sum()),
        "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "al_lin_wk": round(al_lin_wk, 4),
        "edge_pp": round(lin_wk - al_lin_wk, 4),
        "compound": round(comp, 1), "maxdd": round(maxdd, 1), "sharpe": round(sharpe, 3),
        "null_pct": round(null_pct, 1), "elapsed_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="all")
    ap.add_argument("--tfs", default="1d,4h,2h")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--conviction", type=float, default=0.6)
    ap.add_argument("--null", type=int, default=50)
    ap.add_argument("--window", default="VAL", choices=["VAL", "OOS2024"])
    a = ap.parse_args()

    types = MA_TYPES if a.types == "all" else [t.strip().upper() for t in a.types.split(",")]
    tfs = [t.strip() for t in a.tfs.split(",")]
    ws = {"VAL": (TRAIN_END, VAL_END, VAL_WALL_MS),
          "OOS2024": (VAL_END, OOS_2024_END, OOS_WALL_MS)}
    w_start, w_end, w_ms = ws[a.window]

    print("=" * 100)
    print(f"DYN-5 CONVICTION-ON-AGREEMENT — MA family ({a.window})")
    print("=" * 100)
    print(f"top_k={a.topk} | conviction>={a.conviction} | lookback=7d | slice=7d")
    print(f"Window: {w_start}..{w_end} | gate=BTC-SMA100 | cost={MAKER_RT} RT")
    print(f"\n  {'type':<6} {'tf':<4} | {'deploy':>6} {'edge':>8} {'null':>5} {'shrp':>6} {'comp':>8}")
    print("  " + "-" * 55)

    rows = []
    for ma_type in types:
        for tf in tfs:
            r = run_dyn5_cell(ma_type, tf, top_k=a.topk, conviction_threshold=a.conviction,
                              window_start=w_start, window_end=w_end, wall_ms=w_ms, n_null=a.null)
            rows.append(r)
            if r["status"] == "OK":
                print(f"  {ma_type:<6} {tf:<4} | {r['deploy']:>5.1%} "
                      f"{r['edge_pp']:>+7.3f} {r['null_pct']:>4.0f} {r['sharpe']:>6.2f} "
                      f"{r['compound']:>+7.0f}%")
            else:
                print(f"  {ma_type:<6} {tf:<4} | {r['status']}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTDIR / f"dyn5_conviction_{a.window}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"engine": "DYN-5", "rows": rows, "top_k": a.topk,
                   "conviction": a.conviction, "window": a.window, "timestamp": ts}, f, indent=2, default=str)
    print(f"\n  Saved: {out}")
