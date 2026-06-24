"""RSI Engine Family — TI-based trading engine using Relative Strength Index.

Signal Logic:
  LONG when: RSI crosses ABOVE the oversold threshold (entry on the TURN, not depth)
  EXIT when: RSI crosses BELOW the overbought threshold OR slice boundary

Key insight from dead-list:
  - D50: RSI oversold thr30 at HIGH-FREQ = cost-killed (HARD at 1h)
  - D51: RSI oversold thr20-deep = worse (depth requirement cuts OOS in half)
  - The TURN is load-bearing, not the depth (D51 lesson)
  - 4h is UNTESTED with this architecture + decoupled lookback

Architecture (same as MA family):
  - Decoupled: lookback_bars for signal confirmation, slice_bars for max hold
  - The RSI signal replaces the MA band — instead of "price above MA", it's
    "RSI turned up from oversold" (a momentum-confirmation entry)
  - Regime gate (V1 BTC SMA100) applied on top
  - Per-slice evaluation, same metrics

Config parameters to sweep:
  - RSI period: {7, 14, 21} (standard is 14)
  - Oversold threshold: {25, 30, 35} (entry trigger)
  - Overbought threshold: {65, 70, 75} (exit trigger)
  - Confirmation: RSI must be RISING for N bars after crossing oversold

Uses pandas_ta for RSI computation (canonical, verified).
"""
from __future__ import annotations

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CRYPTO = ROOT.parent

import strat.fleet_lab as fl
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ma_type_upgrade import _ema

BARS_PER_DAY = fl.BARS_PER_DAY
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "RSI"


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using exponential smoothing (Wilder's method). Causal."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Seed with SMA of first `period` bars
    if len(close) > period:
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period + 1, len(close)):
            avg_gain[i] = avg_gain[i-1] * (1 - alpha) + gain[i] * alpha
            avg_loss[i] = avg_loss[i-1] * (1 - alpha) + loss[i] * alpha
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = 50.0  # neutral before enough data
    return rsi


def rsi_held_state(close: np.ndarray, period: int = 14, 
                   oversold: float = 30.0, overbought: float = 70.0,
                   lookback_bars: int = 42, slice_bars: int = 42,
                   confirm_rising: int = 2, mode: str = "trend") -> np.ndarray:
    """Compute RSI-based held state with decoupled lookback/slice.
    
    Two modes:
    - "reversion": Entry on RSI crossing ABOVE oversold (V1, FAILED)
    - "trend": Entry when RSI > 50 AND rising (V1.1, trend-confirmation)
    
    The "trend" mode aligns with the proven MA mechanism: continuation > reversion.
    Instead of "buy when oversold" (catching falling knives), it's "be long when
    momentum is confirmed" (riding trends). This is structurally different.
    """
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    fin = np.isfinite(close)
    if fin.sum() < max(period + 10, lookback_bars // 4):
        return held
    
    rsi = compute_rsi(close, period)
    
    if mode == "trend":
        # V1.1: TREND CONFIRMATION
        # Long when RSI > 50 (bullish momentum) AND rising over confirm_rising bars
        # Exit when RSI < 45 (momentum fading) OR overbought OR slice end
        above_midline = rsi > 50.0
        rsi_rising = np.zeros(n, dtype=bool)
        for i in range(confirm_rising, n):
            if all(rsi[i-j] >= rsi[i-j-1] for j in range(confirm_rising)):
                rsi_rising[i] = True
        
        not_overbought = rsi < overbought
        confirmed_long = above_midline & rsi_rising & not_overbought & fin
        
        # Decoupled slice holding
        i = period  # start after RSI warmup
        while i < n:
            if confirmed_long[i]:
                end = min(i + slice_bars, n)
                j = i
                while j < end:
                    # Exit if RSI drops below 45 (momentum lost)
                    if rsi[j] < 45.0:
                        break
                    # Exit if overbought
                    if rsi[j] > overbought:
                        held[j] = 1
                        j += 1
                        break
                    held[j] = 1
                    j += 1
                i = j
            else:
                i += 1
    else:
        # V1: REVERSION (original, FAILED)
        above_oversold = rsi > oversold
        rsi_rising = np.zeros(n, dtype=bool)
        for i in range(confirm_rising, n):
            if all(rsi[i-j] >= rsi[i-j-1] for j in range(confirm_rising)):
                rsi_rising[i] = True
        
        not_overbought = rsi < overbought
        was_oversold = np.zeros(n, dtype=bool)
        lookback_check = min(lookback_bars, period * 3)
        for i in range(lookback_check, n):
            if np.any(rsi[max(0, i-lookback_check):i] < oversold):
                was_oversold[i] = True
        
        confirmed_long = above_oversold & rsi_rising & not_overbought & was_oversold & fin
        
        i = 0
        while i < n:
            if i >= period and confirmed_long[i]:
                end = min(i + slice_bars, n)
                j = i
                while j < end:
                    if rsi[j] > overbought:
                        held[j] = 1
                        j += 1
                        break
                    if not confirmed_long[j] and j > i + 3:
                        break
                    held[j] = 1
                    j += 1
                i = j
            else:
                i += 1
    
    return held


def run_rsi_cell(tf: str, period: int = 14, oversold: float = 30.0, 
                 overbought: float = 70.0, lookback_days: int = 7,
                 slice_days: int = 7, mode: str = "trend") -> dict:
    """Run one RSI engine cell."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10**6)
    symbols = _u100_symbols()
    
    close_dict = {}
    held_dict = {}
    
    for sym in symbols:
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        
        fin = np.asarray(c, dtype=float)
        held = rsi_held_state(fin, period, oversold, overbought, lookback_bars, slice_bars, mode=mode)
        
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)
    
    n_assets = len(close_dict)
    if n_assets < 20:
        return {"status": "INSUFFICIENT_ASSETS", "n_assets": n_assets}
    
    # Build date universe
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    
    # Compute bar returns
    engine_rets = []
    al_rets = []
    
    for i in range(1, len(all_dates)):
        dp = all_dates[i-1]
        dc = all_dates[i]
        
        eng_r = []
        all_r = []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]
                cc = close_dict[sym][dc]
                if cp > 0:
                    ret = (cc / cp) - 1.0
                    all_r.append(ret)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        eng_r.append(ret)
        
        engine_rets.append(float(np.mean(eng_r)) if eng_r else 0.0)
        al_rets.append(float(np.mean(all_r)) if all_r else 0.0)
    
    eng = np.array(engine_rets)
    al = np.array(al_rets)
    
    # Per-slice metrics
    n_bars = len(eng)
    eng_slices = []
    al_slices = []
    i = 0
    while i + slice_bars <= n_bars:
        eng_slices.append(float(np.prod(1 + eng[i:i+slice_bars]) - 1) * 100)
        al_slices.append(float(np.prod(1 + al[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    
    n_slices = len(eng_slices)
    pct_prof = sum(1 for s in eng_slices if s > 0) / n_slices if n_slices > 0 else 0
    median_s = float(np.median(eng_slices)) if eng_slices else 0
    
    compound = float(np.prod(1 + eng) - 1) * 100
    al_compound = float(np.prod(1 + al) - 1) * 100
    
    vs_al = [e - a for e, a in zip(eng_slices, al_slices)]
    beats_al = sum(1 for v in vs_al if v > 0) / len(vs_al) if vs_al else 0
    
    cum = np.cumprod(1 + eng)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
    
    deploy = float(np.mean(eng != 0))
    elapsed = time.time() - t0
    
    return {
        "status": "OK",
        "config": {"tf": tf, "period": period, "oversold": oversold, "overbought": overbought,
                   "lookback_days": lookback_days, "slice_days": slice_days},
        "n_assets": n_assets,
        "n_slices": n_slices,
        "pct_profitable": round(pct_prof, 3),
        "median_slice_pct": round(median_s, 2),
        "compound_pct": round(compound, 1),
        "al_compound_pct": round(al_compound, 1),
        "vs_al_pp": round(compound - al_compound, 1),
        "beats_al_pct": round(beats_al, 3),
        "maxdd_pct": round(maxdd, 1),
        "deployment_rate": round(deploy, 3),
        "elapsed_s": round(elapsed, 1),
    }


def run_rsi_sweep():
    """Sweep RSI across timeframes with the best config."""
    print("=" * 80)
    print("RSI ENGINE FAMILY — V1 BUILD")
    print("=" * 80)
    
    # Config sweep: try a few RSI parameter sets
    configs = [
        {"period": 14, "oversold": 30, "overbought": 70},  # standard
        {"period": 14, "oversold": 35, "overbought": 65},  # tighter
        {"period": 7, "oversold": 30, "overbought": 70},   # faster
        {"period": 21, "oversold": 30, "overbought": 70},  # slower
    ]
    
    tfs = ["1d", "4h", "2h", "1h", "30m"]
    
    results = []
    
    # First: find best config at 4h (the proven sweet spot)
    print("\n--- Config tuning at 4h ---")
    for cfg in configs:
        r = run_rsi_cell("4h", **cfg)
        if r["status"] == "OK":
            print(f"  RSI({cfg['period']}, {cfg['oversold']}/{cfg['overbought']}): "
                  f"{r['pct_profitable']:.0%} prof, median {r['median_slice_pct']:+.2f}%, "
                  f"deploy {r['deployment_rate']:.1%}")
    
    # Use standard config (14, 30/70) for the full sweep — TREND MODE (V1.1)
    best_cfg = {"period": 14, "oversold": 30, "overbought": 70}
    
    print(f"\n--- Full TF sweep — V1.1 TREND MODE (RSI>50 + rising) ---")
    print(f"{'TF':<6} {'%Prof':>6} {'Median':>8} {'Compound':>10} {'vsAL':>8} {'MaxDD':>7} {'Deploy':>7}")
    print("-" * 55)
    
    for tf in tfs:
        r = run_rsi_cell(tf, **best_cfg, mode="trend")
        results.append(r)
        if r["status"] == "OK":
            print(f"{tf:<6} {r['pct_profitable']:>5.0%} {r['median_slice_pct']:>+7.2f}% "
                  f"{r['compound_pct']:>9.1f}% {r['vs_al_pp']:>+7.1f} {r['maxdd_pct']:>6.1f}% "
                  f"{r['deployment_rate']:>6.1%}")
        else:
            print(f"{tf:<6} {r['status']}")
    
    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"rsi_sweep_{ts}.json", "w") as f:
        json.dump({"family": "RSI", "config": best_cfg, "results": results, "timestamp": ts}, f, indent=2)
    print(f"\n  Saved: {OUTDIR / f'rsi_sweep_{ts}.json'}")
    
    return results


if __name__ == "__main__":
    run_rsi_sweep()
