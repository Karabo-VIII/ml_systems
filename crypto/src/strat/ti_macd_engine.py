"""MACD Engine Family — TI-based trading engine using MACD signal-line cross.

Signal Logic:
  LONG when: MACD line crosses ABOVE signal line (momentum increasing)
  EXIT when: MACD line crosses BELOW signal line OR slice boundary

Key difference from MA band:
  - MA band: is price above/below the MA? (level)
  - MACD: is the RATE OF CHANGE of MA-spread increasing? (derivative/momentum-of-momentum)
  
MACD = EMA(fast) - EMA(slow) → measures how fast the price is diverging from its trend
Signal = EMA(MACD, signal_period) → smoothed MACD (removes noise)
Histogram = MACD - Signal → the momentum-of-momentum (acceleration)

Standard params: fast=12, slow=26, signal=9
Crypto-adapted variants: faster (8,21,5), slower (26,52,9)
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
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "MACD"


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """Compute MACD, signal line, and histogram. All causal."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def macd_held_state(close: np.ndarray, fast: int = 12, slow: int = 26, 
                    signal: int = 9, slice_bars: int = 42,
                    mode: str = "cross") -> np.ndarray:
    """MACD-based held state.
    
    Modes:
    - "cross": Long when MACD > Signal (bullish cross), exit when MACD < Signal
    - "histogram": Long when histogram > 0 AND rising (momentum accelerating)
    - "zero_cross": Long when MACD > 0 (above zero line = bullish trend)
    """
    n = len(close)
    held = np.zeros(n, dtype=np.int8)
    
    fin = np.isfinite(close)
    if fin.sum() < slow + signal + 5:
        return held
    
    macd_line, signal_line, histogram = compute_macd(close, fast, slow, signal)
    
    warmup = slow + signal  # bars needed before MACD is valid
    
    if mode == "cross":
        # Standard: long while MACD > Signal (bullish momentum)
        bullish = macd_line > signal_line
        
        i = warmup
        while i < n:
            if fin[i] and bullish[i]:
                end = min(i + slice_bars, n)
                j = i
                while j < end and bullish[j]:
                    held[j] = 1
                    j += 1
                i = j
            else:
                i += 1
    
    elif mode == "histogram":
        # Long when histogram > 0 AND rising (accelerating momentum)
        hist_pos = histogram > 0
        hist_rising = np.zeros(n, dtype=bool)
        for i in range(1, n):
            hist_rising[i] = histogram[i] > histogram[i-1]
        
        confirmed = hist_pos & hist_rising & fin
        
        i = warmup
        while i < n:
            if confirmed[i]:
                end = min(i + slice_bars, n)
                j = i
                while j < end:
                    if histogram[j] < 0:  # momentum reversed
                        break
                    held[j] = 1
                    j += 1
                i = j
            else:
                i += 1
    
    elif mode == "zero_cross":
        # Long when MACD > 0 (price above its trend = bullish)
        above_zero = macd_line > 0
        
        i = warmup
        while i < n:
            if fin[i] and above_zero[i]:
                end = min(i + slice_bars, n)
                j = i
                while j < end and above_zero[j]:
                    held[j] = 1
                    j += 1
                i = j
            else:
                i += 1
    
    return held


def run_macd_cell(tf: str, fast: int = 12, slow: int = 26, signal: int = 9,
                  slice_days: int = 7, mode: str = "cross") -> dict:
    """Run one MACD engine cell."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
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
        held = macd_held_state(fin, fast, slow, signal, slice_bars, mode)
        
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)
    
    n_assets = len(close_dict)
    if n_assets < 20:
        return {"status": "INSUFFICIENT_ASSETS", "n_assets": n_assets}
    
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    
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
        "config": {"tf": tf, "fast": fast, "slow": slow, "signal": signal, "mode": mode},
        "n_assets": n_assets, "n_slices": n_slices,
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


def run_macd_sweep():
    """Run MACD across all modes and timeframes."""
    print("=" * 80)
    print("MACD ENGINE FAMILY — V1 BUILD")
    print("=" * 80)
    
    tfs = ["1d", "4h", "2h", "1h", "30m"]
    modes = ["cross", "histogram", "zero_cross"]
    
    all_results = []
    
    for mode in modes:
        print(f"\n--- Mode: {mode} (standard 12/26/9) ---")
        print(f"{'TF':<6} {'%Prof':>6} {'Median':>8} {'Compound':>10} {'vsAL':>8} {'MaxDD':>7} {'Deploy':>7}")
        print("-" * 55)
        
        for tf in tfs:
            r = run_macd_cell(tf, mode=mode)
            all_results.append(r)
            if r["status"] == "OK":
                print(f"{tf:<6} {r['pct_profitable']:>5.0%} {r['median_slice_pct']:>+7.2f}% "
                      f"{r['compound_pct']:>9.1f}% {r['vs_al_pp']:>+7.1f} {r['maxdd_pct']:>6.1f}% "
                      f"{r['deployment_rate']:>6.1%}")
    
    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"macd_sweep_{ts}.json", "w") as f:
        json.dump({"family": "MACD", "results": all_results, "timestamp": ts}, f, indent=2)
    print(f"\n  Saved: {OUTDIR / f'macd_sweep_{ts}.json'}")
    
    return all_results


if __name__ == "__main__":
    run_macd_sweep()
