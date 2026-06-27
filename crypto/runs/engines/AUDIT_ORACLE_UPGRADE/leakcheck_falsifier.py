"""LEAK-CHECK FALSIFIER for the V1 (1d BTC SMA100 regime gate) -> V2 (4h MA band) book.

The make-or-break check (Claude audit, 2026-06-28). ma_slice_engine.run_cell broadcasts the
1d BTC regime onto the engine's (4h/1d/...) grid like this:

    norm = pd.DatetimeIndex(dates).normalize()              # 4h bar -> its OWN day-start
    idx  = regime.index.union(norm).drop_duplicates()
    ra   = regime.reindex(idx).ffill().reindex(norm).fillna(0)

regime.index are floor('D') of 1d CLOSE-time bars; regime[D] reflects BTC's day-D CLOSE.
A 4h bar at 2024-06-18 04:00 (closes ~04:00) normalizes to 2024-06-18 00:00 and thus reads
regime[2024-06-18] = the day-D close regime, which is NOT known until ~23:59 of day D.
=> SAME-DAY LOOK-AHEAD LEAK (the exact V1.1 align_htf_on START-label bug).

CAUSAL FIX (de-leak): a bar on day D must read the most recent CLOSED 1d bar = regime[D-1].
Implement by SHIFTING the regime series forward by one 1d bar before the broadcast.

This script re-runs run_cell's EXACT engine path but swaps ONLY the regime alignment
(leaked vs strict-lag), reusing the verified _load_asset / band / canonical exits / MtM, on
OOS-2024 (2025 SEALED). Reports lin_wk and pct_profitable_7d_slices BEFORE vs AFTER.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(ROOT))

import strat.ma_slice_engine as E
from strat.ma_slice_engine import (
    btc_regime, _load_asset, _u100_symbols, _cached_band_held, held_rolling, held_fixed,
    BARS_PER_DAY, OOS_WALL_MS, VAL_WALL_MS, OOS_START, OOS_2024_END, VAL_START, VAL_END, TIER,
)
from strat.canonical_exits import COST_TAKER, COST_PROD


def _align_regime(regime, dates, mode):
    """mode='leaked' reproduces run_cell EXACTLY. mode='lag1' strict-lags the 1d regime by one
    full 1d bar (causal: a bar on day D reads regime[D-1] = the last CLOSED 1d bar)."""
    if regime is None:
        return np.ones(len(dates))
    norm = pd.DatetimeIndex(dates).normalize()
    if mode == "lag1":
        regime = regime.shift(1)          # day D now reads the PRIOR closed 1d bar
    idx = regime.index.union(norm).drop_duplicates()
    ra = regime.reindex(idx).ffill().reindex(norm).fillna(0)
    return ra.to_numpy().astype(float)


def run_cell_mode(ma_type, tf, mode, *, slice_mode="rolling", lookback_days=14, slice_days=7,
                  cost_rt=COST_PROD, window="OOS"):
    """Mirror of ma_slice_engine.run_cell with the regime alignment selected by `mode`
    ('leaked' | 'lag1'). Everything else (band, exits, lag-1 MtM) is the verified path."""
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    wall_ms = OOS_WALL_MS if window == "OOS" else VAL_WALL_MS
    win_s = pd.Timestamp(OOS_START if window == "OOS" else VAL_START)
    win_e = pd.Timestamp(OOS_2024_END if window == "OOS" else VAL_END)

    regime = btc_regime(wall_ms)
    held_fn = held_fixed if slice_mode == "fixed" else held_rolling

    close_cols, held_cols = {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        band_held = _cached_band_held(ma_type, tf, sym, o, h, l, c, wall_ms)
        held = held_fn(c, band_held, lookback_bars, slice_bars)
        close_cols[sym] = pd.Series(c, index=dates)
        held_cols[sym] = pd.Series(held, index=dates)
    if len(close_cols) < 10:
        return {"status": "INSUFFICIENT_ASSETS"}

    close_df = pd.DataFrame(close_cols).sort_index()
    held_df = pd.DataFrame(held_cols).reindex(close_df.index).fillna(0).astype(np.int8)
    rets = close_df.pct_change().to_numpy()[1:]
    held_lag = held_df.shift(1).fillna(0).to_numpy()[1:]
    turn = held_df.diff().abs().fillna(0).to_numpy()[1:]
    dates = close_df.index[1:]

    reg_vals = _align_regime(regime, dates, mode)

    mask = (dates >= win_s) & (dates < win_e)
    valid = np.isfinite(rets)
    rets_f = np.where(valid, rets, 0.0)
    n_held = (held_lag * valid).sum(axis=1)
    eng_gross = np.where(n_held > 0, (rets_f * held_lag * valid).sum(axis=1) / np.maximum(n_held, 1), 0.0)
    cost = np.where(n_held > 0, (turn * valid).sum(axis=1) * (cost_rt / 2.0) / np.maximum(n_held, 1), 0.0)
    eng = (eng_gross - cost) * reg_vals

    eng_w = eng[mask]
    deploy = float(np.mean(n_held[mask] > 0)) if mask.sum() else 0.0
    if len(eng_w) == 0:
        return {"status": "NO_BARS"}
    lin_wk = float(np.mean(eng_w)) * bpd * 7 * 100
    comp = float(np.prod(1 + eng_w) - 1) * 100
    cum = np.cumprod(1 + eng_w)
    maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100 if len(cum) > 1 else 0
    sharpe = (float(np.mean(eng_w)) * bpd * 365) / (float(np.std(eng_w)) * np.sqrt(bpd * 365)) if np.std(eng_w) > 1e-12 else 0
    slice_pnls = []
    i = 0
    while i + slice_bars <= len(eng_w):
        slice_pnls.append(float(np.prod(1 + eng_w[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    pct_prof = sum(1 for s in slice_pnls if s > 0) / len(slice_pnls) if slice_pnls else 0
    # fraction of OOS bars where the two regimes would differ (the leak's footprint)
    reg_leak = _align_regime(regime, dates, "leaked")[mask]
    reg_lag = _align_regime(regime, dates, "lag1")[mask]
    frac_diff = float(np.mean(reg_leak != reg_lag)) if mask.sum() else 0.0
    return {"status": "OK", "ma_type": ma_type, "tf": tf, "mode": mode,
            "deploy": round(deploy, 4), "lin_wk": round(lin_wk, 4), "compound": round(comp, 1),
            "maxdd": round(maxdd, 1), "sharpe": round(sharpe, 3),
            "pct_profitable_7d_slices": round(pct_prof, 3), "n_slices": len(slice_pnls),
            "regime_bars_changed_frac": round(frac_diff, 4), "n_assets": len(close_cols)}


def main():
    t0 = time.time()
    # The deployable cells (CURRENT.md / SLICE_STUDY): 4h sweet spot + 1d best-slice.
    cells = [("SMA", "4h"), ("VIDYA", "4h"), ("EMA", "4h"), ("WMA", "4h"), ("KAMA", "4h"),
             ("VIDYA", "1d"), ("SMA", "1d"), ("EMA", "1d"), ("DEMA", "1d")]
    rows = []
    print("="*100)
    print("LEAK-CHECK FALSIFIER — V1 1d-BTC-SMA100 regime gate -> V2 engine, OOS-2024, PROD cost")
    print("leaked = same-day regime (current code) | lag1 = strict-lag (causal, reads last CLOSED 1d bar)")
    print("="*100)
    print(f"  {'cell':<11}{'deploy_L':>9}{'lin_wk_L':>10}{'lin_wk_C':>10}{'delta':>9}"
          f"{'7dprof_L':>9}{'7dprof_C':>9}{'reg_chg':>9}")
    print("  " + "-"*86)
    for ma, tf in cells:
        rl = run_cell_mode(ma, tf, "leaked")
        rc = run_cell_mode(ma, tf, "lag1")
        if rl["status"] != "OK" or rc["status"] != "OK":
            print(f"  {ma}|{tf}: {rl.get('status')} / {rc.get('status')}"); continue
        delta = round(rc["lin_wk"] - rl["lin_wk"], 3)
        print(f"  {ma+'|'+tf:<11}{rl['deploy']:>8.1%}{rl['lin_wk']:>+10.3f}{rc['lin_wk']:>+10.3f}"
              f"{delta:>+9.3f}{rl['pct_profitable_7d_slices']:>8.0%}{rc['pct_profitable_7d_slices']:>8.0%}"
              f"{rl['regime_bars_changed_frac']:>8.1%}")
        rows.append({"cell": f"{ma}|{tf}", "leaked": rl, "lag1": rc,
                     "lin_wk_delta_pp": delta,
                     "lin_wk_pct_retained": (round(rc["lin_wk"]/rl["lin_wk"]*100, 1)
                                             if abs(rl["lin_wk"]) > 1e-9 else None)})
    out = {"check": "V1_regime_gate_to_V2_engine_HTF_alignment_leak",
           "verdict_mechanism": (
               "ma_slice_engine.run_cell broadcasts regime via norm=dates.normalize(); "
               "ra=regime.reindex(idx).ffill().reindex(norm). regime.index=floor('D') of 1d "
               "CLOSE-time bars (regime[D]=day-D close). An intraday bar on day D normalizes to "
               "day-D 00:00 and reads regime[D] = the day-D daily CLOSE regime, unknown until "
               "~23:59 of day D => SAME-DAY LOOK-AHEAD LEAK."),
           "fix": "shift the 1d regime forward one full 1d bar (day D reads regime[D-1], last CLOSED 1d bar)",
           "window": "OOS-2024 (2024-05-15..2024-12-31); 2025 SEALED", "cost": "PROD (taker+slip 0.0028 RT)",
           "cells": rows, "elapsed_s": round(time.time()-t0, 1)}
    p = Path(__file__).resolve().parent / "leakcheck.json"
    json.dump(out, open(p, "w"), indent=1, default=str)
    print(f"\n[json] {p}")
    print(f"[elapsed] {out['elapsed_s']}s")


if __name__ == "__main__":
    main()
