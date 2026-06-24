"""TI Engine Multi-TF Runner — decoupled 7d-slice / {7d,14d}-lookback across all cadences.

The unified runner for the TI engine rebuild campaign. Wires the COMMITTED infra
(band selection, hold provider, wealth accounting) with the DECOUPLED architecture
(lookback != hold) at ALL available cadences {30m, 1h, 2h, 4h, 1d}.

Outputs per-cell JSON with slice-level metrics. Designed for autonomous batch execution.

Usage:
  python -m strat.ti_engine_multitf --ma-type EMA --tf 1h --lookback 14 --slice 7
  python -m strat.ti_engine_multitf --all  (run all cells)
  python -m strat.ti_engine_multitf --sweep-ema  (EMA × all TFs × all lookbacks)
"""
from __future__ import annotations

import argparse
import json
import sys
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
from strat.ma_movecatch_engine import select_band_trainval, make_hold_provider
from strat.ma_type_upgrade import _ema
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT

OUTDIR = CRYPTO / "runs" / "engines" / "V2_subtf_decoupled"
BARS_PER_DAY = fl.BARS_PER_DAY  # {'1d': 1, '4h': 6, '2h': 12, '1h': 24, '30m': 48, '15m': 96}


def _decoupled_held(c: np.ndarray, band_held: np.ndarray,
                    lookback_bars: int, slice_bars: int) -> np.ndarray:
    """Decoupled signal (lookback) + hold (slice). Identical to ma_wealth_driver_decoupled._decoupled_held."""
    n = len(c)
    out = np.zeros(n, dtype=np.int8)
    fin = np.asarray(c, dtype=float)
    m = np.isfinite(fin)
    if m.sum() < max(5, lookback_bars // 4):
        return out

    cseg = fin[m]
    ema_lb = _ema(cseg, lookback_bars)
    above = cseg > ema_lb
    rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])
    bh_seg = np.asarray(band_held, dtype=bool)[m]
    confirmed = bh_seg & above & rising

    held_seg = np.zeros(len(cseg), dtype=np.int8)
    i = 0
    L = len(cseg)
    while i < L:
        if confirmed[i]:
            end = min(i + slice_bars, L)
            j = i
            while j < end and confirmed[j]:
                held_seg[j] = 1
                j += 1
            i = j
        else:
            i += 1
    out[np.where(m)[0]] = held_seg
    return out


def run_cell(ma_type: str, tf: str, lookback_days: int = 14,
             slice_days: int = 7, verbose: bool = True) -> dict:
    """Run one (ma_type, tf, lookback, slice) cell and return per-slice metrics."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd

    if verbose:
        print(f"  [{ma_type}|{tf}|lb{lookback_days}d|sl{slice_days}d] "
              f"lookback={lookback_bars}bars, slice={slice_bars}bars")

    # Band selection
    try:
        sub_band, sel_meta = select_band_trainval(ma_type, tf)
    except KeyError as e:
        return {"status": "NO_BAND", "error": str(e), "ma_type": ma_type, "tf": tf}

    provider = make_hold_provider(sub_band, ma_type)
    wall_ms = int(pd.Timestamp(fl.DEV_END).value // 10**6)

    # Load u100 panel and compute holds
    symbols = _u100_symbols()
    per_asset_data = {}  # sym -> (dates_idx, close, held_decoupled)

    n_loaded = 0
    for sym in symbols:
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        # Committed band ensemble held state
        band_held = provider(sym, o, h, l, c)
        # Decoupled: 14d lookback signal, 7d slice hold
        held = _decoupled_held(c, band_held, lookback_bars, slice_bars)
        per_asset_data[sym] = (dates, c, held)
        n_loaded += 1

    if n_loaded < 20:
        return {"status": "INSUFFICIENT_ASSETS", "n_loaded": n_loaded,
                "ma_type": ma_type, "tf": tf}

    if verbose:
        print(f"    Loaded {n_loaded} assets")

    # Build aligned panel — use pandas date-indexed approach
    # Create a common date index
    all_dates_set = set()
    for sym, (dates, _, _) in per_asset_data.items():
        all_dates_set.update(dates.tolist())
    all_dates = sorted(all_dates_set)

    # Convert per-asset data to pandas Series for fast lookup
    close_dict = {}  # sym -> pd.Series(close, index=dates)
    held_dict = {}   # sym -> pd.Series(held, index=dates)
    for sym, (dates, close, held) in per_asset_data.items():
        close_dict[sym] = pd.Series(close, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)

    # Per-bar portfolio: equal-weight among HELD assets, cash otherwise
    bar_nets = []
    bar_dates = []
    n_active_bars = 0

    for i in range(1, len(all_dates)):
        d_prev = all_dates[i - 1]
        d_curr = all_dates[i]
        bar_dates.append(d_curr)

        held_rets = []
        for sym in close_dict:
            if d_prev not in held_dict[sym].index or d_curr not in close_dict[sym].index:
                continue
            if d_prev not in close_dict[sym].index:
                continue

            h_prev = int(held_dict[sym].get(d_prev, 0))
            if h_prev == 1:
                c_prev = close_dict[sym][d_prev]
                c_curr = close_dict[sym].get(d_curr, np.nan)
                if c_prev > 0 and not np.isnan(c_curr):
                    held_rets.append((c_curr / c_prev) - 1.0)

        if not held_rets:
            bar_nets.append(0.0)
            continue

        n_active_bars += 1
        ew_ret = float(np.mean(held_rets))
        # Simplified cost: count as 1 turnover event per bar with activity
        # (conservative: assumes some entry/exit each active bar)
        cost = MAKER_RT / 2.0 / max(1, len(held_rets)) if len(held_rets) < n_loaded else 0
        bar_nets.append(ew_ret - cost)

    bar_nets = np.array(bar_nets)

    # Always-long baseline
    al_bar_nets = []
    for i in range(1, len(all_dates)):
        d_prev = all_dates[i - 1]
        d_curr = all_dates[i]
        al_rets = []
        for sym in close_dict:
            if d_prev in close_dict[sym].index and d_curr in close_dict[sym].index:
                c_prev = close_dict[sym][d_prev]
                c_curr = close_dict[sym][d_curr]
                if c_prev > 0:
                    al_rets.append((c_curr / c_prev) - 1.0)
        al_bar_nets.append(float(np.mean(al_rets)) if al_rets else 0.0)
    al_bar_nets = np.array(al_bar_nets)

    # === PER-SLICE EVALUATION ===
    n_bars = len(bar_nets)
    slices = []
    i = 0
    while i + slice_bars <= n_bars:
        slice_net = bar_nets[i:i + slice_bars]
        slice_compound = float(np.prod(1 + slice_net) - 1)
        slice_active = int(np.sum(slice_net != 0))
        slices.append({
            "start_idx": i,
            "compound_pct": slice_compound * 100,
            "n_active_bars": slice_active,
        })
        i += slice_bars

    al_slices = []
    i = 0
    while i + slice_bars <= len(al_bar_nets):
        al_compound = float(np.prod(1 + al_bar_nets[i:i + slice_bars]) - 1)
        al_slices.append(al_compound * 100)
        i += slice_bars

    # === METRICS ===
    slice_compounds = [s["compound_pct"] for s in slices]
    n_slices = len(slice_compounds)
    n_profitable = sum(1 for s in slice_compounds if s > 0)
    pct_profitable = n_profitable / n_slices if n_slices > 0 else 0

    # vs always-long
    vs_al = [s - a for s, a in zip(slice_compounds, al_slices)] if len(al_slices) == len(slice_compounds) else []
    n_beats_al = sum(1 for v in vs_al if v > 0) if vs_al else 0

    elapsed = time.time() - t0

    result = {
        "ma_type": ma_type,
        "tf": tf,
        "lookback_days": lookback_days,
        "slice_days": slice_days,
        "lookback_bars": lookback_bars,
        "slice_bars": slice_bars,
        "n_assets": n_loaded,
        "n_bars_total": n_bars,
        "n_active_bars": n_active_bars,
        "deployment_rate": n_active_bars / n_bars if n_bars > 0 else 0,
        "n_slices": n_slices,
        "n_profitable_slices": n_profitable,
        "pct_profitable": pct_profitable,
        "median_slice_pct": float(np.median(slice_compounds)) if slice_compounds else 0,
        "mean_slice_pct": float(np.mean(slice_compounds)) if slice_compounds else 0,
        "std_slice_pct": float(np.std(slice_compounds)) if slice_compounds else 0,
        "best_slice_pct": float(np.max(slice_compounds)) if slice_compounds else 0,
        "worst_slice_pct": float(np.min(slice_compounds)) if slice_compounds else 0,
        "full_compound_pct": float(np.prod(1 + bar_nets) - 1) * 100,
        "full_compound_al_pct": float(np.prod(1 + al_bar_nets) - 1) * 100,
        "vs_always_long_pp": float(np.prod(1 + bar_nets) - 1) * 100 - float(np.prod(1 + al_bar_nets) - 1) * 100,
        "pct_beats_always_long": n_beats_al / n_slices if n_slices > 0 else 0,
        "mean_vs_al_pp": float(np.mean(vs_al)) if vs_al else 0,
        "elapsed_s": elapsed,
        "status": "OK",
    }

    if verbose:
        print(f"    Slices: {n_slices} | Profitable: {pct_profitable:.0%} | "
              f"Median: {result['median_slice_pct']:+.2f}% | "
              f"vs AL: {result['vs_always_long_pp']:+.1f}pp | "
              f"[{elapsed:.1f}s]")

    return result


def run_sweep(ma_types=None, tfs=None, lookbacks=None, slices=None, verbose=True):
    """Run a sweep of cells."""
    if ma_types is None:
        ma_types = ["EMA", "HMA", "DEMA"]
    if tfs is None:
        tfs = ["1d", "4h", "2h", "1h", "30m"]
    if lookbacks is None:
        lookbacks = [7, 14]
    if slices is None:
        slices = [7]

    print("=" * 80)
    print("TI ENGINE MULTI-TF SWEEP")
    print(f"MA types: {ma_types}")
    print(f"TFs: {tfs}")
    print(f"Lookbacks: {lookbacks}d | Slices: {slices}d")
    print("=" * 80)

    results = []
    total = len(ma_types) * len(tfs) * len(lookbacks) * len(slices)
    done = 0

    for ma in ma_types:
        for tf in tfs:
            for lb in lookbacks:
                for sl in slices:
                    done += 1
                    print(f"\n[{done}/{total}] {ma}|{tf}|lb{lb}d|sl{sl}d")
                    r = run_cell(ma, tf, lb, sl, verbose=verbose)
                    results.append(r)

                    # Save individual cell
                    if r.get("status") == "OK":
                        cell_dir = OUTDIR / "cells"
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"{ma}_{tf}_{lb}d_{sl}d.json"
                        with open(cell_dir / fname, "w") as f:
                            json.dump(r, f, indent=2)

    # Summary
    print("\n" + "=" * 80)
    print("SWEEP RESULTS")
    print("=" * 80)

    ok_results = [r for r in results if r.get("status") == "OK"]
    print(f"\n{'Cell':<20} {'Slices':>6} {'%Prof':>6} {'Median':>8} {'Mean':>8} {'vsAL':>8} {'Deploy':>7}")
    print("-" * 75)

    for r in sorted(ok_results, key=lambda x: -x["median_slice_pct"]):
        cell = f"{r['ma_type']}|{r['tf']}|{r['lookback_days']}d"
        print(f"{cell:<20} {r['n_slices']:>6} {r['pct_profitable']:>5.0%} "
              f"{r['median_slice_pct']:>+7.2f}% {r['mean_slice_pct']:>+7.2f}% "
              f"{r['vs_always_long_pp']:>+7.1f} {r['deployment_rate']:>6.1%}")

    # Best cells
    if ok_results:
        best_median = max(ok_results, key=lambda x: x["median_slice_pct"])
        best_pct = max(ok_results, key=lambda x: x["pct_profitable"])
        best_vs_al = max(ok_results, key=lambda x: x["vs_always_long_pp"])

        print(f"\nBEST by median slice ROI: {best_median['ma_type']}|{best_median['tf']}|{best_median['lookback_days']}d = {best_median['median_slice_pct']:+.2f}%")
        print(f"BEST by % profitable: {best_pct['ma_type']}|{best_pct['tf']}|{best_pct['lookback_days']}d = {best_pct['pct_profitable']:.0%}")
        print(f"BEST vs always-long: {best_vs_al['ma_type']}|{best_vs_al['tf']}|{best_vs_al['lookback_days']}d = {best_vs_al['vs_always_long_pp']:+.1f}pp")

    # Save summary
    summary_dir = OUTDIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(summary_dir / f"sweep_{ts}.json", "w") as f:
        json.dump({"results": results, "timestamp": ts}, f, indent=2)

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ma-type", default="EMA")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--lookback", type=int, default=14)
    ap.add_argument("--slice", type=int, default=7)
    ap.add_argument("--sweep-ema", action="store_true", help="Sweep EMA across all TFs")
    ap.add_argument("--sweep-top3", action="store_true", help="Sweep EMA+HMA+DEMA across all TFs")
    ap.add_argument("--all", action="store_true", help="All 8 types × all TFs")
    args = ap.parse_args()

    if args.all:
        run_sweep(ma_types=["EMA", "SMA", "WMA", "HMA", "DEMA", "TEMA", "KAMA", "VIDYA"])
    elif args.sweep_top3:
        run_sweep(ma_types=["EMA", "HMA", "DEMA"])
    elif args.sweep_ema:
        run_sweep(ma_types=["EMA"])
    else:
        r = run_cell(args.ma_type, args.tf, args.lookback, args.slice)
        print(json.dumps(r, indent=2))
