"""OOS Engine Test — rigorous held-out validation of V1 and V2 engines.

Tests the engines on the OOS window (2024-05-15 to 2024-12-31) which is
GENUINELY held-out: band configs were selected on TRAIN+VAL (2020-2024-05-15)
and NEVER re-fitted on this data.

Reports BOTH linear (mean daily/weekly) and compound returns, so we know
the realistic expectation without compounding bias.

Architecture:
1. Load data with wall at 2025-01-01 (extends into OOS)
2. Compute holds using FROZEN band configs (no re-fit)
3. Slice portfolio returns to ONLY the OOS window
4. Report per-slice and aggregate metrics
5. Compare to always-long baseline in same window

This is the PRE-REGISTERED honest test. Results here determine whether
the engines are deployable or require revision.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from datetime import date, datetime

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

BARS_PER_DAY = fl.BARS_PER_DAY
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)

OUTDIR = CRYPTO / "runs" / "engines" / "V2_subtf_decoupled" / "oos_validation"


def run_oos_cell(ma_type: str, tf: str, lookback_days: int = 7,
                 slice_days: int = 7, use_regime_gate: bool = True) -> dict:
    """Run one engine cell on the OOS window."""
    t0 = time.time()
    bpd = BARS_PER_DAY[tf]
    lookback_bars = lookback_days * bpd
    slice_bars = slice_days * bpd
    
    # Load with EXTENDED wall (includes OOS data)
    sub_band, sel_meta = select_band_trainval(ma_type, tf)
    provider = make_hold_provider(sub_band, ma_type)
    
    oos_start_ts = pd.Timestamp(OOS_START)
    
    # BTC regime (for combined V1+V2)
    btc_rec = _load_asset("BTCUSDT", "1d", OOS_WALL_MS)
    btc_regime = None
    if btc_rec and use_regime_gate:
        btc_dates, _, _, _, btc_close = btc_rec
        sma100 = pd.Series(btc_close, index=btc_dates).rolling(100).mean()
        regime_s = pd.Series(0, index=btc_dates, dtype=int)
        state = 0
        for i in range(100, len(btc_dates)):
            s = sma100.iloc[i]
            c = btc_close[i]
            if np.isnan(s):
                continue
            if state == 0 and c > s * 1.02:
                state = 1
            elif state == 1 and c < s * 0.98:
                state = 0
            regime_s.iloc[i] = state
        btc_regime = regime_s
    
    # Load all assets with extended wall
    symbols = _u100_symbols()
    close_dict = {}
    held_dict = {}
    
    for sym in symbols:
        rec = _load_asset(sym, tf, OOS_WALL_MS)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        
        # Compute holds (FROZEN band, no re-fit)
        band_held = provider(sym, o, h, l, c)
        
        # Decoupled signal
        fin = np.asarray(c, dtype=float)
        m = np.isfinite(fin)
        if m.sum() < max(5, lookback_bars // 4):
            continue
        cseg = fin[m]
        ema_lb = _ema(cseg, lookback_bars)
        above = cseg > ema_lb
        rising = np.concatenate([[True], ema_lb[1:] >= ema_lb[:-1]])
        bh_seg = np.asarray(band_held, dtype=bool)[m]
        confirmed = bh_seg & above & rising
        
        held = np.zeros(len(c), dtype=np.int8)
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
        held[np.where(m)[0]] = held_seg
        
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)
    
    n_assets = len(close_dict)
    
    # Build date universe and slice to OOS only
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    oos_dates = [d for d in all_dates if d >= oos_start_ts]
    
    if len(oos_dates) < 10:
        return {"status": "INSUFFICIENT_OOS_DATA", "n_oos_dates": len(oos_dates)}
    
    # Compute bar-by-bar returns in OOS window
    engine_rets = []
    al_rets = []
    combined_rets = []  # with regime gate
    
    for i in range(1, len(oos_dates)):
        dp = oos_dates[i-1]
        dc = oos_dates[i]
        
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
        
        # Combined: apply regime gate
        if btc_regime is not None:
            d_1d = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d_1d]
            regime_val = int(btc_regime[prior[-1]]) if len(prior) > 0 else 0
            if regime_val == 1:
                combined_rets.append(engine_rets[-1])
            else:
                combined_rets.append(0.0)
        else:
            combined_rets.append(engine_rets[-1])
    
    eng = np.array(engine_rets)
    al = np.array(al_rets)
    comb = np.array(combined_rets)
    
    # === METRICS ===
    n_bars = len(eng)
    n_days = n_bars / bpd
    
    # Per-slice
    eng_slices = []
    al_slices = []
    comb_slices = []
    i = 0
    while i + slice_bars <= n_bars:
        eng_slices.append(float(np.prod(1 + eng[i:i+slice_bars]) - 1) * 100)
        al_slices.append(float(np.prod(1 + al[i:i+slice_bars]) - 1) * 100)
        comb_slices.append(float(np.prod(1 + comb[i:i+slice_bars]) - 1) * 100)
        i += slice_bars
    
    def compute_metrics(rets, slices, label):
        compound = float(np.prod(1 + rets) - 1) * 100
        linear_daily = float(np.mean(rets)) * bpd * 100  # daily equivalent
        linear_weekly = linear_daily * 7
        cum = np.cumprod(1 + rets)
        maxdd = float(np.min((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum))) * 100
        pct_prof = sum(1 for s in slices if s > 0) / len(slices) if slices else 0
        median_s = float(np.median(slices)) if slices else 0
        win_rate_bar = float(np.mean(rets > 0))
        
        return {
            "label": label,
            "compound_pct": round(compound, 2),
            "linear_daily_pct": round(linear_daily, 4),
            "linear_weekly_pct": round(linear_weekly, 4),
            "maxdd_pct": round(maxdd, 1),
            "n_slices": len(slices),
            "pct_slices_profitable": round(pct_prof, 3),
            "median_slice_pct": round(median_s, 2),
            "mean_slice_pct": round(float(np.mean(slices)), 2) if slices else 0,
            "win_rate_bar": round(win_rate_bar, 3),
        }
    
    eng_metrics = compute_metrics(eng, eng_slices, "V2 engine (no gate)")
    al_metrics = compute_metrics(al, al_slices, "Always-long (passive)")
    comb_metrics = compute_metrics(comb, comb_slices, "Combined V1+V2 (regime-gated)")
    
    elapsed = time.time() - t0
    
    result = {
        "status": "OK",
        "cell": f"{ma_type}|{tf}|lb{lookback_days}d|sl{slice_days}d",
        "ma_type": ma_type, "tf": tf,
        "lookback_days": lookback_days, "slice_days": slice_days,
        "oos_window": f"{OOS_START} to {OOS_END}",
        "n_assets": n_assets,
        "n_oos_bars": n_bars,
        "n_oos_days": round(n_days, 1),
        "engine": eng_metrics,
        "always_long": al_metrics,
        "combined": comb_metrics,
        "edge_vs_al_compound_pp": round(eng_metrics["compound_pct"] - al_metrics["compound_pct"], 2),
        "combined_edge_vs_al_pp": round(comb_metrics["compound_pct"] - al_metrics["compound_pct"], 2),
        "elapsed_s": round(elapsed, 1),
    }
    
    return result


def run_oos_sweep():
    """Run OOS test on all verified cells."""
    print("=" * 80)
    print("OOS VALIDATION — 2024-05-15 to 2024-12-31 (7.5 months held-out)")
    print("=" * 80)
    print("Band configs FROZEN from TRAIN+VAL. NO re-fitting on this window.")
    print()
    
    cells = [
        ("EMA", "4h", 7, 7),
        ("WMA", "4h", 7, 7),
        ("DEMA", "4h", 7, 7),
        ("EMA", "4h", 14, 7),
        ("EMA", "2h", 7, 7),
        ("EMA", "1d", 14, 7),
    ]
    
    results = []
    for ma, tf, lb, sl in cells:
        print(f"\n--- {ma}|{tf}|lb{lb}d|sl{sl}d ---")
        r = run_oos_cell(ma, tf, lb, sl, use_regime_gate=True)
        results.append(r)
        
        if r["status"] != "OK":
            print(f"  {r['status']}")
            continue
        
        e = r["engine"]
        a = r["always_long"]
        c = r["combined"]
        print(f"  OOS period: {r['n_oos_days']:.0f} days, {r['n_assets']} assets")
        print(f"  V2 ENGINE:    compound {e['compound_pct']:+.2f}%, linear {e['linear_weekly_pct']:+.3f}%/wk, "
              f"slices {e['pct_slices_profitable']:.0%} prof, median {e['median_slice_pct']:+.2f}%")
        print(f"  ALWAYS-LONG:  compound {a['compound_pct']:+.2f}%, linear {a['linear_weekly_pct']:+.3f}%/wk")
        print(f"  COMBINED:     compound {c['compound_pct']:+.2f}%, linear {c['linear_weekly_pct']:+.3f}%/wk, "
              f"slices {c['pct_slices_profitable']:.0%} prof")
        print(f"  Edge vs AL:   {r['edge_vs_al_compound_pp']:+.2f}pp (engine), "
              f"{r['combined_edge_vs_al_pp']:+.2f}pp (combined)")
    
    # Summary table
    print("\n" + "=" * 80)
    print("OOS SUMMARY TABLE")
    print("=" * 80)
    print(f"\n{'Cell':<22} {'Comp%':>8} {'Lin/wk':>8} {'%Prof':>6} {'Med/sl':>7} {'vsAL':>7} {'MaxDD':>7}")
    print("-" * 70)
    
    for r in results:
        if r["status"] != "OK":
            continue
        cell = r["cell"]
        c = r["combined"]  # report the combined (regime-gated) version
        edge = r["combined_edge_vs_al_pp"]
        print(f"{cell:<22} {c['compound_pct']:>+7.2f}% {c['linear_weekly_pct']:>+7.3f}% "
              f"{c['pct_slices_profitable']:>5.0%} {c['median_slice_pct']:>+6.2f}% "
              f"{edge:>+6.2f} {c['maxdd_pct']:>6.1f}%")
    
    # Save
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = OUTDIR / f"oos_results_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({"results": results, "oos_window": f"{OOS_START} to {OOS_END}",
                   "timestamp": ts}, f, indent=2)
    print(f"\n  Saved: {outfile}")
    
    return results


if __name__ == "__main__":
    run_oos_sweep()
