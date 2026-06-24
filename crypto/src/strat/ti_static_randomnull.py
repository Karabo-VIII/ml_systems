"""Random-null + DEV/OOS consistency for the standout STATIC non-MA cells.

The static exhaustion test found that RSI/MACD/Bollinger DO beat always-long when
the confirmation layer + vote threshold are properly engaged (the levers the first
SOTA sweep never pulled). BUT that test selected cells by looking at OOS, so it risks
OOS-selection bias.

This script applies the GOLD-STANDARD bar the MA family cleared (and non-MA never had):

  1. DEV/OOS CONSISTENCY: the edge must show on BOTH the DEV window (<=2024-05-15) and
     the OOS window (2024-05-15->2025-01-01). A cell strong on both, robust to the
     threshold knob, was selectable BEFORE seeing OOS — that defuses selection bias.

  2. RANDOM-NULL: replace the real ensemble signal with a RANDOM held-series at the
     SAME per-asset deploy rate, same regime gate, same confirmation+slice-hold. Repeat
     N times. The real edge must beat the random-null distribution (target >=95th pct).
     This proves it's the TIMING, not just being-long-under-the-gate.

Cells under test (the threshold-robust standouts):
  MACD|1h+confirm, BOLLINGER|1d+confirm, RSI|4h.
"""
from __future__ import annotations

import warnings; warnings.filterwarnings("ignore")
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
from strat.ma_wealth_driver import _load_asset, _u100_symbols, MAKER_RT
from strat.ti_static_exhaustion import ensemble_votes, apply_confirmation, _btc_regime

BARS_PER_DAY = fl.BARS_PER_DAY
DEV_END = "2024-05-15"
OOS_START = "2024-05-15"
OOS_END = "2025-01-01"
DEV_WALL_MS = int(pd.Timestamp(DEV_END).value // 10**6)
OOS_WALL_MS = int(pd.Timestamp(OOS_END).value // 10**6)
OUTDIR = CRYPTO / "runs" / "engines" / "V3_ti_families" / "static_randomnull"

# (family, tf, threshold, confirm)
CELLS = [
    ("MACD", "1h", 0.6, True),
    ("BOLLINGER", "1d", 0.6, True),
    ("RSI", "4h", 0.7, False),
]
N_NULL = 200


def _build_panel(family, tf, threshold, confirm, wall_ms, slice_days=7, lookback_days=7):
    """Load panel, compute real held + per-asset deploy rates."""
    bpd = BARS_PER_DAY[tf]
    slice_bars = slice_days * bpd
    lookback_bars = lookback_days * bpd
    close_dict, held_dict, deploy_rates = {}, {}, {}
    for sym in _u100_symbols():
        rec = _load_asset(sym, tf, wall_ms)
        if rec is None:
            continue
        dates, o, h, l, c = rec
        frac = ensemble_votes(family, np.asarray(c, dtype=float))
        raw = (frac > threshold).astype(np.int8)
        held = apply_confirmation(c, raw, lookback_bars, slice_bars) if confirm else raw
        close_dict[sym] = pd.Series(c, index=dates)
        held_dict[sym] = pd.Series(held, index=dates)
        deploy_rates[sym] = float(np.mean(held))
    return close_dict, held_dict, deploy_rates, slice_bars


def _portfolio_linwk(close_dict, held_dict, btc_regime, window_start, window_end, bpd):
    """Linear %/wk of the regime-gated EW engine over [window_start, window_end)."""
    all_dates = sorted(set().union(*[s.index.tolist() for s in close_dict.values()]))
    ws = pd.Timestamp(window_start); we = pd.Timestamp(window_end)
    win = [d for d in all_dates if ws <= d < we]
    if len(win) < 10:
        return None, None
    eng, al = [], []
    for i in range(1, len(win)):
        dp, dc = win[i-1], win[i]
        er, ar = [], []
        for sym in close_dict:
            if dp in close_dict[sym].index and dc in close_dict[sym].index:
                cp = close_dict[sym][dp]; cc = close_dict[sym][dc]
                if cp > 0:
                    r = (cc/cp) - 1.0
                    ar.append(r)
                    if dp in held_dict[sym].index and held_dict[sym][dp] == 1:
                        er.append(r)
        e = float(np.mean(er)) if er else 0.0
        if btc_regime is not None:
            d1 = pd.Timestamp(dp).normalize()
            prior = btc_regime.index[btc_regime.index <= d1]
            if len(prior) and int(btc_regime[prior[-1]]) == 0:
                e = 0.0
        eng.append(e); al.append(float(np.mean(ar)) if ar else 0.0)
    eng = np.array(eng); al = np.array(al)
    return float(np.mean(eng)) * bpd * 7 * 100, float(np.mean(al)) * bpd * 7 * 100


def _random_held_like(held_dict, deploy_rates, slice_bars, rng):
    """Random held-series per asset matching each asset's deploy rate, in slice-blocks."""
    out = {}
    for sym, hs in held_dict.items():
        n = len(hs)
        dr = deploy_rates[sym]
        rh = np.zeros(n, dtype=np.int8)
        # block-random: decide per slice-block to be on/off at prob dr (preserves hold structure)
        i = 0
        while i < n:
            if rng.random() < dr:
                rh[i:min(i+slice_bars, n)] = 1
            i += slice_bars
        out[sym] = pd.Series(rh, index=hs.index)
    return out


def run_cell(family, tf, threshold, confirm):
    print(f"\n{'='*84}\n  {family}|{tf} thr{threshold} conf{confirm}\n{'='*84}")
    bpd = BARS_PER_DAY[tf]

    # --- DEV window (load DEV-walled) ---
    cd_dev, hd_dev, dr_dev, sb = _build_panel(family, tf, threshold, confirm, DEV_WALL_MS)
    btc_reg_dev = _btc_regime(DEV_WALL_MS)
    dev_eng, dev_al = _portfolio_linwk(cd_dev, hd_dev, btc_reg_dev, "2020-01-01", DEV_END, bpd)

    # --- OOS window (load OOS-walled) ---
    cd, hd, dr, sb = _build_panel(family, tf, threshold, confirm, OOS_WALL_MS)
    btc_reg = _btc_regime(OOS_WALL_MS)
    oos_eng, oos_al = _portfolio_linwk(cd, hd, btc_reg, OOS_START, OOS_END, bpd)

    print(f"  DEV  edge: {dev_eng - dev_al:+.3f}pp  (eng {dev_eng:+.3f}%/wk vs AL {dev_al:+.3f}%/wk)")
    print(f"  OOS  edge: {oos_eng - oos_al:+.3f}pp  (eng {oos_eng:+.3f}%/wk vs AL {oos_al:+.3f}%/wk)")
    consistent = (dev_eng - dev_al > 0) and (oos_eng - oos_al > 0)
    print(f"  DEV/OOS consistent (both positive): {consistent}")

    # --- Random-null on OOS ---
    rng = np.random.default_rng(42)
    null_edges = []
    for _ in range(N_NULL):
        rhd = _random_held_like(hd, dr, sb, rng)
        r_eng, r_al = _portfolio_linwk(cd, rhd, btc_reg, OOS_START, OOS_END, bpd)
        if r_eng is not None:
            null_edges.append(r_eng - r_al)
    null_edges = np.array(null_edges)
    real_edge = oos_eng - oos_al
    pct = float(np.mean(null_edges < real_edge)) * 100
    print(f"  Random-null (n={len(null_edges)}): real edge {real_edge:+.3f}pp vs "
          f"null mean {null_edges.mean():+.3f}pp (sd {null_edges.std():.3f})")
    print(f"  --> Real beats {pct:.0f}th percentile of random-null  "
          f"{'PASS' if pct >= 95 else 'FAIL (edge is just being-long-under-gate)'}")

    return {
        "family": family, "tf": tf, "threshold": threshold, "confirm": confirm,
        "dev_edge_pp": round(dev_eng - dev_al, 3), "oos_edge_pp": round(real_edge, 3),
        "dev_oos_consistent": consistent,
        "null_mean_pp": round(float(null_edges.mean()), 3),
        "null_sd": round(float(null_edges.std()), 3),
        "real_vs_null_pct": round(pct, 1),
        "random_null_pass": bool(pct >= 95),
    }


def main():
    print("=" * 84)
    print("STATIC RANDOM-NULL + DEV/OOS CONSISTENCY — the gold-standard bar")
    print("=" * 84)
    print("A cell is REAL static edge if: DEV/OOS consistent AND beats random-null >=95th pct.")
    results = []
    for cell in CELLS:
        results.append(run_cell(*cell))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(OUTDIR / f"static_randomnull_{ts}.json", "w") as f:
        json.dump({"results": results, "n_null": N_NULL, "timestamp": ts}, f, indent=2)

    print(f"\n{'='*84}\n  SUMMARY\n{'='*84}")
    for r in results:
        verdict = "REAL EDGE" if (r["dev_oos_consistent"] and r["random_null_pass"]) else "not confirmed"
        print(f"  {r['family']}|{r['tf']}: DEV {r['dev_edge_pp']:+.2f}pp, OOS {r['oos_edge_pp']:+.2f}pp, "
              f"null {r['real_vs_null_pct']:.0f}pct -> {verdict}")
    print(f"\n  Saved: static_randomnull_{ts}.json")


if __name__ == "__main__":
    main()
