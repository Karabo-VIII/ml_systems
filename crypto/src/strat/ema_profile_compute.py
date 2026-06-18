"""ema_profile_compute.py -- compute the full EMA profile: block-bootstrap p05 + turnover per TF.

Uses the exact same ironed sleeve as the leaderboard. EMA working band ensemble.
Reports: OOS net, OOS maxdd, block-bootstrap p05, turnover (round-trips per period), coverage (pct time in position).
RWYB: python -m strat.ema_profile_compute
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.seterr(invalid="ignore", divide="ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import strat.ma_2020_config_leaderboard as L
from strat.ma_2020_config_leaderboard import (
    build_panels, config_book, _asset_close, _metrics, SPLITS, YEAR, SYMS, ANN, TRAIL, MINHOLD,
)
from strat.portfolio_replay import apply_trail_stop, MAKER_RT
from strat.replay_distinct_grid import distinct_specs
from strat.ma_type_upgrade import _nums, _MA, MA_TYPES
from strat.ma_2020_breakdown import SPLIT, WARMUP
from strat.structural_fixes import min_hold
from strat.data_expansion import block_bootstrap_distribution

TFS = ["1d", "4h", "2h", "1h", "30m", "15m"]
MA_TYPE = "EMA"

# Load the leaderboard grid
LB_PATH = ROOT.parent / "runs" / "periods" / "TRAIN" / "2020" / "DEEP_DIVE" / "config_leaderboard.json"
with open(LB_PATH) as f:
    LB = json.load(f)
grid = LB["grid"]


def get_band_configs(tf):
    """Return list of config names in the EMA working band for `tf`."""
    cell = grid.get(f"{MA_TYPE}|{tf}")
    if not cell:
        return []
    return cell["band"]["band_configs"]


def build_ema_ensemble_book(tf):
    """Build the fixed-EW ensemble book for the EMA working band at `tf`.
    Returns a per-bar Series of net returns over the 2020 window."""
    band_configs = get_band_configs(tf)
    if not band_configs:
        print(f"  [WARN] No band configs for EMA|{tf}")
        return None

    # Get all unique periods
    all_periods = sorted(set(p for name in band_configs for p in _nums(name)))
    panels = build_panels(tf, MA_TYPE, all_periods)
    if len(panels) < 3:
        print(f"  [WARN] Not enough assets for EMA|{tf}")
        return None

    books = []
    for name in band_configs:
        periods = _nums(name)
        if len(periods) < 2:
            continue
        book = config_book(panels, periods)
        if book is not None and len(book) > 10:
            books.append(book)

    if not books:
        return None, None

    # Fixed-EW ensemble: align and average
    combined = pd.concat(books, axis=1).fillna(0.0).mean(axis=1)
    return combined, panels


def compute_coverage(panels, band_configs, tf):
    """Compute the fraction of bars where the ensemble is in position (pos >= 0.5)."""
    maf = _MA[MA_TYPE]
    pos_list = []
    for sym, (c, ms, win, ret, cache) in panels.items():
        member_pos = []
        for name in band_configs:
            periods = _nums(name)
            if len(periods) < 2:
                continue
            mas = [cache[p] for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c)); pos[1:] = h2[:-1]
            pos_list.append(pos[win])

    if not pos_list:
        return None
    # Ensemble position (mean across members and assets)
    return float(np.mean([p.mean() for p in pos_list]))


def compute_turnover(panels, band_configs, tf):
    """Compute round-trips per year for the ensemble (mean across assets x configs)."""
    annualizer = ANN[tf]  # bars per year
    rt_list = []
    for sym, (c, ms, win, ret, cache) in panels.items():
        for name in band_configs:
            periods = _nums(name)
            if len(periods) < 2:
                continue
            mas = [cache[p] for p in periods]
            if len(periods) == 2:
                h0 = (mas[0] > mas[1]).astype(np.int8)
            else:
                h0 = ((mas[0] > mas[1]) & (mas[1] > mas[2])).astype(np.int8)
            h0 = np.nan_to_num(h0).astype(np.int8)
            h1 = apply_trail_stop(h0.copy(), c, TRAIL)[0].astype(np.int8)
            h2 = min_hold(h1, MINHOLD).astype(np.int8)
            pos = np.zeros(len(c)); pos[1:] = h2[:-1]
            pos_w = pos[win]
            # Round-trips = number of buy signals / total bars * bars_per_year
            n_bars = win.sum()
            if n_bars < 2:
                continue
            flips = np.abs(np.diff(np.concatenate([[0.0], pos_w])))
            n_entries = int(np.sum(flips > 0) / 2)  # entries only (not exits)
            rt_per_year = n_entries / n_bars * annualizer
            rt_list.append(rt_per_year)
    return float(np.mean(rt_list)) if rt_list else None


def main():
    results = {}
    for tf in TFS:
        print(f"\n=== EMA|{tf} ===")
        band_configs = get_band_configs(tf)
        print(f"  Band size: {len(band_configs)} configs")

        all_periods = sorted(set(p for name in band_configs for p in _nums(name)))
        panels = build_panels(tf, MA_TYPE, all_periods)
        print(f"  Assets loaded: {len(panels)}")

        # Build ensemble book (OOS period only: 2020-10-01..2021-01-01)
        books = []
        for name in band_configs:
            periods = _nums(name)
            if len(periods) < 2:
                continue
            book = config_book(panels, periods)
            if book is not None and len(book) > 10:
                books.append(book)

        if not books:
            print(f"  [SKIP] no books")
            continue

        ensemble = pd.concat(books, axis=1).fillna(0.0).mean(axis=1)

        # OOS metrics
        oos_lo, oos_hi = SPLITS["OOS"]
        oos_s = ensemble[(ensemble.index >= pd.Timestamp(oos_lo)) &
                         (ensemble.index < pd.Timestamp(oos_hi))].dropna()
        oos_net = float(np.prod(1 + oos_s.to_numpy()) - 1) * 100 if len(oos_s) > 1 else None
        eq = np.cumprod(1 + oos_s.to_numpy()); pk = np.maximum.accumulate(eq)
        oos_maxdd = float(((eq - pk) / pk).min() * 100) if len(oos_s) > 1 else None

        # Block bootstrap p05 on OOS
        bb = block_bootstrap_distribution(oos_s.to_numpy(), n_boot=1000, block=5, seed=42)
        p05 = round(bb["p05"] * 100, 1)

        # Coverage
        coverage = compute_coverage(panels, band_configs, tf)

        # Turnover (round-trips per period)
        turnover = compute_turnover(panels, band_configs, tf)

        print(f"  OOS net: {oos_net:.1f}%  OOS maxdd: {oos_maxdd:.1f}%")
        print(f"  Block-bootstrap p05: {p05:.1f}%")
        print(f"  Coverage (pct in position): {coverage:.3f}")
        print(f"  Turnover (RT/yr): {turnover:.1f}")

        results[tf] = {
            "ma_type": "EMA",
            "tf": tf,
            "band_size": len(band_configs),
            "oos_net": round(oos_net, 1) if oos_net is not None else None,
            "oos_maxdd": round(oos_maxdd, 1) if oos_maxdd is not None else None,
            "p05_bootstrap": p05,
            "coverage": round(coverage, 3) if coverage is not None else None,
            "turnover_rt_per_yr": round(turnover, 1) if turnover is not None else None,
        }

    # Save
    out_path = ROOT.parent / "runs" / "strat" / "ema_profile_compute.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {out_path}")
    return results


if __name__ == "__main__":
    main()
