"""inspect_strength_scales.py -- empirically measure raw `strength` scales
across the 17 deploy indicators on real fire days.

Verifies the diagnosis that smart_discovery_17_sleeve.py:303's
`sort_values("strength")` is biased because indicator families emit incomparable
raw scales.

Output: a markdown summary of strength distribution per indicator over a sample
of fire days in the OOS window.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "strategy" / "sleeves"))

from smart_discovery_17_sleeve import (
    DEPLOY_17, DEFAULT_U100, _signal_fires_today, _today_regime
)
from pipeline.chimera_loader import ChimeraLoader  # type: ignore


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    loader = ChimeraLoader()
    print("Loading chimera 1d for u100 sample...")
    asset_dfs = {}
    for sym in DEFAULT_U100[:60]:  # sample 60 assets
        try:
            df = loader.load(sym, "1d")
            if df is None: continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            if "date" not in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            if len(df) >= 100:
                asset_dfs[sym] = df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  skip {sym}: {e}")
    print(f"  loaded {len(asset_dfs)} assets")

    # Sample 30 dates spread across 2024-Q1 to 2025-Q4 (2-yr OOS+UNSEEN-ish range)
    sample_dates = []
    for year in (2024, 2025):
        for month in range(1, 13):
            for day in (10, 20):
                sample_dates.append(date(year, month, day))
    sample_dates = sample_dates[::2]  # every other → ~24 dates
    print(f"  sampling {len(sample_dates)} dates")

    rows = []
    for pick_date in sample_dates:
        for sym, df in asset_dfs.items():
            mask = pd.to_datetime(df["date"]) <= pd.Timestamp(pick_date)
            sub = df[mask].tail(200).reset_index(drop=True)
            if len(sub) < 60: continue
            highs = sub["high"].values
            lows = sub["low"].values
            closes = sub["close"].values
            for ind, cfg, _ in DEPLOY_17:
                fired, strength = _signal_fires_today(ind, cfg, highs, lows, closes)
                if fired:
                    rows.append({
                        "date": str(pick_date),
                        "asset": sym,
                        "indicator": ind,
                        "config": cfg,
                        "strength": strength,
                    })

    df_fires = pd.DataFrame(rows)
    print(f"\nTotal fires across sample: {len(df_fires)}")

    if df_fires.empty:
        print("No fires; cannot diagnose.")
        return

    # Per-indicator strength distribution
    print("\n=== STRENGTH SCALE PER INDICATOR (across fire events) ===\n")
    stats = df_fires.groupby("indicator")["strength"].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    stats = stats[["count", "mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    pd.set_option("display.float_format", "{:+.4f}".format)
    print(stats.to_string())

    # Cross-indicator scale ratio (max median / min median)
    medians = df_fires.groupby("indicator")["strength"].median().abs()
    medians = medians[medians > 0]
    if len(medians) > 1:
        scale_ratio = medians.max() / medians.min()
        print(f"\n  Scale ratio (max median / min median): {scale_ratio:.1f}×")
        print(f"  Highest-median indicator : {medians.idxmax()} = {medians.max():+.4f}")
        print(f"  Lowest-median indicator  : {medians.idxmin()} = {medians.min():+.4f}")

    # Simulate K-selection bias: pick the 30 dates with most co-fires
    co_fires = df_fires.groupby("date").size().sort_values(ascending=False)
    print(f"\n=== TOP 10 CO-FIRE DAYS (multiple setups fire) ===\n")
    for d, n in co_fires.head(10).items():
        sub = df_fires[df_fires["date"] == d]
        sub_top5 = sub.sort_values("strength", ascending=False).head(5)
        ind_picked = sub_top5["indicator"].value_counts()
        print(f"  {d}: {n} fires, top-5 raw-strength picks: {ind_picked.to_dict()}")

    # Per-asset uniqueness check
    df_fires["asset_date"] = df_fires["asset"] + "_" + df_fires["date"]
    multi_setup_per_asset = df_fires.groupby("asset_date").size()
    n_multi = (multi_setup_per_asset > 1).sum()
    print(f"\n  (asset, date) cells with 2+ setups firing: {n_multi}")
    print(f"  Total unique (asset, date) cells: {len(multi_setup_per_asset)}")

    # Save
    OUT = ROOT / "runs" / "audit" / "STRENGTH_SCALES_2026_05_20.csv"
    df_fires.to_csv(OUT, index=False)
    print(f"\n[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
