"""Premium / mark-vs-spot derived features layered on top of basis_features_long.

Background:
    The "mark/premium" signal in perp futures is mathematically equivalent to
    the spot-perp basis already computed in basis_features_long.parquet
    (basis_pct = (perp_close - spot_close) / spot_close). This module ADDS
    5 derived features that complement existing basis_* without duplication.

Input:
    data/processed/panels/daily/basis_features_long.parquet
    Schema (relevant cols): date, asset, basis_pct, basis_z30

Output:
    data/processed/panels/daily/premium_features_daily.parquet
    Schema (per date, asset):
      date, asset,
        premium_vol30          -- rolling 30d std of basis_pct (instability regime)
        premium_persistence30  -- rolling 30d lag-1 autocorr of basis_pct
                                   (trend vs mean-revert regime)
        premium_extreme_count30  -- count of |basis_pct| > 0.10 in trailing 30d
                                     (frequent-extreme regime indicator)
        premium_z90            -- 90d z-score (longer regime baseline vs basis_z30)
        premium_apr            -- annualized basis: basis_pct * 365
                                   (proxy for implied carry-trade APR)

Bias-free design:
    All aggregations use trailing-30/90 rolling windows. No future leak.
    Output is per (date, asset); chimera attach via attach_frontier +1d shift.

Per parquet_io contract: atomic_write_parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet

INPUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "basis_features_long.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "premium_features_daily.parquet"

EXTREME_BASIS = 0.10   # |basis_pct| > 10% = "extreme"

__contract__ = {
    "kind": "panel_builder",
    "stage": "premium_features_daily",
    "inputs": {"upstream": "data/processed/panels/daily/basis_features_long.parquet"},
    "outputs": {
        "files": "data/processed/panels/daily/premium_features_daily.parquet",
        "columns": ["date", "asset",
                    "premium_vol30", "premium_persistence30",
                    "premium_extreme_count30", "premium_z90", "premium_apr"],
    },
    "invariants": {
        "no_lookahead": True,
        "no_silent_overwrite": True,
        "atomic_write": True,
    },
}


def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("prem_panel", phase, message, **kw)


def build_panel() -> pl.DataFrame:
    """Read basis_features_long, compute 5 derived premium features per (asset, date)."""
    if not INPUT.exists():
        raise FileNotFoundError(f"input missing: {INPUT}")

    src = pl.read_parquet(INPUT, columns=["date", "asset", "basis_pct"])
    # Normalize date to pl.Date (basis_features stamps datetime[ns])
    if src["date"].dtype != pl.Date:
        src = src.with_columns(pl.col("date").cast(pl.Date))
    src = src.sort(["asset", "date"])
    _pl("BUILD", f"basis_features input: {src.height:,} rows, "
                  f"{src['asset'].n_unique()} assets, "
                  f"{src['date'].min()} -> {src['date'].max()}")

    # premium_apr: simple linear annualization. NOT a forecast; it's the
    # implied carry-trade APR you'd earn if today's basis held for a year.
    out = src.with_columns(
        (pl.col("basis_pct") * 365.0).alias("premium_apr")
    )

    # Rolling 30d std per asset (no lookahead -- rolling is left-anchored).
    out = out.with_columns(
        pl.col("basis_pct")
            .rolling_std(window_size=30, min_samples=5)
            .over("asset")
            .alias("premium_vol30")
    )

    # Rolling 30d lag-1 autocorrelation per asset.
    # Polars doesn't have rolling_corr; implement via rolling cov / std.
    out = out.with_columns(
        pl.col("basis_pct").shift(1).over("asset").alias("_b_lag1")
    )
    # Drop initial null rows where lag1 is null (per asset).
    # Use rolling stats on (b, b_lag1) to derive autocorr.
    # corr = cov(b, b_lag1) / (std(b) * std(b_lag1))
    # Approximation: use rolling_mean of cross-product minus product of means.
    out = out.with_columns([
        (pl.col("basis_pct") * pl.col("_b_lag1"))
            .rolling_mean(window_size=30, min_samples=5)
            .over("asset")
            .alias("_xy_mean"),
        pl.col("basis_pct")
            .rolling_mean(window_size=30, min_samples=5)
            .over("asset")
            .alias("_x_mean"),
        pl.col("_b_lag1")
            .rolling_mean(window_size=30, min_samples=5)
            .over("asset")
            .alias("_y_mean"),
        pl.col("basis_pct")
            .rolling_std(window_size=30, min_samples=5)
            .over("asset")
            .alias("_x_std"),
        pl.col("_b_lag1")
            .rolling_std(window_size=30, min_samples=5)
            .over("asset")
            .alias("_y_std"),
    ])
    out = out.with_columns(
        ((pl.col("_xy_mean") - pl.col("_x_mean") * pl.col("_y_mean")) /
         (pl.col("_x_std") * pl.col("_y_std") + 1e-9))
            .clip(-1.0, 1.0)
            .alias("premium_persistence30")
    )

    # Rolling count of extreme basis in trailing 30d.
    out = out.with_columns(
        (pl.col("basis_pct").abs() > EXTREME_BASIS).cast(pl.Int32).alias("_is_extreme")
    )
    out = out.with_columns(
        pl.col("_is_extreme")
            .rolling_sum(window_size=30, min_samples=5)
            .over("asset")
            .alias("premium_extreme_count30")
    )

    # Rolling 90d z-score.
    out = out.with_columns([
        pl.col("basis_pct")
            .rolling_mean(window_size=90, min_samples=10)
            .over("asset")
            .alias("_mu90"),
        pl.col("basis_pct")
            .rolling_std(window_size=90, min_samples=10)
            .over("asset")
            .alias("_sd90"),
    ])
    out = out.with_columns(
        ((pl.col("basis_pct") - pl.col("_mu90")) / (pl.col("_sd90") + 1e-9))
            .clip(-5.0, 5.0)
            .alias("premium_z90")
    )

    cols_order = ["date", "asset",
                  "premium_vol30", "premium_persistence30",
                  "premium_extreme_count30", "premium_z90", "premium_apr"]
    out = out.select(cols_order)
    return out


def main():
    p = argparse.ArgumentParser(description="Premium-derived features from basis_features_long")
    p.add_argument("--force", action="store_true",
                   help="Re-build even if output exists.")
    args = p.parse_args()

    if not args.force and OUT.exists():
        print(f"[premium_features_daily] SKIP fresh: {OUT.name} (use --force to rebuild)")
        return 0

    print(f"[premium_features_daily] reading {INPUT}")
    panel = build_panel()
    print(f"[premium_features_daily] panel shape: {panel.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, OUT,
                          required_cols=["date", "asset", "premium_vol30",
                                         "premium_persistence30", "premium_z90"])
    print(f"[premium_features_daily] WROTE {OUT}")
    print(f"  assets: {panel['asset'].n_unique()}, "
          f"date range: {panel['date'].min()} -> {panel['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
