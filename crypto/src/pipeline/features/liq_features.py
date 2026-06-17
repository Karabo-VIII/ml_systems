"""Liquidation feature panel — derives 13 features from liq_daily_approx.

Input panel: data/processed/panels/daily/liq_daily_approx.parquet
  Schema: date, asset, liq_long_usd, liq_short_usd, liq_long_count,
          liq_short_count, liq_delta_usd, liq_total_usd

Output panel: data/processed/panels/daily/liq_features_long.parquet
  Schema: date, asset + 13 derived features (see registry liq_features spec)

Features (per asset, per date):
    liq_long_usd        pass-through (already log-clipped) -- raw long-liq USD
    liq_short_usd       pass-through                       -- raw short-liq USD
    liq_delta_usd       pass-through                       -- short - long
    liq_total_usd       pass-through                       -- long + short
    liq_long_z30        per-asset rolling 30d z-score (shifted-1, no leak)
    liq_short_z30       per-asset rolling 30d z-score
    liq_delta_z30       per-asset rolling 30d z-score of liq_delta_usd
    liq_long_xsec_z     cross-sectional z-score per date (vs all assets)
    liq_short_xsec_z    cross-sectional z-score per date
    liq_long_spike      binary: liq_long_z30 > 2 (>2 sigma capitulation buy)
    liq_short_spike     binary: liq_short_z30 > 2 (>2 sigma short squeeze)
    liq_capitulation    binary: long_spike & price_drop (when joined with bars)
                          -- here computed as long_spike & long > 5x median
    liq_short_panic     binary: short_spike & short > 5x median
"""
from __future__ import annotations
import os

# CDAP contract -- declared after __future__ per PEP-236.
__contract__ = {
    "kind": "panel_builder",
    "stage": "liq_features",
    "inputs": {
        "args": ["--input", "--output", "--force", "--assets", "--universe", "--dry-run"],
        "upstream": "data/processed/panels/daily/liq_daily_approx.parquet",
    },
    "outputs": {
        "files": "data/processed/panels/daily/liq_features_long.parquet",
        "columns": ["date", "asset", "liq_long_usd", "liq_short_usd",
                    "liq_delta_usd", "liq_total_usd",
                    "liq_long_z30", "liq_short_z30", "liq_delta_z30",
                    "liq_long_xsec_z", "liq_short_xsec_z",
                    "liq_long_spike", "liq_short_spike",
                    "liq_capitulation", "liq_short_panic"],
    },
    "invariants": {
        "atomic_write": True,
        "column_name_verify": True,
        "universe_agnostic": True,
    },
}

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))
DEFAULT_IN = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "liq_daily_approx.parquet"
DEFAULT_OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "liq_features_long.parquet"

ROLL_WIN = 30
SPIKE_Z_THRESH = 2.0
SPIKE_MEDIAN_MULT = 5.0


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("liq_feat", phase, message, **kw)


def _per_asset_z(df: pl.DataFrame, value_col: str, out_col: str,
                  win: int = ROLL_WIN) -> pl.DataFrame:
    """Add per-asset shifted-1 rolling z-score column.

    Shifted-1 ensures no look-ahead: today's z uses yesterday's window.
    """
    return df.with_columns([
        pl.col(value_col)
          .shift(1)
          .rolling_mean(window_size=win, min_samples=10)
          .over("asset")
          .alias(f"_{value_col}_mu"),
        pl.col(value_col)
          .shift(1)
          .rolling_std(window_size=win, min_samples=10)
          .over("asset")
          .alias(f"_{value_col}_sd"),
    ]).with_columns(
        ((pl.col(value_col) - pl.col(f"_{value_col}_mu"))
         / (pl.col(f"_{value_col}_sd") + 1e-9)).clip(-5.0, 5.0).alias(out_col)
    ).drop([f"_{value_col}_mu", f"_{value_col}_sd"])


def _xsec_z(df: pl.DataFrame, value_col: str, out_col: str) -> pl.DataFrame:
    """Add cross-sectional z-score (per date, across all assets in the panel)."""
    return df.with_columns([
        pl.col(value_col).mean().over("date").alias(f"_{value_col}_xmu"),
        pl.col(value_col).std().over("date").alias(f"_{value_col}_xsd"),
    ]).with_columns(
        ((pl.col(value_col) - pl.col(f"_{value_col}_xmu"))
         / (pl.col(f"_{value_col}_xsd") + 1e-9)).clip(-5.0, 5.0).alias(out_col)
    ).drop([f"_{value_col}_xmu", f"_{value_col}_xsd"])


def build(input_path: Path = DEFAULT_IN) -> pl.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(
            f"liq_features depends on {input_path.name} (built by "
            f"liquidations_approx.py). Missing at {input_path}.")

    df = pl.read_parquet(input_path)
    # Normalize date to Date type (input may be Datetime[ns]).
    if df.schema["date"] != pl.Date:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    df = df.sort(["asset", "date"])

    # Per-asset rolling z-scores (shifted-1, no leak).
    df = _per_asset_z(df, "liq_long_usd", "liq_long_z30")
    df = _per_asset_z(df, "liq_short_usd", "liq_short_z30")
    df = _per_asset_z(df, "liq_delta_usd", "liq_delta_z30")

    # Cross-sectional z (per date).
    df = _xsec_z(df, "liq_long_usd", "liq_long_xsec_z")
    df = _xsec_z(df, "liq_short_usd", "liq_short_xsec_z")

    # Spike flags (binary).
    df = df.with_columns([
        (pl.col("liq_long_z30") > SPIKE_Z_THRESH).cast(pl.Int8).alias("liq_long_spike"),
        (pl.col("liq_short_z30") > SPIKE_Z_THRESH).cast(pl.Int8).alias("liq_short_spike"),
    ])

    # Per-asset rolling median for "5x median" composite flags.
    df = df.with_columns([
        pl.col("liq_long_usd").shift(1).rolling_median(window_size=ROLL_WIN, min_samples=10)
          .over("asset").alias("_liq_long_med"),
        pl.col("liq_short_usd").shift(1).rolling_median(window_size=ROLL_WIN, min_samples=10)
          .over("asset").alias("_liq_short_med"),
    ])

    df = df.with_columns([
        ((pl.col("liq_long_spike") == 1)
         & (pl.col("liq_long_usd") > pl.col("_liq_long_med") * SPIKE_MEDIAN_MULT))
        .cast(pl.Int8).alias("liq_capitulation"),
        ((pl.col("liq_short_spike") == 1)
         & (pl.col("liq_short_usd") > pl.col("_liq_short_med") * SPIKE_MEDIAN_MULT))
        .cast(pl.Int8).alias("liq_short_panic"),
    ]).drop(["_liq_long_med", "_liq_short_med"])

    # Final column ordering per registry spec.
    keep = ["date", "asset", "liq_long_usd", "liq_short_usd",
            "liq_delta_usd", "liq_total_usd",
            "liq_long_z30", "liq_short_z30", "liq_delta_z30",
            "liq_long_xsec_z", "liq_short_xsec_z",
            "liq_long_spike", "liq_short_spike",
            "liq_capitulation", "liq_short_panic"]
    df = df.select(keep)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_IN,
                    help=f"Input liq_daily_approx.parquet (default: {DEFAULT_IN.name})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT,
                    help=f"Output liq_features_long.parquet (default: {DEFAULT_OUT.name})")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild even if OUT panel is fresher than input.")
    # 2026-05-21 contract retrofit
    ap.add_argument("--assets", nargs="+", default=None,
                    help="No-op for cross-section panel. Accepted for pipeline uniformity.")
    ap.add_argument("--universe", default=None, choices=["u10", "u50", "u100"],
                    help="No-op for cross-section panel. Accepted for pipeline uniformity.")
    ap.add_argument("--workers", type=int, default=1, help="Not used.")
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no writes.")
    args = ap.parse_args()
    if args.assets or args.universe:
        print(f"[liq_feat] note: --assets/--universe accepted but no-op for cross-section panel",
              flush=True)

    # Skip-existing: output fresher than input
    if args.output.exists() and not args.force:
        if args.input.exists() and args.output.stat().st_mtime >= args.input.stat().st_mtime:
            _pl("SKIP", f"skip: output fresher than input; --force to rebuild")
            return 0

    if args.dry_run:
        _pl("BUILD", f"DRY-RUN: would rebuild {args.output}")
        return 0

    _pl("BUILD", f"reading {args.input.name}...")
    df = build(args.input)
    print(f"[liq_feat] built: {df.height} rows, {len(df.columns)} cols, "
          f"{df.select('asset').n_unique()} assets, "
          f"{df.select('date').n_unique()} dates")

    # G-AUDIT-020: atomic-tmp-rename + column-name verify.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    written = set(pl.read_parquet_schema(tmp).keys())
    required = {"date", "asset", "liq_long_usd", "liq_long_z30",
                "liq_long_xsec_z", "liq_long_spike", "liq_capitulation",
                "liq_short_panic"}
    missing = required - written
    if missing:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"liq_features panel missing required cols: {sorted(missing)}")
    if args.output.exists():
        args.output.unlink()
    os.replace(str(tmp), str(args.output))  # atomic overwrite (Windows-safe)
    _pl("OK", f"saved: {args.output}")
    # Skip pretty-print sample: polars table renderer emits Unicode that
    # crashes Windows cp1252.
    return 0


if __name__ == "__main__":
    sys.exit(main())
