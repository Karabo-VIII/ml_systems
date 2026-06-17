"""build_coverage_atlas.py — P4-1: Coverage map across existing oracle Layer-3 specialists.

For each indicator class with Layer-3 exhaust data:
  1. Load event_eval_rows (every fire across every config)
  2. Filter to TOP-K configs by expectancy (these are the specialists worth keeping)
  3. Extract fires-with-positive-EV per (asset, date, regime) cell
  4. Union across all specialists into a coverage atlas

Output:
  runs/audit/coverage_atlas.parquet — per (asset, date) cell with N specialists firing
  runs/audit/COVERAGE_ATLAS_2026_05_19.md — verdict on engine coverage
"""
from __future__ import annotations
import os
from pathlib import Path

import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
LAYER3 = ROOT / "runs" / "oracle_layer3"

# Top-K configs per class to consider as specialists
TOP_K_PER_CLASS = 5
# Min expectancy for a config to qualify as specialist
MIN_EXPECTANCY = 0.001  # 0.1% post-cost EV per fire


def load_class_specialists(class_dir: Path) -> tuple[pl.DataFrame, list[str]]:
    """Load fires from top-K configs of an indicator class."""
    eval_rows_path = class_dir / "event_eval_rows.parquet"
    metrics_path = class_dir / "metrics_overall.parquet"
    if not eval_rows_path.exists() or not metrics_path.exists():
        return None, []

    metrics = pl.read_parquet(str(metrics_path))
    if "expectancy" not in metrics.columns:
        return None, []
    metrics = metrics.filter(pl.col("expectancy") >= MIN_EXPECTANCY)
    if len(metrics) == 0:
        return None, []
    top_configs = metrics.sort("expectancy", descending=True).head(TOP_K_PER_CLASS)
    top_config_names = top_configs["indicator_config"].to_list()
    expectancies = dict(zip(top_configs["indicator_config"], top_configs["expectancy"]))

    # Lazy-load eval rows, filter to top configs + fired events
    df = pl.scan_parquet(str(eval_rows_path)).filter(
        (pl.col("indicator_config").is_in(top_config_names)) & (pl.col("fired") == True)
    ).select([
        "asset", "date", "cadence", "def_type",
        "indicator_class", "indicator_config",
        "pnl_post_cost_pct", "btc_regime_30d", "cluster_id", "bucket",
    ]).collect()

    return df, top_config_names


def main():
    classes = [d for d in LAYER3.iterdir() if d.is_dir() and (d / "event_eval_rows.parquet").exists()]
    print(f"Found {len(classes)} Layer-3 indicator classes with event_eval_rows.parquet")

    all_fires = []
    per_class_stats = []
    for cls_dir in classes:
        cls_name = cls_dir.name
        df, configs = load_class_specialists(cls_dir)
        if df is None or len(df) == 0:
            print(f"  {cls_name}: no qualifying specialists")
            continue
        all_fires.append(df)
        per_class_stats.append({
            "class": cls_name,
            "n_specialists": len(configs),
            "n_fires_total": len(df),
            "n_unique_asset_days": df.select(["asset","date"]).unique().height,
            "configs": configs,
        })
        print(f"  {cls_name}: {len(configs)} specialists, {len(df)} fires, {df.select(['asset','date']).unique().height} unique asset-days")

    if not all_fires:
        print("No coverage data — exiting.")
        return

    fires = pl.concat(all_fires, how="diagonal_relaxed").with_columns(
        pl.col("date").cast(pl.Date)
    )
    print(f"\nTotal fires aggregated: {len(fires)}")

    # Coverage atlas: per (asset, date), how many specialists fire?
    atlas = fires.group_by(["asset", "date"]).agg([
        pl.col("indicator_class").n_unique().alias("n_distinct_classes"),
        pl.len().alias("n_fires"),
        pl.col("pnl_post_cost_pct").mean().alias("mean_specialist_pnl"),
        pl.col("pnl_post_cost_pct").sum().alias("sum_specialist_pnl"),
        pl.col("indicator_class").unique().alias("classes_firing"),
        pl.col("indicator_class").first().alias("any_class"),
        pl.col("btc_regime_30d").first().alias("btc_regime"),
        pl.col("cluster_id").first().alias("cluster_id"),
        pl.col("bucket").first().alias("bucket"),
    ]).sort("date", "asset")
    atlas.write_parquet(str(OUT_DIR / "coverage_atlas.parquet"))

    print(f"\nCoverage atlas: {len(atlas)} unique (asset, date) cells with ≥1 specialist firing")
    # Filter to 8Q WF window
    from datetime import date
    atlas_8q = atlas.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))
    print(f"  8Q WF window (24Q1-25Q4): {len(atlas_8q)} cells covered")

    # Univ size: 87 assets x 731 days = 63,597 asset-day cells (in theory)
    n_cells_total = 87 * 731
    print(f"  Universe (87 assets × 731 days): {n_cells_total} possible cells")
    print(f"  Coverage % of universe: {len(atlas_8q)/n_cells_total*100:.1f}%")

    # By regime — cast btc_regime to float defensively
    print("\nCoverage by BTC 30d regime band:")
    atlas_reg = atlas_8q.with_columns(
        pl.col("btc_regime").cast(pl.Float64, strict=False).alias("btc_regime_f")
    )
    by_regime = atlas_reg.with_columns(
        pl.when(pl.col("btc_regime_f") >= 0.05).then(pl.lit("bull"))
          .when(pl.col("btc_regime_f") <= -0.05).then(pl.lit("bear"))
          .otherwise(pl.lit("chop")).alias("regime")
    ).group_by("regime").agg([
        pl.len().alias("n_cells"),
        pl.col("mean_specialist_pnl").mean().alias("mean_pnl_when_fires"),
        (pl.col("mean_specialist_pnl") > 0).cast(pl.Float64).mean().alias("frac_positive_fires"),
    ])
    for r in by_regime.iter_rows(named=True):
        print(f"  {r['regime']:5s}: {r['n_cells']:>6d} cells covered  mean_pnl={r['mean_pnl_when_fires']:+.4f}%  frac_pos={r['frac_positive_fires']*100:.1f}%")

    # By bucket
    print("\nCoverage by asset bucket:")
    by_bucket = atlas_8q.group_by("bucket").agg([
        pl.len().alias("n_cells"),
        pl.col("mean_specialist_pnl").mean().alias("mean_pnl"),
        pl.col("asset").n_unique().alias("n_assets"),
    ]).sort("n_cells", descending=True)
    for r in by_bucket.iter_rows(named=True):
        print(f"  {r['bucket']:10s}: {r['n_cells']:>6d} cells / {r['n_assets']:>3d} assets  mean_pnl={r['mean_pnl']:+.4f}%")

    # By number-of-distinct-classes (the conjunction score)
    print("\nFires conjunction structure (how many classes co-fire per cell):")
    by_conj = atlas_8q.group_by("n_distinct_classes").agg([
        pl.len().alias("n_cells"),
        pl.col("mean_specialist_pnl").mean().alias("mean_pnl"),
        (pl.col("mean_specialist_pnl") > 0).cast(pl.Float64).mean().alias("frac_positive"),
    ]).sort("n_distinct_classes")
    for r in by_conj.iter_rows(named=True):
        print(f"  {r['n_distinct_classes']} classes: {r['n_cells']:>6d} cells  mean_pnl={r['mean_pnl']:+.4f}%  frac_pos={r['frac_positive']*100:.1f}%")

    # Per-day union: across all assets on each day, are SOME firing?
    daily_union = atlas_8q.group_by("date").agg([
        pl.col("asset").n_unique().alias("n_assets_with_fires"),
        pl.len().alias("n_cells_today"),
        pl.col("mean_specialist_pnl").mean().alias("avg_pnl_today"),
    ])
    days_with_any_fire = len(daily_union)
    days_possible = 731  # 8Q
    print(f"\nPer-day coverage:")
    print(f"  Days with ≥1 specialist firing (8Q): {days_with_any_fire}/{days_possible} ({days_with_any_fire/days_possible*100:.1f}%)")
    print(f"  Mean assets-firing-per-active-day: {daily_union['n_assets_with_fires'].mean():.1f}")
    print(f"  Median assets-firing-per-active-day: {daily_union['n_assets_with_fires'].median():.1f}")
    # Distribution
    print(f"  Days with ≥5 assets firing: {(daily_union['n_assets_with_fires'] >= 5).sum()}")
    print(f"  Days with ≥10 assets firing: {(daily_union['n_assets_with_fires'] >= 10).sum()}")
    print(f"  Days with ≥20 assets firing: {(daily_union['n_assets_with_fires'] >= 20).sum()}")

    print("\n[atlas] wrote", OUT_DIR / "coverage_atlas.parquet")
    return atlas_8q, per_class_stats


if __name__ == "__main__":
    main()
