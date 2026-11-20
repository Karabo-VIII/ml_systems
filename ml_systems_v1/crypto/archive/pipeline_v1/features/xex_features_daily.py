"""Cross-exchange spread features (xex_*) for the top-5 liquid assets.

Input:
    data/processed/panels/daily/cross_exchange_daily.parquet
    Schema: date, asset, coinbase_close, bybit_close, okx_close, binance_close,
            cb_bn_spread_bps, by_bn_spread_bps, ok_bn_spread_bps
    Coverage: BTC, ETH, SOL, XRP, DOGE  (top-5 cross-exchange liquid)

Output:
    data/processed/panels/daily/xex_features_daily.parquet
    Schema (per date, asset):
      date, asset,
        xex_cb_bn_spread_bps    -- raw Coinbase-Binance basis (bp)
        xex_by_bn_spread_bps    -- raw Bybit-Binance basis (bp)
        xex_ok_bn_spread_bps    -- raw OKX-Binance basis (bp)
        xex_spread_dispersion   -- std of 3 cross-exchange spreads (regime)
        xex_max_abs_spread      -- max(|spread|) across 3 venues (extreme)
        xex_cb_bn_z30           -- 30d z-score of CB-BN spread (regime move)
        xex_n_venues_active     -- count of non-null spread fields today (data sanity)

Bias-free design:
    All aggregations are within-day or trailing-30. No future leak.
    Output is per (date, asset). Sparse-by-design: only 5 assets covered.

Per parquet_io contract: atomic_write_parquet.
Per @browser B6 (sparse-by-design): the SPARSE_BY_DESIGN_PREFIXES validator
exemption includes 'xex_' -- 95% of u100 won't have these features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pipeline.parquet_io import atomic_write_parquet

INPUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "cross_exchange_daily.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "xex_features_daily.parquet"

__contract__ = {
    "kind": "panel_builder",
    "stage": "xex_features_daily",
    "inputs": {"upstream": "data/processed/panels/daily/cross_exchange_daily.parquet"},
    "outputs": {
        "files": "data/processed/panels/daily/xex_features_daily.parquet",
        "columns": ["date", "asset",
                    "xex_cb_bn_spread_bps", "xex_by_bn_spread_bps", "xex_ok_bn_spread_bps",
                    "xex_spread_dispersion", "xex_max_abs_spread",
                    "xex_cb_bn_z30", "xex_n_venues_active"],
    },
    "invariants": {
        "no_lookahead": True,
        "no_silent_overwrite": True,
        "atomic_write": True,
        "sparse_by_design": True,   # BTC/ETH/SOL/XRP/DOGE only
    },
}


def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("xex_panel", phase, message, **kw)


def build_panel() -> pl.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(f"input missing: {INPUT}")

    src = pl.read_parquet(INPUT)
    if src["date"].dtype != pl.Date:
        src = src.with_columns(pl.col("date").cast(pl.Date))
    src = src.sort(["asset", "date"])
    _pl("BUILD", f"cross_exchange input: {src.height:,} rows, "
                  f"{src['asset'].n_unique()} assets, "
                  f"{src['date'].min()} -> {src['date'].max()}")

    out = src.with_columns([
        pl.col("cb_bn_spread_bps").alias("xex_cb_bn_spread_bps"),
        pl.col("by_bn_spread_bps").alias("xex_by_bn_spread_bps"),
        pl.col("ok_bn_spread_bps").alias("xex_ok_bn_spread_bps"),
    ])

    # Dispersion: std across the 3 spreads (regime intensity).
    # n_venues_active: count of non-null spreads today.
    spreads_arr = ["cb_bn_spread_bps", "by_bn_spread_bps", "ok_bn_spread_bps"]
    out = out.with_columns([
        # n_venues_active: count of non-null
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in spreads_arr])
            .alias("xex_n_venues_active"),
        # max abs spread
        pl.max_horizontal([pl.col(c).abs() for c in spreads_arr])
            .alias("xex_max_abs_spread"),
    ])
    # Std requires careful handling of nulls; compute as numpy if any are null
    # via expression: concat to list, then std with null handling.
    out = out.with_columns(
        pl.concat_list(spreads_arr).list.std().alias("xex_spread_dispersion")
    )

    # Trailing 30d z-score on the canonical CB-BN spread (per asset).
    out = out.with_columns([
        pl.col("cb_bn_spread_bps")
            .rolling_mean(window_size=30, min_samples=5)
            .over("asset")
            .alias("_mu"),
        pl.col("cb_bn_spread_bps")
            .rolling_std(window_size=30, min_samples=5)
            .over("asset")
            .alias("_sd"),
    ])
    out = out.with_columns(
        ((pl.col("cb_bn_spread_bps") - pl.col("_mu")) / (pl.col("_sd") + 1e-9))
            .clip(-5.0, 5.0)
            .alias("xex_cb_bn_z30")
    ).drop(["_mu", "_sd"])

    cols_order = ["date", "asset",
                  "xex_cb_bn_spread_bps", "xex_by_bn_spread_bps", "xex_ok_bn_spread_bps",
                  "xex_spread_dispersion", "xex_max_abs_spread",
                  "xex_cb_bn_z30", "xex_n_venues_active"]
    out = out.select(cols_order)
    return out


def main():
    p = argparse.ArgumentParser(description="Cross-exchange spread features (xex_*)")
    p.add_argument("--force", action="store_true",
                   help="Re-build even if output exists.")
    args = p.parse_args()

    if not args.force and OUT.exists():
        print(f"[xex_features_daily] SKIP fresh: {OUT.name} (use --force to rebuild)")
        return 0

    print(f"[xex_features_daily] reading {INPUT}")
    panel = build_panel()
    print(f"[xex_features_daily] panel shape: {panel.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, OUT,
                          required_cols=["date", "asset", "xex_cb_bn_spread_bps",
                                         "xex_spread_dispersion", "xex_n_venues_active"])
    print(f"[xex_features_daily] WROTE {OUT}")
    print(f"  assets: {panel['asset'].n_unique()}, "
          f"date range: {panel['date'].min()} -> {panel['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
