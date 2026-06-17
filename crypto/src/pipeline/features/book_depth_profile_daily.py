"""Daily aggregation of binance.vision bookDepth-profile data into chimera-ready features.

Input:
    data/raw_external/binance_vision/depth_profile/<SYM>/<YYYY-MM-DD>.parquet
    Schema: ts_ms, symbol, percentage, depth, notional
    (12 bands: -5%, -4%, -3%, -2%, -1%, -0.2%, +0.2%, +1%, +2%, +3%, +4%, +5%)
    ~2632 snapshots/day per asset (~30s cadence).

Output:
    data/processed/panels/daily/book_depth_profile_daily.parquet
    Schema (per date, asset):
        date, asset,

      # Depth at narrow bands (microstructure scale)
        bd_depth_l1pct_mean        -- mean cumulative depth at ±1% (base ccy)
        bd_depth_l1pct_p90         -- 90th-pct depth at ±1% (top-of-day liquidity)
        bd_notional_l1pct_mean     -- mean notional value at ±1% (USD)

      # Imbalance / skew (directional book pressure)
        bd_imbalance_l1            -- mean(depth(-1%) / depth(+1%))  -1 = thick bid
        bd_imbalance_l5            -- mean(depth(-5%) / depth(+5%))  -1 = thick deep bid
        bd_notional_skew           -- mean((notional_bid - notional_ask) / total) at ±5%

      # Liquidity regime
        bd_total_depth_l5_mean     -- total depth ±5% (overall liquidity)
        bd_total_depth_l5_p10      -- 10th-pct (worst-liquidity moments)
        bd_thin_book_frac          -- fraction of snapshots where total l5 depth < 0.5x rolling-30d-median

      # Microstructure precision
        bd_depth_at_02pct          -- mean depth at ±0.2% (top-of-book precision)
        bd_n_snapshots             -- snapshot count for the day

Bias-free design:
    - All aggregations are within-day (no future leak)
    - bd_thin_book_frac uses TRAILING 30-day median (no future)
    - Schema is daily, joined to chimera by (date, asset)

Per parquet_io contract: atomic_write_parquet + is_fresh.
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

INPUT_ROOT = PROJECT_ROOT / "data" / "raw_external" / "binance_vision" / "depth_profile"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "book_depth_profile_daily.parquet"

__contract__ = {
    "kind": "panel_builder",
    "stage": "book_depth_profile_daily",
    "inputs": {"upstream": "data/raw_external/binance_vision/depth_profile/<SYM>/<DATE>.parquet"},
    "outputs": {
        "files": "data/processed/panels/daily/book_depth_profile_daily.parquet",
        "columns": ["date", "asset",
                    "bd_depth_l1pct_mean", "bd_depth_l1pct_p90", "bd_notional_l1pct_mean",
                    "bd_imbalance_l1", "bd_imbalance_l5", "bd_notional_skew",
                    "bd_total_depth_l5_mean", "bd_total_depth_l5_p10",
                    "bd_thin_book_frac",
                    "bd_depth_at_02pct", "bd_n_snapshots"],
    },
    "invariants": {
        "no_lookahead": True,  # within-day aggregations + trailing-30d median
        "no_silent_overwrite": True,
        "atomic_write": True,
    },
}


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("bdp", phase, message, **kw)


def _agg_one_day(df: pl.DataFrame) -> dict:
    """Aggregate one (asset, day) of depth-profile snapshots into daily features.

    df: schema (ts_ms, symbol, percentage, depth, notional) for ONE asset ONE day.
    Returns dict of features (no date / asset; caller adds them).
    """
    if df.is_empty():
        return {}

    # Pivot to per-snapshot, per-band wide so we can compute imbalance directly.
    # Each snapshot has 12 bands; wide will be (n_snaps, 12 percentage cols).
    wide = (df
            .group_by(["ts_ms", "percentage"])
            .agg(pl.col("depth").first(), pl.col("notional").first())
            .pivot(index="ts_ms", on="percentage", values=["depth", "notional"]))

    # Polars pivot returns column names like 'depth_-1.0', 'depth_1.0', etc.
    cols = wide.columns

    def _col_for(prefix: str, pct: float) -> Optional[str]:
        # Polars >=1.0 uses prefix_<value>; handle either casting
        for c in cols:
            if c.startswith(f"{prefix}_") and float(c.split("_", 1)[1]) == pct:
                return c
        return None

    n = len(wide)
    feats: dict = {"bd_n_snapshots": n}

    # Depth at ±0.2%
    d_neg02, d_pos02 = _col_for("depth", -0.2), _col_for("depth", 0.2)
    if d_neg02 and d_pos02:
        depth_02 = wide.select((pl.col(d_neg02) + pl.col(d_pos02)).alias("d"))["d"]
        feats["bd_depth_at_02pct"] = float(depth_02.mean()) if not depth_02.is_empty() else 0.0
    else:
        feats["bd_depth_at_02pct"] = 0.0

    # Depth + imbalance at ±1%
    d_neg1, d_pos1 = _col_for("depth", -1.0), _col_for("depth", 1.0)
    n_neg1, n_pos1 = _col_for("notional", -1.0), _col_for("notional", 1.0)
    if d_neg1 and d_pos1:
        depth_1 = wide.select((pl.col(d_neg1) + pl.col(d_pos1)).alias("d"))["d"]
        feats["bd_depth_l1pct_mean"] = float(depth_1.mean())
        feats["bd_depth_l1pct_p90"] = float(depth_1.quantile(0.9))
        # Imbalance: bid-heavy => depth(-1%)/depth(+1%) > 1
        imb = (wide.select(
            (pl.col(d_neg1) / (pl.col(d_pos1) + 1e-9)).alias("imb")
        )["imb"])
        # Clamp extreme outliers; log the mean
        imb = imb.clip(0.1, 10.0)
        feats["bd_imbalance_l1"] = float(imb.mean())
    else:
        feats.update({"bd_depth_l1pct_mean": 0.0, "bd_depth_l1pct_p90": 0.0,
                      "bd_imbalance_l1": 1.0})
    if n_neg1 and n_pos1:
        notional_1 = wide.select(
            (pl.col(n_neg1) + pl.col(n_pos1)).alias("n")
        )["n"]
        feats["bd_notional_l1pct_mean"] = float(notional_1.mean())
    else:
        feats["bd_notional_l1pct_mean"] = 0.0

    # ±5% bands: depth + imbalance + notional skew
    d_neg5, d_pos5 = _col_for("depth", -5.0), _col_for("depth", 5.0)
    n_neg5, n_pos5 = _col_for("notional", -5.0), _col_for("notional", 5.0)
    if d_neg5 and d_pos5:
        depth_5 = wide.select((pl.col(d_neg5) + pl.col(d_pos5)).alias("d"))["d"]
        feats["bd_total_depth_l5_mean"] = float(depth_5.mean())
        feats["bd_total_depth_l5_p10"] = float(depth_5.quantile(0.1))
        imb5 = wide.select(
            (pl.col(d_neg5) / (pl.col(d_pos5) + 1e-9)).alias("imb")
        )["imb"].clip(0.1, 10.0)
        feats["bd_imbalance_l5"] = float(imb5.mean())
    else:
        feats.update({"bd_total_depth_l5_mean": 0.0, "bd_total_depth_l5_p10": 0.0,
                      "bd_imbalance_l5": 1.0})
    if n_neg5 and n_pos5:
        skew = wide.select(
            ((pl.col(n_neg5) - pl.col(n_pos5)) /
             (pl.col(n_neg5) + pl.col(n_pos5) + 1e-9)).alias("s")
        )["s"]
        feats["bd_notional_skew"] = float(skew.mean())
    else:
        feats["bd_notional_skew"] = 0.0

    return feats


def build_panel(symbols: Optional[list[str]] = None,
                 thin_book_window: int = 30) -> pl.DataFrame:
    """Aggregate all (symbol, day) parquets under INPUT_ROOT into a daily long panel.

    Returns DataFrame with one row per (date, asset). Includes
    bd_thin_book_frac which uses a TRAILING window for no-lookahead.
    """
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"input root missing: {INPUT_ROOT}")

    if symbols is None:
        symbols = sorted([d.name for d in INPUT_ROOT.iterdir() if d.is_dir()])

    rows = []
    for sym in symbols:
        sym_dir = INPUT_ROOT / sym
        files = sorted(sym_dir.glob("*.parquet"))
        if not files:
            continue
        asset = sym.upper().replace("USDT", "")
        # Map 1000-prefix futures pairs to base chimera asset name
        # (1000PEPE → PEPE so the join to chimera works)
        if asset.startswith("1000"):
            asset = asset[4:]
        _pl("BUILD", f"{asset:<8s}: aggregating {len(files)} days...")
        for f in files:
            date_str = f.stem  # YYYY-MM-DD
            try:
                df = pl.read_parquet(f)
            except Exception as e:
                _pl("WARN", f"read failed {f.name}: {e}")
                continue
            feats = _agg_one_day(df)
            if not feats:
                continue
            feats["date"] = date_str
            feats["asset"] = asset
            rows.append(feats)

    if not rows:
        raise RuntimeError("no rows produced")

    panel = pl.DataFrame(rows)
    # Cast date to date dtype
    panel = panel.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )
    # Trailing 30-day thin-book frac (per asset)
    # bd_thin_book_frac = fraction of snapshots in this day where total_depth < median
    # APPROXIMATION: use median across days (we don't have per-snap data here;
    # this is the daily-aggregate proxy for the regime feature)
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns(
        pl.col("bd_total_depth_l5_mean")
            .rolling_median(window_size=thin_book_window, min_samples=5)
            .over("asset")
            .alias("_median_l5")
    )
    panel = panel.with_columns(
        (pl.col("bd_total_depth_l5_mean") < 0.5 * pl.col("_median_l5"))
            .cast(pl.Float64)
            .alias("bd_thin_book_frac")
    ).drop("_median_l5")

    cols_order = ["date", "asset",
                  "bd_depth_l1pct_mean", "bd_depth_l1pct_p90", "bd_notional_l1pct_mean",
                  "bd_imbalance_l1", "bd_imbalance_l5", "bd_notional_skew",
                  "bd_total_depth_l5_mean", "bd_total_depth_l5_p10",
                  "bd_thin_book_frac",
                  "bd_depth_at_02pct", "bd_n_snapshots"]
    panel = panel.select(cols_order)
    return panel


def main():
    p = argparse.ArgumentParser(description="bookDepth-profile daily aggregator")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to aggregate. Default: all under INPUT_ROOT.")
    p.add_argument("--force", action="store_true",
                   help="Re-aggregate even if output exists.")
    args = p.parse_args()

    if not args.force and OUT.exists():
        print(f"[book_depth_profile_daily] SKIP fresh: {OUT.name} (use --force to re-aggregate)")
        return 0

    print(f"[book_depth_profile_daily] aggregating from {INPUT_ROOT}")
    panel = build_panel(symbols=args.symbols)
    print(f"[book_depth_profile_daily] panel shape: {panel.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, OUT,
                          required_cols=["date", "asset", "bd_depth_l1pct_mean",
                                         "bd_imbalance_l1", "bd_n_snapshots"])
    print(f"[book_depth_profile_daily] WROTE {OUT}")
    print(f"  assets: {panel['asset'].n_unique()}, "
          f"date range: {panel['date'].min()} -> {panel['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
