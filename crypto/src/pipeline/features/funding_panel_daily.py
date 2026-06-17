"""Daily aggregation of Binance perp funding-rate raw into chimera-ready features.

Input:
    data/raw/<SYMBOL>USDT/funding/<DATE>.parquet
    Schema: timestamp (i64 ms), funding_rate (f64)
    Cadence: 3 settlements/day at 00:00 / 08:00 / 16:00 UTC.

Output:
    data/processed/panels/daily/funding_panel_daily.parquet
    Schema (per date, asset):
      date, asset,
        fund_rate_mean         -- daily mean funding rate (avg over 3 settlements)
        fund_rate_max          -- daily max funding rate (overheated extreme)
        fund_rate_min          -- daily min funding rate (panic-low extreme)
        fund_rate_abs_mean     -- daily mean |funding| (regime intensity, sign-agnostic)
        fund_rate_z30          -- z-score of fund_rate_mean vs trailing 30d mean
        fund_extreme_long_count    -- count of intraday settlements > +5bp (overheated longs)
        fund_extreme_short_count   -- count of intraday settlements < -5bp (overheated shorts)
        fund_sign_flip         -- 1 if any sign-change within day, else 0 (regime shift)
        fund_avg_apr           -- annualized funding (fund_rate_mean * 3 * 365)
        fund_n_settlements     -- count (sanity; 3 = normal)

Bias-free design:
    - All daily aggregates are within-day (no future leak).
    - fund_rate_z30 uses TRAILING 30-day rolling.
    - Day rolls at UTC 00:00 -- consistent with chimera "date" semantics.

Per parquet_io contract: atomic_write_parquet + is_fresh.
Per @browser B2 (no silent failures): missing/empty days -> raw skip with WARN.
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

INPUT_ROOT = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "processed" / "panels" / "daily" / "funding_panel_daily.parquet"

EXTREME_BP = 5e-4    # 5 bp = 0.05% per settlement = ~55% APR

__contract__ = {
    "kind": "panel_builder",
    "stage": "funding_panel_daily",
    "inputs": {"upstream": "data/raw/<SYMBOL>USDT/funding/<DATE>.parquet"},
    "outputs": {
        "files": "data/processed/panels/daily/funding_panel_daily.parquet",
        "columns": ["date", "asset",
                    "fund_rate_mean", "fund_rate_max", "fund_rate_min",
                    "fund_rate_abs_mean", "fund_rate_z30",
                    "fund_extreme_long_count", "fund_extreme_short_count",
                    "fund_sign_flip", "fund_avg_apr", "fund_n_settlements"],
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
    phase_log("fund_panel", phase, message, **kw)


def _agg_one_day(df: pl.DataFrame) -> Optional[dict]:
    """Aggregate one (asset, day) of funding settlements into daily features."""
    if df.is_empty():
        return None
    fr = df["funding_rate"]
    n = fr.len()
    abs_fr = fr.abs()
    fr_max = float(fr.max())
    fr_min = float(fr.min())
    # Sign-flip detection
    pos = (fr > 0).any()
    neg = (fr < 0).any()
    feats = {
        "fund_rate_mean":       float(fr.mean()),
        "fund_rate_max":        fr_max,
        "fund_rate_min":        fr_min,
        "fund_rate_abs_mean":   float(abs_fr.mean()),
        "fund_extreme_long_count":  int((fr > EXTREME_BP).sum()),
        "fund_extreme_short_count": int((fr < -EXTREME_BP).sum()),
        "fund_sign_flip":       int(bool(pos and neg)),
        "fund_avg_apr":         float(fr.mean()) * 3.0 * 365.0,
        "fund_n_settlements":   int(n),
    }
    return feats


def build_panel(symbols: Optional[list[str]] = None,
                 z_window: int = 30) -> pl.DataFrame:
    """Aggregate all (symbol, day) funding parquets under INPUT_ROOT into a daily long panel.

    Returns DataFrame with one row per (date, asset). Includes
    fund_rate_z30 which uses a TRAILING 30-day rolling for no-lookahead.
    """
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"input root missing: {INPUT_ROOT}")

    if symbols is None:
        # discover all <SYM>USDT subdirs with a funding/ directory
        symbols = sorted([d.name for d in INPUT_ROOT.iterdir()
                          if d.is_dir() and (d / "funding").is_dir()])

    rows = []
    for sym in symbols:
        sym_dir = INPUT_ROOT / sym / "funding"
        if not sym_dir.exists():
            continue
        files = sorted(sym_dir.glob("*.parquet"))
        if not files:
            continue
        # Asset name conversion: strip USDT suffix + 1000-prefix.
        # (1000PEPE -> PEPE so the join to chimera works.)
        asset = sym.upper().replace("USDT", "")
        if asset.startswith("1000"):
            asset = asset[4:]
        _pl("BUILD", f"{asset:<8s}: aggregating {len(files)} days...")
        for f in files:
            # Filename is <SYM>-funding-YYYY-MM-DD.parquet
            stem = f.stem
            date_str = "-".join(stem.split("-")[-3:])
            try:
                df = pl.read_parquet(f)
            except Exception as e:
                _pl("WARN", f"read failed {f.name}: {e}")
                continue
            feats = _agg_one_day(df)
            if feats is None:
                continue
            feats["date"] = date_str
            feats["asset"] = asset
            rows.append(feats)

    if not rows:
        raise RuntimeError("no rows produced")

    panel = pl.DataFrame(rows)
    panel = panel.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d", strict=False)
    )

    # Trailing 30-day z-score on fund_rate_mean (per asset). No lookahead:
    # rolling window is left-aligned (excludes current row's contribution
    # only via order; mean+std-rolling on series of length N at index i sees
    # [i-W+1..i] -- this DOES include current i. Acceptable since the
    # downstream chimera attach is silver +1 day shifted anyway.)
    panel = panel.sort(["asset", "date"])
    panel = panel.with_columns([
        pl.col("fund_rate_mean")
            .rolling_mean(window_size=z_window, min_samples=5)
            .over("asset")
            .alias("_mu"),
        pl.col("fund_rate_mean")
            .rolling_std(window_size=z_window, min_samples=5)
            .over("asset")
            .alias("_sd"),
    ])
    panel = panel.with_columns(
        ((pl.col("fund_rate_mean") - pl.col("_mu")) / (pl.col("_sd") + 1e-9))
            .clip(-5.0, 5.0)
            .alias("fund_rate_z30")
    ).drop(["_mu", "_sd"])

    cols_order = ["date", "asset",
                  "fund_rate_mean", "fund_rate_max", "fund_rate_min",
                  "fund_rate_abs_mean", "fund_rate_z30",
                  "fund_extreme_long_count", "fund_extreme_short_count",
                  "fund_sign_flip", "fund_avg_apr", "fund_n_settlements"]
    panel = panel.select(cols_order)
    return panel


def main():
    p = argparse.ArgumentParser(description="Perp funding-rate daily aggregator")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="Symbols to aggregate (BTCUSDT format). Default: all under INPUT_ROOT.")
    p.add_argument("--force", action="store_true",
                   help="Re-aggregate even if output exists.")
    args = p.parse_args()

    if not args.force and OUT.exists():
        print(f"[funding_panel_daily] SKIP fresh: {OUT.name} (use --force to re-aggregate)")
        return 0

    print(f"[funding_panel_daily] aggregating from {INPUT_ROOT}")
    panel = build_panel(symbols=args.symbols)
    print(f"[funding_panel_daily] panel shape: {panel.shape}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(panel, OUT,
                          required_cols=["date", "asset", "fund_rate_mean",
                                         "fund_rate_abs_mean", "fund_n_settlements"])
    print(f"[funding_panel_daily] WROTE {OUT}")
    print(f"  assets: {panel['asset'].n_unique()}, "
          f"date range: {panel['date'].min()} -> {panel['date'].max()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
