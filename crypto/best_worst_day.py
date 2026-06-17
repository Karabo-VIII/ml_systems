#!/usr/bin/env python3
"""best_worst_day.py — rank daily crypto bars by single-day return.

For a target date, compute each asset's intraday return (close/open - 1) from
this repo's daily chimera bars, rank all assets, and print the best- and
worst-performing quartiles.

Usage:
    python best_worst_day.py [YYYY-MM-DD]

If no date is given, defaults to the most recent date for which at least 30
assets have a bar.

Data: data/processed/chimera/1d/*.parquet — one file per asset, named like
'btcusdt_v51_chimera_1d_DATE.parquet'. The ticker is the filename prefix
before '_v51'. Columns include timestamp (13-digit ms epoch), open, high,
low, close, volume.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed" / "chimera" / "1d"
MIN_ASSETS = 30


def ticker_from_filename(path: Path) -> str:
    """Derive the asset ticker from the filename prefix before '_v51'."""
    stem = path.name
    return stem.split("_v51", 1)[0].upper()


def load_bars() -> pl.DataFrame:
    """Load (ticker, date, open, close) for every asset/day across all files."""
    frames = []
    for path in sorted(DATA_DIR.glob("*.parquet")):
        ticker = ticker_from_filename(path)
        df = pl.read_parquet(path, columns=["timestamp", "open", "close"])
        if df.is_empty():
            continue
        df = df.with_columns(
            pl.lit(ticker).alias("ticker"),
            # 13-digit ms epoch -> calendar date (UTC)
            (pl.col("timestamp").cast(pl.Int64) * 1000)
            .cast(pl.Datetime("us"))
            .dt.date()
            .alias("date"),
        )
        frames.append(df.select(["ticker", "date", "open", "close"]))
    if not frames:
        raise SystemExit(f"No parquet files found under {DATA_DIR}")
    return pl.concat(frames)


def resolve_date(bars: pl.DataFrame, argv_date: str | None) -> datetime:
    if argv_date is not None:
        try:
            return datetime.strptime(argv_date, "%Y-%m-%d").date()
        except ValueError:
            raise SystemExit(f"Invalid date '{argv_date}' (expected YYYY-MM-DD)")
    # Default: most recent date with >= MIN_ASSETS assets having a bar.
    counts = (
        bars.group_by("date")
        .agg(pl.col("ticker").n_unique().alias("n"))
        .filter(pl.col("n") >= MIN_ASSETS)
        .sort("date", descending=True)
    )
    if counts.is_empty():
        raise SystemExit(f"No date has at least {MIN_ASSETS} assets with a bar.")
    return counts.row(0, named=True)["date"]


def main() -> None:
    argv_date = sys.argv[1] if len(sys.argv) > 1 else None
    bars = load_bars()
    target = resolve_date(bars, argv_date)

    day = (
        bars.filter(pl.col("date") == target)
        .filter(pl.col("open") > 0)  # guard against div-by-zero / bad bars
        .with_columns((pl.col("close") / pl.col("open") - 1.0).alias("ret"))
        .unique(subset=["ticker"], keep="first")
        .drop_nulls("ret")
        .sort("ret", descending=True)
    )

    n = day.height
    if n == 0:
        raise SystemExit(f"No bars found for {target}.")

    quartile = max(1, n // 4)
    rows = day.select(["ticker", "ret"]).rows()

    print(f"Date:        {target}")
    print(f"Asset count: {n}")
    print()

    print(f"TOP 25% - best performers ({quartile}):")
    for ticker, ret in rows[:quartile]:
        print(f"  {ticker:<14} {ret * 100:+.2f}%")

    print()
    print(f"BOTTOM 25% - worst performers ({quartile}):")
    for ticker, ret in rows[-quartile:]:
        print(f"  {ticker:<14} {ret * 100:+.2f}%")


if __name__ == "__main__":
    main()
