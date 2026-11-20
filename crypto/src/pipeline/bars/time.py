"""True time bars. 1m is the base; coarser bars sum the 1m bars.

This is the opposite of the v51 cadence view, which copied the last dollar
bar in the bucket and called it a 1h candle.
"""

from __future__ import annotations

import polars as pl

from pipeline.bars.schema import (
    aggregate_child_bars,
    aggregate_trades,
    close_of,
    finalize,
)
from pipeline.constants import TIME_WIDTH_MS


def trades_to_time(trades: pl.DataFrame, width_ms: int, asset: str, bar_type: str) -> pl.DataFrame:
    if trades.is_empty():
        return finalize(trades, asset, bar_type)
    df = trades.sort(["ts_us", "agg_trade_id"] if "ts_us" in trades.columns else ["ts_ms"])
    df = df.with_columns((pl.col("price") * pl.col("qty")).alias("_usd"))
    df = df.with_columns(close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = aggregate_trades(df, "_close")
    grouped = grouped.with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
    ]).drop(["_close", "ts_close_trade_ms"])
    return finalize(grouped, asset, bar_type)


def bars_to_time(bars_1m: pl.DataFrame, width_ms: int, asset: str, bar_type: str) -> pl.DataFrame:
    """Aggregate 1m bars into a coarser clock. Volumes and trade counts sum."""
    if bars_1m.is_empty():
        return finalize(bars_1m, asset, bar_type)
    df = bars_1m.sort("ts_ms").with_columns(close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = aggregate_child_bars(df, "_close")
    grouped = grouped.with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
    ]).drop("_close")
    return finalize(grouped, asset, bar_type)


def build_time_ladder(trades: pl.DataFrame, asset: str, bar_types: list[str] | None = None) -> dict[str, pl.DataFrame]:
    """1m from trades, then every requested coarser width from those 1m bars."""
    wanted = list(bar_types) if bar_types is not None else list(TIME_WIDTH_MS)
    base = trades_to_time(trades, TIME_WIDTH_MS["time_1m"], asset, "time_1m")
    out: dict[str, pl.DataFrame] = {}
    if "time_1m" in wanted:
        out["time_1m"] = base
    for name, width in TIME_WIDTH_MS.items():
        if name == "time_1m" or name not in wanted:
            continue
        out[name] = bars_to_time(base, width, asset, name)
    return out


def merge_partial_time(partials: list[pl.DataFrame], asset: str, bar_type: str, width_ms: int) -> pl.DataFrame:
    """Merge per-day 1m partials. A bucket can straddle two daily files at midnight."""
    frames = [p for p in partials if p is not None and not p.is_empty()]
    if not frames:
        return finalize(pl.DataFrame(), asset, bar_type)
    df = pl.concat(frames, how="diagonal_relaxed").sort("ts_ms")
    # Re-aggregate on the 1m close so a split midnight bucket becomes one bar.
    df = df.with_columns(pl.col("ts_ms").alias("_close"))
    grouped = aggregate_child_bars(df, "_close")
    grouped = grouped.with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
    ]).drop("_close")
    return finalize(grouped, asset, bar_type)


def gaps_ms(bars_1m: pl.DataFrame, width_ms: int = 60_000) -> list[int]:
    """Missing clock closes between the first and last bar. Empty minutes are gaps, not zero bars."""
    if bars_1m.height < 2:
        return []
    closes = bars_1m.sort("ts_ms")["ts_ms"].to_list()
    missing: list[int] = []
    for a, b in zip(closes, closes[1:]):
        t = a + width_ms
        while t < b:
            missing.append(int(t))
            t += width_ms
            if len(missing) > 1_000_000:
                return missing
    return missing


def contiguous_hole_ms(missing: list[int], width_ms: int = 60_000) -> int:
    """Longest run of consecutive missing closes, in milliseconds."""
    if not missing:
        return 0
    longest = 1
    run = 1
    for a, b in zip(missing, missing[1:]):
        if b - a == width_ms:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return longest * width_ms
