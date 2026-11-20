"""Shared bar schema and OHLCV aggregation."""

from __future__ import annotations

import polars as pl

from pipeline_v2.constants import MARKET_COLUMNS, TIME_WIDTH_MS

BAR_FLOATS = (
    "open", "high", "low", "close", "vwap", "volume", "volume_usd",
    "buy_volume", "sell_volume", "buy_usd", "sell_usd", "signed_usd",
)

BAR_SCHEMA: dict[str, pl.DataType] = {
    "asset": pl.Utf8,
    "bar_type": pl.Utf8,
    "bar_id": pl.Int64,
    "ts_open_ms": pl.Int64,
    "ts_ms": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "vwap": pl.Float64,
    "volume": pl.Float64,
    "volume_usd": pl.Float64,
    "buy_volume": pl.Float64,
    "sell_volume": pl.Float64,
    "buy_usd": pl.Float64,
    "sell_usd": pl.Float64,
    "n_trades": pl.Int64,
    "signed_usd": pl.Float64,
}


def empty_bars() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


def close_of(ts_ms: pl.Expr, width_ms: int) -> pl.Expr:
    """Right edge of the left-open, right-closed bucket containing ``ts_ms``.

    A trade at exactly a boundary belongs to the bucket that closes there.
    """
    return (
        pl.when(ts_ms % width_ms == 0)
        .then(ts_ms)
        .otherwise((ts_ms // width_ms + 1) * width_ms)
    )


def cast_bars(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return empty_bars()
    df = df.with_columns([pl.col(c).cast(dt) for c, dt in BAR_SCHEMA.items() if c in df.columns])
    # Stable column order.
    return df.select([c for c in MARKET_COLUMNS if c in df.columns])


def collapse_duplicate_closes(df: pl.DataFrame) -> pl.DataFrame:
    """Several event bars can close inside one millisecond. Merge them.

    The label stays ``ts_ms``. Trade counts and volume are summed, so the
    reconciliation gate still holds. A burst that crosses the dollar
    threshold more than once in the same millisecond becomes one bar.
    """
    if df.is_empty() or "ts_ms" not in df.columns or df["ts_ms"].n_unique() == df.height:
        return df
    df = df.with_row_index("_ord")
    return df.group_by("ts_ms", maintain_order=True).agg([
        pl.col("ts_open_ms").sort_by("_ord").first().alias("ts_open_ms"),
        pl.col("open").sort_by("_ord").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").sort_by("_ord").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("volume_usd").sum().alias("volume_usd"),
        pl.col("buy_volume").sum().alias("buy_volume"),
        pl.col("sell_volume").sum().alias("sell_volume"),
        pl.col("buy_usd").sum().alias("buy_usd"),
        pl.col("sell_usd").sum().alias("sell_usd"),
        pl.col("n_trades").sum().cast(pl.Int64).alias("n_trades"),
    ])


def finalize(df: pl.DataFrame, asset: str, bar_type: str) -> pl.DataFrame:
    if df.is_empty():
        return empty_bars()
    df = collapse_duplicate_closes(df)
    df = df.sort("ts_ms").with_columns([
        pl.lit(asset).alias("asset"),
        pl.lit(bar_type).alias("bar_type"),
        pl.when(pl.col("volume") > 0)
        .then(pl.col("volume_usd") / pl.col("volume"))
        .otherwise(None)
        .alias("vwap"),
        (pl.col("buy_usd") - pl.col("sell_usd")).alias("signed_usd"),
    ])
    df = df.drop("bar_id") if "bar_id" in df.columns else df
    df = df.with_row_index("bar_id").with_columns(pl.col("bar_id").cast(pl.Int64))
    return cast_bars(df)


def _side_sums(prefix: str = "") -> list[pl.Expr]:
    """Buy is the aggressive buyer: is_buyer_maker is False on Binance."""
    maker = pl.col("is_buyer_maker")
    qty = pl.col(f"{prefix}qty") if prefix else pl.col("qty")
    usd = pl.col("_usd")
    return [
        pl.col("qty").filter(~maker).sum().fill_null(0.0).alias("buy_volume"),
        pl.col("qty").filter(maker).sum().fill_null(0.0).alias("sell_volume"),
        usd.filter(~maker).sum().fill_null(0.0).alias("buy_usd"),
        usd.filter(maker).sum().fill_null(0.0).alias("sell_usd"),
    ]


def aggregate_trades(df: pl.DataFrame, key: str) -> pl.DataFrame:
    """OHLCV from trades already tagged with a group key. Trades must be time-sorted."""
    if df.is_empty():
        return df
    if "_seq" not in df.columns:
        df = df.with_row_index("_seq")
    order = "_seq"
    return df.group_by(key, maintain_order=True).agg([
        pl.col("ts_ms").sort_by(order).first().alias("ts_open_ms"),
        pl.col("ts_ms").sort_by(order).last().alias("ts_close_trade_ms"),
        pl.col("price").sort_by(order).first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").sort_by(order).last().alias("close"),
        pl.col("qty").sum().alias("volume"),
        pl.col("_usd").sum().alias("volume_usd"),
        *_side_sums(),
        pl.len().cast(pl.Int64).alias("n_trades"),
    ])


def aggregate_child_bars(df: pl.DataFrame, key: str) -> pl.DataFrame:
    """Aggregate finer bars. open/close follow time, volumes sum. Not `.last()`."""
    if df.is_empty():
        return df
    return df.group_by(key, maintain_order=True).agg([
        pl.col("open").sort_by("ts_open_ms").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").sort_by("ts_ms").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.col("volume_usd").sum().alias("volume_usd"),
        pl.col("buy_volume").sum().alias("buy_volume"),
        pl.col("sell_volume").sum().alias("sell_volume"),
        pl.col("buy_usd").sum().alias("buy_usd"),
        pl.col("sell_usd").sum().alias("sell_usd"),
        pl.col("n_trades").sum().cast(pl.Int64).alias("n_trades"),
        pl.col("ts_open_ms").min().alias("ts_open_ms"),
    ])


def width_of(bar_type: str) -> int:
    if bar_type not in TIME_WIDTH_MS:
        raise KeyError(bar_type)
    return TIME_WIDTH_MS[bar_type]
