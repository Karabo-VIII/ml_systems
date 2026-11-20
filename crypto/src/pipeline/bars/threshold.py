"""Dollar, tick, and volume bars.

Assignment is the global ``floor(cumsum / threshold)`` rule named in the
contract, applied as one series so chunk boundaries cannot renumber bars.
The open tail (the last id, when the running total is not an exact multiple
of the threshold) is not written. Its trade count is returned so the
reconciliation gate can add it back.
"""

from __future__ import annotations

import math

import polars as pl

from pipeline.bars.schema import aggregate_trades, finalize


def _prepare(trades: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    df = trades.sort(["ts_us", "agg_trade_id"] if "ts_us" in trades.columns else ["ts_ms"])
    return df.with_columns([
        (pl.col("price") * pl.col("qty")).alias("_usd"),
        value.alias("_value"),
    ])


def _ids(df: pl.DataFrame, threshold: float, cum_start: float) -> pl.DataFrame:
    return df.with_columns(
        ((pl.col("_value").cum_sum() + cum_start) / threshold).floor().cast(pl.Int64).alias("_raw")
    )


def _partial(df: pl.DataFrame) -> pl.DataFrame:
    grouped = aggregate_trades(df, "_raw")
    return grouped.rename({"ts_close_trade_ms": "ts_ms"})


def _exact(total: float, threshold: float) -> bool:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    ratio = total / threshold
    frac = ratio - math.floor(ratio)
    return frac < 1e-8 or frac > 1 - 1e-8


def threshold_bars(
    trades: pl.DataFrame,
    *,
    threshold: float,
    asset: str,
    bar_type: str,
    value: pl.Expr,
    cum_start: float = 0.0,
) -> tuple[pl.DataFrame, int, float]:
    """Return (complete bars, excluded open-tail trade count, new cum of value)."""
    if trades.is_empty():
        return finalize(trades, asset, bar_type), 0, cum_start
    df = _ids(_prepare(trades, value), threshold, cum_start)
    total = float(df["_value"].sum()) + cum_start
    partial = _partial(df)
    excluded = 0
    if not _exact(total, threshold):
        last = int(df["_raw"].max())
        excluded = int(df.filter(pl.col("_raw") == last).height)
        partial = partial.filter(pl.col("_raw") < last)
    if partial.is_empty():
        return finalize(partial, asset, bar_type), excluded, total
    partial = partial.drop("_raw")
    return finalize(partial, asset, bar_type), excluded, total


def dollar_bars(trades: pl.DataFrame, threshold: float, asset: str) -> tuple[pl.DataFrame, int]:
    bars, excluded, _ = threshold_bars(
        trades, threshold=threshold, asset=asset, bar_type="dollar",
        value=pl.col("price") * pl.col("qty"),
    )
    return bars, excluded


def tick_bars(trades: pl.DataFrame, trades_per_bar: int, asset: str) -> tuple[pl.DataFrame, int]:
    bars, excluded, _ = threshold_bars(
        trades, threshold=float(trades_per_bar), asset=asset, bar_type="tick",
        value=pl.lit(1.0),
    )
    return bars, excluded


def volume_bars(trades: pl.DataFrame, qty_per_bar: float, asset: str) -> tuple[pl.DataFrame, int]:
    bars, excluded, _ = threshold_bars(
        trades, threshold=float(qty_per_bar), asset=asset, bar_type="volume",
        value=pl.col("qty"),
    )
    return bars, excluded


class DollarAccumulator:
    """Day-at-a-time dollar bars. The running sum is the only state carried across days."""

    def __init__(self, threshold: float):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self.cum = 0.0
        self._partials: list[pl.DataFrame] = []

    def add(self, trades: pl.DataFrame) -> None:
        if trades is None or trades.is_empty():
            return
        df = _ids(_prepare(trades, pl.col("price") * pl.col("qty")), self.threshold, self.cum)
        self._partials.append(_partial(df))
        self.cum += float(df["_value"].sum())

    def finish(self, asset: str) -> tuple[pl.DataFrame, int]:
        if not self._partials:
            return finalize(pl.DataFrame(), asset, "dollar"), 0
        df = pl.concat(self._partials, how="diagonal_relaxed")
        # Fragments of the same raw id, in time order, collapse to one bar.
        df = df.with_row_index("_seq")
        merged = df.group_by("_raw", maintain_order=True).agg([
            pl.col("ts_open_ms").sort_by("_seq").first().alias("ts_open_ms"),
            pl.col("ts_ms").sort_by("_seq").last().alias("ts_ms"),
            pl.col("open").sort_by("_seq").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").sort_by("_seq").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("volume_usd").sum().alias("volume_usd"),
            pl.col("buy_volume").sum().alias("buy_volume"),
            pl.col("sell_volume").sum().alias("sell_volume"),
            pl.col("buy_usd").sum().alias("buy_usd"),
            pl.col("sell_usd").sum().alias("sell_usd"),
            pl.col("n_trades").sum().cast(pl.Int64).alias("n_trades"),
        ])
        excluded = 0
        if not _exact(self.cum, self.threshold):
            last = int(merged["_raw"].max())
            excluded = int(merged.filter(pl.col("_raw") == last)["n_trades"].sum() or 0)
            merged = merged.filter(pl.col("_raw") < last)
        if merged.is_empty():
            return finalize(merged, asset, "dollar"), excluded
        return finalize(merged.drop("_raw"), asset, "dollar"), excluded


def dollar_bars_chunked(
    chunks: list[pl.DataFrame],
    threshold: float,
    asset: str,
) -> tuple[pl.DataFrame, int]:
    """Same bar ids as concatenating the chunks first. Residual crosses chunks."""
    if not chunks:
        return finalize(pl.DataFrame(), asset, "dollar"), 0
    prepared = []
    cum = 0.0
    for chunk in chunks:
        if chunk is None or chunk.is_empty():
            continue
        df = _ids(_prepare(chunk, pl.col("price") * pl.col("qty")), threshold, cum)
        prepared.append(df)
        cum += float(df["_value"].sum())
    if not prepared:
        return finalize(pl.DataFrame(), asset, "dollar"), 0
    df = pl.concat(prepared, how="diagonal_relaxed")
    partial = _partial(df)
    excluded = 0
    if not _exact(cum, threshold):
        last = int(df["_raw"].max())
        excluded = int(df.filter(pl.col("_raw") == last).height)
        partial = partial.filter(pl.col("_raw") < last)
    if partial.is_empty():
        return finalize(partial, asset, "dollar"), excluded
    return finalize(partial.drop("_raw"), asset, "dollar"), excluded
