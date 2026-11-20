"""Shared bar schema and OHLCV aggregation."""

from __future__ import annotations

import polars as pl

from pipeline.constants import MARKET_COLUMNS, TIME_WIDTH_MS

# Kept when present. Crypto bars do not have them.
FX_BAR_COLUMNS = ("bid_close", "ask_close", "spread_mean", "spread_max")

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
    for col in FX_BAR_COLUMNS:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    order = [c for c in MARKET_COLUMNS if c in df.columns]
    order += [c for c in FX_BAR_COLUMNS if c in df.columns]
    return df.select(order)


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
    vwap = (
        pl.when(pl.col("volume_usd").is_not_null() & (pl.col("volume") > 0))
        .then(pl.col("volume_usd") / pl.col("volume"))
        .otherwise(pl.col("vwap") if "vwap" in df.columns else pl.lit(None, dtype=pl.Float64))
    )
    df = df.sort("ts_ms").with_columns([
        pl.lit(asset).alias("asset"),
        pl.lit(bar_type).alias("bar_type"),
        vwap.alias("vwap"),
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

"""True time bars. 1m is the base; coarser bars sum the 1m bars.

This is the opposite of the v51 cadence view, which copied the last dollar
bar in the bucket and called it a 1h candle.
"""


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

"""Dollar, tick, and volume bars.

Assignment is the global ``floor(cumsum / threshold)`` rule named in the
contract, applied as one series so chunk boundaries cannot renumber bars.
The open tail (the last id, when the running total is not an exact multiple
of the threshold) is not written. Its trade count is returned so the
reconciliation gate can add it back.
"""


import math

import polars as pl



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


def _fx_null_sides() -> list[pl.Expr]:
    return [
        pl.lit(None, dtype=pl.Float64).alias("volume_usd"),
        pl.lit(None, dtype=pl.Float64).alias("buy_volume"),
        pl.lit(None, dtype=pl.Float64).alias("sell_volume"),
        pl.lit(None, dtype=pl.Float64).alias("buy_usd"),
        pl.lit(None, dtype=pl.Float64).alias("sell_usd"),
    ]


def fx_ticks_to_time(ticks: pl.DataFrame, width_ms: int, asset: str, bar_type: str) -> pl.DataFrame:
    """Time bars from mid quotes. ``volume`` and ``n_trades`` are the tick count.

    ``volume_usd`` stays null: spot FX has no traded notional in this feed.
    ``vwap`` is the average mid. Bid, ask, and spread are quote statistics.
    """
    if ticks.is_empty():
        return finalize(ticks, asset, bar_type)
    df = ticks.sort("ts_ms").with_columns(close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = df.group_by("_close", maintain_order=True).agg([
        pl.col("mid").sort_by("ts_ms").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").sort_by("ts_ms").last().alias("close"),
        pl.col("mid").mean().alias("vwap"),
        pl.len().cast(pl.Float64).alias("volume"),
        *_fx_null_sides(),
        pl.len().cast(pl.Int64).alias("n_trades"),
        pl.col("bid").sort_by("ts_ms").last().alias("bid_close"),
        pl.col("ask").sort_by("ts_ms").last().alias("ask_close"),
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("spread").max().alias("spread_max"),
    ]).with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
    ]).drop("_close")
    return finalize(grouped, asset, bar_type)


def fx_bars_to_time(bars_1m: pl.DataFrame, width_ms: int, asset: str, bar_type: str) -> pl.DataFrame:
    """Coarser FX bars. Tick counts sum. Spread mean is tick-weighted. Sides stay null."""
    if bars_1m.is_empty():
        return finalize(bars_1m, asset, bar_type)
    df = bars_1m.sort("ts_ms").with_columns(close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = df.group_by("_close", maintain_order=True).agg([
        pl.col("open").sort_by("ts_open_ms").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").sort_by("ts_ms").last().alias("close"),
        ((pl.col("vwap") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
        pl.col("volume").sum().alias("volume"),
        *_fx_null_sides(),
        pl.col("n_trades").sum().cast(pl.Int64).alias("n_trades"),
        pl.col("bid_close").sort_by("ts_ms").last().alias("bid_close"),
        pl.col("ask_close").sort_by("ts_ms").last().alias("ask_close"),
        (
            (pl.col("spread_mean") * pl.col("n_trades").cast(pl.Float64)).sum()
            / pl.col("n_trades").sum()
        ).alias("spread_mean"),
        pl.col("spread_max").max().alias("spread_max"),
        pl.col("ts_open_ms").min().alias("ts_open_ms"),
    ]).with_columns(pl.col("_close").alias("ts_ms")).drop("_close")
    return finalize(grouped, asset, bar_type)


def build_fx_ladder(
    ticks: pl.DataFrame, asset: str, bar_types: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    wanted = list(bar_types) if bar_types is not None else list(TIME_WIDTH_MS)
    base = fx_ticks_to_time(ticks, TIME_WIDTH_MS["time_1m"], asset, "time_1m")
    out: dict[str, pl.DataFrame] = {}
    if "time_1m" in wanted:
        out["time_1m"] = base
    for name, width in TIME_WIDTH_MS.items():
        if name == "time_1m" or name not in wanted:
            continue
        out[name] = fx_bars_to_time(base, width, asset, name)
    return out
