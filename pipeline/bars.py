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
    """Close of the bucket that a *finished child bar* belongs to.

    A child bar is already stamped at its own close. A 1m bar that closes at
    exactly 12:00:00 is the last child of the 1h bar that closes at 12:00:00,
    so a stamp on a boundary stays on it (left-open, right-closed). Raw trades
    and ticks use :func:`event_close_of` instead.

    Buckets are aligned to 00:00:00 UTC: ``width_ms`` always divides a UTC
    day, and the Unix epoch used here is midnight UTC, so 2h/6h/8h closes
    land on 00:00, 02:00, … or 00:00, 06:00, 12:00, 18:00, and so on.
    """
    return (
        pl.when(ts_ms % width_ms == 0)
        .then(ts_ms)
        .otherwise((ts_ms // width_ms + 1) * width_ms)
    )


def event_close_of(ts_ms: pl.Expr, width_ms: int) -> pl.Expr:
    """Close of the bucket that a raw event (trade or tick) belongs to.

    A bar covers ``[t_open, t_close)`` and is stamped ``t_close``. An event at
    exactly ``t_close`` belongs to the *next* bar. This is the Binance kline
    convention (``open_time`` inclusive, the next ``open_time`` exclusive), so
    a 1m bar built from trades matches the exchange kline exactly. It is also
    causal: the bar stamped ``t`` holds only events strictly before ``t``, and
    nothing in a bar is knowable later than its stamp. A trade at 00:00:00.000
    on the first of a month lands in the first 1m bar of that month, so no time
    bar straddles a month boundary.
    """
    return (ts_ms // width_ms + 1) * width_ms


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
        *([pl.col("bar_id").sort_by("_ord").first().alias("bar_id")] if "bar_id" in df.columns else []),
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
    # bar_id is unique per asset and bar type across every partition: time bars
    # carry their close index (ts_ms // width) and event bars carry the global
    # sequence number the accumulator assigned. Only a bare frame is numbered here.
    if "bar_id" not in df.columns:
        df = df.with_row_index("bar_id")
    df = df.with_columns(pl.col("bar_id").cast(pl.Int64))
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
    df = df.with_columns(event_close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = aggregate_trades(df, "_close")
    grouped = grouped.with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
        (pl.col("_close") // width_ms).alias("bar_id"),
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
        (pl.col("_close") // width_ms).alias("bar_id"),
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
    df = pl.concat(frames, how="diagonal_relaxed").sort("ts_ms", maintain_order=True)
    # Re-aggregate on the 1m close so a split midnight bucket becomes one bar.
    df = df.with_columns(pl.col("ts_ms").alias("_close"))
    grouped = aggregate_child_bars(df, "_close")
    grouped = grouped.with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
        (pl.col("_close") // width_ms).alias("bar_id"),
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

"""Dollar, tick, and volume bars (event bars).

A bar closes on the trade that makes its running total reach the threshold,
and that trade is the LAST trade of the bar. The next bar starts at zero with
the next trade. So a bar stamped ``t`` contains only trades up to ``t`` and it
is knowable at ``t``: nothing about its close depends on a later trade. Every
completed bar holds at least ``threshold`` (a single large print can overshoot
it). The trades after the last completed bar are the open tail. They are not
written, and their count is returned so the reconciliation gate can add it back.

The dollar threshold can change from one UTC day to the next (see
:class:`DollarAccumulator`). It is fixed when a bar opens and never changes
mid-bar, so a bar's size is known from information before its first trade.
The first day of an adaptive chain has no history and closes no bar.
"""

import math

import numpy as np
import polars as pl

from pipeline.constants import DAY_MS

# A trade that lands within this fraction of the threshold counts as reaching it.
# It only absorbs float error in the running sum (1e-9 of a $2M bar is $0.002).
_EPS = 1e-9


def _prepare(trades: pl.DataFrame, value: pl.Expr) -> pl.DataFrame:
    df = trades.sort(["ts_us", "agg_trade_id"] if "ts_us" in trades.columns else ["ts_ms"])
    return df.with_columns([
        (pl.col("price") * pl.col("qty")).alias("_usd"),
        value.alias("_value"),
    ])


def assign_ids(
    values: np.ndarray,
    *,
    open_threshold: float | None,
    open_cum: float,
    new_threshold: float,
    first_id: int,
) -> tuple[np.ndarray, int, float | None, float]:
    """Number the trades of one chunk into bars.

    ``open_threshold`` / ``open_cum`` describe a bar already open from earlier
    chunks (``None`` when there is none). Returns ``(ids, next_id,
    open_threshold, open_cum)``. When the chunk ends exactly on a close, there
    is no open bar afterwards and ``open_threshold`` is ``None``.
    """
    n = len(values)
    ids = np.empty(n, dtype=np.int64)
    if n == 0:
        return ids, first_id, open_threshold, open_cum
    c = np.cumsum(values, dtype=np.float64)
    cur = first_id
    pos = 0
    base = 0.0
    if open_threshold is not None:
        target = base + (open_threshold - open_cum)
        j = int(np.searchsorted(c, target - _EPS * open_threshold, side="left"))
        if j >= n:
            ids[:] = cur
            return ids, cur, open_threshold, open_cum + float(c[-1])
        ids[: j + 1] = cur
        pos, base, cur = j + 1, float(c[j]), cur + 1
    t = float(new_threshold)
    while pos < n:
        j = int(np.searchsorted(c, base + t - _EPS * t, side="left"))
        if j >= n:
            ids[pos:] = cur
            return ids, cur, t, float(c[-1]) - base
        ids[pos : j + 1] = cur
        base, pos, cur = float(c[j]), j + 1, cur + 1
    return ids, cur, None, 0.0


def _partial(df: pl.DataFrame) -> pl.DataFrame:
    from pipeline.bars import aggregate_trades

    grouped = aggregate_trades(df, "_raw")
    return grouped.rename({"ts_close_trade_ms": "ts_ms"})


def threshold_bars(
    trades: pl.DataFrame,
    *,
    threshold: float,
    asset: str,
    bar_type: str,
    value: pl.Expr,
    cum_start: float = 0.0,
) -> tuple[pl.DataFrame, int, float]:
    """One-shot event bars. Returns (complete bars, open-tail trade count, open-tail total)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if trades.is_empty():
        return finalize(trades, asset, bar_type), 0, cum_start
    df = _prepare(trades, value)
    open_t = float(threshold) if cum_start > 0 else None
    ids, _next, tail_t, tail_cum = assign_ids(
        df["_value"].to_numpy(), open_threshold=open_t, open_cum=cum_start,
        new_threshold=float(threshold), first_id=0,
    )
    df = df.with_columns(pl.Series("_raw", ids))
    partial = _partial(df)
    excluded = 0
    if tail_t is not None:
        last = int(ids[-1])
        excluded = int((ids == last).sum())
        partial = partial.filter(pl.col("_raw") < last)
    if partial.is_empty():
        return finalize(partial, asset, bar_type), excluded, tail_cum
    partial = partial.with_columns(pl.col("_raw").alias("bar_id")).drop("_raw")
    return finalize(partial, asset, bar_type), excluded, tail_cum


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


def trailing_threshold(history: list[float], target_bars_per_day: float) -> float | None:
    """Dollar threshold for a new UTC day, or ``None`` when there is no history.

    ``history`` holds the notional of the most recent complete days, oldest first.
    The threshold is their median divided by the bars-per-day target, so the
    fabric carries about the same number of bars per day in 2020 and in 2026.
    Only days before the day being opened ever enter ``history``. With no usable
    history (the first day of a chain) there is no size: that day calibrates.
    """
    usable = [float(x) for x in history if x is not None and math.isfinite(x) and x > 0]
    if not usable:
        return None
    return float(np.median(usable)) / float(target_bars_per_day)


class DollarAccumulator:
    """Day-at-a-time dollar bars with a carried open bar.

    Constant mode (``target_bars_per_day=None``): every bar uses ``threshold``.
    Adaptive mode (``threshold=None``): the threshold for UTC day D is
    ``median(notional of the last window_days complete days before D) /
    target_bars_per_day`` (see :func:`trailing_threshold`). It is causal: day D's
    own trades never move day D's threshold. A bar that is still open at
    midnight keeps the threshold it opened with. The first day of the chain has
    no history and therefore no size: it is a calibration day. Its trades seed
    the history and close no bar; they are counted in ``calibration_trades``.

    State across calls (and across months, via :meth:`state` / :meth:`restore`):
    the open bar's partial trades, its running total and threshold, the next bar
    id, the day-notional history, the current day's running notional, and the
    calibration count.
    """

    def __init__(
        self,
        threshold: float | None,
        *,
        target_bars_per_day: float | None = None,
        window_days: int = 30,
    ):
        if target_bars_per_day is None:
            if threshold is None or threshold <= 0:
                raise ValueError("a constant dollar size must be positive")
        elif target_bars_per_day <= 0:
            raise ValueError("target_bars_per_day must be positive")
        elif threshold is not None:
            raise ValueError("adaptive dollar bars take no static size (the first day calibrates)")
        self.threshold = None if threshold is None else float(threshold)  # constant size only
        self.target_bars_per_day = None if target_bars_per_day is None else float(target_bars_per_day)
        self.window_days = int(window_days)
        self.history: list[float] = []
        self.cur_day: int | None = None
        self.cur_day_notional = 0.0
        self.cur_day_threshold: float | None = self.threshold
        self.cum = 0.0  # running total inside the open bar
        self.open_threshold: float | None = None
        self.next_id = 0
        self.calibration_trades = 0  # trades of the calibration day (in no bar)
        self.calibration_day: int | None = None  # UTC day index (ts_ms // DAY_MS) it calibrated on
        self.through: str | None = None  # last month whose bars are written ("YYYY-MM")
        self._open: list[pl.DataFrame] = []  # fragments of the open bar
        self._done: list[pl.DataFrame] = []  # completed bars not yet emitted

    # -- threshold schedule -------------------------------------------------
    def _open_day(self, day: int) -> None:
        if self.cur_day is not None:
            self.history.append(self.cur_day_notional)
            self.history = self.history[-self.window_days:]
        self.cur_day = day
        self.cur_day_notional = 0.0
        if self.target_bars_per_day is None:
            self.cur_day_threshold = self.threshold
        else:
            self.cur_day_threshold = trailing_threshold(self.history, self.target_bars_per_day)

    # -- feeding trades -----------------------------------------------------
    def add(self, trades: pl.DataFrame) -> None:
        if trades is None or trades.is_empty():
            return
        df = _prepare(trades, pl.col("price") * pl.col("qty"))
        days = (df["ts_ms"] // DAY_MS).to_numpy()
        if days[0] != days[-1] or not (np.diff(days) == 0).all():
            for day in sorted({int(d) for d in days}):
                self._add_day(df.filter(pl.col("ts_ms") // DAY_MS == day), day)
        else:
            self._add_day(df, int(days[0]))

    def _add_day(self, df: pl.DataFrame, day: int) -> None:
        if self.cur_day is None or day != self.cur_day:
            if self.cur_day is not None and day < self.cur_day:
                raise ValueError("trades must be added in day order")
            self._open_day(day)
        usd = df["_value"].to_numpy()
        if self.cur_day_threshold is None and self.open_threshold is None:
            # Calibration: no history yet, so no size. The trades seed the history.
            if self.calibration_day is None:
                self.calibration_day = int(day)
            self.calibration_trades += int(df.height)
            self.cur_day_notional += float(usd.sum())
            return
        carried_id = self.next_id if self.open_threshold is not None else None
        ids, next_id, open_t, open_cum = assign_ids(
            usd, open_threshold=self.open_threshold, open_cum=self.cum,
            new_threshold=self.cur_day_threshold, first_id=self.next_id,
        )
        parts = _partial(df.with_columns(pl.Series("_raw", ids)))
        last = int(ids[-1])
        still_open = open_t is not None
        done = parts.filter(pl.col("_raw") < last) if still_open else parts
        tail = parts.filter(pl.col("_raw") == last) if still_open else parts.clear()
        if carried_id is not None:
            # Fold the fragments carried from earlier chunks into the first id of this one.
            first = parts.filter(pl.col("_raw") == carried_id)
            if still_open and last == carried_id:
                self._open = [*self._open, first]
                tail = parts.clear()
            else:
                merged = self._merge(pl.concat([*self._open, first], how="diagonal_relaxed"))
                done = pl.concat([merged, done.filter(pl.col("_raw") != carried_id)], how="diagonal_relaxed")
                self._open = []
        if still_open and not tail.is_empty():
            self._open = [tail]
        if not still_open:
            self._open = []
        if not done.is_empty():
            self._done.append(done)
        self.cur_day_notional += float(usd.sum())
        self.next_id, self.open_threshold, self.cum = next_id, open_t, open_cum

    @staticmethod
    def _merge(df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_row_index("_seq")
        return df.group_by("_raw", maintain_order=True).agg([
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

    # -- reading bars out ---------------------------------------------------
    def tail_trades(self) -> int:
        """Trades in the bar that is still open."""
        if not self._open:
            return 0
        return int(pl.concat(self._open, how="diagonal_relaxed")["n_trades"].sum() or 0)

    def excluded_trades(self) -> int:
        """Trades fed so far that are in no completed bar: the open tail plus the calibration day."""
        return self.tail_trades() + int(self.calibration_trades)

    def emit_completed(self, asset: str) -> tuple[pl.DataFrame, int]:
        """Completed bars since the last call, and the trades in no completed bar yet.

        The open tail stays for the next chunk. The count includes the
        calibration day, so bars + count always equals the trades fed in.
        """
        excluded = self.excluded_trades()
        if not self._done:
            return finalize(pl.DataFrame(), asset, "dollar"), excluded
        merged = pl.concat(self._done, how="diagonal_relaxed").sort("_raw")
        self._done = []
        merged = merged.with_columns(pl.col("_raw").alias("bar_id")).drop("_raw")
        return finalize(merged, asset, "dollar"), excluded

    def finish(self, asset: str) -> tuple[pl.DataFrame, int]:
        return self.emit_completed(asset)

    # -- persistence --------------------------------------------------------
    def state(self) -> tuple[dict, pl.DataFrame | None]:
        """JSON-able scalars and the open bar's fragments, for ``_state/``."""
        meta = {
            "threshold": self.threshold,
            "target_bars_per_day": self.target_bars_per_day,
            "window_days": self.window_days,
            "history": self.history,
            "cur_day": self.cur_day,
            "cur_day_notional": self.cur_day_notional,
            "cur_day_threshold": self.cur_day_threshold,
            "cum": self.cum,
            "open_threshold": self.open_threshold,
            "next_id": self.next_id,
            "calibration_trades": self.calibration_trades,
            "calibration_day": self.calibration_day,
            "through": self.through,
            "n_partials": len(self._open),
            "excluded": self.excluded_trades(),
        }
        frag = pl.concat(self._open, how="diagonal_relaxed") if self._open else None
        return meta, frag

    @classmethod
    def restore(cls, meta: dict, frag: pl.DataFrame | None) -> "DollarAccumulator":
        raw = meta.get("threshold")
        acc = cls(
            None if raw is None else float(raw),
            target_bars_per_day=meta.get("target_bars_per_day"),
            window_days=int(meta.get("window_days", 30)),
        )
        acc.history = [float(x) for x in meta.get("history", [])]
        acc.cur_day = meta.get("cur_day")
        acc.cur_day_notional = float(meta.get("cur_day_notional", 0.0))
        day_t = meta.get("cur_day_threshold", acc.threshold)
        acc.cur_day_threshold = None if day_t is None else float(day_t)
        acc.cum = float(meta.get("cum", 0.0))
        acc.open_threshold = meta.get("open_threshold")
        acc.next_id = int(meta.get("next_id", 0))
        acc.calibration_trades = int(meta.get("calibration_trades", 0))
        cal_day = meta.get("calibration_day")
        acc.calibration_day = None if cal_day is None else int(cal_day)
        acc.through = meta.get("through")
        if frag is not None and not frag.is_empty():
            acc._open = [frag]
        return acc


def dollar_bars_chunked(
    chunks: list[pl.DataFrame],
    threshold: float,
    asset: str,
) -> tuple[pl.DataFrame, int]:
    """Same bars as concatenating the chunks first. The open bar crosses chunks."""
    acc = DollarAccumulator(threshold)
    for chunk in chunks:
        acc.add(chunk)
    return acc.emit_completed(asset)


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
    # Stable sort: ticks in the same millisecond keep the feed's order, so the
    # open and close of a bar never depend on how a sort breaks a tie.
    df = ticks.sort("ts_ms", maintain_order=True).with_columns(event_close_of(pl.col("ts_ms"), width_ms).alias("_close"))
    grouped = df.group_by("_close", maintain_order=True).agg([
        pl.col("mid").first().alias("open"),
        pl.col("mid").max().alias("high"),
        pl.col("mid").min().alias("low"),
        pl.col("mid").last().alias("close"),
        pl.col("mid").mean().alias("vwap"),
        pl.len().cast(pl.Float64).alias("volume"),
        *_fx_null_sides(),
        pl.len().cast(pl.Int64).alias("n_trades"),
        pl.col("bid").last().alias("bid_close"),
        pl.col("ask").last().alias("ask_close"),
        pl.col("spread").mean().alias("spread_mean"),
        pl.col("spread").max().alias("spread_max"),
    ]).with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
        (pl.col("_close") // width_ms).alias("bar_id"),
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
    ]).with_columns([
        (pl.col("_close") - width_ms).alias("ts_open_ms"),
        pl.col("_close").alias("ts_ms"),
        (pl.col("_close") // width_ms).alias("bar_id"),
    ]).drop("_close")
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
