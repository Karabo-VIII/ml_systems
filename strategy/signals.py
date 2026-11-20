"""The first three shortlisted strategies. Each returns a target in [-1, 1].

The value at bar t is decided from information knowable at that close. The
engine fills it on the next bar. Do not shift these outputs yourself.
Do not read ``target_return_*``.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from strategy.engine import bars_per_year


def _close_high_low(bars: pl.DataFrame) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if "close" not in bars.columns:
        raise ValueError("bars must have a close column")
    close = np.asarray(bars["close"].to_numpy(), dtype=np.float64)
    high = np.asarray(bars["high"].to_numpy(), dtype=np.float64) if "high" in bars.columns else None
    low = np.asarray(bars["low"].to_numpy(), dtype=np.float64) if "low" in bars.columns else None
    return close, high, low


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    c = np.cumsum(np.where(np.isfinite(x), x, 0.0))
    counts = np.cumsum(np.isfinite(x).astype(np.int64))
    c = np.insert(c, 0, 0.0)
    counts = np.insert(counts, 0, 0)
    width = counts[window:] - counts[:-window]
    total = c[window:] - c[:-window]
    ok = width == window
    out[window - 1 :] = np.where(ok, total / window, np.nan)
    return out


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_window_stat(x, window, lambda sl: float(sl.max()))


def _rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling_window_stat(x, window, lambda sl: float(sl.min()))


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        raise ValueError("vol_lookback must be at least 2")
    return _rolling_window_stat(x, window, lambda sl: float(np.std(sl, ddof=1)))


def _rolling_window_stat(x: np.ndarray, window: int, fn) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(x.shape, np.nan, dtype=np.float64)
    for i in range(window - 1, x.size):
        sl = x[i - window + 1 : i + 1]
        if np.all(np.isfinite(sl)):
            out[i] = fn(sl)
    return out


def regime_gate(bars: pl.DataFrame, window: int = 200) -> np.ndarray:
    """Long 1 when close is above its trailing ``window``-bar mean, else cash.

    Default window is 200, the slow regime use in docs/STRATEGY_LAYER.md.
    Warmup bars are 0.
    """
    close, _, _ = _close_high_low(bars)
    ma = _rolling_mean(close, window)
    out = np.zeros(close.shape[0], dtype=np.float64)
    ok = np.isfinite(ma) & np.isfinite(close)
    out[ok & (close > ma)] = 1.0
    return out


def donchian_breakout(
    bars: pl.DataFrame,
    entry: int = 20,
    exit: int = 10,
) -> np.ndarray:
    """Long when close breaks the prior ``entry``-bar high; flat on the prior ``exit``-bar low.

    The channel uses the previous bars only (the current bar is not in the
    extremum). Between entry and exit the previous target is held. Warmup is 0.
    """
    close, high, low = _close_high_low(bars)
    if high is None or low is None:
        raise ValueError("Donchian needs high and low")
    n = close.size
    prior_high = np.full(n, np.nan, dtype=np.float64)
    prior_low = np.full(n, np.nan, dtype=np.float64)
    # rolling_*[t] is the extremum of x[t-window+1:t+1]. Shift so bar t sees
    # the extremum of the prior ``window`` bars, x[t-window:t].
    prior_high[entry:] = _rolling_max(high, entry)[entry - 1 : n - 1]
    prior_low[exit:] = _rolling_min(low, exit)[exit - 1 : n - 1]
    out = np.zeros(n, dtype=np.float64)
    pos = 0.0
    ready = max(entry, exit)
    for t in range(n):
        if t < ready or not np.isfinite(close[t]):
            out[t] = 0.0
            pos = 0.0
            continue
        if np.isfinite(prior_high[t]) and close[t] > prior_high[t]:
            pos = 1.0
        elif np.isfinite(prior_low[t]) and close[t] < prior_low[t]:
            pos = 0.0
        out[t] = pos
    return out


def vol_scaled_momentum(
    bars: pl.DataFrame,
    lookback: int = 63,
    vol_lookback: int = 20,
    target_ann_vol: float = 0.10,
    long_only: bool = True,
) -> np.ndarray:
    """Sign of the trailing ``lookback``-bar return, sized to a vol target.

    Weight is ``target_ann_vol / annualised trailing std``, clipped to
    ``[-1, 1]``. ``target_ann_vol`` is a research parameter (default 0.10),
    not a measured constant. Realized vol is the trailing standard deviation
    of one-bar close-to-close returns, annualised with the same cadence rule
    as the engine. FX ``rv_*`` is the same object; it is recomputed here so
    the function runs on bare OHLCV.

    The shortlist is long-or-cash: a negative trailing return is 0 when
    ``long_only`` is True. A later book of assets each sized to the same
    target is equal-risk; this function is single-asset.

    Warmup bars are 0.
    """
    close, _, _ = _close_high_low(bars)
    n = close.size
    out = np.zeros(n, dtype=np.float64)
    if n < max(lookback, vol_lookback) + 1:
        return out
    prev = np.empty(n, dtype=np.float64)
    prev[:lookback] = np.nan
    prev[lookback:] = close[: n - lookback]
    mom = close / prev - 1.0
    r1 = np.full(n, np.nan, dtype=np.float64)
    valid = (close[:-1] > 0) & np.isfinite(close[:-1]) & np.isfinite(close[1:])
    r1[1:] = np.where(valid, close[1:] / close[:-1] - 1.0, np.nan)
    rv = _rolling_std(r1, vol_lookback)
    ts = (
        np.asarray(bars["ts_ms"].to_numpy(), dtype=np.int64)
        if "ts_ms" in bars.columns
        else np.arange(n, dtype=np.int64)
    )
    bar_type = str(bars["bar_type"][0]) if "bar_type" in bars.columns and n else None
    bpy = bars_per_year(ts, bar_type)
    ann_rv = rv * (math.sqrt(bpy) if bpy > 0 else float("nan"))
    scale = np.where((ann_rv > 0) & np.isfinite(ann_rv), target_ann_vol / ann_rv, np.nan)
    scale = np.clip(scale, 0.0, 1.0)
    if long_only:
        raw = np.where(mom > 0, scale, 0.0)
    else:
        raw = np.clip(np.sign(mom) * scale, -1.0, 1.0)
    ok = np.isfinite(raw)
    out[ok] = raw[ok]
    return out
