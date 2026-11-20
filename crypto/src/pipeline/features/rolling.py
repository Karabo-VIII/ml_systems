"""Causal rolling helpers. Warmup stays null. Nothing is filled with 0."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


def rolling_z(col: str, window: int) -> pl.Expr:
    """Trailing z-score. Null until ``window`` finite observations. Null when sigma is ~0."""
    x = pl.col(col)
    mu = x.rolling_mean(window_size=window, min_samples=window)
    sd = x.rolling_std(window_size=window, min_samples=window)
    z = (x - mu) / (sd + 1e-5)
    return pl.when(sd.is_null() | (sd < 1e-12) | x.is_null()).then(None).otherwise(z.clip(-5.0, 5.0))


def frac_diff(series: np.ndarray, d: float, window: int) -> np.ndarray:
    """Fractional difference. ``out[i]`` uses ``series[i-window:i]`` only (excludes i)."""
    n = len(series)
    out = np.full(n, np.nan)
    if window < 1 or n <= window:
        return out
    w = np.empty(window, dtype=np.float64)
    w[0] = 1.0
    for k in range(1, window):
        w[k] = -w[k - 1] / k * (d - k + 1)
    weights = np.ascontiguousarray(w[::-1])
    values = np.asarray(series, dtype=np.float64)
    if not np.isfinite(values).all():
        for i in range(window, n):
            chunk = values[i - window:i]
            if np.isfinite(chunk).all():
                out[i] = float(np.dot(weights, chunk))
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window)
    # view[i-window] = values[i-window:i] for i in [window, n)
    out[window:] = view[: n - window] @ weights
    return out


def hurst_rs(log_returns: np.ndarray, window: int) -> np.ndarray:
    """Rescaled-range Hurst on a trailing window of log returns. Null during warmup."""
    n = len(log_returns)
    out = np.full(n, np.nan)
    if n <= window:
        return out
    values = np.asarray(log_returns, dtype=np.float64)
    log_w = math.log(window)
    for i in range(window, n):
        chunk = values[i - window:i]
        if not np.isfinite(chunk).all():
            continue
        if np.max(chunk) == np.min(chunk):
            out[i] = 0.5
            continue
        mean = chunk.mean()
        z = np.cumsum(chunk - mean)
        r = float(np.max(z) - np.min(z))
        s = float(np.std(chunk))
        if s < 1e-12 or r <= 0:
            out[i] = 0.5
        else:
            out[i] = math.log(r / s) / log_w
    return out


def _pattern3(a: float, b: float, c: float) -> int:
    """Ordinal pattern of a triple. Ties keep the earlier index smaller."""
    if a <= b:
        if b <= c:
            return 0
        if a <= c:
            return 1
        return 2
    if a <= c:
        return 3
    if b <= c:
        return 4
    return 5


def permutation_entropy(returns: np.ndarray, window: int, m: int = 3) -> np.ndarray:
    """Normalized permutation entropy in [0, 1]. Null until the window is full.

    The value at t uses returns in ``[t-window+1, t]`` only.
    """
    n = len(returns)
    out = np.full(n, np.nan)
    if m != 3 or window < m or n < window:
        return out
    values = np.asarray(returns, dtype=np.float64)
    log_nf = math.log(6.0)
    for t in range(window - 1, n):
        counts = [0, 0, 0, 0, 0, 0]
        left = t - window + 1
        total = 0
        last_start = t - m + 1
        for i in range(left, last_start + 1):
            a, b, c = values[i], values[i + 1], values[i + 2]
            if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
                continue
            counts[_pattern3(a, b, c)] += 1
            total += 1
        if total == 0:
            continue
        entropy = 0.0
        for c in counts:
            if c:
                p = c / total
                entropy -= p * math.log(p)
        out[t] = entropy / log_nf
    return out


def kyle_lambda(close: np.ndarray, signed_usd: np.ndarray, window: int) -> np.ndarray:
    """Trailing correlation of log return with signed USD. Null during warmup."""
    n = len(close)
    out = np.full(n, np.nan)
    log_ret = np.full(n, np.nan)
    for i in range(1, n):
        prev = close[i - 1]
        if prev > 0 and math.isfinite(prev) and math.isfinite(close[i]) and close[i] > 0:
            log_ret[i] = math.log(close[i] / prev)
    signed = np.asarray(signed_usd, dtype=np.float64)
    for t in range(window, n):
        lr = log_ret[t - window:t]
        sv = signed[t - window:t]
        mask = np.isfinite(lr) & np.isfinite(sv)
        if int(mask.sum()) < window:
            continue
        x = sv[mask]
        y = lr[mask]
        x = x - x.mean()
        y = y - y.mean()
        denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
        if denom > 1e-15:
            out[t] = float(np.dot(x, y) / denom)
    return out


def to_series(name: str, values: np.ndarray) -> pl.Series:
    s = pl.Series(name, values)
    return s.fill_nan(None)
