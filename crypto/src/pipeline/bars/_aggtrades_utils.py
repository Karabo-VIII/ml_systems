"""Shared aggTrades read-side utilities for bar builders.

Two responsibilities, packaged together:

1. Timestamp-scale normalization. Binance switched aggTrades from 13-digit
   ms to 16-digit us (microseconds) somewhere between 2024 and 2026. The
   CLAUDE.md project invariant is 13-digit ms in [1.5e12, 2.0e12]. Bar
   builders passing timestamps through unmodified inherit the scale of
   their input -- corrupting output for any year where the source switched.

2. Sort by timestamp ascending. Starting around 2026-03, Binance aggTrades
   arrive out-of-order (observed ~50% unsorted rows on 2026-03-04 BTCUSDT).
   Range / DIB / runs builders assume sorted input; unsorted input causes
   adjacent rows to jump in price by ~3% on average, triggering range-bar
   close every 1-2 ticks -> 250K bars/day instead of 30.

prepare_aggtrades() is the canonical entry point: normalizes ts scale,
then sorts. Idempotent on already-clean input.

Scale magnitudes at year 2026:
    ms:  ~1.77e12 (13 digits) -- CLAUDE.md target
    us:  ~1.77e15 (16 digits) -- Binance 2026 format
    ns:  ~1.77e18 (19 digits) -- (defensive: not observed but handled)
"""
from __future__ import annotations

import numpy as np
import polars as pl

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def _wrap(f):
            return f
        return f if (len(a) == 1 and callable(a[0])) else _wrap

__contract__ = {
    "kind": "support",
    "outputs": {
        "callable": "prepare_aggtrades(df, ts_col='timestamp') -> pl.DataFrame",
    },
    "invariants": {
        "output_ts_in_ms_range": "ts in [1.5e12, 2.0e12] after normalization",
        "output_monotone_asc": "ts strictly non-decreasing after sort",
        "idempotent": "calling twice on already-clean df is a noop",
    },
}

# Scale boundaries (year 2026 reference):
#   ms < 2e12  (13 digits)
#   us < 2e15  (16 digits)
#   ns < 2e18  (19 digits)
_MS_MAX = 2_000_000_000_000          # 2e12
_US_MAX = 2_000_000_000_000_000      # 2e15


def normalize_ts_to_ms(df: pl.DataFrame, ts_col: str = "timestamp") -> pl.DataFrame:
    """Ensure ts_col is in 13-digit ms scale. Idempotent.

    Detection (year 2026 magnitudes):
        ts_max <  2e12  -> ms (already), return as-is
        ts_max <  2e15  -> us (16-digit), divide by 1_000
        ts_max >= 2e15  -> ns (19-digit), divide by 1_000_000

    Args:
        df: polars DataFrame from aggTrades parquet
        ts_col: timestamp column name (default 'timestamp')

    Returns:
        df with ts_col downscaled to ms if needed.
    """
    if df.is_empty() or ts_col not in df.columns:
        return df
    ts_max = df[ts_col].max()
    if ts_max is None or ts_max < _MS_MAX:
        return df
    if ts_max < _US_MAX:
        # us -> ms
        return df.with_columns((pl.col(ts_col) // 1_000).alias(ts_col))
    # ns -> ms
    return df.with_columns((pl.col(ts_col) // 1_000_000).alias(ts_col))


def prepare_aggtrades(df: pl.DataFrame, ts_col: str = "timestamp") -> pl.DataFrame:
    """Canonical aggTrades preparation: normalize ts scale to ms, then sort by ts asc.

    Use this in bar builders instead of raw normalize_ts_to_ms. Idempotent.

    Args:
        df: polars DataFrame from aggTrades parquet
        ts_col: timestamp column name (default 'timestamp')

    Returns:
        df with ts_col in ms scale, rows sorted by ts asc.
    """
    df = normalize_ts_to_ms(df, ts_col=ts_col)
    if df.is_empty() or ts_col not in df.columns:
        return df
    return df.sort(ts_col)


@njit(cache=False)
def _imbalance_bar_ids_nb(signed: np.ndarray, threshold: float) -> np.ndarray:
    n = len(signed)
    ids = np.empty(n, dtype=np.int64)
    cur = 0.0
    bid = 0
    for i in range(n):
        cur += signed[i]
        ids[i] = bid
        # AFML imbalance bar: the trade that pushes |accumulated| past the
        # threshold CLOSES the current bar; accumulator resets for the next bar.
        if cur >= threshold or cur <= -threshold:
            bid += 1
            cur = 0.0
    return ids


def imbalance_bar_ids(signed_values, threshold: float) -> np.ndarray:
    """Assign per-trade bar ids using AFML imbalance-bar RESET semantics.

    A bar closes when the running sum of signed values SINCE THE LAST BAR
    crosses +/- threshold (then the accumulator resets to 0). This is the
    canonical definition -- distinct from floor(cumulative_sum / threshold),
    which never resets and groups by absolute cumulative level (wrong: a
    mean-reverting signed flow revisits the same level repeatedly).

    Returns a monotone-non-decreasing int64 array of bar ids (0,1,2,...),
    contiguous within the input. Global uniqueness across files is the
    caller's responsibility (reassign after concatenation).
    """
    arr = np.asarray(signed_values, dtype=np.float64)
    if arr.size == 0:
        return np.empty(0, dtype=np.int64)
    return _imbalance_bar_ids_nb(arr, float(threshold))


__all__ = ["normalize_ts_to_ms", "prepare_aggtrades", "imbalance_bar_ids"]
