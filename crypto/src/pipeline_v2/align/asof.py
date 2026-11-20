"""Backward as-of join on knowable_at.

A feature row attaches to a bar only when ``event_time + lag_ms <= bar.ts_ms``.
The join is the one producers call. It is not a self-test that nothing imports.
"""

from __future__ import annotations

import warnings

import polars as pl


def asof_join_knowable(
    base: pl.DataFrame,
    feature: pl.DataFrame,
    feature_cols: list[str],
    publication_lag_ms: int,
    *,
    base_time_col: str = "ts_ms",
    feature_time_col: str = "ts_ms",
    by: str | None = None,
    tolerance_ms: int | None = None,
) -> pl.DataFrame:
    """Attach ``feature_cols`` where ``knowable_at <= base ts``. Base row count is unchanged."""
    if publication_lag_ms < 0:
        raise ValueError(f"publication_lag_ms must be >= 0, got {publication_lag_ms}")
    if base.is_empty():
        return base
    for col in feature_cols:
        if col not in feature.columns:
            raise ValueError(f"feature missing {col}")
        if col in base.columns:
            raise ValueError(f"feature column {col} collides with a base column")
    if feature.is_empty():
        return base.with_columns([pl.lit(None, dtype=feature.schema.get(c, pl.Float64)).alias(c) for c in feature_cols])

    knowable = "_knowable_at"
    feat = feature.with_columns(
        (pl.col(feature_time_col).cast(pl.Int64) + pl.lit(publication_lag_ms)).alias(knowable)
    )
    keep = [knowable, *feature_cols]
    if by is not None:
        keep.append(by)
    feat = feat.select(keep).drop_nulls([knowable])
    # If several feature rows share a knowable instant, keep the last.
    feat = feat.sort(knowable).unique(subset=([by, knowable] if by else [knowable]), keep="last", maintain_order=True)

    order = "_asof_order"
    base_tagged = base.with_row_index(order)
    base_sorted = base_tagged.sort([by, base_time_col] if by else [base_time_col])
    feat_sorted = feat.sort([by, knowable] if by else [knowable])

    join_kwargs = dict(left_on=base_time_col, right_on=knowable, strategy="backward")
    if tolerance_ms is not None:
        join_kwargs["tolerance"] = int(tolerance_ms)
    if by is not None:
        join_kwargs["by"] = by
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")
        joined = base_sorted.join_asof(feat_sorted, **join_kwargs)
    # The right-hand clock is not part of the result. Selecting the original
    # columns also drops the ``_right`` suffix polars adds when several
    # as-of joins run on the same frame.
    original = list(base.columns)
    return joined.sort(order).select([order, *original, *feature_cols]).drop(order)


def daily_event_ms(day_ordinal_utc_midnight_ms: int) -> int:
    """Event time for a daily aggregate of the UTC day that starts at this midnight."""
    return int(day_ordinal_utc_midnight_ms)
