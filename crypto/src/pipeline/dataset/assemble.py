"""Gold dataset: targets, frozen splits, embargo, manifest."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import polars as pl

from pipeline.constants import CLIP_H1, CLIP_OTHER
from pipeline.contract import Splits, membership_flags


def add_targets(
    df: pl.DataFrame,
    horizons: tuple[int, ...] = (1, 4, 16, 64),
    *,
    clipped: bool = False,
    voladj: bool = False,
) -> pl.DataFrame:
    """Uncapped forward returns. The last ``h`` rows stay null. Nothing is filled with 0.

    ``voladj`` divides by future realized volatility. It is a label, never a feature.
    """
    if df.is_empty():
        return df
    df = df.sort("ts_ms")
    for h in horizons:
        raw = (pl.col("close").shift(-h) - pl.col("close")) / pl.col("close")
        df = df.with_columns(raw.alias(f"target_return_{h}"))
        if clipped:
            cap = CLIP_H1 if h == 1 else CLIP_OTHER
            df = df.with_columns(
                pl.col(f"target_return_{h}").clip(-cap, cap).alias(f"target_return_{h}_clipped")
            )
        if voladj:
            ret = pl.col("close").pct_change()
            fut = ret.rolling_std(window_size=max(h, 2), min_samples=max(h, 2)).shift(-h)
            ratio = pl.col(f"target_return_{h}") / pl.when(fut.is_null()).then(None).when(fut < 1e-3).then(pl.lit(1e-3)).otherwise(fut)
            sym = ratio.sign() * (ratio.abs() + 1.0).log()
            df = df.with_columns([
                sym.alias(f"target_voladj_{h}"),
                fut.alias(f"target_vol_{h}"),
            ])
    return df


def _utc_date(ts_ms: pl.Expr) -> pl.Expr:
    return pl.from_epoch(ts_ms, time_unit="ms").dt.date()


def assign_split(df: pl.DataFrame, splits: Splits) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("split"))
    d = _utc_date(pl.col("ts_ms"))
    label = (
        pl.when(d < splits.train_end).then(pl.lit("train"))
        .when(d < splits.val_end).then(pl.lit("val"))
        .when(d < splits.oos_end).then(pl.lit("oos"))
        .otherwise(pl.lit("unseen"))
    )
    return df.with_columns(label.alias("split"))


def apply_embargo(df: pl.DataFrame, embargo_bars: int) -> pl.DataFrame:
    """Drop ``embargo_bars`` at the tail of the earlier segment and the head of the later one.

    The embargo is not shrunk to fit a short segment. Rows that fall inside it
    are labelled ``embargo`` and belong to no training segment.
    """
    if df.is_empty() or embargo_bars <= 0 or "split" not in df.columns:
        return df
    df = df.sort("ts_ms")
    labels = df["split"].to_list()
    order = ["train", "val", "oos", "unseen"]
    segments: list[tuple[str, int, int]] = []
    i = 0
    n = len(labels)
    while i < n:
        j = i + 1
        while j < n and labels[j] == labels[i]:
            j += 1
        segments.append((str(labels[i]), i, j))
        i = j
    flagged = [False] * n
    for (lab_a, a0, a1), (lab_b, b0, b1) in zip(segments, segments[1:]):
        if lab_a not in order or lab_b not in order:
            continue
        if order.index(lab_b) <= order.index(lab_a):
            continue
        for k in range(max(a0, a1 - embargo_bars), a1):
            flagged[k] = True
        for k in range(b0, min(b1, b0 + embargo_bars)):
            flagged[k] = True
    return _apply_flag(df, flagged)


def _apply_flag(df: pl.DataFrame, flagged: list[bool]) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.Series("flag", flagged))
        .then(pl.lit("embargo"))
        .otherwise(pl.col("split"))
        .alias("split")
    )


def add_universe_columns(df: pl.DataFrame, asset: str, listing: date | None) -> pl.DataFrame:
    flags = membership_flags(asset)
    return df.with_columns([
        pl.lit(bool(flags["is_u10"])).alias("is_u10"),
        pl.lit(bool(flags["is_u50"])).alias("is_u50"),
        pl.lit(bool(flags["is_u100"])).alias("is_u100"),
        pl.lit(flags["asset_dna"]).cast(pl.Utf8).alias("asset_dna"),
        pl.lit(listing.isoformat() if listing else None).alias("listing_date"),
    ])


def null_fraction(df: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    out = {}
    n = max(df.height, 1)
    for c in cols:
        if c not in df.columns or df.height == 0:
            continue
        out[c] = float(df[c].null_count()) / n
    return out


def manifest_payload(
    *,
    asset: str,
    bar_type: str,
    df: pl.DataFrame,
    feature_cols: list[str],
    params: dict,
    embargo_bars: int,
    inputs: list[dict],
    lags: dict[str, int],
    built_at_ms: int,
) -> dict:
    ts_min = int(df["ts_ms"].min()) if df.height else None
    ts_max = int(df["ts_ms"].max()) if df.height else None
    return {
        "contract_version": "2.0.0",
        "schema_name": "dataset_v1",
        "producer": "pipeline",
        "asset": asset,
        "bar_type": bar_type,
        "params_hash": params.get("params_hash"),
        "params": params,
        "ts_ms_min": ts_min,
        "ts_ms_max": ts_max,
        "n_rows": df.height,
        "n_feature_cols": len(feature_cols),
        "n_null_by_col": null_fraction(df, feature_cols),
        "inputs": inputs,
        "lags_applied": lags,
        "split_boundaries": params.get("splits"),
        "purge_bars": embargo_bars,
        "embargo_bars": embargo_bars,
        "built_at_ms": built_at_ms,
    }


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def symlog_is_finite(x: float) -> bool:
    return math.isfinite(x)
