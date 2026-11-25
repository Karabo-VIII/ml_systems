"""Gold dataset: targets, frozen splits, embargo, manifest."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import polars as pl

from pipeline.constants import CLIP_H1, CLIP_OTHER, CONTRACT_VERSION
from pipeline.config import Splits, membership_flags


def add_targets(
    df: pl.DataFrame,
    horizons: tuple[int, ...] = (1, 4, 16, 64),
    *,
    clipped: bool = False,
    voladj: bool = False,
    width_ms: int | None = None,
) -> pl.DataFrame:
    """Uncapped forward returns. Nothing is filled with 0.

    With ``width_ms`` (crypto time bars) the horizon is clock time: the target
    of the bar closing at ``t`` uses the last close at or before ``t + h * width``
    (the last trade price then; a minute without trades has no bar and leaves
    the price where it was), and is null when ``t + h * width`` is past the last
    bar of the frame. Without it (event bars, FX) the target is ``h`` bars ahead
    and the last ``h`` rows stay null.

    ``voladj`` divides by future realized volatility over the next ``h`` bars.
    It is a label, never a feature.
    """
    if df.is_empty():
        return df
    df = df.sort("ts_ms")
    last_ts = int(df["ts_ms"].max())
    for h in horizons:
        if width_ms:
            ahead = int(h) * int(width_ms)
            probe = df.select((pl.col("ts_ms") + ahead).alias("_tq"))
            ref = df.select(pl.col("ts_ms").alias("_tf"), pl.col("close").alias("_cf"))
            fwd = probe.join_asof(ref, left_on="_tq", right_on="_tf", strategy="backward")
            fwd_close = fwd.select(
                pl.when(pl.col("_tq") <= last_ts).then(pl.col("_cf")).otherwise(None).alias("_cf")
            )["_cf"]
            raw = (pl.lit(fwd_close) - pl.col("close")) / pl.col("close")
        else:
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


# Series families that are attached to a bar by an as-of join. The key is the
# ``lags.families`` entry that sets the publication lag; the tuple lists the
# column names or prefixes that belong to the family.
INPUT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("funding", ("funding_rate",)),
    ("daily_panel", ("wh_", "rv_", "hbr_")),
    ("funding_daily", ("funding_day_",)),
    ("metrics_daily", ("s3_",)),
    ("basis", ("bs_",)),
    ("defillama", ("stbl_",)),
    ("deribit", ("dv_",)),
    ("etf", ("etf_",)),
    ("wikipedia", ("soc_",)),
    ("open_interest", ("oi_value", "norm_oi_")),
    ("policy_rate", ("carry_pct",)),
)

# Columns that are empty because the source did not publish yet (not a bug).
_SOURCE_FAMILIES = {"deribit", "etf", "wikipedia", "defillama", "metrics_daily", "basis", "open_interest", "funding", "funding_daily", "policy_rate"}


def input_families(columns: list[str], lag_table: dict[str, int]) -> tuple[list[dict], dict[str, int]]:
    """Families present in a frame, with their columns and the lag that was applied."""
    inputs: list[dict] = []
    lags: dict[str, int] = {}
    for family, patterns in INPUT_FAMILIES:
        cols = [c for c in columns if any(c == p or (p.endswith("_") and c.startswith(p)) for p in patterns)]
        if not cols:
            continue
        lag = int(lag_table.get(family, 0))
        inputs.append({"family": family, "columns": cols, "lag_ms": lag})
        lags[family] = lag
    return inputs, lags


def null_by_design(df: pl.DataFrame, feature_cols: list[str], families: list[dict]) -> dict[str, str]:
    """Columns that are 100% null in this partition, each with the reason."""
    owner = {c: f["family"] for f in families for c in f["columns"]}
    out: dict[str, str] = {}
    if df.height == 0:
        return out
    for c in feature_cols:
        if c not in df.columns or df[c].null_count() != df.height:
            continue
        family = owner.get(c)
        if family in _SOURCE_FAMILIES:
            out[c] = f"source '{family}' has no data knowable in this period"
        elif c.startswith(("norm_", "hurst_", "xd_")):
            out[c] = "rolling-window warmup (or too few peers for xd_*) covers the partition"
        else:
            out[c] = "all null; not explained by a known source or warmup"
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
    splits: object | None = None,
) -> dict:
    from pipeline.io import params_hash

    ts_min = int(df["ts_ms"].min()) if df.height else None
    ts_max = int(df["ts_ms"].max()) if df.height else None
    boundaries = None
    if splits is not None:
        boundaries = {
            k: getattr(splits, k).isoformat()
            for k in ("train_end", "val_end", "oos_end", "unseen_start")
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "schema_name": "dataset_v1",
        "producer": "pipeline",
        "asset": asset,
        "bar_type": bar_type,
        "params_hash": params_hash(params),
        "params": params,
        "ts_ms_min": ts_min,
        "ts_ms_max": ts_max,
        "n_rows": df.height,
        "n_feature_cols": len(feature_cols),
        "n_null_by_col": null_fraction(df, feature_cols),
        "null_by_design": null_by_design(df, feature_cols, inputs),
        "inputs": inputs,
        "lags_applied": lags,
        "split_boundaries": boundaries,
        "purge_bars": embargo_bars,
        "embargo_bars": embargo_bars,
        "built_at_ms": built_at_ms,
    }


def utc_now_ms() -> int:
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def symlog_is_finite(x: float) -> bool:
    return math.isfinite(x)

"""Read gold datasets produced by this pipeline.

:func:`load_gold` is the reader. It does not import the ``ml_systems_v1`` snapshot.
"""


from pathlib import Path

import polars as pl

from pipeline.partitions import list_partition_paths, load_partitions
from pipeline.paths import default_data_root
from pipeline.symbols import canonical_instrument, canonical_symbol


def gold_dir(root: Path, bar_type: str) -> Path:
    return root / "gold" / bar_type


def latest_gold_path(root: Path, asset: str, bar_type: str) -> Path | None:
    """Last monthly (or legacy) partition. Prefer :func:`load_gold` for the full series."""
    asset = canonical_instrument(asset)
    files = list_partition_paths(gold_dir(root, bar_type), asset)
    return files[-1] if files else None


def load_gold(
    asset: str,
    bar_type: str = "time_1h",
    *,
    root: Path | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Load every gold partition for one asset and bar type, in time order.

    Raises FileNotFoundError when nothing has been built. Does not fall back
    to v1 chimera files.
    """
    root = root or default_data_root()
    asset = canonical_instrument(asset)
    df = load_partitions(gold_dir(root, bar_type), asset, columns=columns)
    if df is None:
        raise FileNotFoundError(
            f"no v2 gold for {asset} {bar_type} under {root / 'gold'}. "
            f"Build it with: python -m pipeline.cli run --assets {asset} "
            f"--bar-types {bar_type}"
        )
    return df


def load_gold_universe(
    assets: list[str],
    bar_type: str = "time_1h",
    *,
    root: Path | None = None,
    columns: list[str] | None = None,
) -> dict[str, pl.DataFrame]:
    out: dict[str, pl.DataFrame] = {}
    for asset in assets:
        try:
            out[canonical_symbol(asset)] = load_gold(asset, bar_type, root=root, columns=columns)
        except FileNotFoundError as exc:
            print(f"[gold] {exc}")
    return out
