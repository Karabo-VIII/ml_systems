"""Monthly silver/gold partitions, skip-fresh coverage, and partition readers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from pipeline.io import file_sha256, params_hash, read_parquet_metadata
from pipeline.paths import (
    bars_month_path,
    features_month_path,
    gold_month_path,
    month_partition_name,
)


def sidecar_ok(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    # A leftover temp file is never a finished partition.
    if ".tmp." in path.name:
        return False
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        return False
    try:
        expected = sidecar.read_text().strip()
        return file_sha256(path) == expected
    except OSError:
        return False


def partition_fresh(path: Path, params: dict, *, force: bool = False) -> bool:
    """True when the file is complete, hashed, and was built with the same params."""
    if force:
        return False
    if not sidecar_ok(path):
        return False
    try:
        meta = read_parquet_metadata(path)
    except Exception:
        return False
    return meta.get("params_hash") == params_hash(params)


def partition_params(
    *,
    asset: str,
    bar_type: str,
    start: date,
    end: date,
    extra: dict | None = None,
) -> dict:
    payload = {
        "asset": asset,
        "bar_type": bar_type,
        "range_start": start.isoformat(),
        "range_end": end.isoformat(),
    }
    if extra:
        payload.update(extra)
    return payload


def list_partition_paths(directory: Path, asset: str) -> list[Path]:
    """Monthly ``ASSET_YYYY-MM.parquet`` plus any legacy ``ASSET_YYYYMMDD.parquet``."""
    if not directory.exists():
        return []
    files = [p for p in directory.glob(f"{asset}_*.parquet") if p.is_file() and ".tmp." not in p.name]
    return sorted(files)


def load_partitions(
    directory: Path,
    asset: str,
    *,
    columns: list[str] | None = None,
) -> pl.DataFrame | None:
    files = list_partition_paths(directory, asset)
    if not files:
        return None
    frames = []
    for path in files:
        df = pl.read_parquet(path, columns=columns) if columns else pl.read_parquet(path)
        frames.append(df)
    out = pl.concat(frames, how="diagonal_relaxed").sort("ts_ms")
    if "ts_ms" in out.columns:
        out = out.unique(subset=["ts_ms"], keep="last", maintain_order=True)
    return out


def expected_months_present(
    directory: Path,
    asset: str,
    start: date,
    end: date,
    params_for_month,
) -> tuple[int, int, list[str]]:
    """Return (present, expected, missing stamps) for a date range."""
    from pipeline.calendar import clip_month, iter_months, month_stamp

    missing = []
    present = 0
    expected = 0
    for y, m in iter_months(start, end):
        expected += 1
        first, last = clip_month(y, m, start, end)
        path = directory / month_partition_name(asset, y, m)
        params = params_for_month(first, last)
        if partition_fresh(path, params):
            present += 1
        else:
            missing.append(month_stamp(y, m))
    return present, expected, missing
