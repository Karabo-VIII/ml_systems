"""On-disk layout. Silver and gold names use the data-end date, never ``--start``."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from pipeline.constants import TIME_WIDTH_MS


def crypto_root() -> Path:
    # pipeline/paths.py -> src -> crypto
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return crypto_root() / "data" / "pipeline"


def stamp_from_ts_ms(ts_ms: int) -> str:
    """YYYYMMDD of the UTC calendar date of this close. Not the run's start year."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y%m%d")


def date_from_ts_ms(ts_ms: int) -> date:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).date()


def trades_path(root: Path, asset: str, day: date) -> Path:
    return root / "bronze" / "trades" / "spot" / asset / f"{day.isoformat()}.parquet"


def klines_1m_path(root: Path, asset: str, day: date) -> Path:
    return root / "bronze" / "klines" / "spot_1m" / asset / f"{day.isoformat()}.parquet"


def funding_path(root: Path, asset: str, year: int, month: int) -> Path:
    return root / "bronze" / "funding" / "um" / asset / f"{year:04d}-{month:02d}.parquet"


def metrics_path(root: Path, asset: str, day: date) -> Path:
    return root / "bronze" / "metrics" / "um" / asset / f"{day.isoformat()}.parquet"


def perp_1d_path(root: Path, asset: str, day: date) -> Path:
    return root / "bronze" / "klines" / "um_1d" / asset / f"{day.isoformat()}.parquet"


def manifest_path(root: Path, source: str, asset: str) -> Path:
    return root / "bronze" / "_manifest" / source / f"{asset}.json"


def bars_path(root: Path, bar_type: str, asset: str, ts_ms_max: int) -> Path:
    stamp = stamp_from_ts_ms(ts_ms_max)
    return root / "silver" / "bars" / bar_type / f"{asset}_{stamp}.parquet"


def features_path(root: Path, bar_type: str, asset: str, ts_ms_max: int) -> Path:
    stamp = stamp_from_ts_ms(ts_ms_max)
    return root / "silver" / "features" / bar_type / f"{asset}_{stamp}.parquet"


def gold_path(root: Path, bar_type: str, asset: str, ts_ms_max: int) -> Path:
    stamp = stamp_from_ts_ms(ts_ms_max)
    return root / "gold" / bar_type / f"{asset}_{stamp}.parquet"


def gold_manifest_path(gold_file: Path) -> Path:
    return gold_file.with_suffix(".manifest.json")


def gap_path(root: Path, asset: str, ts_ms_max: int) -> Path:
    stamp = stamp_from_ts_ms(ts_ms_max)
    return root / "silver" / "bars" / "_gaps" / f"{asset}_{stamp}.json"


def is_time_bar(bar_type: str) -> bool:
    return bar_type in TIME_WIDTH_MS
