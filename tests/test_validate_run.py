"""Unit tests for scripts/validate_run.py helpers and severity rules."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from pipeline import validate_run as vr  # noqa: E402
from pipeline.io import atomic_write_parquet  # noqa: E402


def _minute_bars(start: datetime, n: int, skip: set[int]) -> pl.DataFrame:
    """Right-closed 1m bars: skip is a set of close timestamps (ms) to omit."""
    start_ms = int(start.timestamp() * 1000)
    rows = []
    for i in range(1, n + 1):
        ts = start_ms + i * 60_000
        if ts in skip:
            continue
        rows.append({"ts_ms": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "n_trades": 1})
    return pl.DataFrame(rows)


def test_isolated_fx_quiet_minutes_are_warn_not_fail():
    # Monday 10 Nov 2026 12:00–13:00 UTC, two isolated missing closes.
    start = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    skip = {
        int(datetime(2026, 11, 10, 12, 15, tzinfo=timezone.utc).timestamp() * 1000),
        int(datetime(2026, 11, 10, 12, 40, tzinfo=timezone.utc).timestamp() * 1000),
    }
    report = vr.Report()
    vr.check_gaps(report, {"EURUSD": {"time_1m": _minute_bars(start, 60, skip)}})
    assert len(report.checks) == 1
    assert report.checks[0].severity == "WARN"
    assert "2 missing minutes" in report.checks[0].detail
    assert "longest hole=1 min" in report.checks[0].detail
    assert report.code == 1


def test_long_fx_weekday_hole_is_fail():
    start = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    # 16 consecutive missing minutes mid-session is a fetch-sized hole.
    skip = {
        int(datetime(2026, 11, 10, 12, 10, tzinfo=timezone.utc).timestamp() * 1000) + i * 60_000
        for i in range(16)
    }
    report = vr.Report()
    vr.check_gaps(report, {"EURUSD": {"time_1m": _minute_bars(start, 60, skip)}})
    assert report.checks[0].severity == "FAIL"
    assert "longest hole=16 min" in report.checks[0].detail
    assert report.code == 2


def test_fx_weekend_gap_is_pass():
    # Two Saturday bars an hour apart: every missing close is inside the NY weekend.
    start = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)
    rows = []
    for minutes in (1, 61):
        ts = int(start.timestamp() * 1000) + minutes * 60_000
        rows.append({
            "ts_ms": ts,
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0, "n_trades": 1,
        })
    report = vr.Report()
    vr.check_gaps(report, {"EURUSD": {"time_1m": pl.DataFrame(rows)}})
    assert report.checks[0].severity == "PASS", report.checks[0].detail
    assert report.code == 0


def test_fx_null_volume_and_short_lookbacks_are_allowed(tmp_path):
    folder = tmp_path / "gold" / "time_1d"
    folder.mkdir(parents=True)
    path = folder / "EURUSD_20261116.parquet"
    n = 6
    frame = pl.DataFrame({
        "asset": ["EURUSD"] * n,
        "bar_type": ["time_1d"] * n,
        "ts_ms": list(range(n)),
        "close": [1.1] * n,
        "volume_usd": [None] * n,
        "buy_volume": [None] * n,
        "sell_volume": [None] * n,
        "buy_usd": [None] * n,
        "sell_usd": [None] * n,
        "signed_usd": [None] * n,
        "base_strength_ret": [None] * n,
        "ret_10": [None] * n,
        "ret_50": [None] * n,
        "rv_20": [None] * n,
        "spread_z": [None] * n,
    })
    atomic_write_parquet(frame, path, schema_name="dataset_v1", params={"v": 1}, sort_cols=["ts_ms"])
    report = vr.Report()
    vr.check_dead_columns(report, {"time_1d": {"EURUSD": path}})
    assert len(report.checks) == 1
    assert report.checks[0].severity == "PASS", report.checks[0].detail
    assert "no unexpected dead series" in report.checks[0].detail
