"""Inspect summaries and scoped independent validation."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from pipeline.inspect import candle_x, collect_views, format_inspect
from pipeline.io import atomic_write_parquet
from pipeline.validate_run import run_validation


def _bars(start: datetime, n: int, width_s: int, asset: str, bar_type: str) -> pl.DataFrame:
    start_ms = int(start.timestamp() * 1000)
    rows = []
    for i in range(1, n + 1):
        ts = start_ms + i * width_s * 1000
        rows.append({
            "asset": asset, "bar_type": bar_type, "ts_open_ms": ts - width_s * 1000,
            "ts_ms": ts, "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0 + i * 0.01,
            "volume": 2.0, "volume_usd": 200.0, "n_trades": 4,
        })
    return pl.DataFrame(rows)


def _write(root: Path, layer: str, bar_type: str, asset: str, df: pl.DataFrame) -> None:
    folder = root / layer / bar_type if layer == "gold" else root / "silver" / "bars" / bar_type
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{asset}_2020-03.parquet"
    atomic_write_parquet(df, path, schema_name="bars_v1", params={"asset": asset})


def test_inspect_and_scoped_validate(tmp_path):
    start = datetime(2020, 3, 11, tzinfo=timezone.utc)
    btc_1m = _bars(start, 3 * 1440, 60, "BTCUSDT", "time_1m")
    btc_1d = _bars(start, 3, 86400, "BTCUSDT", "time_1d")
    # Force volume agreement: 1d volume is the sum of that day's 1m volume.
    # close_of 1d bars: day 1 closes at 2020-03-12 00:00, etc.
    # Our _bars from start 00:00: first 1d close is 2020-03-12 00:00 (i=1).
    # 1m first close is 00:01. Sum of all 1m volume should equal sum of 1d if we set it.
    day_vol = float(btc_1m["volume"].sum() / 3)
    btc_1d = btc_1d.with_columns(pl.lit(day_vol).alias("volume"))
    # last 1d close equals last 1m close
    btc_1d = btc_1d.with_columns(
        pl.when(pl.col("ts_ms") == btc_1d["ts_ms"].max()).then(pl.lit(float(btc_1m["close"][-1]))).otherwise(pl.col("close")).alias("close")
    )
    eth_1m = _bars(start, 3 * 1440, 60, "ETHUSDT", "time_1m")
    _write(tmp_path, "silver", "time_1m", "BTCUSDT", btc_1m)
    _write(tmp_path, "silver", "time_1d", "BTCUSDT", btc_1d)
    _write(tmp_path, "gold", "time_1m", "BTCUSDT", btc_1m)
    _write(tmp_path, "gold", "time_1d", "BTCUSDT", btc_1d)
    _write(tmp_path, "silver", "time_1m", "ETHUSDT", eth_1m)
    _write(tmp_path, "gold", "time_1m", "ETHUSDT", eth_1m)

    views = collect_views(tmp_path, "BTCUSDT", date(2020, 3, 11), date(2020, 3, 13))
    assert set(views) >= {"time_1m", "time_1d"}
    assert views["time_1m"].height == 3 * 1440
    text = format_inspect("BTCUSDT", date(2020, 3, 11), date(2020, 3, 13), tmp_path, views)
    assert "rows=4320" in text
    assert "match=yes" in text
    assert "1m_vs_1d=yes" in text

    report = run_validation(
        tmp_path,
        start=date(2020, 3, 11),
        end=date(2020, 3, 13),
        assets=["BTCUSDT"],
        market="crypto",
        skip_network=True,
    )
    names = {c.name for c in report.checks}
    assert not any(n.endswith("ETHUSDT") or ":ETHUSDT" in n or n.endswith("ETHUSDT") for n in names)
    # Scoped to BTC; ETH must not appear.
    assert all("ETHUSDT" not in (c.name + c.detail) for c in report.checks)


def test_candle_x_uses_interval_midpoint():
    day = 24 * 60 * 60 * 1000
    open_ms = int(datetime(2020, 3, 12, tzinfo=timezone.utc).timestamp() * 1000)
    close_ms = open_ms + day
    mid, width = candle_x(open_ms, close_ms)
    assert width == day
    assert mid == open_ms + day // 2
    assert mid == int(datetime(2020, 3, 12, 12, tzinfo=timezone.utc).timestamp() * 1000)

    hour = 60 * 60 * 1000
    h_open = int(datetime(2020, 3, 12, 14, tzinfo=timezone.utc).timestamp() * 1000)
    mid_h, width_h = candle_x(h_open, h_open + hour)
    assert width_h == hour
    assert mid_h == int(datetime(2020, 3, 12, 14, 30, tzinfo=timezone.utc).timestamp() * 1000)
