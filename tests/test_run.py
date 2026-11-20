"""End-to-end offline run: bronze on disk, one orchestrator, gates green."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl

from pipeline.cli import main
from pipeline.io import file_sha256
from pipeline.orchestrator import RunConfig, run
from fixtures import small_windows, write_bronze_day, write_funding


def test_cli_dry_run_has_date_flags(capsys):
    code = main([
        "run", "--start", "2024-06-01", "--end", "2024-06-02",
        "--assets", "BTC", "--dry-run", "--stages", "bars",
        "--bar-types", "time_1m,time_1h",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "BTCUSDT" in out
    assert "2024-06-01" in out


def _bursty_day(seed: int, price0: float, qty: float):
    rng = np.random.default_rng(seed)
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)
    ts_ms, price, q, maker, ids = [], [], [], [], []
    px = price0
    tid = 1
    for minute in range(24 * 60):
        n = int(rng.integers(2, 6))
        for _ in range(n):
            sec = int(rng.integers(0, 59))
            ms = int(rng.integers(1, 900))
            stamp = start + timedelta(minutes=minute, seconds=sec, milliseconds=ms)
            px *= 1.0 + float(rng.normal(0, 0.0007))
            size = float(qty * rng.uniform(0.3, 2.5))
            if rng.random() < 0.01:
                size *= 30
            ts_ms.append(int(stamp.timestamp() * 1000))
            price.append(px)
            q.append(size)
            maker.append(bool(rng.random() < 0.45))
            ids.append(tid)
            tid += 1
    return pl.DataFrame({
        "agg_trade_id": ids,
        "ts_us": [t * 1000 for t in ts_ms],
        "ts_ms": ts_ms,
        "price": price,
        "qty": q,
        "is_buyer_maker": maker,
    }).sort(["ts_us", "agg_trade_id"])


def test_offline_run_reconciles_and_is_idempotent(tmp_path):
    day = date(2024, 6, 1)
    btc = _bursty_day(11, 100.0, 0.2)
    eth = _bursty_day(12, 10.0, 2.0)
    write_bronze_day(tmp_path, "BTCUSDT", day, btc)
    write_bronze_day(tmp_path, "ETHUSDT", day, eth)
    # Funding inside the day, several distinct rates.
    base_ms = int(datetime(2024, 6, 1, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [(base_ms + i * 3_600_000, 0.0001 * (i - 6)) for i in range(8)]
    write_funding(tmp_path, "BTCUSDT", rows)
    write_funding(tmp_path, "ETHUSDT", [(t, r * 0.5) for t, r in rows])

    windows = small_windows()
    cfg = RunConfig(
        start=day,
        end=day,
        assets=["BTCUSDT", "ETHUSDT"],
        stages=["bars", "features", "dataset", "validate"],
        bar_types=["time_1m", "time_1h", "time_1d", "dollar"],
        data_root=tmp_path,
        force=True,
        windows=windows,
        threshold_overrides={"BTCUSDT": 50.0, "ETHUSDT": 30.0},
    )
    code = run(cfg)
    assert code == 0, "validate must pass on the synthetic book"
    gold = list((tmp_path / "gold").rglob("*.parquet"))
    assert gold
    # Monthly partitions are named ASSET_YYYY-MM.parquet, never a start year of 2020.
    names = " ".join(p.name for p in gold)
    assert "2020" not in names
    assert "2024-06" in names
    first = {p: file_sha256(p) for p in gold}
    code2 = run(cfg)
    assert code2 == 0
    for path, digest in first.items():
        assert file_sha256(path) == digest


def test_cli_status(tmp_path, capsys):
    code = main(["status", "--data-root", str(tmp_path)])
    assert code == 0
