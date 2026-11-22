"""Interrupt mid-stage and resume: output matches a clean run."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl
import pytest

from pipeline.cli import main
from pipeline.io import file_sha256
from pipeline.orchestrator import RunConfig, run
from fixtures import small_windows, write_bronze_day


def _day(seed: int, price0: float, day: date) -> pl.DataFrame:
    rng_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    # A few trades every minute is enough to close time and dollar bars.
    ts, price, qty, maker, ids = [], [], [], [], []
    px = price0
    tid = 1
    import numpy as np
    rng = np.random.default_rng(seed)
    for minute in range(24 * 60):
        for k in range(2):
            stamp = rng_start + timedelta(minutes=minute, seconds=10 * k + 1)
            px *= 1.0 + float(rng.normal(0, 0.0005))
            ts.append(int(stamp.timestamp() * 1000))
            price.append(px)
            qty.append(0.5)
            maker.append(k == 0)
            ids.append(tid)
            tid += 1
    return pl.DataFrame({
        "agg_trade_id": ids,
        "ts_us": [t * 1000 for t in ts],
        "ts_ms": ts,
        "price": price,
        "qty": qty,
        "is_buyer_maker": maker,
    }).sort(["ts_us", "agg_trade_id"])


def test_interrupt_mid_bars_then_resume_matches_clean_run(tmp_path, monkeypatch):
    """A KeyboardInterrupt after the first monthly write is not treated as done.

    Re-running the same config finishes the remaining partitions and matches a
    clean run byte-for-byte.
    """
    from pipeline import chunked

    d1 = date(2024, 5, 31)
    d2 = date(2024, 6, 1)
    write_bronze_day(tmp_path, "BTCUSDT", d1, _day(1, 100.0, d1))
    write_bronze_day(tmp_path, "BTCUSDT", d2, _day(2, 101.0, d2))

    windows = small_windows()
    cfg_kwargs = dict(
        start=d1,
        end=d2,
        assets=["BTCUSDT"],
        stages=["bars", "features", "dataset"],
        bar_types=["time_1m", "time_1h", "time_1d"],
        data_root=tmp_path,
        windows=windows,
        threshold_overrides={"BTCUSDT": 50.0},
    )

    clean_root = tmp_path / "clean"
    # Need bronze under clean_root too.
    for day, seed, px in ((d1, 1, 100.0), (d2, 2, 101.0)):
        write_bronze_day(clean_root, "BTCUSDT", day, _day(seed, px, day))
    code = run(RunConfig(**{**cfg_kwargs, "data_root": clean_root, "force": True}))
    assert code == 0
    clean_gold = {p.relative_to(clean_root): file_sha256(p) for p in (clean_root / "gold").rglob("*.parquet")}
    assert clean_gold

    original = chunked.write_bars_month
    calls = {"n": 0}

    def boom(cfg, asset, year, month, acc, session=None, **kwargs):
        result = original(cfg, asset, year, month, acc, session, **kwargs)
        calls["n"] += 1
        if calls["n"] >= 1:
            raise KeyboardInterrupt()
        return result

    import pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "write_bars_month", boom)
    monkeypatch.setattr(chunked, "write_bars_month", boom)

    interrupted = tmp_path / "interrupted"
    for day, seed, px in ((d1, 1, 100.0), (d2, 2, 101.0)):
        write_bronze_day(interrupted, "BTCUSDT", day, _day(seed, px, day))
    with pytest.raises(KeyboardInterrupt):
        run(RunConfig(**{**cfg_kwargs, "data_root": interrupted, "force": False}))

    # First month should exist (atomic write finished); second may be missing.
    written = list((interrupted / "silver" / "bars").rglob("*.parquet"))
    assert written, "the first completed month must have been written before the interrupt"

    # Resume with the original writer.
    monkeypatch.setattr(orch, "write_bars_month", original)
    monkeypatch.setattr(chunked, "write_bars_month", original)
    code = run(RunConfig(**{**cfg_kwargs, "data_root": interrupted, "force": False}))
    assert code == 0
    resumed = {p.relative_to(interrupted): file_sha256(p) for p in (interrupted / "gold").rglob("*.parquet")}
    assert set(resumed) == set(clean_gold)
    for rel, digest in clean_gold.items():
        assert resumed[rel] == digest


def test_status_reports_range_completeness(tmp_path, capsys):
    code = main([
        "status", "--data-root", str(tmp_path),
        "--start", "2024-06-01", "--end", "2024-06-02",
        "--assets", "BTC", "--stages", "fetch,bars",
        "--bar-types", "time_1m,time_1h",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "BTCUSDT" in out
    assert "fetch" in out
    assert "bars" in out
