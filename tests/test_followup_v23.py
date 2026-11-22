"""Follow-up regressions (contract 2.3.0): fetch streaming, empty-day cache, whale size, embargo."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pipeline.config import WhaleSpec, embargo_bars_for, whale_spec
from pipeline.constants import CONTRACT_VERSION
from pipeline.features import (
    quantile_from_hists,
    summarize_day,
    trade_notional_hist,
    whale_threshold,
)
from pipeline.fetch_crypto import (
    _finalize_aggtrades,
    fetch_trades,
    iter_aggtrades,
    parse_aggtrades,
)
from pipeline.orchestrator import RunConfig, run
from fixtures import small_windows, write_bronze_day
from test_resume import _day


def _ms(y, m, d, h=0, mi=0, s=0) -> int:
    return int(datetime(y, m, d, h, mi, s, tzinfo=timezone.utc).timestamp() * 1000)


def _zip_csv(rows: bytes, name: str = "x.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, rows)
    return buf.getvalue()


def _trade_csv(rows: list[tuple[int, float, float, int]]) -> bytes:
    # agg_trade_id, price, qty, first, last, timestamp_us, is_buyer_maker, is_best_match
    lines = []
    for tid, price, qty, ts_ms in rows:
        lines.append(f"{tid},{price},{qty},{tid},{tid},{ts_ms * 1000},false,true")
    return ("\n".join(lines) + "\n").encode()


# --- fetch -------------------------------------------------------------------

def test_chunked_aggtrades_match_one_shot(tmp_path):
    rows = []
    tid = 1
    for d in (1, 2, 3):
        base = _ms(2024, 5, d)
        for i in range(800):
            rows.append((tid, 10.0 + (i % 5), 0.5, base + i * 1000))
            tid += 1
    raw = _zip_csv(_trade_csv(rows))
    one = parse_aggtrades(raw)
    zpath = tmp_path / "m.zip"
    zpath.write_bytes(raw)
    parts = list(iter_aggtrades(zpath, chunk_rows=150))
    assert len(parts) > 1
    got = _finalize_aggtrades(pl.concat(parts, how="diagonal_relaxed"))
    assert got.height == one.height == 2400
    assert got["agg_trade_id"].to_list() == one["agg_trade_id"].to_list()
    assert got["ts_us"].to_list() == one["ts_us"].to_list()
    assert got["price"].to_list() == one["price"].to_list()


def test_empty_day_in_monthly_zip_is_recorded_and_not_refetched(tmp_path, monkeypatch):
    # Trades on the 1st and 3rd only; the 2nd is legitimately empty.
    rows = []
    tid = 1
    for d, n in ((1, 20), (3, 15)):
        base = _ms(2024, 5, d, 12)
        for i in range(n):
            rows.append((tid, 100.0, 1.0, base + i * 1000))
            tid += 1
    blob = _zip_csv(_trade_csv(rows))
    calls = {"n": 0}

    def to_path(url, dest, timeout=120):
        calls["n"] += 1
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(blob)
        return 200, None

    def no_daily(url, timeout=120):
        raise AssertionError(f"daily fallback should not run: {url}")

    import pipeline.fetch_crypto as fc
    monkeypatch.setattr(fc, "http_get_to_path", to_path)
    monkeypatch.setattr(fc, "http_get", no_daily)

    start, end = date(2024, 5, 1), date(2024, 5, 3)
    r1 = fetch_trades(tmp_path, "BTCUSDT", start, end)
    statuses = {rec.key: rec.status for rec in r1.records}
    assert statuses["2024-05-01"] == "ok"
    assert statuses["2024-05-02"] == "empty"
    assert statuses["2024-05-03"] == "ok"
    assert not (tmp_path / "bronze" / "trades" / "spot" / "BTCUSDT" / "2024-05-02.parquet").exists()
    assert calls["n"] == 1

    r2 = fetch_trades(tmp_path, "BTCUSDT", start, end)
    assert {rec.key: rec.status for rec in r2.records} == statuses
    assert all(rec.error == "cached" for rec in r2.records)
    assert calls["n"] == 1  # monthly zip was not downloaded again

    fetch_trades(tmp_path, "BTCUSDT", start, end, force=True)
    assert calls["n"] == 2


# --- whale -------------------------------------------------------------------

def test_trailing_whale_quantile_is_causal_and_bootstraps():
    spec = WhaleSpec(bootstrap=1_000.0, quantile=0.9, window_days=3, version="t")
    assert whale_threshold([], spec) == 1_000.0
    small = trade_notional_hist(np.full(100, 10.0))
    t2 = whale_threshold([small], spec)
    assert 5.0 < t2 < 20.0
    huge = trade_notional_hist(np.full(100, 1_000_000.0))
    # Day D's own (huge) prints must not move day D's size.
    assert whale_threshold([small], spec) == t2
    later = whale_threshold([small, huge], spec)
    assert later > t2


def test_bnb_2020_scale_prints_are_not_all_zero_after_the_first_day():
    """A $250k static floor made BNB 2020 `wh_*` a constant 0; the adaptive size must not."""
    spec = whale_spec("BNBUSDT")
    assert spec.bootstrap >= 100_000  # the old static floor
    # Day 1: typical 2020 BNB prints, plus a handful of larger ones at $8k.
    day1 = np.concatenate([np.full(10_000, 80.0), np.full(2, 8_000.0)])
    hist = trade_notional_hist(day1)
    thresh = whale_threshold([hist], spec)  # day 2's size
    assert thresh < 20_000
    trades = pl.DataFrame({
        "price": [80.0] * 100 + [8_000.0] * 5,
        "qty": [1.0] * 105,
        "is_buyer_maker": [False] * 105,
        "ts_ms": list(range(105)),
    })
    row = summarize_day(trades, None, asset="BNBUSDT", day=date(2020, 8, 31), whale_usd=thresh, liq_usd=1e18)
    assert row["wh_large_count"] == 5
    assert row["wh_large_usd"] == 40_000.0
    assert row["wh_threshold_usd"] == thresh


def test_histogram_quantile_matches_a_known_distribution():
    # 80 values of 10, 20 values of 1000: the 0.9 quantile sits in the upper group.
    usd = np.concatenate([np.full(80, 10.0), np.full(20, 1000.0)])
    q = quantile_from_hists([trade_notional_hist(usd)], 0.90)
    assert 200 < q < 2000
    lo = quantile_from_hists([trade_notional_hist(np.full(100, 10.0))], 0.90)
    assert 5 < lo < 20


# --- embargo -----------------------------------------------------------------

def test_dollar_embargo_is_identical_across_a_month_boundary(tmp_path, monkeypatch):
    """May is busy (short dollar bars), June is quiet (long ones). Length must still match."""
    import pipeline.chunked as chunked
    import pipeline.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_embargo_policy", lambda path=None: (1, 90))
    monkeypatch.setattr(chunked, "load_embargo_policy", lambda path=None: (1, 90))

    may_days = [date(2024, 5, d) for d in range(20, 32)]
    jun_days = [date(2024, 6, d) for d in range(1, 9)]
    # May: many small prints so bars close often. June: sparse large prints.
    for i, d in enumerate(may_days):
        write_bronze_day(tmp_path, "BTCUSDT", d, _day(i + 1, 100.0, d))
    for i, d in enumerate(jun_days):
        start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        ts = [int((start + timedelta(hours=h, minutes=5)).timestamp() * 1000) for h in range(24)]
        trades = pl.DataFrame({
            "agg_trade_id": list(range(1, 25)),
            "ts_us": [t * 1000 for t in ts],
            "ts_ms": ts,
            "price": [100.0] * 24,
            "qty": [80.0] * 24,  # $8k each, 24 a day
            "is_buyer_maker": [False] * 24,
        })
        write_bronze_day(tmp_path, "BTCUSDT", d, trades)

    splits_obj = __import__("pipeline.config", fromlist=["Splits"]).Splits(
        train_end=date(2024, 6, 1), val_end=date(2030, 1, 1), oos_end=date(2031, 1, 1),
        unseen_start=date(2031, 1, 1), live_start=date(2032, 1, 1),
    )
    cfg = RunConfig(
        start=date(2024, 5, 20), end=date(2024, 6, 8), assets=["BTCUSDT"],
        stages=["bars", "features", "dataset"], bar_types=["dollar"],
        data_root=tmp_path, windows=small_windows(
            z=3, fast=3, vol_cluster=3, frac_diff=4, hurst=4, kurtosis=3, efficiency=2,
            perm_entropy=4, kyle=3, yz=3, cs_block=2, funding_mom=3, hawkes_span=3, ma=3,
            horizons=(1, 2),
        ),
        threshold_overrides={"BTCUSDT": 500.0},
        splits=splits_obj,
    )
    assert run(cfg) == 0
    may_man = json.loads((tmp_path / "gold" / "dollar" / "BTCUSDT_2024-05.manifest.json").read_text())
    jun_man = json.loads((tmp_path / "gold" / "dollar" / "BTCUSDT_2024-06.manifest.json").read_text())
    assert may_man["embargo_bars"] == jun_man["embargo_bars"]
    n = may_man["embargo_bars"]
    assert n > 0
    # A per-month median would have given different lengths (May is busy, June is quiet).
    may = pl.read_parquet(tmp_path / "gold" / "dollar" / "BTCUSDT_2024-05.parquet")
    jun = pl.read_parquet(tmp_path / "gold" / "dollar" / "BTCUSDT_2024-06.parquet")
    def _median(df):
        d = df["ts_ms"].diff().drop_nulls()
        return int(d.median()) if d.len() else None
    per_month = (
        embargo_bars_for("dollar", cfg.windows, _median(may), calendar_floor_days=1, max_lookback_days=90),
        embargo_bars_for("dollar", cfg.windows, _median(jun), calendar_floor_days=1, max_lookback_days=90),
    )
    assert per_month[0] != per_month[1]
    both = pl.concat([may, jun]).sort("ts_ms")
    edge = _ms(2024, 6, 1)
    pre = [l for t, l in zip(both["ts_ms"], both["split"]) if t < edge]
    post = [l for t, l in zip(both["ts_ms"], both["split"]) if t >= edge]
    k_pre, k_post = min(n, len(pre)), min(n, len(post))
    assert k_pre > 0 and k_post > 0
    assert pre[-k_pre:] == ["embargo"] * k_pre
    assert post[:k_post] == ["embargo"] * k_post
    if len(pre) > n:
        assert pre[-n - 1] == "train"
    if len(post) > n:
        assert post[n] == "val"
    assert CONTRACT_VERSION == "2.3.0"
