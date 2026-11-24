"""Parsers, symbols, atomic writes, and the decisions that live in config."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from pipeline.constants import OMITTED_BAR_TYPES
from pipeline.config import dollar_spec, load_windows
from pipeline.fetch_crypto import (
    empty_venue_frame,
    funding_url,
    parse_aggtrades,
    parse_funding,
    parse_metrics,
)
from pipeline.io import atomic_write_parquet, file_sha256
from pipeline.symbols import canonical_symbol


def test_symbols():
    assert canonical_symbol("btc") == "BTCUSDT"
    assert canonical_symbol("BTC/USDT") == "BTCUSDT"
    assert canonical_symbol("BTCUSDT") == "BTCUSDT"
    assert canonical_symbol("eth-usdt") == "ETHUSDT"


def test_dollar_sizing_is_adaptive_with_no_static_table():
    # The v1 per-asset sizes (BTC $2M, ...) were removed in 2.5.0: the first day calibrates.
    spec = dollar_spec("ETH")
    assert spec.constant is None and spec.target_bars_per_day == 288


def test_embargo_is_cadence_aware_not_a_flat_1200():
    from pipeline.config import embargo_bars_for
    windows = load_windows()
    daily = embargo_bars_for("time_1d", windows)
    hourly = embargo_bars_for("time_1h", windows)
    minute = embargo_bars_for("time_1m", windows)
    dollar = embargo_bars_for("dollar", windows, median_bar_ms=5 * 60_000)
    assert daily != 1200
    assert daily < hourly
    assert minute >= 7 * 24 * 60
    assert dollar >= 7 * 24 * 12
    assert daily >= 7
    assert "dib" in OMITTED_BAR_TYPES


def test_aggtrades_sorted_and_deduped_keeping_id():
    # Deliberately unsorted, with a duplicate id whose later row must win,
    # and two trades that share price/qty but are different ids (v1 unique()
    # on price/qty would have collapsed them).
    csv = (
        "900,10,1,1,1,1700000000000000,true,true\n"
        "100,10,1,1,1,1600000000000000,false,true\n"
        "100,11,2,1,1,1600000000001000,false,true\n"
        "50,10,1,1,1,1500000000000000,true,true\n"
    ).encode()
    raw = _zip(csv)
    df = parse_aggtrades(raw)
    ids = df["agg_trade_id"].to_list()
    assert ids == sorted(ids)
    assert ids == [50, 100, 900]
    # The duplicate id 100 keeps the later row (price 11), not the first.
    assert df.filter(pl.col("agg_trade_id") == 100)["price"][0] == 11
    assert df["ts_us"].is_sorted()
    assert df["ts_ms"][0] == df["ts_us"][0] // 1000
    assert "is_buyer_maker" in df.columns


def test_funding_monthly_columns_not_the_dead_daily_path():
    url = funding_url("BTCUSDT", 2026, 10)
    assert "/monthly/fundingRate/" in url
    assert "/daily/fundingRate/" not in url
    csv = "calc_time,funding_interval_hours,last_funding_rate\n1700000000000,8,0.0001\n1700028800000,8,-0.0002\n".encode()
    df = parse_funding(_zip(csv))
    assert df["funding_rate"].to_list() == [0.0001, -0.0002]
    assert df["ts_ms"].is_sorted()
    # Legacy header still accepted.
    legacy = "calcTime,fundingRate\n1700000000000,0.01\n".encode()
    old = parse_funding(_zip(legacy))
    assert old["funding_rate"][0] == 0.01


def test_headered_kline_uses_vision_names():
    from pipeline.fetch_crypto import parse_klines
    csv = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        "1700000000000,1,2,0.5,1.5,10,1700086399999,15,100,4,6,0\n"
    ).encode()
    df = parse_klines(_zip(csv), width_ms=86_400_000)
    assert df["n_trades"][0] == 100
    assert df["taker_buy_base"][0] == 4
    assert df["ts_ms"][0] == 1700000000000 + 86_400_000


def test_metrics_do_not_invent_long_short_ratio():
    csv = "create_time,symbol,sum_open_interest,sum_open_interest_value\n1700000000000,BTCUSDT,10,1000.5\n".encode()
    df = parse_metrics(_zip(csv))
    assert "long_short_ratio" not in df.columns
    assert df["sum_open_interest_value"][0] == pytest.approx(1000.5)
    assert df["ts_ms"][0] == 1700000000000


def test_empty_venue_does_not_keyerror():
    empty = pl.DataFrame()
    out = empty_venue_frame(empty)
    assert "date" in out.columns
    assert out.is_empty()
    # A frame that is non-empty but missing the column used to crash the job.
    weird = pl.DataFrame({"close": [1.0]})
    out2 = empty_venue_frame(weird)
    assert out2.is_empty()


def test_atomic_write_is_idempotent(tmp_path):
    df = pl.DataFrame({"ts_ms": [1, 2], "asset": ["BTCUSDT", "BTCUSDT"]}).sort("ts_ms")
    path = tmp_path / "a.parquet"
    params = {"k": 1}
    atomic_write_parquet(df, path, schema_name="bars_v1", params=params, sort_cols=["ts_ms"])
    first = file_sha256(path)
    atomic_write_parquet(df, path, schema_name="bars_v1", params=params, sort_cols=["ts_ms"])
    assert file_sha256(path) == first
    # A crash mid-write must not leave the temp file as the destination.
    assert not list(tmp_path.glob("*.tmp.*"))


def _zip(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("file.csv", payload)
    return buf.getvalue()


def test_u10_listing_dates_floor_sol_and_avax_at_binance_spot_launch():
    """SOL and AVAX listed on Binance after 2020-01-01; a Jan start must not fetch earlier."""
    from datetime import date
    from pipeline.config import effective_start, load_listing_dates, universe_symbols

    listings = load_listing_dates()
    assert listings["SOLUSDT"] == date(2020, 8, 11)
    assert listings["AVAXUSDT"] == date(2020, 9, 22)
    jan = date(2020, 1, 1)
    dec = date(2020, 12, 31)
    assert effective_start("SOLUSDT", jan, listings) == date(2020, 8, 11)
    assert effective_start("AVAXUSDT", jan, listings) == date(2020, 9, 22)
    # Already-listed U10 names still start at the 2020-01-01 history floor.
    assert effective_start("BTCUSDT", jan, listings) == jan
    assert "SOLUSDT" in universe_symbols("u10")
    assert "AVAXUSDT" in universe_symbols("u10")
    # A window that ends before the listing is unusable as a start>end signal.
    assert effective_start("SOLUSDT", jan, listings) > date(2020, 8, 10)
