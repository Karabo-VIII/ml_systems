"""Parsers, symbols, atomic writes, and the decisions that live in config."""

from __future__ import annotations

import io
import zipfile
from datetime import date

import polars as pl
import pytest

from pipeline_v2.constants import OMITTED_BAR_TYPES
from pipeline_v2.contract import dollar_threshold, load_windows
from pipeline_v2.fetch.vision import (
    empty_venue_frame,
    funding_url,
    parse_aggtrades,
    parse_funding,
    parse_metrics,
)
from pipeline_v2.io import atomic_write_parquet, file_sha256
from pipeline_v2.symbols import canonical_symbol


def test_symbols():
    assert canonical_symbol("btc") == "BTCUSDT"
    assert canonical_symbol("BTC/USDT") == "BTCUSDT"
    assert canonical_symbol("BTCUSDT") == "BTCUSDT"
    assert canonical_symbol("eth-usdt") == "ETHUSDT"


def test_dollar_table_matches_the_fixed_v1_sizes():
    assert dollar_threshold("BTCUSDT") == 2_000_000
    assert dollar_threshold("ETH") == 700_000
    assert dollar_threshold("LTCUSDT") == 50_000
    with pytest.raises(KeyError):
        dollar_threshold("NOSUCHUSDT")


def test_embargo_is_max_horizon_window():
    windows = load_windows()
    assert windows.embargo_bars() == max(64, 1000 + 200)
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
    from pipeline_v2.fetch.vision import parse_klines
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
