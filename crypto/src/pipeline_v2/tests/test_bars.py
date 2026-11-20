"""Time bars, dollar bars, reconciliation, gaps."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import polars as pl

from pipeline_v2.bars.threshold import DollarAccumulator, dollar_bars, dollar_bars_chunked
from pipeline_v2.bars.time import bars_to_time, contiguous_hole_ms, gaps_ms, trades_to_time
from pipeline_v2.constants import TIME_WIDTH_MS
from pipeline_v2.tests.fixtures import make_trades
from pipeline_v2.validate.gates import gate_ohlc, gate_time_reconcile, gate_trade_count


def _clock_trades(n_minutes: int = 180, start: datetime | None = None) -> pl.DataFrame:
    start = start or datetime(2024, 6, 1, 0, 0, 30, tzinfo=timezone.utc)
    return make_trades(n_minutes, start=start, step=timedelta(minutes=1), price0=50.0, qty=2.0, seed=1, asset_noise=0.0)


def test_time_bars_sum_and_right_closed_boundary():
    trades = _clock_trades(120)
    one_m = trades_to_time(trades, 60_000, "BTCUSDT", "time_1m")
    one_h = bars_to_time(one_m, TIME_WIDTH_MS["time_1h"], "BTCUSDT", "time_1h")
    one_d = bars_to_time(one_m, TIME_WIDTH_MS["time_1d"], "BTCUSDT", "time_1d")
    assert one_m.height == 120
    assert int(one_m["n_trades"].sum()) == trades.height
    assert abs(float(one_m["volume"].sum()) - float(one_h["volume"].sum())) < 1e-8
    assert abs(float(one_m["volume_usd"].sum()) - float(one_d["volume_usd"].sum())) < 1e-6
    assert int(one_m["n_trades"].sum()) == int(one_h["n_trades"].sum()) == int(one_d["n_trades"].sum())
    # A 1h bar's volume is the hour, not one constituent minute.
    assert float(one_h["volume"][0]) == float(one_m.head(60)["volume"].sum())
    # Label is the close of the bucket. First trade is 00:00:30 -> bar closes 00:01:00.
    first_close = int(datetime(2024, 6, 1, 0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert one_m["ts_ms"][0] == first_close
    assert one_m["ts_open_ms"][0] == first_close - 60_000
    assert gate_ohlc(one_m).severity == "pass"
    ladder = {"time_1m": one_m, "time_1h": one_h, "time_1d": one_d}
    assert gate_time_reconcile(ladder).severity == "pass"
    assert gate_trade_count(int(one_m["n_trades"].sum()), trades.height).severity == "pass"


def test_trade_on_the_boundary_belongs_to_the_closing_bar():
    # Exactly 00:01:00.000 belongs to the bar that closes at 00:01, not the next one.
    ts = int(datetime(2024, 6, 1, 0, 1, tzinfo=timezone.utc).timestamp() * 1000)
    trades = pl.DataFrame({
        "agg_trade_id": [1, 2],
        "ts_us": [ts * 1000, (ts + 1) * 1000],
        "ts_ms": [ts, ts + 1],
        "price": [10.0, 11.0],
        "qty": [1.0, 1.0],
        "is_buyer_maker": [False, True],
    })
    bars = trades_to_time(trades, 60_000, "ETHUSDT", "time_1m")
    assert bars.height == 2
    assert bars["ts_ms"].to_list() == [ts, ts + 60_000]
    assert bars["close"].to_list() == [10.0, 11.0]


def test_dollar_bars_match_one_shot_and_drop_the_open_tail():
    trades = make_trades(30, start=datetime(2024, 6, 1, tzinfo=timezone.utc), step=timedelta(seconds=10), price0=10.0, qty=1.0, seed=2, asset_noise=0.0)
    # Flat price 10, qty ~1, threshold 25 closes a bar every few trades.
    threshold = 25.0
    one, excluded = dollar_bars(trades, threshold, "BTCUSDT")
    # Chunked path must agree, including across the boundary.
    mid = trades.height // 2
    chunked, excluded2 = dollar_bars_chunked([trades.head(mid), trades.tail(trades.height - mid)], threshold, "BTCUSDT")
    assert excluded == excluded2
    assert one.height == chunked.height
    assert one["n_trades"].to_list() == chunked["n_trades"].to_list()
    assert one["close"].to_list() == chunked["close"].to_list()
    assert int(one["n_trades"].sum()) + excluded == trades.height
    # The excluded tail is a suffix and is strictly the incomplete id.
    assert one["ts_ms"].is_sorted()
    assert one["bar_id"].to_list() == list(range(one.height))
    # Buyer-maker true is the sell aggressor.
    assert abs(float((one["buy_volume"] + one["sell_volume"]).sum()) - float(one["volume"].sum())) < 1e-8


def test_dollar_accumulator_matches_one_shot():
    trades = make_trades(40, start=datetime(2024, 6, 2, tzinfo=timezone.utc), step=timedelta(seconds=5), price0=20.0, qty=1.0, seed=3, asset_noise=0.002)
    threshold = 40.0
    one, excluded = dollar_bars(trades, threshold, "SOLUSDT")
    acc = DollarAccumulator(threshold)
    acc.add(trades.head(15))
    acc.add(trades.slice(15, 15))
    acc.add(trades.slice(30))
    streamed, excluded2 = acc.finish("SOLUSDT")
    assert excluded == excluded2
    assert one.select(["n_trades", "volume_usd", "close", "open"]).to_dicts() == streamed.select(["n_trades", "volume_usd", "close", "open"]).to_dicts()


def test_same_millisecond_burst_collapses_to_one_bar():
    ts = 1_700_000_000_000
    n = 10
    trades = pl.DataFrame({
        "agg_trade_id": list(range(1, n + 1)),
        "ts_us": [ts * 1000 + i for i in range(n)],
        "ts_ms": [ts] * n,
        "price": [100.0 + i for i in range(n)],
        "qty": [1.0] * n,
        "is_buyer_maker": [i % 2 == 0 for i in range(n)],
    })
    # Each trade is $100+, so a $150 threshold closes several bars inside one ms.
    bars, excluded = dollar_bars(trades, 150.0, "BTCUSDT")
    assert bars["ts_ms"].n_unique() == bars.height
    assert int(bars["n_trades"].sum()) + excluded == n
    assert bars["ts_ms"].is_sorted()


def test_gap_list_and_btc_hole():
    start = datetime(2024, 6, 1, 12, 0, 30, tzinfo=timezone.utc)
    a = make_trades(10, start=start, step=timedelta(minutes=1), seed=4, asset_noise=0.0)
    b = make_trades(10, start=start + timedelta(minutes=40), step=timedelta(minutes=1), seed=5, asset_noise=0.0)
    b = b.with_columns(pl.col("agg_trade_id") + 100)
    trades = pl.concat([a, b])
    bars = trades_to_time(trades, 60_000, "BTCUSDT", "time_1m")
    missing = gaps_ms(bars)
    assert missing
    assert contiguous_hole_ms(missing) >= 15 * 60_000
    # Every trade still lands in a bar. The hole is minutes with no trades.
    assert int(bars["n_trades"].sum()) == trades.height
