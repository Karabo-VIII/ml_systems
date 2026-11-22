"""Offline FX: synthetic bi5, sessions, weekend gap, causal features, shared orchestrator."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

from pipeline.bars import build_fx_ladder, fx_ticks_to_time
from pipeline.config import fx_pip, fx_price_divisor, universe_symbols
from pipeline.fetch_crypto import BinanceVisionSource
from pipeline.fetch_fx import DukascopySource, decode_bi5, encode_bi5, hour_url
from pipeline.features import compute_fx_features, add_usd_strength
from pipeline.orchestrator import RunConfig, run


def _hour(y, m, d, h) -> datetime:
    return datetime(y, m, d, h, tzinfo=timezone.utc)


def test_bi5_roundtrip_scales_jpy_and_majors():
    assert fx_price_divisor("EURUSD") == 100_000
    assert fx_price_divisor("USDJPY") == 1_000
    assert fx_pip("USDJPY") == 0.01
    assert fx_pip("EURUSD") == 0.0001
    start = _hour(2024, 1, 15, 12)
    start_ms = int(start.timestamp() * 1000)
    ticks = pl.DataFrame({
        "ts_ms": [start_ms + 1_000, start_ms + 2_000],
        "bid": [1.08500, 1.08510],
        "ask": [1.08502, 1.08514],
        "ask_vol": [1.0, 1.0],
        "bid_vol": [1.0, 1.0],
    })
    blob = encode_bi5(ticks, start_ms, 100_000)
    back = decode_bi5(blob, start_ms, 100_000)
    assert back["bid"].to_list() == [1.08500, 1.08510]
    assert abs(back["mid"][0] - (1.08500 + 1.08502) / 2) < 1e-12
    jpy = pl.DataFrame({
        "ts_ms": [start_ms + 5],
        "bid": [150.250],
        "ask": [150.253],
        "ask_vol": [0.0],
        "bid_vol": [0.0],
    })
    back_j = decode_bi5(encode_bi5(jpy, start_ms, 1_000), start_ms, 1_000)
    assert abs(back_j["bid"][0] - 150.250) < 1e-9
    assert "2024/00/15/12h_ticks.bi5" in hour_url("EURUSD", start)


def test_tick_bar_is_stamped_at_close_and_uses_tick_count():
    # A bar covers [open, close) and is stamped at close, as for crypto trades: a print
    # exactly on the boundary is the first tick of the NEXT bar.
    close_ms = int(_hour(2024, 1, 15, 13).timestamp() * 1000)
    ticks = pl.DataFrame({
        "ts_ms": [close_ms - 30_000, close_ms, close_ms + 30_000],
        "bid": [1.10, 1.20, 1.30],
        "ask": [1.10, 1.20, 1.30],
        "mid": [1.10, 1.20, 1.30],
        "spread": [0.0, 0.0, 0.0],
    })
    bars = fx_ticks_to_time(ticks, 3_600_000, "EURUSD", "time_1h")
    assert bars["ts_ms"].to_list() == [close_ms, close_ms + 3_600_000]
    first = bars.filter(pl.col("ts_ms") == close_ms)
    assert first["n_trades"][0] == 1
    assert first["volume"][0] == 1.0
    assert first["close"][0] == 1.10
    assert first["volume_usd"][0] is None
    assert first["buy_volume"][0] is None
    second = bars.filter(pl.col("ts_ms") == close_ms + 3_600_000)
    assert second["n_trades"][0] == 2
    assert second["open"][0] == 1.20


def test_sessions_follow_dst_and_weekend_gap_is_causal():
    # January: 12:00 UTC is 07:00 New York, before the 08:00 local open.
    # July: 12:00 UTC is 08:00 New York, the open.
    def one_hour(day: date, hour: int) -> pl.DataFrame:
        start = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
        ms = int(start.timestamp() * 1000) + 60_000
        return pl.DataFrame({
            "ts_ms": [ms],
            "bid": [1.1], "ask": [1.1002], "mid": [1.1001], "spread": [0.0002],
        })

    jan = compute_fx_features(
        fx_ticks_to_time(one_hour(date(2024, 1, 15), 12), 3_600_000, "EURUSD", "time_1h"),
        0.0001,
    )
    jul = compute_fx_features(
        fx_ticks_to_time(one_hour(date(2024, 7, 15), 12), 3_600_000, "EURUSD", "time_1h"),
        0.0001,
    )
    assert jan["is_newyork"][0] == 0
    assert jul["is_newyork"][0] == 1
    assert jan["is_london"][0] == 1
    assert jan["is_tokyo"][0] == 0

    # Friday 21:00 UTC close, Sunday 23:00 UTC reopen. Gap is one pip, only on the reopen bar.
    friday = int(datetime(2024, 1, 12, 21, tzinfo=timezone.utc).timestamp() * 1000)
    sunday = int(datetime(2024, 1, 14, 23, tzinfo=timezone.utc).timestamp() * 1000)
    ticks = pl.DataFrame({
        "ts_ms": [friday - 1_800_000, sunday - 1_800_000],
        "bid": [1.1000, 1.1001],
        "ask": [1.1000, 1.1001],
        "mid": [1.1000, 1.1001],
        "spread": [0.0, 0.0],
    })
    bars = fx_ticks_to_time(ticks, 3_600_000, "EURUSD", "time_1h")
    feat = compute_fx_features(bars, 0.0001)
    gaps = feat["weekend_gap_pips"].to_list()
    assert gaps[0] is None
    assert abs(gaps[1] - 1.0) < 1e-6


def test_features_do_not_look_ahead():
    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    rows = []
    px = 1.10
    for i in range(400):
        px *= 1.00001 if i % 5 else 0.99999
        ms = int((start + timedelta(minutes=i, seconds=30)).timestamp() * 1000)
        half = 0.00005 + (i % 3) * 0.00001
        rows.append({
            "ts_ms": ms, "bid": px - half, "ask": px + half,
            "mid": px, "spread": 2 * half,
        })
    ticks = pl.DataFrame(rows)
    bars = build_fx_ladder(ticks, "EURUSD", ["time_1m"])["time_1m"]
    base = compute_fx_features(bars, 0.0001)
    bumped = bars.with_columns(
        pl.when(pl.col("ts_ms") == bars["ts_ms"].max())
        .then(pl.col("close") * 1.05)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    # high must stay above the bumped close or the bar is nonsense; bump high too.
    bumped = bumped.with_columns(pl.max_horizontal("high", "close").alias("high"))
    alt = compute_fx_features(bumped, 0.0001)
    early = base.head(base.height - 100)
    same = alt.head(alt.height - 100)
    for col in ("ret_1", "rv_20", "parkinson_20", "garman_klass_20", "atr_pips_20", "efficiency_ratio_20"):
        assert early[col].to_list() == same[col].to_list()


def test_usd_strength_needs_two_legs_and_signs():
    ts = [1_700_000_000_000, 1_700_003_600_000]
    def frame(pair, rets):
        close = [1.0]
        for r in rets:
            close.append(close[-1] * (1.0 + r))
        # ret_1 is recomputed from close, so build closes that yield rets.
        return pl.DataFrame({
            "ts_ms": ts,
            "close": close[1:],
            "ret_1": rets,
        })
    # EURUSD up 1% is USD down. USDJPY up 1% is USD up. Index is 0.
    frames = add_usd_strength({
        "EURUSD": frame("EURUSD", [0.01, 0.0]),
        "USDJPY": frame("USDJPY", [0.01, 0.0]),
    })
    assert abs(frames["EURUSD"]["usd_strength_ret"][0]) < 1e-12
    single = frame("EURUSD", [0.01, 0.0]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("usd_strength_ret")
    )
    alone = add_usd_strength({"EURUSD": single})
    assert alone["EURUSD"]["usd_strength_ret"][0] is None


def test_offline_fx_run_reconciles(tmp_path):
    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    for pair, px0 in (("EURUSD", 1.10), ("USDJPY", 148.0)):
        rows = []
        px = px0
        for i in range(18 * 60):
            px *= 1.000002 if i % 2 == 0 else 0.999998
            ms = int((start + timedelta(minutes=i, seconds=20)).timestamp() * 1000)
            half = (0.0001 if pair != "USDJPY" else 0.01)
            for k in range(1 + (i % 3)):
                rows.append({
                    "ts_ms": ms + k,
                    "bid": px - half, "ask": px + half,
                    "mid": px, "spread": 2 * half,
                })
        ticks = pl.DataFrame(rows)
        # One parquet per hour, which is what a resumed fetch leaves behind.
        from pipeline.io import atomic_write_parquet
        from pipeline.paths import fx_hour_path
        for hour in range(18):
            h0 = int((start + timedelta(hours=hour)).timestamp() * 1000)
            h1 = h0 + 3_600_000
            chunk = ticks.filter((pl.col("ts_ms") >= h0) & (pl.col("ts_ms") < h1))
            path = fx_hour_path(tmp_path, pair, start.date(), hour)
            atomic_write_parquet(chunk, path, schema_name="fx_ticks_v1", params={"pair": pair, "hour": hour})
    code = run(RunConfig(
        start=date(2024, 1, 15),
        end=date(2024, 1, 15),
        assets=["EURUSD", "USDJPY"],
        market="fx",
        stages=["bars", "features", "dataset", "validate"],
        bar_types=["time_1m", "time_1h"],
        data_root=tmp_path,
        force=True,
        skip_network=True,
    ))
    assert code < 2
    assert universe_symbols("majors")[0] == "EURUSD"
    assert DukascopySource(["eur/usd"]).calendar() == "fx_week"
    assert BinanceVisionSource(["BTC"]).calendar() == "24/7"
    assert DukascopySource(["EURUSD"]).list_instruments() == ["EURUSD"]
def test_fx_coarse_bars_use_clock_open_when_minutes_are_missing():
    # Ticks only in the last 10 minutes of the 13:00 UTC hour.
    close = int(_hour(2024, 1, 15, 14).timestamp() * 1000)
    ticks = pl.DataFrame({
        "ts_ms": [close - 600_000, close - 120_000, close - 1],
        "bid": [1.10, 1.11, 1.12],
        "ask": [1.10, 1.11, 1.12],
        "mid": [1.10, 1.11, 1.12],
        "spread": [0.0, 0.0, 0.0],
    })
    ladder = build_fx_ladder(ticks, "EURUSD", ["time_1m", "time_1h"])
    hour = ladder["time_1h"]
    assert hour.height == 1
    assert hour["ts_ms"][0] == close
    assert hour["ts_open_ms"][0] == close - 3_600_000
    assert hour["open"][0] == 1.10
    assert hour["close"][0] == 1.12
    assert hour["n_trades"][0] == 3
def test_fx_weekend_gap_follows_new_york_close():
    from pipeline.orchestrator import _fx_weekday_gaps
    # Friday 16:30 New York is still open in winter (21:30 UTC) and already
    # closed in summer (20:30 UTC). The filter must follow the zone, not a
    # fixed 21:00 UTC cut.
    winter = int(datetime(2024, 1, 12, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
    summer = int(datetime(2024, 7, 12, 21, 30, tzinfo=timezone.utc).timestamp() * 1000)
    monday = int(datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert _fx_weekday_gaps([winter]) == [winter]
    assert _fx_weekday_gaps([summer]) == []
    assert _fx_weekday_gaps([monday]) == [monday]
def test_fetch_fx_range_retries_then_continues(tmp_path, monkeypatch):
    from pipeline.fetch_fx import fetch_fx_range, encode_bi5, decode_bi5
    from pipeline.io import atomic_write_parquet
    from pipeline.paths import fx_hour_path
    calls = {"n": 0}
    sleeps = []
    def fake_sleep(s):
        sleeps.append(s)
    def fake_hour(root, pair, hour_start, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("HTTP 503")
        path = fx_hour_path(root, pair, hour_start.date(), hour_start.hour)
        atomic_write_parquet(
            pl.DataFrame(schema={"ts_ms": pl.Int64, "bid": pl.Float64, "ask": pl.Float64, "mid": pl.Float64, "spread": pl.Float64}),
            path, schema_name="fx_ticks_v1", params={"pair": pair},
        )
    monkeypatch.setattr("pipeline.fetch_fx.fetch_hour", fake_hour)
    monkeypatch.setattr("pipeline.fetch_fx.time.sleep", fake_sleep)
    n = fetch_fx_range(tmp_path, "EURUSD", date(2024, 1, 15), date(2024, 1, 15))
    assert n == 24
    assert calls["n"] == 24 + 2  # first hour failed twice, then all 24 written
    assert sleeps == [45.0, 90.0]
    # The failed hour was not cached as empty before it finally wrote.
    from pipeline.paths import fx_hour_path as pth
    assert pth(tmp_path, "EURUSD", date(2024, 1, 15), 0).exists()
