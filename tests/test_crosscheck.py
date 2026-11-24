"""Public-API cross-check and the every-view report, offline (the venue is faked)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import numpy as np
import polars as pl
import pytest

import pipeline.crosscheck as cc
from pipeline.io import atomic_write_parquet

T0 = int(datetime(2020, 3, 2, tzinfo=timezone.utc).timestamp() * 1000)


def _minutes(n: int = 180, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.1, n))
    open_ = np.r_[100.0, close[:-1]]
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    vol = rng.uniform(1, 3, n)
    return pl.DataFrame({
        "asset": "BTCUSDT", "bar_type": "time_1m",
        "ts_open_ms": [T0 + i * 60_000 for i in range(n)],
        "ts_ms": [T0 + (i + 1) * 60_000 for i in range(n)],
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "volume_usd": vol * close, "buy_volume": vol / 2, "buy_usd": vol * close / 2,
        "n_trades": rng.integers(5, 20, n).astype(np.int64),
        "funding_rate": [None] * n,
    }, schema_overrides={"funding_rate": pl.Float64})


def _venue(ours: pl.DataFrame, extra_trades: int = 3) -> pl.DataFrame:
    """What the venue would serve: the same candles, counting the trades inside each aggTrade."""
    return ours.select(cc.KLINE_COLS).with_columns(pl.col("n_trades") + extra_trades)


def _hourly(m: pl.DataFrame) -> pl.DataFrame:
    return (
        m.group_by(((pl.col("ts_open_ms") - T0) // 3_600_000).alias("h"), maintain_order=True)
        .agg(pl.col("ts_open_ms").first(), pl.col("open").first(), pl.col("high").max(), pl.col("low").min(),
             pl.col("close").last(), pl.col("volume").sum(), pl.col("volume_usd").sum(),
             pl.col("buy_volume").sum(), pl.col("buy_usd").sum(), pl.col("n_trades").sum())
        .with_columns((pl.col("ts_open_ms") + 3_600_000).alias("ts_ms"))
        .drop("h")
    )


def test_identical_bars_pass_and_each_defect_is_caught():
    ours = _minutes()
    venue = _venue(ours)
    ok = cc.compare_bars(ours, venue, 60_000)
    assert cc._judge_bars(ok) == "PASS" and ok["compared"] == ours.height

    # A wrong close in the middle of the series is a scattered mismatch.
    bad = ours.with_columns(pl.when(pl.col("ts_ms") == T0 + 90 * 60_000).then(pl.col("close") * 1.001)
                            .otherwise(pl.col("close")).alias("close"))
    s = cc.compare_bars(bad, venue, 60_000)
    assert s["scattered"] == 1 and cc._judge_bars(s) == "FAIL"  # 1/180 is far over 1e-4

    # A minute the venue traded that we lost fails.
    s = cc.compare_bars(ours.filter(pl.col("ts_ms") != T0 + 50 * 60_000), venue, 60_000)
    assert s["missing"] == 1 and cc._judge_bars(s) == "FAIL"

    # More aggTrades than venue trades is impossible.
    s = cc.compare_bars(ours, venue.with_columns(pl.col("n_trades") - 10), 60_000)
    assert s["n_trades_over_venue"] > 0 and cc._judge_bars(s) == "FAIL"


def test_a_venue_outage_warns_on_1m_and_explains_the_coarser_bar():
    ours = _minutes()
    hole = [T0 + k * 60_000 for k in (70, 71, 72)]
    venue = _venue(ours).filter(~pl.col("ts_ms").is_in(hole))
    s1 = cc.compare_bars(ours, venue, 60_000)
    assert s1["only_ours"] == 3 and s1["scattered"] == 0 and cc._judge_bars(s1) == "WARN"
    outage = set(s1["holes"])
    assert set(hole) <= outage

    # The venue's hourly bar lacks the outage minutes; ours has them.
    venue_h = _hourly(_minutes().filter(~pl.col("ts_ms").is_in(hole)))
    ours_h = _hourly(ours)
    explained = cc.compare_bars(ours_h, _venue(venue_h), 3_600_000, outage)
    assert explained["outage_bars"] == 1 and explained["scattered"] == 0
    assert cc._judge_bars(explained) == "WARN"
    unexplained = cc.compare_bars(ours_h, _venue(venue_h), 3_600_000, set())
    assert unexplained["scattered"] == 1 and cc._judge_bars(unexplained) == "FAIL"


def test_dollar_bars_sit_in_the_minute_envelope_and_conserve_notional():
    m = _minutes()
    # Three dollar bars that split the minutes 0-59, 60-119, 120-179 exactly at minute edges.
    rows = []
    for k in range(3):
        part = m.slice(k * 60, 60)
        rows.append({"ts_open_ms": int(part["ts_open_ms"][0]), "ts_ms": int(part["ts_ms"][-1]) - 1,
                     "high": float(part["high"].max()), "low": float(part["low"].min()),
                     "volume_usd": float(part["volume_usd"].sum())})
    dollar = pl.DataFrame(rows)
    s = cc.check_dollar_against_minutes(dollar, _venue(m), set())
    assert s["outside"] == 0
    assert abs(s["notional_ours"] - s["notional_venue"]) <= s["slack"] + 1e-6 * s["notional_venue"]
    spiked = dollar.with_columns(pl.when(pl.col("ts_open_ms") == rows[1]["ts_open_ms"])
                                 .then(pl.col("high") * 1.01).otherwise(pl.col("high")).alias("high"))
    assert cc.check_dollar_against_minutes(spiked, _venue(m), set())["outside"] == 1


def test_gold_funding_must_be_the_last_venue_event_at_the_close():
    venue = pl.DataFrame({"ts_ms": [T0, T0 + 8 * 3_600_000], "funding_rate": [1e-4, -2e-4]})
    bars = pl.DataFrame({"ts_ms": [T0 + h * 3_600_000 for h in range(1, 12)]})
    good = bars.join_asof(venue, on="ts_ms", strategy="backward")
    assert cc.compare_gold_funding(good, venue)["differ"] == 0
    # A one-bar-early join (the 07:00-08:00 bar seeing the 08:00 rate) is a leak.
    early = bars.with_columns((pl.col("ts_ms") + 3_600_000).alias("k")).join_asof(
        venue.rename({"ts_ms": "k"}), on="k", strategy="backward").drop("k")
    assert cc.compare_gold_funding(early, venue)["differ"] == 1


def test_a_bar_stamped_one_day_off_fails_the_venue_lag_test():
    rng = np.random.default_rng(1)
    days = 200
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, days)))
    ts = [T0 + (i + 1) * 86_400_000 for i in range(days)]
    ours = pl.DataFrame({"ts_ms": ts, "close": px})
    venue = pl.DataFrame({"ts_ms": ts, "close_v": px * (1 + rng.normal(0, 0.001, days))})
    s = cc.compare_venue(ours, venue)
    assert cc._judge_venue(s) == "PASS" and s["median_dev"] < 0.005
    shifted = ours.with_columns(pl.col("ts_ms") + 86_400_000)  # our close stamped a day late
    assert cc._judge_venue(cc.compare_venue(shifted, venue)) == "FAIL"


def test_the_crosscheck_command_end_to_end_against_a_fake_venue(tmp_path, monkeypatch):
    m = _minutes()
    h = _hourly(m).with_columns(pl.lit("time_1h").alias("bar_type"), pl.lit("BTCUSDT").alias("asset"),
                                pl.lit(None, dtype=pl.Float64).alias("funding_rate"))
    for bt, df in (("time_1m", m), ("time_1h", h)):
        atomic_write_parquet(df, tmp_path / "gold" / bt / "BTCUSDT_2020-03.parquet",
                             schema_name="dataset_v1", params={})
    served = {"1m": m, "1h": h}

    def fake(url: str):
        assert url.startswith(cc.BINANCE_REST)
        interval = url.split("interval=")[1].split("&")[0]
        start = int(url.split("startTime=")[1].split("&")[0])
        df = served[interval].filter(pl.col("ts_open_ms") >= start).head(cc.REST_LIMIT)
        return [[r["ts_open_ms"], str(r["open"]), str(r["high"]), str(r["low"]), str(r["close"]),
                 str(r["volume"]), r["ts_ms"] - 1, str(r["volume_usd"]), r["n_trades"] + 1,
                 str(r["buy_volume"]), str(r["buy_usd"]), "0"] for r in df.iter_rows(named=True)]

    monkeypatch.setattr(cc, "_get_json", fake)
    report = cc.crosscheck(tmp_path, ["BTCUSDT"], date(2020, 3, 2), date(2020, 3, 2),
                           ["time_1m", "time_1h"], venue=False, funding=False)
    assert {c.name: c.severity for c in report.checks} == {"rest_time_1m": "PASS", "rest_time_1h": "PASS"}
    saved = json.loads((tmp_path / "crosscheck" / "report.json").read_text())
    assert saved["per_asset"]["BTCUSDT"]["time_1m"]["compared"] == m.height
    # Settled months are cached: a second run makes no request.
    monkeypatch.setattr(cc, "_get_json", lambda url: pytest.fail(f"refetched {url}"))
    assert cc.crosscheck(tmp_path, ["BTCUSDT"], date(2020, 3, 2), date(2020, 3, 2),
                         ["time_1m", "time_1h"], venue=False, funding=False).code == 0
    # --refresh fetches live again and proves the cache equals the venue, without overwriting it.
    monkeypatch.setattr(cc, "_get_json", fake)
    cache = tmp_path / "crosscheck" / "binance_rest" / "1h" / "BTCUSDT_2020-03.parquet"
    before = cache.read_bytes()
    live = cc.crosscheck(tmp_path, ["BTCUSDT"], date(2020, 3, 2), date(2020, 3, 2),
                         ["time_1m", "time_1h"], venue=False, funding=False, refresh=True)
    checks = {c.name: c for c in live.checks}
    assert checks["live_refetch"].severity == "PASS" and "2 had a copy" in checks["live_refetch"].detail
    assert cache.read_bytes() == before
    # A cached copy that no longer matches the venue is reported, and still not overwritten.
    served["1h"] = h.with_columns((pl.col("close") * 1.01).alias("close"))
    stale = cc.crosscheck(tmp_path, ["BTCUSDT"], date(2020, 3, 2), date(2020, 3, 2),
                          ["time_1h"], venue=False, funding=False, refresh=True)
    assert {c.name: c.severity for c in stale.checks}["live_refetch"] == "FAIL"
    assert cache.read_bytes() == before


def test_the_every_view_report_writes_an_index_and_a_page(tmp_path):
    pytest.importorskip("plotly")
    from pipeline.inspect import write_report

    m = _minutes().with_columns(pl.lit(False).alias("is_warmup"), pl.lit(0.5).alias("norm_example"))
    atomic_write_parquet(m, tmp_path / "gold" / "time_1m" / "BTCUSDT_2020-03.parquet",
                         schema_name="dataset_v1", params={})
    index = write_report(tmp_path / "out", tmp_path, ["BTCUSDT"], date(2020, 3, 2), date(2020, 3, 2))
    assert "BTCUSDT.html" in index.read_text(encoding="utf-8")
    page = (tmp_path / "out" / "BTCUSDT.html").read_text(encoding="utf-8")
    assert "feature coverage after warmup" in page and "norm_example" in page
