"""Contract 2.4.x regressions: what a full-history, resumable, incremental rebuild needs.

Each test pins one defect found in the 2.3.0 audit and fails on 2.3.0 code.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import pipeline.chunked as chunked
import pipeline.fetch_crypto as fc
from pipeline.config import (
    Windows, dollar_threshold, embargo_bars_for, load_dollar_thresholds,
    load_splits, load_windows, scale_windows,
)
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.io import atomic_write_parquet, file_sha256
from pipeline.orchestrator import RunConfig, run
from pipeline.paths import trades_path
from fixtures import small_windows, write_bronze_day
from test_resume import _day


# --- embargo: the chained lookback is capped, so val and oos survive ------------------

def test_chained_lookback_never_exceeds_the_cap_on_any_cadence():
    w = load_windows()
    for bt, width in TIME_WIDTH_MS.items():
        scaled = scale_windows(w, bt, width, max_lookback_days=90)
        assert scaled.max_lookback * width <= 90 * DAY_MS, bt
        n = embargo_bars_for(bt, w, width, calendar_floor_days=7, max_lookback_days=90)
        assert n * width <= 90 * DAY_MS, bt


def test_daily_val_and_oos_keep_rows_after_both_side_embargo():
    """2.3.0: 180-day embargo on 1d left val (319 days) and oos (304 days) empty."""
    w, s = load_windows(), load_splits()
    n = embargo_bars_for("time_1d", w, DAY_MS, calendar_floor_days=7, max_lookback_days=90)
    for a, b in ((s.train_end, s.val_end), (s.val_end, s.oos_end)):
        assert (b - a).days - 2 * n > 100


def test_fine_cadences_keep_their_windows():
    w = load_windows()
    for bt in ("time_1m", "time_5m", "time_30m", "time_1h"):
        assert scale_windows(w, bt, TIME_WIDTH_MS[bt]) == w


def test_lookback_counts_every_chained_input():
    w = Windows(z=10, frac_diff=5, hurst=5, perm_entropy=5, kyle=5, vol_cluster=5, ma=5,
                kurtosis=40, funding_mom=5, yz=5, efficiency=5, fast=5, slow_span=5, cs_block=2)
    # kurtosis(40) is the longest input; 2.3.0 ignored it.
    assert w.max_lookback == 41 + 10


# --- fetch: adopt finished files, record permanent absences, retry transients ----------

def test_a_finished_file_without_a_manifest_record_is_adopted(tmp_path):
    day = date(2020, 1, 1)
    dest = trades_path(tmp_path, "BTCUSDT", day)
    write_bronze_day(tmp_path, "BTCUSDT", day, _day(1, 100.0, day))
    arts: dict = {}
    assert fc._check_fresh(arts, day.isoformat(), dest, force=False)
    rec = arts[day.isoformat()]
    assert rec["status"] == "ok" and rec["sha256"] == file_sha256(dest) and rec["adopted"]
    assert not fc._check_fresh({}, day.isoformat(), dest, force=True)


def test_a_torn_file_is_not_adopted(tmp_path):
    day = date(2020, 1, 1)
    dest = trades_path(tmp_path, "BTCUSDT", day)
    write_bronze_day(tmp_path, "BTCUSDT", day, _day(1, 100.0, day))
    dest.write_bytes(dest.read_bytes()[:-10])
    assert not fc._check_fresh({}, day.isoformat(), dest, force=False)


def test_old_404_is_absent_and_not_refetched():
    assert fc._miss_status(date(2020, 1, 1), today=date(2026, 1, 1)) == "absent"
    assert fc._miss_status(date(2025, 12, 30), today=date(2026, 1, 1)) == "missing_not_published"
    from pathlib import Path
    assert fc._fresh({"status": "absent"}, trades_path(Path("x"), "BTCUSDT", date(2020, 1, 1)), False)


def test_http_get_retries_transient_errors(monkeypatch):
    calls = []

    def once(url, timeout):
        calls.append(url)
        return (503, b"", "HTTP 503") if len(calls) < 3 else (200, b"ok", None)

    monkeypatch.setattr(fc, "_http_get_once", once)
    monkeypatch.setattr(fc, "_sleep", lambda s: None)
    assert fc.http_get("u") == (200, b"ok", None)
    assert len(calls) == 3


def test_http_get_does_not_retry_a_404(monkeypatch):
    calls = []
    monkeypatch.setattr(fc, "_http_get_once", lambda u, t: calls.append(u) or (404, b"", "HTTP 404"))
    monkeypatch.setattr(fc, "_sleep", lambda s: None)
    assert fc.http_get("u")[0] == 404
    assert len(calls) == 1


def test_monthly_zip_404_marks_the_month_absent_without_daily_requests(tmp_path, monkeypatch):
    daily = []
    monkeypatch.setattr(fc, "http_get_to_path", lambda url, dest, timeout=120: (404, "HTTP 404"))
    monkeypatch.setattr(fc, "http_get", lambda url, timeout=120: daily.append(url) or (404, b"", "HTTP 404"))
    report = fc.fetch_trades(tmp_path, "PEPEUSDT", date(2020, 3, 1), date(2020, 3, 31))
    assert not daily
    man = json.loads((tmp_path / "bronze" / "_manifest" / "trades" / "PEPEUSDT.json").read_text())
    assert {v["status"] for v in man["artifacts"].values()} == {"absent"}
    assert len(report.records) == 31


# --- externals merge instead of overwrite ---------------------------------------------

def test_external_write_merges_and_new_rows_win(tmp_path):
    from pipeline.fetch_external import _write

    path = tmp_path / "x.parquet"
    a = pl.DataFrame({"asset": ["BTCUSDT"] * 2, "event_time_ms": [0, DAY_MS], "v": [1.0, 2.0]})
    b = pl.DataFrame({"asset": ["BTCUSDT"] * 2, "event_time_ms": [DAY_MS, 2 * DAY_MS], "v": [9.0, 3.0]})
    _write(a, path, "t", {})
    _write(b, path, "t", {})
    out = pl.read_parquet(path).sort("event_time_ms")
    assert out["event_time_ms"].to_list() == [0, DAY_MS, 2 * DAY_MS]
    assert out["v"].to_list() == [1.0, 9.0, 3.0]


# --- dollar sizes beyond u10 ------------------------------------------------------------

def test_dollar_bootstrap_for_an_asset_outside_the_table_uses_its_measured_volume():
    table = load_dollar_thresholds()
    assert "SUIUSDT" not in table["assets"]
    size = dollar_threshold("SUIUSDT", table)
    assert size == pytest.approx(23323552 / 288)


def test_dollar_bootstrap_without_any_source_is_an_error():
    with pytest.raises(KeyError):
        dollar_threshold("NOTACOINUSDT", load_dollar_thresholds())


# --- runs: resume, incremental extension, listings, cohort -------------------------------

def _cfg(root, d1, d2, **kw):
    base = dict(
        start=d1, end=d2, assets=["BTCUSDT"], stages=["bars", "features", "dataset"],
        bar_types=["time_1m", "time_1h", "time_1d", "dollar"], data_root=root,
        windows=small_windows(), threshold_overrides={"BTCUSDT": 5000.0, "ETHUSDT": 5000.0, "SOLUSDT": 5000.0},
    )
    base.update(kw)
    return RunConfig(**base)


def _seed(root, asset, days, offset=0):
    for i, d in enumerate(days):
        write_bronze_day(root, asset, d, _day(i + 1 + offset, 100.0 + i, d))


def _hashes(root, sub):
    return {str(p.relative_to(root)): file_sha256(p) for p in (root / sub).rglob("*.parquet")}


def test_extending_the_end_does_not_rebuild_the_dollar_chain_from_the_start(tmp_path, monkeypatch):
    """2.3.0: a catch-up run (end moved inside the last month) rebuilt every dollar month."""
    days = [date(2024, 4, 29) + timedelta(days=i) for i in range(5)]  # Apr 29 .. May 3
    _seed(tmp_path, "BTCUSDT", days)
    assert run(_cfg(tmp_path, days[0], date(2024, 5, 2))) == 0
    april = tmp_path / "silver" / "bars" / "dollar" / "BTCUSDT_2024-04.parquet"
    before = file_sha256(april)
    seen = []
    real = chunked.build_crypto_month

    def spy(cfg, asset, first, last, acc):
        seen.append((first, acc is not None))
        return real(cfg, asset, first, last, acc)

    monkeypatch.setattr(chunked, "build_crypto_month", spy)
    assert run(_cfg(tmp_path, days[0], days[-1])) == 0
    assert file_sha256(april) == before
    assert all(first.month == 5 for first, _ in seen), seen
    # And the result equals a clean build of the longer range.
    clean = tmp_path.parent / "clean"
    _seed(clean, "BTCUSDT", days)
    monkeypatch.setattr(chunked, "build_crypto_month", real)
    assert run(_cfg(clean, days[0], days[-1])) == 0
    assert _hashes(tmp_path, "silver/bars/dollar") == _hashes(clean, "silver/bars/dollar")


def test_a_refetched_bronze_day_rebuilds_bars_features_and_gold(tmp_path):
    days = [date(2024, 5, 1), date(2024, 5, 2)]
    _seed(tmp_path, "BTCUSDT", days)
    cfg = _cfg(tmp_path, days[0], days[-1], bar_types=["time_1m", "time_1h"])
    assert run(cfg) == 0
    gold = tmp_path / "gold" / "time_1h" / "BTCUSDT_2024-05.parquet"
    feats = tmp_path / "silver" / "features" / "time_1h" / "BTCUSDT_2024-05.parquet"
    before = (file_sha256(gold), file_sha256(feats))
    write_bronze_day(tmp_path, "BTCUSDT", days[1], _day(99, 120.0, days[1]))  # a corrected day
    assert run(cfg) == 0
    assert file_sha256(feats) != before[1]
    assert file_sha256(gold) != before[0]


def test_an_asset_listed_mid_run_does_not_break_the_cohort_month(tmp_path):
    """2.3.0: clip_month raised for SOL in July 2020 and no asset got July gold."""
    july = [date(2020, 7, 30), date(2020, 7, 31)]
    aug = [date(2020, 8, 11), date(2020, 8, 12)]  # SOL lists 2020-08-11
    _seed(tmp_path, "BTCUSDT", july + aug)
    _seed(tmp_path, "SOLUSDT", aug, offset=50)
    cfg = _cfg(tmp_path, july[0], aug[-1], assets=["BTCUSDT", "SOLUSDT"], bar_types=["time_1h"])
    assert run(cfg) == 0
    gold = tmp_path / "gold" / "time_1h"
    assert (gold / "BTCUSDT_2020-07.parquet").exists()
    assert (gold / "SOLUSDT_2020-08.parquet").exists()
    assert not (gold / "SOLUSDT_2020-07.parquet").exists()
    man = json.loads((gold / "BTCUSDT_2020-08.manifest.json").read_text())
    assert man["cohort"] == ["BTCUSDT", "SOLUSDT"]


def test_cross_asset_peers_include_members_whose_gold_is_already_fresh(tmp_path):
    """2.3.0: xd_* used only the stale assets as peers, and the cohort was not in params."""
    days = [date(2024, 5, 1), date(2024, 5, 2)]
    _seed(tmp_path, "BTCUSDT", days)
    _seed(tmp_path, "ETHUSDT", days, offset=30)
    one = _cfg(tmp_path, days[0], days[-1], bar_types=["time_1h"])
    assert run(one) == 0
    gold = tmp_path / "gold" / "time_1h" / "BTCUSDT_2024-05.parquet"
    alone = pl.read_parquet(gold)
    assert all(alone[c].null_count() == alone.height for c in alone.columns if c.startswith("xd_"))
    both = _cfg(tmp_path, days[0], days[-1], assets=["BTCUSDT", "ETHUSDT"], bar_types=["time_1h"])
    assert run(both) == 0
    joint = pl.read_parquet(gold)
    xd = [c for c in joint.columns if c.startswith("xd_") and c not in ("xd_btc_return", "xd_btc_volatility")]
    assert xd and any(joint[c].drop_nulls().len() for c in xd)


def test_fx_rerun_of_a_finished_range_is_not_a_failure(tmp_path):
    """2.3.0: a cached FX month reported 0 ticks, so the repeat run failed with exit 2."""
    from pipeline.paths import fx_hour_path

    start = datetime(2024, 1, 15, tzinfo=timezone.utc)
    rows = []
    px = 1.10
    import numpy as np
    rng = np.random.default_rng(3)
    for i in range(6 * 60):
        px *= 1.0 + float(rng.normal(0, 2e-5))
        ms = int((start + timedelta(minutes=i, seconds=20)).timestamp() * 1000)
        rows.append({"ts_ms": ms, "bid": px - 1e-4, "ask": px + 1e-4, "mid": px, "spread": 2e-4})
    ticks = pl.DataFrame(rows)
    for hour in range(6):
        h0 = int((start + timedelta(hours=hour)).timestamp() * 1000)
        chunk = ticks.filter((pl.col("ts_ms") >= h0) & (pl.col("ts_ms") < h0 + 3_600_000))
        atomic_write_parquet(chunk, fx_hour_path(tmp_path, "EURUSD", start.date(), hour),
                             schema_name="fx_ticks_v1", params={"pair": "EURUSD", "hour": hour})
    cfg = dict(start=date(2024, 1, 15), end=date(2024, 1, 15), assets=["EURUSD"], market="fx",
               stages=["bars", "features", "dataset", "validate"], bar_types=["time_1m", "time_1h"],
               data_root=tmp_path, skip_network=True)
    first = run(RunConfig(**cfg))
    second = run(RunConfig(**cfg))
    assert first < 2
    assert second == first


# --- validator pieces ---------------------------------------------------------------------

def test_expected_asof_uses_the_given_lag():
    from pipeline.validate_run import _expected_asof

    events = [0, DAY_MS, 2 * DAY_MS]
    vals = [1.0, 2.0, 3.0]
    lag = 2 * DAY_MS
    ts = [2 * DAY_MS - 1, 2 * DAY_MS, 3 * DAY_MS, 10 * DAY_MS]
    assert _expected_asof(ts, events, vals, lag) == [None, 1.0, 2.0, 3.0]


def test_fx_weekend_rule_covers_the_sunday_reopen_bar():
    from pipeline.validate_run import _fx_weekend_expected

    # Sunday 2024-01-14 17:00 America/New_York = 22:00 UTC (EST).
    reopen = int(datetime(2024, 1, 14, 22, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert _fx_weekend_expected(reopen)
    assert not _fx_weekend_expected(reopen + 60_000)


def test_unknown_bar_types_are_rejected_not_silently_skipped(tmp_path):
    with pytest.raises(ValueError):
        run(_cfg(tmp_path, date(2024, 5, 1), date(2024, 5, 1), bar_types=["time_1m", "tick"]))


def test_gaps_tell_venue_downtime_from_lost_data(tmp_path):
    """A hole the venue's own klines also lack is downtime (pass); a traded minute we lack fails."""
    from pipeline.paths import klines_1m_path
    from pipeline.validate_run import Report, check_gaps

    day = date(2020, 2, 19)
    t0 = int(datetime(2020, 2, 19, tzinfo=timezone.utc).timestamp() * 1000)
    closes = [t0 + 60_000 * (i + 1) for i in range(120)]
    hole = set(closes[30:60])  # 30 minutes missing from our bars
    bars = pl.DataFrame({"ts_ms": [c for c in closes if c not in hole]})

    def klines(traded_in_hole: bool):
        opens = [c - 60_000 for c in closes]
        n = [1 if (c not in hole or traded_in_hole) else 0 for c in closes]
        df = pl.DataFrame({"ts_open_ms": opens, "n_trades": n})
        atomic_write_parquet(df, klines_1m_path(tmp_path, "BTCUSDT", day), schema_name="kline_1m_v1", params={})

    klines(traded_in_hole=False)
    r = Report()
    check_gaps(r, {"BTCUSDT": {"time_1m": bars}}, tmp_path, day, day)
    assert r.checks[0].severity == "PASS" and "venue downtime" in r.checks[0].detail
    klines(traded_in_hole=True)
    r = Report()
    check_gaps(r, {"BTCUSDT": {"time_1m": bars}}, tmp_path, day, day)
    assert r.checks[0].severity == "FAIL"


# --- review round 2 -------------------------------------------------------------------------

def test_interrupted_dollar_cascade_does_not_keep_months_from_the_old_chain(tmp_path, monkeypatch):
    """A corrected bronze day in an early month must rebuild every later dollar month, even
    when the first rebuild is interrupted after the early month is written."""
    days = [date(2024, 3, 30), date(2024, 3, 31), date(2024, 4, 1), date(2024, 4, 2), date(2024, 5, 1)]
    _seed(tmp_path, "BTCUSDT", days)
    cfg = _cfg(tmp_path, days[0], days[-1], bar_types=["time_1m", "dollar"], stages=["bars"])
    assert run(cfg) == 0
    write_bronze_day(tmp_path, "BTCUSDT", days[1], _day(77, 130.0, days[1]))  # March corrected
    real = chunked.write_bars_month
    calls = {"n": 0}

    def boom(*a, **k):
        out = real(*a, **k)
        calls["n"] += 1
        if calls["n"] == 1:  # March rebuilt, then Ctrl+C before April
            raise KeyboardInterrupt()
        return out

    import pipeline.orchestrator as orch
    monkeypatch.setattr(orch, "write_bars_month", boom)
    with pytest.raises(KeyboardInterrupt):
        run(cfg)
    monkeypatch.setattr(orch, "write_bars_month", real)
    assert run(cfg) == 0
    clean = tmp_path.parent / "clean_cascade"
    _seed(clean, "BTCUSDT", days)
    write_bronze_day(clean, "BTCUSDT", days[1], _day(77, 130.0, days[1]))
    assert run(_cfg(clean, days[0], days[-1], bar_types=["time_1m", "dollar"], stages=["bars"])) == 0
    assert _hashes(tmp_path, "silver/bars/dollar") == _hashes(clean, "silver/bars/dollar")


def test_a_snapshot_whose_partition_was_rewritten_is_refused(tmp_path):
    days = [date(2024, 3, 31), date(2024, 4, 1)]
    _seed(tmp_path, "BTCUSDT", days)
    cfg = _cfg(tmp_path, days[0], days[-1], bar_types=["time_1m", "dollar"], stages=["bars"])
    assert run(cfg) == 0
    spec = chunked.dollar_spec_for(cfg, "BTCUSDT")
    assert chunked.read_dollar_state(tmp_path, "BTCUSDT", spec, through="2024-03") is not None
    part = tmp_path / "silver" / "bars" / "dollar" / "BTCUSDT_2024-03.parquet"
    df = pl.read_parquet(part)
    atomic_write_parquet(df.head(max(df.height - 1, 0)), part, schema_name="bars_v1", params={})
    assert chunked.read_dollar_state(tmp_path, "BTCUSDT", spec, through="2024-03") is None


def test_a_recent_month_without_a_monthly_zip_falls_back_to_daily_files(tmp_path, monkeypatch):
    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 4)  # October ended four days ago

    monkeypatch.setattr(fc, "date", FakeDate)
    first = date(2026, 10, 1)
    daily = []
    monkeypatch.setattr(fc, "http_get_to_path", lambda url, dest, timeout=120: (404, "HTTP 404"))
    monkeypatch.setattr(fc, "http_get", lambda url, timeout=120: daily.append(url) or (404, b"", "HTTP 404"))
    fc.fetch_trades(tmp_path, "BTCUSDT", first, first + timedelta(days=2))
    assert len(daily) == 3  # the monthly zip may not be published yet: try each day
    man = json.loads((tmp_path / "bronze" / "_manifest" / "trades" / "BTCUSDT.json").read_text())
    # Each day is judged on its own 404 age; the month was not written off wholesale.
    assert sorted(man["artifacts"]) == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert all("/daily/" in v["source_url"] for v in man["artifacts"].values())


def test_an_old_month_without_a_monthly_zip_is_absent(tmp_path, monkeypatch):
    daily = []
    monkeypatch.setattr(fc, "http_get_to_path", lambda url, dest, timeout=120: (404, "HTTP 404"))
    monkeypatch.setattr(fc, "http_get", lambda url, timeout=120: daily.append(url) or (404, b"", "HTTP 404"))
    fc.fetch_trades(tmp_path, "BTCUSDT", date(2019, 3, 1), date(2019, 3, 3))
    assert daily == []
    man = json.loads((tmp_path / "bronze" / "_manifest" / "trades" / "BTCUSDT.json").read_text())
    assert {v["status"] for v in man["artifacts"].values()} == {"absent"}


def test_bronze_coverage_fails_on_a_day_missing_from_trades_and_klines(tmp_path):
    from pipeline.validate_run import Report, check_bronze_coverage

    for d in (date(2020, 3, 1), date(2020, 3, 3)):
        write_bronze_day(tmp_path, "BTCUSDT", d, _day(1, 100.0, d))
    r = Report()
    check_bronze_coverage(r, tmp_path, "BTCUSDT", date(2020, 3, 1), date(2020, 3, 3))
    assert r.checks[0].severity == "FAIL" and "2020, 3, 2" in r.checks[0].detail


def test_fx_embargo_covers_the_fixed_100_bar_windows(tmp_path):
    from pipeline.features import FX_MAX_WINDOW
    cfg = RunConfig(start=date(2024, 1, 1), end=date(2024, 1, 2), assets=["EURUSD"], market="fx",
                    data_root=tmp_path, windows=load_windows())
    n, pad = chunked._embargo_padding(cfg, "EURUSD", "time_1d", load_windows(), 7, 90)
    assert n >= FX_MAX_WINDOW + 1
    assert pad * 5 >= n * 7  # padding in calendar days covers n trading days


# ---------------------------------------------------------------- 2.4.1: found by the u10 2020 build

CADENCES = ["time_1m", "time_5m", "time_30m", "time_1h", "time_2h", "time_4h", "time_6h", "time_8h", "time_1d"]


@pytest.mark.parametrize("cadence", CADENCES)
def test_every_feature_starts_within_the_declared_lookback(cadence):
    # The embargo is max_lookback bars. A feature whose first value comes later
    # (2.4.0 kurtosis at 1d: bar 131 of a declared 90) reaches past the embargo.
    from pipeline.bars import trades_to_time
    from pipeline.config import load_windows, scale_windows
    from pipeline.features import compute_physics
    from fixtures import make_trades

    w = scale_windows(load_windows(), cadence)
    n = w.max_lookback + 80
    trades = make_trades(n * 3, start=datetime(2023, 1, 1, 0, 0, 10), step=timedelta(seconds=20),
                         seed=5, asset_noise=0.002)
    # Drop a random third of the prints so arrival counts vary (a constant count has no z-score).
    keep = np.random.default_rng(9).random(trades.height) > 0.33
    bars = trades_to_time(trades.filter(pl.Series(keep)), 60_000, "BTCUSDT", "time_1m")
    bars = bars.with_columns((0.0001 * ((pl.int_range(pl.len()) % 7) - 3)).alias("funding_rate"))
    out = compute_physics(bars, w)
    late, dead = {}, []
    for c in out.columns:
        if not c.startswith(("norm_", "hurst_regime", "regime_label")):
            continue
        idx = out[c].is_not_null().arg_true()
        if len(idx) == 0:
            dead.append(c)
        elif int(idx[0]) > w.max_lookback:
            late[c] = int(idx[0])
    assert not late, f"{cadence}: first value past max_lookback={w.max_lookback}: {late}"
    assert not dead, f"{cadence}: never produced a value: {dead}"


def test_lookahead_correlation_fails_only_when_significant():
    from pipeline.validate import gate_lookahead

    rng = np.random.default_rng(0)

    def frame(n, k):
        y = rng.normal(size=n)
        return pl.DataFrame({"norm_ma_distance": k * y + rng.normal(size=n), "target_return_1": y})

    small = gate_lookahead(frame(40, 0.8), ["norm_ma_distance"])   # |r| ~ 0.6, z < 5: noise-sized sample
    assert [r.severity for r in small if r.name == "lookahead" and "norm_ma" in r.detail] == ["warn"]
    big = gate_lookahead(frame(2000, 0.35), ["norm_ma_distance"])  # |r| ~ 0.33, z ~ 15
    assert any(r.severity == "fail" for r in big)


def test_an_all_null_slow_feature_on_a_late_listing_is_warmup():
    from pipeline.validate import gate_constants

    def frame(n, warm):
        return pl.DataFrame({
            "norm_return_kurtosis": pl.Series([None] * n, dtype=pl.Float64),
            "is_warmup": [True] * warm + [False] * (n - warm),
        })

    assert all(r.severity != "fail" for r in gate_constants(frame(101, 90), ["norm_return_kurtosis"], min_history=90))
    assert any(r.severity == "fail" for r in gate_constants(frame(400, 90), ["norm_return_kurtosis"], min_history=90))


def _kline_day(tmp_path, base, drop=(), alter=()):
    from pipeline.paths import klines_1m_path

    k = base.select([
        (pl.col("ts_ms") - 60_000).alias("ts_open_ms"), "open", "high", "low", "close", "volume",
        pl.col("volume_usd").alias("quote_volume"), pl.col("n_trades"),
    ])
    k = k.filter(~pl.col("ts_open_ms").is_in([t - 60_000 for t in drop]))
    k = k.with_columns(
        pl.when(pl.col("ts_open_ms").is_in([t - 60_000 for t in alter])).then(pl.col("close") * 1.01)
        .otherwise(pl.col("close")).alias("close")
    )
    path = klines_1m_path(tmp_path, "SOLUSDT", date(2020, 12, 21))
    path.parent.mkdir(parents=True, exist_ok=True)
    k.write_parquet(path)


def _minute_bars(n=1440):
    t0 = int(datetime(2020, 12, 21, tzinfo=timezone.utc).timestamp() * 1000)
    ts = [t0 + 60_000 * (i + 1) for i in range(n)]
    px = [10.0 + 0.001 * i for i in range(n)]
    return pl.DataFrame({
        "ts_ms": ts, "open": px, "high": px, "low": px, "close": px,
        "volume": [1.0] * n, "volume_usd": px, "n_trades": [3] * n,
    })


def test_a_venue_kline_outage_warns_and_scattered_disagreement_fails(tmp_path):
    from pipeline.validate_run import Report, check_klines

    base = _minute_bars()
    ts = base["ts_ms"].to_list()
    hole = ts[800:827]                       # 27 traded minutes the kline file lacks
    _kline_day(tmp_path, base, drop=hole, alter=[ts[799], ts[827]])  # partial candles on both edges
    rep = Report()
    check_klines(rep, tmp_path, {"SOLUSDT": {"time_1m": base}})
    assert [c.severity for c in rep.checks] == ["WARN"], rep.checks
    assert "1 kline-file outage" in rep.checks[0].detail

    _kline_day(tmp_path, base, alter=ts[100:110])  # ten scattered mismatches in one day: an aggregation bug
    rep = Report()
    check_klines(rep, tmp_path, {"SOLUSDT": {"time_1m": base}})
    assert [c.severity for c in rep.checks] == ["FAIL"], rep.checks


def _one_day_run(tmp_path, stages):
    from pipeline.orchestrator import RunConfig, run
    from fixtures import small_windows, write_bronze_day
    from test_run import _bursty_day

    day = date(2024, 6, 1)
    write_bronze_day(tmp_path, "BTCUSDT", day, _bursty_day(11, 100.0, 0.2))
    cfg = RunConfig(
        start=day, end=day, assets=["BTCUSDT"], stages=stages,
        bar_types=["time_1m", "time_1h"], data_root=tmp_path, force=True,
        windows=small_windows(), threshold_overrides={"BTCUSDT": 50.0}, skip_network=True,
    )
    run(cfg)
    return cfg, day


def test_the_causality_gate_passes_a_clean_build_and_catches_a_planted_leak(tmp_path, monkeypatch):
    import pipeline.orchestrator as orch

    cfg, day = _one_day_run(tmp_path / "clean", ["bars", "features"])
    clean = orch._causality_gate(cfg, "BTCUSDT", "time_1m", day)
    assert clean is not None and clean.severity == "pass", clean

    honest = orch._attach_slow_features

    def leaky(cfg, asset, start, bars, daily):
        out = honest(cfg, asset, start, bars, daily)
        return {k: v.with_columns(pl.col("close").shift(-1).alias("norm_planted_leak")) for k, v in out.items()}

    monkeypatch.setattr(orch, "_attach_slow_features", leaky)
    cfg, day = _one_day_run(tmp_path / "leaky", ["bars", "features"])
    caught = orch._causality_gate(cfg, "BTCUSDT", "time_1m", day)
    assert caught is not None and caught.severity == "fail"
    assert "norm_planted_leak" in caught.detail
