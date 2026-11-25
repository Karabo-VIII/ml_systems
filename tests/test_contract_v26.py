"""Contract 2.6.0 regressions: the FX layer brought to the crypto layer's standard.

FX windows and embargo inside 90 calendar days, the vectorised and
deterministic tick decode, the polite fetcher (closed hours, adaptive pacing),
tick-digested bar freshness, currency strength, the daily realized-variance
panel and the carry (both as-of joined with their lags), and the ECB reference
cross-check, including a planted clock shift and a planted price-scale error.
Also the rebuild's idempotence: correctly rounded daily aggregates, and stable
sorts wherever the order of equal timestamps decides a value.
"""

from __future__ import annotations

import lzma
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

import pipeline.fetch_fx as fx
from pipeline.bars import fx_ticks_to_time
from pipeline.chunked import attach_fx_slow, write_fx_bars_month
from pipeline.config import lag_ms
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.crosscheck import (
    _judge_reference,
    compare_reference,
    ecb_pair_rate,
    fixing_ms,
    parse_ecb_exr_csv,
)
from pipeline.features import (
    compute_fx_features,
    fx_daily_panel,
    fx_embargo_bars,
    fx_lookback,
    fx_z_window,
    realized_day,
    rolling_z,
)
from pipeline.fetch_external import parse_bis_policy_csv
from pipeline.io import atomic_write_parquet
from pipeline.paths import fx_hour_path

UTC = timezone.utc


# --- windows and embargo ------------------------------------------------------------------

def test_fx_windows_and_embargo_stay_inside_90_calendar_days():
    for bt, width in TIME_WIDTH_MS.items():
        n = fx_embargo_bars(bt, (1, 4, 16, 64))
        assert n >= fx_lookback(bt) and n >= 64
        # FX bars are trading time: five trading days in seven calendar days.
        assert n * width * 7 / 5 <= 90 * DAY_MS + 1, bt
    assert fx_z_window("time_1d") == 63 and fx_z_window("time_1h") == 100
    assert fx_lookback("time_1d") == 63 and fx_lookback("time_1h") == 100


def test_spread_z_uses_the_cadence_window():
    n = 150
    rng = np.random.default_rng(1)
    ts = DAY_MS * np.arange(1, n + 1) + 1_577_836_800_000
    close = 1.1 + np.cumsum(rng.normal(0, 0.003, n))
    bars = pl.DataFrame({
        "bar_type": ["time_1d"] * n, "ts_open_ms": ts - DAY_MS, "ts_ms": ts,
        "open": close, "high": close + 0.002, "low": close - 0.002, "close": close,
        "n_trades": rng.integers(1000, 2000, n), "spread_mean": rng.uniform(1e-5, 3e-5, n),
    })
    out = compute_fx_features(bars, 1e-4)
    want = bars.select(rolling_z("spread_mean", 63).alias("z"))["z"]
    assert out["spread_z"].to_list() == want.to_list()
    assert "spread_z_100" not in out.columns and "tick_intensity_z" in out.columns


# --- fetch ----------------------------------------------------------------------------------

def _reference_decode(blob, hour_start_ms, div):
    raw = lzma.decompress(blob, format=lzma.FORMAT_ALONE)
    rows = []
    for off in range(0, len(raw), 20):
        ms, ai, bi, _, _ = fx._REC.unpack_from(raw, off)
        a, b = ai / div, bi / div
        rows.append((hour_start_ms + int(ms), b, a, (a + b) / 2.0, a - b))
    return pl.DataFrame(rows, schema=list(fx._TICK_SCHEMA), orient="row")


@pytest.mark.parametrize("div", [1e5, 1e3])
def test_vectorised_decode_equals_the_record_loop(div):
    rng = np.random.default_rng(2)
    ms = np.sort(rng.integers(0, 3_600_000, 5000))
    bid = 100_000 + rng.integers(0, 500, 5000)
    ask = bid + rng.integers(0, 30, 5000)
    raw = b"".join(fx._REC.pack(int(m), int(a), int(b), 1.0, 2.0) for m, a, b in zip(ms, ask, bid))
    blob = lzma.compress(raw, format=lzma.FORMAT_ALONE)
    assert fx.decode_bi5(blob, 1_577_836_800_000, div).equals(_reference_decode(blob, 1_577_836_800_000, div))


def test_closed_weekend_hours_are_cached_without_a_request(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fx, "http_get", lambda url, timeout=60: calls.append(url) or (404, b"", None))
    monkeypatch.setattr(fx._PACER, "wait", lambda m: None)
    sat = datetime(2020, 1, 4, 12, tzinfo=UTC)
    assert fx.fx_market_closed(sat)
    assert fx.fetch_hour(tmp_path, "EURUSD", sat).is_empty() and not calls
    from pipeline.io import read_parquet_metadata  # noqa: F401  (written atomically, with a sidecar)
    assert fx_hour_path(tmp_path, "EURUSD", sat.date(), 12).exists()
    # The edges are fetched: Friday 21:00 UTC (summer close at 21:00) and Sunday 20:00 UTC.
    for h in (datetime(2020, 7, 3, 21, tzinfo=UTC), datetime(2020, 1, 5, 20, tzinfo=UTC), datetime(2020, 1, 6, 9, tzinfo=UTC)):
        assert not fx.fx_market_closed(h)
        fx.fetch_hour(tmp_path, "EURUSD", h)
    assert len(calls) == 3


def test_pacer_backs_off_on_throttling_and_recovers_slowly():
    p = fx._Pacer()
    p.interval = 1.0
    p.throttled()
    assert p.interval == 2.0
    for _ in range(10):
        p.throttled()
    assert p.interval == fx._Pacer.MAX_INTERVAL_S
    p.answered(1.0)
    assert p.interval == pytest.approx(60.0 * 0.85)
    for _ in range(100):
        p.answered(1.0)
    assert p.interval == 1.0  # never below the configured floor


def test_ticks_in_the_same_millisecond_keep_feed_order():
    t0 = 1_577_952_000_000  # 2020-01-02 08:00 UTC
    ticks = pl.DataFrame({
        "ts_ms": [t0 + 5, t0 + 5, t0 + 5],
        "bid": [1.0, 2.0, 3.0], "ask": [1.0002, 2.0002, 3.0002],
    }).with_columns([((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread")])
    for _ in range(5):
        bar = fx_ticks_to_time(ticks, 60_000, "EURUSD", "time_1m")
        assert bar["open"][0] == pytest.approx(1.0001) and bar["close"][0] == pytest.approx(3.0001)


def test_a_refetched_hour_rebuilds_the_bar_month(tmp_path):
    from pipeline.orchestrator import RunConfig
    from pipeline.paths import bars_month_path
    from pipeline.io import read_parquet_metadata

    hour = datetime(2020, 1, 6, 9, tzinfo=UTC)

    def write_hour(px):
        ms = np.arange(0, 3_600_000, 10_000)
        df = pl.DataFrame({"ts_ms": int(hour.timestamp() * 1000) + ms, "bid": np.full(len(ms), px),
                           "ask": np.full(len(ms), px + 1e-4)}).with_columns([
            ((pl.col("bid") + pl.col("ask")) / 2).alias("mid"), (pl.col("ask") - pl.col("bid")).alias("spread")])
        atomic_write_parquet(df, fx_hour_path(tmp_path, "EURUSD", hour.date(), 9), schema_name="fx_ticks_v1", params={})

    cfg = RunConfig(start=date(2020, 1, 6), end=date(2020, 1, 6), assets=["EURUSD"], market="fx",
                    data_root=tmp_path, bar_types=["time_1m", "time_1h"])
    write_hour(1.10)
    assert not write_fx_bars_month(cfg, "EURUSD", 2020, 1)["cached"]
    path = bars_month_path(tmp_path, "time_1h", "EURUSD", 2020, 1)
    first = read_parquet_metadata(path)["params_hash"]
    assert write_fx_bars_month(cfg, "EURUSD", 2020, 1)["cached"]
    write_hour(1.12)  # the hour is refetched with different ticks
    assert not write_fx_bars_month(cfg, "EURUSD", 2020, 1)["cached"]
    assert read_parquet_metadata(path)["params_hash"] != first
    assert pl.read_parquet(path)["close"][0] == pytest.approx(1.12005)


# --- slow series ------------------------------------------------------------------------------

def _minutes(day: date, n: int = 300, seed: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    t0 = int(datetime(day.year, day.month, day.day, 8, tzinfo=UTC).timestamp() * 1000)
    ts = t0 + 60_000 * np.arange(1, n + 1)
    return pl.DataFrame({"ts_open_ms": ts - 60_000, "ts_ms": ts, "close": 1.1 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))})


def test_daily_panel_is_realized_variance_per_utc_day():
    one_m = pl.concat([_minutes(date(2020, 1, 6)), _minutes(date(2020, 1, 7), seed=4)])
    panel = fx_daily_panel(one_m, "EURUSD")
    assert panel.height == 2 and panel["n_bars"].to_list() == [300, 300]
    want = realized_day(_minutes(date(2020, 1, 6)))
    assert panel["rv_realized"][0] == pytest.approx(want["rv_realized"], rel=1e-12)


def test_slow_series_join_with_their_lags_and_carry_is_base_minus_quote():
    d6 = int(datetime(2020, 1, 6, tzinfo=UTC).timestamp() * 1000)
    bars = pl.DataFrame({"ts_ms": [d6 + 12 * 3_600_000, d6 + DAY_MS, d6 + DAY_MS + 3_600_000]})
    daily = pl.DataFrame({"event_time_ms": [d6], "rv_realized": [4e-6], "rv_bpv": [3e-6], "rv_jv": [1e-6], "rv_jump_frac": [0.25]})
    policy = pl.DataFrame({"area": ["XM", "US", "XM", "US"], "event_time_ms": [d6 - DAY_MS, d6 - DAY_MS, d6, d6],
                           "rate_pct": [0.0, 1.625, 0.0, 1.875]})
    out = attach_fx_slow(bars, "EURUSD", daily, policy)
    # Day 6's realized variance is knowable at day 7 00:00, not during day 6.
    assert out["rv_realized"].to_list() == [None, 4e-6, 4e-6]
    # EUR - USD; the day-6 rate (1.875) is only knowable from day 7.
    assert out["carry_pct"].to_list() == pytest.approx([-1.625, -1.875, -1.875])
    assert lag_ms("policy_rate") == DAY_MS
    missing = attach_fx_slow(bars, "EURUSD", daily, policy.filter(pl.col("area") == "US"))
    assert missing["carry_pct"].null_count() == 3


def test_bis_and_ecb_parsers():
    bis = parse_bis_policy_csv("FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nD,US,2020-01-02,1.625\nD,JP,2020-01-02,-0.1\nD,JP,2020-01-03,\n")
    assert bis.height == 2 and bis.filter(pl.col("area") == "JP")["rate_pct"][0] == -0.1
    ecb = parse_ecb_exr_csv("KEY,CURRENCY,TIME_PERIOD,OBS_VALUE\nX,USD,2020-01-06,1.1194\nX,JPY,2020-01-06,121.02\nX,GBP,2020-01-06,0.85173\n")
    assert ecb_pair_rate(ecb, "EURUSD")["rate"][0] == pytest.approx(1.1194)
    assert ecb_pair_rate(ecb, "USDJPY")["rate"][0] == pytest.approx(121.02 / 1.1194)
    assert ecb_pair_rate(ecb, "GBPUSD")["rate"][0] == pytest.approx(1.1194 / 0.85173)


# --- ECB reference cross-check ------------------------------------------------------------------

def _fx_minutes_and_reference(shift_ms: int = 0, scale: float = 1.0, *, days=None, shift_regime=None, trend=True):
    """1m mids over weekdays (with an intraday trend unless ``trend=False``), and the
    'ECB' rate equal to the true mid at 14:15 Frankfurt each day. ``shift_ms`` moves
    our stamps on every day, or only on days of ``shift_regime`` (CET or CEST)."""
    from pipeline.crosscheck import fixing_regime
    days = days or [date(2020, 1, 6) + timedelta(days=i) for i in range(5)]
    rows_ts, rows_px, ref = [], [], []
    for d in days:
        t0 = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)
        ts = t0 + 60_000 * np.arange(1, 1441)
        px = 1.10 + (0.0001 * np.arange(1440) / 10 if trend else 0.0)  # +1 pip every 10 minutes
        px = np.broadcast_to(px, (1440,)).astype(float)
        moved = shift_regime is None or fixing_regime(d) == shift_regime
        rows_ts.append(ts + (shift_ms if moved else 0))
        rows_px.append(px * scale)
        fix = fixing_ms(d)
        ref.append((d, float(px[np.searchsorted(ts, fix, side="right") - 1])))
    minutes = pl.DataFrame({"ts_ms": np.concatenate(rows_ts), "close": np.concatenate(rows_px)})
    reference = pl.DataFrame({"day": [r[0] for r in ref], "rate": [r[1] for r in ref]})
    return minutes, reference


def test_ecb_check_passes_aligned_data():
    minutes, ref = _fx_minutes_and_reference()
    s = compare_reference(minutes, ref, 1e-4)
    assert s["days"] == 5 and s["median_pips"] == pytest.approx(0.0, abs=1e-6) and s["best_offset_min"] == 0
    assert _judge_reference(s) == "PASS"


def test_ecb_check_fails_a_one_hour_clock_shift():
    """The daylight-saving bug class: bars stamped an hour late match the reference
    better at an offset than at the fixing itself."""
    minutes, ref = _fx_minutes_and_reference(shift_ms=3_600_000)
    s = compare_reference(minutes, ref, 1e-4)
    assert s["best_offset_min"] == 60 and _judge_reference(s) == "FAIL"


def test_ecb_check_fails_a_price_scale_error_by_level():
    """On flat prices no offset matches better, so the level threshold alone must fail it."""
    minutes, ref = _fx_minutes_and_reference(scale=1.001, trend=False)
    s = compare_reference(minutes, ref, 1e-4)
    assert s["shifted_to_min"] is None and s["median_pips"] > 10
    assert _judge_reference(s) == "FAIL"


def test_ecb_check_fails_a_shift_confined_to_one_dst_regime():
    """A clock error on the winter (CET) days only: the pooled median barely moves,
    so each daylight-saving regime is judged alone."""
    days = [date(2020, 3, 16) + timedelta(days=i) for i in range(28)]
    days = [d for d in days if d.weekday() < 5]  # CET until 2020-03-29, CEST after
    minutes, ref = _fx_minutes_and_reference(days=days, shift_ms=3_600_000, shift_regime="CET", trend=True)
    s = compare_reference(minutes, ref, 1e-4)
    assert s["regimes"]["CET"]["shifted_to_min"] == 60 and s["regimes"]["CEST"]["shifted_to_min"] is None
    assert _judge_reference(s) == "FAIL"


def test_ecb_outage_is_a_warning_not_a_crash(tmp_path, monkeypatch):
    import pipeline.crosscheck as cc
    from pipeline.validate_run import Report
    from collections import defaultdict

    monkeypatch.setattr(cc, "http_get", lambda url, timeout=60: (503, b"", "HTTP 503"))
    report = Report()
    cc.fx_crosscheck(tmp_path, ["EURUSD"], date(2020, 1, 6), date(2020, 1, 10), report, defaultdict(dict))
    assert [(c.name, c.severity) for c in report.checks] == [("ecb_reference", "WARN")]
    # A weekend-only range is answered 200 with no body: no fixing day, not an error.
    monkeypatch.setattr(cc, "http_get", lambda url, timeout=60: (200, b"", None))
    assert cc.ecb_reference(tmp_path, date(2020, 1, 4), date(2020, 1, 5)).is_empty()


def test_bis_nan_days_never_reach_the_carry():
    bis = parse_bis_policy_csv("FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE\nD,NZ,2020-01-03,1.0\nD,NZ,2020-01-04,NaN\n"
                               "D,NZ,2020-01-05,NaN\nD,US,2020-01-03,1.625\nD,US,2020-01-04,1.625\nD,US,2020-01-05,1.625\n")
    assert bis.filter(pl.col("area") == "NZ").height == 1
    d6 = int(datetime(2020, 1, 6, tzinfo=UTC).timestamp() * 1000)
    bars = pl.DataFrame({"ts_ms": [d6 + 3_600_000]})
    # Even a stored NaN row (an older fetch) is ignored: the last finite rate carries.
    stored = pl.concat([bis, pl.DataFrame({"area": ["NZ"], "event_time_ms": [d6 - DAY_MS], "rate_pct": [float("nan")]})])
    out = attach_fx_slow(bars, "NZDUSD", None, stored)
    assert out["carry_pct"].to_list() == pytest.approx([1.0 - 1.625])


def test_dead_carry_and_weekend_gap_are_reported_but_not_their_legitimate_nulls(tmp_path):
    from pipeline import validate_run as vr

    def gold(n_days: int, carry, gap):
        n = n_days
        start = datetime(2020, 1, 6, tzinfo=UTC)  # a Monday
        ts = [int((start + timedelta(days=i + 1)).timestamp() * 1000) for i in range(n)]
        # A gap is set on the bar after each weekend only, and differs from weekend to weekend.
        gaps = [(gap + i / 7) if gap is not None and i % 7 == 6 else None for i in range(n)]
        frame = pl.DataFrame({"asset": ["EURUSD"] * n, "bar_type": ["time_1d"] * n, "ts_ms": ts,
                              "close": list(np.linspace(1.1, 1.2, n)), "carry_pct": [carry] * n,
                              "weekend_gap_pips": gaps},
                             schema_overrides={"carry_pct": pl.Float64, "weekend_gap_pips": pl.Float64})
        path = tmp_path / f"gold_{n_days}_{carry}_{gap}" / "time_1d" / "EURUSD_2020-01.parquet"
        atomic_write_parquet(frame, path, schema_name="dataset_v1", params={}, sort_cols=["ts_ms"])
        r = vr.Report()
        vr.check_dead_columns(r, {"time_1d": {"EURUSD": path}})
        return r.checks[0]

    long_dead = gold(60, None, None)
    assert long_dead.severity == "WARN" and "carry_pct" in long_dead.detail and "weekend_gap_pips" in long_dead.detail
    steady = gold(60, -1.625, 3.0)  # a constant carry is a step series, not a dead one
    assert steady.severity == "PASS", steady.detail
    one_week = gold(4, -1.625, None)  # Tue..Fri: no weekend in range, so no gap is due
    assert one_week.severity == "PASS", one_week.detail


def _hourly_gold(tmp_path, name: str, days: int, extra: dict) -> "Path":
    start = datetime(2020, 1, 6, 1, tzinfo=UTC)  # Monday 01:00
    ts = [int((start + timedelta(hours=h)).timestamp() * 1000) for h in range(24 * days - 1)]
    n = len(ts)
    frame = pl.DataFrame({"asset": ["EURUSD"] * n, "bar_type": ["time_1h"] * n, "ts_ms": ts,
                          "close": list(np.linspace(1.1, 1.2, n)),
                          **{k: (v(n) if callable(v) else [v] * n) for k, v in extra.items()}})
    path = tmp_path / name / "time_1h" / "EURUSD_2020-01.parquet"
    atomic_write_parquet(frame, path, schema_name="dataset_v1", params={}, sort_cols=["ts_ms"])
    return path


def test_rv_n_bars_may_be_constant_only_inside_one_week(tmp_path):
    """Every full weekday's daily figure rests on 1440 one-minute bars; only the Sunday
    session's is shorter. A Monday-to-Friday range is constant for real; a range that
    holds a weekend and is still constant is not."""
    from pipeline import validate_run as vr
    from pipeline.validate import gate_constants

    for days, expected in ((5, "PASS"), (19, "WARN")):
        path = _hourly_gold(tmp_path, f"d{days}", days, {"rv_n_bars": 1440})
        frame = pl.read_parquet(path)
        gate = [g.severity for g in gate_constants(frame, ["rv_n_bars"])]
        assert gate == (["pass"] if expected == "PASS" else ["warn"]), (days, gate)
        r = vr.Report()
        vr.check_dead_columns(r, {"time_1h": {"EURUSD": path}})
        assert r.checks[0].severity == expected, (days, r.checks[0].detail)
        if expected == "WARN":
            assert "constant: rv_n_bars=1440" in r.checks[0].detail


def test_the_column_check_reports_dead_and_constant_columns_together(tmp_path):
    """A dead column used to hide every constant one in the same frame, and lists stopped
    at eight names without saying so."""
    from pipeline import validate_run as vr

    dead = {f"norm_dead_{i}": None for i in range(10)}
    path = _hourly_gold(tmp_path, "both", 19, {**{k: (lambda n: pl.Series([None] * n, dtype=pl.Float64))
                                                      for k in dead},
                                                   "norm_flat": 2.5})
    r = vr.Report()
    vr.check_dead_columns(r, {"time_1h": {"EURUSD": path}})
    detail = r.checks[0].detail
    assert r.checks[0].severity == "WARN"
    assert "all-null: " in detail and "(+2 more)" in detail
    assert "constant: norm_flat=2.5" in detail


def test_cross_asset_columns_do_not_depend_on_later_listings_or_frame_order():
    """A gold month's cohort includes a peer that lists later in the month (its column is
    null before that). polars' sum_horizontal rounds differently with one more column, so
    xd_* before the listing moved by an ulp: a truncation proof could not pass bit for bit.
    The means are now a left fold over the peers in name order."""
    from pipeline.features import XD_COLUMNS, add_cross_asset

    rng = np.random.default_rng(26)
    t0 = 1_577_836_800_000
    n = 120
    ts = t0 + DAY_MS * np.arange(1, n + 1)

    def asset(k: int, start: int = 0):
        close = 100.0 * (1 + k) * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
        return pl.DataFrame({"ts_ms": ts[start:], "close": close[start:],
                             "funding_rate": rng.normal(1e-4, 5e-5, n)[start:]})

    names = [f"A{k}USDT" for k in range(8)] + ["BTCUSDT"]
    frames = {a: asset(k) for k, a in enumerate(names)}
    late = asset(99, start=90)  # lists on bar 90
    with_late = add_cross_asset({**frames, "A0LATEUSDT": late}, bar_type="time_1d", fast=20, ma=20)
    without = add_cross_asset(frames, bar_type="time_1d", fast=20, ma=20)
    shuffled = add_cross_asset(dict(reversed(list({**frames, "A0LATEUSDT": late}.items()))),
                               bar_type="time_1d", fast=20, ma=20)
    for a in names:
        before = pl.col("ts_ms") < int(ts[90])
        x = with_late[a].filter(before).select(XD_COLUMNS)
        assert x.equals(without[a].filter(before).select(XD_COLUMNS)), a
        assert with_late[a].select(XD_COLUMNS).equals(shuffled[a].select(XD_COLUMNS)), a
    # The planted defect: with sum_horizontal the later listing does move the earlier bars.
    import pipeline.features as features_module

    real = features_module._fold_sum
    features_module._fold_sum = lambda exprs: pl.sum_horizontal([e.cast(pl.Float64).fill_null(0.0) for e in exprs])
    try:
        old_with = add_cross_asset({**frames, "A0LATEUSDT": late}, bar_type="time_1d", fast=20, ma=20)
        old_without = add_cross_asset(frames, bar_type="time_1d", fast=20, ma=20)
    finally:
        features_module._fold_sum = real
    before = pl.col("ts_ms") < int(ts[90])
    assert any(not old_with[a].filter(before).select(XD_COLUMNS).equals(old_without[a].filter(before).select(XD_COLUMNS))
               for a in names), "the test no longer exercises the column-count rounding it guards against"


def test_currency_strength_does_not_depend_on_the_order_of_the_pairs():
    from pipeline.features import add_currency_strength

    rng = np.random.default_rng(27)
    ts = 1_578_268_800_000 + 3_600_000 * np.arange(1, 200)
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]
    frames = {p: pl.DataFrame({"ts_ms": ts, "log_ret_1": rng.normal(0, 1e-3, ts.size)}) for p in pairs}
    frames["USDCAD"] = frames["USDCAD"].with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("log_ret_1")).alias("log_ret_1"))
    a = add_currency_strength(frames)
    b = add_currency_strength(dict(reversed(list(frames.items()))))
    for p in pairs:
        assert a[p].select(["base_strength_ret", "quote_strength_ret"]).equals(
            b[p].select(["base_strength_ret", "quote_strength_ret"])), p


def test_the_run_log_keeps_every_finding_at_a_gates_worst_severity():
    from pipeline.orchestrator import _merge_gate
    from pipeline.validate import GateResult

    results: dict = {}
    _merge_gate(results, GateResult("constant", "pass", "ok"), "2020")
    _merge_gate(results, GateResult("constant", "warn", "a is constant"), "2020")
    _merge_gate(results, GateResult("constant", "warn", "b is constant"), "2020")
    assert results["constant"] == ("warn", ["[2020] a is constant", "[2020] b is constant"])
    _merge_gate(results, GateResult("constant", "fail", "close is constant"), "2021")
    _merge_gate(results, GateResult("constant", "warn", "c is constant"), "2021")
    _merge_gate(results, GateResult("constant", "fail", "open is constant"), "2022")
    assert results["constant"] == ("fail", ["[2021] close is constant", "[2022] open is constant"])


def test_a_lookback_cap_too_short_for_the_fx_windows_is_an_error():
    with pytest.raises(ValueError):
        fx_lookback("time_1d", 60)


def test_fx_gold_says_how_many_bars_a_daily_figure_rests_on():
    d6 = int(datetime(2020, 1, 6, tzinfo=UTC).timestamp() * 1000)
    bars = pl.DataFrame({"ts_ms": [d6 + DAY_MS + 3_600_000]})
    daily = pl.DataFrame({"event_time_ms": [d6], "rv_realized": [4e-6], "rv_bpv": [3e-6], "rv_jv": [1e-6],
                          "rv_jump_frac": [0.25], "n_bars": [150]})
    assert attach_fx_slow(bars, "EURUSD", daily, pl.DataFrame())["rv_n_bars"].to_list() == [150]


# --- idempotence -------------------------------------------------------------------------------

def test_daily_aggregates_are_bitwise_identical_whatever_the_frame_layout():
    """The u10 rebuild under 2.6.0 changed the s3_* means by 1e-16: polars' group-by can
    sum the same group along a different path from run to run. The per-day sums are now
    correctly rounded (math.fsum), so neither the layout nor the row order can matter,
    not even among rows that share a timestamp."""
    import math

    from pipeline.features import funding_daily, metrics_daily

    rng = np.random.default_rng(16)
    t0 = 1_598_918_400_000  # 2020-09-01
    ts = t0 + 300_000 * np.arange(288 * 20)
    m = pl.DataFrame({"ts_ms": ts, "sum_open_interest_value": rng.uniform(1e9, 2e9, ts.size),
                      "count_long_short_ratio": rng.uniform(0.5, 2.0, ts.size)})
    m = m.with_columns(pl.when(pl.int_range(pl.len()) % 97 == 0).then(None)
                       .otherwise(pl.col("count_long_short_ratio")).alias("count_long_short_ratio"))
    parts = [m.slice(i, 777) for i in range(0, m.height, 777)]
    chunked_frame = pl.concat(parts, rechunk=False)
    a = metrics_daily(m, "BTCUSDT")
    for other in (chunked_frame, m.sample(fraction=1.0, shuffle=True, seed=1), m.reverse()):
        assert metrics_daily(other, "BTCUSDT").equals(a)
    # Exactly the correctly rounded daily sum over the defined values, divided by their count.
    day = (ts // DAY_MS).astype(np.int64)
    v = m["count_long_short_ratio"].to_numpy()
    expected = []
    for d in np.unique(day):
        x = v[(day == d) & np.isfinite(v)]
        expected.append(math.fsum(x) / x.size)
    assert np.array_equal(a["s3_count_long_short_ratio_mean"].to_numpy(), np.array(expected))
    # Rows sharing a timestamp: the sum cannot depend on which comes first.
    dup = pl.concat([m, m.with_columns(pl.col("sum_open_interest_value") * 1.000001)])
    assert metrics_daily(dup, "BTCUSDT").equals(metrics_daily(dup.reverse(), "BTCUSDT"))
    f = pl.DataFrame({"ts_ms": t0 + 28_800_000 * np.arange(60), "funding_rate": rng.normal(1e-4, 5e-5, 60)})
    assert funding_daily(f.reverse(), "BTCUSDT").equals(funding_daily(f, "BTCUSDT"))
    assert funding_daily(f, "BTCUSDT")["funding_day_last"][0] == f["funding_rate"][2]
    fs = f["funding_rate"].to_numpy()
    assert funding_daily(f, "BTCUSDT")["funding_day_sum"][0] == math.fsum(fs[:3])


def _unstable_keep_last_chains(source: str) -> tuple[int, list[int]]:
    """Every unique(keep="last") whose row order comes from a sort: (count, lines where the sort is unstable).

    The receiver chain is followed through order-preserving calls, and a bare name is
    resolved to its latest assignment above the call in the same function.
    """
    import ast

    passthrough = {"with_columns", "select", "filter", "drop_nulls", "rename", "cast",
                   "with_row_index", "fill_null", "fill_nan", "head", "tail", "slice"}
    def own_nodes(scope):
        """The scope's nodes, not descending into the functions defined inside it."""
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            node = stack.pop()
            yield node
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stack.extend(ast.iter_child_nodes(node))

    tree = ast.parse(source)
    checked, bad = 0, []
    scopes = [n for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        assigns = [(n.lineno, t.id, n.value) for n in own_nodes(scope) if isinstance(n, ast.Assign)
                   for t in n.targets if isinstance(t, ast.Name)]
        for call in own_nodes(scope):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "unique"):
                continue
            keep = {k.arg: k.value for k in call.keywords}.get("keep")
            if not (isinstance(keep, ast.Constant) and keep.value == "last"):
                continue
            expr, line = call.func.value, call.lineno
            for _ in range(20):
                if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute):
                    if expr.func.attr == "sort":
                        checked += 1
                        stable = {k.arg: k.value for k in expr.keywords}.get("maintain_order")
                        if not (isinstance(stable, ast.Constant) and stable.value is True):
                            bad.append(expr.lineno)
                        break
                    if expr.func.attr in passthrough:
                        expr = expr.func.value
                        continue
                    break
                if isinstance(expr, ast.Name):
                    earlier = [(ln, val) for ln, name, val in assigns if name == expr.id and ln < line]
                    if not earlier:
                        break
                    line, expr = max(earlier, key=lambda p: p[0])
                    continue
                break
    return checked, sorted(set(bad))


def test_every_keep_last_dedup_follows_a_stable_sort():
    """unique(keep="last") after a sort means "the later row wins" only if the sort is
    stable. Polars promises tie order only with maintain_order=True (1.44 happens to keep
    it anyway), so every such chain in pipeline/ asks for it explicitly."""
    from pathlib import Path

    # The guard must catch a planted unstable chain, directly and through a name.
    planted = (
        "def f(df):\n"
        "    a = df.sort('ts_ms').unique(subset=['ts_ms'], keep='last')\n"
        "    out = df.sort('ts_ms').with_columns(x=1)\n"
        "    b = out.unique(subset=['ts_ms'], keep='last')\n"
        "    c = df.sort('ts_ms', maintain_order=True).unique(subset=['ts_ms'], keep='last')\n"
        "    d = df.unique(subset=['ts_ms'], keep='last')\n"
    )
    assert _unstable_keep_last_chains(planted) == (3, [2, 3])

    root = Path(__file__).resolve().parents[1] / "pipeline"
    total, offenders = 0, {}
    for path in sorted(root.glob("*.py")):
        n, bad = _unstable_keep_last_chains(path.read_text(encoding="utf-8"))
        total += n
        if bad:
            offenders[path.name] = bad
    assert not offenders, f"unstable sort feeding unique(keep='last'): {offenders}"
    assert total >= 12, f"the guard only saw {total} sorted keep-last chains; it may have gone blind"


def test_fx_tick_reader_keeps_the_shipped_order_of_ticks_sharing_a_millisecond(tmp_path):
    """The last tick of a bar is its close, and Dukascopy can ship several ticks in one
    millisecond. The reader and the validator keep the shipped order, as the bar builder does."""
    from pipeline.validate_run import _read_fx_ticks

    day = date(2024, 3, 5)
    h0 = int(datetime(2024, 3, 5, tzinfo=timezone.utc).timestamp() * 1000)
    frames = []
    for hour in range(3):
        n = 3_000
        ts = h0 + hour * 3_600_000 + np.repeat(np.arange(30) * 1_000, n // 30)
        seq = hour * n + np.arange(n)
        mid = 1.08 + seq * 1e-7
        f = pl.DataFrame({"ts_ms": ts, "bid": mid - 5e-6, "ask": mid + 5e-6, "mid": mid,
                          "spread": np.full(n, 1e-5)})
        path = fx_hour_path(tmp_path, "EURUSD", day, hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        f.write_parquet(path)
        frames.append(f)
    shipped = pl.concat(frames)["mid"].to_numpy()
    assert np.array_equal(fx.read_fx_ticks(tmp_path, "EURUSD", day, day)["mid"].to_numpy(), shipped)
    assert np.array_equal(_read_fx_ticks(tmp_path, "EURUSD", [day])["mid"].to_numpy(), shipped)
