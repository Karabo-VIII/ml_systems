"""Contract 2.5.0 regressions.

Each corrected feature is recomputed here from its published formula (numpy,
no pipeline helpers except ``rolling_z``, which is itself checked by hand), and
the structural fixes are pinned: finite memory (a partition's values do not
depend on where its frame starts), the dropped and renamed columns, clock-time
targets, the dollar calibration day, whale sizes without a static table, and
the metrics lag.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from pipeline.align import asof_join_knowable
from pipeline.bars import DollarAccumulator
from pipeline.config import WhaleSpec, Windows, lag_ms, load_windows, scale_windows
from pipeline.constants import DAY_MS, DROPPED_FAMILIES, RENAMED_COLUMNS, TIME_WIDTH_MS
from pipeline.dataset import add_targets
from pipeline.features import (
    PHYSICS_FEATURES,
    _corr_roll,
    add_cross_asset,
    compute_physics,
    ewm_finite,
    hurst_rs,
    kyle_lambda,
    permutation_entropy,
    quantile_from_hists,
    rolling_z,
    rows_to_frame,
    summarize_day,
    trade_notional_hist,
    whale_threshold,
)

T0 = 1_577_836_800_000  # 2020-01-01 00:00 UTC


def _bars(n: int, *, seed: int = 0, width: int = 3_600_000, bar_type: str = "time_1h") -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
    vol = rng.uniform(5, 50, n)
    buy = vol * rng.uniform(0.2, 0.8, n)
    ts = T0 + width * np.arange(1, n + 1)
    return pl.DataFrame({
        "asset": ["BTCUSDT"] * n, "bar_type": [bar_type] * n,
        "ts_open_ms": ts - width, "ts_ms": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol, "buy_volume": buy, "sell_volume": vol - buy,
        "buy_usd": buy * close, "sell_usd": (vol - buy) * close,
        "n_trades": rng.integers(10, 200, n), "signed_usd": (2 * buy - vol) * close,
    })


def _z(values: np.ndarray, window: int) -> np.ndarray:
    """rolling_z applied to a numpy series (rolling_z is checked by hand below)."""
    out = pl.DataFrame({"x": values}).with_columns(pl.col("x").fill_nan(None)).select(rolling_z("x", window).alias("x"))
    return out["x"].to_numpy().astype(float)


def _same(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert (np.isfinite(a) == np.isfinite(b)).all(), "null patterns differ"
    m = np.isfinite(a)
    assert m.sum() > 0
    assert np.max(np.abs(a[m] - b[m])) <= tol * max(1.0, float(np.max(np.abs(b[m]))))


# --- z-score -------------------------------------------------------------------------

def _hand_z(x: np.ndarray, window: int) -> np.ndarray:
    """Mean and sample std of the defined values among the last `window`, >= 4/5 of them."""
    need = max(2, -(-4 * window // 5))
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        w = x[max(0, i - window + 1):i + 1]
        w = w[np.isfinite(w)]
        if np.isfinite(x[i]) and len(w) >= need:
            out[i] = np.clip((x[i] - w.mean()) / w.std(ddof=1), -5, 5)
    return out


def test_rolling_z_is_the_plain_z_score_with_no_epsilon():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, 300)
    got = _z(x, 50)
    _same(got, _hand_z(x, 50), 1e-12)
    assert np.isnan(got[:39]).all() and np.isfinite(got[39:]).all()  # 40 of 50 values needed


def test_one_undefined_input_does_not_blank_the_next_window():
    """2.4.x/2.5.0-rc: a single null input nulled the next `z` z-scores (16% of DOGE dollar efficiency)."""
    rng = np.random.default_rng(13)
    x = rng.normal(0, 1, 400)
    x[[100, 180, 181]] = np.nan
    got = _z(x, 50)
    assert np.isnan(got[[100, 180, 181]]).all()
    assert np.isfinite(np.delete(got, [100, 180, 181])[60:]).all()
    _same(got, _hand_z(x, 50), 1e-12)
    many = x.copy()
    many[200:212] = np.nan  # 12 of 50 undefined: under 4/5 left
    assert np.isnan(_z(many, 50)[212:221]).all()


def test_rolling_z_is_scale_free_where_the_epsilon_was_not():
    # 2.4.x divided by std + 1e-5; a 1m volatility series has a std of that order.
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 400)
    _same(_z(x * 1e-6, 100), _z(x, 100), 1e-9)


def test_rolling_z_is_null_on_a_flat_window():
    x = np.concatenate([np.full(60, 3.0), np.arange(40, dtype=float)])
    z = _z(x, 50)
    assert np.isnan(z[:60]).all()


# --- finite-memory EMA and partition independence ---------------------------------------

def test_ewm_finite_is_the_normalised_truncated_ema():
    rng = np.random.default_rng(3)
    x = rng.normal(size=500)
    got = ewm_finite(x, 26, 120)
    a = 2 / 27
    wts = (1 - a) ** np.arange(120)
    for t in (119, 250, 499):
        want = np.dot(wts, x[t::-1][:120]) / wts.sum()
        assert got[t] == pytest.approx(want, rel=1e-12)
    assert np.isnan(got[:119]).all()
    # Over the whole history it is the adjusted (normalised) EMA.
    full = ewm_finite(x, 26, len(x))[-1]
    ref = pl.Series(x).ewm_mean(span=26, adjust=True)[-1]
    assert full == pytest.approx(ref, rel=1e-10)
    x[300] = np.nan
    assert np.isnan(ewm_finite(x, 26, 120)[300:420]).all()


@pytest.mark.parametrize("bar_type", ["time_1d", "time_8h", "time_4h", "time_1h"])
def test_features_do_not_depend_on_where_the_frame_starts(bar_type):
    """2.4.x recursive EMAs made a 1d bar's value depend on its month's warm-up start (up to 1.7 z)."""
    w = scale_windows(load_windows(), bar_type, TIME_WIDTH_MS[bar_type])
    n = w.max_lookback * 3 + 50
    bars = _bars(n, seed=4, width=TIME_WIDTH_MS[bar_type], bar_type=bar_type)
    full = compute_physics(bars, w)
    for k in (1, 17, w.max_lookback // 2):
        part = compute_physics(bars[k:], w)
        for col in PHYSICS_FEATURES:
            a = full[col][k + w.max_lookback:].to_numpy().astype(float)
            b = part[col][w.max_lookback:].to_numpy().astype(float)
            _same(b, a, 1e-9)


def test_every_window_chain_fits_the_cap_on_every_cadence():
    for bt, width in TIME_WIDTH_MS.items():
        w = scale_windows(load_windows(), bt, width)
        cap = int(90 * DAY_MS / width)
        assert w.max_lookback <= cap, bt
        assert (w.ema + w.vol_cluster - 1) + w.z <= w.max_lookback, bt
        assert (w.oi_median + w.ema) + w.z <= w.max_lookback, bt


# --- corrected formulas ----------------------------------------------------------------

def test_vol_price_corr_and_flow_persistence_are_pearson_correlations():
    rng = np.random.default_rng(5)
    a, b = rng.normal(10, 2, 200), rng.normal(0, 1, 200) + np.linspace(0, 3, 200)
    got = pl.DataFrame({"a": a, "b": b}).select(_corr_roll("a", "b", 20, "r"))["r"].to_numpy()
    want = np.full(200, np.nan)
    for t in range(19, 200):
        want[t] = np.corrcoef(a[t - 19:t + 1], b[t - 19:t + 1])[0, 1]
    _same(got, want, 1e-10)


def test_physics_correlations_match_numpy_on_bars():
    w = Windows(z=30, fast=15, ema=40, oi_median=10)
    bars = _bars(400, seed=6)
    out = compute_physics(bars, w)
    vol = bars["volume"].to_numpy()
    close = bars["close"].to_numpy()
    flow = (bars["buy_volume"] - bars["sell_volume"]).to_numpy() / vol
    abs_ret = np.abs(np.concatenate([[np.nan], close[1:] / close[:-1] - 1]))
    flow_lag = np.concatenate([[np.nan], flow[:-1]])
    vp, fp = np.full(400, np.nan), np.full(400, np.nan)
    for t in range(15, 400):
        s = slice(t - 14, t + 1)
        vp[t] = np.corrcoef(np.log(vol[s]), abs_ret[s])[0, 1]
        fp[t] = np.corrcoef(flow[s], flow_lag[s])[0, 1]
    _same(out["norm_vol_price_corr"].to_numpy().astype(float), _z(vp, 30), 1e-8)
    _same(out["norm_flow_persistence"].to_numpy().astype(float), _z(fp, 30), 1e-8)


def test_corwin_schultz_uses_the_sum_of_both_blocks():
    w = Windows(z=30, cs_block=3, ema=40, oi_median=10)
    bars = _bars(300, seed=7)
    out = compute_physics(bars, w)
    hi, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    k = 3 - 2 * math.sqrt(2)
    s = np.full(300, np.nan)
    for t in range(2 * 3 - 1, 300):
        h1, l1 = hi[t - 2:t + 1].max(), lo[t - 2:t + 1].min()
        h0, l0 = hi[t - 5:t - 2].max(), lo[t - 5:t - 2].min()
        beta = math.log(h1 / l1) ** 2 + math.log(h0 / l0) ** 2
        gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
        s[t] = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    _same(out["norm_cs_spread"].to_numpy().astype(float), _z(s, 30), 1e-8)


def test_corwin_schultz_is_defined_on_a_block_without_range():
    """A flat block is ln(H/L) = 0, not undefined (2.4.x nulled it: 9.4% of DOGE dollar bars)."""
    bars = _bars(300, seed=14)
    flat = pl.Series([100.0] * 300)
    rows = pl.Series(range(300))
    is_flat = (rows >= 150) & (rows < 160)
    bars = bars.with_columns([
        pl.when(is_flat).then(flat).otherwise(pl.col(c)).alias(c) for c in ("open", "high", "low", "close")
    ])
    w = Windows(z=30, cs_block=3, ema=40, oi_median=10)
    out = compute_physics(bars, w)
    hi, lo = bars["high"].to_numpy(), bars["low"].to_numpy()
    k = 3 - 2 * math.sqrt(2)
    s = np.full(300, np.nan)
    for t in range(5, 300):
        h1, l1 = hi[t - 2:t + 1].max(), lo[t - 2:t + 1].min()
        h0, l0 = hi[t - 5:t - 2].max(), lo[t - 5:t - 2].min()
        beta = math.log(h1 / l1) ** 2 + math.log(h0 / l0) ** 2
        gamma = math.log(max(h0, h1) / min(l0, l1)) ** 2
        alpha = (math.sqrt(2 * beta) - math.sqrt(beta)) / k - math.sqrt(gamma / k)
        s[t] = 2 * (math.exp(alpha) - 1) / (1 + math.exp(alpha))
    assert s[159] == 0.0  # six flat bars: S = 0
    _same(out["norm_cs_spread"].to_numpy().astype(float), _z(s, 30), 1e-8)


def test_kyle_lambda_is_the_ols_slope_of_return_on_signed_flow():
    rng = np.random.default_rng(8)
    flow = rng.normal(0, 1e6, 200)
    close = 100 * np.exp(np.cumsum(2e-9 * flow + rng.normal(0, 1e-3, 200)))
    got = kyle_lambda(close, flow, 40)
    lr = np.concatenate([[np.nan], np.diff(np.log(close))])
    for t in (41, 100, 199):
        slope = np.polyfit(flow[t - 40:t], lr[t - 40:t], 1)[0]
        assert got[t] == pytest.approx(slope, rel=1e-9)
    assert got[60] == pytest.approx(2e-9, rel=0.5)  # recovers the planted impact


def test_yang_zhang_uses_sample_variances_and_the_rs_mean():
    n_yz = 12
    w = Windows(z=30, yz=n_yz, ema=40, oi_median=10)
    bars = _bars(300, seed=9)
    out = compute_physics(bars, w)
    o, h, l, c = (bars[x].to_numpy() for x in ("open", "high", "low", "close"))
    on = np.concatenate([[np.nan], np.log(o[1:] / c[:-1])])
    oc = np.log(c / o)
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)
    k = 0.34 / (1.34 + (n_yz + 1) / (n_yz - 1))
    yz = np.full(300, np.nan)
    for t in range(n_yz, 300):
        s = slice(t - n_yz + 1, t + 1)
        yz[t] = math.sqrt(np.var(on[s], ddof=1) + k * np.var(oc[s], ddof=1) + (1 - k) * rs[s].mean())
    _same(out["norm_yz_volatility"].to_numpy().astype(float), _z(yz, 30), 1e-8)


def test_undefined_values_are_null_not_constants():
    flat = np.zeros(100)
    assert np.isnan(hurst_rs(flat, 20)).all()  # 2.4.x wrote 0.5
    bars = _bars(200, seed=10).with_columns(pl.lit(100.0).alias("close"), pl.lit(100.0).alias("open"),
                                            pl.lit(100.0).alias("high"), pl.lit(100.0).alias("low"))
    out = compute_physics(bars, Windows(z=20, ema=30, oi_median=10, efficiency=5))
    assert out["norm_efficiency"].null_count() == out.height  # 0/0 path, not 0


def test_permutation_entropy_ties_do_not_depend_on_rounding_noise():
    rng = np.random.default_rng(11)
    base = rng.choice([-0.001, 0.0, 0.001, 0.002], size=400)
    noisy = base * (1 + rng.choice([-1, 0, 1], size=400) * 2e-16)
    _same(permutation_entropy(noisy, 50), permutation_entropy(base, 50), 0.0)


# --- dropped and renamed ---------------------------------------------------------------

def test_dropped_columns_are_gone_and_documented():
    out = compute_physics(_bars(300), Windows(z=20, ema=30, oi_median=10))
    for gone in ("norm_funding", "norm_funding_momentum", "norm_spread_bps",
                 "norm_hawkes_buy_intensity", "norm_hawkes_sell_intensity", "norm_hl_spread"):
        assert gone not in out.columns and gone not in PHYSICS_FEATURES
    assert "norm_rs_volatility" in out.columns and RENAMED_COLUMNS["norm_hl_spread"] == "norm_rs_volatility"
    assert len(PHYSICS_FEATURES) == 26
    for key in ("norm_funding", "norm_funding_momentum", "liq_proxy", "norm_spread_bps"):
        assert key in DROPPED_FAMILIES
    row = summarize_day(_trades_day(), None, asset="BTCUSDT", day=date(2020, 1, 2), whale_usd=500.0)
    assert not any(k.startswith("liq_") for k in row)


# --- cross-asset rank --------------------------------------------------------------------

def test_momentum_rank_is_a_cohort_percentile():
    ts = [T0 + 3_600_000 * i for i in range(1, 4)]

    def frame(closes):
        return pl.DataFrame({"ts_ms": ts, "close": closes, "funding_rate": [None] * 3},
                            schema_overrides={"funding_rate": pl.Float64})

    frames = {
        "BTCUSDT": frame([100.0, 101.0, 102.0]),   # +1.0%, ...
        "ETHUSDT": frame([100.0, 103.0, 103.0]),   # +3.0% then 0
        "SOLUSDT": frame([100.0, 99.0, 103.0]),    # -1.0% then best
        "XRPUSDT": frame([100.0, 101.0, 100.0]),   # ties BTC at bar 2
    }
    out = add_cross_asset(frames, bar_type="time_1h", fast=2, ma=2)
    at = {a: out[a]["xd_momentum_rank"][1] for a in frames}
    assert at["ETHUSDT"] == 1.0 and at["SOLUSDT"] == 0.0
    assert at["BTCUSDT"] == at["XRPUSDT"] == pytest.approx((1 + 0.5) / 3)
    for a in frames:
        assert 0.0 <= out[a]["xd_momentum_rank"][2] <= 1.0


# --- targets -----------------------------------------------------------------------------

def test_time_bar_targets_use_clock_time_through_quiet_minutes():
    m = 60_000
    df = pl.DataFrame({"ts_ms": [T0 + m * k for k in (1, 2, 3, 6, 7)],
                       "close": [10.0, 11.0, 12.0, 13.0, 14.0]})
    out = add_targets(df, (1, 4), width_ms=m)
    t1 = out["target_return_1"].to_list()
    assert t1[1] == pytest.approx(12 / 11 - 1)   # minute 2 -> minute 3
    assert t1[2] == 0.0                          # minute 3 -> 4: no trade, price unchanged
    assert t1[3] == pytest.approx(14 / 13 - 1)
    assert t1[4] is None                         # past the last bar
    t4 = out["target_return_4"].to_list()
    assert t4[1] == pytest.approx(13 / 11 - 1)   # minute 2 -> minute 6
    assert t4[3] is None
    rows = add_targets(df, (1,))["target_return_1"].to_list()  # event bars / FX: next row
    assert rows[2] == pytest.approx(13 / 12 - 1)


# --- lags --------------------------------------------------------------------------------

def test_daily_metrics_wait_for_the_last_window_to_publish():
    # The last row of day D is stamped D 23:55 and is knowable 10 minutes after its stamp.
    assert lag_ms("metrics_daily") == (DAY_MS - 5 * 60_000) + lag_ms("open_interest")
    panel = pl.DataFrame({"event_time_ms": [T0], "s3_x_mean": [1.0]})
    bars = pl.DataFrame({"ts_ms": [T0 + DAY_MS, T0 + DAY_MS + 4 * 60_000, T0 + DAY_MS + 5 * 60_000]})
    j = asof_join_knowable(bars, panel, ["s3_x_mean"], lag_ms("metrics_daily"), feature_time_col="event_time_ms")
    assert j["s3_x_mean"].to_list() == [None, None, 1.0]


# --- dollar calibration day --------------------------------------------------------------

def _trades_day(day_ms: int = T0, n: int = 400, usd: float = 50.0, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    ts = day_ms + np.sort(rng.integers(0, DAY_MS, n))
    return pl.DataFrame({
        "ts_ms": ts, "ts_us": ts * 1000, "agg_trade_id": np.arange(n),
        "price": np.full(n, 10.0), "qty": np.full(n, usd / 10.0),
        "is_buyer_maker": rng.integers(0, 2, n).astype(bool),
    })


def test_adaptive_dollar_bars_calibrate_on_the_first_day():
    acc = DollarAccumulator(None, target_bars_per_day=40.0, window_days=30)
    acc.add(_trades_day(T0, 400, seed=1))
    bars, excluded = acc.emit_completed("BTCUSDT")
    assert bars.height == 0 and excluded == 400 and acc.calibration_trades == 400
    acc.add(_trades_day(T0 + DAY_MS, 400, seed=2))
    bars, excluded = acc.emit_completed("BTCUSDT")
    assert bars.height > 0
    assert int(bars["ts_open_ms"].min()) >= T0 + DAY_MS  # nothing opened on the calibration day
    assert (bars["volume_usd"] >= 400 * 50.0 / 40.0 - 1e-6).all()  # size = day-1 notional / 40
    assert int(bars["n_trades"].sum()) + excluded == 800
    meta, frag = acc.state()
    again = DollarAccumulator.restore(meta, frag)
    assert again.calibration_trades == 400 and again.excluded_trades() == excluded
    with pytest.raises(ValueError):
        DollarAccumulator(1_000.0, target_bars_per_day=40.0)  # no static first-day size


# --- whale ---------------------------------------------------------------------------------

def test_whale_columns_are_null_without_history_and_measured_after():
    spec = WhaleSpec(quantile=0.999, window_days=30, version="t")
    assert whale_threshold([], spec) is None
    trades = _trades_day()
    first = summarize_day(trades, None, asset="BTCUSDT", day=date(2020, 1, 1), whale_usd=None)
    assert first["wh_large_count"] is None and first["wh_threshold_usd"] is None
    assert first["wh_max_print_usd"] == pytest.approx(50.0)
    frame = rows_to_frame([first])
    assert frame.schema["wh_large_count"] == pl.Int64 and frame.schema["wh_large_usd"] == pl.Float64
    later = summarize_day(trades, None, asset="BTCUSDT", day=date(2020, 1, 2), whale_usd=40.0)
    assert later["wh_large_count"] == 400


def test_whale_quantile_is_within_a_fraction_of_a_bin_of_the_exact_one():
    rng = np.random.default_rng(12)
    usd = np.exp(rng.normal(6, 1.5, 300_000))
    exact = float(np.quantile(usd, 0.999))
    est = quantile_from_hists([trade_notional_hist(usd)], 0.999)
    assert abs(est / exact - 1) < 0.01


# --- review round 2: accounting, warm-up, freshness ---------------------------------------

def _fixture_days(root, days):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from fixtures import write_bronze_day
    from test_resume import _day
    for i, d in enumerate(days):
        write_bronze_day(root, "BTCUSDT", d, _day(i + 1, 100.0 + i, d))


def test_validate_counts_dollar_trades_when_only_other_bars_were_rebuilt(tmp_path, monkeypatch):
    """The bars stage used to hand validate 0 excluded trades when no month rebuilt the
    dollar fabric (a cadence added on a rerun), so the exact trade-count gate failed."""
    from datetime import timedelta

    import pipeline.orchestrator as orch
    from pipeline.chunked import dollar_spec_for, read_dollar_state
    from pipeline.orchestrator import RunConfig, run
    from fixtures import small_windows

    days = [date(2024, 5, 30) + timedelta(days=i) for i in range(3)]
    _fixture_days(tmp_path, days)
    base = dict(start=days[0], end=days[-1], assets=["BTCUSDT"], data_root=tmp_path,
                windows=small_windows(), skip_network=True)
    assert run(RunConfig(stages=["bars"], bar_types=["dollar"], **base)) == 0
    calls = []
    real_gate = orch.gate_trade_count

    def spy(n_bars, n_bronze, *, excluded_tail=0, what="bars"):
        r = real_gate(n_bars, n_bronze, excluded_tail=excluded_tail, what=what)
        calls.append((what, r.severity, excluded_tail))
        return r

    monkeypatch.setattr(orch, "gate_trade_count", spy)
    monkeypatch.setattr(orch, "_independent_validate", lambda cfg, worst, reports=None: worst)
    run(RunConfig(stages=["bars", "features", "dataset", "validate"], bar_types=["time_1h", "dollar"], **base))
    state = read_dollar_state(tmp_path, "BTCUSDT", dollar_spec_for(RunConfig(**base), "BTCUSDT"), through="2024-06")
    dollar = [c for c in calls if c[0] == "dollar"]
    assert dollar and dollar[-1][1] == "pass"
    assert dollar[-1][2] == state.excluded_trades() >= state.calibration_trades > 0


def test_dollar_validation_of_a_later_sub_range_keeps_the_chain_origin(tmp_path):
    from datetime import timedelta

    from pipeline.orchestrator import RunConfig, run
    from pipeline.validate_run import Report, check_dollar_fabric
    from fixtures import small_windows

    days = [date(2024, 5, 28) + timedelta(days=i) for i in range(5)]
    _fixture_days(tmp_path, days)
    cfg = RunConfig(start=days[0], end=days[-1], assets=["BTCUSDT"], data_root=tmp_path,
                    stages=["bars"], bar_types=["dollar"], windows=small_windows(), skip_network=True)
    assert run(cfg) == 0
    for lo in (days[0], days[2]):
        report = Report()
        check_dollar_fabric(report, tmp_path, "BTCUSDT", lo, days[-1])
        sev = {c.name: c.severity for c in report.checks}
        assert sev["dollar_notional:BTCUSDT"] == "PASS", (lo, [c.detail for c in report.checks])


def test_feature_months_rebuild_when_windows_or_lags_change(tmp_path):
    from datetime import timedelta

    from pipeline.io import read_parquet_metadata
    from pipeline.orchestrator import RunConfig, run
    from pipeline.paths import features_month_path
    from fixtures import small_windows

    days = [date(2024, 5, 30) + timedelta(days=i) for i in range(3)]
    _fixture_days(tmp_path, days)
    base = dict(start=days[0], end=days[-1], assets=["BTCUSDT"], data_root=tmp_path,
                stages=["bars", "features"], bar_types=["time_1h"], skip_network=True)
    assert run(RunConfig(windows=small_windows(), **base)) == 0
    path = features_month_path(tmp_path, "time_1h", "BTCUSDT", 2024, 6)
    first = read_parquet_metadata(path)["params_hash"]
    assert run(RunConfig(windows=small_windows(), **base)) == 0
    assert read_parquet_metadata(path)["params_hash"] == first  # unchanged config: cached
    assert run(RunConfig(windows=small_windows(ema=12), **base)) == 0
    assert read_parquet_metadata(path)["params_hash"] != first  # an edited window rebuilds


def test_fx_feature_months_warm_up_by_bars_not_calendar_days(tmp_path):
    """FX trades five days a week: 92 calendar days held fewer 1d bars than the
    100-bar windows, so month heads were null (or differed from full history)."""
    from datetime import timedelta

    from pipeline.chunked import load_feature_months, write_frame, write_fx_features_month
    from pipeline.config import load_windows
    from pipeline.features import FX_FEATURES, compute_fx_features
    from pipeline.orchestrator import RunConfig
    from pipeline.paths import bars_month_path

    rng = np.random.default_rng(15)
    days = [date(2020, 1, 1) + timedelta(days=i) for i in range(274)]
    days = [d for d in days if d.weekday() < 5]
    n = len(days)
    ts = np.array([int(datetime_utc(d).timestamp() * 1000) + DAY_MS for d in days])
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.concatenate([[1.1], close[:-1]])
    bars = pl.DataFrame({
        "asset": ["EURUSD"] * n, "bar_type": ["time_1d"] * n, "bar_id": ts // DAY_MS,
        "ts_open_ms": ts - DAY_MS, "ts_ms": ts,
        "open": open_, "high": np.maximum(open_, close) * 1.002, "low": np.minimum(open_, close) * 0.998,
        "close": close, "vwap": close, "volume": rng.uniform(1e4, 2e4, n),
        "volume_usd": [None] * n, "buy_volume": [None] * n, "sell_volume": [None] * n,
        "buy_usd": [None] * n, "sell_usd": [None] * n, "n_trades": rng.integers(10_000, 20_000, n),
        "signed_usd": [None] * n, "bid_close": close - 5e-5, "ask_close": close + 5e-5,
        "spread_mean": rng.uniform(5e-5, 2e-4, n), "spread_max": rng.uniform(2e-4, 4e-4, n),
    }, schema_overrides={c: pl.Float64 for c in ("volume_usd", "buy_volume", "sell_volume", "buy_usd", "sell_usd", "signed_usd")})
    for (y, m), part in bars.with_columns(
        pl.from_epoch(pl.col("ts_open_ms"), time_unit="ms").dt.year().alias("_y"),
        pl.from_epoch(pl.col("ts_open_ms"), time_unit="ms").dt.month().alias("_m"),
    ).group_by(["_y", "_m"]):
        write_frame(part.drop(["_y", "_m"]), bars_month_path(tmp_path, "time_1d", "EURUSD", y, m), "bars_v1", {})
    cfg = RunConfig(start=date(2020, 1, 1), end=date(2020, 9, 30), assets=["EURUSD"], market="fx",
                    data_root=tmp_path, bar_types=["time_1d"], windows=load_windows())
    assert write_fx_features_month(cfg, ["EURUSD"], "time_1d", 2020, 7)["EURUSD"] == "written"
    got = load_feature_months(tmp_path, "EURUSD", "time_1d", date(2020, 7, 1), date(2020, 7, 31))
    want = compute_fx_features(bars, 1e-4).join(got.select("ts_ms"), on="ts_ms", how="semi").sort("ts_ms")
    assert got.height == want.height > 15
    for col in FX_FEATURES:
        if col in ("base_strength_ret", "quote_strength_ret"):
            continue
        a = got[col].cast(pl.Float64).to_numpy()
        b = want[col].cast(pl.Float64).to_numpy()
        assert (np.isfinite(a) == np.isfinite(b)).all(), col
        assert np.allclose(a[np.isfinite(a)], b[np.isfinite(b)], rtol=1e-12, atol=0), col
    assert got["spread_z"].null_count() == 0  # the month head is not blank


def datetime_utc(d):
    from datetime import datetime, timezone
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
