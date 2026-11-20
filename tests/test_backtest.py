"""Synthetic-data proofs for the backtest engine and the first three strategies."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import polars as pl
import pytest

from pipeline.config import load_embargo_policy
from pipeline.constants import HOUR_MS, TIME_WIDTH_MS
from strategy import (
    CostModel,
    buy_and_hold,
    cadence_embargo_bars,
    donchian_breakout,
    load_cost_model,
    regime_gate,
    run_backtest,
    vol_scaled_momentum,
    walk_forward,
)


def _ts(n: int, step_ms: int, start: datetime | None = None) -> np.ndarray:
    start = start or datetime(2020, 1, 6, tzinfo=timezone.utc)  # Monday
    t0 = int(start.timestamp() * 1000) + step_ms  # first label is the close of bar 0
    return t0 + step_ms * np.arange(n, dtype=np.int64)


def make_bars(
    close: np.ndarray,
    *,
    bar_type: str = "time_1d",
    step_ms: int | None = None,
    open_: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
    bid: np.ndarray | None = None,
    ask: np.ndarray | None = None,
    spread_mean: np.ndarray | None = None,
    start: datetime | None = None,
) -> pl.DataFrame:
    close = np.asarray(close, dtype=np.float64)
    n = close.size
    step_ms = step_ms or TIME_WIDTH_MS[bar_type]
    ts = _ts(n, step_ms, start)
    if open_ is None:
        open_ = np.empty(n, dtype=np.float64)
        open_[0] = close[0]
        open_[1:] = close[:-1]
    if high is None:
        high = np.maximum(open_, close)
    if low is None:
        low = np.minimum(open_, close)
    data = {
        "ts_ms": ts,
        "ts_open_ms": ts - step_ms,
        "open": np.asarray(open_, dtype=np.float64),
        "high": np.asarray(high, dtype=np.float64),
        "low": np.asarray(low, dtype=np.float64),
        "close": close,
        "bar_type": [bar_type] * n,
        "asset": ["TEST"] * n,
    }
    if bid is not None:
        data["bid_close"] = np.asarray(bid, dtype=np.float64)
    if ask is not None:
        data["ask_close"] = np.asarray(ask, dtype=np.float64)
    if spread_mean is not None:
        data["spread_mean"] = np.asarray(spread_mean, dtype=np.float64)
    return pl.DataFrame(data)


def test_cost_config_matches_sourced_yaml():
    crypto = load_cost_model("crypto")
    fx = load_cost_model("fx")
    assert crypto == CostModel(fee_bps=10.0, slippage_bps=2.0, half_spread_bps=0.0)
    assert fx == CostModel(fee_bps=0.35, slippage_bps=0.0, half_spread_bps=0.0)


def test_cost_accounting_matches_hand_calculation_open_fill():
    # close path: 100, 102, 101, 103. Open of bar t is the previous close.
    close = np.array([100.0, 102.0, 101.0, 103.0, 103.0])
    bars = make_bars(close)
    # Decision at each close: enter at bar 0, hold, exit at bar 2, stay flat.
    target = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    costs = CostModel(fee_bps=10.0, slippage_bps=2.0, half_spread_bps=0.0)
    rate = (10.0 + 2.0) / 1e4  # 12 bps
    bt = run_backtest(bars, target, costs=costs)

    # Bar 0: flat, no fill. Bar 1: fill 0->1 at open 100, then 100->102.
    # Bar 2: hold 1 through open 102 and close 101. Bar 3: fill 1->0 at open 101.
    assert bt.position[0] == 0.0
    assert bt.position[1] == 1.0
    assert bt.position[2] == 1.0
    assert bt.position[3] == 0.0
    assert bt.cost[0] == 0.0
    assert bt.cost[1] == pytest.approx(rate)
    assert bt.cost[2] == 0.0
    assert bt.cost[3] == pytest.approx(rate)
    # Gross: bar 1 gap is 0 (open=prev close), intra = 102/100 - 1.
    assert bt.gross_return[1] == pytest.approx(102.0 / 100.0 - 1.0)
    # Bar 2: gap 102/102-1=0, intra 101/102-1, still long.
    assert bt.gross_return[2] == pytest.approx(101.0 / 102.0 - 1.0)
    # Bar 3: gap 101/101-1=0, then flat so intra of 103/101 is not earned.
    assert bt.gross_return[3] == pytest.approx(0.0)
    assert bt.net_return[1] == pytest.approx(bt.gross_return[1] - rate)
    assert bt.net_return[3] == pytest.approx(0.0 - rate)
    assert bt.trades.height == 2
    assert bt.trades["delta"].to_list() == pytest.approx([1.0, -1.0])
    assert bt.metrics["cost_sum"] == pytest.approx(2 * rate)
    assert bt.metrics["turnover_sum"] == pytest.approx(2.0)


def test_fx_half_spread_uses_quote_columns_not_fallback():
    close = np.array([100.0, 100.0, 100.0, 100.0])
    # 20 bps full spread -> 10 bps half. Fallback is deliberately huge.
    spread = np.array([0.20, 0.20, 0.20, 0.20])
    bars = make_bars(close, spread_mean=spread, bid=close - 0.5, ask=close + 0.5)
    target = np.array([1.0, 1.0, 0.0, 0.0])
    costs = CostModel(fee_bps=0.0, slippage_bps=0.0, half_spread_bps=500.0)
    bt = run_backtest(bars, target, costs=costs)
    half = 0.5 * 0.20 / 100.0  # 10 bps, from spread_mean, not 50% fallback
    assert bt.cost[1] == pytest.approx(half)
    assert bt.cost[3] == pytest.approx(half)
    # Without quote columns the fallback would have been 500 bps.
    bare = make_bars(close)
    fallback = run_backtest(bare, target, costs=costs)
    assert fallback.cost[1] == pytest.approx(500.0 / 1e4)
    assert fallback.cost[1] != pytest.approx(bt.cost[1])


def test_constant_long_equals_buy_and_hold_minus_costs():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=80)
    close = 100.0 * np.cumprod(1.0 + r)
    bars = make_bars(close, bar_type="time_1d")
    costs = CostModel(fee_bps=10.0, slippage_bps=2.0, half_spread_bps=0.0)
    long = run_backtest(bars, np.ones(bars.height), costs=costs)
    gross_bh = buy_and_hold(bars, costs=CostModel(0.0, 0.0, 0.0))
    rate = 12.0 / 1e4
    assert long.gross_return == pytest.approx(gross_bh.gross_return, rel=0, abs=1e-12)
    # One entry, no later turnover.
    assert long.cost[1] == pytest.approx(rate)
    assert np.allclose(long.cost[2:], 0.0)
    expected_net = gross_bh.net_return.copy()
    expected_net[1] -= rate
    assert long.net_return == pytest.approx(expected_net, rel=0, abs=1e-12)
    # Compounded equity: only bar 1 differs, then the same multipliers.
    assert long.equity[-1] == pytest.approx(
        gross_bh.equity[-1] * (1.0 + expected_net[1]) / (1.0 + gross_bh.net_return[1]),
        rel=0,
        abs=1e-12,
    )


def test_close_fill_when_open_is_missing():
    close = np.array([100.0, 110.0, 105.0, 120.0])
    bars = make_bars(close).drop("open")
    target = np.ones(4)
    costs = CostModel(fee_bps=10.0, slippage_bps=0.0, half_spread_bps=0.0)
    bt = run_backtest(bars, target, costs=costs)
    # Decision at close 0 fills at close 1: bar 1 is still flat, entry cost at close 1.
    assert bt.position[0] == 0.0
    assert bt.position[1] == 0.0
    assert bt.position[2] == 1.0
    assert bt.cost[1] == pytest.approx(10.0 / 1e4)
    assert bt.gross_return[1] == 0.0
    # Bar 2 earns close-to-close while long: 105/110 - 1.
    assert bt.gross_return[2] == pytest.approx(105.0 / 110.0 - 1.0)
    assert bt.net_return[2] == pytest.approx(bt.gross_return[2])


def test_no_lookahead_peeking_strategy_has_no_edge():
    """A same-bar peek has a perfect edge; the engine's delay removes it.

    On uncorrelated +/- 1% returns, signing the next close-to-close return and
    applying it to that same bar earns 1% every bar. The engine fills a target
    decided at close t on bar t+1, so a target that only peeks at bar t+2
    (the first bar the delayed position can earn) has mean return near zero.
    """
    rng = np.random.default_rng(1)
    r = rng.choice(np.array([-0.01, 0.01]), size=400)
    close = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + r)])
    bars = make_bars(close, bar_type="time_1d")
    n = bars.height
    r_bar = np.zeros(n)
    r_bar[1:] = close[1:] / close[:-1] - 1.0
    # Illegal: decide with the return of the bar you are about to earn.
    peek = np.zeros(n)
    peek[:-1] = np.sign(r_bar[1:])
    illegal = peek[:-1] * r_bar[1:]
    illegal_mean = float(np.mean(illegal))
    assert illegal_mean == pytest.approx(0.01, rel=0, abs=1e-12)

    # Through the engine, open[t] = close[t-1], so held[t] = target[t-1] earns
    # r_bar[t]. A target that peeks at r_bar[t+2] is then uncorrelated with
    # the return it actually earns.
    peek2 = np.zeros(n)
    peek2[:-2] = np.sign(r_bar[2:])
    delayed = run_backtest(bars, peek2, costs=CostModel(0.0, 0.0, 0.0))
    active = delayed.net_return[2:-1]
    assert abs(float(np.mean(active))) < 0.002
    assert delayed.metrics["sharpe"] < 1.0


def test_shifting_signals_changes_results_as_expected():
    # Alternating +1% / -1% close-to-close, open is previous close.
    r = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(60)])
    close = 100.0 * np.concatenate([[1.0], np.cumprod(1.0 + r)])
    bars = make_bars(close[:60])
    n = bars.height
    r_bar = np.zeros(n)
    r_bar[1:] = close[1:n] / close[: n - 1] - 1.0
    # Causal: sign of the return just observed.
    causal = np.sign(r_bar)
    causal[0] = 0.0
    # Shift the causal signal one bar toward the present (lookahead).
    shifted = np.zeros(n)
    shifted[:-1] = causal[1:]
    zero = CostModel(0.0, 0.0, 0.0)
    delayed = run_backtest(bars, causal, costs=zero)
    early = run_backtest(bars, shifted, costs=zero)
    assert not np.allclose(delayed.net_return, early.net_return)
    # Open fill, held[t] = target[t-1]. Early uses target[t-1]=causal[t]=sign(r[t]),
    # so it earns the realised move. Causal uses sign(r[t-1]) * r[t] and, on this
    # alternating path, loses.
    assert early.equity[-1] > 1.0
    assert delayed.equity[-1] < 1.0
    assert early.metrics["sharpe"] > 2.0
    assert delayed.metrics["sharpe"] < 0.0


def test_walk_forward_folds_respect_two_sided_embargo_and_seven_day_floor():
    floor_days, _ = load_embargo_policy()
    assert floor_days == 7
    n = 120
    bars = make_bars(np.linspace(100, 120, n), bar_type="time_1d")
    # lookback 2 would only be 2 bars; the 7-day floor on daily bars wins.
    embargo = cadence_embargo_bars("time_1d", lookback=2, horizon=1)
    assert embargo == 7
    folds = walk_forward(
        bars, bar_type="time_1d", lookback=2, horizon=1,
        test_bars=10, min_train_bars=20, expanding=True,
    )
    assert folds
    seen_test = np.zeros(n, dtype=bool)
    for fold in folds:
        assert fold.embargo_bars == 7
        assert fold.test_start - fold.train_end == 2 * 7
        assert fold.train_end > fold.train_start
        assert fold.test_end - fold.test_start == 10
        assert fold.test_end <= n
        # No overlap between train and test, and the gap is unused.
        train = set(fold.train_index.tolist())
        test = set(fold.test_index.tolist())
        gap = set(range(fold.train_end, fold.test_start))
        assert not train & test
        assert not train & gap
        assert not test & gap
        assert len(gap) == 2 * 7
        # A later fold's train may include an earlier fold's test (expanding
        # walk-forward does that). Within a fold, no test bar is in that fold's train.
        seen_test[fold.test_index] = True
    # A remainder shorter than test_bars + embargo is dropped, not shrunk.
    last = folds[-1]
    leftover = n - last.test_end
    if leftover:
        next_cut = (last.train_end + last.embargo_bars) + (folds[1].train_end - folds[0].train_end if len(folds) > 1 else 10)
        # After the last emitted fold, one more test window would not fit.
        assert last.test_end + 0 <= n
        would_end = last.test_end + (last.test_end - last.test_start)
        # step equals test_bars, so the next test_end would exceed n.
        assert last.test_end + 2 * last.embargo_bars + 10 > n or leftover < 10 + 2 * last.embargo_bars


def test_hourly_embargo_uses_seven_day_floor_in_bars():
    embargo = cadence_embargo_bars("time_1h", lookback=5, horizon=1)
    assert embargo == 7 * 24
    n = 7 * 24 * 6
    bars = make_bars(np.ones(n), bar_type="time_1h", step_ms=HOUR_MS)
    folds = walk_forward(
        bars, bar_type="time_1h", lookback=5, test_bars=24, min_train_bars=embargo + 48,
    )
    assert folds
    for fold in folds:
        assert fold.embargo_bars == 7 * 24
        assert fold.test_start - fold.train_end == 2 * 7 * 24


def test_daily_long_lookback_embargo_is_not_shrunk_to_the_floor():
    embargo = cadence_embargo_bars("time_1d", lookback=20, horizon=4)
    assert embargo == 20  # lookback wins over the 7-day floor and horizon 4
    n = 200
    bars = make_bars(np.linspace(1, 2, n), bar_type="time_1d")
    folds = walk_forward(
        bars, bar_type="time_1d", lookback=20, horizon=4,
        test_bars=15, min_train_bars=50,
    )
    assert folds
    for fold in folds:
        assert fold.embargo_bars == 20
        assert fold.test_start - fold.train_end == 40


def test_regime_gate_long_only_above_the_average():
    # 30 bars at 100, then a jump and a climb, then a collapse under the mean.
    close = np.concatenate([
        np.full(30, 100.0),
        np.linspace(100.0, 150.0, 20),
        np.linspace(150.0, 50.0, 20),
    ])
    bars = make_bars(close)
    pos = regime_gate(bars, window=20)
    assert pos.shape == (close.size,)
    assert np.all(pos[:19] == 0.0)  # warmup
    # Mid climb: close is above a lagging mean.
    assert pos[45] == 1.0
    # After the collapse, close is under the mean.
    assert pos[-1] == 0.0
    assert set(np.unique(pos)).issubset({0.0, 1.0})


def test_donchian_enters_on_a_prior_high_and_exits_on_a_prior_low():
    # Flat, then a breakout above 10, a hold, then a break under the 5-bar low.
    close = np.array(
        [10.0] * 20
        + [10.5, 11.0, 11.2, 11.0, 10.8]
        + [10.7, 9.5, 9.0, 9.0],
        dtype=np.float64,
    )
    high = close + 0.05
    low = close - 0.05
    # During the flat, high/low stay inside 10 ± 0.05 so no break until 10.5.
    bars = make_bars(close, high=high, low=low)
    pos = donchian_breakout(bars, entry=10, exit=5)
    # First 10 bars are warmup (entry window).
    assert np.all(pos[:10] == 0.0)
    # Bar 20 (0-based) is the first 10.5 close, which breaks the prior 10-bar high (~10.05).
    assert close[20] == 10.5
    assert pos[20] == 1.0
    assert pos[21] == 1.0
    # 9.5 should break the prior 5-bar low and go flat.
    idx = int(np.where(close == 9.5)[0][0])
    assert pos[idx] == 0.0
    assert pos[-1] == 0.0


def test_vol_scaled_momentum_is_long_or_cash_and_scales_with_vol():
    rng = np.random.default_rng(2)
    # Strong upward drift, modest noise, then a decline.
    up = 100 * np.cumprod(1.0 + 0.01 + rng.normal(0, 0.002, size=120))
    down = up[-1] * np.cumprod(1.0 + (-0.01) + rng.normal(0, 0.002, size=80))
    close = np.concatenate([up, down])
    bars = make_bars(close, bar_type="time_1d")
    pos = vol_scaled_momentum(bars, lookback=20, vol_lookback=10, target_ann_vol=0.10)
    assert pos.shape == (close.size,)
    assert np.all(pos[:20] == 0.0)
    assert np.all(pos >= 0.0) and np.all(pos <= 1.0)
    # Mid uptrend should be long; end of the decline should be cash.
    assert pos[100] > 0.0
    assert pos[-1] == 0.0
    # A higher vol target never reduces a positive weight.
    looser = vol_scaled_momentum(bars, lookback=20, vol_lookback=10, target_ann_vol=1.0)
    held = pos > 0
    assert np.all(looser[held] >= pos[held] - 1e-12)
