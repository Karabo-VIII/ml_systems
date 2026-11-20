"""Vectorised single-asset backtest, costs, metrics, and walk-forward splits.

A self-written numpy engine on gold bars. vectorbt was considered and not
adopted as a dependency: its license is Apache 2.0 plus the Commons Clause
(see docs/STRATEGY_LAYER.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from pipeline.config import bar_duration_ms, load_embargo_policy
from pipeline.constants import DAY_MS, TIME_WIDTH_MS
from pipeline.paths import repo_root

_BPS = 1e4
_TRADE_EPS = 1e-12


@dataclass(frozen=True)
class CostModel:
    """Per-side costs, in basis points of notional, charged on turnover.

    ``half_spread_bps`` is the fallback used only when the bar frame has no
    quote spread. FX gold should pass ``bid_close`` / ``ask_close`` or
    ``spread_mean`` and those take precedence.
    """

    fee_bps: float
    slippage_bps: float
    half_spread_bps: float = 0.0

    @property
    def fixed_frac(self) -> float:
        return (self.fee_bps + self.slippage_bps + self.half_spread_bps) / _BPS


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold. Indices are into the bar frame; ends are exclusive.

    The embargo is applied on both sides of the cut and is not shrunk.
    Bars in ``[train_end, test_start)`` belong to neither side.
    """

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_bars: int

    @property
    def train_index(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def test_index(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)


@dataclass
class BacktestResult:
    ts_ms: np.ndarray
    target: np.ndarray
    position: np.ndarray
    fill_price: np.ndarray
    gross_return: np.ndarray
    cost: np.ndarray
    net_return: np.ndarray
    equity: np.ndarray
    trades: pl.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def load_cost_model(market: str, path: Path | None = None) -> CostModel:
    """Load the ``backtest.<market>`` block from ``config/pipeline.yaml``."""
    path = path or (repo_root() / "config" / "pipeline.yaml")
    raw = yaml.safe_load(path.read_text()) or {}
    block = (raw.get("backtest") or {}).get(market)
    if not isinstance(block, dict):
        raise KeyError(f"config/pipeline.yaml has no backtest.{market} section")
    missing = [k for k in ("fee_bps", "slippage_bps", "half_spread_bps") if k not in block]
    if missing:
        raise KeyError(f"backtest.{market} is missing {missing}")
    return CostModel(
        fee_bps=float(block["fee_bps"]),
        slippage_bps=float(block["slippage_bps"]),
        half_spread_bps=float(block["half_spread_bps"]),
    )


def _col(bars: pl.DataFrame, name: str) -> np.ndarray | None:
    if name not in bars.columns:
        return None
    return np.asarray(bars[name].to_numpy(), dtype=np.float64)


def _positions(positions, n: int) -> np.ndarray:
    if isinstance(positions, pl.Series):
        arr = positions.to_numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(positions, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"positions must be a length-{n} vector, got shape {arr.shape}")
    if np.nanmax(np.abs(np.where(np.isfinite(arr), arr, 0.0)), initial=0.0) > 1.0 + 1e-12:
        arr = np.clip(arr, -1.0, 1.0)
    else:
        arr = np.clip(arr, -1.0, 1.0)
    return np.where(np.isfinite(arr), arr, 0.0)


def _half_spread_frac(bars: pl.DataFrame, fill_price: np.ndarray, fallback_bps: float) -> np.ndarray:
    """Half-spread as a fraction of the fill price.

    Preference: ``spread_mean`` (bar-level quote average), then
    ``0.5 * (ask_close - bid_close)``, then the configured fallback.
    ``spread_mean`` is used first because FX gold's bid/ask are the last
    quotes in the bar, which are the wrong time for an open fill.
    """
    n = fill_price.shape[0]
    out = np.full(n, fallback_bps / _BPS, dtype=np.float64)
    mid = _col(bars, "close")
    bid = _col(bars, "bid_close")
    ask = _col(bars, "ask_close")
    spread = _col(bars, "spread_mean")
    if mid is None:
        mid = np.where(np.isfinite(fill_price) & (fill_price > 0), fill_price, np.nan)
    if bid is not None and ask is not None:
        quote_mid = (bid + ask) / 2.0
        half = 0.5 * (ask - bid) / quote_mid
        ok = np.isfinite(half) & (quote_mid > 0) & (ask >= bid)
        out = np.where(ok, half, out)
    if spread is not None:
        denom = np.where(np.isfinite(mid) & (mid > 0), mid, fill_price)
        half = 0.5 * spread / denom
        ok = np.isfinite(half) & (denom > 0) & (spread >= 0)
        out = np.where(ok, half, out)
    return out


def _open_fill_mask(open_px: np.ndarray | None, n: int) -> np.ndarray:
    if open_px is None:
        return np.zeros(n, dtype=bool)
    return np.isfinite(open_px) & (open_px > 0)


def bars_per_year(ts_ms: np.ndarray, bar_type: str | None = None) -> float:
    """Cadence-aware annualisation from the sample span, not a fixed 252.

    ``(n_bars - 1) / years`` uses the first and last stamps, so weekend gaps
    on FX and 24/7 crypto both come out honest. Falls back to the named
    time-bar width, then to a 365.25-day year, when the span is unusable.
    """
    ts = np.asarray(ts_ms, dtype=np.int64)
    if ts.size >= 2:
        span = float(ts[-1] - ts[0])
        if span > 0:
            years = span / (365.25 * DAY_MS)
            if years > 0:
                return (ts.size - 1) / years
    if bar_type and bar_type in TIME_WIDTH_MS:
        return 365.25 * DAY_MS / TIME_WIDTH_MS[bar_type]
    return 365.25


def cadence_embargo_bars(
    bar_type: str,
    lookback: int,
    horizon: int = 1,
    median_bar_ms: int | None = None,
) -> int:
    """``max(horizon, lookback)`` in this cadence's bars, with the 7-day floor.

    Reuses ``load_embargo_policy`` and ``bar_duration_ms`` from the pipeline.
    The floor is not shrunk. This is the strategy lookback, not the full
    feature-window embargo used when gold is built.
    """
    floor_days, _ = load_embargo_policy()
    width = bar_duration_ms(bar_type, median_bar_ms)
    floor_bars = max(1, math.ceil(floor_days * DAY_MS / width))
    return max(int(horizon), int(lookback), floor_bars)


def _median_bar_ms(ts_ms: np.ndarray) -> int | None:
    if ts_ms.size < 2:
        return None
    d = np.diff(ts_ms.astype(np.int64))
    d = d[d > 0]
    if d.size == 0:
        return None
    return int(np.median(d))


def walk_forward(
    bars: pl.DataFrame | np.ndarray,
    *,
    bar_type: str,
    lookback: int,
    horizon: int = 1,
    test_bars: int,
    min_train_bars: int | None = None,
    train_bars: int | None = None,
    step_bars: int | None = None,
    expanding: bool = True,
) -> list[Fold]:
    """Expanding (default) or rolling train/test folds with a two-sided embargo.

    After a cut at index ``t``, train stops at ``t - embargo`` and test starts
    at ``t + embargo``, matching ``pipeline.dataset.apply_embargo``. A short
    remainder does not shrink the embargo or the test window; it is dropped.
    """
    if isinstance(bars, pl.DataFrame):
        if "ts_ms" not in bars.columns:
            raise ValueError("bars must have a ts_ms column")
        ts_ms = np.asarray(bars["ts_ms"].to_numpy(), dtype=np.int64)
        if "bar_type" in bars.columns and bars.height:
            named = str(bars["bar_type"][0])
            bar_type = bar_type or named
    else:
        ts_ms = np.asarray(bars, dtype=np.int64)
    n = int(ts_ms.size)
    embargo = cadence_embargo_bars(
        bar_type, lookback, horizon, median_bar_ms=_median_bar_ms(ts_ms),
    )
    if test_bars <= 0:
        raise ValueError("test_bars must be positive")
    min_train = int(min_train_bars or (max(int(lookback), 1) + embargo))
    step = int(step_bars or test_bars)
    if step <= 0:
        raise ValueError("step_bars must be positive")
    folds: list[Fold] = []
    cut = min_train
    fold_id = 0
    while True:
        train_end = cut - embargo
        test_start = cut + embargo
        test_end = test_start + int(test_bars)
        if train_end <= 0 or test_end > n:
            break
        if expanding:
            train_start = 0
        else:
            width = int(train_bars or min_train)
            train_start = max(0, train_end - width)
        if train_end - train_start <= 0:
            cut += step
            continue
        folds.append(
            Fold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_bars=embargo,
            )
        )
        fold_id += 1
        cut += step
    return folds


def _metrics(
    *,
    ts_ms: np.ndarray,
    net: np.ndarray,
    gross: np.ndarray,
    cost: np.ndarray,
    position: np.ndarray,
    equity: np.ndarray,
    trades: pl.DataFrame,
    bar_type: str | None,
) -> dict[str, float]:
    bpy = bars_per_year(ts_ms, bar_type)
    span = float(ts_ms[-1] - ts_ms[0]) if ts_ms.size >= 2 else 0.0
    years = span / (365.25 * DAY_MS) if span > 0 else float("nan")
    # Skip the leading flat bar so the first fill does not dilute vol.
    active = np.isfinite(net) & (np.arange(net.size) > 0)
    r = net[active]
    mu = float(np.mean(r)) if r.size else float("nan")
    sig = float(np.std(r, ddof=1)) if r.size >= 2 else float("nan")
    down = r[r < 0]
    down_sig = float(np.std(down, ddof=1)) if down.size >= 2 else (0.0 if down.size == 0 and r.size else float("nan"))
    start = float(equity[0]) if equity.size else 1.0
    end = float(equity[-1]) if equity.size else 1.0
    if years is not None and years > 0 and start > 0 and end > 0:
        cagr = (end / start) ** (1.0 / years) - 1.0
    else:
        cagr = float("nan")
    ann_vol = sig * math.sqrt(bpy) if sig == sig and bpy > 0 else float("nan")
    sharpe = (mu / sig) * math.sqrt(bpy) if sig and sig == sig and sig > 0 else float("nan")
    if down_sig == 0 and r.size and np.all(r >= 0):
        sortino = float("inf")
    elif down_sig and down_sig == down_sig and down_sig > 0:
        sortino = (mu / down_sig) * math.sqrt(bpy)
    else:
        sortino = float("nan")
    peak = np.maximum.accumulate(np.where(np.isfinite(equity), equity, 1.0))
    dd = equity / np.where(peak > 0, peak, np.nan) - 1.0
    max_dd = float(np.nanmin(dd)) if dd.size else float("nan")
    delta = np.diff(position, prepend=0.0)
    turnover_sum = float(np.sum(np.abs(delta)))
    turnover = turnover_sum / years if years and years > 0 else float("nan")
    in_mkt = position[active] != 0
    if np.any(in_mkt):
        hit_rate = float(np.mean(r[in_mkt] > 0))
    else:
        hit_rate = float("nan")
    exposure = float(np.mean(np.abs(position[active]))) if np.any(active) else 0.0
    time_in_market = float(np.mean(position[active] != 0)) if np.any(active) else 0.0
    # Gross equity path with the same fills, for cost drag vs net CAGR.
    gross_eq = np.empty_like(equity)
    gross_eq[0] = start
    for i in range(1, gross.size):
        gross_eq[i] = gross_eq[i - 1] * (1.0 + gross[i])
    if years and years > 0 and start > 0 and gross_eq[-1] > 0:
        cagr_gross = (float(gross_eq[-1]) / start) ** (1.0 / years) - 1.0
        cost_drag = cagr_gross - cagr
    else:
        cagr_gross = float("nan")
        cost_drag = float("nan")
    return {
        "cagr": cagr,
        "cagr_gross": cagr_gross,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "turnover": turnover,
        "turnover_sum": turnover_sum,
        "hit_rate": hit_rate,
        "exposure": exposure,
        "time_in_market": time_in_market,
        "cost_drag": cost_drag,
        "cost_sum": float(np.sum(cost)),
        "bars_per_year": bpy,
        "years": years,
        "n_bars": float(ts_ms.size),
        "n_trades": float(trades.height),
        "final_equity": end,
    }


def run_backtest(
    bars: pl.DataFrame,
    positions,
    *,
    costs: CostModel | None = None,
    market: str | None = None,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Apply a target-position series to gold bars.

    ``positions[t]`` is the target decided at the close of bar ``t``. The
    engine fills it on bar ``t+1`` at ``open`` if that value is present and
    positive, otherwise at ``close``. Strategies must not pre-shift.

    Costs are ``|delta position| * (fee + slippage + half-spread)`` in return
    units, charged on the fill bar. ``market`` selects the block in
    ``config/pipeline.yaml``; a passed ``CostModel`` wins. With neither, costs
    are zero.
    """
    if "close" not in bars.columns or "ts_ms" not in bars.columns:
        raise ValueError("bars must have ts_ms and close")
    n = bars.height
    close = _col(bars, "close")
    open_px = _col(bars, "open")
    ts_ms = np.asarray(bars["ts_ms"].to_numpy(), dtype=np.int64)
    if close is None or n == 0:
        raise ValueError("bars are empty or missing close")
    target = _positions(positions, n)
    if costs is None:
        costs = load_cost_model(market) if market else CostModel(0.0, 0.0, 0.0)

    use_open = _open_fill_mask(open_px, n)
    fill_price = np.where(use_open, open_px, close)
    half = _half_spread_frac(bars, fill_price, costs.half_spread_bps)
    cost_rate = (costs.fee_bps + costs.slippage_bps) / _BPS + half

    gross = np.zeros(n, dtype=np.float64)
    cost = np.zeros(n, dtype=np.float64)
    held = np.zeros(n, dtype=np.float64)
    trade_rows: list[dict] = []

    if np.all(use_open[1:] if n > 1 else True):
        # Uniform open fills: vectorised.
        held[1:] = target[:-1]
        prev = np.zeros(n, dtype=np.float64)
        prev[2:] = target[:-2]
        gap = np.zeros(n, dtype=np.float64)
        intra = np.zeros(n, dtype=np.float64)
        valid_gap = (close[:-1] > 0) & np.isfinite(close[:-1]) & np.isfinite(open_px[1:])
        valid_intra = (open_px[1:] > 0) & np.isfinite(open_px[1:]) & np.isfinite(close[1:])
        gap[1:] = np.where(valid_gap, open_px[1:] / close[:-1] - 1.0, 0.0)
        intra[1:] = np.where(valid_intra, close[1:] / open_px[1:] - 1.0, 0.0)
        gross = prev * gap + held * intra
        delta = held - prev
        delta[0] = 0.0
        cost = np.abs(delta) * cost_rate
        cost[0] = 0.0
        for t in np.nonzero(np.abs(delta) > _TRADE_EPS)[0]:
            trade_rows.append(_trade_row(t, ts_ms, fill_price, delta, held, cost))
    elif not np.any(use_open[1:] if n > 1 else False):
        # Uniform close fills: the new target starts earning on the next bar.
        held[2:] = target[:-2]
        new = np.zeros(n, dtype=np.float64)
        new[1:] = target[:-1]
        ret = np.zeros(n, dtype=np.float64)
        valid = (close[:-1] > 0) & np.isfinite(close[:-1]) & np.isfinite(close[1:])
        ret[1:] = np.where(valid, close[1:] / close[:-1] - 1.0, 0.0)
        gross = held * ret
        delta = new - held
        delta[0] = 0.0
        cost = np.abs(delta) * cost_rate
        cost[0] = 0.0
        for t in np.nonzero(np.abs(delta) > _TRADE_EPS)[0]:
            trade_rows.append(_trade_row(t, ts_ms, fill_price, delta, new, cost))
    else:
        pos = 0.0
        for t in range(1, n):
            new_pos = target[t - 1]
            delta = new_pos - pos
            prev_c = close[t - 1]
            if use_open[t] and prev_c > 0 and np.isfinite(prev_c) and np.isfinite(open_px[t]):
                gap = open_px[t] / prev_c - 1.0
                intra = (close[t] / open_px[t] - 1.0) if open_px[t] > 0 and np.isfinite(close[t]) else 0.0
                gross[t] = pos * gap + new_pos * intra
                held[t] = new_pos
                after = new_pos
                pos = new_pos
            else:
                ret = (close[t] / prev_c - 1.0) if prev_c > 0 and np.isfinite(prev_c) and np.isfinite(close[t]) else 0.0
                gross[t] = pos * ret
                held[t] = pos
                after = new_pos
                pos = new_pos
            cost[t] = abs(delta) * cost_rate[t]
            if abs(delta) > _TRADE_EPS:
                trade_rows.append({
                    "ts_ms": int(ts_ms[t]),
                    "fill_price": float(fill_price[t]),
                    "delta": float(delta),
                    "position": float(after),
                    "cost": float(cost[t]),
                })

    net = gross - cost
    equity = np.empty(n, dtype=np.float64)
    equity[0] = float(initial_equity)
    for i in range(1, n):
        equity[i] = equity[i - 1] * (1.0 + net[i])
    trades = pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame(
        schema={
            "ts_ms": pl.Int64,
            "fill_price": pl.Float64,
            "delta": pl.Float64,
            "position": pl.Float64,
            "cost": pl.Float64,
        }
    )
    bar_type = str(bars["bar_type"][0]) if "bar_type" in bars.columns and n else None
    metrics = _metrics(
        ts_ms=ts_ms, net=net, gross=gross, cost=cost, position=held,
        equity=equity, trades=trades, bar_type=bar_type,
    )
    return BacktestResult(
        ts_ms=ts_ms,
        target=target,
        position=held,
        fill_price=fill_price,
        gross_return=gross,
        cost=cost,
        net_return=net,
        equity=equity,
        trades=trades,
        metrics=metrics,
    )


def _trade_row(
    t: int,
    ts_ms: np.ndarray,
    fill_price: np.ndarray,
    delta: np.ndarray,
    position_after: np.ndarray,
    cost: np.ndarray,
) -> dict:
    return {
        "ts_ms": int(ts_ms[t]),
        "fill_price": float(fill_price[t]),
        "delta": float(delta[t]),
        "position": float(position_after[t]),
        "cost": float(cost[t]),
    }


def buy_and_hold(
    bars: pl.DataFrame,
    *,
    costs: CostModel | None = None,
    market: str | None = None,
    long: bool = True,
) -> BacktestResult:
    """Constant unit long (or short) through the same engine and fill rule.

    The first fill is still on bar 1, so this is not a same-bar close-to-close
    from the first stamp. Pass ``costs=CostModel(0, 0, 0)`` for a gross path;
    pass ``market`` to apply the configured entry cost.
    """
    n = bars.height
    pos = np.full(n, 1.0 if long else -1.0, dtype=np.float64)
    return run_backtest(bars, pos, costs=costs, market=market)
