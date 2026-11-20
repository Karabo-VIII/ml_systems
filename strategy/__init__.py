"""Minimal strategy layer: signals in, a vectorised backtest out.

The engine, not the strategy, delays a close-labelled target by one bar and
fills it at the next open (or the next close if ``open`` is missing). Costs
and the cadence-aware embargo live in ``config/pipeline.yaml``.
"""

from strategy.engine import (
    BacktestResult,
    CostModel,
    Fold,
    buy_and_hold,
    cadence_embargo_bars,
    load_cost_model,
    run_backtest,
    walk_forward,
)
from strategy.signals import donchian_breakout, regime_gate, vol_scaled_momentum

__all__ = [
    "BacktestResult",
    "CostModel",
    "Fold",
    "buy_and_hold",
    "cadence_embargo_bars",
    "donchian_breakout",
    "load_cost_model",
    "regime_gate",
    "run_backtest",
    "vol_scaled_momentum",
    "walk_forward",
]
