"""
Reward Functions for Trading Agent
===================================
All reward functions take (pnl, costs, portfolio_state) and return a scalar reward.

Design principles:
  - Asymmetric: losses hurt more than gains help (capital preservation)
  - Cost-aware: transaction costs are explicit penalties
  - Regime-adaptive: optional drawdown penalty scales with portfolio state
"""

import numpy as np
from collections import deque

from config import (
    REWARD_SCALE,
    REWARD_ASYMMETRY,
    REWARD_COST_PENALTY,
    REWARD_DRAWDOWN_PENALTY,
    REWARD_SHARPE_WINDOW,
    BARS_PER_DAY,
)


class RewardCalculator:
    """Computes trading rewards with asymmetric penalties and cost awareness."""

    def __init__(
        self,
        scale: float = REWARD_SCALE,
        asymmetry: float = REWARD_ASYMMETRY,
        cost_penalty: float = REWARD_COST_PENALTY,
        drawdown_penalty: float = REWARD_DRAWDOWN_PENALTY,
        sharpe_window: int = REWARD_SHARPE_WINDOW,
    ):
        self.scale = scale
        self.asymmetry = asymmetry
        self.cost_penalty = cost_penalty
        self.drawdown_penalty = drawdown_penalty

        # Rolling stats for Sharpe component
        self.returns_buffer = deque(maxlen=sharpe_window)
        self.peak_value = 0.0

    def reset(self, initial_capital: float):
        """Reset state for a new episode."""
        self.returns_buffer.clear()
        self.peak_value = initial_capital

    def compute(
        self,
        pnl: float,
        transaction_cost: float,
        funding_cost: float,
        portfolio_value: float,
    ) -> float:
        """
        Compute the reward for a single step.

        Args:
            pnl: Raw profit/loss from position changes in asset value
            transaction_cost: Cost of trades executed this step (>= 0)
            funding_cost: Holding cost from funding rates (can be + or -)
            portfolio_value: Current total portfolio value

        Returns:
            Scalar reward
        """
        # Costs are subtracted from PnL once (realistic net return).
        # No separate cost penalty -- double-counting caused DoNothing convergence
        # (expected gain ~0.03% vs double-counted cost penalty ~0.40%).
        net_pnl = pnl - transaction_cost - funding_cost

        # Normalize by portfolio value to get return
        if portfolio_value > 0:
            ret = net_pnl / portfolio_value
        else:
            ret = 0.0

        self.returns_buffer.append(ret)

        # --- Asymmetric scaling ---
        # Losses are penalized more heavily than gains are rewarded
        if ret < 0:
            scaled_ret = ret * self.asymmetry
        else:
            scaled_ret = ret

        reward = scaled_ret * self.scale

        # --- Drawdown penalty ---
        # Penalize being in drawdown (encourages recovery and capital preservation)
        self.peak_value = max(self.peak_value, portfolio_value)
        if self.peak_value > 0:
            drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if drawdown > 0.01:  # Only penalize meaningful drawdowns (> 1%)
                reward -= drawdown * self.drawdown_penalty

        return reward

    def get_episode_stats(self) -> dict:
        """Get summary statistics for the episode so far."""
        if len(self.returns_buffer) == 0:
            return {"sharpe": 0.0, "mean_ret": 0.0, "std_ret": 0.0, "max_dd": 0.0}

        returns = np.array(self.returns_buffer)
        mean_ret = returns.mean()
        std_ret = returns.std() + 1e-8

        # Compute max drawdown as fractional decline from peak
        max_dd = 0.0
        if self.peak_value > 0:
            # peak_value tracks the portfolio peak; current is approximated from returns
            cumulative = np.cumprod(1.0 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (running_max - cumulative) / running_max
            max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        return {
            "sharpe": mean_ret / std_ret * np.sqrt(252 * BARS_PER_DAY),  # Annualized
            "mean_ret": mean_ret,
            "std_ret": std_ret,
            "max_dd": max_dd,
        }
