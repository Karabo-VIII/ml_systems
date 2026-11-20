"""risk_manager -- kill switch + tripwires for the paper-trade runner.

Three independent tripwires:
  1. Rolling drawdown vs peak equity exceeds max_drawdown_pct
  2. Consecutive losing trades exceeds max_consecutive_losses
  3. Whale-data age (hours since last whale-flow observation) exceeds
     whale_freshness_max_hours -- stale whale data invalidates the filter
     and downgrades the bot to "trade blind", which the static rule was NOT
     validated under.

__contract__:
  inputs:
    - equity_curve: list[float] of running USD equity values
    - recent_trade_pnls: list[float] pct returns (decimals)
    - whale_data_age_hours: float
  outputs: tuple (should_halt: bool, reason: str)
  invariants:
    - kill switch is monotonic -- once halted, never un-halts (caller responsibility)
    - empty equity_curve returns (False, "no_equity_history")
    - max_drawdown_pct is positive (e.g., 25.0 means 25% drop from peak)
"""
from __future__ import annotations

__contract__ = {
    "kind": "risk_manager",
    "owner": "wealth_bot/bot/risk_manager",
    "purpose": "Independent kill-switch tripwires for paper-trade runner",
    "invariants": [
        "halt is monotonic (caller never un-halts)",
        "DD measured from running peak, not entry capital",
        "consecutive-loss counter resets on a winner",
        "whale freshness gate compares to risk.whale_freshness_max_hours",
    ],
}


class RiskManager:
    """Stateless evaluator -- caller passes the rolling state in each call."""

    def __init__(
        self,
        max_drawdown_pct: float = 25.0,
        max_consecutive_losses: int = 10,
        whale_freshness_max_hours: float = 28.0,
    ) -> None:
        if max_drawdown_pct <= 0:
            raise ValueError(f"max_drawdown_pct must be > 0, got {max_drawdown_pct}")
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.whale_freshness_max_hours = float(whale_freshness_max_hours)

    def current_drawdown_pct(self, equity_curve: list[float]) -> float:
        """Return current DD vs running peak (positive number, e.g., 18.4 means -18.4%)."""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        for e in equity_curve:
            if e > peak:
                peak = e
        if peak <= 0:
            return 0.0
        current = equity_curve[-1]
        dd = (peak - current) / peak * 100.0
        return float(max(0.0, dd))

    def check_consecutive_loss(self, recent_trade_pnls: list[float]) -> bool:
        """True if the last `max_consecutive_losses` trades were ALL losers.

        Strictly negative-or-zero counts as a loss (a zero-pnl trade is a
        cost-paid round-trip with no edge, treat as a loss for the tripwire).
        """
        if len(recent_trade_pnls) < self.max_consecutive_losses:
            return False
        tail = recent_trade_pnls[-self.max_consecutive_losses:]
        return all(p <= 0 for p in tail)

    def check_kill_switch(
        self,
        equity_curve: list[float],
        whale_data_age_hours: float,
        recent_trade_pnls: list[float] | None = None,
    ) -> tuple[bool, str]:
        """Return (should_halt, reason).

        Evaluates the three tripwires in order. First trigger wins; caller
        should stop the loop immediately on True.
        """
        if not equity_curve:
            return False, "no_equity_history"

        dd = self.current_drawdown_pct(equity_curve)
        if dd > self.max_drawdown_pct:
            return True, f"max_drawdown_exceeded: {dd:.2f}% > {self.max_drawdown_pct:.2f}%"

        if recent_trade_pnls is not None and self.check_consecutive_loss(recent_trade_pnls):
            return True, (
                f"max_consecutive_losses_exceeded: "
                f"last {self.max_consecutive_losses} trades all <= 0"
            )

        if whale_data_age_hours > self.whale_freshness_max_hours:
            return True, (
                f"whale_data_stale: age={whale_data_age_hours:.1f}h > "
                f"{self.whale_freshness_max_hours:.1f}h"
            )

        return False, ""
