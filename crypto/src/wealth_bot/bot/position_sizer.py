"""position_sizer -- Kelly-fraction sizing on top of SignalDecision.

Quarter-Kelly by default (kelly_fraction=0.25) per config invariants.
Falls back to a fixed pct of capital when rolling stats are unavailable
(< 5 trades observed) so the bot is not paralyzed at cold-start.

__contract__:
  inputs:
    - SignalDecision
    - current capital (USD)
    - rolling win-rate / mean-win / mean-loss (per-trade pct returns, decimals)
    - kelly_fraction (default 0.25 = quarter Kelly)
    - max_position_pct (cap on capital risked per trade)
  outputs: dollar amount to risk on the trade
  invariants:
    - return >= 0 always
    - return <= max_position_pct * current_capital
    - Kelly numerator (p*b - q) clipped to >= 0 -- no negative-edge sizing
    - degenerate inputs (no losses observed, b=inf) -> cap at max_position_pct
"""
from __future__ import annotations

__contract__ = {
    "kind": "position_sizer",
    "owner": "wealth_bot/bot/position_sizer",
    "purpose": "Quarter-Kelly sizing capped by risk.max_position_pct",
    "invariants": [
        "size >= 0",
        "size <= max_position_pct * capital",
        "negative-edge Kelly clipped to zero",
        "cold-start fallback when < 5 trades observed",
    ],
}

from dataclasses import dataclass

from .signal_engine import SignalDecision


@dataclass
class SizingResult:
    """Diagnostic packet alongside the dollar amount."""
    dollar_size: float
    kelly_raw: float            # raw Kelly fraction f* (pre-quarter, pre-cap)
    kelly_applied: float        # fraction actually applied after quarter + cap
    reason: str                 # "kelly" | "cold_start_fallback" | "capped" | "zero_edge"


class PositionSizer:
    """Kelly-fraction sizer with cold-start fallback + max-position cap."""

    def __init__(
        self,
        max_position_pct: float = 1.0,
        cold_start_fallback_pct: float = 0.25,
        min_trades_for_kelly: int = 5,
    ) -> None:
        if not (0.0 < max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0, 1], got {max_position_pct}")
        self.max_position_pct = max_position_pct
        self.cold_start_fallback_pct = cold_start_fallback_pct
        self.min_trades_for_kelly = min_trades_for_kelly

    def size(
        self,
        decision: SignalDecision,
        current_capital: float,
        recent_winrate: float,
        recent_mean_win: float,
        recent_mean_loss: float,
        kelly_fraction: float = 0.25,
        n_observed_trades: int = 0,
    ) -> SizingResult:
        """Return dollar amount to risk.

        Args:
          decision: SignalDecision; if not firing -> zero size.
          current_capital: total bot equity in USD.
          recent_winrate: p in [0, 1].
          recent_mean_win: average pct return of winners (decimal, e.g., 0.04 = +4%).
          recent_mean_loss: average pct return of losers (decimal, NEGATIVE expected, e.g., -0.03).
          kelly_fraction: scale on raw Kelly (default 0.25 = quarter).
          n_observed_trades: count of completed trades observed so far.

        Notes:
          Kelly formula f* = (p*b - q) / b where b = avg_win/|avg_loss|, q = 1 - p.
          When (a) no trades observed yet OR (b) |recent_mean_loss| is ~0,
          we cannot evaluate b safely -- fall back to cold_start_fallback_pct.
        """
        if not decision.fire:
            return SizingResult(0.0, 0.0, 0.0, "no_fire")

        cap_dollars = self.max_position_pct * current_capital

        # Cold-start fallback: not enough history to estimate Kelly reliably.
        if n_observed_trades < self.min_trades_for_kelly:
            fallback = self.cold_start_fallback_pct * current_capital
            applied = min(fallback, cap_dollars)
            return SizingResult(
                dollar_size=float(applied),
                kelly_raw=0.0,
                kelly_applied=applied / current_capital if current_capital > 0 else 0.0,
                reason="cold_start_fallback",
            )

        # Guard against degenerate inputs.
        p = float(max(0.0, min(1.0, recent_winrate)))
        q = 1.0 - p
        avg_win = float(recent_mean_win)
        avg_loss_abs = abs(float(recent_mean_loss))

        if avg_loss_abs <= 1e-9:
            # No observed loss magnitude -- cap to max_position_pct (treat as
            # all-wins edge; conservatively bounded by cap).
            return SizingResult(
                dollar_size=float(cap_dollars),
                kelly_raw=float("inf"),
                kelly_applied=self.max_position_pct,
                reason="capped",
            )

        b = avg_win / avg_loss_abs
        if b <= 0:
            return SizingResult(0.0, 0.0, 0.0, "zero_edge")

        kelly_raw = (p * b - q) / b
        if kelly_raw <= 0:
            return SizingResult(0.0, float(kelly_raw), 0.0, "zero_edge")

        f_applied = kelly_raw * kelly_fraction
        dollars = f_applied * current_capital
        if dollars > cap_dollars:
            return SizingResult(
                dollar_size=float(cap_dollars),
                kelly_raw=float(kelly_raw),
                kelly_applied=self.max_position_pct,
                reason="capped",
            )

        return SizingResult(
            dollar_size=float(dollars),
            kelly_raw=float(kelly_raw),
            kelly_applied=float(f_applied),
            reason="kelly",
        )
