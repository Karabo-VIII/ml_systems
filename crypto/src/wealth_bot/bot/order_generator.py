"""order_generator -- Binance-style paper-trade order dicts.

Each order is a plain dict suitable for JSONL serialization. The simulated
fill applies a slippage = cost_per_side_pct on each side (entry pays buy-side,
exit pays sell-side; round-trip cost = 2 * cost_per_side_pct).

PAPER-TRADE ONLY: no live API calls are wired here. The "order_id" is a uuid
prefix and "filled_at_ms" == "submitted_at_ms" (instant simulated fill).

__contract__:
  inputs:
    - bar_idx, SignalDecision, position_size_usd, current_price, timestamp_ms
    - cost_per_side_pct (from risk config, e.g., 0.22 for 0.22% taker)
  outputs:
    - make_entry / make_exit return Binance-style dict
    - realized_pnl returns decimal pct net of round-trip cost
  invariants:
    - notional_usd > 0 on entries; quantity_token = notional / price
    - filled_price embeds simulated slippage (buy-side higher, sell-side lower)
    - realized_pnl is purely a function of (entry, exit) -- no global state
"""
from __future__ import annotations

__contract__ = {
    "kind": "order_generator",
    "owner": "wealth_bot/bot/order_generator",
    "purpose": "Binance-style paper-trade order dict emitter",
    "invariants": [
        "PAPER-TRADE only -- no live API",
        "quantity_token = notional_usd / current_price",
        "filled_price embeds cost_per_side_pct slippage",
        "realized_pnl computed from filled_prices (already net of slippage)",
    ],
}

import uuid
from typing import Any


class OrderGenerator:
    """Emits Binance-style order dicts for the paper-trade journal."""

    def __init__(
        self,
        symbol: str = "PEPEUSDT",
        cost_per_side_pct: float = 0.22,
    ) -> None:
        self.symbol = symbol
        self.cost_per_side_pct = float(cost_per_side_pct)

    # ------------------------------------------------------------------
    # Entry
    # ------------------------------------------------------------------
    def make_entry(
        self,
        bar_idx: int,
        decision: Any,                # SignalDecision (avoid hard import cycle)
        position_size_usd: float,
        current_price: float,
        timestamp_ms: int,
    ) -> dict:
        if position_size_usd <= 0:
            raise ValueError(f"position_size_usd must be > 0 on entry, got {position_size_usd}")
        if current_price <= 0:
            raise ValueError(f"current_price must be > 0, got {current_price}")

        # Buy-side slippage: pay slightly above the bar close.
        cost = self.cost_per_side_pct / 100.0
        filled_price = current_price * (1.0 + cost)
        quantity_token = position_size_usd / filled_price

        return {
            "order_id": uuid.uuid4().hex[:16],
            "type": "MARKET",
            "side": "BUY",
            "symbol": self.symbol,
            "quantity_token": float(quantity_token),
            "notional_usd": float(position_size_usd),
            "submitted_at_ms": int(timestamp_ms),
            "filled_at_ms": int(timestamp_ms),
            "filled_price": float(filled_price),
            "bar_idx": int(bar_idx),
            "strategy_idx": int(decision.chosen_strategy_idx),
            "predicted_fwd_ret": float(decision.predicted_fwd_ret)
                if decision.predicted_fwd_ret == decision.predicted_fwd_ret  # NaN-safe
                else None,
            "confidence": float(decision.confidence),
        }

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------
    def make_exit(
        self,
        entry_order: dict,
        current_price: float,
        timestamp_ms: int,
        exit_reason: str = "fwd_bars_elapsed",
    ) -> dict:
        if current_price <= 0:
            raise ValueError(f"current_price must be > 0, got {current_price}")

        cost = self.cost_per_side_pct / 100.0
        # Sell-side slippage: receive slightly below the bar close.
        filled_price = current_price * (1.0 - cost)
        quantity_token = entry_order["quantity_token"]
        notional_usd = quantity_token * filled_price

        return {
            "order_id": uuid.uuid4().hex[:16],
            "type": "MARKET",
            "side": "SELL",
            "symbol": self.symbol,
            "quantity_token": float(quantity_token),
            "notional_usd": float(notional_usd),
            "submitted_at_ms": int(timestamp_ms),
            "filled_at_ms": int(timestamp_ms),
            "filled_price": float(filled_price),
            "linked_entry_order_id": entry_order["order_id"],
            "strategy_idx": entry_order["strategy_idx"],
            "exit_reason": exit_reason,
        }

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------
    @staticmethod
    def realized_pnl(entry_order: dict, exit_order: dict) -> float:
        """Decimal pct return on the round-trip (slippage already embedded).

        Both filled_prices already embed cost_per_side_pct slippage, so the
        difference is already net of round-trip cost; no extra subtraction.
        """
        ep = float(entry_order["filled_price"])
        xp = float(exit_order["filled_price"])
        if ep <= 0:
            return 0.0
        return (xp / ep) - 1.0
