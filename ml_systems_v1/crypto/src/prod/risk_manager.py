"""
Risk Manager
===============

Portfolio-level and per-asset risk controls for live trading.

Controls:
  1. Portfolio circuit breaker (max drawdown from peak)
  2. Per-asset drawdown limit (from entry price)
  3. Kill switch (file-based emergency halt)
  4. Position size limits (max % per asset)
  5. Capital allocation (% of available USDT)
"""
import logging
from pathlib import Path
from typing import Dict, Optional

from prod.config import (
    MAX_PORTFOLIO_DRAWDOWN, MAX_ASSET_DRAWDOWN,
    CAPITAL_ALLOCATION, MAX_POSITION_PER_ASSET,
    KILL_SWITCH_FILE, SPOT_FEE, SPOT_SLIPPAGE,
)

logger = logging.getLogger("prod.risk")


class RiskManager:
    """Enforces risk limits before order execution."""

    def __init__(self):
        self.peak_equity = 0.0
        self.halted = False
        self.force_exit_all = False  # RED TEAM: force-exit on circuit breaker
        self.halt_reason = ""

    def check_kill_switch(self) -> bool:
        """Check if the kill switch file exists."""
        if KILL_SWITCH_FILE.exists():
            self.halted = True
            self.halt_reason = "KILL_SWITCH file detected"
            logger.critical("KILL SWITCH ACTIVE: %s", KILL_SWITCH_FILE)
            return True
        return False

    def update_equity(self, current_equity: float):
        """Update peak equity tracking."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def check_portfolio_drawdown(self, current_equity: float) -> bool:
        """Check if portfolio drawdown exceeds limit.

        Returns True if trading should be halted.
        """
        if self.peak_equity <= 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > MAX_PORTFOLIO_DRAWDOWN:
            self.halted = True
            self.force_exit_all = True  # RED TEAM FIX: force-exit existing positions
            self.halt_reason = (
                f"Portfolio drawdown {drawdown:.1%} exceeds "
                f"limit {MAX_PORTFOLIO_DRAWDOWN:.1%} -- FORCE EXIT ALL"
            )
            logger.critical("PORTFOLIO CIRCUIT BREAKER: %s", self.halt_reason)
            return True
        return False

    def check_asset_drawdown(self, entry_price: float,
                              current_price: float) -> bool:
        """Check if per-asset drawdown from entry exceeds limit.

        Returns True if position should be closed.
        """
        if entry_price <= 0:
            return False

        drawdown = (entry_price - current_price) / entry_price
        if drawdown > MAX_ASSET_DRAWDOWN:
            logger.warning("Asset DD %.1f%% exceeds limit %.1f%%",
                          drawdown * 100, MAX_ASSET_DRAWDOWN * 100)
            return True
        return False

    def compute_position_size(self, equity: float, signal_position: float,
                                n_active_assets: int) -> float:
        """Compute dollar amount to allocate.

        Args:
            equity: Total portfolio USDT value
            signal_position: Strategy signal (0-1)
            n_active_assets: Number of assets currently active

        Returns:
            Dollar amount to allocate to this asset
        """
        if self.halted:
            return 0.0

        # Available capital
        available = equity * CAPITAL_ALLOCATION

        # Per-asset cap
        max_per_asset = equity * MAX_POSITION_PER_ASSET

        # Scale by signal strength
        target = available * signal_position / max(n_active_assets, 1)

        # Cap
        target = min(target, max_per_asset)

        # Floor -- minimum $10 for Binance
        if target < 10.0:
            return 0.0

        return target

    def should_enter(self, asset: str, signal: Dict,
                      current_equity: float) -> bool:
        """Full pre-entry check.

        Returns True if entry is allowed.
        """
        if self.halted:
            logger.info("Entry blocked for %s: %s", asset, self.halt_reason)
            return False

        if self.check_kill_switch():
            return False

        if self.check_portfolio_drawdown(current_equity):
            return False

        # Only enter on LONG signal
        if signal.get("position", 0) < 0.5:
            return False

        return True

    def should_exit(self, asset: str, entry_price: float,
                     current_price: float, signal: Dict) -> bool:
        """Check if position should be closed.

        Returns True if exit is warranted.
        """
        # Kill switch -> immediate exit
        if self.check_kill_switch():
            return True

        # Signal says flat
        if signal.get("position", 0) < 0.5:
            return True

        # Per-asset drawdown
        if self.check_asset_drawdown(entry_price, current_price):
            return True

        return False

    def reset_halt(self):
        """Manually reset halt state (after investigation)."""
        self.halted = False
        self.halt_reason = ""
        logger.info("Risk halt manually reset")
