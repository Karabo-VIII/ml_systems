"""
Exchange Wrapper
=================

Binance SPOT connection via ccxt with retry logic, precision formatting,
and structured logging. Ported from TMP/4 live engine with improvements.
"""
import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict

try:
    import ccxt
except ImportError:
    ccxt = None

from prod.config import (
    EXCHANGE_ID, USE_TESTNET, SPOT_MODE,
    PROJECT_ROOT, LOG_DIR,
)

logger = logging.getLogger("prod.exchange")


class ExchangeWrapper:
    """Binance SPOT exchange interface with safety features."""

    def __init__(self, testnet: bool = None, paper: bool = False):
        """
        Args:
            testnet: Override USE_TESTNET config. None = use config default.
            paper: If True, simulate all orders (no real exchange calls).
        """
        if ccxt is None:
            raise ImportError(
                "ccxt not installed. Run: pip install ccxt python-dotenv")

        self.paper = paper
        self.testnet = testnet if testnet is not None else USE_TESTNET

        # Load API keys from .env
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=str(PROJECT_ROOT / ".env"))
        except ImportError:
            logger.warning("python-dotenv not installed, using env vars directly")

        if self.testnet:
            api_key = os.getenv("BINANCE_TESTNET_API_KEY", "")
            api_secret = os.getenv("BINANCE_TESTNET_API_SECRET", "")
        else:
            api_key = os.getenv("BINANCE_API_KEY", "")
            api_secret = os.getenv("BINANCE_API_SECRET", "")

        if not api_key and not paper:
            logger.warning("No API key found. Exchange calls will fail.")

        self.exchange = None
        self.public_exchange = None  # For paper mode trade data fetching

        if not paper:
            exchange_class = getattr(ccxt, EXCHANGE_ID)
            self.exchange = exchange_class({
                "apiKey": api_key,
                "secret": api_secret,
                "options": {"defaultType": "spot"},
                "enableRateLimit": True,
            })
            if self.testnet:
                self.exchange.set_sandbox_mode(True)

            self.exchange.load_markets()
            mode_str = "TESTNET" if self.testnet else "MAINNET"
            logger.info("Connected to Binance %s (SPOT)", mode_str)
        else:
            # Paper mode: create public-only client for fetching trade data
            # No API keys needed -- just public aggTrades endpoint
            try:
                exchange_class = getattr(ccxt, EXCHANGE_ID)
                self.public_exchange = exchange_class({
                    "options": {"defaultType": "spot"},
                    "enableRateLimit": True,
                })
                self.public_exchange.load_markets()
                logger.info("PAPER mode -- public data feed connected")
            except Exception as e:
                logger.warning("PAPER mode -- no data feed: %s", e)
                self.public_exchange = None

        # Trade log
        self._trade_log_path = LOG_DIR / "trades"
        self._trade_log_path.mkdir(parents=True, exist_ok=True)

    def get_usdt_balance(self) -> float:
        """Get free USDT balance."""
        if self.paper:
            return 0.0
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0.0))
        except Exception as e:
            logger.error("Failed to fetch USDT balance: %s", e)
            return 0.0

    def get_total_equity(self) -> float:
        """Get total portfolio equity in USDT (including held assets)."""
        if self.paper:
            return 0.0
        try:
            balance = self.exchange.fetch_balance()
            total = float(balance.get("total", {}).get("USDT", 0.0))
            for asset, amount in balance.get("total", {}).items():
                if asset != "USDT" and amount > 1e-8:
                    try:
                        ticker = self.exchange.fetch_ticker(f"{asset}/USDT")
                        total += amount * ticker["last"]
                    except Exception:
                        continue
            return total
        except Exception as e:
            logger.error("Failed to fetch total equity: %s", e)
            return 0.0

    def get_open_positions(self) -> Dict[str, float]:
        """Get all non-USDT holdings {asset: amount}."""
        if self.paper:
            return {}
        try:
            balance = self.exchange.fetch_balance()
            return {
                asset: float(amount)
                for asset, amount in balance.get("total", {}).items()
                if float(amount) > 1e-8 and asset != "USDT"
            }
        except Exception as e:
            logger.error("Failed to fetch open positions: %s", e)
            return {}

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get last price for a ccxt symbol (e.g., 'BTC/USDT')."""
        if self.paper:
            return None
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error("Failed to fetch price for %s: %s", symbol, e)
            return None

    def place_market_order(self, symbol: str, side: str, amount: float,
                           reason: str = "") -> Optional[Dict]:
        """Place a market order with safety checks.

        Args:
            symbol: ccxt format, e.g., 'BTC/USDT'
            side: 'buy' or 'sell'
            amount: Asset quantity (not USDT value)
            reason: Human-readable reason for audit log

        Returns:
            Order dict on success, None on failure.
        """
        # Paper mode -- simulate
        if self.paper:
            order = {
                "id": f"paper_{int(time.time()*1000)}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "filled": amount,
                "price": 0.0,  # Would need live price
                "status": "filled",
                "paper": True,
            }
            self._log_trade(order, reason)
            logger.info("PAPER %s %s %.6f %s [%s]",
                        side.upper(), symbol, amount, reason, order["id"])
            return order

        # Real exchange
        market_info = self.exchange.markets.get(symbol)
        if not market_info:
            logger.error("Market not found: %s", symbol)
            return None

        # Min amount check
        min_amount = market_info.get("limits", {}).get("amount", {}).get("min")
        if min_amount is not None and amount < min_amount:
            logger.warning("Amount %.8f below minimum %.8f for %s",
                          amount, min_amount, symbol)
            return None

        # Precision formatting
        formatted = self.exchange.amount_to_precision(symbol, amount)
        if float(formatted) <= 0:
            logger.warning("Formatted amount '%s' too small for %s",
                          formatted, symbol)
            return None

        # Min cost check ($10 Binance minimum)
        min_cost = market_info.get("limits", {}).get("cost", {}).get("min", 10)
        price = self.get_current_price(symbol)
        if price and float(formatted) * price < min_cost:
            logger.warning("Order cost $%.2f below minimum $%.2f for %s",
                          float(formatted) * price, min_cost, symbol)
            return None

        # Execute with retry
        for attempt in range(3):
            try:
                logger.info("Placing MARKET %s %.6f %s (attempt %d) [%s]",
                           side.upper(), float(formatted), symbol,
                           attempt + 1, reason)
                order = self.exchange.create_market_order(
                    symbol, side, formatted)
                self._log_trade(order, reason)
                logger.info("Order filled: %s %s %.6f @ %.4f [%s]",
                           side.upper(), symbol,
                           order.get("filled", 0),
                           order.get("price", 0),
                           order["id"])
                return order
            except ccxt.InsufficientFunds as e:
                logger.error("Insufficient funds for %s %s: %s",
                            side, symbol, e)
                return None
            except ccxt.NetworkError as e:
                logger.warning("Network error (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                continue
            except Exception as e:
                logger.error("Order failed: %s", e)
                return None

        logger.error("All 3 order attempts failed for %s %s", side, symbol)
        return None

    def verify_position_closed(self, base_asset: str,
                                max_wait: int = 60) -> bool:
        """Poll exchange to verify a position is fully closed.

        Args:
            base_asset: e.g., 'BTC'
            max_wait: Maximum seconds to wait

        Returns:
            True if position is confirmed closed.
        """
        if self.paper:
            return True
        for i in range(max_wait // 5):
            time.sleep(5)
            positions = self.get_open_positions()
            if base_asset not in positions:
                return True
        return False

    def sell_all(self) -> bool:
        """Emergency liquidation of all non-USDT holdings."""
        positions = self.get_open_positions()
        if not positions:
            logger.info("No positions to liquidate")
            return True

        success = True
        for asset, amount in positions.items():
            symbol = f"{asset}/USDT"
            if self.exchange and symbol in self.exchange.markets:
                order = self.place_market_order(
                    symbol, "sell", amount, reason="EMERGENCY_LIQUIDATION")
                if order is None:
                    logger.error("Failed to liquidate %s", symbol)
                    success = False
            else:
                logger.warning("Cannot liquidate %s -- market not found", symbol)
        return success

    def _log_trade(self, order: Dict, reason: str):
        """Append trade to JSONL audit log."""
        try:
            log_file = self._trade_log_path / "trade_log.jsonl"
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "order_id": order.get("id"),
                "symbol": order.get("symbol"),
                "side": order.get("side"),
                "amount": order.get("filled", order.get("amount")),
                "price": order.get("price"),
                "status": order.get("status"),
                "reason": reason,
                "paper": order.get("paper", False),
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to log trade: %s", e)
