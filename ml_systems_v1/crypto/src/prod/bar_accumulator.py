"""
Dollar Bar Accumulator
========================

Accumulates aggTrade data into dollar bars in real-time.
When cumulative dollar volume crosses the threshold, a bar is emitted.

Two modes:
  1. WebSocket: stream aggTrades via Binance WS (primary)
  2. REST poll: fetch recent aggTrades via REST (fallback)

The accumulator maintains a rolling buffer of completed bars for
feature computation and strategy signal generation.
"""
import time
import json
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Callable, List

import numpy as np

from prod.config import (
    BAR_BUFFER_SIZE, get_dollar_threshold,
    ws_symbol, POLL_INTERVAL_SECONDS,
)

logger = logging.getLogger("prod.bar_accumulator")


class InProgressBar:
    """State of the bar currently being accumulated."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.open = 0.0
        self.high = -1e18
        self.low = 1e18
        self.close = 0.0
        self.volume = 0.0         # Base asset volume
        self.volume_usd = 0.0     # Dollar volume
        self.buy_vol = 0.0        # Dollar buy volume
        self.sell_vol = 0.0       # Dollar sell volume
        self.tick_count = 0
        self.first_timestamp = 0
        self.last_timestamp = 0
        self.has_data = False

    def add_trade(self, price: float, qty: float, timestamp_ms: int,
                  is_buyer_maker: bool):
        """Add a single trade to the in-progress bar."""
        dollar_value = price * qty

        if not self.has_data:
            self.open = price
            self.first_timestamp = timestamp_ms
            self.has_data = True

        if price > self.high:
            self.high = price
        if price < self.low:
            self.low = price

        self.close = price
        self.volume += qty
        self.volume_usd += dollar_value
        self.tick_count += 1
        self.last_timestamp = timestamp_ms

        # Binance: is_buyer_maker=True means the buyer was the maker,
        # so the trade was a SELL aggressor
        if is_buyer_maker:
            self.sell_vol += dollar_value
        else:
            self.buy_vol += dollar_value

    def to_dict(self, bar_id: int) -> Dict:
        """Convert completed bar to dict matching pipeline schema."""
        return {
            "bar_id": bar_id,
            "timestamp": self.last_timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "volume_usd": self.volume_usd,
            "buy_vol": self.buy_vol,
            "sell_vol": self.sell_vol,
            "tick_count": self.tick_count,
        }


class BarAccumulator:
    """Accumulates trades into dollar bars for a single asset.

    When a bar completes (dollar volume >= threshold), it's appended
    to the rolling buffer and the on_bar_complete callback fires.
    """

    def __init__(self, asset: str, on_bar_complete: Callable = None):
        """
        Args:
            asset: e.g., 'btcusdt'
            on_bar_complete: callback(asset, bar_dict, buffer) when bar closes
        """
        self.asset = asset
        self.threshold = get_dollar_threshold(asset)
        self.on_bar_complete = on_bar_complete

        # Rolling buffer of completed bars
        self.buffer: deque = deque(maxlen=BAR_BUFFER_SIZE)

        # Current in-progress bar
        self.current_bar = InProgressBar()
        self.next_bar_id = 0

        # Stats
        self.total_trades = 0
        self.total_bars = 0

        logger.info("%s: threshold=$%s, buffer=%d",
                    asset.upper(), f"{self.threshold:,.0f}", BAR_BUFFER_SIZE)

    def process_trade(self, price: float, qty: float, timestamp_ms: int,
                      is_buyer_maker: bool):
        """Process a single aggTrade. May emit 0 or more completed bars."""
        self.total_trades += 1

        # Add to current bar
        self.current_bar.add_trade(price, qty, timestamp_ms, is_buyer_maker)

        # Check if bar is complete
        while self.current_bar.volume_usd >= self.threshold:
            # Bar is complete -- emit it
            bar_dict = self.current_bar.to_dict(self.next_bar_id)
            self.buffer.append(bar_dict)
            self.next_bar_id += 1
            self.total_bars += 1

            # Calculate overflow (trades that spill into next bar)
            overflow = self.current_bar.volume_usd - self.threshold

            # Reset for next bar
            last_price = self.current_bar.close
            last_ts = self.current_bar.last_timestamp
            self.current_bar.reset()

            # If significant overflow, seed the next bar with residual
            # (simplified: we can't split a single trade, so we just
            # note the overflow. For production accuracy, you'd need
            # to handle partial trade attribution, but the error is
            # small relative to threshold)
            if overflow > self.threshold * 0.01:
                logger.debug("%s: bar overflow $%.0f (%.1f%% of threshold)",
                            self.asset, overflow,
                            overflow / self.threshold * 100)

            # Notify
            if self.on_bar_complete:
                self.on_bar_complete(self.asset, bar_dict, self.buffer)

            break  # One trade can't produce 2+ bars in practice

    def process_trades_batch(self, trades: List[Dict]):
        """Process a batch of trades (from REST poll).

        Each trade dict: {price, qty, timestamp, is_buyer_maker}
        """
        for t in trades:
            self.process_trade(
                float(t["price"]),
                float(t["qty"]),
                int(t["timestamp"]),
                bool(t["is_buyer_maker"]),
            )

    def get_buffer_arrays(self) -> Optional[Dict[str, np.ndarray]]:
        """Convert buffer to numpy arrays matching strategy_lab format.

        Returns None if buffer has fewer than 100 bars.
        """
        if len(self.buffer) < 100:
            return None

        bars = list(self.buffer)
        n = len(bars)

        return {
            "n": n,
            "close": np.array([b["close"] for b in bars], dtype=np.float64),
            "open": np.array([b["open"] for b in bars], dtype=np.float64),
            "high": np.array([b["high"] for b in bars], dtype=np.float64),
            "low": np.array([b["low"] for b in bars], dtype=np.float64),
            "volume": np.array([b["volume"] for b in bars], dtype=np.float64),
            "volume_usd": np.array([b["volume_usd"] for b in bars], dtype=np.float64),
            "buy_vol": np.array([b["buy_vol"] for b in bars], dtype=np.float64),
            "sell_vol": np.array([b["sell_vol"] for b in bars], dtype=np.float64),
            "tick_count": np.array([b["tick_count"] for b in bars], dtype=np.float64),
            "timestamp": np.array([b["timestamp"] for b in bars], dtype=np.int64),
        }

    def seed_from_chimera(self, n_bars: int = 2000) -> int:
        """Pre-seed buffer from chimera parquet (cold-start elimination).

        Loads the last n_bars from the processed chimera file so the model
        can make decisions immediately without waiting for live warmup.

        Returns number of bars seeded.
        """
        import polars as pl

        chimera_path = (Path(__file__).resolve().parent.parent.parent
                        / "data" / "processed"
                        / f"{self.asset.upper()}_v50_chimera.parquet")

        if not chimera_path.exists():
            logger.warning("%s: No chimera file for seeding at %s",
                          self.asset.upper(), chimera_path)
            return 0

        try:
            df = pl.read_parquet(chimera_path).sort("timestamp")
            n = len(df)
            start = max(0, n - n_bars)
            df_tail = df.slice(start, n - start)

            count = 0
            for row in df_tail.iter_rows(named=True):
                bar_dict = {
                    "bar_id": self.next_bar_id,
                    "timestamp": int(row.get("timestamp", 0)),
                    "open": float(row.get("open", 0)),
                    "high": float(row.get("high", 0)),
                    "low": float(row.get("low", 0)),
                    "close": float(row.get("close", 0)),
                    "volume": float(row.get("volume", 0)),
                    "volume_usd": float(row.get("volume_usd", 0)),
                    "buy_vol": float(row.get("buy_vol", 0)),
                    "sell_vol": float(row.get("sell_vol", 0)),
                    "tick_count": int(row.get("tick_count", 0)),
                }
                self.buffer.append(bar_dict)
                self.next_bar_id += 1
                count += 1

            self.total_bars = count
            logger.info("%s: Seeded %d bars from chimera (latest: %s)",
                       self.asset.upper(), count,
                       datetime.utcfromtimestamp(
                           self.buffer[-1]["timestamp"] / 1000
                       ).strftime("%Y-%m-%d %H:%M") if self.buffer else "N/A")
            return count
        except Exception as e:
            logger.error("%s: Failed to seed from chimera: %s",
                        self.asset.upper(), e)
            return 0

    @property
    def bars_in_buffer(self) -> int:
        return len(self.buffer)

    @property
    def bar_in_progress_pct(self) -> float:
        """How full is the current bar (0-100%)."""
        if self.threshold <= 0:
            return 0
        return min(self.current_bar.volume_usd / self.threshold * 100, 100)


class MultiAssetAccumulator:
    """Manages BarAccumulators for multiple assets."""

    def __init__(self, assets: List[str],
                 on_bar_complete: Callable = None):
        self.accumulators = {}
        for asset in assets:
            self.accumulators[asset] = BarAccumulator(
                asset, on_bar_complete=on_bar_complete)

    def get(self, asset: str) -> Optional[BarAccumulator]:
        return self.accumulators.get(asset)

    def status(self) -> Dict:
        """Return status summary for all assets."""
        return {
            asset: {
                "bars": acc.bars_in_buffer,
                "total_bars": acc.total_bars,
                "total_trades": acc.total_trades,
                "current_bar_pct": round(acc.bar_in_progress_pct, 1),
            }
            for asset, acc in self.accumulators.items()
        }
