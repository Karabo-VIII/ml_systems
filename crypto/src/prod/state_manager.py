"""
State Manager
===============

Persists trading state to JSON between restarts.
Handles position tracking, equity history, and trade log.

State file is atomically written (write temp -> rename) to prevent
corruption on crash.
"""
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List

from prod.config import STATE_DIR, LOG_DIR

logger = logging.getLogger("prod.state")


class Position:
    """Represents an open position for one asset."""

    def __init__(self, symbol: str, entry_price: float, amount: float,
                 entry_time: str = None, strategy: str = ""):
        self.symbol = symbol
        self.entry_price = entry_price
        self.amount = amount
        self.entry_time = entry_time or datetime.now(timezone.utc).isoformat()
        self.strategy = strategy

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "amount": self.amount,
            "entry_time": self.entry_time,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "Position":
        return cls(
            symbol=d["symbol"],
            entry_price=d["entry_price"],
            amount=d["amount"],
            entry_time=d.get("entry_time"),
            strategy=d.get("strategy", ""),
        )


class StateManager:
    """Manages persistent trading state."""

    def __init__(self, name: str = "default"):
        """
        Args:
            name: Instance name (for running multiple bots).
        """
        self.name = name
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        self.state_path = STATE_DIR / f"state_{name}.json"
        self.equity_log_path = LOG_DIR / f"equity_{name}.jsonl"

        # Core state
        self.positions: Dict[str, Position] = {}  # {asset: Position}
        self.peak_equity = 0.0
        self.last_equity = 0.0
        self.total_trades = 0
        self.total_pnl = 0.0
        self.start_time = datetime.now(timezone.utc).isoformat()

        # Load existing state
        self._load()

    def _load(self):
        """Load state from disk."""
        if not self.state_path.exists():
            logger.info("No existing state at %s, starting fresh", self.state_path)
            return

        try:
            with open(self.state_path) as f:
                data = json.load(f)

            for asset, pos_dict in data.get("positions", {}).items():
                self.positions[asset] = Position.from_dict(pos_dict)

            self.peak_equity = data.get("peak_equity", 0.0)
            self.last_equity = data.get("last_equity", 0.0)
            self.total_trades = data.get("total_trades", 0)
            self.total_pnl = data.get("total_pnl", 0.0)
            self.start_time = data.get("start_time", self.start_time)

            n_pos = sum(1 for p in self.positions.values() if p.amount > 0)
            logger.info("Loaded state: %d positions, equity=$%.2f, "
                       "%d total trades",
                       n_pos, self.last_equity, self.total_trades)

        except Exception as e:
            logger.error("Failed to load state: %s", e)

    def save(self):
        """Atomically save state to disk."""
        data = {
            "name": self.name,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "start_time": self.start_time,
            "peak_equity": self.peak_equity,
            "last_equity": self.last_equity,
            "total_trades": self.total_trades,
            "total_pnl": self.total_pnl,
            "positions": {
                asset: pos.to_dict()
                for asset, pos in self.positions.items()
                if pos.amount > 0
            },
        }

        # Atomic write: write to temp, then rename
        tmp_path = self.state_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w") as f:
                json.dump(data, f, indent=2)
            shutil.move(str(tmp_path), str(self.state_path))
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    def record_entry(self, asset: str, symbol: str, entry_price: float,
                      amount: float, strategy: str = ""):
        """Record a new position entry."""
        self.positions[asset] = Position(
            symbol=symbol,
            entry_price=entry_price,
            amount=amount,
            strategy=strategy,
        )
        self.total_trades += 1
        self.save()
        logger.info("ENTRY: %s %.6f @ $%.4f [%s]",
                    symbol, amount, entry_price, strategy)

    def record_exit(self, asset: str, exit_price: float):
        """Record a position exit and compute PnL."""
        pos = self.positions.get(asset)
        if pos is None or pos.amount <= 0:
            return

        pnl_pct = (exit_price / pos.entry_price - 1) * 100
        pnl_usd = pos.amount * (exit_price - pos.entry_price)
        self.total_pnl += pnl_usd

        logger.info("EXIT: %s @ $%.4f (entry $%.4f, PnL %+.2f%% / $%+.2f)",
                    pos.symbol, exit_price, pos.entry_price,
                    pnl_pct, pnl_usd)

        # Clear position
        pos.amount = 0
        self.save()

    def update_equity(self, equity: float):
        """Update equity tracking and log."""
        self.last_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Append to equity log
        try:
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "equity": round(equity, 2),
                "peak": round(self.peak_equity, 2),
                "drawdown_pct": round(
                    (self.peak_equity - equity) / self.peak_equity * 100
                    if self.peak_equity > 0 else 0, 2),
                "n_positions": sum(
                    1 for p in self.positions.values() if p.amount > 0),
            }
            with open(self.equity_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to log equity: %s", e)

        self.save()

    def get_position(self, asset: str) -> Optional[Position]:
        """Get current position for an asset (or None)."""
        pos = self.positions.get(asset)
        if pos and pos.amount > 0:
            return pos
        return None

    def has_position(self, asset: str) -> bool:
        """Check if we have an open position for an asset."""
        return self.get_position(asset) is not None

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return {a: p for a, p in self.positions.items() if p.amount > 0}

    def status_summary(self) -> str:
        """Return a human-readable status string."""
        positions = self.get_all_positions()
        lines = [
            f"Equity: ${self.last_equity:,.2f} (peak: ${self.peak_equity:,.2f})",
            f"Trades: {self.total_trades} | PnL: ${self.total_pnl:+,.2f}",
            f"Positions: {len(positions)}",
        ]
        for asset, pos in positions.items():
            lines.append(f"  {pos.symbol}: {pos.amount:.6f} @ ${pos.entry_price:.4f}")
        return "\n".join(lines)
