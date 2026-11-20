"""telemetry -- JSONL journal + alert log for the paper-trade runner.

Trade journal: append-only JSONL of order dicts (one per line). Alert log:
plain-text level-tagged messages, also written line-by-line. WARNING and
CRIT alerts mirror to stdout for live visibility.

__contract__:
  inputs:
    - journal_path: Path -- JSONL file (parent dir auto-created)
    - optional alert_path: Path -- defaults to journal_path.with_suffix('.alerts.log')
  outputs:
    - log_trade(order_dict): one JSONL line, fsync-on-write
    - alert(level, msg): one line in alert log + optional stdout mirror
    - summary(): in-memory counts of trades / alerts / halt events
  invariants:
    - journal lines are valid JSON (one object per line)
    - alert log lines have ISO-8601 wall-clock prefix + level tag
    - summary() never raises -- safe to call even with no events
"""
from __future__ import annotations

__contract__ = {
    "kind": "telemetry",
    "owner": "wealth_bot/bot/telemetry",
    "purpose": "JSONL trade journal + leveled alert log",
    "invariants": [
        "journal append-only, one JSON object per line",
        "alert log line-prefixed with ISO-8601 + level tag",
        "WARNING / CRIT alerts mirror to stdout",
        "summary() never raises",
    ],
}

import json
import time
from pathlib import Path
from typing import Any


VALID_LEVELS = {"INFO", "WARNING", "CRIT"}


class Telemetry:
    """Per-run telemetry sink."""

    def __init__(
        self,
        journal_path: str | Path,
        alert_path: str | Path | None = None,
    ) -> None:
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate-on-init -- each run gets a fresh journal.
        self.journal_path.write_text("")

        self.alert_path = (
            Path(alert_path)
            if alert_path is not None
            else self.journal_path.with_suffix(".alerts.log")
        )
        self.alert_path.write_text("")

        self.n_trades_logged = 0
        self.n_alerts_by_level: dict[str, int] = {lvl: 0 for lvl in VALID_LEVELS}
        self.n_halt_events = 0

    # ------------------------------------------------------------------
    # Trade journal
    # ------------------------------------------------------------------
    def log_trade(self, order_dict: dict) -> None:
        """Append one order to the JSONL journal."""
        with open(self.journal_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(order_dict, default=str) + "\n")
        self.n_trades_logged += 1

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------
    def alert(self, level: str, msg: str) -> None:
        """Write a leveled alert; WARNING/CRIT also print to stdout."""
        lvl = level.upper()
        if lvl not in VALID_LEVELS:
            lvl = "INFO"
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"{ts} [{lvl}] {msg}"
        with open(self.alert_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")
        self.n_alerts_by_level[lvl] += 1
        if lvl == "CRIT" and "halt" in msg.lower():
            self.n_halt_events += 1
        if lvl in ("WARNING", "CRIT"):
            print(line, flush=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Return in-memory counts. Never raises."""
        try:
            return {
                "n_trades_logged": int(self.n_trades_logged),
                "n_alerts_by_level": dict(self.n_alerts_by_level),
                "n_halt_events": int(self.n_halt_events),
                "journal_path": str(self.journal_path),
                "alert_path": str(self.alert_path),
            }
        except Exception:
            return {"n_trades_logged": 0, "error": "summary_failed"}
