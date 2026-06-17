"""router -- PEPE Regime-Routed Deploy Orchestrator (STUB).

WARNING: THIS IS A DESIGN STUB.
  - The RegimeRouter class skeleton + docstrings are COMPLETE.
  - Execution logic (make_decision, run_paper_trade) bodies are NOT IMPLEMENTED.
  - L3 destructive-ops review required before filling in execution logic.
  - No live API calls exist in this file.  This file is safe to import.

L3 REVIEW STATUS (2026-05-26): PAPER-TRADE-ONLY
  R12 STATIC IS REFUTED_AT_G1 (commit 850d05a; combined +78.22% < +130% G1 floor).
  R12 sub-strategy replaced with R12+WF-LGBM (R60, commit 7203886; PARTIALLY_PASS).
  R23a remains PARTIALLY_PASS (SMALL_POSITION_ONLY). Both candidates fail G3.
  Both wallets use quarter-Kelly (kelly_fraction=0.25). See deploy YAML.

Purpose:
  At each 4h bar, classify the PEPE regime using past-only closes, look up
  the routing decision from the empirical R57a / R57a-followup table, and
  delegate signal/entry/exit to the appropriate sub-strategy (R12+WF-LGBM or R23a).
  HALT cells suppress all new entries and trigger immediate exit of any
  position opened under a now-forbidden regime.

Empirical routing table (OOS + UNSEEN combined, 2026-05-26):
  NOTE: R12 cell performance figures reflect R12 STATIC per-trade data (n=60 trades).
  R12+WF-LGBM per-cell distribution NOT YET MEASURED (pre-live item #10).

  Cell                      | Route         | Basis
  --------------------------+---------------+-------------------------------
  chop_x_low_vol            | R12+WF-LGBM   | R12static +103.7% (n=8) vs R23a +19.3%
  trending_down_x_low_vol   | R12+WF-LGBM   | R12static +32.0% (n=12) vs R23a +27.6%
  trending_up_x_low_vol     | R23a          | R23a +46.1% (n=15) vs R12 +6.8%
  trending_up_x_high_vol    | R23a          | R23a +46.8% (n=6) vs R12 -1.1% [LOW_CONF]
  trending_up_x_med_vol     | R12+WF-LGBM   | R12static +5.9% (n=11) vs R23a -26.6%
  chop_x_med_vol            | HALT          | Both lose (-11.8% / -19.5%)
  trending_down_x_med_vol   | HALT          | Both lose (-4.7% / -10.9%) [default HALT]
  chop_x_high_vol           | DEFER         | LOW_CONF (R23a n=3)
  trending_down_x_high_vol  | DEFER         | LOW_CONF (R23a n=1)

Transition rule (HOLD-TO-EXIT):
  If a position is open under regime A and the regime switches to B:
    - B is HALT/DEFER:  EXIT immediately (end-of-bar close, next bar).
    - B routes to a different strategy (A=R12 -> B=R23a or vice versa):
      HOLD existing position to its own exit signal.  No new entries until
      B's strategy is eligible.  This prevents double-trade confusion.
    - B routes to the SAME strategy as A: continue normally.

__contract__:
  kind: regime_router_stub
  owner: wealth_bot/regime_router/router
  purpose: Orchestrate R12 + R23a sub-strategies via empirical regime routing
  invariants:
    - no live API calls
    - no execution logic in this stub (L3 sign-off required)
    - classifier is past-only (see regime_classifier.py)
    - transition rule documented above enforced at run-time
    - HALT/DEFER cells never produce new entries
    - audit trail: every regime classification logged to JSONL
"""
from __future__ import annotations

__contract__ = {
    "kind": "regime_router_stub",
    "owner": "wealth_bot/regime_router/router",
    "purpose": "Orchestrate R12+WF-LGBM + R23a sub-strategies via empirical regime routing",
    "status": "STUB -- execution bodies not implemented (L3 sign-off required)",
    "l3_review": "PAPER-TRADE-ONLY (2026-05-26) -- both sub-strategies PARTIALLY_PASS; G3 FAIL on both",
    "r12_static_status": "REFUTED_AT_G1 (commit 850d05a) -- R12 route now executes R12+WF-LGBM (R60)",
    "invariants": [
        "no live API calls",
        "classifier is past-only",
        "HALT/DEFER cells never produce new entries",
        "transition rule: HOLD to sub-strategy exit on regime switch",
        "audit trail logged to JSONL at every 4h bar",
        "both wallets use kelly_fraction=0.25 (quarter-Kelly) given G3 FAIL on both sub-strategies",
        "R12 route executes R12+WF-LGBM, NOT R12 static (which is REFUTED_AT_G1)",
    ],
}

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .regime_classifier import (
    RegimeClassifierConfig,
    classify_regime,
    VALID_CELLS,
)


# ---------------------------------------------------------------------------
# Routing table (empirical, from R57a + R57a-followup 2026-05-26)
# L3 REVISED 2026-05-26: R12 static REFUTED_AT_G1; R12 route now executes
# R12+WF-LGBM (R60, PARTIALLY_PASS). Per-cell data still sourced from R12
# static OOS+UNSEEN trades (n=60). WF-LGBM per-cell NOT YET MEASURED.
# ---------------------------------------------------------------------------

class RouteDecision(str, Enum):
    R12 = "R12"       # R12+WF-LGBM (rolling retrain LGBM gating WMA10/30+whale) on 1000PEPEUSDT perp
    R23A = "R23a"     # EMA30_dist>1% + whale_net>0 on PEPEUSDT spot
    HALT = "HALT"     # No new entries; exit existing positions immediately
    DEFER = "DEFER"   # LOW_CONF cell -- no new entries, wait for n to grow


# Map cell -> route + confidence annotation
_ROUTING_TABLE: dict[str, tuple[RouteDecision, str]] = {
    # R12 cells
    "chop_x_low_vol":           (RouteDecision.R12,   "R12 +103.7% n=8 vs R23a +19.3% n=11"),
    "trending_down_x_low_vol":  (RouteDecision.R12,   "R12 +32.0% n=12 vs R23a +27.6% n=11; DD R12 -5.9% < -9.7%"),
    "trending_up_x_med_vol":    (RouteDecision.R12,   "R12 +5.9% n=11 vs R23a -26.6% n=18; R23a KILL CELL"),
    # R23a cells
    "trending_up_x_low_vol":    (RouteDecision.R23A,  "R23a +46.1% n=15 vs R12 +6.8% n=9; 6.8x improvement"),
    "trending_up_x_high_vol":   (RouteDecision.R23A,  "R23a +46.8% n=6 vs R12 -1.1% n=3 [LOW_CONF on n]"),
    # HALT cells
    "chop_x_med_vol":           (RouteDecision.HALT,  "Both lose: R12 -11.8% n=8, R23a -19.5% n=12"),
    "trending_down_x_med_vol":  (RouteDecision.HALT,  "Both lose: R12 -4.7% n=5, R23a -10.9% n=9 [default HALT]"),
    # DEFER cells (LOW_CONF)
    "chop_x_high_vol":          (RouteDecision.DEFER, "R23a +6.6% n=3 LOW_CONF; R12 -1.8% n=2 LOW_CONF"),
    "trending_down_x_high_vol": (RouteDecision.DEFER, "R23a +1.6% n=1 LOW_CONF; R12 +3.9% n=2 LOW_CONF"),
    # Warmup -- no entry
    "WARMUP":                   (RouteDecision.HALT,  "Warmup period: insufficient history for classification"),
}


def lookup_route(cell: str) -> tuple[RouteDecision, str]:
    """Return (RouteDecision, rationale) for a given cell name.

    Raises KeyError if cell is not in the routing table (never happens if
    the classifier always returns a VALID_CELL string).
    """
    if cell not in _ROUTING_TABLE:
        raise KeyError(f"Unknown regime cell: {cell!r}. Valid cells: {sorted(VALID_CELLS)}")
    return _ROUTING_TABLE[cell]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """Top-level regime-router configuration.

    sub_r12_yaml : str
        Path to R12 bot config YAML (src/wealth_bot/configs/pepe_strat_B_perp_candidate.yaml).
    sub_r23a_yaml : str
        Path to R23a bot config YAML (src/wealth_bot/configs/pepe_ema_bot_static_1strat_dist30_lgbm.yaml).
    classifier_cfg : RegimeClassifierConfig
        Regime classifier thresholds.  Defaults to R57a calibration.
    audit_log_path : str
        Path to JSONL audit log.  Every classification + routing decision appended.
    capital_r12_usd : float
        Wallet capital for R12 sub-strategy (e.g. R5000).
    capital_r23a_usd : float
        Wallet capital for R23a sub-strategy (e.g. R5000).
    """
    sub_r12_yaml: str = "src/wealth_bot/configs/pepe_strat_B_perp_candidate.yaml"
    sub_r23a_yaml: str = "src/wealth_bot/configs/pepe_ema_bot_static_1strat_dist30_lgbm.yaml"
    classifier_cfg: RegimeClassifierConfig = field(default_factory=RegimeClassifierConfig)
    audit_log_path: str = "runs/paper_trade/regime_router/regime_audit.jsonl"
    capital_r12_usd: float = 5000.0
    capital_r23a_usd: float = 5000.0


# ---------------------------------------------------------------------------
# RegimeRouter (STUB)
# ---------------------------------------------------------------------------

class RegimeRouter:
    """Regime-routed orchestrator for R12 + R23a sub-strategies.

    All public methods are stubbed.  Docstrings describe the intended contract
    that the implementation must satisfy after L3 sign-off.

    Usage (intended, post-implementation):

        router = RegimeRouter(cfg)
        router.load_sub_strategies()
        result = router.run_paper_trade(df, segment_mask, segment_name="UNSEEN")
        router.write_audit_log()
    """

    def __init__(self, cfg: RouterConfig) -> None:
        """Initialize router with configuration.

        Sets up classifier, loads sub-strategy configs (but does NOT
        instantiate live signal engines -- those require separate call to
        load_sub_strategies()).
        """
        self.cfg = cfg
        self._classifier_cfg = cfg.classifier_cfg
        self._audit_entries: list[dict] = []
        # Sub-strategy handles -- populated by load_sub_strategies()
        self._r12: Any = None
        self._r23a: Any = None
        # Transition state: track which sub-strategy has an open position
        self._open_position_strategy: RouteDecision | None = None
        self._open_position_entry_bar: int | None = None

    def load_sub_strategies(self) -> None:
        """Load R12 and R23a BotConfig objects from their YAML files.

        STUB: implementation must call framework.config.load_config() for each
        sub_yaml, validate that cadence == '4h' and asset == expected, and
        store handles in self._r12 / self._r23a.

        Raises FileNotFoundError if sub-YAML paths do not exist.
        Raises ValueError if cadence or asset mismatch.
        """
        raise NotImplementedError(
            "STUB -- awaiting L3 destructive-ops sign-off before execution logic is filled. "
            "Implementation: load_config(cfg.sub_r12_yaml) + load_config(cfg.sub_r23a_yaml)."
        )

    def classify_bar(self, closes: np.ndarray, t: int) -> tuple[str, RouteDecision, str]:
        """Classify regime at bar t and return routing decision.

        This method IS IMPLEMENTED (classifier is past-only, safe).

        Parameters
        ----------
        closes : np.ndarray
            Full close array.  Only closes[0..t-1] consumed.
        t : int
            Current bar index.

        Returns
        -------
        cell : str
            Regime cell name.
        route : RouteDecision
            Routing decision for this bar.
        rationale : str
            Human-readable explanation.
        """
        cell = classify_regime(closes, t, self._classifier_cfg)
        route, rationale = lookup_route(cell)
        return cell, route, rationale

    def log_classification(self, t: int, timestamp_ms: int,
                           cell: str, route: RouteDecision,
                           rationale: str, action_taken: str) -> None:
        """Append one regime classification event to the audit buffer.

        STUB for the write-to-JSONL path; the dict assembly is complete.
        Callers should call flush_audit_log() at end of session.
        """
        entry = {
            "bar_idx": t,
            "timestamp_ms": timestamp_ms,
            "cell": cell,
            "route": route.value,
            "rationale": rationale,
            "action_taken": action_taken,
        }
        self._audit_entries.append(entry)

    def flush_audit_log(self) -> None:
        """Write audit buffer to JSONL file at cfg.audit_log_path.

        STUB: implementation must use atomic_write or append mode to avoid
        truncating an existing log on crash.

        File format: one JSON object per line (JSONL / ndjson).
        """
        raise NotImplementedError(
            "STUB -- flush_audit_log not yet implemented. "
            "Implementation: open(path, 'a') and json.dumps each entry."
        )

    def apply_transition_rule(
        self,
        t: int,
        new_route: RouteDecision,
        closes: np.ndarray,
    ) -> str:
        """Evaluate regime transition at bar t given a new routing decision.

        Implements the documented HOLD-TO-EXIT transition rule:
          - If no open position: return 'no_position'.
          - If open position under same route as new_route: return 'continue'.
          - If open position under different route AND new_route is R12/R23a:
            return 'hold_to_exit' (keep existing position until its sub-strategy exits).
          - If new_route is HALT or DEFER:
            return 'exit_now' (force immediate exit on next bar open).

        STUB: tracking of self._open_position_strategy must be maintained by
        run_paper_trade() loop; this method only reads that state.
        """
        raise NotImplementedError(
            "STUB -- apply_transition_rule not yet implemented. "
            "See docstring for the four cases."
        )

    def run_paper_trade(
        self,
        df: Any,                   # pd.DataFrame with close + timestamp columns
        segment_mask: np.ndarray,
        segment_name: str = "UNSEEN",
        verbose: bool = True,
    ) -> dict:
        """Walk segment bar-by-bar, classify regime, route to sub-strategy.

        STUB: Implementation must:
          1. At each bar t: call classify_bar(closes, t).
          2. Call apply_transition_rule(t, route, closes).
          3. Depending on action_taken:
               - 'exit_now': call sub-strategy.force_exit(t) and clear open position state.
               - 'hold_to_exit': let active sub-strategy handle naturally.
               - 'continue' or 'no_position': delegate to route's sub-strategy.
          4. call log_classification() with action_taken.
          5. After loop: call flush_audit_log().
          6. Aggregate per-route PnL and return summary dict.

        Returns dict with at minimum:
          n_trades_r12, n_trades_r23a, n_halted_bars, n_defer_bars,
          total_return_pct, max_dd_pct, equity_curve,
          regime_cell_distribution (counts per cell).

        INVARIANTS the implementation must enforce:
          - No new entries in HALT or DEFER cells.
          - At bar t, only closes[0..t-1] used for classification.
          - exit_now completes at t+1 (next bar open), not t (same-bar).
          - No double-trade on transition (only one sub-strategy holds a position at a time).
        """
        raise NotImplementedError(
            "STUB -- run_paper_trade not yet implemented (L3 sign-off required). "
            "See method docstring for the complete implementation contract."
        )


# ---------------------------------------------------------------------------
# Convenience: routing table dump for inspection
# ---------------------------------------------------------------------------

def dump_routing_table() -> list[dict]:
    """Return routing table as a list of dicts for logging / inspection."""
    rows = []
    for cell, (route, rationale) in _ROUTING_TABLE.items():
        rows.append({"cell": cell, "route": route.value, "rationale": rationale})
    return rows


if __name__ == "__main__":
    import json
    print("Routing table:")
    print(json.dumps(dump_routing_table(), indent=2))
    print("\nClassifier smoke test (synthetic data):")
    from .regime_classifier import smoke_test
    print(json.dumps(smoke_test(), indent=2))
