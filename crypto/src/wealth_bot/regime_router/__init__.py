"""wealth_bot.regime_router -- PEPE Regime-Routed Deploy Orchestrator.

Modules:
  regime_classifier : past-only 9-cell regime classifier (FULL implementation)
  router            : RegimeRouter orchestrator + routing table (STUB bodies)

Design doc: docs/dossiers/PEPE_REGIME_ROUTER_DEPLOY_SPEC_2026_05_26.md
"""
from .regime_classifier import (
    classify_regime,
    classify_all,
    recalibrate,
    RegimeClassifierConfig,
    VALID_CELLS,
    WARMUP_BARS,
)
from .router import (
    RouteDecision,
    RouterConfig,
    RegimeRouter,
    lookup_route,
    dump_routing_table,
)

__all__ = [
    "classify_regime",
    "classify_all",
    "recalibrate",
    "RegimeClassifierConfig",
    "VALID_CELLS",
    "WARMUP_BARS",
    "RouteDecision",
    "RouterConfig",
    "RegimeRouter",
    "lookup_route",
    "dump_routing_table",
]
