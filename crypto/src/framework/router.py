"""src/framework/router.py -- the METHOD ROUTER: classify a problem to Layer A/B/C + method + adapter.

Three isolated solutioning layers each with their own routing, cross-pollinating via the bus:
  Layer A -- GAMES / self-play  (projects/chess_zero, GameAdapter pipeline)
  Layer B -- General WM / forecasting / time-series / science
  Layer C -- Crypto WM  (src/wm + src/strat)

Shared spine: src/strat/battery.py + CDAP (domain-general -- every layer passes through it).

The routing is a DETERMINISTIC cascade over the `problem` dict keys:
  name, has_exact_simulator (bool), action_space (str), objective_type (str), domain (str),
  + optional: stochastic_transitions (bool), imperfect_information (bool),
              data_budget (int), sim_budget (int).

Canonical trees sourced from:
  docs/GENERAL_PROBLEM_SOLVING_HARNESS_2026_06_09.md (method router table)
  projects/chess_zero/docs/ENGINE_AGNOSTIC_FRAMEWORK.md (games decision tree)

No emoji (Windows cp1252 safety).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """The output of `route()`. All fields are plain-text so they can be serialised / logged."""
    layer: str                   # "A", "B", or "C"
    method: str                  # e.g. "alphazero", "supervised_seq_wm", "geometric_dl"
    variant: str                 # sub-variant inside the method family
    adapter_cls: str             # dotted import path to the adapter class
    rationale: str               # 1-2 sentence human-readable justification
    warnings: list[str] = field(default_factory=list)  # non-fatal notices (unimplemented, etc.)


# ---------------------------------------------------------------------------
# Routing constants
# ---------------------------------------------------------------------------

_LAYER_A_ADAPTER = "games.az.game_adapter.GameAdapter"
_LAYER_B_ADAPTER = "src.framework.general_adapter.GeneralAdapter"
_LAYER_C_ADAPTER = "src.framework.crypto_adapter.CryptoAdapter"

_VALID_ACTION_SPACES = {"discrete_small", "discrete_large", "continuous", "none"}
_VALID_OBJECTIVE_TYPES = {"decision", "forecasting", "structured_prediction"}
_VALID_DOMAINS = {"games", "time_series", "crypto", "science"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn(warns: list[str], msg: str) -> None:
    warns.append(msg)
    warnings.warn(f"[router] {msg}", stacklevel=3)


def _route_games(p: dict, warns: list[str]) -> RoutingResult:
    """Layer A sub-router: follow the ENGINE_AGNOSTIC_FRAMEWORK.md decision tree."""
    # Default True here is intentional: once we know a problem is games-domain,
    # having an exact simulator is the expected case (chess/go/tic-tac-toe).
    # The outer route() defaults to False so that unspecified non-game problems
    # never accidentally hit this branch -- see the _domain_is_game_compatible guard.
    has_sim = bool(p.get("has_exact_simulator", True))  # games sub-router default: yes
    action = p.get("action_space", "discrete_small")
    stochastic = bool(p.get("stochastic_transitions", False))
    imperfect = bool(p.get("imperfect_information", False))
    sim_budget = p.get("sim_budget")       # int or None
    tight_sims = sim_budget is not None and int(sim_budget) <= 32

    if not has_sim:
        # Per ENGINE_AGNOSTIC_FRAMEWORK.md Q1 branch:
        #   data_budget tiny (<=100k steps) -> EfficientZero V2
        #   else                            -> MuZero
        data_budget = p.get("data_budget")
        _tiny = False
        if isinstance(data_budget, str):
            _tiny = data_budget.lower() == "tiny"
        elif data_budget is not None:
            try:
                _tiny = int(data_budget) <= 100_000
            except (TypeError, ValueError):
                pass
        if _tiny:
            variant = "efficient_zero_v2"
            rationale = (
                "No exact simulator + tiny data budget (<=100k steps): EfficientZero V2. "
                "Data-efficient model-based RL that learns a world model from scarce experience "
                "(discrete OR continuous action spaces)."
            )
        else:
            variant = "muzero"
            rationale = (
                "No exact simulator: MuZero (learns dynamics model in latent space). "
                "For scarce data (<=100k steps) prefer EfficientZero V2 -- "
                "pass data_budget='tiny' or data_budget=<int<=100000> to trigger that branch."
            )
        return RoutingResult(
            layer="A", method="model_based_rl", variant=variant,
            adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
        )

    if imperfect:
        variant = "student_of_games"
        rationale = (
            "Imperfect information + exact simulator: Student of Games (sound CFR + search). "
            "Naive AlphaZero converges to an exploitable policy -- don't."
        )
        return RoutingResult(
            layer="A", method="self_play_rl", variant=variant,
            adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
        )

    if stochastic:
        variant = "stochastic_muzero"
        rationale = (
            "Stochastic transitions (dice, cards): Stochastic MuZero via afterstate factorisation "
            "(deterministic afterstate -> chance node -> next state)."
        )
        return RoutingResult(
            layer="A", method="self_play_rl", variant=variant,
            adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
        )

    if action == "continuous":
        variant = "sampled_muzero"
        rationale = "Continuous action space: Sampled MuZero (sample a subset of actions per node)."
        return RoutingResult(
            layer="A", method="self_play_rl", variant=variant,
            adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
        )

    if action == "discrete_large" or tight_sims:
        variant = "gumbel_alphazero"
        reason = "huge discrete action space" if action == "discrete_large" else "tight sim budget (<= 32)"
        rationale = (
            f"Gumbel AlphaZero/MuZero ({reason}): Sequential Halving on top-m actions -- "
            "policy-improvement guarantee even at low simulation counts."
        )
        return RoutingResult(
            layer="A", method="self_play_rl", variant=variant,
            adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
        )

    # default: standard AlphaZero
    variant = "alphazero"
    rationale = (
        "Perfect-information game, exact simulator, small-medium discrete action space, deterministic: "
        "standard AlphaZero (MCTS + PUCT + self-play). The proven baseline for games."
    )
    return RoutingResult(
        layer="A", method="self_play_rl", variant=variant,
        adapter_cls=_LAYER_A_ADAPTER, rationale=rationale, warnings=warns,
    )


def _route_crypto(p: dict, warns: list[str]) -> RoutingResult:
    """Layer C sub-router: decision -> model-based RL; forecasting -> supervised WM."""
    obj = p.get("objective_type", "forecasting")
    if obj == "decision":
        variant = "dreamerv3_or_muzero"
        rationale = (
            "Crypto decision problem: model-based RL (DreamerV3 or MuZero). "
            "No exact simulator -> the WM (src/wm/*) is the learned dynamics model. "
            "WARNING: a policy that plans over an imperfect WM exploits WM errors -- "
            "the robustness/eval-trust stack (10/10 seeds, block-bootstrap p05>0, maxDD<30%) is MANDATORY."
        )
    else:
        variant = "wm_predict_then_rule"
        rationale = (
            "Crypto forecasting: supervised sequence WM (src/wm V1-V25 zoo) + rule-based position sizing. "
            "Objective: robust held-out compound return (per-bar IC is BANNED as the primary metric). "
            "Validated by battery.py (block-bootstrap + jackknife + 10/10 seeds)."
        )
    return RoutingResult(
        layer="C", method="model_based_rl" if obj == "decision" else "supervised_seq_wm",
        variant=variant, adapter_cls=_LAYER_C_ADAPTER, rationale=rationale, warnings=warns,
    )


def _route_layer_b(p: dict, warns: list[str]) -> RoutingResult:
    """Layer B: general time-series / science / forecasting."""
    return RoutingResult(
        layer="B", method="supervised_seq_wm", variant="general_forecasting",
        adapter_cls=_LAYER_B_ADAPTER,
        rationale=(
            "General time-series / science forecasting: supervised sequence model via GeneralAdapter. "
            "Features loaded from DataFrame/parquet; robustness battery applies (domain-general spine)."
        ),
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route(problem: dict) -> RoutingResult:
    """Classify a problem dict to the correct layer, method, variant, and adapter.

    Parameters
    ----------
    problem : dict with keys:
        name (str)                          -- human label
        has_exact_simulator (bool)          -- True for games; False for crypto/real-world
        action_space (str)                  -- "discrete_small|discrete_large|continuous|none"
        objective_type (str)                -- "decision|forecasting|structured_prediction"
        domain (str)                        -- "games|time_series|crypto|science"
        stochastic_transitions (bool, opt)  -- dice/cards/market noise
        imperfect_information (bool, opt)   -- hidden cards/units
        data_budget (int, opt)              -- training samples available
        sim_budget (int, opt)               -- simulations per move (games)

    Returns
    -------
    RoutingResult
    """
    warns: list[str] = []

    # validate inputs (soft: warn, don't crash)
    action = p_action = problem.get("action_space", "none")
    if action not in _VALID_ACTION_SPACES:
        _warn(warns, f"Unknown action_space '{action}'; treating as 'none'.")
        p_action = "none"

    obj = problem.get("objective_type", "forecasting")
    if obj not in _VALID_OBJECTIVE_TYPES:
        _warn(warns, f"Unknown objective_type '{obj}'; defaulting to 'forecasting'.")
        obj = "forecasting"

    domain = problem.get("domain", "")
    if domain and domain not in _VALID_DOMAINS:
        _warn(warns, f"Unknown domain '{domain}'; will fall through to Layer B.")

    # -----------------------------------------------------------------------
    # ROUTING CASCADE (ordered -- first match wins)
    # -----------------------------------------------------------------------

    # 1. Structured prediction (AlphaFold-class) -- currently unimplemented
    if obj == "structured_prediction":
        # Use _warn() so the message both enters the warns list AND fires Python's warnings
        # channel -- callers using warnings.catch_warnings(record=True) will see it.
        _warn(
            warns,
            "NOT IMPLEMENTED: structured_prediction (equivariant/geometric DL, AlphaFold-class). "
            "This is a method bucket with zero implementation in the current codebase (no equivariant "
            "architecture family, no geometry-aware data pipeline). Named honestly as out-of-current-reach."
        )
        # P2 fix: if domain is a non-empty value that is incompatible with structured_prediction
        # (e.g. domain='games'), it is silently ignored by routing to Layer B regardless.
        # Emit a warning so the caller knows their domain key was discarded.
        if domain and domain != "":
            _warn(
                warns,
                f"domain='{domain}' was provided but is ignored: objective_type="
                f"'structured_prediction' always routes to Layer B (geometric_dl). "
                f"To route by domain instead, change objective_type to "
                f"'decision' or 'forecasting'."
            )
        return RoutingResult(
            layer="B", method="geometric_dl", variant="equivariant_net",
            adapter_cls=_LAYER_B_ADAPTER,
            rationale=(
                "Structured prediction (geometry/graph) -> supervised geometric DL (AlphaFold-class). "
                "WARN: not implemented -- requires equivariant/geometric net family + structural data."
            ),
            warnings=warns,
        )

    # 2. Games: domain==games OR (exact simulator + decision objective AND domain is not explicit
    #    non-game domain).  The simulator-implies-games inference ONLY fires when no explicit
    #    domain overrides it: a crypto or time_series problem with has_exact_simulator=True
    #    (e.g. a backtest harness wrapping market data) must NOT be shunted to Layer A/AlphaZero.
    #    Explicit crypto -> Layer C below; explicit time_series/science -> Layer B below.
    _domain_is_game_compatible = domain in (None, "", "games")
    if domain == "games" or (
        _domain_is_game_compatible
        and bool(problem.get("has_exact_simulator", False))  # line-247: default False for non-games
        and obj == "decision"
    ):
        p_with_action = dict(problem)
        p_with_action["action_space"] = p_action
        return _route_games(p_with_action, warns)

    # 3. Crypto -> Layer C
    if domain == "crypto":
        return _route_crypto(problem, warns)

    # 4. Time-series / science / generic forecasting -> Layer B
    if domain in {"time_series", "science"} or obj == "forecasting":
        return _route_layer_b(problem, warns)

    # 5. Default: Layer B supervised + warn underdetermined
    _warn(warns, f"Underdetermined problem spec (domain='{domain}', objective='{obj}'). "
                 "Defaulting to Layer B supervised_seq. Provide domain/objective_type for precise routing.")
    return RoutingResult(
        layer="B", method="supervised_seq", variant="default",
        adapter_cls=_LAYER_B_ADAPTER,
        rationale="Underdetermined problem -- defaulted to Layer B supervised sequence model.",
        warnings=warns,
    )


# ---------------------------------------------------------------------------
# Self-test (5 canonical problems)
# ---------------------------------------------------------------------------

_TEST_PROBLEMS = [
    {
        "name": "chess",
        "has_exact_simulator": True,
        "action_space": "discrete_small",
        "objective_type": "decision",
        "domain": "games",
        "expected_layer": "A",
        "expected_variant": "alphazero",
    },
    {
        "name": "crypto_decision",
        "has_exact_simulator": False,
        "action_space": "continuous",
        "objective_type": "decision",
        "domain": "crypto",
        "expected_layer": "C",
        "expected_variant": "dreamerv3_or_muzero",
    },
    {
        "name": "crypto_forecast",
        "has_exact_simulator": False,
        "action_space": "none",
        "objective_type": "forecasting",
        "domain": "crypto",
        "expected_layer": "C",
        "expected_variant": "wm_predict_then_rule",
    },
    {
        "name": "generic_time_series",
        "has_exact_simulator": False,
        "action_space": "none",
        "objective_type": "forecasting",
        "domain": "time_series",
        "expected_layer": "B",
        "expected_variant": "general_forecasting",
    },
    {
        "name": "protein_structure_placeholder",
        "has_exact_simulator": False,
        "action_space": "none",
        "objective_type": "structured_prediction",
        "domain": "science",
        "expected_layer": "B",
        "expected_variant": "equivariant_net",
    },
    # Regression tests for fix 1: simulator-implies-games must NOT fire for explicit non-game domains
    {
        "name": "crypto_decision_with_simulator_regression",
        "has_exact_simulator": True,   # e.g. a backtest harness
        "action_space": "continuous",
        "objective_type": "decision",
        "domain": "crypto",            # explicit crypto -> Layer C, NOT Layer A
        "expected_layer": "C",
        "expected_variant": "dreamerv3_or_muzero",
    },
    {
        "name": "ts_decision_with_simulator_regression",
        "has_exact_simulator": True,
        "action_space": "none",
        "objective_type": "decision",
        "domain": "time_series",       # explicit time_series -> Layer B, NOT Layer A
        "expected_layer": "B",
        "expected_variant": "general_forecasting",
    },
    # P1 RWYB: no-simulator + tiny data_budget -> efficient_zero_v2
    {
        "name": "no_sim_tiny_budget",
        "has_exact_simulator": False,
        "action_space": "discrete_small",
        "objective_type": "decision",
        "domain": "games",
        "data_budget": "tiny",
        "expected_layer": "A",
        "expected_variant": "efficient_zero_v2",
    },
    # P1 RWYB: no-simulator + normal data_budget -> muzero
    {
        "name": "no_sim_normal_budget",
        "has_exact_simulator": False,
        "action_space": "discrete_small",
        "objective_type": "decision",
        "domain": "games",
        "data_budget": "normal",
        "expected_layer": "A",
        "expected_variant": "muzero",
    },
    # P1 RWYB: no-simulator + int data_budget <=100k -> efficient_zero_v2
    {
        "name": "no_sim_int_tiny_budget",
        "has_exact_simulator": False,
        "action_space": "discrete_small",
        "objective_type": "decision",
        "domain": "games",
        "data_budget": 50_000,
        "expected_layer": "A",
        "expected_variant": "efficient_zero_v2",
    },
    # P1 RWYB: no-simulator + int data_budget >100k -> muzero
    {
        "name": "no_sim_int_large_budget",
        "has_exact_simulator": False,
        "action_space": "discrete_small",
        "objective_type": "decision",
        "domain": "games",
        "data_budget": 500_000,
        "expected_layer": "A",
        "expected_variant": "muzero",
    },
]


def _selftest(verbose: bool = True) -> int:
    """Route canonical problems and assert expected layer + variant. Returns 0 on pass."""
    failures = 0
    for prob in _TEST_PROBLEMS:
        expected_layer = prob["expected_layer"]
        expected_variant = prob["expected_variant"]
        result = route(prob)
        ok_layer = result.layer == expected_layer
        ok_variant = result.variant == expected_variant
        status = "PASS" if (ok_layer and ok_variant) else "FAIL"
        if status == "FAIL":
            failures += 1
        if verbose:
            print(f"  [{status}] {prob['name']:35s} -> Layer {result.layer} / {result.variant}")
            if result.warnings:
                for w in result.warnings:
                    print(f"         WARN: {w[:100]}")
            if status == "FAIL":
                print(f"         EXPECTED layer={expected_layer} variant={expected_variant}")

    # P2 RWYB: structured_prediction + non-empty domain fires a domain-discard warning.
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        result_p2 = route({
            "objective_type": "structured_prediction",
            "domain": "games",
        })
    domain_warns = [str(c.message) for c in caught if "domain='games' was provided" in str(c.message)]
    p2_ok = result_p2.layer == "B" and len(domain_warns) >= 1
    p2_status = "PASS" if p2_ok else "FAIL"
    if not p2_ok:
        failures += 1
    if verbose:
        print(f"  [{p2_status}] {'P2_domain_discard_warn':35s} -> "
              f"Layer {result_p2.layer} / warns_fired={len(domain_warns)}")
        if not p2_ok:
            print("         EXPECTED layer=B and at least 1 domain-discard warning")

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="router",
        description="Method router: classify a problem to Layer A/B/C + method + adapter.",
    )
    sub = ap.add_subparsers(dest="cmd")

    # selftest
    sub.add_parser("selftest", help="Route 5 canonical problems and assert expectations.")

    # route (single problem via JSON or flags)
    rp = sub.add_parser("route", help="Route a single problem (JSON string or flags).")
    rp.add_argument("--json", default=None,
                    help='Problem as JSON string, e.g. \'{"domain":"crypto","objective_type":"decision"}\'')
    rp.add_argument("--domain", default="", help="Problem domain (games/time_series/crypto/science).")
    rp.add_argument("--objective-type", dest="objective_type", default="forecasting",
                    help="objective_type (decision/forecasting/structured_prediction).")
    rp.add_argument("--has-exact-simulator", dest="has_exact_simulator",
                    action="store_true", default=False)
    rp.add_argument("--action-space", dest="action_space", default="none",
                    help="discrete_small|discrete_large|continuous|none")
    rp.add_argument("--stochastic", action="store_true", default=False)
    rp.add_argument("--imperfect-info", dest="imperfect_information", action="store_true", default=False)
    rp.add_argument("--sim-budget", dest="sim_budget", type=int, default=None)
    rp.add_argument("--name", default="unnamed")

    return ap


if __name__ == "__main__":
    ap = _build_cli()
    args = ap.parse_args()

    if args.cmd == "selftest" or args.cmd is None:
        n = len(_TEST_PROBLEMS) + 1  # +1 for the P2 warning probe
        print(f"[router] self-test: {n} problems")
        failures = _selftest(verbose=True)
        if failures:
            print(f"[router] FAIL: {failures} assertion(s) failed")
            sys.exit(1)
        print(f"[router] PASS: all {n} routed correctly")
        sys.exit(0)

    elif args.cmd == "route":
        if args.json:
            problem = json.loads(args.json)
        else:
            problem = {
                "name": args.name,
                "domain": args.domain,
                "objective_type": args.objective_type,
                "has_exact_simulator": args.has_exact_simulator,
                "action_space": args.action_space,
                "stochastic_transitions": args.stochastic,
                "imperfect_information": args.imperfect_information,
            }
            if args.sim_budget is not None:
                problem["sim_budget"] = args.sim_budget

        result = route(problem)
        print(f"Layer   : {result.layer}")
        print(f"Method  : {result.method}")
        print(f"Variant : {result.variant}")
        print(f"Adapter : {result.adapter_cls}")
        print(f"Rationale: {result.rationale}")
        if result.warnings:
            for w in result.warnings:
                print(f"WARN: {w}")
