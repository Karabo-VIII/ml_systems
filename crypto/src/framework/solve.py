"""src/framework/solve.py -- ORCHESTRATION CAPSTONE: plan_solve(problem) -> SolvePlan.

This is the END-TO-END entry-point that ties the 3-layer engine into a single
"give the engine a problem -> it produces a coherent, honest solve-plan" flow.

WHAT IT DOES (5 steps from docs/GENERAL_PROBLEM_SOLVING_HARNESS_2026_06_09.md):
  1. REPRESENT   -> validates the problem dict (the ProblemAdapter contract check)
  2. ROUTE       -> calls router.route(problem) to classify layer/method/variant/adapter
  3. LEARN       -> resolves the adapter class (import-by-string, graceful on fail)
  4. VALIDATE    -> names the SHARED VALIDATION SPINE (battery.py / CDAP / honesty rules)
  5. HONEST CEILING -> emits per-layer honest ceiling note (compound-bound / compute-bound /
                       not-implemented). NEVER overclaims.

Returns a SolvePlan dataclass -- a structured, serialisable report of HOW the engine would
approach the problem, not a trained result. Training is deferred (this is planning/orchestration).

EXECUTE path (added 2026-06-11):
  plan_solve() PLANS. The EXECUTOR that actually trains + validates lives in the sibling module
  framework.execute (execute_solve(problem, segments=...) -> SolveResult). It is re-exported here
  for convenience -- `from framework.solve import execute_solve`. plan_solve() is unchanged: the
  executor calls it internally for routing + the honest ceiling, then adds the real run (Layer B
  only; Layer A/C return an honest execution_deferred result).

CLI:
  python -m framework.solve --name "BTCUSDT 4h WM" --domain crypto --objective-type forecasting
  python -m framework.solve --demo        # 3 example problems end-to-end (planner)
  python -m framework.solve --execute     # run the EXECUTOR controls (pos + neg) end-to-end
  python -m framework.solve --selftest    # planner self-test + executor self-test

No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "autonomy") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))
# Layer-A games adapter (`import games.*`) lives EITHER as a child of the framework
# root (in-place layout) OR as a SIBLING of crypto/ after the repo split
# (parent-of-crypto). Put whichever dir actually contains games/ on sys.path so the
# lazy, optional Layer-A routing resolves regardless of layout.
for _cand in (ROOT, ROOT.parent):
    if (_cand / "games").is_dir() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))

# ---------------------------------------------------------------------------
# CDAP contract
# ---------------------------------------------------------------------------
__contract__ = {
    "kind": "orchestration_capstone",
    "module": "SolvePlan producer + CLI",
    "inputs": ["problem: dict with keys name/domain/objective_type/..."],
    "outputs": ["SolvePlan: {problem, routing, adapter_status, spine, lessons_digest, "
                "honest_ceiling, next_actions}"],
    "invariants": {
        "no_training": "plan_solve() does not train; it plans and defers execution",
        "honest_ceiling": "honest_ceiling is REQUIRED and must not overclaim",
        "graceful_import": "adapter import failures are reported, not raised",
        "spine_named": "validation spine (battery.py + CDAP) is always named in the plan",
    },
}


# ---------------------------------------------------------------------------
# SolvePlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdapterStatus:
    adapter_cls: str           # dotted import path
    importable: bool           # whether the import succeeded
    import_error: str          # populated if importable=False
    adapter_note: str          # brief note on what the adapter does / its status


@dataclass
class ValidationSpine:
    """The shared validation spine -- the honesty guarantee of the engine."""
    primary: str = "src/strat/battery.py"
    gates: List[str] = field(default_factory=lambda: [
        "Lens A (strict): n>=15, all-4-window-positive, jk2>0, jk3>0, p05>0, maxDD<30%",
        "Lens B (pragmatic): all-4-positive, UNSEEN>0, jk2>0, jk3>0, n_eff>=8, maxDD<30%",
        "Lens C (temporal): UNSEEN>0, n_months>=3, monthly-positive>=60%, worst_month>-10%",
    ])
    cdap: str = "src/audit/check_invariants.py (unskippable pre-commit gate)"
    no_lookahead_rule: str = (
        "held-out/UNSEEN segment NEVER touched during development; "
        "purge gap=400 bars between splits; no full-history standardisation (G-AUDIT-011)"
    )
    robustness_requirements: List[str] = field(default_factory=lambda: [
        "10/10 seeds positive on UNSEEN",
        "block-bootstrap p05 > 0 (B=5, N=2000)",
        "jackknife K=2 and K=3 positive",
        "maxDD < 30% (project binding: 20%)",
    ])
    note: str = (
        "The spine is DOMAIN-GENERAL: it runs on any layer (A/B/C). "
        "The objective is robust held-out COMPOUND return -- per-bar IC is banned as the primary metric."
    )


@dataclass
class SolvePlan:
    """Structured output of plan_solve(). Serialisable to JSON."""
    problem: Dict[str, Any]
    routing: Dict[str, Any]          # RoutingResult as dict
    adapter_status: Dict[str, Any]   # AdapterStatus as dict
    spine: Dict[str, Any]            # ValidationSpine as dict
    lessons_digest: str              # prompt-ready cross-layer lessons
    honest_ceiling: str              # HONEST ceiling for this layer/domain -- no overclaim
    next_actions: List[str]          # what a human/agent does next to START a solve
    plan_notes: List[str]            # any warnings or caveats from routing


# ---------------------------------------------------------------------------
# Honest-ceiling registry (one entry per layer -- grounded in MEMORY.md + docs)
# ---------------------------------------------------------------------------

_HONEST_CEILINGS: Dict[str, str] = {
    "A": (
        "Layer A (Games/self-play) -- COMPUTE-BOUND ceiling. "
        "The generic UCT/AlphaZero pipeline is proven (TicTacToe never loses to random). "
        "For chess-strength: the ceiling is compute (a 4060 reaches ~1800-2000 ELO self-play; "
        "top-tier requires A100-class + 10^4 self-play games + Gumbel/MCTS optimisation). "
        "The engine routes, the adapter plugs in, the champion gate prevents regression -- "
        "but model STRENGTH is compute-bound, not engineering-bound. "
        "No shortcut: pure self-play on a weaker-than-teacher net DEGRADES (chess-to-crypto "
        "transfer lesson: small-sample noise looks like signal; optimising a proxy degrades real)."
    ),
    "B": (
        "Layer B (General WM / time-series) -- IMPLEMENTATION-PARTIAL ceiling. "
        "Forecasting: GENUINELY WIRED (no longer a stub). src/framework/general_trainer.py:"
        "train_layer_b is a REAL minimal forecaster -- it consumes GeneralAdapter.to_segments() "
        "directly, splits walk-forward (4-way + purge via anti_fragile.WalkForwardSplitter), "
        "trains a small CPU-friendly GRU/MLP/linear head, and validates HELD-OUT IC on the UNSEEN "
        "split with a shuffled-IC memorization probe. PROVEN to learn via a two-sided control: "
        "positive control (planted lagged signal) -> held-out IC ~0.65-0.71; negative control "
        "(pure noise) -> held-out IC ~0 (mean ~-0.007, |IC|<0.05 across 5 seeds) -- it does NOT "
        "hallucinate signal. HONEST SCOPE: this is a MINIMAL BASELINE forecaster, NOT a SOTA "
        "model -- the richer WM zoo (src/wm V1-V25: TwoHot heads, RSSM/JEPA latents, NCL ensembles) "
        "is the upgrade path, parameterised by input_dim and reachable via the same segments "
        "contract. Compound return is the correct TRADING objective (per-bar IC is banned as a "
        "primary metric; here IC is a within-trainer LEARNING DIAGNOSTIC / control gate only). "
        "Structured prediction (AlphaFold-class, geometry/graph): NOT IMPLEMENTED -- "
        "no equivariant/geometric net family, no structural data pipeline. "
        "This is the REMAINING honest scope gap: the method bucket exists in the router, the "
        "structured-prediction implementation does not. Claiming otherwise is a Layer-2 violation."
    ),
    "C": (
        "Layer C (Crypto WM) -- COMPOUND-RETURN-BOUND ceiling (VERIFIED-HONEST per MEMORY.md). "
        "Current apparatus: V1.1 ShIC~0.033 (modest); no verified active alpha post-2026-06-04 reset "
        "(the prior +20.25% figure was apparatus-inflated -- do not cite). "
        "The regime>IC finding (2026-06-06): at 1d/4h/1h/30m bar-level entry-timing (MA + orderflow + "
        "momentum + micro + liq, linear + GBM) and liquidation events are ALL null held-out; "
        "3 avenues converge on sub-bar/HF. Cost cliff: 30m costs ~89.5% of return. "
        "The WM is a SIZER/FILTER (IC~0.033 < Trader tier IC>0.05), not a standalone alpha. "
        "Target: robust held-out compound return, NOT Sharpe. 10/10 seeds + p05>0 + maxDD<20% "
        "are the MINIMUM bars -- do not ship below them. "
        "Do NOT re-mine: (a) 1d MA depth (D01-D63 dead-list), (b) vol-expansion lead (magnitude "
        "not directional, drift+concentrated+maker-fragile, killed 2026-06-09). "
        "Next viable avenues: sub-bar/HF features, wavelet causal features (watch G-AUDIT-011 "
        "look-ahead contamination in published literature), oracle-framing setups."
    ),
}


# ---------------------------------------------------------------------------
# Adapter import (graceful)
# ---------------------------------------------------------------------------

def _try_import_adapter(adapter_cls_path: str) -> AdapterStatus:
    """Attempt to import the adapter class by dotted path. Return an AdapterStatus."""
    parts = adapter_cls_path.rsplit(".", 1)
    if len(parts) != 2:
        return AdapterStatus(
            adapter_cls=adapter_cls_path,
            importable=False,
            import_error=f"Cannot parse dotted path: {adapter_cls_path!r}",
            adapter_note="",
        )
    module_path, cls_name = parts

    # The router uses 'src.framework.*' and 'projects.*' dotted paths.
    # When sys.path already includes ROOT/src and ROOT, direct import works.
    # Try both the canonical path and a stripped 'src.' prefix.
    candidates = [module_path]
    if module_path.startswith("src."):
        candidates.append(module_path[4:])  # strip leading 'src.'

    last_err = ""
    for mod_path in candidates:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            note = _adapter_note(adapter_cls_path, cls)

            # Distinguish IMPORTABLE_ABSTRACT (ABC with unimplemented methods) from a
            # fully usable concrete class.  A caller cannot instantiate an abstract class
            # without first providing a concrete subclass -- label it honestly so the
            # plan does not imply the adapter is ready to use as-is.
            is_abstract = bool(getattr(cls, "__abstractmethods__", None)) or inspect.isabstract(cls)
            if is_abstract:
                abstract_methods = sorted(getattr(cls, "__abstractmethods__", set()))
                return AdapterStatus(
                    adapter_cls=adapter_cls_path,
                    importable=True,
                    import_error="",
                    adapter_note=(
                        f"IMPORTABLE_ABSTRACT (needs a concrete subclass before use). "
                        f"Abstract methods that must be implemented: {abstract_methods}. "
                        f"{note}"
                    ),
                )

            return AdapterStatus(
                adapter_cls=adapter_cls_path,
                importable=True,
                import_error="",
                adapter_note=note,
            )
        except ImportError as exc:
            last_err = f"ImportError on '{mod_path}': {exc}"
        except AttributeError as exc:
            last_err = f"AttributeError on '{mod_path}': {exc}"
        except Exception as exc:
            last_err = f"{type(exc).__name__} on '{mod_path}': {exc}"

    return AdapterStatus(
        adapter_cls=adapter_cls_path,
        importable=False,
        import_error=last_err,
        adapter_note=(
            "Adapter could not be imported in this environment. "
            "Common reasons: optional dependency (yaml/pandas/narrate) unavailable. "
            "The routing and plan are still valid -- the adapter is resolved at SOLVE time."
        ),
    )


def _adapter_note(cls_path: str, cls: Any) -> str:
    """One-line note describing the adapter's role."""
    if "game_adapter" in cls_path or "GameAdapter" in cls_path:
        return (
            "GameAdapter: engine-agnostic games contract (initial_state/legal_actions/apply/"
            "is_terminal/returns). Concrete adapters: TicTacToe (game_adapter.py) AND chess "
            "(projects/chess_zero/az/chess_adapter.py::ChessGameAdapter -- backed by python-chess "
            "+ the verified move<->index encoding; the generic uct_search plays LEGAL chess over it). "
            "Honest ceiling: the generic adapter-driven search proves chess PLUGS IN (contract + "
            "legal play); for chess STRENGTH the project also has the neural PUCT pipeline "
            "(projects/chess_zero/az/train_robust.py), which is compute-bound, not engineering-bound."
        )
    if "general_adapter" in cls_path or "GeneralAdapter" in cls_path:
        return (
            "GeneralAdapter: accepts any DataFrame/parquet/CSV with feature cols + target col; "
            "converts to the segments format the anti-fragile WM loop consumes. Layer B default."
        )
    if "crypto_adapter" in cls_path or "CryptoAdapter" in cls_path:
        return (
            "CryptoAdapter: wraps the project's chimera loader + taker cost model + universe YAMLs "
            "behind the generic 5-method interface. Layer C home."
        )
    # fallback: use the class docstring first line if available
    doc = getattr(cls, "__doc__", "") or ""
    first = doc.strip().splitlines()[0] if doc.strip() else ""
    return first[:120] if first else f"Adapter class {cls_path} (no docstring)."


# ---------------------------------------------------------------------------
# Lessons digest (cross-pollination bus)
# ---------------------------------------------------------------------------

def _get_lessons_digest(layer: str) -> str:
    """Pull cross-layer lessons from the cross_pollination_bus for the routed layer."""
    try:
        from cross_pollination_bus import cross_layer_digest, _seed
        _seed()  # ensure seed lessons are written
        digest = cross_layer_digest(layer, k=5)
        return digest
    except ImportError as exc:
        return (
            f"(cross_pollination_bus unavailable in this environment: {exc}. "
            "Lessons exist in runs/autonomy/cross_pollination.jsonl -- read directly if needed.)"
        )
    except Exception as exc:
        return f"(cross_pollination_bus error: {exc})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_solve(problem: dict) -> SolvePlan:
    """Produce a SolvePlan for the given problem dict.

    Parameters
    ----------
    problem : dict -- see router.route() for the full key contract.
        Required keys:
          name (str)            -- human label
          domain (str)          -- games / time_series / crypto / science
          objective_type (str)  -- decision / forecasting / structured_prediction
        Optional keys:
          has_exact_simulator (bool), action_space (str), stochastic_transitions (bool),
          imperfect_information (bool), data_budget (int), sim_budget (int)

    Returns
    -------
    SolvePlan -- structured plan; NOT a trained model.
    """
    from framework.router import route

    # 1. REPRESENT: normalise + validate the problem dict (soft)
    problem = dict(problem)  # copy; don't mutate caller's dict
    if "name" not in problem:
        problem["name"] = "unnamed"
    if "domain" not in problem:
        problem["domain"] = ""
    if "objective_type" not in problem:
        problem["objective_type"] = "forecasting"

    # 2. ROUTE
    routing_result = route(problem)

    # 3. LEARN: resolve the adapter class
    adapter_status = _try_import_adapter(routing_result.adapter_cls)

    # 4. VALIDATE: name the spine
    spine = ValidationSpine()

    # 5. HONEST CEILING
    honest_ceiling = _HONEST_CEILINGS.get(routing_result.layer, "(no ceiling note for this layer)")

    # Structured-prediction note: the router flags it in warnings, we amplify it here
    if routing_result.method == "geometric_dl":
        honest_ceiling = _HONEST_CEILINGS["B"]

    # Cross-layer lessons
    lessons_digest = _get_lessons_digest(routing_result.layer)

    # Next actions: layer-specific, grounded
    next_actions = _next_actions(routing_result.layer, routing_result.variant, problem)

    plan_notes = list(routing_result.warnings)

    return SolvePlan(
        problem=problem,
        routing={
            "layer": routing_result.layer,
            "method": routing_result.method,
            "variant": routing_result.variant,
            "adapter_cls": routing_result.adapter_cls,
            "rationale": routing_result.rationale,
        },
        adapter_status=asdict(adapter_status),
        spine=asdict(spine),
        lessons_digest=lessons_digest,
        honest_ceiling=honest_ceiling,
        next_actions=next_actions,
        plan_notes=plan_notes,
    )


def execute_solve(problem: dict, segments: Optional[List[dict]] = None, **kw):
    """Convenience re-export of framework.execute.execute_solve (the EXECUTE path).

    plan_solve() PLANS; this RUNS (represent->route->LEARN->VALIDATE) and returns a SolveResult.
    Lazy import avoids a module-load circular dependency (execute.py imports plan_solve from here).

    See framework.execute.execute_solve for the full signature/contract.
    """
    from framework.execute import execute_solve as _exec
    return _exec(problem, segments=segments, **kw)


def _next_actions(layer: str, variant: str, problem: dict) -> List[str]:
    """Produce a short, grounded list of concrete next actions to START the solve."""
    name = problem.get("name", "this problem")
    domain = problem.get("domain", "")

    if layer == "A":
        is_chess = "chess" in str(name).lower()
        if is_chess:
            first = (
                "ChessGameAdapter ALREADY EXISTS (projects/chess_zero/az/chess_adapter.py): "
                "run `python -m projects.chess_zero.az.chess_adapter` to RWYB the contract + "
                "the generic uct_search playing legal chess. No adapter to write."
            )
        else:
            first = (
                f"Implement GameAdapter for '{name}': initial_state, legal_actions, apply, "
                "is_terminal, returns (7 methods = one engine). Model it on TicTacToe "
                "(game_adapter.py) or ChessGameAdapter (chess_adapter.py)."
            )
        return [
            first,
            "Run: python -m projects.chess_zero.az.game_adapter (proof: generic UCT solves "
            "the TicTacToe adapter without a net -- the contract is sufficient).",
            "For STRENGTH (chess): wire the net pipeline (projects/chess_zero/az/train_robust.py "
            "= search + net + self-play + champion gate). Generic random-rollout UCT plays LEGAL "
            "but WEAK chess -- strength is compute-bound, not contract-bound.",
            "Run 3 training cycles; confirm champion gate fires at least once (proves monotonic "
            "gate is live, not cosmetic).",
            "Evaluate: Wilson-CI win-rate vs a random baseline at N=100 games.",
        ]
    elif layer == "C":
        return [
            "Run: python -m framework.pipeline status crypto <instrument> to check the workspace stage.",
            "If stage < 03_strat: run the oracle decomposer first "
            "(python src/mining/decompose.py --asset <SYM> --cadence <TF>).",
            "Select a candidate strategy; wrap it via src/strat/candidate_gate.py.",
            "Run: python src/strat/battery.py on the candidate's held-out returns "
            "(Lens A/B/C + block-bootstrap p05).",
            "If ALL gates pass (10/10 seeds, p05>0, maxDD<20%): record in the workspace "
            "(python -m framework.pipeline run crypto <instrument> 03_strat ...).",
            "HARD STOP: do NOT cite the pre-reset +20.25% figure; re-establish under current apparatus.",
        ]
    elif layer == "B":
        if problem.get("objective_type") == "structured_prediction":
            return [
                "HONEST NOTE: structured_prediction (equivariant/geometric DL) is NOT IMPLEMENTED.",
                "No equivariant architecture family and no geometry-aware data pipeline exist in-repo.",
                "To proceed: (a) adopt an existing library (e.g. PyTorch Geometric, e3nn) for the "
                "equivariant net; (b) implement a structural data loader; (c) adapt the battery.py "
                "spine for the non-temporal domain (shuffle test semantics change).",
                "Then wire via GeneralAdapter (override to_segments for structural data).",
            ]
        return [
            f"Prepare a DataFrame/parquet for '{name}' with named feature columns + a target column.",
            "Instantiate: GeneralAdapter(data_source, target_col='y', instrument='<name>', cadence='<tf>').",
            "Run: adapter.to_segments() -> validate with GeneralAdapter.validate_segment().",
            "Train the WIRED minimal baseline: src/framework/general_trainer.py::train_layer_b(segments) "
            "-- splits walk-forward (4-way + purge via anti_fragile), trains a CPU-friendly GRU/MLP/linear "
            "head, validates held-out IC on UNSEEN with a shuffled-IC memorization probe + two-sided "
            "controls. RWYB: `python -m framework.general_trainer --controls`.",
            "UPGRADE PATH (richer model): feed the SAME segments into a WM-zoo trainer "
            "(e.g. src/wm/v1/v1_1_training/train_world_model.py) -- parameterised by input_dim, same "
            "segments contract. src/anti_fragile.py is the shared component toolkit (splitter, augmentor, "
            "dataset, early-stopping); each version wires these manually (no single entry-point).",
            "After training: run battery.py Lens A/B/C on held-out returns.",
            "Report: honest compound% + block-bootstrap p05 + maxDD. No overclaim.",
        ]

    # fallback
    return [
        "Clarify the problem domain and objective_type (see router.py for valid values).",
        "Re-run plan_solve() with the refined problem dict.",
    ]


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------

_SEP = "-" * 70


def _format_plan(plan: SolvePlan) -> str:
    """Render a SolvePlan as a human-readable report."""
    lines = []

    def _section(title: str) -> None:
        lines.append("")
        lines.append(_SEP)
        lines.append(f"  {title}")
        lines.append(_SEP)

    def _item(label: str, value: str, indent: int = 2) -> None:
        pad = " " * indent
        wrapped = textwrap.fill(str(value), width=72, subsequent_indent=pad + "  ")
        lines.append(f"{pad}{label}: {wrapped}")

    # Header
    lines.append("")
    lines.append("=" * 70)
    lines.append("  SOLVE PLAN")
    lines.append("=" * 70)

    _section("1. PROBLEM")
    _item("name", plan.problem.get("name", "unnamed"))
    _item("domain", plan.problem.get("domain", "(unset)"))
    _item("objective_type", plan.problem.get("objective_type", "(unset)"))
    extras = {k: v for k, v in plan.problem.items()
              if k not in ("name", "domain", "objective_type",
                           "expected_layer", "expected_variant")}
    if extras:
        _item("extras", json.dumps(extras))

    _section("2. ROUTING")
    r = plan.routing
    _item("layer", f"{r['layer']}  (A=Games  B=General-WM  C=Crypto-WM)")
    _item("method", r["method"])
    _item("variant", r["variant"])
    _item("adapter_cls", r["adapter_cls"])
    _item("rationale", r["rationale"])
    if plan.plan_notes:
        for note in plan.plan_notes:
            _item("WARN", note)

    _section("3. ADAPTER STATUS")
    a = plan.adapter_status
    status_str = "IMPORTABLE" if a["importable"] else "NOT importable in this env"
    _item("status", status_str)
    _item("class", a["adapter_cls"])
    if a["adapter_note"]:
        _item("note", a["adapter_note"])
    if not a["importable"] and a["import_error"]:
        _item("import_error", a["import_error"])

    _section("4. VALIDATION SPINE  (the honesty guarantee)")
    s = plan.spine
    _item("primary", s["primary"])
    _item("CDAP gate", s["cdap"])
    _item("no-lookahead rule", s["no_lookahead_rule"])
    lines.append("  gates:")
    for g in s["gates"]:
        lines.append(f"    - {g}")
    lines.append("  robustness requirements:")
    for req in s["robustness_requirements"]:
        lines.append(f"    - {req}")
    if s.get("note"):
        _item("note", s["note"])

    _section("5. CROSS-LAYER LESSONS  (read-forward; do not re-mine refuted veins)")
    digest = plan.lessons_digest
    # indent the digest
    for dline in digest.splitlines():
        lines.append("  " + dline)

    _section("6. HONEST CEILING")
    for cline in textwrap.wrap(plan.honest_ceiling, width=68):
        lines.append("  " + cline)

    _section("7. NEXT ACTIONS  (to start the solve)")
    for i, action in enumerate(plan.next_actions, 1):
        wrapped = textwrap.fill(action, width=66,
                                initial_indent=f"  {i}. ",
                                subsequent_indent="     ")
        lines.append(wrapped)

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo problems
# ---------------------------------------------------------------------------

_DEMO_PROBLEMS = [
    {
        "name": "Chess (generic board game)",
        "domain": "games",
        "objective_type": "decision",
        "has_exact_simulator": True,
        "action_space": "discrete_small",
        "stochastic_transitions": False,
        "imperfect_information": False,
        "sim_budget": 800,
    },
    {
        "name": "BTCUSDT 4h World-Model Forecast",
        "domain": "crypto",
        "objective_type": "forecasting",
        "has_exact_simulator": False,
        "action_space": "none",
    },
    {
        "name": "Generic daily time-series (e.g. energy demand)",
        "domain": "time_series",
        "objective_type": "forecasting",
        "has_exact_simulator": False,
        "action_space": "none",
        "data_budget": 5000,
    },
]


def _run_demo() -> int:
    """Run 3 example problems and print each SolvePlan. Returns 0 on success."""
    print("=" * 70)
    print("  SOLVE.PY --demo: 3 example problems -> SolvePlan")
    print("=" * 70)

    failures = 0
    for i, prob in enumerate(_DEMO_PROBLEMS, 1):
        print(f"\n[demo {i}/{len(_DEMO_PROBLEMS)}] problem: {prob['name']!r}")
        try:
            plan = plan_solve(prob)
            report = _format_plan(plan)
            print(report)
            # minimal assertions
            assert plan.routing["layer"] in ("A", "B", "C"), \
                f"routing layer not A/B/C: {plan.routing['layer']}"
            assert plan.honest_ceiling, "honest_ceiling is empty"
            assert plan.next_actions, "next_actions is empty"
            assert plan.spine["primary"] == "src/strat/battery.py", \
                "spine.primary is wrong"
            print(f"  [PASS] demo {i}: plan valid (layer={plan.routing['layer']}, "
                  f"method={plan.routing['method']}, variant={plan.routing['variant']})")
        except Exception as exc:
            print(f"  [FAIL] demo {i}: {exc}")
            failures += 1

    print()
    if failures:
        print(f"[solve --demo] FAIL: {failures} demo(s) failed")
    else:
        print(f"[solve --demo] PASS: all {len(_DEMO_PROBLEMS)} demos produced valid SolvePlans")
    return failures


# ---------------------------------------------------------------------------
# Self-test (for framework.selftest integration)
# ---------------------------------------------------------------------------

def _selftest(verbose: bool = True) -> int:
    """Smoke-test: route 3 canonical problems, assert plan fields populated. Returns 0 on pass."""
    failures = 0

    cases = [
        # (problem, expected_layer, expected_method)
        (
            {"name": "tictactoe_test", "domain": "games", "objective_type": "decision",
             "has_exact_simulator": True, "action_space": "discrete_small"},
            "A", "self_play_rl",
        ),
        (
            {"name": "btc_forecast_test", "domain": "crypto", "objective_type": "forecasting"},
            "C", "supervised_seq_wm",
        ),
        (
            {"name": "ts_test", "domain": "time_series", "objective_type": "forecasting"},
            "B", "supervised_seq_wm",
        ),
    ]

    for prob, exp_layer, exp_method in cases:
        try:
            plan = plan_solve(prob)

            ok_layer = plan.routing["layer"] == exp_layer
            ok_method = plan.routing["method"] == exp_method
            ok_spine = plan.spine["primary"] == "src/strat/battery.py"
            ok_ceiling = bool(plan.honest_ceiling)
            ok_actions = len(plan.next_actions) >= 3

            passed = ok_layer and ok_method and ok_spine and ok_ceiling and ok_actions
            tag = "PASS" if passed else "FAIL"
            if not passed:
                failures += 1
            if verbose:
                print(f"  [{tag}] solve({prob['name']!r}): "
                      f"layer={plan.routing['layer']} method={plan.routing['method']} "
                      f"spine={'OK' if ok_spine else 'MISSING'} "
                      f"ceiling={'OK' if ok_ceiling else 'EMPTY'} "
                      f"actions={len(plan.next_actions)}")
                if not ok_layer:
                    print(f"         expected layer={exp_layer}, got {plan.routing['layer']}")
                if not ok_method:
                    print(f"         expected method={exp_method}, got {plan.routing['method']}")
        except Exception as exc:
            print(f"  [FAIL] solve({prob.get('name')!r}): raised {type(exc).__name__}: {exc}")
            failures += 1

    if verbose:
        if failures:
            print(f"[solve] planner self-test: {failures} case(s) failed")
        else:
            print("[solve] planner self-test PASS: all 3 plan cases passed")

    # Executor self-test hook: prove the EXECUTE path (Layer-B two-sided control + A/C deferral).
    # Lazy import (execute.py imports plan_solve from this module). Kept lightweight (CPU, seconds).
    try:
        from framework.execute import selftest as _exec_selftest
        if verbose:
            print("[solve] executor self-test: Layer-B two-sided control + A/C deferral")
        exec_fails = _exec_selftest(verbose=verbose, seed=0)
        failures += exec_fails
        if verbose and not exec_fails:
            print("[solve] executor self-test PASS")
    except Exception as exc:  # executor selftest must not silently vanish
        print(f"[solve] executor self-test ERROR: {type(exc).__name__}: {exc}")
        failures += 1

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="framework.solve",
        description=(
            "Orchestration capstone: plan_solve(problem) -> SolvePlan. "
            "Routes the problem, resolves the adapter, names the validation spine, "
            "emits an honest ceiling note, and lists concrete next actions."
        ),
    )
    ap.add_argument("--demo", action="store_true",
                    help="Run 3 example problems end-to-end and print each SolvePlan.")
    ap.add_argument("--execute", action="store_true",
                    help="Run the EXECUTOR (framework.execute) on the pos+neg Layer-B controls "
                         "end-to-end and print each SolveResult (real train+validate, not a plan).")
    ap.add_argument("--selftest", action="store_true",
                    help="Run the internal self-test (planner 3 cases + executor two-sided control).")
    ap.add_argument("--json-out", action="store_true",
                    help="Output the SolvePlan as JSON instead of a human-readable report.")
    ap.add_argument("--name", default="unnamed", help="Human label for the problem.")
    ap.add_argument("--domain", default="",
                    help="Problem domain: games / time_series / crypto / science.")
    ap.add_argument("--objective-type", dest="objective_type", default="forecasting",
                    help="Objective: decision / forecasting / structured_prediction.")
    ap.add_argument("--has-exact-simulator", dest="has_exact_simulator",
                    action="store_true", default=False)
    ap.add_argument("--action-space", dest="action_space", default="none",
                    help="discrete_small / discrete_large / continuous / none.")
    ap.add_argument("--stochastic", dest="stochastic_transitions",
                    action="store_true", default=False)
    ap.add_argument("--imperfect-info", dest="imperfect_information",
                    action="store_true", default=False)
    ap.add_argument("--sim-budget", dest="sim_budget", type=int, default=None)
    ap.add_argument("--data-budget", dest="data_budget", type=int, default=None)
    return ap


if __name__ == "__main__":
    ap = _build_cli()
    args = ap.parse_args()

    if args.demo:
        sys.exit(_run_demo())

    if args.execute:
        # Run the EXECUTOR end-to-end on the two return-scaled Layer-B controls.
        from framework.execute import (
            execute_solve as _exec, format_result,
            make_positive_return_control, make_negative_return_control,
        )
        _ov = {"seed": 0, "verbose": False, "model_kind": "gru",
               "seq_len": 16, "max_epochs": 60, "purge_gap_bars": 16}
        _pos = _exec({"name": "solve --execute positive control", "domain": "time_series",
                      "objective_type": "forecasting", "target_is_return": True},
                     segments=make_positive_return_control(seed=0), trainer_overrides=_ov)
        _neg = _exec({"name": "solve --execute negative control", "domain": "time_series",
                      "objective_type": "forecasting", "target_is_return": True},
                     segments=make_negative_return_control(seed=0), trainer_overrides=_ov)
        print(format_result(_pos))
        print()
        print(format_result(_neg))
        _ok = (_pos.battery.get("unseen_only_pass") is True
               and _neg.battery.get("unseen_only_pass") is False)
        print(f"\n[solve --execute] two-sided result: "
              f"{'PASS' if _ok else 'FAIL'} (positive screens, negative does not)")
        sys.exit(0 if _ok else 1)

    if args.selftest:
        print("[solve] self-test: planner (3 cases) + executor (two-sided control + A/C deferral)")
        fails = _selftest(verbose=True)
        sys.exit(0 if not fails else 1)

    # single problem from CLI flags
    problem: Dict[str, Any] = {
        "name": args.name,
        "domain": args.domain,
        "objective_type": args.objective_type,
        "has_exact_simulator": args.has_exact_simulator,
        "action_space": args.action_space,
        "stochastic_transitions": args.stochastic_transitions,
        "imperfect_information": args.imperfect_information,
    }
    if args.sim_budget is not None:
        problem["sim_budget"] = args.sim_budget
    if args.data_budget is not None:
        problem["data_budget"] = args.data_budget

    plan = plan_solve(problem)

    if args.json_out:
        print(json.dumps(asdict(plan), indent=2, ensure_ascii=False))
    else:
        print(_format_plan(plan))
