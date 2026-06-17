"""Operator GRAPH -- crypto-consumer SHIM over the canonical harness.metaop.graph (G-A dedup 2026-06-07).

The awake-loop orchestrator (plan->dispatch->judge->reflect->route, the mechanical AlphaProof-Nexus verifier with
rejection-as-gradient, the planner verify-seam, HITL park, durable checkpointer) now lives ONCE in
harness/metaop/graph.py. The harness make_nodes/build are PARAMETERIZED (cwd / workspace / persona_dir); this shim
re-exports the whole engine and only injects the crypto specifics the live loop has always used:

  - the mechanical verifier + the worker build cwd run from the repo ROOT (cwd=ROOT)
  - observability traces + learnings land under the repo's runs/autonomy (workspace=runs/autonomy), so TRACE_DIR is
    runs/autonomy/traces/<run_id>.jsonl exactly as before
  - the crypto .claude/agents persona dir is used in expert mode (persona_dir=ROOT/.claude/agents)

Crypto callers (manager.py, _proof_loop_delivers.py, _test_verifier.py, _test_planner_verify.py) call make_nodes/
build with NO cwd/workspace arg, so the wrappers default those to the crypto paths. ROOT, TRACE_DIR, _run_verify,
_screen_verify_cmd, _VERIFY_DENY, DEFAULT_VERIFY_RETRIES, _seed_verify_defaults, OpState are all re-exported so the
existing imports/tests are unchanged. No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
from pathlib import Path

from harness.metaop import graph as _h  # the canonical engine
# re-export the full graph surface (functions + constants + state) so existing imports keep working.
from harness.metaop.graph import (  # noqa: F401
    OpState, make_nodes as _h_make_nodes, build as _h_build,
    _run_verify, _screen_verify_cmd, _VERIFY_DENY, _seed_verify_defaults,
    DEFAULT_VERIFY_RETRIES, VERIFY_TIMEOUT, _open_nodes, _is_irreversible, _trace,
    # REPLANNER surface (N3): re-exported so the crypto path inherits plan-execute recovery + tests can import it.
    _replan_reason, _merge_replan, _done_count,
    DEFAULT_REPLAN_STALL, DEFAULT_MAX_REPLANS, REPLAN_LESSON_SENTINEL,
)

from .experts import ALIASES as _CRYPTO_ALIASES  # crypto role->expert-file map (auditor -> expert-auditor, ...)

ROOT = Path(__file__).resolve().parents[3]                     # repo root (…/ml_systems)
_WORKSPACE = str(ROOT / "runs" / "autonomy")                   # harness trace_dir(ws) = ws/traces -> runs/autonomy/traces
TRACE_DIR = ROOT / "runs" / "autonomy" / "traces"              # what manager.status / _proof_loop_delivers read
_PERSONA_DIR = str(ROOT / ".claude" / "agents")                # crypto expert personas (expert mode)

# The 4 anti-drift / idea-generation modules live in scripts/autonomy (the parent of this metaop package). Ensure
# that dir is importable regardless of how the shim was loaded (run_metaop adds it; the repo-root test imports do not).
_AUTONOMY_DIR = str(ROOT / "scripts" / "autonomy")
if _AUTONOMY_DIR not in sys.path:
    sys.path.insert(0, _AUTONOMY_DIR)


# ---------------------------------------------------------------------------------------------------------------
# CRYPTO ANTI-DRIFT INJECTION (G-C 2026-06-07): wire the breadth/anti-drift/reuse modules INTO the loop's plan node.
# These were CLI-only; now the loop's planner receives them every run. The harness stays agnostic (framer/recaller/
# recorder default None there); the crypto specifics are composed HERE and injected via make_nodes/build.
# ---------------------------------------------------------------------------------------------------------------

def _crypto_framer(objective: str) -> dict:
    """BREADTH + COGNITION framing for the plan payload (payload["framing"]).
    = problem_framing.frame(objective)  (depth+breadth axes, standing lenses, anti-impossible rail, seed nodes)
    + resourcefulness.check(objective)  (LLM-failure-mode lenses + decompose-the-ideal protocols).
    So the planner seeds a BROADER frontier and never tunnel-visions / declares impossible."""
    import problem_framing
    import resourcefulness
    fr = problem_framing.frame(objective)
    rc = resourcefulness.check(objective)
    return {
        "coverage_grid": fr["coverage_grid"],
        "not_explored_axes": fr["not_explored"],
        "standing_lenses": fr["standing_lenses"],
        "jolts": fr["jolts"],
        "breadth_seed_nodes": fr["seed_nodes"],
        "anti_impossible_rail": fr["anti_impossible_rail"],
        "cognition_failure_modes": rc["flagged_failure_modes"],
        "resourceful_protocols": rc["resourceful_protocols"],
        "cognition_meta_questions": rc["meta_questions"],
    }


def _crypto_recaller(objective: str) -> dict:
    """REUSE + DEAD-VEIN recall for the plan payload (payload["recall"]).
    = skill_library.digest(query=objective)  (reuse validated assets before building)
    + hypothesis_register.open_ranked()      (EV-ranked OPEN hypotheses to pursue)
    + hypothesis_register.dead_catalog()     (REFUTED veins -- do NOT re-mine).
    So the planner reuses prior skills and does not re-pay for refuted lessons (monotonic memory)."""
    import skill_library
    import hypothesis_register
    return {
        "reusable_assets_digest": skill_library.digest(query=objective),
        "open_hypotheses": [{"id": h["id"], "spec": h["spec"], "ev": h.get("ev"), "note": h.get("note", "")}
                            for h in hypothesis_register.open_ranked()],
        "dead_veins_do_not_remine": [{"id": h["id"], "spec": h["spec"],
                                      "why_refuted": h.get("verdict_detail", "")[:120]}
                                     for h in hypothesis_register.dead_catalog()],
    }


def _node_to_spec(node: dict) -> dict:
    """Map a graph node to a hypothesis SPEC so refuted nodes dedupe by identity. The node's task/id IS the vein."""
    return {"node_id": node.get("id", ""), "task": (node.get("task", "") or "")[:300], "kind": node.get("kind", "")}


def _crypto_recorder(node: dict, verdict: str) -> None:
    """When a node is TERMINALLY refuted, persist the vein to hypothesis_register so a future cycle's recaller can
    mark it dead (refuted-vein dedup). register() seeds the hypothesis, then record_verdict() flips it to 'refuted'
    -> is_dead(spec) becomes True. Idempotent + never raises (the harness wraps this, but be defensive)."""
    import hypothesis_register
    spec = _node_to_spec(node)
    detail = (str(node.get("verify_error") or node.get("result") or "")[:200]) or "refuted by judge"
    try:
        reg = hypothesis_register.register(spec, ev=float(node.get("ev", 0.0) or 0.0), note="auto: metaop loop node")
        hid = reg.get("id")
        if hid:
            hypothesis_register.record_verdict(hid, "refuted", detail)
    except Exception:
        pass


# kinds the auto-harvester will register (the node ASSERTED a reusable artifact). verify/diverge/analysis nodes
# don't produce a reusable tool, so they are NOT harvested (avoids flooding the library with non-assets).
_HARVEST_KINDS = {"build", "engine", "tool", "harness", "gate", "probe", "dataset"}
# map a metaop node kind to a skill_library KIND (its valid set is {tool,probe,harness,engine,gate,dataset}).
_KIND_MAP = {"build": "tool", "engine": "engine", "tool": "tool", "harness": "harness",
             "gate": "gate", "probe": "probe", "dataset": "dataset"}


def _crypto_harvester(node: dict) -> None:
    """H4 (2026-06-09): MONOTONIC harvest. When a node passes the MECHANICAL verifier, register the validated,
    re-runnable artifact into the Voyager skill library so the NEXT cycle reuses it (recaller's reusable_assets_digest)
    instead of re-discovering it. Best-effort + idempotent (register() updates in place by name) + never raises.

    Gate: only nodes that (a) ASSERT a reusable artifact (kind in _HARVEST_KINDS) AND (b) carry a verify_cmd (the
    re-runnable proof) are harvested -- a mechanical pass with no verify_cmd cannot happen (the verifier IS the cmd),
    but the kind gate keeps verify/diverge passes out. The harvested asset's `tested_on` IS the passing verify_cmd, so
    the registry entry is itself re-runnable. Tagged 'auto_harvest' so curated vs auto entries stay distinguishable."""
    kind = (node.get("kind") or "build").lower()
    if kind not in _HARVEST_KINDS or not node.get("verify_cmd"):
        return
    import subprocess
    import skill_library
    # a worker MAY emit a rich harvest spec (node["harvest"] = {name,kind,path,entrypoint,signature,summary,tags});
    # else fall back to a coarse-but-honest entry built from the node itself.
    h = node.get("harvest") if isinstance(node.get("harvest"), dict) else {}
    nid = str(node.get("id") or "node")
    name = str(h.get("name") or f"metaop_{nid}")[:80]
    task = (node.get("task", "") or "")[:240]
    try:
        import os as _os
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10,
                             creationflags=(subprocess.CREATE_NO_WINDOW if _os.name == "nt" else 0)).stdout.strip() or "unknown"
    except Exception:
        sha = "unknown"
    try:
        skill_library.register(
            name=name,
            kind=_KIND_MAP.get(str(h.get("kind") or kind).lower(), "tool"),
            path=str(h.get("path") or ""),
            entrypoint=str(h.get("entrypoint") or ""),
            signature=str(h.get("signature") or ""),
            summary=str(h.get("summary") or task or "metaop node passed its mechanical verifier"),
            tested_on=str(h.get("tested_on") or f"verify_cmd PASS: {node.get('verify_cmd')}")[:240],
            provenance_sha=sha,
            tags=list(h.get("tags") or []) + ["auto_harvest", "metaop", kind],
        )
    except Exception:
        pass


def make_nodes(brain, parallel: int, max_steps: int, judges: int, taper: int,
               expert_mode: bool = False, channel: str = "default",
               replan_stall: int = DEFAULT_REPLAN_STALL, max_replans: int = DEFAULT_MAX_REPLANS,
               plan_critique: bool = True, fill_window: bool = False):
    """Crypto wrapper: the canonical make_nodes with the build cwd pinned to the repo ROOT, traces/learnings under
    runs/autonomy, the crypto .claude/agents persona dir + ALIASES for expert mode, AND the crypto anti-drift
    framer/recaller/recorder injected so the planner gets breadth + reuse + dead-vein dedup (G-C). The REPLANNER
    (N3: replan_stall / max_replans) + the PLAN SELF-CRITIQUE (N7: plan_critique, default ON) are inherited from the
    canonical engine with the same defaults. fill_window (NO-IDLE-STOP drain-replan) is forwarded too -- kept in
    parity with the canonical make_nodes + the shim build() so a direct make_nodes caller can't silently drop it."""
    return _h_make_nodes(brain, parallel, max_steps, judges, taper, expert_mode=expert_mode, channel=channel,
                         workspace=_WORKSPACE, cwd=str(ROOT), persona_dir=_PERSONA_DIR,
                         persona_aliases=_CRYPTO_ALIASES,
                         framer=_crypto_framer, recaller=_crypto_recaller, recorder=_crypto_recorder,
                         harvester=_crypto_harvester,
                         replan_stall=replan_stall, max_replans=max_replans, plan_critique=plan_critique,
                         fill_window=fill_window)


def build(brain, parallel=2, max_steps=6, judges=3, taper=3, checkpointer=None, expert_mode=False, channel="default",
          replan_stall: int = DEFAULT_REPLAN_STALL, max_replans: int = DEFAULT_MAX_REPLANS,
          plan_critique: bool = True, fill_window: bool = False):
    """Crypto wrapper: the canonical build with crypto cwd/workspace/persona_dir/ALIASES + the anti-drift
    framer/recaller/recorder injected (same call surface the live loop + the metaop tests use). The REPLANNER
    (N3) + the PLAN SELF-CRITIQUE breadth guard (N7: plan_critique, default ON) are inherited so the crypto loop gets
    a broad, recoverable plan; defaults = no replan on a healthy run, falsifier+generalization guaranteed.
    fill_window (NO-IDLE-STOP) MUST forward to the canonical build -- manager.launch passes it (--fill-window); if
    the shim drops it the canonical default silently wins. Was missing here -> every run_metaop launch crashed with
    `build() got an unexpected keyword argument 'fill_window'` (signature drift the symbol-only parity test missed)."""
    return _h_build(brain, parallel=parallel, max_steps=max_steps, judges=judges, taper=taper,
                    checkpointer=checkpointer, expert_mode=expert_mode, channel=channel,
                    workspace=_WORKSPACE, cwd=str(ROOT), persona_dir=_PERSONA_DIR,
                    persona_aliases=_CRYPTO_ALIASES,
                    framer=_crypto_framer, recaller=_crypto_recaller, recorder=_crypto_recorder,
                    harvester=_crypto_harvester,
                    replan_stall=replan_stall, max_replans=max_replans, plan_critique=plan_critique,
                    fill_window=fill_window)
