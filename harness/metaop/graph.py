"""Harness GRAPH -- the awake-loop orchestrator (LangGraph) wiring real tool-using workers.

The plan->dispatch->judge->reflect->route loop:
  - plan     : the Brain seeds an EV-ranked frontier of nodes (build / verify-falsifier / diverge-generalize)
  - dispatch : runs up to `parallel` nodes CONCURRENTLY; irreversible nodes are PARKED for human approval (HITL)
  - judge    : MECHANICAL VERIFIER first (verify_cmd exit==0 = ground-truth PASS, overrides the LLM panel);
               else an adversarial N-judge LLM vote. Rejection becomes a GRADIENT (the concrete error is fed back).
  - reflect  : distils a transferable lesson (persisted to a learnings channel) + GENERATES adjacent problems
  - route    : loop until solved / budget spent / frontier empty

OBSERVABILITY: every node/decision -> a JSONL trace under the harness workspace (traces/<run_id>.jsonl).
DURABLE: pass a checkpointer (SqliteSaver) to survive + resume across processes.

Project-agnostic: all paths come from config (the harness WORKSPACE), and the mechanical verifier runs in the
configurable BUILD CWD (the target project) -- nothing is pinned to any one repo. No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import operator
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from . import experts
from . import learnings
from . import memory as _memory
from .config import trace_dir, build_cwd


class OpState(TypedDict, total=False):
    objective: str
    success_criteria: str
    frontier: list
    ledger: Annotated[list, operator.add]
    budget: int
    cycle: int
    status: str
    parallel: int
    run_id: str
    awaiting_approval: Annotated[list, operator.add]
    # --- REPLANNER bookkeeping (optional; absent on legacy/healthy runs -> behaves exactly as before) ----------
    # done_count    : snapshot of how many frontier nodes have reached 'done' (for STALL detection across cycles)
    # stall_cycles  : consecutive reflect-cycles with NO node newly reaching 'done' (STALL trigger A)
    # replan_count  : total replans performed (guard: capped at max_replans -> then proceed/END, never infinite)
    # replan_reason : why the LAST replan fired (passed to the brain's replan role; cleared after consumption)
    done_count: int
    stall_cycles: int
    replan_count: int
    replan_reason: str
    # drain_empty   : consecutive DRAIN-replans (frontier empty + budget remaining) that added NO new work. The
    #                 loop-level "use the window / no idle-stop" gate: route REPLENISHES instead of ending while the
    #                 budget is unspent; only after DEFAULT_DRAIN_REPLAN_EMPTY_CAP empty drain-replans is the frontier
    #                 declared genuinely exhausted and the loop ENDs honestly.
    drain_empty: int


def _trace(run_id: str, event: str, data: dict, workspace: str | None = None):
    try:
        with open(trace_dir(workspace) / f"{run_id}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": round(time.time(), 2), "event": event, **data}, default=str) + "\n")
    except Exception:
        pass


def _open_nodes(frontier):
    return [n for n in frontier if n.get("status") == "open"]


def _record_refuted(recorder, node, verdict: str, run_id: str, workspace: str | None):
    """ANTI-DRIFT hook: when a node is TERMINALLY refuted, hand it to the host `recorder(node, verdict)` so the
    refuted vein is persisted (e.g. the crypto hypothesis_register) and a future cycle's recaller can mark it dead.
    No-op when recorder is None (agnostic path). Wrapped so a host error never wedges the judge node."""
    if recorder is None:
        return
    try:
        recorder(node, verdict)
        _trace(run_id, "record", {"node": node.get("id"), "verdict": verdict}, workspace)
    except Exception as e:
        _trace(run_id, "record", {"node": node.get("id"), "error": f"{type(e).__name__}: {e}"}, workspace)


def _harvest_passed(harvester, node, run_id: str, workspace: str | None):
    """H4 (2026-06-09): the MONOTONICITY hook (Voyager skill-library symmetric to _record_refuted). When a node
    passes the MECHANICAL verifier (ground truth -- NOT an LLM-believed/inconclusive pass, see H3), hand it to the
    host `harvester(node)` so the validated, re-runnable artifact is REGISTERED in the reusable-asset library. Without
    this, every CONFIRM is forgotten and the next cycle re-discovers it (a monotonicity violation). No-op when
    harvester is None (agnostic path). Wrapped so a host error never wedges the judge node."""
    if harvester is None:
        return
    try:
        harvester(node)
        _trace(run_id, "harvest", {"node": node.get("id"), "verify_cmd": node.get("verify_cmd")}, workspace)
    except Exception as e:
        _trace(run_id, "harvest", {"node": node.get("id"), "error": f"{type(e).__name__}: {e}"}, workspace)


def _is_irreversible(node) -> bool:
    t = (node.get("task", "") + " " + node.get("kind", "")).lower()
    return node.get("irreversible") is True or any(
        k in t for k in ("deploy real", "live capital", "external send", "force-push"))


# --- MECHANICAL VERIFIER (the JUDGE becomes a gradient) -------------------------------------------------------
# A node MAY carry verify_cmd (a shell command string). When present, the judge RUNS it from the BUILD CWD and
# treats exit==0 as ground-truth PASS (overrides the LLM panel) / exit!=0 as REFUTE (storing the concrete error on
# node['verify_error']). The error is APPENDED to the node's next dispatch prompt so rejection is a usable gradient.
VERIFY_TIMEOUT = 120  # seconds; a hung verify_cmd must never wedge the loop

# TRUST-CORE INTEGRITY (2026-06-07 audit): verify_cmd is BRAIN-AUTHORED and runs shell=True. The mechanical
# verifier is the harness's anchor of trust, so it must itself be unspoofable -- critical once the brain is a
# SWAPPED weak local model (Ollama). Two rules screened BEFORE it runs:
#   (1) NON-DESTRUCTIVE: no irreversible side effect that runs unfenced from the build cwd.
#   (2) NON-TRIVIAL: `true`/`exit 0`/`echo ok` would let the brain fake a green PASS on ANY node -- treated as
#       REFUTE so the verification GAP is loud, never a silent green (he_verifier_falsify).
_VERIFY_DENY = [r"\brm\s+-rf?\b", r"\bgit\s+(push|reset\s+--hard|clean\s+-[a-zA-Z]*f)\b", r">\s*/dev/sd",
                r"\bmkfs\b", r"\bdd\s+if=", r"(^|\s)sudo\s", r"\bshutdown\b", r"\breboot\b", r":\s*\(\s*\)\s*\{",
                r"\bformat\s+[a-zA-Z]:", r"Remove-Item.*-Recurse", r"\b(curl|wget)\b.*\|\s*(sh|bash)"]


def _screen_verify_cmd(verify_cmd: str):
    """Return (reject_code, reason) if verify_cmd must NOT run (destructive or trivial), else None. Non-zero codes
    => the caller treats a rejected cmd as REFUTE (a loud gap, never a silent pass)."""
    import re
    vc = (verify_cmd or "").strip()
    if not vc:
        return (125, "verify_cmd REJECTED: empty -- a verifier must assert the artifact (he_verifier_falsify)")
    if not re.search(r"&&|\|\||;|\|", vc) and re.match(r"^(true|:|exit(\s+0)?|echo(\s+\S.*)?)$", vc, re.IGNORECASE):
        return (125, f"verify_cmd REJECTED: trivial no-op ({vc!r}) cannot prove an artifact -- author a real "
                     "assertion that exits non-zero when the artifact is WRONG (he_verifier_falsify)")
    for pat in _VERIFY_DENY:
        if re.search(pat, vc):
            return (126, f"verify_cmd REJECTED by safety fence /{pat}/ -- a verifier must be a NON-DESTRUCTIVE "
                         "independent assertion, not a side-effecting command")
    return None


def _run_verify(verify_cmd: str, cwd: str):
    """Run verify_cmd mechanically from the BUILD CWD. Returns (exit_code, tail) where tail is the last ~1500 chars
    of combined stdout+stderr. A timeout / launch failure is a non-zero exit (REFUTE) with the reason in tail.
    TRUST GUARD: a destructive or trivial verify_cmd is REJECTED (non-zero) before it can run -- see _screen_verify_cmd."""
    bad = _screen_verify_cmd(verify_cmd)
    if bad:
        return bad
    import os, subprocess
    _nw = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        r = subprocess.run(verify_cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=VERIFY_TIMEOUT, creationflags=_nw)
        combined = (r.stdout or "") + (r.stderr or "")
        return r.returncode, combined[-1500:]
    except subprocess.TimeoutExpired as e:
        out = ((e.stdout or "") if isinstance(e.stdout, str) else "") + ((e.stderr or "") if isinstance(e.stderr, str) else "")
        return 124, (f"TIMEOUT after {VERIFY_TIMEOUT}s\n" + out)[-1500:]
    except Exception as e:
        return 1, f"verify_cmd failed to launch: {type(e).__name__}: {e}"[-1500:]


DEFAULT_VERIFY_RETRIES = 2  # rejection-as-gradient retries default for build nodes that carry a verify_cmd


def _seed_verify_defaults(frontier):
    """PLANNER seam (he_planner_verify_cmd): set build nodes up to USE the mechanical-verifier loop, and make the
    verification GAP visible instead of silent.

    For every kind=='build' node:
      - DEFAULT verify_retries to 2 if it does not already set it (so rejection-as-gradient retry engages WHEN a
        verify_cmd is present). Nodes that already carry verify_retries are untouched.
      - If it carries NO verify_cmd, mark node['verify_missing'] = True and emit a one-line WARNING -- the planner
        CANNOT reliably author an EXTERNAL check (per he_verifier_falsify a verify_cmd must assert known outputs, not
        re-run the worker's own test), so we surface the gap rather than auto-generating a bogus verify_cmd.
    Non-build nodes and nodes already carrying verify_cmd/verify_retries are left exactly as-is. Mutates in place
    and returns the same list (so callers can keep the planner's node order/identity)."""
    for n in frontier:
        if not isinstance(n, dict) or n.get("kind") != "build":
            continue
        if "verify_retries" not in n:
            n["verify_retries"] = DEFAULT_VERIFY_RETRIES
        if not n.get("verify_cmd"):
            n["verify_missing"] = True
            print(f"[verify] build node {n.get('id')} has NO external verify_cmd -- it will be LLM-judged, "
                  "not mechanically verified (he_verifier_falsify)")
    return frontier


# --- REPLANNER (LangGraph plan-execute "replan" pattern, adapted to our frontier model) ----------------------
# The ONE-SHOT planner cannot recover from a bad initial plan (the #1 fragility): `route` only ever returned
# dispatch/budget/END and `reflect` only APPENDED adjacent nodes. The replanner closes that loop: on STALL /
# REPEATED-FAILURE / an explicit approach-wrong SIGNAL, the graph routes back to a `replan` node that asks the brain
# for a REVISED frontier -- it may PRUNE doomed/refuted nodes, KEEP good/open ones, and ADD new-approach nodes
# (not merely append). A guard caps total replans so it can NEVER infinite-loop.
DEFAULT_REPLAN_STALL = 2     # consecutive reflect-cycles with no new 'done' before STALL fires (trigger A)
DEFAULT_MAX_REPLANS = 3      # hard cap on total replans per run -> then proceed/END (anti-infinite-loop guard)

# DRAIN-REPLAN (the loop-level "use the window / no idle-stop" gate, 2026-06-08). The idle-stop bug at the LOOP level:
# route() used to END the instant the frontier drained, even with budget/window remaining -- the mechanical twin of an
# overseer who finishes the explicit list and stops with time left. FIX: when the frontier drains but the budget is
# NOT spent and the objective is NOT solved, the loop ROUTES TO REPLAN to replenish with the next adjacent work
# (n+-k). Bounded: this many CONSECUTIVE drain-replans that add NO new work => the frontier is genuinely exhausted =>
# END honestly (the brain was asked for adjacent work and had none). The cycle `budget` is the outer wall so this can
# never run past the window. Separate from max_replans (which caps STALL/FAILURE recovery replans).
DEFAULT_DRAIN_REPLAN_EMPTY_CAP = 2

# A reflect lesson containing this sentinel (case-insensitive substring) flags the whole APPROACH as wrong and
# forces a replan -- the brain's escape hatch when it realizes the plan's premise is unsound (trigger C / SIGNAL).
REPLAN_LESSON_SENTINEL = "approach-wrong"


def _done_count(frontier) -> int:
    """How many frontier nodes have terminally reached 'done' (a PASS). Used for STALL detection across cycles."""
    return sum(1 for n in frontier if isinstance(n, dict) and n.get("status") == "done")


def _judge_count(node: dict, judges: int) -> int:
    """SELF-CONSISTENCY (test-time scaling): how many independent LLM judges to sample for this node.
    verify nodes get the full panel (ground-truth-adjacent, costly-if-wrong); HIGH-EV build/diverge nodes get a
    small panel (>=2) -- a single hallucinated 'pass' on a load-bearing node is the costliest error; everything else
    gets one cheap pass. Scaling by KIND *and EV* (cost-if-wrong), not KIND alone, is the difficulty-adaptive-compute
    pattern. Backward-compatible: with judges>=2, verify is unchanged; only high-EV non-verify nodes gain samples."""
    kind = node.get("kind", "build")
    if kind == "verify":
        return max(1, judges)
    try:
        ev = float(node.get("ev", 0) or 0)
    except (TypeError, ValueError):
        ev = 0.0
    if ev >= 0.7:
        return max(2, (judges + 1) // 2)  # half-panel (>=2) for load-bearing nodes
    return 1


_PANEL_LENSES = ["correctness", "skeptic", "reproducibility", "edge-cases", "evidence-quality"]


def _panel_lenses(n: int) -> list:
    """DE-NAIVE the judge panel (anti self-preference): our judges are the SAME model family that did the work, so
    N identical samples just re-confirm one bias. Give each panel member a DISTINCT audit LENS instead -- perspective
    diversity catches failure modes redundancy can't, and is a stronger de-correlator than answer-order shuffling for
    a single-result verdict. n==1 -> plain correctness; n>1 -> distinct lenses round-robin."""
    if n <= 1:
        return ["correctness"]
    return [_PANEL_LENSES[i % len(_PANEL_LENSES)] for i in range(n)]


def _replan_reason(state, replan_stall: int = DEFAULT_REPLAN_STALL) -> str | None:
    """Return a non-empty REASON string when the loop should REPLAN instead of dispatch, else None. Checks (in order):
      (a) STALL    : `stall_cycles` >= replan_stall  (consecutive reflect-cycles with no node newly reaching 'done');
      (b) FAILURE  : an open/refuted node exhausted its verify_retries AND no other open node progressed this cycle
                     (i.e. nothing reached 'done' this cycle AND >=1 node is terminally 'refuted');
      (c) SIGNAL   : any node carries node['replan_signal'] truthy, OR a ledger/reflect lesson contains the sentinel.
    Returns None (NO replan) on a healthy run (a node reaching 'done' resets stall_cycles to 0). The guard against
    infinite replanning lives in `route` (max_replans), NOT here, so the reason is always reportable for the trace."""
    fr = state.get("frontier", []) or []
    # (c) SIGNAL: an explicit per-node flag, or a reflect lesson that flagged the approach as wrong.
    sig_nodes = [n.get("id") for n in fr if isinstance(n, dict) and n.get("replan_signal")]
    if sig_nodes:
        return f"signal: node(s) {sig_nodes} flagged the approach wrong (replan_signal)"
    ledger = state.get("ledger", []) or []
    for lesson in ledger:
        if isinstance(lesson, str) and REPLAN_LESSON_SENTINEL in lesson.lower():
            return f"signal: a reflect lesson flagged the approach wrong ({REPLAN_LESSON_SENTINEL!r})"
    # (a) STALL: N consecutive reflect-cycles with no newly-done node.
    if int(state.get("stall_cycles", 0)) >= int(replan_stall):
        return (f"stall: {state.get('stall_cycles')} consecutive cycle(s) with no node reaching done "
                f"(>= replan_stall={replan_stall})")
    # (b) REPEATED-FAILURE: a node terminally refuted (retries exhausted) AND nothing progressed to done this cycle.
    refuted = [n.get("id") for n in fr if isinstance(n, dict) and n.get("status") == "refuted"]
    no_done_this_cycle = int(state.get("stall_cycles", 0)) >= 1
    if refuted and no_done_this_cycle:
        return (f"repeated-failure: node(s) {refuted} terminally refuted (verify_retries exhausted) and no other "
                "open node progressed to done this cycle")
    return None


def _merge_replan(old_frontier, revised, run_id: str, reason: str, workspace: str | None):
    """Merge a brain-revised frontier into the live one with KEEP/PRUNE/ADD semantics (NOT a blind replace):
      - PRESERVE every terminal `done` node from the old frontier (history + the overseer ledger are immutable);
      - REPLACE the OPEN set (open / awaiting_approval / refuted) with the revised plan's nodes;
      - so the revised plan may PRUNE doomed/refuted veins (omit them), KEEP good open ones (re-list them), and ADD
        brand-new-approach nodes. `_seed_verify_defaults` is applied to the revised nodes (same planner seam).
    Emits a `replan` trace event {reason, pruned:[ids], kept:[ids], added:[ids]} and returns the merged list.
    Defensive: if the brain returns no usable revised frontier, the OPEN set is left intact (no-op, never wipes work)."""
    old = [n for n in old_frontier if isinstance(n, dict)]
    done_nodes = [n for n in old if n.get("status") == "done"]
    done_ids = {n.get("id") for n in done_nodes}
    old_open = [n for n in old if n.get("status") != "done"]
    old_open_ids = {n.get("id") for n in old_open}

    revised = [n for n in (revised or []) if isinstance(n, dict) and n.get("id")]
    # never let the brain re-touch a node we already completed (done is terminal); drop dup-of-done from the revision.
    revised = [n for n in revised if n.get("id") not in done_ids]
    if not revised:
        # brain gave us nothing usable -> keep the existing open set (do NOT wipe in-flight work).
        _trace(run_id, "replan", {"reason": reason, "pruned": [], "kept": sorted(old_open_ids), "added": [],
                                  "note": "brain returned no usable revised frontier -> open set kept as-is"}, workspace)
        return done_nodes + old_open

    for n in revised:  # the revised plan starts fresh: every revised node is OPEN and worker-ready.
        n["status"] = "open"
    _seed_verify_defaults(revised)
    # PRESERVE the mechanical verifier on KEPT nodes (N6-exposed gap, 2026-06-07): a revised re-list of an existing
    # open node may omit its verify_cmd -> that node would silently drop from mechanical verification to LLM-judging.
    # Carry the original verify_cmd/verify_retries forward so a kept node stays mechanically verified (trust core).
    _old_by_id = {n.get("id"): n for n in old_open if n.get("id")}
    for n in revised:
        o = _old_by_id.get(n.get("id"))
        if o and o.get("verify_cmd") and not n.get("verify_cmd"):
            n["verify_cmd"] = o["verify_cmd"]
            n.setdefault("verify_retries", o.get("verify_retries", DEFAULT_VERIFY_RETRIES))
            n.pop("verify_missing", None)  # no longer missing -- the original assertion is restored
    revised_ids = {n.get("id") for n in revised}
    kept = sorted(old_open_ids & revised_ids)        # open nodes the revision retained (by id)
    pruned = sorted(old_open_ids - revised_ids)       # open nodes the revision DROPPED (doomed/refuted veins removed)
    added = sorted(revised_ids - old_open_ids)        # brand-new-approach nodes
    _trace(run_id, "replan", {"reason": reason, "pruned": pruned, "kept": kept, "added": added}, workspace)
    return done_nodes + revised


# --- PLAN SELF-CRITIQUE (breadth guard) ----------------------------------------------------------------------
# The planner prompt instructs the brain to self-critique once before returning, but a WEAK brain (a swapped local
# model) may still return a single-path frontier. This is the MECHANICAL backstop: after the brain plans, detect a
# missing FALSIFIER (kind=verify) and/or GENERALIZATION (kind=diverge) and ask the brain for the MINIMAL extra node(s)
# via a CHEAP `plan_critique` call. Optional (default ON, fully degradable): any error / no-op leaves the drafted
# frontier byte-identical, so the agnostic path is unchanged when the plan is already broad or the brain lacks the
# role. This is the seam DSPy/eval can later measure against solve_rate.
def _plan_gaps(frontier) -> list:
    """Names of the breadth guards the drafted frontier is MISSING: 'falsifier' (no kind==verify node) and/or
    'generalization' (no kind==diverge node). Empty list => the frontier already covers both (no critique needed)."""
    kinds = {(n.get("kind") or "").lower() for n in frontier if isinstance(n, dict)}
    gaps = []
    if "verify" not in kinds:
        gaps.append("falsifier")
    if "diverge" not in kinds:
        gaps.append("generalization")
    return gaps


def _self_critique_plan(brain, frontier, objective, run_id, workspace):
    """ONE cheap critique pass: if the drafted frontier lacks a falsifier/generalization, ask the brain (role
    'plan_critique') for ONLY the minimal extra node(s) and APPEND them (dedup by id, status forced open). Returns the
    (possibly extended) frontier. NEVER raises -- on any error / unusable output the frontier is returned unchanged."""
    gaps = _plan_gaps(frontier)
    if not gaps:
        return frontier
    try:
        existing_ids = {n.get("id") for n in frontier if isinstance(n, dict)}
        out = brain.decide("plan_critique", {"objective": objective, "missing": gaps,
                                             "frontier": [{"id": n.get("id"), "kind": n.get("kind"),
                                                           "task": (n.get("task", "") or "")[:160]}
                                                          for n in frontier if isinstance(n, dict)]})
        add = out.get("add", []) if isinstance(out, dict) else []
        added_ids = []
        for n in add:
            if not isinstance(n, dict) or not n.get("id") or n.get("id") in existing_ids:
                continue
            n["status"] = "open"
            n.setdefault("ev", 0.5)
            frontier.append(n)
            existing_ids.add(n["id"])
            added_ids.append(n["id"])
        _trace(run_id, "plan_critique", {"missing": gaps, "added": added_ids,
                                         "remaining_gaps": _plan_gaps(frontier)}, workspace)
    except Exception as e:
        _trace(run_id, "plan_critique", {"missing": gaps, "error": f"{type(e).__name__}: {e}"}, workspace)
    return frontier


def make_nodes(brain, parallel: int, max_steps: int, judges: int, taper: int,
               expert_mode: bool = False, channel: str = "default",
               workspace: str | None = None, cwd: str | None = None, persona_dir: str | None = None,
               persona_aliases: dict | None = None,
               framer=None, recaller=None, recorder=None, harvester=None,
               replan_stall: int = DEFAULT_REPLAN_STALL, max_replans: int = DEFAULT_MAX_REPLANS,
               plan_critique: bool = True, fill_window: bool = False):
    """make_nodes builds the plan/dispatch/judge/reflect/route closures.

    fill_window (NO-IDLE-STOP, default False -> EXACT prior behavior): when True, a loop that completes its planned
    frontier does NOT END while the cycle `budget` remains -- it DRAIN-REPLANS for the next adjacent work (n+-k) and
    only stops when the budget is spent OR the brain returns no new work DEFAULT_DRAIN_REPLAN_EMPTY_CAP times in a
    row. This is the loop-level "use the allocated window" mode the autonomy driver turns on for timed/agentic runs;
    a normal one-shot loop leaves it False and completes-then-stops as before.

    ANTI-DRIFT / BREADTH injection (optional, default None -> EXACT current behavior, agnostic path unchanged):
      framer(objective)   -> str|dict : called at `plan`; output added to the plan payload as payload["framing"]
                                        so the planner seeds a BROADER frontier (depth+breadth axes; never-impossible).
      recaller(objective) -> str|dict : called at `plan`; output added as payload["recall"] = reusable-skills digest
                                        + open hypotheses + the DEAD-list, so the planner REUSES prior assets and does
                                        NOT re-mine refuted veins.
      recorder(node, verdict)         : called from `judge`/`reflect` on REFUTED nodes so the refuted vein is written
                                        to a persistent register -> next cycle's recaller can mark it dead (dedup).
      harvester(node)                 : called from `judge` on a MECHANICALLY-PASSED node (ground truth, NOT an
                                        LLM-believed/inconclusive pass) so the validated re-runnable artifact is
                                        REGISTERED in the reusable-asset (skill) library -> next cycle's recaller
                                        reuses it instead of re-discovering it (monotonicity; the CONFIRM-harvest twin
                                        of recorder's REFUTE-record).
    All four are HOST-injected (the harness stays project-agnostic): the crypto shim supplies them; an agnostic run
    leaves them None.

    REPLANNER params (default = the standard plan-execute recovery behavior; a HEALTHY run never replans):
      replan_stall : consecutive reflect-cycles with no node newly reaching 'done' before a STALL replan fires (2).
      max_replans  : hard cap on total replans per run -> then proceed/END (anti-infinite-loop guard, default 3).

    PLAN SELF-CRITIQUE param:
      plan_critique : when True (default), the plan node runs ONE cheap `plan_critique` call that ADDS a missing
                      falsifier (kind=verify) / generalization (kind=diverge) so a weak brain can't ship a
                      single-path frontier. Fully degradable -> set False to disable; never changes a broad plan."""
    verify_cwd = str(build_cwd(cwd))

    def plan(state: OpState):
        rid = state["run_id"]
        if state.get("frontier"):
            return {}
        payload = {"objective": state["objective"], "success_criteria": state["success_criteria"],
                   "prior_project_learnings": learnings.summary_for_plan(channel=channel, workspace=workspace),
                   # G-B (mem0-style): TASK-SIMILARITY recall -- prior cycles similar to THIS objective (not just
                   # recent), so an old/other-lane lesson on a similar task resurfaces (fixes 'forget after compaction').
                   "similar_past_cycles": learnings.similar_for_plan(state["objective"], channel=channel,
                                                                     workspace=workspace),
                   # N5 (Mem0 backend): ADDITIVE semantic recall. When the LOCAL Mem0 backend (ollama embedder +
                   # on-disk qdrant) is available it returns embedding-similar lessons FUSED with the TF-IDF digest;
                   # when Mem0 is unavailable it degrades to EXACTLY the TF-IDF `similar_for_plan` output above, so
                   # this key is always safe + never changes the fallback behavior. Best-effort: never raises here.
                   "mem0_recall": _memory.recall(state["objective"], channel=channel, workspace=workspace)}
        # ANTI-DRIFT seam: enrich the plan payload with BREADTH (framer) + REUSE/dead-veins (recaller). Both optional;
        # default None keeps the agnostic path byte-identical. Each is wrapped so a host module error never wedges plan.
        if framer is not None:
            try:
                payload["framing"] = framer(state["objective"])
                _trace(rid, "frame", {"keys": list(payload["framing"].keys()) if isinstance(payload["framing"], dict)
                                      else "str"}, workspace)
            except Exception as e:
                _trace(rid, "frame", {"error": f"{type(e).__name__}: {e}"}, workspace)
        if recaller is not None:
            try:
                payload["recall"] = recaller(state["objective"])
                _trace(rid, "recall", {"keys": list(payload["recall"].keys()) if isinstance(payload["recall"], dict)
                                       else "str"}, workspace)
            except Exception as e:
                _trace(rid, "recall", {"error": f"{type(e).__name__}: {e}"}, workspace)
        if expert_mode:
            payload["expert_mode"] = True
            payload["available_experts"] = experts.available(persona_dir, persona_aliases)
        out = brain.decide("plan", payload)
        fr = out.get("frontier", []) if isinstance(out, dict) else []
        # PLAN SELF-CRITIQUE (breadth guard, optional default ON): if the drafted frontier lacks a FALSIFIER
        # (kind=verify) and/or a GENERALIZATION (kind=diverge), ask the brain (cheap `plan_critique` call) for ONLY
        # the minimal extra node(s) and append them. Fully degradable: any error / no-op leaves `fr` unchanged.
        if plan_critique:
            fr = _self_critique_plan(brain, fr, state["objective"], rid, workspace)
        # PLANNER seam: set build nodes up to USE the mechanical-verifier loop + flag the verification GAP (silent no
        # longer) per he_planner_verify_cmd. Backward-compatible: nodes already carrying verify_cmd/verify_retries are
        # untouched; non-build nodes untouched.
        _seed_verify_defaults(fr)
        _trace(rid, "plan", {"seeded": len(fr), "expert_mode": expert_mode, "channel": channel,
                             "gaps_after_critique": _plan_gaps(fr),
                             "verify_missing": [n.get("id") for n in fr if isinstance(n, dict) and n.get("verify_missing")]},
               workspace)
        return {"frontier": fr, "status": "running"}

    def dispatch(state: OpState):
        rid = state["run_id"]
        fr = [dict(n) for n in state["frontier"]]
        # HITL: park irreversible nodes for approval instead of running them
        approvals = []
        runnable = []
        for n in _open_nodes(fr):
            if _is_irreversible(n):
                n["status"] = "awaiting_approval"; approvals.append(n["id"])
            else:
                runnable.append(n)
        batch = sorted(runnable, key=lambda n: n.get("ev", 0), reverse=True)[:max(1, parallel)]
        if not batch:
            _trace(rid, "dispatch", {"runnable": 0, "parked_for_approval": approvals}, workspace)
            return {"frontier": fr, "awaiting_approval": approvals}

        def _do(node):
            ename, persona = ("", "")
            if expert_mode:
                ename, persona = experts.persona_for_node(node, persona_dir, persona_aliases)
            task = node["task"]
            # BUILD-TO-THE-TEST: give the worker the node's verify_cmd UP FRONT so it builds the EXACT interface the
            # mechanical verifier checks (file name, function name, signature). Without this the worker builds blind,
            # the planner-written verify_cmd guesses a different interface, and a CORRECT artifact gets refuted on a
            # name mismatch (observed: worker built calc() while the verify_cmd expected another name). exit 0 on this
            # command is the definition of done.
            if node.get("verify_cmd") and not node.get("verify_error"):
                task = task + (
                    f"\n\nDEFINITION OF DONE: your artifact MUST make this exact command exit 0 (it is how you are "
                    f"graded -- build precisely the file/function names + signatures it expects):\n"
                    f"  {node.get('verify_cmd')}\nRun it yourself and fix until it passes, then stop.")
            # REJECTION-AS-GRADIENT: a prior mechanical verify failure is appended so the worker fixes the REAL error.
            if node.get("verify_error"):
                task = task + (
                    f"\n\nA MECHANICAL VERIFIER ran `{node.get('verify_cmd', '')}` on your artifact and it FAILED "
                    f"(exit!=0):\n{node['verify_error']}\nFix the artifact so this command exits 0. Do not claim "
                    "success -- the verifier is the ground truth.")
            # U4: wire the cascade's MECHANICAL escalation tier. A CascadeBrain accepts the cheap result IFF the
            # node's verify_cmd exits 0 (ground truth); without this it falls back to the weaker quality heuristic.
            # Duck-typed: only CascadeBrain has set_node_context -> a no-op for every other brain (agnostic path
            # unchanged). Pass the node's verify_cmd + the build cwd the verifier runs in (verify_cwd, in scope).
            _snc = getattr(brain, "set_node_context", None)
            if callable(_snc):
                _snc(verify_cmd=node.get("verify_cmd"), cwd=verify_cwd, needs_strong=bool(node.get("needs_strong")))
            res = brain.work(task, persona=persona)
            if not res.get("ok"):  # one retry
                if callable(_snc):
                    _snc(verify_cmd=node.get("verify_cmd"), cwd=verify_cwd, needs_strong=bool(node.get("needs_strong")))
                res = brain.work(task + " (retry: be concrete, run a tool, verify by running)", persona=persona)
            return node["id"], res, ename

        results = {}
        with ThreadPoolExecutor(max_workers=len(batch)) as ex:
            for nid, res, ename in ex.map(_do, batch):
                results[nid] = (res, ename)
        for n in fr:
            if n["id"] in results:
                r, ename = results[n["id"]]
                n["status"] = "worked"; n["result"] = r["result"]; n["worker_ok"] = r["ok"]
                if ename:
                    n["expert_used"] = ename
        _trace(rid, "dispatch", {"ran": list(results.keys()), "parallel": len(batch),
                                 "experts": {k: v[1] for k, v in results.items() if v[1]},
                                 "parked_for_approval": approvals}, workspace)
        return {"frontier": fr, "awaiting_approval": approvals}

    def judge(state: OpState):
        rid = state["run_id"]
        fr = [dict(n) for n in state["frontier"]]
        for n in [x for x in fr if x.get("status") == "worked"]:
            # MECHANICAL VERIFIER FIRST: a verify_cmd exit==0 is ground-truth PASS (overrides the LLM panel);
            # exit!=0 REFUTES + captures the concrete error for rejection-as-gradient.
            if n.get("verify_cmd"):
                code, tail = _run_verify(n["verify_cmd"], verify_cwd)
                if code == 0:
                    n["verdict"] = "pass"; n["status"] = "done"; n["evidence_type"] = "mechanical"
                    n.pop("verify_error", None)
                    _harvest_passed(harvester, n, rid, workspace)  # H4: monotonic skill-library harvest (ground-truth only)
                    _trace(rid, "judge", {"node": n["id"], "verdict": "pass", "evidence_type": "mechanical",
                                          "mechanical": True, "exit": 0}, workspace)
                    continue
                n["verify_error"] = tail; n["evidence_type"] = "mechanical_refuted"
                budget = int(n.get("verify_retries", 0))
                if budget > 0:
                    n["verify_retries"] = budget - 1
                    n["verdict"] = "refuted"; n["status"] = "open"  # re-open -> dispatch re-runs it WITH the error
                else:
                    n["verdict"] = "refuted"; n["status"] = "refuted"
                    _record_refuted(recorder, n, "refuted", rid, workspace)  # TERMINAL refute -> dead-vein register
                _trace(rid, "judge", {"node": n["id"], "verdict": "refuted", "mechanical": True, "exit": code,
                                      "reopened": n["status"] == "open",
                                      "retries_left": int(n.get("verify_retries", 0))}, workspace)
                continue
            # --- LLM-vote judge: SELF-CONSISTENCY (adaptive K) + DE-NAIVE perspective-diverse panel ---
            # K scales with KIND and EV (cost-if-wrong), not KIND alone (_judge_count). Each panel member audits
            # under a DISTINCT lens and is told it is auditing ANOTHER worker's output with default skepticism --
            # so a multi-vote panel is perspective-diverse, not N redundant self-preferring samples. The
            # answer-frequency (passed/K) is surfaced as judge_confidence (a calibrated signal for VOI stopping).
            n_judges = _judge_count(n, judges)
            jp = experts.load("auditor", persona_dir, persona_aliases) if (expert_mode and n.get("kind") == "verify") else ""
            _skeptic = ("You are auditing ANOTHER worker's output, not your own. Default to skepticism: return "
                        "'pass' ONLY if the evidence in 'result' actually supports the claim; else 'refuted' or "
                        "'inconclusive'.")
            verdicts = [brain.decide("judge", {"node": n, "audit_lens": lens, "_skeptic_note": _skeptic},
                                     persona=jp).get("verdict", "inconclusive")
                        for lens in _panel_lenses(n_judges)]
            passed = sum(1 for v in verdicts if v == "pass")
            n["judge_confidence"] = round(passed / n_judges, 3) if n_judges else 0.0  # answer-frequency confidence
            n["self_judged"] = True  # honest limitation marker: same model family judged its own work
            raw = "pass" if passed > n_judges / 2 else (verdicts[0] if n_judges == 1 else "refuted")
            # H3 (2026-06-09): EVIDENCE-TYPED verdicts. A mechanical verify_cmd exit==0 is ground truth (handled
            # above); an LLM-panel "pass" with NO mechanical check is a BELIEF, not a verification. Tag it, and
            # AUTO-DOWNGRADE an unverified pass on a node that ASSERTS a checkable artifact (kind=build/verify with
            # no verify_cmd) to "inconclusive" -- UNLESS a verify-node panel was UNANIMOUS (stronger evidence). The
            # node still TERMINATES (status=done, so the loop never stalls on it) but is honestly labeled low-evidence
            # so the overseer / skill-harvest (H4) / completeness critic re-checks before trusting it. Rationale: the
            # loop previously reported an LLM-believed artifact identically to a mechanically-verified one -> false
            # "solved". Healthy mechanical runs are unaffected (this branch only runs when verify_cmd is absent).
            n["evidence_type"] = "llm_panel"
            unanimous = (passed == n_judges and n_judges > 0)
            verifiable_kind = (n.get("kind") or "build") in ("build", "verify")
            if raw == "pass" and verifiable_kind and not n.get("verify_cmd") and not (n.get("kind") == "verify" and unanimous):
                raw = "inconclusive"
                n["evidence_type"] = "llm_panel_unverified"
                n["unverified_pass"] = True
            n["verdict"] = raw
            n["status"] = "done" if raw in ("pass", "inconclusive") else "refuted"
            if raw == "refuted":  # LLM-vote terminal refute -> dead-vein register (next recaller marks dead)
                _record_refuted(recorder, n, "refuted", rid, workspace)
            _trace(rid, "judge", {"node": n["id"], "verdict": raw, "evidence_type": n["evidence_type"],
                                  "unverified_pass": bool(n.get("unverified_pass")), "votes": verdicts,
                                  "judge_confidence": n.get("judge_confidence"), "k": n_judges}, workspace)
        return {"frontier": fr}

    def reflect(state: OpState):
        rid = state["run_id"]
        cyc = state["cycle"] + 1
        # REFLEXION (SOTA): surface THIS cycle's REFUTED nodes (+ their concrete errors) to the reflect brain so the
        # lesson it writes is a DIRECTED post-mortem ('why it failed + what to try differently'), not a vague summary.
        # A refutation thereby becomes a directed-retry signal the next plan/replan can act on, not a silent dead end.
        refuted_now = [{"id": n.get("id"), "task": (n.get("task", "") or "")[:200],
                        "error": (str(n.get("verify_error") or n.get("result", "")) or "")[:300]}
                       for n in state.get("frontier", []) if isinstance(n, dict) and n.get("verdict") == "refuted"]
        payload = {"cycle": cyc, "taper": taper, "ledger_len": len(state.get("ledger", [])),
                   "external_guidance": learnings.summary_for_plan(channel=channel, workspace=workspace)}
        if refuted_now:
            payload["refuted_nodes"] = refuted_now[:8]
            payload["reflexion_instruction"] = (
                "Some nodes were REFUTED this cycle (see refuted_nodes). Write 'lesson' as a concrete, transferable "
                "post-mortem: WHY did they fail and WHAT should the next attempt do differently? One or two sentences.")
        out = brain.decide("reflect", payload)
        adjacent = out.get("adjacent", []) if isinstance(out, dict) else []
        fr = state["frontier"] + adjacent
        lesson = out.get("lesson") if isinstance(out, dict) else None
        if lesson:  # PERSIST to this run's learnings CHANNEL -> compounds across future runs in the same lane
            learnings.record(lesson, rid.rsplit("-", 1)[0], state.get("objective", ""), cyc,
                             channel=channel, workspace=workspace)
        # STALL tracking for the REPLANNER: compare this cycle's 'done' count to the prior snapshot. No NEW done ->
        # increment the stall counter; progress (>=1 new done) RESETS it to 0 (so a healthy run never replans).
        prev_done = int(state.get("done_count", 0))
        cur_done = _done_count(fr)
        stall_cycles = 0 if cur_done > prev_done else int(state.get("stall_cycles", 0)) + 1
        open_left = _open_nodes(fr)
        # 'solved' requires NO remaining work AND nothing terminally REFUTED -- an all-refuted frontier is a FAILURE
        # that the replanner should get a chance to recover (route inspects status before the replan trigger), not a
        # false victory. Previously 'no open nodes' alone meant solved; that masked a dead plan as success.
        refuted_left = [n for n in fr if isinstance(n, dict) and n.get("status") == "refuted"]
        # NO-IDLE-STOP gate (fill_window): 'all planned work done' != 'objective solved'. Declaring solved the instant
        # the frontier drains is the loop-level idle-stop (route honors status==solved BEFORE the drain-replan gate).
        # With fill_window ON, stay 'running' on a drained-but-unspent frontier so route can drain-replan the next
        # adjacent work; only call it solved once the budget is spent OR the drain-replan gate is exhausted. With
        # fill_window OFF (default), behaviour is exactly as before: all work done -> solved.
        all_work_done = (not open_left and not refuted_left)
        if fill_window and all_work_done:
            budget_spent = cyc >= int(state.get("budget", 0))
            drain_exhausted = int(state.get("drain_empty", 0)) >= DEFAULT_DRAIN_REPLAN_EMPTY_CAP
            status = "solved" if (budget_spent or drain_exhausted) else "running"
        else:
            status = "solved" if all_work_done else "running"
        _trace(rid, "reflect", {"cycle": cyc, "adjacent": len(adjacent), "open_left": len(open_left),
                                "refuted_left": len(refuted_left), "status": status, "done": cur_done,
                                "stall_cycles": stall_cycles}, workspace)
        return {"frontier": fr, "ledger": [lesson] if lesson else [], "cycle": cyc, "status": status,
                "done_count": cur_done, "stall_cycles": stall_cycles}

    def replan(state: OpState):
        """REPLAN node: ask the brain for a REVISED frontier (prune doomed/refuted, keep good/open, add new-approach)
        and merge it with KEEP/PRUNE/ADD semantics. Fires only via `route` (STALL / REPEATED-FAILURE / SIGNAL) and is
        capped by max_replans. Resets stall_cycles + bumps replan_count so the loop returns to a healthy cadence."""
        rid = state["run_id"]
        fr = state.get("frontier", []) or []
        # DRAIN replan (frontier empty + budget remaining) vs RECOVERY replan (stall/failure/signal, open nodes
        # present). A drain replan asks the brain for NEXT adjacent work (use the window); a recovery replan revises
        # a stuck plan. Distinguished by whether any open node remains.
        is_drain = not _open_nodes(fr)
        if is_drain:
            reason = ("drain: the frontier is EMPTY but the budget/window is NOT spent. Propose the NEXT "
                      "highest-value ADJACENT work toward the objective as NEW nodes (n+-k: foundational soundness "
                      "checks, derived/adjacent solutions, the general class). Return an EMPTY frontier ONLY if the "
                      "objective is genuinely complete AND nothing adjacent is worth doing.")
        else:
            reason = state.get("replan_reason") or _replan_reason(state, replan_stall) or "manual replan"
        # compact node view for the brain -- only what it needs to revise the plan (no bulky 'result' blobs).
        node_view = [{"id": n.get("id"), "status": n.get("status"), "kind": n.get("kind"),
                      "task": (n.get("task", "") or "")[:300], "verdict": n.get("verdict"),
                      "verify_error": (str(n.get("verify_error"))[:300] if n.get("verify_error") else None)}
                     for n in fr if isinstance(n, dict)]
        payload = {"objective": state.get("objective", ""), "success_criteria": state.get("success_criteria", ""),
                   "current_frontier": node_view, "replan_reason": reason,
                   "lessons": learnings.summary_for_plan(channel=channel, workspace=workspace),
                   "similar_past_cycles": learnings.similar_for_plan(state.get("objective", ""), channel=channel,
                                                                     workspace=workspace)}
        out = brain.decide("replan", payload)
        revised = out.get("frontier", []) if isinstance(out, dict) else []
        merged = _merge_replan(fr, revised, rid, reason, workspace)
        replans = int(state.get("replan_count", 0)) + 1
        # DRAIN tracking: a drain replan that adds NO new open work increments drain_empty (route ENDs honestly once
        # it hits DEFAULT_DRAIN_REPLAN_EMPTY_CAP); one that adds work resets it so the window keeps being used. A
        # RECOVERY (non-drain) replan leaves drain_empty untouched.
        drain_empty = int(state.get("drain_empty", 0))
        if is_drain:
            drain_empty = 0 if _open_nodes(merged) else drain_empty + 1
        _trace(rid, "replan_done", {"replan_count": replans, "frontier_after": len(merged),
                                    "open_after": len(_open_nodes(merged)), "drain": is_drain,
                                    "drain_empty": drain_empty}, workspace)
        # reset stall + the consumed reason + refresh the done snapshot so post-replan progress is measured cleanly.
        return {"frontier": merged, "status": "running", "stall_cycles": 0, "replan_count": replans,
                "replan_reason": "", "done_count": _done_count(merged), "drain_empty": drain_empty}

    def route(state: OpState) -> str:
        if state["status"] == "solved":
            return END
        if state["cycle"] >= state["budget"]:
            return "budget"
        # REPLANNER trigger (plan-execute recovery): on STALL / REPEATED-FAILURE / SIGNAL, route to `replan` instead
        # of dispatch -- UNLESS the max_replans guard is exhausted (then fall through to normal routing, never loop).
        reason = _replan_reason(state, replan_stall)
        if reason and int(state.get("replan_count", 0)) < max_replans:
            _trace(state["run_id"], "route", {"to": "replan", "reason": reason,
                                              "replan_count": int(state.get("replan_count", 0))}, workspace)
            return "replan"
        if reason:  # trigger active but guard exhausted -> proceed (do NOT replan again); note it loudly in the trace.
            _trace(state["run_id"], "route", {"to": "proceed", "reason": reason,
                                              "note": f"max_replans={max_replans} reached -> no further replan"},
                   workspace)
        if _open_nodes(state["frontier"]):
            return "dispatch"
        # FRONTIER DRAINED. With fill_window ON and budget remaining, do NOT idle-stop: route to `replan` to
        # REPLENISH with the next adjacent work (n+-k), bounded by DEFAULT_DRAIN_REPLAN_EMPTY_CAP consecutive
        # empty drain-replans -> then genuinely exhausted -> END honestly. With fill_window OFF (default), END on
        # drain exactly as before. (Closes the loop-level idle-stop: route used to END the instant the frontier drained.)
        if fill_window and int(state.get("drain_empty", 0)) < DEFAULT_DRAIN_REPLAN_EMPTY_CAP:
            _trace(state["run_id"], "route",
                   {"to": "replan", "reason": "drain: frontier empty + budget remains -> replenish adjacent work",
                    "drain_empty": int(state.get("drain_empty", 0))}, workspace)
            return "replan"
        if fill_window:
            _trace(state["run_id"], "route",
                   {"to": "END", "reason": "frontier genuinely exhausted: drain-replan returned no new work "
                    f"{DEFAULT_DRAIN_REPLAN_EMPTY_CAP}x -> honest stop",
                    "drain_empty": int(state.get("drain_empty", 0))}, workspace)
        return END

    return plan, dispatch, judge, reflect, route, replan


def build(brain, parallel=2, max_steps=6, judges=3, taper=3, checkpointer=None, expert_mode=False,
          channel="default", workspace: str | None = None, cwd: str | None = None, persona_dir: str | None = None,
          persona_aliases: dict | None = None, framer=None, recaller=None, recorder=None, harvester=None,
          replan_stall: int = DEFAULT_REPLAN_STALL, max_replans: int = DEFAULT_MAX_REPLANS,
          plan_critique: bool = True, fill_window: bool = False):
    plan, dispatch, judge, reflect, route, replan = make_nodes(
        brain, parallel, max_steps, judges, taper, expert_mode, channel, workspace, cwd, persona_dir,
        persona_aliases, framer=framer, recaller=recaller, recorder=recorder, harvester=harvester,
        replan_stall=replan_stall, max_replans=max_replans, plan_critique=plan_critique, fill_window=fill_window)
    g = StateGraph(OpState)
    for name, fn in (("plan", plan), ("dispatch", dispatch), ("judge", judge), ("reflect", reflect),
                     ("replan", replan)):
        g.add_node(name, fn)
    g.add_node("budget", lambda s: {"status": "budget_spent"})
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "judge")
    g.add_edge("judge", "reflect")
    # route may now also send the loop to `replan`; after replanning, go straight back to dispatch (execute the
    # REVISED frontier). The replan->dispatch edge keeps a healthy cadence and never re-enters reflect without work.
    g.add_conditional_edges("reflect", route,
                            {"dispatch": "dispatch", "replan": "replan", "budget": "budget", END: END})
    g.add_edge("replan", "dispatch")
    g.add_edge("budget", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())
