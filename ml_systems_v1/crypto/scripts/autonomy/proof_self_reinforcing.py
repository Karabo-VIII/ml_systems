#!/usr/bin/env python3
"""N8 CAPSTONE -- the END-TO-END SELF-REINFORCING PROOF.

Demonstrates + ASSERTS that the integrated metaop engine, given a problem, behaves like a human driving an LLM over
many prompting cycles: it PLANS (with breadth), ACTS, a MECHANICAL verifier AUDITS, on failure it REPLANS + recovers,
it LEARNS (writes durable memory), and the NEXT cycle RECALLS the lesson (compounding). All the pieces wired in N3
(replanner) + N5 (mem0/memory) + N6 (verifier) + N7 (planner-breadth) working IN SYNERGY -- proven on the REAL
`harness.metaop.graph.build` app, not a unit-test stub.

DETERMINISM: a single SCRIPTED brain (ScriptedBrain, a MockBrain subclass) drives the whole loop so the proof is
fast + reproducible + does not depend on a flaky weak model. The MECHANICS are what we prove; the brain is scripted
so they fire on every run. A separate, honest REAL datapoint (litellm -> ollama/qwen2.5-coder:3b) is recorded at the
end and reported without inflation (it is a weak 3b; it may not pass -- that is reported truthfully).

ISOLATION (does NOT pollute live state): every run uses a fresh TEMP workspace (traces + learnings) and a fresh TEMP
build dir (artifacts), both removed at the end. Mem0's slow/locking on-disk path is disabled (HARNESS_MEM0_DISABLE)
so recall() uses the GUARANTEED TF-IDF floor deterministically -- N5's semantic layer DEGRADES to TF-IDF here by
design (recall() is contractually identical on that path); the live ollama datapoint exercises the real model.

THE SCRIPTED FAILURE->RECOVERY CHAIN (drives the real graph):
  plan      -> frontier = [ n1 build (verify_cmd asserts answer()==42; verify_retries=0 -> first refute is TERMINAL),
                            n2 verify  (the FALSIFIER / -k breadth node),
                            n3 diverge (the GENERALIZATION / +k breadth node) ]
  cycle 1   -> WORKER writes the WRONG artifact for n1 -> mechanical verifier REFUTES (exit!=0), n1 terminally refuted
  reflect   -> no node reached done -> stall -> route fires REPLAN (replan_stall=1)
  replan    -> ScriptedBrain prunes refuted n1, keeps n2/n3, ADDS n_fix (writes the CORRECT artifact + verify_cmd)
  cycle 2+  -> WORKER writes the CORRECT artifact for n_fix -> verifier PASSES (exit 0) -> RECOVERY; n2/n3 judged pass
  reflect   -> a durable LESSON is written to the learnings channel (compounds across runs in the same lane)
  run 2     -> on a SIMILAR objective, memory.recall(objective) surfaces run 1's lesson (cross-cycle compounding)

Run:  python scripts/autonomy/proof_self_reinforcing.py
Exit: 0 iff links 1-5 all PASS (the real ollama datapoint never fails the proof -- SKIPPED is acceptable).
No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# repo root on sys.path so `harness.metaop` imports without PYTHONPATH (mirrors eval_harness_run.py).
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ISOLATION: disable Mem0's on-disk qdrant path for the IN-PROCESS proof. recall() then uses the guaranteed TF-IDF
# floor -- deterministic, fast, no embedded-store file locks. Set BEFORE importing memory (it reads the env at import).
os.environ.setdefault("HARNESS_MEM0_DISABLE", "1")

from harness.metaop import graph as _graph             # noqa: E402  the canonical engine (plan/dispatch/judge/reflect/route/replan)
from harness.metaop import learnings as _learnings     # noqa: E402  durable per-channel lessons (the LEARN channel)
from harness.metaop import memory as _memory           # noqa: E402  mem0 + TF-IDF recall seam (N5)
from harness.metaop.brain import MockBrain             # noqa: E402  deterministic base brain
from harness.metaop.config import trace_dir            # noqa: E402


# ----------------------------------------------------------------------------------------------------------------
# The artifact the proof builds: a file `proof_artifact.py` defining answer() -> 42. The mechanical verify_cmd (a
# NON-trivial, harness-authored assertion -- not the worker's own test) exits 0 IFF answer()==42, else non-zero.
_PY = f'"{sys.executable}"'
_VERIFY_BODY = ("import sys; sys.path.insert(0, '.'); from proof_artifact import answer; "
                "assert answer() == 42, ('WRONG', answer()); print('ARTIFACT_OK')")
VERIFY_CMD = f'{_PY} -c "{_VERIFY_BODY}"'

WRONG_ARTIFACT = "def answer():\n    return 0  # WRONG on purpose -- the verifier must REFUTE this\n"
CORRECT_ARTIFACT = "def answer():\n    return 42  # CORRECT -- the verifier must PASS this\n"

# a stable, transferable lesson the scripted reflect emits -- and what run 2's recall must surface (compounding).
LESSON_MARKER = "PROOF-LESSON-N8"
SCRIPTED_LESSON = (f"{LESSON_MARKER}: the first build attempt returned 0 instead of 42; the mechanical verifier "
                   "refuted it and a replan with a corrected artifact recovered. Reuse the corrected build, do not "
                   "re-mine the wrong-constant vein.")


class ScriptedBrain(MockBrain):
    """Deterministic brain that scripts the failure->replan->recovery chain on the REAL graph.

    - plan   : breadth frontier (build n1 + falsifier n2[kind=verify] + generalization n3[kind=diverge]); n1 carries
               the mechanical verify_cmd with verify_retries=0 (first refute is TERMINAL -> stalls -> triggers replan).
    - work   : writes the WRONG artifact for the original build node (n1) and the CORRECT artifact for the replan's
               fix node (n_fix). The artifact lands in the worker's build cwd, where the verifier runs.
    - replan : prune the refuted n1, KEEP the open breadth nodes, ADD n_fix (correct artifact + the SAME verify_cmd).
    - reflect: emit the durable SCRIPTED_LESSON so it is recorded to the learnings channel (the LEARN link).
    - judge  : inherited MockBrain LLM-vote for the no-verify_cmd breadth nodes (result present -> pass).
    """

    FIX_ID = "n_fix"

    def decide(self, role, payload, persona=""):
        if role == "plan":
            obj = payload.get("objective", "objective")
            return {"frontier": [
                {"id": "n1", "task": f"build: {obj} -- write proof_artifact.py with answer()",
                 "ev": 0.95, "kind": "build", "status": "open",
                 "verify_cmd": VERIFY_CMD, "verify_retries": 0},   # 0 -> first mechanical refute is TERMINAL
                {"id": "n2", "task": f"falsifier (-k): audit whether '{obj}' is sound (positive control / leak check)",
                 "ev": 0.80, "kind": "verify", "status": "open"},
                {"id": "n3", "task": f"generalization (+k): extend '{obj}' to the adjacent / more-general case",
                 "ev": 0.60, "kind": "diverge", "status": "open"},
            ]}
        if role == "replan":
            # KEEP the open non-refuted nodes, PRUNE the refuted ones (by omission), ADD the corrective fix node.
            cur = payload.get("current_frontier", []) or []
            kept = [{"id": n.get("id"), "task": n.get("task", ""), "ev": n.get("ev", 0.5),
                     "kind": n.get("kind", "build"), "status": "open"}
                    for n in cur if isinstance(n, dict) and n.get("status") not in ("refuted", "done")]
            fix = {"id": self.FIX_ID,
                   "task": "NEW-APPROACH (replan): rewrite proof_artifact.py so answer() returns 42 (the corrected build)",
                   "ev": 0.99, "kind": "build", "status": "open",
                   "verify_cmd": VERIFY_CMD, "verify_retries": 1}
            return {"frontier": kept + [fix]}
        if role == "reflect":
            # emit the durable lesson once (graph.reflect records it to the learnings channel -> the LEARN link).
            return {"lesson": SCRIPTED_LESSON, "adjacent": []}
        # judge + everything else: inherited MockBrain behavior (breadth nodes with no verify_cmd -> LLM-vote pass).
        return super().decide(role, payload, persona)

    def work(self, task: str, persona: str = "") -> dict:
        """Write the WRONG artifact for the original build, the CORRECT artifact for the replan's fix node. Writes to
        the worker's build cwd (the graph's cwd=) so the mechanical verifier finds it. Uses the harness Tools so this
        is a REAL file-write through the real worker surface, not a side-channel."""
        from harness.metaop.tools import Tools
        t = (task or "")
        # the fix-node task says 'returns 42 (the corrected build)'; the original build node does not.
        content = CORRECT_ARTIFACT if ("corrected build" in t or "returns 42" in t) else WRONG_ARTIFACT
        # build cwd: prefer an explicit per-instance cwd (set by the driver), else the default build_cwd.
        tools = Tools(cwd=getattr(self, "cwd", None))
        res = tools.write_file("proof_artifact.py", content)
        kind = "CORRECT" if content is CORRECT_ARTIFACT else "WRONG"
        return {"ok": bool(res.get("ok")),
                "result": f"[scripted] wrote {kind} proof_artifact.py -> {res.get('output', res.get('error', ''))}"}


# ----------------------------------------------------------------------------------------------------------------
def _events(workspace: str, run_id: str) -> list:
    """Read the JSONL trace the graph wrote for run_id under workspace/traces."""
    tr = trace_dir(workspace) / f"{run_id}.jsonl"
    if not tr.exists():
        return []
    out = []
    for ln in tr.read_text(encoding="utf-8").strip().splitlines():
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out


def _init_state(objective: str, run_id: str, budget: int) -> dict:
    return {"objective": objective, "success_criteria": "proof_artifact.py exists AND its verify_cmd exits 0",
            "frontier": [], "ledger": [], "budget": budget, "cycle": 0, "status": "running",
            "parallel": 1, "run_id": run_id, "awaiting_approval": []}


def _drive_run(objective: str, workspace: str, build_dir: str, run_id: str, budget: int = 8):
    """Run the objective through the REAL graph.build app in an isolated workspace+build dir. parallel=1 isolates the
    failing build node so its terminal refute drives the stall->replan->recover chain deterministically. Returns
    (last_state, trace_events)."""
    brain = ScriptedBrain(domain="self-reinforcing proof (build a verifiable artifact)")
    brain.cwd = build_dir  # worker writes the artifact where the mechanical verifier runs
    app = _graph.build(brain, parallel=1, judges=1, taper=1, channel="proof_n8",
                       workspace=workspace, cwd=build_dir,
                       replan_stall=1, max_replans=3, plan_critique=True)
    cfg = {"configurable": {"thread_id": run_id}}
    last = None
    t0 = time.time()
    for step in app.stream(_init_state(objective, run_id, budget), cfg, stream_mode="values"):
        last = step
        if time.time() - t0 > 90:   # hard wall-clock guard: the proof must never hang
            print("  [guard] run exceeded 90s -- breaking (should not happen on the scripted path)")
            break
    return last, _events(workspace, run_id)


# ----------------------------------------------------------------------------------------------------------------
def _real_ollama_datapoint() -> str:
    """HONEST real datapoint: run the eval harness once on litellm->ollama/qwen2.5-coder:3b for 2 tasks and report the
    measured solve_rate WITHOUT inflation (it is a weak 3b). If litellm/ollama is truly unavailable, return a clear
    SKIPPED line -- this NEVER fails the proof. Runs as a SUBPROCESS (the canonical CLI) so its temp dirs/teardown
    are fully isolated from the in-process proof state."""
    cli = ROOT / "scripts" / "autonomy" / "eval_harness_run.py"
    cmd = [sys.executable, str(cli), "--brain", "litellm", "--model", "ollama/qwen2.5-coder:3b",
           "--tasks", "fib,is_prime", "--budget", "3", "--timeout", "120",
           "--label", f"n8_proof_ollama_{time.strftime('%Y%m%d_%H%M%S')}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(ROOT),
                           creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    except subprocess.TimeoutExpired:
        return "REAL OLLAMA DATAPOINT: SKIPPED (eval subprocess timed out > 600s)"
    except Exception as e:
        return f"REAL OLLAMA DATAPOINT: SKIPPED (could not launch eval: {type(e).__name__}: {e})"
    out = (r.stdout or "") + (r.stderr or "")
    # parse the SOLVE_RATE line the CLI prints.
    rate = None
    for ln in out.splitlines():
        if "SOLVE_RATE" in ln:
            rate = ln.strip()
            break
    if rate is None:
        tail = out.strip()[-400:]
        return ("REAL OLLAMA DATAPOINT: SKIPPED (no SOLVE_RATE parsed -- litellm/ollama likely unavailable). "
                f"tail: {tail!r}")
    # also surface per-task PASS/FAIL lines for honesty
    per = [ln.strip() for ln in out.splitlines() if ln.strip().startswith("[") and ("PASS" in ln or "FAIL" in ln)]
    note = ("  (HONEST: qwen2.5-coder:3b is a weak local 3b model; a low/zero solve_rate is expected and is reported "
            "truthfully -- this datapoint proves the engine runs a real swapped model end-to-end, not that the model "
            "is strong.)")
    return "REAL OLLAMA DATAPOINT (litellm -> ollama/qwen2.5-coder:3b, 2 tasks):\n  " + rate + \
           ("\n  " + "\n  ".join(per) if per else "") + "\n" + note


# ----------------------------------------------------------------------------------------------------------------
def main() -> int:
    print("=" * 90)
    print("N8 CAPSTONE -- END-TO-END SELF-REINFORCING PROOF (plan->act->AUDIT->replan->recover->LEARN->RECALL)")
    print("=" * 90)

    # isolated temp workspace (traces + learnings) + temp build dirs (artifacts). All removed in `finally`.
    workspace = tempfile.mkdtemp(prefix="n8_proof_ws_")
    build1 = tempfile.mkdtemp(prefix="n8_proof_build1_")
    build2 = tempfile.mkdtemp(prefix="n8_proof_build2_")
    print(f"  isolation: workspace={workspace}")
    print(f"             build1={build1}")
    print(f"             build2={build2}")
    print(f"  memory   : mem0 disabled (HARNESS_MEM0_DISABLE={os.environ.get('HARNESS_MEM0_DISABLE')}) -> recall uses "
          "the guaranteed TF-IDF floor (N5 degrades to TF-IDF by contract)")
    print("-" * 90)

    results: dict[str, bool] = {}
    try:
        # ============================================================= RUN 1 (the full chain) =====================
        obj1 = "build a verified proof_artifact answer-function on the first integrated cycle"
        rid1 = f"n8-run1-{int(time.time()*1000)}"
        print(f"[RUN 1] objective: {obj1!r}\n        run_id: {rid1}")
        last1, ev1 = _drive_run(obj1, workspace, build1, rid1, budget=8)
        ev_by = lambda name: [e for e in ev1 if e.get("event") == name]  # noqa: E731

        # ---- LINK 1: PLAN with breadth (N7) -- falsifier (kind=verify) AND/OR generalization (kind=diverge) -----
        plan_ev = ev_by("plan")
        seeded_frontier = last1.get("frontier", []) if last1 else []
        # the planned (pre-mutation) kinds: read from the SEEDED frontier nodes the plan node emitted.
        kinds_planned = {(n.get("kind") or "").lower() for n in seeded_frontier}
        # n1/n2/n3 are the original planned nodes; n_fix arrives only via replan.
        has_falsifier = any(n.get("id") == "n2" and n.get("kind") == "verify" for n in seeded_frontier)
        has_general = any(n.get("id") == "n3" and n.get("kind") == "diverge" for n in seeded_frontier)
        gaps_after = plan_ev[0].get("gaps_after_critique") if plan_ev else "?"
        link1 = bool(plan_ev) and has_falsifier and has_general
        results["LINK 1 (PLAN breadth: falsifier + generalization)"] = link1
        print(f"\n  [LINK 1] PLAN breadth (N7): plan event fired={bool(plan_ev)}; planned kinds={sorted(kinds_planned)}; "
              f"falsifier(kind=verify)={has_falsifier}; generalization(kind=diverge)={has_general}; "
              f"gaps_after_critique={gaps_after}")
        print(f"           -> {'PASS' if link1 else 'FAIL'}")

        # ---- LINK 2: AUDIT (mechanical) -- the FIRST artifact is WRONG and the verifier REFUTED it (not LLM) -----
        judge_ev = ev_by("judge")
        n1_refute = [e for e in judge_ev if e.get("node") == "n1" and e.get("verdict") == "refuted"
                     and e.get("mechanical") is True and int(e.get("exit", 0)) != 0]
        # cross-check: there exists a mechanical PASS somewhere too (so 'refuted' isn't because the verifier is broken).
        any_mech_pass = [e for e in judge_ev if e.get("mechanical") is True and e.get("verdict") == "pass"]
        link2 = bool(n1_refute) and bool(any_mech_pass)
        results["LINK 2 (AUDIT mechanical: wrong artifact REFUTED, exit!=0)"] = link2
        print(f"\n  [LINK 2] AUDIT mechanical (N6): n1 mechanically REFUTED on first attempt={bool(n1_refute)} "
              f"(exit={n1_refute[0].get('exit') if n1_refute else '?'}); a mechanical PASS also exists "
              f"(verifier is not stuck-refusing)={bool(any_mech_pass)}")
        print(f"           -> {'PASS' if link2 else 'FAIL'}")

        # ---- LINK 3: REPLAN + RECOVER (N3) -- replan fired on the failure AND a node reached mechanical PASS ------
        replan_ev = ev_by("replan")
        replan_done = ev_by("replan_done")
        route_to_replan = [e for e in ev_by("route") if e.get("to") == "replan"]
        # recovery: the replan's fix node reached mechanical PASS (done via verify_cmd exit 0).
        fix_pass = [e for e in judge_ev if e.get("node") == ScriptedBrain.FIX_ID and e.get("verdict") == "pass"
                    and e.get("mechanical") is True]
        fix_node_done = any(n.get("id") == ScriptedBrain.FIX_ID and n.get("status") == "done"
                            for n in (last1.get("frontier", []) if last1 else []))
        added_fix = any(ScriptedBrain.FIX_ID in (e.get("added") or []) for e in replan_ev)
        link3 = bool(replan_ev) and bool(route_to_replan) and bool(fix_pass) and fix_node_done and added_fix
        results["LINK 3 (REPLAN fired + RECOVERED to mechanical PASS)"] = link3
        reason0 = replan_ev[0].get("reason") if replan_ev else "?"
        print(f"\n  [LINK 3] REPLAN + RECOVER (N3): route->replan fired={bool(route_to_replan)}; replan event(s)="
              f"{len(replan_ev)} (reason: {str(reason0)[:70]!r}); replan ADDED fix node '{ScriptedBrain.FIX_ID}'="
              f"{added_fix}; fix node reached mechanical PASS={bool(fix_pass)} and status=done={fix_node_done}")
        print(f"           replan_done count={len(replan_done)} (capped, no infinite loop)")
        print(f"           -> {'PASS' if link3 else 'FAIL'}")

        # ---- LINK 4: LEARN -- a lesson was written to the learnings channel during the run -----------------------
        lessons_after1 = _learnings.recent(10 ** 9, channel="proof_n8", workspace=workspace)
        learned = [r for r in lessons_after1 if LESSON_MARKER in (r.get("lesson") or "")]
        # also write it into the unified memory seam (remember -> learnings + best-effort mem0) so RECALL exercises N5.
        wrote_mem0 = _memory.remember(SCRIPTED_LESSON, objective=obj1, channel="proof_n8", workspace=workspace)
        link4 = bool(learned)
        results["LINK 4 (LEARN: lesson persisted to memory/learnings)"] = link4
        print(f"\n  [LINK 4] LEARN: lessons recorded this run={len(lessons_after1)}; the proof lesson "
              f"('{LESSON_MARKER}') persisted={bool(learned)}; memory.remember also-stored-in-mem0={wrote_mem0} "
              f"(False is expected -- mem0 disabled for isolation; lesson is safe in learnings)")
        print(f"           -> {'PASS' if link4 else 'FAIL'}")

        # ============================================================= RUN 2 (compounding RECALL) =================
        # a SIMILAR objective (same lane) -- recall must surface run 1's lesson BEFORE run 2 does any work.
        obj2 = "build a verified proof_artifact answer-function again, reusing what the prior cycle learned"
        print(f"\n[RUN 2] objective (SIMILAR): {obj2!r}")
        # ---- LINK 5: RECALL (compounding, N5) -- recall(obj2) surfaces run 1's lesson --------------------------
        recall_digest = _memory.recall(obj2, k=5, channel="proof_n8", workspace=workspace)
        recalled = LESSON_MARKER in (recall_digest or "")
        # prove it is genuinely cross-cycle: a recall on a FRESH/empty channel must NOT surface it (negative control).
        empty_ws = tempfile.mkdtemp(prefix="n8_proof_emptyws_")
        try:
            control_digest = _memory.recall(obj2, k=5, channel="proof_n8", workspace=empty_ws)
        finally:
            shutil.rmtree(empty_ws, ignore_errors=True)
        control_clean = LESSON_MARKER not in (control_digest or "")
        link5 = recalled and control_clean
        results["LINK 5 (RECALL: prior lesson surfaces on a similar objective)"] = link5
        print(f"\n  [LINK 5] RECALL compounding (N5): recall(obj2) surfaced the prior lesson={recalled}; "
              f"negative control (empty workspace) does NOT surface it={control_clean}")
        # show the actual recall evidence (trimmed)
        rd = (recall_digest or "").strip().replace("\n", "\n             ")
        print(f"           recall digest:\n             {rd[:600]}")
        print(f"           -> {'PASS' if link5 else 'FAIL'}")

        # run 2 actually drives the loop too (a real second cycle), to show the engine RE-runs with memory available
        # (the same failure->replan->recover chain, now with run 1's lesson recallable in the plan payload).
        rid2 = f"n8-run2-{int(time.time()*1000)}"
        last2, _ev2 = _drive_run(obj2, workspace, build2, rid2, budget=8)
        print(f"  [RUN 2] re-ran the full loop with memory available -> terminal status="
              f"{last2.get('status') if last2 else '?'}")

        # ============================================================= TRACE EVIDENCE =============================
        print("\n" + "-" * 90)
        print("TRACE EVIDENCE (run 1 -- the actual JSONL the graph wrote; the audit cannot be faked):")
        for e in ev1:
            ev = e.get("event")
            if ev in ("plan", "judge", "reflect", "route", "replan", "replan_done", "dispatch"):
                slim = {k: v for k, v in e.items() if k not in ("t",)}
                print("   " + json.dumps(slim, default=str)[:200])

        # ============================================================= VERDICT ====================================
        print("\n" + "=" * 90)
        print("SELF-REINFORCING LOOP -- PER-LINK VERDICT")
        print("=" * 90)
        for name, ok in results.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        overall = all(results.values())
        print("-" * 90)
        print(f"  OVERALL: {'PASS' if overall else 'FAIL'}  "
              f"({sum(results.values())}/{len(results)} links proven on the REAL graph.build engine)")
        print("=" * 90)

        # ============================================================= REAL OLLAMA DATAPOINT (honest) =============
        print("\n" + "-" * 90)
        print("REAL DATAPOINT (separate from the deterministic proof above -- never fails it):")
        print(_real_ollama_datapoint())
        print("-" * 90)

        return 0 if overall else 1
    finally:
        for d in (workspace, build1, build2):
            shutil.rmtree(d, ignore_errors=True)
        # confirm cleanup (isolation evidence)
        leftover = [d for d in (workspace, build1, build2) if Path(d).exists()]
        print(f"\n  [isolation] temp dirs removed: {'all clean' if not leftover else 'LEFTOVER: ' + str(leftover)}")


if __name__ == "__main__":
    raise SystemExit(main())
