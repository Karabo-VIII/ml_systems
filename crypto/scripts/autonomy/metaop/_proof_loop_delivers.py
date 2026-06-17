"""PROOF DRIVER -- does the metaop LOOP autonomously DELIVER a mechanically-verified artifact end-to-end?

This closes the "worker builds but the loop doesn't deliver" gap. It runs the REAL metaop graph (graph.build,
the same construction manager.launch uses -- NOT a reimplementation) with the SDK brain (AgentSdkBrain, Opus on
the subscription), seeded with ONE build node that carries a Windows-form verify_cmd. The judge (AlphaProof-Nexus
mechanical verifier) RUNS that verify_cmd: exit==0 is ground-truth PASS (overrides the LLM panel), exit!=0 REFUTES
and feeds the concrete error back to the worker (rejection-as-gradient) for up to verify_retries fixes.

Run:  .venv\Scripts\python.exe scripts\autonomy\metaop\_proof_loop_delivers.py
No emoji (Windows cp1252). Does NOT commit (worker fence + this driver never touches git).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# --- make `metaop` importable exactly like the package expects (scripts/autonomy on sys.path) ----------------
HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]                          # repo root (…/ml_systems)
AUTONOMY = HERE.parents[1]                       # …/scripts/autonomy  (contains the `metaop` package)
sys.path.insert(0, str(AUTONOMY))

from metaop.graph import build, TRACE_DIR        # the REAL graph construction (same as manager.launch)
from metaop.brain import make_brain

# --- the artifact the loop must DELIVER, and the MECHANICAL command that is ground truth --------------------
ARTIFACT = ROOT / "runs" / "autonomy" / "_proof_fib.py"
# Windows form: ABSOLUTE interpreter + ABSOLUTE script path, BACKSLASHES (shell=True on cmd.exe rejects /-paths).
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
VERIFY_CMD = f"{str(VENV_PY)} {str(ARTIFACT)}"   # both already use OS-native backslashes on Windows

SEED_NODE = {
    "id": "fib1",
    "task": (
        "Create the file runs/autonomy/_proof_fib.py containing a function fib(n) (iterative) and a "
        "__main__ block that prints fib(10) and asserts fib(10) == 55 and fib(20) == 6765. Keep it tiny."
    ),
    "ev": 0.95,
    "kind": "build",
    "status": "open",
    "verify_cmd": VERIFY_CMD,
    "verify_retries": 2,
}


def _init_state(objective: str, budget: int, parallel: int, run_id: str) -> dict:
    """OpState seed. Pre-seeding `frontier` makes the graph's plan node return {} early (graph.py L94-95), so
    OUR single build node is what the loop works -- no LLM planning, fully bounded + deterministic frontier."""
    return {
        "objective": objective,
        "success_criteria": "the artifact exists AND its verify_cmd exits 0 (mechanical ground truth)",
        "frontier": [dict(SEED_NODE)],
        "ledger": [],
        "budget": budget,
        "cycle": 0,
        "status": "running",
        "parallel": parallel,
        "run_id": run_id,
        "awaiting_approval": [],
    }


def main() -> int:
    print("=" * 78)
    print("METAOP LOOP DELIVERY PROOF -- mechanical verifier (AlphaProof-Nexus judge)")
    print("=" * 78)
    print(f"  repo ROOT     : {ROOT}")
    print(f"  artifact      : {ARTIFACT}")
    print(f"  verify_cmd    : {VERIFY_CMD}")

    # Clean slate: the loop's WORKER must create the artifact, not a stale prior run.
    if ARTIFACT.exists():
        ARTIFACT.unlink()
        print(f"  (removed pre-existing artifact so the loop must re-create it)")
    print(f"  artifact exists BEFORE run: {ARTIFACT.exists()}")

    brain = make_brain("sdk")
    print(f"  brain         : {brain.name}  (expect AgentSdkBrain -- Opus on subscription)")
    if brain.name != "AgentSdkBrain":
        print("  WARNING: SDK brain not selected -- the proof would not exercise a real Claude worker.")

    budget, parallel = 4, 1                       # BOUNDED: 1 node, few cycles, single worker
    run_id = f"proofdeliver-{int(time.time())}"
    app = build(brain, parallel=parallel, judges=1, taper=1)   # taper=1 -> reflect generates no adjacent padding
    cfg = {"configurable": {"thread_id": run_id}}

    t0 = time.time()
    last = None
    print("\n--- LIVE LOOP TRACE (one line per graph super-step) -------------------------")
    for step in app.stream(_init_state("prove the loop delivers a verified artifact", budget, parallel, run_id),
                            cfg, stream_mode="values"):
        last = step
        fr = step.get("frontier", [])
        node = next((n for n in fr if n["id"] == "fib1"), {})
        print(f"  cycle={step.get('cycle'):>2}  status={step.get('status'):<12}  "
              f"node.status={node.get('status','?'):<16}  verdict={node.get('verdict','-'):<12}  "
              f"verify_retries_left={node.get('verify_retries','-')}")
    dt = time.time() - t0

    print(f"\n--- RUN COMPLETE in {dt:.1f}s -----------------------------------------------")
    fr = last["frontier"]
    print(f"  final loop status : {last['status']}")
    print(f"  cycles run        : {last['cycle']}")
    print(f"  frontier statuses : {dict(Counter(n.get('status') for n in fr))}")

    node = next((n for n in fr if n["id"] == "fib1"), {})
    print(f"  fib1 final verdict: {node.get('verdict')}")
    print(f"  fib1 final status : {node.get('status')}")

    # --- replay the JSONL trace so every dispatch/judge (incl. the MECHANICAL exit code) is visible ----------
    tr = TRACE_DIR / f"{run_id}.jsonl"
    print(f"\n--- FULL JSONL TRACE  ({tr}) ------------------------------------------------")
    if tr.exists():
        for line in tr.read_text(encoding="utf-8").strip().splitlines():
            ev = json.loads(line)
            kind = ev.get("event")
            if kind == "judge":
                print(f"  [judge]   node={ev.get('node')}  verdict={ev.get('verdict')}  "
                      f"mechanical={ev.get('mechanical')}  exit={ev.get('exit')}  "
                      f"reopened={ev.get('reopened','-')}  retries_left={ev.get('retries_left','-')}  "
                      f"votes={ev.get('votes','-')}")
            elif kind == "dispatch":
                print(f"  [dispatch] ran={ev.get('ran')}  parallel={ev.get('parallel')}")
            else:
                print(f"  [{kind}] {({k: v for k, v in ev.items() if k not in ('t', 'event')})}")
    else:
        print("  (no trace file written)")

    # --- RWYB: independently confirm the artifact exists + the verifier exits 0 ------------------------------
    print("\n--- RWYB INDEPENDENT VERIFICATION ------------------------------------------")
    print(f"  artifact exists AFTER run : {ARTIFACT.exists()}")
    verdict_ok = node.get("verdict") == "pass" and node.get("status") == "done"
    delivered = ARTIFACT.exists() and verdict_ok
    print(f"  loop reached PASS via mechanical judge: {verdict_ok}")
    print(f"\n  HONEST VERDICT: loop autonomously delivered a mechanically-verified artifact = {delivered}")
    return 0 if delivered else 1


if __name__ == "__main__":
    raise SystemExit(main())
