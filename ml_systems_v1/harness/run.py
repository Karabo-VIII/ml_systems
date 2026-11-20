"""harness/run.py -- the clean standalone entrypoint for the autonomous engine-builder loop.

Run ONE objective through the plan->dispatch->judge->reflect->route loop and report the result. Project-agnostic:
it imports ONLY the harness package (no host project). Works with zero credentials via the MockBrain.

  # smallest possible run -- no API key, no network: proves the whole machinery end-to-end
  python harness/run.py --objective "trivial" --backend mock

  # a single BUILD node verified by a MECHANICAL command (exit 0 = ground-truth pass, overrides any LLM vote):
  python harness/run.py --objective "create hello.py that prints hi" --backend mock \
      --verify-cmd "python -c \"print('ok')\""

  # real Claude (in-process SDK) building in a chosen target dir:
  python harness/run.py --objective "build X" --backend sdk --cwd /path/to/project

Backends: mock (default, no creds) | sdk (claude-agent-sdk) | api (ANTHROPIC_API_KEY) | cli (claude on PATH)
  | ollama (a LOCAL open-source model via the Ollama server -- runs with NO Claude; proves model-portability).
The --verify-cmd is the mechanical-verifier contract: it runs from --cwd; exit 0 == PASS. No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

# Import the engine in BOTH layouts: standalone (harness/ is the repo root -> top-level `metaop`) AND in this repo
# (harness/ is a subdir -> namespace `harness.metaop`). Put run.py's OWN dir on the path (for `metaop`) and its parent
# (for `harness.metaop`), then try the standalone name first.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # harness/   -> `metaop`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # repo root  -> `harness.metaop`
try:
    from metaop.brain import make_brain
    from metaop.graph import build
    from metaop.config import workspace_root, trace_dir, build_cwd
except ImportError:  # pragma: no cover -- in-repo fallback if the namespace form is preferred
    from harness.metaop.brain import make_brain
    from harness.metaop.graph import build
    from harness.metaop.config import workspace_root, trace_dir, build_cwd


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harness.run", description="Standalone autonomous engine-builder loop.")
    ap.add_argument("--objective", required=True, help="what to build / investigate")
    ap.add_argument("--success", default="the objective is achieved and verified by evidence")
    ap.add_argument("--backend", default="mock", choices=["mock", "sdk", "api", "cli", "auto", "ollama"],
                    help="brain backend (mock needs no creds; ollama = a local open-source model, no Claude)")
    ap.add_argument("--verify-cmd", default=None,
                    help="mechanical verifier: a shell command run from --cwd; exit 0 == ground-truth PASS")
    ap.add_argument("--verify-retries", type=int, default=1,
                    help="how many times a refuted verify_cmd node is re-dispatched with the concrete error")
    ap.add_argument("--cwd", default=None, help="build cwd where the worker/verifier run (default: current dir)")
    ap.add_argument("--workspace", default=None, help="harness bookkeeping dir (default: ./.harness_runs)")
    ap.add_argument("--domain", default=None, help="task-flavor injected into the brain prompts (optional)")
    ap.add_argument("--budget", type=int, default=2, help="max reflect cycles")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--judges", type=int, default=1)
    ap.add_argument("--taper", type=int, default=1, help="stop generating adjacent nodes after this cycle")
    ap.add_argument("--skills-dir", default=None, help="dir to HARVEST a verified capability into as a new SKILL.md")
    ap.add_argument("--harvest", action="store_true",
                    help="SELF-AUGMENT: if the build passes its --verify-cmd, author a SKILL.md for it into "
                         "--skills-dir (so the harness grows its own skill library). Requires --skills-dir + --verify-cmd.")
    args = ap.parse_args(argv)

    if args.workspace:
        os.environ["HARNESS_WORKSPACE"] = args.workspace
    ws = str(workspace_root(args.workspace))
    cwd = str(build_cwd(args.cwd))

    brain = make_brain(args.backend, domain=args.domain or "a software engine-builder project (build verifiable artifacts in any domain)", cwd=args.cwd)
    run_id = f"run-{int(time.time())}"

    # ONE seed node. If --verify-cmd is given, the node carries it (the judge runs it mechanically) + a retry budget.
    seed = {"id": "n1", "task": f"build: {args.objective}", "ev": 1.0, "kind": "build", "status": "open"}
    if args.verify_cmd:
        seed["verify_cmd"] = args.verify_cmd
        seed["verify_retries"] = args.verify_retries

    harvester = None
    if args.harvest and args.skills_dir:  # SELF-AUGMENT: grow the skill library from a verified build
        try:
            from metaop.skills import skill_harvester
        except ImportError:
            from harness.metaop.skills import skill_harvester
        harvester = skill_harvester(args.skills_dir)

    app = build(brain, parallel=args.parallel, judges=args.judges, taper=args.taper,
                channel="run", workspace=args.workspace, cwd=args.cwd, harvester=harvester)

    init = {"objective": args.objective, "success_criteria": args.success,
            "frontier": [seed],  # pre-seeded so the run is a deterministic ONE-build-node demonstration
            "ledger": [], "budget": args.budget, "cycle": 0, "status": "running",
            "parallel": args.parallel, "run_id": run_id, "awaiting_approval": []}

    print(f"=== HARNESS run  brain={brain.name}  backend={args.backend}  verify_cmd={bool(args.verify_cmd)} ===")
    print(f"  objective : {args.objective}")
    print(f"  workspace : {ws}")
    print(f"  build cwd : {cwd}")

    cfg = {"configurable": {"thread_id": run_id}}
    last = None
    for step in app.stream(init, cfg, stream_mode="values"):
        last = step

    fr = last["frontier"]
    n1 = next((n for n in fr if n["id"] == "n1"), {})
    print("\n=== RESULT ===")
    print(f"  status        : {last['status']}   cycles: {last['cycle']}")
    print(f"  frontier      : {dict(Counter(n.get('status') for n in fr))}  ({len(fr)} nodes)")
    print(f"  n1 verdict    : {n1.get('verdict')}  (worker_ok={n1.get('worker_ok')})")
    if n1.get("verify_cmd"):
        print(f"  n1 verify_cmd : {n1.get('verify_cmd')}")
        if n1.get("verify_error"):
            print(f"  n1 verify_err : {str(n1.get('verify_error'))[:300]}")
    print(f"  n1 result     : {str(n1.get('result'))[:300]}")
    print(f"  ledger        : {len(last['ledger'])} lesson(s): {last['ledger']}")
    print(f"  trace         : {trace_dir(args.workspace)}/{run_id}.jsonl")

    # exit 0 only when the single build node reached a 'done'/'pass' terminal state; else NON-ZERO so callers can
    # script on it (this was a bug: both branches returned 0, making the exit code meaningless).
    return 0 if (n1.get("status") == "done" or n1.get("verdict") == "pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
