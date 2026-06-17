#!/usr/bin/env python3
"""CLI for the metaop EVAL / FITNESS HARNESS (the keystone metric).

Runs the built-in benchmark through the metaop engine and writes a SCORECARD (solve_rate + per-task) to
runs/autonomy/eval/<label>.json. The score is MECHANICAL (graph._run_verify exit 0 == PASS) -- it cannot be faked
by the brain's self-report. This single number is what later unlocks DSPy (optimize the planner prompt against it)
and OpenEvolve (evolve the engine against it).

TWO MODES:
  - PRE-SEEDED (default): the harness pre-seeds ONE node/task -> measures the WORKER+VERIFIER (NOT the planner).
  - PLANNER (--planner-mode): the harness passes ONLY the objective -> the BRAIN must DECOMPOSE it into the right
    multi-step plan and the loop must execute it; the harness scores the FINAL COMPOSED artifact. solve_rate here =
    PLANNER QUALITY. This is the honest objective a future DSPy pass must beat. Tasks default to PLANNER_BENCHMARK.

Examples:
  # PLUMBING (deterministic, no creds): proves scorer + isolation + timeout work -> solve_rate should be 1.0
  python scripts/autonomy/eval_harness_run.py --brain mock

  # PLANNER-MODE PLUMBING: the PlannerOracleMockBrain decomposes + builds -> solve_rate should be 1.0
  python scripts/autonomy/eval_harness_run.py --planner-mode --brain mock

  # REAL local datapoint: a weak 3b model on 2 tasks -> an honest measured number (may not pass all -- fine)
  python scripts/autonomy/eval_harness_run.py --brain litellm --model ollama/qwen2.5-coder:3b --tasks fib,is_prime

  # REAL PLANNER baseline (the number DSPy must beat): a weak 3b must decompose AND build
  python scripts/autonomy/eval_harness_run.py --planner-mode --brain litellm --model ollama/qwen2.5-coder:3b

Flags:
  --brain        mock|litellm|ollama|sdk|cli|api|auto   (mock => the honest oracle plumbing path)
  --model        model string for litellm/ollama (e.g. ollama/qwen2.5-coder:3b)
  --tasks        comma-separated task ids to run (default: ALL of the active mode's benchmark). Use --list.
  --planner-mode measure the PLANNER (brain decomposes; default benchmark = PLANNER_BENCHMARK)
  --budget       max graph cycles per task (default 4)
  --timeout      per-task wall-clock cap in seconds (default 240)
  --label        scorecard filename stem (default: derived from brain/model + mode + a timestamp)
  --list         print the active mode's benchmark task ids and exit

Repo root is added to sys.path so `harness.metaop` imports without PYTHONPATH. No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (…/ml_systems)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.metaop import eval_harness as ev   # noqa: E402  the canonical, agnostic eval harness
from harness.metaop.brain import make_brain      # noqa: E402

# write scorecards under the repo's runs/autonomy (same workspace convention the live crypto loop uses).
_WORKSPACE = str(ROOT / "runs" / "autonomy")


def _select_tasks(spec: str | None, bench: list):
    """Pick tasks from the ACTIVE benchmark (BENCHMARK pre-seeded, PLANNER_BENCHMARK in planner mode)."""
    if not spec:
        return bench
    want = {s.strip() for s in spec.split(",") if s.strip()}
    chosen = [t for t in bench if t["id"] in want]
    unknown = want - {t["id"] for t in chosen}
    if unknown:
        print(f"[eval] WARNING: unknown task id(s) ignored: {sorted(unknown)} "
              f"(known: {ev.benchmark_ids(bench)})")
    return chosen or bench


def main() -> int:
    ap = argparse.ArgumentParser(description="metaop eval/fitness harness -- benchmark solve_rate")
    ap.add_argument("--brain", default="mock",
                    help="mock|litellm|ollama|sdk|cli|api|auto  (mock => honest OracleMockBrain plumbing path)")
    ap.add_argument("--model", default=None, help="model string for litellm/ollama (e.g. ollama/qwen2.5-coder:3b)")
    ap.add_argument("--tasks", default=None, help="comma-separated task ids (default: ALL of the active benchmark)")
    ap.add_argument("--planner-mode", dest="planner_mode", action="store_true",
                    help="measure the PLANNER: the brain decomposes the objective (default benchmark = PLANNER_BENCHMARK)")
    ap.add_argument("--budget", type=int, default=4, help="max graph cycles per task")
    ap.add_argument("--timeout", type=int, default=240, help="per-task wall-clock cap (seconds)")
    ap.add_argument("--label", default=None, help="scorecard filename stem")
    ap.add_argument("--list", action="store_true", help="print the active mode's benchmark task ids and exit")
    args = ap.parse_args()

    bench = ev.PLANNER_BENCHMARK if args.planner_mode else ev.BENCHMARK
    if args.list:
        mode = "PLANNER_BENCHMARK" if args.planner_mode else "BENCHMARK"
        print(f"{mode} tasks:", ev.benchmark_ids(bench))
        return 0

    tasks = _select_tasks(args.tasks, bench)
    oracle_mock = (args.brain == "mock")

    print("=" * 78)
    print("METAOP EVAL / FITNESS HARNESS -- mechanical solve_rate (cannot be self-certified)")
    print("=" * 78)
    print(f"  brain   : {args.brain}" + (f"  model={args.model}" if args.model else ""))
    print(f"  eval    : {'PLANNER (brain decomposes -- measures PLANNER QUALITY)' if args.planner_mode else 'PRE-SEEDED (measures WORKER+VERIFIER)'}")
    print(f"  tasks   : {[t['id'] for t in tasks]}")
    print(f"  budget  : {args.budget} cycles/task   timeout: {args.timeout}s/task")
    if oracle_mock:
        oname = "PlannerOracleMockBrain" if args.planner_mode else "OracleMockBrain"
        print(f"  mode    : PLUMBING ({oname} {'decomposes + ' if args.planner_mode else ''}writes correct "
              "artifacts; the MECHANICAL verifier still scores)")
        brain = None
        label_default = ("planner_" if args.planner_mode else "") + "mock_plumbing"
    else:
        brain = make_brain(args.brain, domain="eval benchmark (build verifiable artifacts)", model=args.model)
        print(f"  resolved brain object: {brain.name}")
        label_default = (("planner_" if args.planner_mode else "")
                         + (args.model or args.brain).replace("/", "_").replace(":", "-"))
    print("-" * 78)

    t0 = time.time()
    card = ev.run_eval(brain, tasks=tasks, budget=args.budget, timeout=args.timeout, oracle_mock=oracle_mock,
                       planner_mode=args.planner_mode)
    dt = time.time() - t0

    print("-" * 78)
    print(f"  SOLVE_RATE = {card['solve_rate']:.4f}   ({card['n_passed']}/{card['n']} tasks)   in {dt:.1f}s")
    print("-" * 78)

    label = args.label or f"{label_default}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = ev.write_scorecard(card, label, workspace=_WORKSPACE)
    print(f"  scorecard -> {out}")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
