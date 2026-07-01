"""DAG orchestrator — topo-walk config/asset_dag.yaml, run stages in order.

A stage runs only after ALL its deps have succeeded; a stage whose dep failed is
skipped (so the gold dataset is never built on missing bars). Each producer is
self-sufficient (own skip-fresh + parallelism); refresh.py adds ordering + gating.

    python src/pipeline/refresh.py --all --universe u7 --start 2026-06-01 --end 2026-07-01
    python src/pipeline/refresh.py --target dataset --universe u7   # dataset + its deps
    python src/pipeline/refresh.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_PIPELINE_DIR = Path(__file__).resolve().parent
_PIPELINE_PARENT = str(_PIPELINE_DIR.parent)
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout  # noqa: E402

__contract__ = {
    "kind": "pipeline_orchestrator",
    "stage": "refresh",
    "inputs": {"config": "config/asset_dag.yaml",
               "args": ["--all/--target", "--universe/--pairs", "--start", "--end",
                        "--cadences", "--bar-cadences", "--workers"]},
    "outputs": {"side_effects": "runs producer stages in dependency order"},
    "invariants": {"deps_gate_downstream": True, "topo_order": True},
    "rationale": "One DAG-aware command from raw fetch to gold dataset + metrics.",
}

DAG_PATH = layout.PROJECT_ROOT / "config" / "asset_dag.yaml"


def _load_dag() -> dict:
    with open(DAG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)["stages"]


def _topo(stages: dict, targets: list[str]) -> list[str]:
    """Return stages needed for `targets`, in dependency order."""
    order, seen = [], set()

    def visit(name: str):
        if name in seen:
            return
        seen.add(name)
        for dep in stages[name].get("deps", []):
            visit(dep)
        order.append(name)

    for t in targets:
        visit(t)
    return order


def _stage_cmd(name: str, spec: dict, args) -> list[str]:
    py = sys.executable
    cmd = [py, str(layout.PROJECT_ROOT / spec["producer"])]
    takes = set(spec.get("takes", []))
    # target selector
    if "target" in takes:
        if args.pairs:
            cmd += ["--pairs", *args.pairs]
        elif args.universe:
            cmd += ["--universe", args.universe]
    if "source" in takes:
        cmd += ["--source", args.source]
    if "bar_cadences" in takes:
        cmd += ["--cadences", *args.bar_cadences]
    elif "cadences" in takes:
        cmd += ["--cadences", *args.cadences]
    if "date_window" in takes:
        cmd += ["--start", args.start]
        if args.end:
            cmd += ["--end", args.end]
    if "workers" in takes:
        cmd += ["--workers", str(args.workers)]
    if "report" in takes:
        cmd += ["--report"]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex pipeline DAG orchestrator")
    ap.add_argument("--all", action="store_true", help="Run every stage.")
    ap.add_argument("--target", default=None, help="Run this stage + its deps.")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--universe", default=None, choices=["u7", "u28", "u52"])
    ap.add_argument("--source", default="dukascopy", choices=["dukascopy", "histdata", "oanda"])
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--bar-cadences", nargs="+",
                    default=["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
    ap.add_argument("--cadences", nargs="+", default=["1h", "1d"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="List stages + deps and exit.")
    args = ap.parse_args()

    stages = _load_dag()
    if args.list:
        for name, spec in stages.items():
            print(f"  {name:<10} deps={spec.get('deps', [])}  -- {spec.get('desc', '')}")
        return

    if args.all:
        targets = list(stages)
    elif args.target:
        if args.target not in stages:
            raise SystemExit(f"unknown target {args.target!r}; stages: {list(stages)}")
        targets = [args.target]
    else:
        raise SystemExit("pass --all or --target <stage> (see --list)")
    if not args.universe and not args.pairs:
        args.universe = "u7"

    order = _topo(stages, targets)
    print(f"[refresh] plan: {' -> '.join(order)}", flush=True)
    failed: set[str] = set()
    for name in order:
        spec = stages[name]
        blocked = [d for d in spec.get("deps", []) if d in failed]
        if blocked:
            print(f"\n[refresh] SKIP {name} (dep failed: {blocked})", flush=True)
            failed.add(name)
            continue
        cmd = _stage_cmd(name, spec, args)
        print(f"\n{'=' * 70}\n[refresh] STAGE {name}\n  $ {' '.join(cmd)}\n{'=' * 70}", flush=True)
        rc = subprocess.run(cmd).returncode
        # quality returns the worst-status code in gate mode; in --report it's 0.
        # Treat rc>=2 as a hard failure that blocks downstream deps.
        if rc >= 2:
            print(f"[refresh] STAGE {name} FAILED rc={rc}", flush=True)
            failed.add(name)
        else:
            print(f"[refresh] STAGE {name} ok rc={rc}", flush=True)

    print(f"\n[refresh] DONE. ran={[s for s in order if s not in failed]} "
          f"failed={sorted(failed)}", flush=True)
    sys.exit(2 if failed else 0)


if __name__ == "__main__":
    main()
