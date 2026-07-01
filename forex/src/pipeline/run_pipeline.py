"""End-to-end pipeline orchestrator: fetch -> bars -> data-quality.

Thin runner over the self-sufficient stages (each is independently runnable).
Runs them in order via subprocess so a stage failure is isolated and visible,
then prints what is usable after each stage.

    # bring up the majors over the last week and validate
    python src/pipeline/run_pipeline.py --universe u7 --start 2026-06-22 --end 2026-07-01

    # just re-derive bars + metrics from already-fetched ticks
    python src/pipeline/run_pipeline.py --universe u7 --skip-fetch --cadences 1m 1h 1d
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent
_PIPELINE_PARENT = str(_PIPELINE_DIR.parent)
if _PIPELINE_PARENT not in sys.path:
    sys.path.insert(0, _PIPELINE_PARENT)

from pipeline import layout  # noqa: E402

__contract__ = {
    "kind": "pipeline_orchestrator",
    "stage": "run_pipeline",
    "inputs": {"args": ["--universe/--pairs", "--start", "--end", "--cadences",
                        "--source", "--skip-fetch", "--skip-bars", "--skip-quality"]},
    "outputs": {"side_effects": "runs fetch_all + build_bars + data_quality in order"},
    "invariants": {"explicit_stage_order": True, "stage_failure_visible": True},
    "rationale": "One command to go from nothing to validated bars + metrics.",
}


def _target_args(args) -> list[str]:
    """The --pairs / --universe selector shared by every stage."""
    if args.pairs:
        return ["--pairs", *args.pairs]
    return ["--universe", args.universe or "u7"]


def _run(label: str, cmd: list[str]) -> int:
    print(f"\n{'=' * 70}\n[run_pipeline] STAGE: {label}\n  $ {' '.join(cmd)}\n{'=' * 70}",
          flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"[run_pipeline] STAGE {label} -> rc={rc}", flush=True)
    return rc


def main() -> None:
    ap = argparse.ArgumentParser(description="Forex pipeline orchestrator")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--universe", default=None, choices=["u7", "u28", "u52"])
    ap.add_argument("--start", default="2026-06-22")
    ap.add_argument("--end", default=None)
    ap.add_argument("--cadences", nargs="+", default=["1m", "5m", "15m", "1h", "4h", "1d"])
    ap.add_argument("--source", default="dukascopy", choices=["dukascopy", "histdata", "oanda"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-bars", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    fetch_py = str(_PIPELINE_DIR / "fetch_all.py")
    bars_py = str(_PIPELINE_DIR / "build_bars.py")
    dq_py = str(_PIPELINE_DIR / "data_quality.py")
    tgt = _target_args(args)
    end_args = ["--end", args.end] if args.end else []

    rc_total = 0
    if not args.skip_fetch:
        rc_total |= _run("fetch", [py, fetch_py, "--source", args.source, *tgt,
                                   "--start", args.start, *end_args,
                                   "--workers", str(args.workers)])
    if not args.skip_bars:
        rc_total |= _run("build_bars", [py, bars_py, *tgt,
                                        "--cadences", *args.cadences,
                                        "--start", args.start, *end_args,
                                        "--workers", str(args.workers)])
    if not args.skip_quality:
        # report mode: never hard-fails the orchestrator; metrics JSON is written.
        _run("data_quality", [py, dq_py, *tgt, "--cadences", *args.cadences, "--report"])

    print(f"\n[run_pipeline] DONE. Bars under: "
          f"{(layout.PROCESSED / 'bars').relative_to(layout.PROJECT_ROOT)}/<cadence>/  "
          f"Metrics under: runs/pipeline_metrics/", flush=True)
    sys.exit(0 if rc_total == 0 else 1)


if __name__ == "__main__":
    main()
