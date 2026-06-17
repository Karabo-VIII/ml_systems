"""Unified data-layer refresh + validation command.

Runs the full update/upgrade/validate cycle for the entire data pipeline
in ONE command. Composes:

  1. registry_contract_test       — gate feature_registry.yaml edits
  2. CDAP check_invariants        — verify cross-cutting invariants survive
  3. refresh.py --all             — DAG-walk every producer (T0-T6),
                                    streams subprocess stdout to terminal,
                                    runs pre_train_gate as the final stage
  4. validate_chimera --asset X   — per-asset 20-check validation (sample)
  5. CDAP check_invariants (again) — post-build invariant sweep

Convention spec: docs/PIPELINE_PROGRESS_CONVENTION_2026_05_22.md
Canonical reference: docs/DATA_LAYER_CANONICAL_REFERENCE_2026_05_22.md

Usage:
    python src/pipeline/refresh_full.py                    # u100 default
    python src/pipeline/refresh_full.py --universe u10     # smaller universe
    python src/pipeline/refresh_full.py --universe u100 --force  # full rebuild
    python src/pipeline/refresh_full.py --skip-refresh     # validators only
    python src/pipeline/refresh_full.py --asset BTC --quick # single-asset smoke

Exit codes:
    0  all stages clean (or warn-only)
    1  at least one stage emitted WARNs
    2  at least one stage emitted CRITICAL / FAIL — halt before training

Provenance: oracle pipeline-A+ closure 2026-05-22. Closes the "no single
command to update everything" gap surfaced by the user this session.
"""
from __future__ import annotations

# CDAP contract
__contract__ = {
    "kind": "pipeline_orchestrator",
    "stage": "refresh_full",
    "inputs": {
        "args": ["--universe", "--asset", "--force", "--skip-refresh", "--quick", "--dry-run"],
        "upstream": ["all bronze + silver + gold + registry"],
    },
    "outputs": {
        "exit_code": "0 clean / 1 warn / 2 CRITICAL",
        "stdout": "phase_log lines from each composed stage",
    },
    "invariants": {
        "deterministic_order": True,
        "fail_fast_on_critical": True,
        "no_data_write_by_self": True,
    },
    "rationale": "Single unified command for full data-layer refresh + validation. Composes registry contract test + CDAP + refresh DAG walk + validate_chimera + post-build CDAP.",
}

import argparse
import subprocess
import sys
from pathlib import Path
from time import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from progress import phase_log, stage_run
except ImportError:
    from pipeline.progress import phase_log, stage_run


def _run_step(label: str, cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    """Run a step with phase_log instrumentation. Streams stdout to terminal."""
    phase_log("refresh_full", "START", f"step: {label}", counters=None)
    phase_log("refresh_full", "BUILD", f"  cmd: {' '.join(cmd)}")
    t0 = time()
    try:
        # Stream stdout/stderr to terminal directly (do not capture).
        rc = subprocess.run(cmd, cwd=str(cwd)).returncode
    except FileNotFoundError as e:
        phase_log("refresh_full", "FAIL", f"step '{label}' executable not found: {e}",
                  counters={"elapsed": time() - t0})
        return 2
    except KeyboardInterrupt:
        phase_log("refresh_full", "FAIL", f"step '{label}' interrupted by user",
                  counters={"elapsed": time() - t0})
        return 130
    elapsed = time() - t0
    if rc == 0:
        phase_log("refresh_full", "OK", f"step '{label}' clean",
                  counters={"elapsed": elapsed})
    elif rc == 1:
        phase_log("refresh_full", "WARN", f"step '{label}' WARN-only (rc=1)",
                  counters={"elapsed": elapsed})
    else:
        phase_log("refresh_full", "FAIL", f"step '{label}' CRITICAL (rc={rc})",
                  counters={"elapsed": elapsed})
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Unified data-layer refresh + validation (single command).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Composes 5 stages:\n"
            "  1. registry_contract_test\n"
            "  2. CDAP check_invariants (pre-refresh)\n"
            "  3. refresh.py --all (DAG walk; includes pre_train_gate)\n"
            "  4. validate_chimera (sample asset)\n"
            "  5. CDAP check_invariants (post-refresh)\n"
            "\n"
            "Exit codes: 0 clean / 1 warn / 2 CRITICAL.\n"
            "\n"
            "See docs/DATA_LAYER_CANONICAL_REFERENCE_2026_05_22.md for full spec.\n"
        ),
    )
    ap.add_argument("--universe", default="u100", choices=["u10", "u50", "u100"],
                    help="Universe to refresh (default: u100)")
    ap.add_argument("--asset", default="BTC",
                    help="Sample asset for validate_chimera step (default: BTC)")
    ap.add_argument("--force", action="store_true",
                    help="Force rebuild of all producer outputs (skip freshness checks)")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="Skip the refresh.py DAG walk; only run validators + CDAP")
    ap.add_argument("--quick", action="store_true",
                    help="Quick mode: skip CDAP post-check + skip non-essential steps")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show the command sequence; don't run anything")
    args = ap.parse_args()

    PYTHON = sys.executable

    # Build step sequence.
    steps: list[tuple[str, list[str]]] = []

    steps.append((
        "registry_contract_test",
        [PYTHON, "src/pipeline/registry_contract_test.py"],
    ))

    steps.append((
        "cdap_pre_refresh",
        [PYTHON, "src/audit/check_invariants.py"],
    ))

    if not args.skip_refresh:
        refresh_cmd = [PYTHON, "src/pipeline/refresh.py", "--all",
                       "--universe", args.universe]
        if args.force:
            refresh_cmd.append("--force")
        steps.append(("refresh_dag_walk", refresh_cmd))

    steps.append((
        f"validate_chimera_{args.asset}",
        [PYTHON, "src/pipeline/validate_chimera.py", "--asset", args.asset],
    ))

    if not args.quick:
        steps.append((
            "cdap_post_refresh",
            [PYTHON, "src/audit/check_invariants.py"],
        ))

    # Banner.
    with stage_run("refresh_full",
                    f"unified data-layer refresh ({args.universe}, "
                    f"force={args.force}, skip_refresh={args.skip_refresh}, "
                    f"quick={args.quick})"):

        if args.dry_run:
            for label, cmd in steps:
                phase_log("refresh_full", "SCAN",
                          f"[DRY-RUN] would run '{label}': {' '.join(cmd)}")
            return 0

        # Run each step in order. Track worst exit code.
        worst_rc = 0
        results: list[tuple[str, int]] = []
        for label, cmd in steps:
            rc = _run_step(label, cmd)
            results.append((label, rc))
            if rc >= 2:
                # HALT on ANY hard failure, not just rc==2. rc>2 (e.g. 130 SIGINT,
                # timeouts, kill signals) previously did NOT break, so downstream
                # steps ran on partial/corrupt output.
                phase_log("refresh_full", "FAIL",
                          f"hard failure rc={rc} on step '{label}' -- halting "
                          f"before further steps")
                worst_rc = max(worst_rc, rc)
                break
            if rc == 1 and worst_rc == 0:
                worst_rc = 1

        # Summary.
        phase_log("refresh_full", "OK", "step-by-step results:")
        for label, rc in results:
            status = "OK" if rc == 0 else ("WARN" if rc == 1 else "FAIL")
            phase_log("refresh_full", status, f"  {label}: rc={rc}")

        if worst_rc == 0:
            phase_log("refresh_full", "DONE",
                      "all stages clean — data layer fully refreshed + validated")
        elif worst_rc == 1:
            phase_log("refresh_full", "WARN",
                      "completed with WARNs — review before training")
        else:
            phase_log("refresh_full", "FAIL",
                      f"halted on CRITICAL (rc={worst_rc}) — fix before retry")

        return worst_rc


if __name__ == "__main__":
    sys.exit(main())
