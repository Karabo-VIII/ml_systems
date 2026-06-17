"""Test-gate runner for projects/chess_zero.

Discovers all _test_*.py files under projects/chess_zero/ (recursively),
converts each to a dotted module name, and runs each one as a module via
  python -m <module>
with PYTHONPATH=repo-root and cwd=repo-root.

This avoids the two known pitfalls:
  - Relative imports: running as a bare script breaks from .selfplay import ...
  - Spawn-pool hang: Windows 'spawn' workers re-import the entry module; a
    stdin/heredoc script has no importable __file__ and the workers HANG.
    Running as -m az._test_selfplay_pool gives a real
    __file__ and the workers can re-import cleanly.

Usage:
  python projects/chess_zero/run_tests.py           # all tests
  python projects/chess_zero/run_tests.py --fast    # skip slow (spawn-heavy) tests
  python projects/chess_zero/run_tests.py --only checkpoint  # substring filter

Exit code = number of failures (0 = all green).
No emoji (Windows cp1252).
"""
from __future__ import annotations

__contract__ = {
    "kind": "test-runner",
    "inputs": [
        "projects/chess_zero/**/_test_*.py -- all test modules under the chess_zero subtree"
    ],
    "outputs": [
        "stdout: [PASS]/[SKIP]/[FAIL] lines per module + SUMMARY line",
        "exit_code: number of failed tests (0 = all green)"
    ],
    "invariants": [
        "each test is run as 'python -m <dotted.module>' with PYTHONPATH=repo_root",
        "cwd is always repo_root so relative data paths inside tests resolve correctly",
        "SLOW_PATTERNS tests are skipped under --fast (never hung)",
        "per-test timeout default 300s; killed and counted as FAIL on breach",
        "sys.executable is used so the venv python runs child processes",
    ],
}

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tests whose dotted name contains any of these substrings are considered SLOW
# (they spawn 8-16 worker processes and take ~60-90s each).
SLOW_PATTERNS = ("selfplay_pool", "teacher", "eval_parallel")

# Per-test timeout in seconds. Selfplay pool tests can legitimately take ~90s;
# 300s is a generous ceiling that catches true hangs.
TIMEOUT_S = 300

# How many tail lines of subprocess output to show on failure.
TAIL_LINES = 15


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """The games-engine root: the directory that holds this run_tests.py (and az/, tests/)."""
    start = start.resolve()
    return start.parent if start.is_file() else start


def discover_tests(repo_root: Path) -> list[str]:
    """Return sorted dotted module names for every _test_*.py under the repo
    (tests live in tests/; the virtualenv / VCS / cache dirs are skipped)."""
    SKIP = {".venv", "venv", ".git", "__pycache__", "site-packages", ".pytest_cache"}
    modules: list[str] = []
    for path in sorted(repo_root.rglob("_test_*.py")):
        rel = path.relative_to(repo_root)
        if any(part in SKIP for part in rel.parts):
            continue
        # rel is something like  tests/_test_auto_balance.py  ->  tests._test_auto_balance
        dotted = str(rel).replace(os.sep, "/").replace("/", ".").removesuffix(".py")
        modules.append(dotted)
    return modules


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_module(module: str, repo_root: Path, timeout: int) -> tuple[bool, float, str]:
    """Run `python -m <module>` as a subprocess.

    Returns:
        (passed: bool, elapsed_s: float, output_tail: str)
    """
    env = {**os.environ, "PYTHONPATH": str(repo_root)}
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        elapsed = time.perf_counter() - t0
        passed = result.returncode == 0
        combined = (result.stdout + result.stderr).rstrip()
        tail = "\n".join(combined.splitlines()[-TAIL_LINES:])
        return passed, elapsed, tail
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return False, elapsed, f"[TIMEOUT after {elapsed:.0f}s]"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Discover and run all _test_*.py modules under projects/chess_zero/."
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Skip slow tests (those whose name matches any SLOW_PATTERNS: "
            + ", ".join(SLOW_PATTERNS)
            + "). Fast tests only."
        ),
    )
    p.add_argument(
        "--only",
        metavar="SUBSTR",
        default=None,
        help="Run only modules whose dotted name contains SUBSTR (case-sensitive).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=TIMEOUT_S,
        metavar="S",
        help=f"Per-test timeout in seconds (default {TIMEOUT_S}).",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # The games-engine root is this file's own directory (it holds az/, tests/, ...).
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)

    all_modules = discover_tests(repo_root)
    if not all_modules:
        print(f"ERROR: no _test_*.py files found under {repo_root / 'tests'}")
        return 1

    # Apply filters
    selected: list[str] = []
    skipped: list[str] = []
    for m in all_modules:
        if args.only is not None and args.only not in m:
            skipped.append(m)
            continue
        if args.fast and any(pat in m for pat in SLOW_PATTERNS):
            skipped.append(m)
            continue
        selected.append(m)

    print(f"Discovered {len(all_modules)} test(s), running {len(selected)}, skipping {len(skipped)}.")
    print(f"Repo root : {repo_root}")
    print(f"Python    : {sys.executable}")
    if skipped:
        for s in skipped:
            print(f"  [SKIP] {s}")
    print()

    passed_list: list[str] = []
    failed_list: list[str] = []

    for module in selected:
        print(f"Running   {module} ...", flush=True)
        ok, elapsed, tail = run_module(module, repo_root, args.timeout)
        elapsed_str = f"{elapsed:.1f}s"
        if ok:
            print(f"  [PASS] {module} ({elapsed_str})")
            passed_list.append(module)
        else:
            print(f"  [FAIL] {module} (exit nonzero, {elapsed_str})")
            print("  --- last output ---")
            for line in tail.splitlines():
                print(f"    {line}")
            print("  ---")
            failed_list.append(module)
        print()

    # Summary
    n_pass = len(passed_list)
    n_fail = len(failed_list)
    n_skip = len(skipped)
    print(f"SUMMARY: {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
