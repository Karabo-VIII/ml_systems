"""Test-gate runner for src/oracle.

Discovers all _test_*.py files under src/oracle/ (recursively), converts
each to a dotted module name, and runs each one as a module via
  python -m oracle._test_*
with PYTHONPATH=<repo_root>/src:<repo_root> and cwd=<repo_root>.

Usage:
  python src/oracle/run_tests.py           # all tests
  python src/oracle/run_tests.py --fast    # skip slow tests
  python src/oracle/run_tests.py --only system  # substring filter

Exit code = number of failures (0 = all green).
No emoji (Windows cp1252).
"""
from __future__ import annotations

__contract__ = {
    "kind": "test-runner",
    "inputs": [
        "src/oracle/**/_test_*.py -- all test modules under the oracle subtree"
    ],
    "outputs": [
        "stdout: [PASS]/[SKIP]/[FAIL] lines per module + SUMMARY line",
        "exit_code: number of failed tests (0 = all green)"
    ],
    "invariants": [
        "each test is run as 'python -m <dotted.module>' with "
        "PYTHONPATH=<repo_root>/src:<repo_root>",
        "cwd is always repo_root so relative data paths inside tests resolve correctly",
        "SLOW_PATTERNS tests are skipped under --fast",
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

# Tests whose dotted name contains any of these substrings are considered SLOW.
# Add patterns here as needed (e.g. for tests that do a full multi-date sweep).
SLOW_PATTERNS = ("_sweep", "_multidate", "_grid_large")

# Per-test timeout in seconds.
TIMEOUT_S = 300

# How many tail lines of subprocess output to show on failure.
TAIL_LINES = 20


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path) -> Path:
    """Walk up from `start` until we find a directory that contains both
    'src' and (a Makefile or pyproject.toml or .git).  Falls back to `start`."""
    p = start.resolve()
    for candidate in [p, *p.parents]:
        if ((candidate / "src").is_dir()
                and (
                    (candidate / ".git").exists()
                    or (candidate / "pyproject.toml").exists()
                    or (candidate / "CLAUDE.md").exists()
                )):
            return candidate
    return start.resolve()


def discover_tests(repo_root: Path) -> list[str]:
    """Return sorted list of dotted module names for every _test_*.py under
    src/oracle/."""
    base = repo_root / "src" / "oracle"
    modules: list[str] = []
    for path in sorted(base.rglob("_test_*.py")):
        # Convert absolute path -> relative to repo_root/src -> dotted name.
        try:
            rel = path.relative_to(repo_root / "src")
        except ValueError:
            # Fallback: relative to repo_root.
            rel = path.relative_to(repo_root)
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
    # PYTHONPATH = repo_root/src + repo_root (matches the project's standard env).
    src_dir = str(repo_root / "src")
    root_dir = str(repo_root)
    existing_pp = os.environ.get("PYTHONPATH", "")
    if existing_pp:
        new_pp = f"{src_dir}{os.pathsep}{root_dir}{os.pathsep}{existing_pp}"
    else:
        new_pp = f"{src_dir}{os.pathsep}{root_dir}"

    env = {**os.environ,
           "PYTHONPATH": new_pp,
           "PYTHONIOENCODING": "utf-8"}
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-m", module],
            cwd=str(repo_root),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
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
        description="Discover and run all _test_*.py modules under src/oracle/."
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

    # Locate repo root relative to this file's location.
    # This file lives at  <repo>/src/oracle/run_tests.py
    # so repo_root is two levels up.
    here = Path(__file__).resolve()
    repo_root = find_repo_root(here)

    all_modules = discover_tests(repo_root)
    if not all_modules:
        print(f"ERROR: no _test_*.py files found under "
              f"{repo_root / 'src' / 'oracle'}")
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

    print(f"Discovered {len(all_modules)} test(s), running {len(selected)}, "
          f"skipping {len(skipped)}.")
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
