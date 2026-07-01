"""Test runner for the forex pipeline. Discovers tests/_test_*.py, runs every
`test_*` function, reports pass/fail. No pytest dependency.

    python run_tests.py
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))          # so `import pipeline...` works


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    test_files = sorted((ROOT / "tests").glob("_test_*.py"))
    if not test_files:
        print("no test files found under tests/")
        return 1
    n_pass = n_fail = 0
    for tf in test_files:
        mod = _load(tf)
        fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
        for fn in fns:
            try:
                fn()
                n_pass += 1
                print(f"  PASS {tf.name}::{fn.__name__}")
            except Exception as e:
                n_fail += 1
                print(f"  FAIL {tf.name}::{fn.__name__}: {type(e).__name__}: {e}")
                traceback.print_exc()
    print(f"\n[run_tests] {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
