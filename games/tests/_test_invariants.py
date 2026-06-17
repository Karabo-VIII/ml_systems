"""
Test-suite wrapper for the pre-training invariant gate (run_invariants_check.py / audit I13).

Makes the catastrophic-correctness gate (value-sign backup, z-perspective, policy<->index
bijection, terminal values, target integrity, never-hang, NaN-guard) part of the CI test gate
(`run_tests.py`), so the gate itself can't silently rot between restarts. Fast (<60s), CPU.

Run:  .venv/Scripts/python.exe -m az._test_invariants
Exit: 0 if all invariants hold; nonzero otherwise.
"""
from __future__ import annotations

from run_invariants_check import main as _gate


def main() -> int:
    rc = _gate()
    assert rc == 0, f"invariant gate returned {rc} (a catastrophic invariant is broken)"
    print("[ok] pre-training invariant gate holds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
