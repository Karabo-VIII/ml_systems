#!/usr/bin/env python3
"""RWYB proof for the N7 PLANNER BREADTH guard (n+-k decomposition + self-critique).

Drives the CRYPTO make_nodes(...) plan closure directly with a CAPTURING MockBrain and asserts the seeded frontier
carries BOTH breadth guards that the upgraded planner contract requires:
  - a FALSIFIER     node  (kind == "verify")  -- the '-k' (is the approach/data/spec sound?)
  - a GENERALIZATION node (kind == "diverge") -- the '+k' (the adjacent / general case)

Two scenarios:
  (1) BRAIN COMPLIES        : MockBrain emits build+verify+diverge -> frontier already broad (critique adds nothing).
  (2) WEAK BRAIN single-path: a brain that returns ONLY a build node -> the graph's plan_critique mechanically ADDS
                              the missing falsifier + generalization (the backstop for a weak swapped local model).

Run: python scripts/autonomy/metaop/_test_planner_breadth.py   (exit 0 = all pass). No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
AUTONOMY = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(AUTONOMY)):
    if p not in sys.path:
        sys.path.insert(0, p)

# import via the package name `metaop` so this exercises the CRYPTO shim (which inherits the canonical engine).
from metaop.graph import make_nodes              # noqa: E402  -- crypto shim (build cwd/workspace/anti-drift injected)
from harness.metaop.brain import MockBrain       # noqa: E402  -- engine brain (upgraded plan + plan_critique roles)


def _plan_node(brain, plan_critique=True):
    """Build the plan closure via the crypto make_nodes and run it on a fresh state; return the seeded frontier."""
    plan, *_ = make_nodes(brain, parallel=1, max_steps=1, judges=1, taper=1, plan_critique=plan_critique)
    state = {"objective": "find a robust setup on SOL", "success_criteria": "verified by evidence",
             "frontier": [], "ledger": [], "budget": 2, "cycle": 0, "status": "running", "parallel": 1,
             "run_id": f"breadth-{int(time.time()*1000)}", "awaiting_approval": []}
    out = plan(state)
    return out.get("frontier", [])


class SinglePathBrain(MockBrain):
    """A WEAK brain that ignores the breadth contract and returns ONLY a build node (no falsifier, no generalization).
    The graph's plan_critique backstop must add the two missing guards."""

    def decide(self, role, payload, persona=""):
        if role == "plan":
            return {"frontier": [{"id": "only", "task": "build: single path", "ev": 0.9, "kind": "build",
                                  "status": "open"}]}
        return super().decide(role, payload, persona)  # plan_critique etc. inherit the deterministic mock


def _kinds(fr):
    return {(n.get("kind") or "").lower() for n in fr if isinstance(n, dict)}


def test_compliant_brain_has_both_guards():
    fr = _plan_node(MockBrain())
    kinds = _kinds(fr)
    assert "verify" in kinds, f"frontier missing FALSIFIER (kind=verify); kinds={kinds}"
    assert "diverge" in kinds, f"frontier missing GENERALIZATION (kind=diverge); kinds={kinds}"
    falsifiers = [n["id"] for n in fr if (n.get("kind") or "").lower() == "verify"]
    generalizations = [n["id"] for n in fr if (n.get("kind") or "").lower() == "diverge"]
    print(f"[PASS] compliant brain: frontier={len(fr)} nodes; FALSIFIER(verify)={falsifiers} "
          f"GENERALIZATION(diverge)={generalizations}")


def test_weak_brain_gets_guards_added_by_critique():
    fr = _plan_node(SinglePathBrain(), plan_critique=True)
    kinds = _kinds(fr)
    assert "verify" in kinds, f"plan_critique did NOT add a FALSIFIER for a single-path brain; kinds={kinds}"
    assert "diverge" in kinds, f"plan_critique did NOT add a GENERALIZATION for a single-path brain; kinds={kinds}"
    added = [n["id"] for n in fr if n.get("id") != "only"]
    print(f"[PASS] weak single-path brain: plan_critique ADDED {added} -> kinds now {sorted(kinds)}")


def test_critique_off_leaves_single_path():
    """Degradability: with plan_critique=False the weak frontier is NOT backfilled (the agnostic path is unchanged)."""
    fr = _plan_node(SinglePathBrain(), plan_critique=False)
    kinds = _kinds(fr)
    assert kinds == {"build"}, f"plan_critique=False should leave the single-path frontier untouched; kinds={kinds}"
    print(f"[PASS] critique OFF: single-path frontier untouched (kinds={sorted(kinds)}) -- fully degradable")


def main():
    tests = [test_compliant_brain_has_both_guards, test_weak_brain_gets_guards_added_by_critique,
             test_critique_off_leaves_single_path]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print()
    if failed:
        print(f"PLANNER BREADTH TEST: {failed}/{len(tests)} FAILED")
        return 1
    print(f"ALL {len(tests)} PLANNER BREADTH TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
