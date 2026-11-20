"""RWYB test for the PLANNER verify seam (he_planner_verify_cmd) -- MockBrain, no API, no network.

Feeds a plan whose frontier contains:
  - b_noverify : build node WITHOUT verify_cmd            -> must get verify_retries=2 + verify_missing=True + WARNING
  - b_withverify: build node WITH verify_cmd (no retries) -> must get verify_retries=2, verify_missing NOT set
  - b_preset   : build node that ALREADY sets verify_retries=5 -> retries UNTOUCHED (still 5)
  - v_check    : non-build (kind=='verify') node          -> entirely untouched

Then it runs ONLY the plan/seed step (the planner seam) and asserts the four contract conditions, including that
the one-line WARNING fired for the no-verify_cmd build node and did NOT fire for the others.

Run: .venv/Scripts/python.exe -m scripts.autonomy.metaop._test_planner_verify   (or run this file directly)
"""
from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

# allow running as a bare script (python _test_planner_verify.py) as well as -m
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.autonomy.metaop.brain import MockBrain  # noqa: E402
from scripts.autonomy.metaop.graph import make_nodes, DEFAULT_VERIFY_RETRIES  # noqa: E402


class PlanBrain(MockBrain):
    """MockBrain whose plan returns a FIXED, contract-exercising frontier (build w/o verify_cmd, build w/ verify_cmd,
    build w/ preset retries, and a non-build node)."""

    def decide(self, role, payload, persona=""):
        if role == "plan":
            return {"frontier": [
                {"id": "b_noverify", "task": "build: thing A", "ev": 0.9, "kind": "build", "status": "open"},
                {"id": "b_withverify", "task": "build: thing B", "ev": 0.8, "kind": "build", "status": "open",
                 "verify_cmd": "python -c \"assert 1==1\""},
                {"id": "b_preset", "task": "build: thing C", "ev": 0.7, "kind": "build", "status": "open",
                 "verify_retries": 5},
                {"id": "v_check", "task": "-k falsifier: is the approach sound?", "ev": 0.6, "kind": "verify",
                 "status": "open"},
            ]}
        return super().decide(role, payload, persona)


def _run_plan_seed():
    """Drive ONLY the plan node (the planner seam) with the fixed-frontier brain; capture its stdout (warnings).
    plan_critique=False ISOLATES the verify-seam under test: the N7 breadth backstop (which would ADD a generalization
    node here, since PlanBrain's frontier has no kind==diverge node) is exercised by _test_planner_breadth.py instead,
    so this test asserts ONLY the verify_retries/verify_missing contract on the brain-authored nodes."""
    # make_nodes returns a 6-tuple since N3 (the replanner added a `replan` node); unpack it (only plan is used here).
    plan, _dispatch, _judge, _reflect, _route, _replan = make_nodes(PlanBrain(), parallel=2, max_steps=6, judges=3,
                                                                    taper=3, plan_critique=False)
    state = {"objective": "obj", "success_criteria": "sc", "frontier": [], "ledger": [], "budget": 8,
             "cycle": 0, "status": "running", "parallel": 2, "run_id": "test-planner-verify",
             "awaiting_approval": []}
    buf = io.StringIO()
    with redirect_stdout(buf):
        out = plan(state)
    return out["frontier"], buf.getvalue()


def main():
    fr, stdout = _run_plan_seed()
    by_id = {n["id"]: n for n in fr}
    assert set(by_id) == {"b_noverify", "b_withverify", "b_preset", "v_check"}, by_id.keys()

    # (a) every build node gets verify_retries defaulted to 2 UNLESS already set
    assert by_id["b_noverify"]["verify_retries"] == DEFAULT_VERIFY_RETRIES == 2, by_id["b_noverify"]
    assert by_id["b_withverify"]["verify_retries"] == 2, by_id["b_withverify"]
    assert by_id["b_preset"]["verify_retries"] == 5, "preset verify_retries must be UNTOUCHED"
    print("[PASS] (a) build nodes default verify_retries=2 unless already set (preset 5 untouched)")

    # (b) build nodes WITHOUT a verify_cmd get verify_missing=True AND the warning fired. verify_missing tracks the
    #     ABSENCE of verify_cmd, INDEPENDENT of verify_retries -- so b_preset (preset retries, no verify_cmd) is also
    #     flagged. Only the verify_cmd-bearing build node and the non-build node are exempt.
    assert by_id["b_noverify"].get("verify_missing") is True, by_id["b_noverify"]
    assert by_id["b_preset"].get("verify_missing") is True, "preset-retries node still lacks verify_cmd -> flagged"
    assert "[verify] build node b_noverify has NO external verify_cmd" in stdout, stdout
    assert "[verify] build node b_preset has NO external verify_cmd" in stdout, stdout
    # the warning fires for EXACTLY the two no-verify_cmd build nodes (not the verify_cmd node, not the non-build node)
    assert stdout.count("[verify] build node") == 2, f"warning fired wrong number of times:\n{stdout!r}"
    print("[PASS] (b) no-verify_cmd build nodes -> verify_missing=True + one-line WARNING fired (per gap node)")

    # (c) the with-verify_cmd node is unchanged apart from the defaulted retries: keeps its verify_cmd, NO verify_missing
    assert by_id["b_withverify"]["verify_cmd"] == "python -c \"assert 1==1\"", by_id["b_withverify"]
    assert "verify_missing" not in by_id["b_withverify"], "with-verify_cmd node must NOT be flagged missing"
    assert "b_withverify" not in stdout, "no warning should mention the with-verify_cmd node"
    print("[PASS] (c) with-verify_cmd build node unchanged (verify_cmd kept, not flagged missing)")

    # (d) the non-build node is entirely untouched -- no verify_retries, no verify_missing, never warned
    assert "verify_retries" not in by_id["v_check"], "non-build node must NOT get verify_retries"
    assert "verify_missing" not in by_id["v_check"], "non-build node must NOT get verify_missing"
    assert "v_check" not in stdout, "no warning should mention the non-build node"
    print("[PASS] (d) non-build node untouched")

    print("\nALL PLANNER-VERIFY ASSERTIONS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
