"""RWYB regression test for H3 (2026-06-09): EVIDENCE-TYPED judge verdicts + unverified-pass auto-downgrade.

Exercises the REAL judge closure from graph.make_nodes (MockBrain -- no API/network) to prove:
  1. mechanical verify_cmd exit 0      -> verdict=pass,        evidence_type="mechanical".
  2. mechanical verify_cmd exit 1      -> verdict=refuted,     evidence_type="mechanical_refuted".
  3. build node, NO verify_cmd, panel-pass -> AUTO-DOWNGRADE to verdict="inconclusive",
     evidence_type="llm_panel_unverified", unverified_pass=True, status="done" (still TERMINAL -> no stall).
  4. verify node, NO verify_cmd, UNANIMOUS panel-pass -> stays verdict="pass" (stronger evidence),
     evidence_type="llm_panel" (NOT downgraded).
  5. non-verifiable kind (diverge), panel-pass -> stays verdict="pass" (nothing to verify), evidence_type="llm_panel".

WHY: before H3 an LLM-panel "pass" with no mechanical check was reported IDENTICALLY to a mechanically
verified pass (both status=done) -> the loop could declare "solved" on an LLM-BELIEVED artifact. H3 tags the
evidence and downgrades the weakest case so the overseer / skill-harvest (H4) / completeness critic re-checks.

Run from repo ROOT:
  .venv/Scripts/python.exe scripts/autonomy/metaop/_test_evidence_type.py
No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "autonomy"))

from scripts.autonomy.metaop.graph import make_nodes  # noqa: E402
from scripts.autonomy.metaop.brain import MockBrain    # noqa: E402
import skill_library                                   # noqa: E402

# H4 is wired into the shim judge: a mechanical-pass BUILD node would auto-harvest into the REAL registry.
# Redirect to a throwaway index so this test never pollutes runs/autonomy/skill_library/INDEX.json.
skill_library.INDEX_PATH = Path(tempfile.mkdtemp(prefix="h3_skill_")) / "INDEX.json"

PY = f'"{sys.executable}"'
PASS_CMD = f'{PY} -c "print(42)"'                 # exit 0
FAIL_CMD = f'{PY} -c "import sys;sys.exit(1)"'    # exit 1


def _state(frontier):
    return {"objective": "test evidence typing", "success_criteria": "verdicts carry evidence_type",
            "frontier": frontier, "ledger": [], "budget": 8, "cycle": 0, "status": "running",
            "parallel": 4, "run_id": "test-evidence-0", "awaiting_approval": []}


def main() -> int:
    brain = MockBrain()
    # judges=3 so a verify-kind node gets a 3-vote panel (MockBrain passes a node that has a 'result' -> unanimous).
    plan, dispatch, judge, reflect, route, _replan = make_nodes(brain, parallel=4, max_steps=6, judges=3, taper=3)
    fails = []

    nodes = [
        {"id": "mech_ok", "task": "x", "ev": 0.9, "kind": "build", "status": "worked",
         "result": "ok", "verify_cmd": PASS_CMD},
        {"id": "mech_no", "task": "x", "ev": 0.9, "kind": "build", "status": "worked",
         "result": "ok", "verify_cmd": FAIL_CMD},                                       # no retry budget -> terminal
        {"id": "build_unver", "task": "x", "ev": 0.5, "kind": "build", "status": "worked", "result": "claim"},
        {"id": "verify_unan", "task": "x", "ev": 0.5, "kind": "verify", "status": "worked", "result": "claim"},
        {"id": "diverge_ok", "task": "x", "ev": 0.5, "kind": "diverge", "status": "worked", "result": "idea"},
    ]
    fr = {n["id"]: n for n in judge(_state(nodes))["frontier"]}

    def check(nid, **expect):
        n = fr[nid]
        for k, v in expect.items():
            if n.get(k) != v:
                fails.append(f"{nid}: expected {k}={v!r}, got {n.get(k)!r}  (full: { {kk: n.get(kk) for kk in expect} })")

    # 1. mechanical PASS
    check("mech_ok", verdict="pass", status="done", evidence_type="mechanical")
    # 2. mechanical REFUTE (terminal, no budget)
    check("mech_no", verdict="refuted", status="refuted", evidence_type="mechanical_refuted")
    # 3. unverified build pass -> downgraded
    check("build_unver", verdict="inconclusive", status="done",
          evidence_type="llm_panel_unverified", unverified_pass=True)
    # 4. unanimous verify panel -> stays pass, NOT downgraded
    check("verify_unan", verdict="pass", status="done", evidence_type="llm_panel")
    # 5. non-verifiable kind -> stays pass
    check("diverge_ok", verdict="pass", status="done", evidence_type="llm_panel")

    print("-" * 70)
    if fails:
        print("RESULT: FAILED")
        for f in fails:
            print("  -", f)
        return 1
    print("RESULT: ALL PASS -- mechanical(pass/refute) tagged; unverified build-pass DOWNGRADED to inconclusive; "
          "unanimous-verify + non-verifiable kinds keep pass; all terminal (no stall).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
