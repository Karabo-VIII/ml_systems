"""MECHANICAL OVERRIDE (end-to-end): a verify_cmd exit==0 is ground-truth PASS and OVERRIDES the LLM panel; exit!=0
REFUTES even when the worker reports ok; and a trivial (spoof) verify_cmd can NEVER fake a green. Drives the real
plan/dispatch/judge/reflect/route loop with the credential-free MockBrain."""
from __future__ import annotations

from metaop.brain import MockBrain
from metaop.graph import build

SOLVED = "solved"


def _run(verify_cmd, cwd, budget=1, retries=0):
    app = build(MockBrain(), parallel=1, judges=1, taper=1, channel="test", cwd=cwd)
    seed = {"id": "n1", "task": "build: x", "ev": 1.0, "kind": "build", "status": "open",
            "verify_cmd": verify_cmd, "verify_retries": retries}
    init = {"objective": "x", "success_criteria": "y", "frontier": [seed], "ledger": [], "budget": budget,
            "cycle": 0, "status": "running", "parallel": 1, "run_id": "t-mech", "awaiting_approval": []}
    last = None
    for step in app.stream(init, {"configurable": {"thread_id": "t-mech"}}, stream_mode="values"):
        last = step
    n1 = next((n for n in last["frontier"] if n.get("id") == "n1"), {})
    return last, n1


def test_exit0_passes_and_solves(tmp_path):
    last, n1 = _run('python -c "assert 1+1==2"', str(tmp_path))
    assert n1.get("verdict") == "pass"
    assert n1.get("evidence_type") == "mechanical"
    assert last["status"] == SOLVED


def test_exit_nonzero_refutes_despite_worker_ok(tmp_path):
    last, n1 = _run('python -c "assert 1+1==3"', str(tmp_path))
    assert n1.get("worker_ok") is True          # the mock worker 'succeeded'...
    assert n1.get("verdict") == "refuted"       # ...but the MECHANICAL verifier overrides it
    assert last["status"] != SOLVED


def test_trivial_verify_cannot_fake_pass(tmp_path):
    # a brain-authored spoof (`true && true`) is REJECTED by the screen -> refuted, never a fabricated green
    last, n1 = _run("true && true", str(tmp_path))
    assert n1.get("verdict") == "refuted"
    assert "trivial no-op" in str(n1.get("verify_error", ""))
    assert last["status"] != SOLVED
