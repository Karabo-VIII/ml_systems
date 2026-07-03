"""NEVER-STALL + NO FALSE VICTORY: a brain whose decide() returns the `_error` sentinel (a backend failure) must
NOT crash the graph AND must NOT be reported as 'solved'. An empty/failed plan terminates honestly as empty_plan.
Pins the fail-closed plan handling + the dispatch/decide never-raise guards (2026-07-03)."""
from __future__ import annotations

from metaop.brain import MockBrain
from metaop.graph import build


class _ErrorBrain(MockBrain):
    """Every structured decision fails to parse -> the honest _error sentinel (as _extract_json emits on bad JSON)."""
    name = "ErrorBrain"

    def decide(self, role, payload, persona=""):
        return {"_error": "simulated backend/decode failure"}


class _RaisingBrain(MockBrain):
    """A misbehaving worker that RAISES -- the dispatch guard must catch it so the loop never wedges."""
    name = "RaisingBrain"

    def work(self, task, persona=""):
        raise RuntimeError("worker exploded")


def _drive(brain, cwd, frontier):
    app = build(brain, parallel=1, judges=1, taper=1, channel="test", cwd=cwd)
    init = {"objective": "x", "success_criteria": "y", "frontier": frontier, "ledger": [], "budget": 2,
            "cycle": 0, "status": "running", "parallel": 1, "run_id": "t-ns", "awaiting_approval": []}
    last = None
    for step in app.stream(init, {"configurable": {"thread_id": "t-ns"}}, stream_mode="values"):
        last = step
    return last


def test_error_plan_never_solved(tmp_path):
    # no pre-seed -> plan() calls decide('plan') -> _error -> empty frontier -> must NOT be 'solved'
    last = _drive(_ErrorBrain(), str(tmp_path), [])
    assert last["status"] != "solved"
    assert last["status"] in ("empty_plan", "failed", "budget_spent", "running")


def test_raising_worker_does_not_crash(tmp_path):
    # a worker that raises is caught in dispatch; the node fails honestly, the loop terminates (no exception escapes)
    seed = {"id": "n1", "task": "build: x", "ev": 1.0, "kind": "build", "status": "open"}
    last = _drive(_RaisingBrain(), str(tmp_path), [seed])
    assert last["status"] != "solved"
    n1 = next((n for n in last["frontier"] if n.get("id") == "n1"), {})
    assert "worker raised" in str(n1.get("result", "")) or n1.get("worker_ok") is False
