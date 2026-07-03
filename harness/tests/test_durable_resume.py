"""DURABLE RESUME IN PLACE: a resumed run must continue on the SAME thread_id (so status/approve/stop still address
it) and must NOT duplicate the operator.add reducer channels (ledger / awaiting_approval). Pins the 2026-07-03
manager resume fix (the old code streamed the continuation into thread+'-r', orphaning the run and re-reading stale
state on a second resume). Skips cleanly if the sqlite checkpointer extra isn't installed."""
from __future__ import annotations

import pytest

from metaop.brain import MockBrain
from metaop.graph import build


def test_resume_same_thread_no_reducer_double(tmp_path, monkeypatch):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    monkeypatch.setenv("HARNESS_WORKSPACE", str(tmp_path / "ws"))
    from langgraph.checkpoint.sqlite import SqliteSaver

    from metaop.config import checkpoint_db
    db = checkpoint_db(None, "tt")
    db.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"configurable": {"thread_id": "tt"}}
    with SqliteSaver.from_conn_string(str(db)) as cp:
        app = build(MockBrain(), parallel=1, judges=1, taper=1, channel="test", checkpointer=cp,
                    cwd=str(tmp_path / "build"))
        seed = {"id": "n1", "task": "build: x", "ev": 1.0, "kind": "build", "status": "open"}
        init = {"objective": "x", "success_criteria": "y", "frontier": [seed], "ledger": [], "budget": 2,
                "cycle": 0, "status": "running", "parallel": 1, "run_id": "tt-1", "awaiting_approval": []}
        for _ in app.stream(init, cfg, stream_mode="values"):
            pass
        s1 = app.get_state(cfg).values
        led1, cyc1 = len(s1["ledger"]), s1["cycle"]

        # RESUME IN PLACE with the manager's MINIMAL input (only mutated scalars) on the SAME thread.
        for _ in app.stream({"budget": 4, "status": "running"}, cfg, stream_mode="values"):
            pass
        s2 = app.get_state(cfg).values

    assert s2["cycle"] > cyc1, "resume must continue on the same thread (cycle advanced)"
    added_cycles = s2["cycle"] - cyc1
    # ledger grew by at most ~one lesson per new cycle -- NOT doubled (a reducer-double would re-add all led1 lessons)
    assert len(s2["ledger"]) <= led1 + added_cycles + 1, "ledger must not be duplicated on resume"
    assert len(s2["ledger"]) >= led1, "resume must not lose prior lessons"
