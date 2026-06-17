#!/usr/bin/env python3
"""RWYB proof for the REPLANNER (N3) -- LangGraph plan-execute "replan" adapted to our frontier model.

The ONE-SHOT planner could never recover from a bad initial plan (the #1 fragility): `route` only ever returned
dispatch/budget/END and `reflect` only APPENDED. This test proves the replanner closes that loop, deterministically
(MockBrain only -- no API, no network):

  CORE (test_fires_and_recovers):
    A MockBrain variant whose initial-plan nodes ALWAYS fail (a verify_cmd that exits non-zero) drives the loop into
    a STALL (>= replan_stall cycles with no node reaching done). We assert:
      (a) a `replan` trace event FIRES, carrying a non-empty reason;
      (b) the frontier was REVISED -- >=1 node PRUNED and/or >=1 NEW-APPROACH node ADDED (not merely appended);
      (c) it does NOT infinite-loop: total replans <= max_replans, and the run terminates.
    Recovery is shown by the NEW-APPROACH node reaching 'done' (the loop escapes the doomed initial plan).

  GUARD (test_max_replans_caps):  a never-recovering plan still terminates with replan_count == max_replans.

  SIGNAL (test_signal_trigger):   a node carrying replan_signal forces a replan even without a stall.

  HEALTHY (test_healthy_no_replan): a plan that succeeds first cycle NEVER replans (backward-compatible default).

Run: python scripts/autonomy/metaop/_test_replanner.py   (exit 0 = all pass). No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# allow running as a bare script AND import the crypto-shim engine (which inherits the canonical replanner).
ROOT = Path(__file__).resolve().parents[3]
AUTONOMY = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(AUTONOMY)):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness.metaop.brain import MockBrain          # noqa: E402  -- engine brain (replan role inherited)
from harness.metaop.graph import build              # noqa: E402  -- agnostic engine build (no host injection)
from harness.metaop.config import trace_dir         # noqa: E402

# A verify_cmd that ALWAYS exits non-zero (it is a REAL non-trivial assertion, so it passes the trust screen yet
# always REFUTES) -- this is how we manufacture a never-succeeding node deterministically.
ALWAYS_FAIL = 'python -c "import sys; sys.exit(7)"'


class StallBrain(MockBrain):
    """Initial plan = two build nodes whose verify_cmd ALWAYS fails -> the loop stalls. `replan` (inherited from
    MockBrain) prunes the refuted nodes + adds a NEW-APPROACH node; that new node has NO verify_cmd, so the LLM-judge
    path passes it -> the loop RECOVERS. judge for the new node is the inherited MockBrain judge (result present -> pass)."""

    def decide(self, role, payload, persona=""):
        if role == "plan":
            return {"frontier": [
                {"id": "bad1", "task": "build: doomed approach A", "ev": 0.9, "kind": "build", "status": "open",
                 "verify_cmd": ALWAYS_FAIL, "verify_retries": 1},
                {"id": "bad2", "task": "build: doomed approach B", "ev": 0.8, "kind": "build", "status": "open",
                 "verify_cmd": ALWAYS_FAIL, "verify_retries": 1},
            ]}
        return super().decide(role, payload, persona)


class NeverRecoverBrain(MockBrain):
    """Every plan/replan node fails (verify_cmd always non-zero) -> proves the max_replans guard terminates the run."""

    def decide(self, role, payload, persona=""):
        if role == "plan":
            return {"frontier": [
                {"id": "x1", "task": "build: doomed", "ev": 0.9, "kind": "build", "status": "open",
                 "verify_cmd": ALWAYS_FAIL, "verify_retries": 0},
            ]}
        if role == "replan":
            # always add ANOTHER doomed node (a real revision: new id) -- never recovers, must hit the cap + stop.
            cur = payload.get("current_frontier", []) or []
            i = sum(1 for n in cur) + int(time.time() * 1000) % 7
            return {"frontier": [
                {"id": f"doomed_{i}_{len(cur)}", "task": "build: still doomed", "ev": 0.5, "kind": "build",
                 "status": "open", "verify_cmd": ALWAYS_FAIL, "verify_retries": 0},
            ]}
        return super().decide(role, payload, persona)


class SignalBrain(MockBrain):
    """Plan node carries replan_signal -> route must replan on the FIRST cycle (no stall needed). The replan adds a
    clean node that passes -> recovers immediately."""

    def decide(self, role, payload, persona=""):
        if role == "plan":
            return {"frontier": [
                {"id": "sig1", "task": "build: approach known to be wrong", "ev": 0.9, "kind": "build",
                 "status": "open", "replan_signal": True},
            ]}
        if role == "judge":
            # sig1 would otherwise be judged PASS by MockBrain (it has a result); but the SIGNAL must trip a replan
            # BEFORE judge can matter. Keep inherited behavior for everything else.
            return super().decide(role, payload, persona)
        return super().decide(role, payload, persona)


def _init(objective, budget, run_id):
    return {"objective": objective, "success_criteria": "verified by evidence", "frontier": [], "ledger": [],
            "budget": budget, "cycle": 0, "status": "running", "parallel": 1, "run_id": run_id,
            "awaiting_approval": []}


def _drive(brain, objective, budget, run_id, taper=99, **build_kw):
    """Run the agnostic engine to completion; return (last_state, trace_events list). taper controls adjacent-node
    generation in reflect (taper=99 = keep generating; low taper = stop padding so a healthy run can solve cleanly)."""
    app = build(brain, parallel=1, judges=1, taper=taper, channel="replantest", **build_kw)
    cfg = {"configurable": {"thread_id": run_id}}
    last = None
    for step in app.stream(_init(objective, budget, run_id), cfg, stream_mode="values"):
        last = step
    tr = trace_dir(None) / f"{run_id}.jsonl"
    events = [json.loads(ln) for ln in tr.read_text(encoding="utf-8").strip().splitlines()] if tr.exists() else []
    return last, events


def _replan_events(events):
    return [e for e in events if e.get("event") == "replan"]


def test_healthy_no_replan():
    """A plan that succeeds first cycle must NEVER replan (backward-compatible default)."""
    rid = f"rp-healthy-{int(time.time()*1000)}"
    last, events = _drive(MockBrain(), "healthy objective", budget=3, run_id=rid, taper=1)
    assert last["status"] == "solved", f"healthy run should solve, got {last['status']}"
    assert not _replan_events(events), f"healthy run must not replan; got {_replan_events(events)}"
    assert all(e.get("to") != "replan" for e in events if e.get("event") == "route"), "no route->replan on healthy"
    print("[PASS] healthy run: solved, ZERO replan events (backward-compatible)")


def test_fires_and_recovers():
    """CORE: stall -> replan fires with a reason -> frontier REVISED (prune and/or add) -> recovers, no infinite loop."""
    rid = f"rp-core-{int(time.time()*1000)}"
    last, events = _drive(StallBrain(), "find a working approach", budget=12, run_id=rid,
                          replan_stall=2, max_replans=3)
    rps = _replan_events(events)
    # (a) a replan event fired, with a non-empty reason
    assert rps, "REPLAN never fired despite a stalling plan -- the #1 fragility is NOT closed"
    assert all(e.get("reason") for e in rps), f"replan event(s) missing a reason: {rps}"
    print(f"[PASS] (a) replan FIRED {len(rps)}x with reasons; first reason: {rps[0]['reason'][:80]!r}")

    # (b) the frontier was REVISED -- not a mere append: at least one PRUNE or one ADD across the replan events.
    total_pruned = sum(len(e.get("pruned", [])) for e in rps)
    total_added = sum(len(e.get("added", [])) for e in rps)
    assert total_pruned >= 1 or total_added >= 1, (
        f"frontier not revised (no prune, no add): pruned={total_pruned} added={total_added}")
    assert total_added >= 1, f"replan must ADD a new-approach node (got added={total_added})"
    print(f"[PASS] (b) frontier REVISED: pruned={total_pruned} added={total_added} (a NEW-APPROACH node, not append)")

    # recovery: a brand-new-approach node reached 'done' (the loop escaped the doomed initial plan)
    fr = last["frontier"]
    new_done = [n for n in fr if n.get("id", "").startswith("replan_alt") and n.get("status") == "done"]
    assert new_done, f"no new-approach node reached done -> did not RECOVER. frontier={[ (n['id'],n.get('status')) for n in fr]}"
    print(f"[PASS] recovery: new-approach node(s) {[n['id'] for n in new_done]} reached 'done'")

    # (c) did NOT infinite-loop: total replans <= max_replans, and the run terminated.
    replan_done = [e for e in events if e.get("event") == "replan_done"]
    max_count = max((e.get("replan_count", 0) for e in replan_done), default=0)
    assert max_count <= 3, f"replan_count {max_count} exceeded max_replans=3 (infinite-loop guard FAILED)"
    assert last["status"] in ("solved", "budget_spent", "running"), f"unexpected terminal status {last['status']}"
    print(f"[PASS] (c) no infinite loop: max replan_count={max_count} <= max_replans=3; terminal={last['status']}")


def test_max_replans_caps():
    """A never-recovering plan must still TERMINATE: replans hit the cap, then the loop proceeds/ends (no infinite loop)."""
    rid = f"rp-cap-{int(time.time()*1000)}"
    last, events = _drive(NeverRecoverBrain(), "unsolvable objective", budget=20, run_id=rid,
                          replan_stall=1, max_replans=2)
    replan_done = [e for e in events if e.get("event") == "replan_done"]
    max_count = max((e.get("replan_count", 0) for e in replan_done), default=0)
    assert max_count == 2, f"expected exactly max_replans=2 replans, got {max_count}"
    # after the cap, route must announce 'proceed' (guard exhausted) at least once, never replanning again.
    proceeded = [e for e in events if e.get("event") == "route" and e.get("to") == "proceed"]
    assert proceeded, "after max_replans the router must PROCEED (not replan again)"
    assert last["status"] in ("budget_spent", "solved", "running"), f"did not terminate cleanly: {last['status']}"
    print(f"[PASS] max_replans guard: stopped at replan_count={max_count}; router PROCEEDED after cap; "
          f"terminal={last['status']}")


def test_signal_trigger():
    """SIGNAL trigger: a node with replan_signal forces a replan even without a stall (first cycle)."""
    rid = f"rp-signal-{int(time.time()*1000)}"
    last, events = _drive(SignalBrain(), "signalled objective", budget=8, run_id=rid,
                          replan_stall=99, max_replans=3)  # stall disabled -> ONLY the signal can trigger
    rps = _replan_events(events)
    assert rps, "replan_signal did not trigger a replan"
    assert "signal" in rps[0].get("reason", ""), f"first replan reason should cite the signal: {rps[0]}"
    print(f"[PASS] signal trigger: replan fired on signal (reason: {rps[0]['reason'][:70]!r})")


def main():
    tests = [test_healthy_no_replan, test_fires_and_recovers, test_max_replans_caps, test_signal_trigger]
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
        print(f"REPLANNER TEST: {failed}/{len(tests)} FAILED")
        return 1
    print(f"ALL {len(tests)} REPLANNER TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
