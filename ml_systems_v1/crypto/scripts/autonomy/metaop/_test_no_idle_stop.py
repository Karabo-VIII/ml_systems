"""RWYB test for the loop-level NO-IDLE-STOP fix (drain-replan gate) in harness/metaop/graph.py.

THE BUG (mechanical twin of an overseer stopping with the window unspent): route() used to return END the instant the
frontier drained (no open nodes), even with budget/window remaining -- so a loop that completed its explicit plan quit
early instead of generating the next adjacent work. THE FIX: when the frontier drains but the budget is NOT spent and
the objective is NOT solved, route() goes to `replan` to REPLENISH; only after DEFAULT_DRAIN_REPLAN_EMPTY_CAP
consecutive drain-replans that add nothing does it END honestly.

Proves:
  UNIT (route, called directly):
    A. drain + budget remains + drain_empty<cap  -> 'replan'   (THE FIX: no idle-stop)
    B. drain + drain_empty == cap                -> END        (honest stop: frontier genuinely exhausted)
    C. cycle >= budget                           -> 'budget'   (window spent -> stop)
    D. status solved                             -> END        (objective met -> stop)
    E. open nodes present                        -> 'dispatch' (unchanged)
    F. stall reason + open nodes                 -> 'replan'   (recovery replan path intact)
  UNIT (replan node): a drain-replan that ADDS work resets drain_empty to 0; one that adds nothing increments it.
  E2E (compiled graph, controllable brain): a run whose plan completes with budget remaining REPLENISHES adjacent
    work (>1 node done) and then ENDs honestly once the brain has no more -- it does NOT stop after the first node.

Run from repo ROOT:  .venv/Scripts/python.exe scripts/autonomy/metaop/_test_no_idle_stop.py
No emoji (Windows cp1252).
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.graph import END  # noqa: E402
from harness.metaop.graph import make_nodes, build, DEFAULT_DRAIN_REPLAN_EMPTY_CAP  # noqa: E402
from harness.metaop.brain import MockBrain, Brain  # noqa: E402
import scripts.autonomy.metaop.graph as _shim  # noqa: E402  (copy-parity: shim must re-export the engine)

PASS_CMD = f'"{sys.executable}" -c "assert 1==1"'   # non-trivial, exits 0 (passes the verify screen)
fails = []
def ok(c, label, detail=""):
    print(("  PASS" if c else "  FAIL"), label, ("" if c else f":: {detail}"))
    if not c: fails.append(label)


def _node(nid, status="open"):
    return {"id": nid, "status": status, "kind": "build", "task": f"build {nid}", "ev": 0.9, "verify_cmd": PASS_CMD}


print("=" * 84); print("UNIT -- route() drain gate (THE idle-stop fix), fill_window ON vs OFF"); print("=" * 84)
_, _, _, _, route_on, _ = make_nodes(MockBrain(), parallel=1, max_steps=6, judges=3, taper=3, fill_window=True)
_, _, _, _, route_off, _ = make_nodes(MockBrain(), parallel=1, max_steps=6, judges=3, taper=3, fill_window=False)

def st(**kw):
    base = {"status": "running", "cycle": 0, "budget": 50, "frontier": [], "run_id": "t", "ledger": [],
            "drain_empty": 0, "stall_cycles": 0, "replan_count": 0}
    base.update(kw); return base

ok(route_on(st(frontier=[_node("a", "done")])) == "replan",
   "A. fill_window ON: drain + budget remains -> REPLAN (no idle-stop)")
ok(route_on(st(frontier=[_node("a", "done")], drain_empty=DEFAULT_DRAIN_REPLAN_EMPTY_CAP)) == END,
   f"B. ON: drain + drain_empty=={DEFAULT_DRAIN_REPLAN_EMPTY_CAP} (cap) -> END (honest stop)")
ok(route_on(st(frontier=[_node("a", "done")], cycle=50)) == "budget",
   "C. ON: cycle>=budget -> 'budget' (window spent)")
ok(route_on(st(status="solved", frontier=[_node("a")])) == END,
   "D. ON: solved -> END")
ok(route_on(st(frontier=[_node("a", "open")])) == "dispatch",
   "E. ON: open nodes -> dispatch (unchanged)")
ok(route_on(st(frontier=[_node("a", "open")], stall_cycles=5)) == "replan",
   "F. ON: stall + open nodes -> replan (recovery path intact)")
ok(route_off(st(frontier=[_node("a", "done")])) == END,
   "G. fill_window OFF (default): drain -> END (prior behaviour preserved -> no regression)")

print("=" * 84); print("UNIT -- replan node drain_empty tracking"); print("=" * 84)

class _ReplanBrain(MockBrain):
    """decide('replan') returns whatever frontier we set in .next_replan (controls add-vs-empty)."""
    next_replan = []
    def decide(self, role, payload, persona=""):
        if role == "replan":
            return {"frontier": list(self.next_replan)}
        return super().decide(role, payload, persona)

rb = _ReplanBrain()
p2, d2, j2, r2, route2, replan2 = make_nodes(rb, parallel=1, max_steps=6, judges=3, taper=3)
drained = st(frontier=[_node("done1", "done")], drain_empty=0)
# (i) drain-replan that ADDS a new open node -> drain_empty resets to 0, frontier gains open work
rb.next_replan = [_node("fresh")]
out_add = replan2(drained)
ok(out_add.get("drain_empty") == 0 and len([n for n in out_add["frontier"] if n.get("status") == "open"]) == 1,
   "drain-replan that ADDS work -> drain_empty=0 + 1 open node", out_add.get("drain_empty"))
# (ii) drain-replan that adds NOTHING -> drain_empty increments
rb.next_replan = []
out_empty = replan2(st(frontier=[_node("done1", "done")], drain_empty=1))
ok(out_empty.get("drain_empty") == 2 and not [n for n in out_empty["frontier"] if n.get("status") == "open"],
   "drain-replan that adds NOTHING -> drain_empty 1->2 (toward honest stop)", out_empty.get("drain_empty"))

print("=" * 84); print("E2E -- compiled graph REPLENISHES until exhausted (does not idle-stop after node 1)"); print("=" * 84)

class _E2EBrain(Brain):
    """plan -> one node; reflect -> no adjacent (force drain); replan -> add a node for the first 2 drain-replans,
    then empty (genuinely exhausted). work -> ok artifact. Nodes carry a passing verify_cmd so the mechanical judge
    auto-passes them (no judge-vote needed)."""
    name = "E2EBrain"
    def __init__(self): super().__init__(); self._n = 0; self._replans = 0
    def decide(self, role, payload, persona=""):
        if role == "plan":
            self._n += 1; return {"frontier": [_node(f"n{self._n}")]}
        if role == "replan":
            self._replans += 1
            if self._replans <= 2:           # propose adjacent work twice, then nothing
                self._n += 1; return {"frontier": [_node(f"n{self._n}")]}
            return {"frontier": []}
        if role == "reflect":
            return {"lesson": "", "adjacent": []}   # NO adjacent -> the frontier drains -> exercises the gate
        return {}
    def act(self, task, tools_schema, history): return {"final": "done"}
    def work(self, task, persona=""): return {"ok": True, "result": "ARTIFACT built + verified by running"}

def _init(tid):
    return {"objective": "demo no-idle-stop", "success_criteria": "complete all nodes", "frontier": [],
            "budget": 50, "cycle": 0, "status": "running", "parallel": 1, "run_id": tid,
            "ledger": [], "awaiting_approval": [], "drain_empty": 0, "stall_cycles": 0, "replan_count": 0,
            "done_count": 0}

# ON: replenishes adjacent work until exhausted, then honest stop
app_on = build(_E2EBrain(), parallel=1, max_steps=6, judges=3, taper=3, fill_window=True)
final = app_on.invoke(_init("e2e-on"), {"recursion_limit": 200, "configurable": {"thread_id": "e2e-on"}})
done_nodes = [n for n in final.get("frontier", []) if n.get("status") == "done"]
ok(len(done_nodes) >= 3,
   f"ON: loop REPLENISHED adjacent work: {len(done_nodes)} nodes done (NOT 1 -> no idle-stop after the first)",
   [n.get("id") for n in done_nodes])
ok(final.get("drain_empty", 0) >= DEFAULT_DRAIN_REPLAN_EMPTY_CAP,
   "ON: loop ENDed HONESTLY only after the brain had no more adjacent work (drain_empty hit the cap)",
   final.get("drain_empty"))

# OFF (default): completes its plan and stops -- NO regression, NO window-filling
app_off = build(_E2EBrain(), parallel=1, max_steps=6, judges=3, taper=3, fill_window=False)
final_off = app_off.invoke(_init("e2e-off"), {"recursion_limit": 200, "configurable": {"thread_id": "e2e-off"}})
done_off = [n for n in final_off.get("frontier", []) if n.get("status") == "done"]
ok(len(done_off) == 1 and final_off.get("status") == "solved",
   "OFF: loop completes its plan (1 node) and stops as before -- default behaviour unchanged",
   {"done": len(done_off), "status": final_off.get("status")})

print("=" * 84); print("COPY-PARITY -- the shim re-exports the engine the fix lives in"); print("=" * 84)
ok(hasattr(_shim, "build") and hasattr(_shim, "make_nodes"),
   "scripts.autonomy.metaop.graph re-exports build + make_nodes (both copies get the fix)")

print("=" * 84)
print(f"NO-IDLE-STOP TEST: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
print("=" * 84)
sys.exit(0 if not fails else 1)
