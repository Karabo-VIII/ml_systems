#!/usr/bin/env python3
"""Exercise the REAL meta_graph judge->reflect cycle on n1's output (RWYB demo, no commit).

Drives the actual node functions returned by meta_graph.make_nodes() -- NOT a reimplementation --
through: plan (seed n1) -> dispatch (work n1) -> judge (n1's output) -> reflect (lesson + adjacent).
Then asserts the reflect node produced a CONCRETE lesson + >=1 adjacent node. Capability proven by
running the code path, not by asserting it exists.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "autonomy"))

import meta_graph as mg

def main():
    backend = "api" if os.environ.get("ANTHROPIC_API_KEY") else ("cli" if mg._which("claude") else "auto")
    brain = mg.make_brain("auto")
    print(f"brain = {type(brain).__name__}")

    TAPER = 3
    plan, dispatch, judge, reflect, route = mg.make_nodes(brain, taper=TAPER)

    # ---- initial state (same shape main() uses) ----
    state: mg.MetaState = {
        "objective": "characterize the liquidation-cascade opportunity surface",
        "success_criteria": "n1 verified + adjacent neighborhood surfaced",
        "frontier": [], "ledger": [], "budget": 8, "cycle": 0,
        "status": "running", "log": [],
    }

    def apply(delta):
        """Mimic LangGraph reducers: operator.add appends for ledger+log; others replace."""
        for k, v in delta.items():
            if k in ("ledger", "log"):
                state[k] = state.get(k, []) + v
            else:
                state[k] = v

    # ---- 1. PLAN: seed the frontier (must contain n1) ----
    apply(plan(state))
    n1 = next((n for n in state["frontier"] if n["id"] == "n1"), None)
    assert n1 is not None, "plan did not seed n1"
    print(f"\n[plan]    n1 seeded: id={n1['id']} kind={n1['kind']} ev={n1['ev']} status={n1['status']}")
    print(f"          n1.task = {n1['task']!r}")

    # ---- 2. DISPATCH: work n1 -> produces n1's OUTPUT (the thing judge+reflect act on) ----
    apply(dispatch(state))
    n1w = next(n for n in state["frontier"] if n["id"] == "n1")
    assert n1w["status"] == "worked", f"dispatch did not work n1 (status={n1w['status']})"
    print(f"\n[dispatch] n1 worked. n1.OUTPUT (result) =")
    print(f"          {n1w['result']!r}")

    # ---- 3. JUDGE: adjudicate n1's output ----
    apply(judge(state))
    n1j = next(n for n in state["frontier"] if n["id"] == "n1")
    print(f"\n[judge]   n1 -> status={n1j['status']} verdict={n1j.get('verdict')!r}")
    assert n1j["status"] in ("done", "refuted"), "judge did not adjudicate n1"
    assert "verdict" in n1j, "judge attached no verdict to n1"

    # ---- 4. REFLECT: the capability under test -> lesson + adjacent nodes ----
    pre_ledger = len(state["ledger"])
    pre_frontier_ids = {n["id"] for n in state["frontier"]}
    apply(reflect(state))
    post_frontier = state["frontier"]
    new_nodes = [n for n in post_frontier if n["id"] not in pre_frontier_ids]
    lessons_added = state["ledger"][pre_ledger:]

    print(f"\n[reflect] cycle now = {state['cycle']}")
    print(f"          LESSON named   : {lessons_added!r}")
    print(f"          ADJACENT nodes : {json.dumps(new_nodes, default=str)}")

    # ---- VERIFY the reflect capability (the assertions ARE the proof) ----
    checks = []
    concrete_lesson = bool(lessons_added) and all(isinstance(l, str) and len(l.strip()) > 0 for l in lessons_added)
    checks.append(("reflect named a concrete (non-empty) lesson", concrete_lesson))

    names_cycle = any(("cycle" in l.lower()) for l in lessons_added)
    checks.append(("lesson references the cycle (is grounded, not generic boilerplate)", names_cycle))

    has_adjacent = len(new_nodes) >= 1
    checks.append(("reflect surfaced >=1 adjacent node", has_adjacent))

    adj_well_formed = all(
        ("id" in n and "task" in n and n.get("status") == "open" and "ev" in n and "kind" in n)
        for n in new_nodes
    )
    checks.append(("each adjacent node is well-formed (id/task/ev/kind/status=open)", adj_well_formed))

    adj_is_adjacent = all(("adjacent" in n.get("task", "").lower()) for n in new_nodes)
    checks.append(("adjacent nodes are framed as adjacent (+k) problems", adj_is_adjacent))

    cycle_advanced = state["cycle"] == 1
    checks.append(("reflect advanced the cycle counter (mutates meta-state)", cycle_advanced))

    print("\n=== VERIFICATION (judge->reflect cycle on n1) ===")
    all_ok = True
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        all_ok = all_ok and ok

    # The concrete artifacts the task asks us to confirm:
    print("\n--- CONCRETE EVIDENCE ---")
    print(f"  concrete lesson   = {lessons_added[0]!r}" if lessons_added else "  concrete lesson   = <NONE>")
    print(f"  adjacent node id  = {new_nodes[0]['id']!r}" if new_nodes else "  adjacent node id  = <NONE>")
    print(f"  adjacent node task= {new_nodes[0]['task']!r}" if new_nodes else "  adjacent node task= <NONE>")

    print("\nRESULT_OK" if all_ok else "\nRESULT_FAIL")
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
