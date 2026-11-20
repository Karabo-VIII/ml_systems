#!/usr/bin/env python3
"""META-GRAPH -- the genuine persistent meta-orchestrator (LangGraph), replacing the linear Stop-hook trick.

THE GAP this closes (user, 2026-06-05): a chat instance is LINEAR (plan->act->halt). The "awake loop" cannot be
the LLM -- it must be a PROGRAM. This is that program: a LangGraph StateGraph whose RUNTIME is the awake loop,
calling a (pluggable) Claude "brain" as a stateless decision-function per node. The graph holds state, dispatches
work, judges results, reflects lessons into project-evolution, GENERATES ADJACENT PROBLEMS dynamically, routes,
and CHECKPOINTS (so it survives + resumes) -- none of which a linear instance can do.

The loop:  plan --> dispatch --> judge --> reflect --> route --(open nodes & budget)--> dispatch
                                                              \--(solved | budget spent)--> END

Brain backends (auto-selected; the GRAPH is real regardless of which brain runs):
  1. ANTHROPIC_API_KEY  -> AnthropicBrain (langchain-anthropic / anthropic SDK)   [real Claude]
  2. `claude` on PATH   -> CliBrain (claude -p)                                    [real Claude, no API key]
  3. else               -> MockBrain (deterministic, role-aware)                   [proves the loop, no creds]

Run:  python scripts/autonomy/meta_graph.py --objective "..." --budget 8 [--backend mock|cli|api]
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import operator
import os
import subprocess
import sys
from typing import Annotated, TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------- State (the persistent meta-memory)
class MetaState(TypedDict):
    objective: str
    success_criteria: str
    frontier: list                         # [{id, task, ev, kind, status, result, verdict}]
    ledger: Annotated[list, operator.add]  # append-only lessons / feedback (project evolution)
    budget: int
    cycle: int
    status: str                            # running | solved | budget_spent
    log: Annotated[list, operator.add]     # human-readable trace


# ---------------------------------------------------------------- Brain (pluggable Claude decision-function)
class Brain:
    """role-based stateless decision-function. Subclasses implement .think(role, payload) -> dict."""
    def think(self, role: str, payload: dict) -> dict:
        raise NotImplementedError


class MockBrain(Brain):
    """Deterministic, role-aware. Makes the loop do REAL structural work (seed/work/judge/generate-adjacent) so
    the awake-loop machinery is fully exercised + reproducible WITHOUT credentials. Swap for a real brain to think."""
    def think(self, role: str, payload: dict) -> dict:
        if role == "plan":
            obj = payload["objective"]
            return {"frontier": [
                {"id": "n1", "task": f"primary: {obj}", "ev": 0.9, "kind": "build", "status": "open"},
                {"id": "n2", "task": f"-k falsifier: is the approach for '{obj}' sound?", "ev": 0.8, "kind": "verify", "status": "open"},
            ]}
        if role == "work":
            n = payload["node"]
            return {"result": f"worked node {n['id']} ({n['kind']}): produced evidence for '{n['task'][:40]}'"}
        if role == "judge":
            # deterministic: falsifier nodes 'refute' ~ rarely; build nodes pass
            n = payload["node"]
            ok = not (n["kind"] == "verify" and n["id"].endswith("9"))
            return {"verdict": "pass" if ok else "refuted", "reason": "acceptance evidence present" if ok else "falsifier fired"}
        if role == "reflect":
            cyc = payload["cycle"]
            lesson = f"cycle {cyc}: node done + verified; pattern holds."
            # GENERATE ADJACENT PROBLEMS -- taper so the loop terminates naturally (not just on budget)
            adj = []
            if cyc < payload["taper"]:
                adj = [{"id": f"a{cyc}", "task": f"+k adjacent problem surfaced in cycle {cyc}", "ev": 0.55, "kind": "build", "status": "open"}]
            return {"lesson": lesson, "adjacent": adj}
        return {}


class CliBrain(Brain):
    """Real Claude via the `claude -p` headless CLI (no API key needed -- uses Claude Code auth)."""
    def think(self, role: str, payload: dict) -> dict:
        prompt = _role_prompt(role, payload)
        try:
            r = subprocess.run(["claude", "-p", prompt], capture_output=True, text=True, timeout=900,
                               creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            return _parse_json(r.stdout)
        except Exception as e:
            return {"_error": f"cli brain failed: {e}"}


class AnthropicBrain(Brain):
    """Real Claude via the API (ANTHROPIC_API_KEY)."""
    def __init__(self):
        import anthropic  # lazy
        self.client = anthropic.Anthropic()
        self.model = os.environ.get("META_GRAPH_MODEL", "claude-opus-4-8")

    def think(self, role: str, payload: dict) -> dict:
        prompt = _role_prompt(role, payload)
        msg = self.client.messages.create(model=self.model, max_tokens=1500,
                                           messages=[{"role": "user", "content": prompt}])
        return _parse_json(msg.content[0].text)


def _role_prompt(role: str, payload: dict) -> str:
    return (f"[META-GRAPH node: {role}] Respond ONLY with a JSON object.\n"
            f"ROLE CONTRACT: plan->{{frontier:[{{id,task,ev,kind,status:'open'}}]}}; "
            f"work->{{result:str}}; judge->{{verdict:'pass'|'refuted',reason:str}}; "
            f"reflect->{{lesson:str, adjacent:[{{id,task,ev,kind,status:'open'}}]}}.\n"
            f"PAYLOAD: {json.dumps(payload, default=str)[:4000]}")


def _parse_json(text: str) -> dict:
    import re
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        return json.loads(m.group(0)) if m else {"_error": "no json"}
    except Exception as e:
        return {"_error": f"parse: {e}"}


def make_brain(kind: str) -> Brain:
    if kind == "api" or (kind == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            return AnthropicBrain()
        except Exception:
            pass
    if kind == "cli" or (kind == "auto" and _which("claude")):
        return CliBrain()
    return MockBrain()


def _which(exe):
    from shutil import which
    return which(exe)


# ---------------------------------------------------------------- Nodes (each calls the brain; the GRAPH loops)
def _top_open(frontier):
    op = [n for n in frontier if n.get("status") == "open"]
    return max(op, key=lambda n: n.get("ev", 0)) if op else None


def make_nodes(brain: Brain, taper: int):
    def plan(state: MetaState):
        if state.get("frontier"):
            return {"log": ["plan: frontier already seeded"]}
        out = brain.think("plan", {"objective": state["objective"]})
        return {"frontier": out.get("frontier", []), "status": "running",
                "log": [f"plan: seeded {len(out.get('frontier', []))} nodes"]}

    def dispatch(state: MetaState):
        nxt = _top_open(state["frontier"])
        if nxt is None:
            return {"log": ["dispatch: no open node"]}
        out = brain.think("work", {"node": nxt})
        fr = [dict(n) for n in state["frontier"]]
        for n in fr:
            if n["id"] == nxt["id"]:
                n["status"] = "worked"; n["result"] = out.get("result", "")
        return {"frontier": fr, "log": [f"dispatch: worked {nxt['id']} (EV={nxt['ev']})"]}

    def judge(state: MetaState):
        fr = [dict(n) for n in state["frontier"]]
        worked = [n for n in fr if n.get("status") == "worked"]
        if not worked:
            return {"log": ["judge: nothing to judge"]}
        n = worked[0]
        v = brain.think("judge", {"node": n})
        n["status"] = "done" if v.get("verdict") == "pass" else "refuted"
        n["verdict"] = v.get("verdict")
        return {"frontier": fr, "log": [f"judge: {n['id']} -> {n['status']} ({v.get('reason','')})"]}

    def reflect(state: MetaState):
        cyc = state["cycle"] + 1
        out = brain.think("reflect", {"cycle": cyc, "taper": taper, "ledger_len": len(state.get("ledger", []))})
        fr = state["frontier"] + out.get("adjacent", [])
        ledger = [out.get("lesson", "")] if out.get("lesson") else []
        # solved? (demo acceptance: all originally-seeded nodes done + no open + adjacent tapered)
        open_left = [n for n in fr if n.get("status") == "open"]
        status = "running"
        if not open_left:
            status = "solved"
        return {"frontier": fr, "ledger": ledger, "cycle": cyc, "status": status,
                "log": [f"reflect: +{len(out.get('adjacent', []))} adjacent, lesson logged, cycle={cyc}, open_left={len(open_left)}"]}

    def route(state: MetaState) -> str:
        if state["status"] == "solved":
            return END
        if state["cycle"] >= state["budget"]:
            return "budget"
        if _top_open(state["frontier"]) is not None:
            return "dispatch"
        return END

    return plan, dispatch, judge, reflect, route


def build_graph(brain: Brain, taper: int, checkpointer=None):
    plan, dispatch, judge, reflect, route = make_nodes(brain, taper)
    g = StateGraph(MetaState)
    g.add_node("plan", plan)
    g.add_node("dispatch", dispatch)
    g.add_node("judge", judge)
    g.add_node("reflect", reflect)
    g.add_node("budget", lambda s: {"status": "budget_spent", "log": ["budget spent -> stop"]})
    g.add_edge(START, "plan")
    g.add_edge("plan", "dispatch")
    g.add_edge("dispatch", "judge")
    g.add_edge("judge", "reflect")
    g.add_conditional_edges("reflect", route, {"dispatch": "dispatch", "budget": "budget", END: END})
    g.add_edge("budget", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def _run(app, stream_input, cfg):
    last = None
    for step in app.stream(stream_input, cfg, stream_mode="values"):
        last = step
        if step.get("log"):
            print("  " + step["log"][-1])
    return last


def _report(last, thread):
    from collections import Counter
    print("=== TERMINATED ===")
    print(f"status      : {last['status']}")
    print(f"cycles      : {last['cycle']}")
    fr = last["frontier"]
    print(f"frontier    : {dict(Counter(n.get('status') for n in fr))}  (total {len(fr)} nodes, incl. dynamically-generated)")
    print(f"ledger      : {len(last['ledger'])} lessons (project-evolution memory)")
    print(f"checkpointed: thread '{thread}' (resumable -- the program survives, not a chat instance)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", default="characterize the opportunity surface")
    ap.add_argument("--success", default="all seeded nodes verified + adjacent neighborhood exhausted")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--taper", type=int, default=3, help="cycles for which reflect generates adjacent problems")
    ap.add_argument("--backend", default="auto", choices=["auto", "mock", "cli", "api"])
    ap.add_argument("--thread", default="run1")
    ap.add_argument("--durable", action="store_true", help="use SqliteSaver (durable cross-process checkpoint)")
    ap.add_argument("--resume", action="store_true", help="resume the thread from its durable checkpoint (extends --budget)")
    args = ap.parse_args()

    brain = make_brain(args.backend)
    cfg = {"configurable": {"thread_id": args.thread}}
    init: MetaState = {"objective": args.objective, "success_criteria": args.success,
                       "frontier": [], "ledger": [], "budget": args.budget, "cycle": 0,
                       "status": "running", "log": []}
    print(f"=== META-GRAPH awake loop  brain={type(brain).__name__}  budget={args.budget}  durable={args.durable}  resume={args.resume} ===")
    if args.durable:
        from langgraph.checkpoint.sqlite import SqliteSaver
        db = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runs", "autonomy", "meta_graph.db"))
        os.makedirs(os.path.dirname(db), exist_ok=True)
        with SqliteSaver.from_conn_string(db) as cp:
            app = build_graph(brain, args.taper, cp)
            if args.resume:
                snap = app.get_state(cfg)                     # load the DURABLE checkpoint (survives across processes)
                base = dict(snap.values) if (snap and snap.values) else init
                base["budget"] = args.budget                 # extend the budget
                base["status"] = "running"; base["log"] = []  # un-terminate; continue the work
                print(f"  [resume] loaded durable checkpoint: cycle={base.get('cycle')}, "
                      f"frontier={len(base.get('frontier', []))} nodes, ledger={len(base.get('ledger', []))} lessons -> continuing")
                rcfg = {"configurable": {"thread_id": args.thread + "_resumed"}}
                last = _run(app, base, rcfg)                  # continue from the loaded state (carries frontier + ledger)
                _report(last, rcfg["configurable"]["thread_id"])
            else:
                last = _run(app, init, cfg)
                _report(last, args.thread)
    else:
        app = build_graph(brain, args.taper)
        last = _run(app, init, cfg)
        _report(last, args.thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
