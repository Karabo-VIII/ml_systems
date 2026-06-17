"""Harness WORKER -- a tool-using ReAct agent that EXECUTES one frontier node.

The graph's `dispatch` node hands a task here; the worker runs the genuine agent loop: brain.act -> pick a tool ->
tools.call -> observe -> repeat -> final. This is where real work happens (running analyses, reading/writing
artifacts), fenced by tools.py's safety screen. Returns {ok, result, steps, transcript}. No emoji (cp1252).

NOTE: this ReAct Worker is used by Brains that implement act() (MockBrain/AnthropicBrain). The SDK/CLI brains do
the whole task inside their own work() instead, so the graph calls brain.work() directly -- this class remains the
canonical reference loop + is handy for testing a Brain's act() in isolation.
"""
from __future__ import annotations

from .tools import Tools


class Worker:
    def __init__(self, brain, tools: Tools | None = None, max_steps: int = 12):
        self.brain = brain
        self.tools = tools or Tools()
        self.max_steps = max_steps

    def run(self, task: str) -> dict:
        # Reference ReAct loop (the raw-LLM brains use the equivalent _run_react_work in brain.py). Robust to a weak /
        # local model's long self-debug loop: the LAST step forbids tools + FORCES a final, and a built artifact (last
        # successful tool output) is reported instead of a bare "hit max steps".
        history: list = []
        last_obs = ""
        n = max(1, self.max_steps)
        for step in range(1, n + 1):
            t = task if step < n else (task + "\n\nFINAL STEP: do NOT call any more tools. Reply with "
                                       '{"action":"final","result":"<one-line summary>"} now.')
            decision = self.brain.act(t, self.tools.schema(), history)
            history.append({"role": "assistant", "content": decision})
            if decision.get("_error"):
                return {"ok": False, "result": f"brain error: {decision['_error']}", "steps": step, "transcript": history}
            if decision.get("action") == "final":
                return {"ok": True, "result": decision.get("result", "") or last_obs, "steps": step, "transcript": history}
            tool, args = decision.get("tool"), decision.get("args", {}) or {}
            if not tool:
                return {"ok": False, "result": "no tool/final in decision", "steps": step, "transcript": history}
            obs = self.tools.call(tool, args)
            history.append({"role": "tool", "content": obs})
            if obs.get("ok"):
                last_obs = str(obs.get("output", ""))[:500]
        return {"ok": bool(last_obs), "result": last_obs or f"hit max_steps={self.max_steps} without final",
                "steps": self.max_steps, "transcript": history}


if __name__ == "__main__":
    from .brain import make_brain
    w = Worker(make_brain("mock"))
    out = w.run("inspect the python version and report it")
    print("ok    :", out["ok"], "| steps:", out["steps"])
    print("result:", out["result"])
