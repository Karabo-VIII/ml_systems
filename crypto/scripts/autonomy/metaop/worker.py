"""Operator WORKER -- crypto-consumer SHIM over the canonical harness.metaop.worker (G-A dedup 2026-06-07).

The tool-using ReAct Worker (brain.act -> tools.call -> observe -> repeat -> final) now lives ONCE in
harness/metaop/worker.py. This shim re-exports it and only swaps the DEFAULT tool surface to the crypto-fenced
Tools (this package's tools.py: .claude control-surface fences + runs/autonomy/permission_policy.json), so a Worker
constructed with no explicit tools still runs under the crypto safety fence. The graph itself calls brain.work()
directly; this Worker is the canonical reference loop + handy for testing a Brain's act() in isolation. No emoji.
"""
from __future__ import annotations

from harness.metaop.worker import Worker as _HarnessWorker  # canonical ReAct loop
from .tools import Tools  # crypto-fenced tools (control-surface deny + permission_policy.json)


class Worker(_HarnessWorker):
    """Canonical harness Worker with the crypto-fenced Tools as the default tool surface."""

    def __init__(self, brain, tools: Tools | None = None, max_steps: int = 6):
        super().__init__(brain, tools or Tools(), max_steps)


if __name__ == "__main__":
    from .brain import make_brain
    w = Worker(make_brain())
    out = w.run("inspect the repository HEAD and report it")
    print("ok    :", out["ok"], "| steps:", out["steps"])
    print("result:", out["result"])
