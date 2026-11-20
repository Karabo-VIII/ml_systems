"""Operator MEMORY -- crypto-consumer SHIM over the canonical harness.metaop.memory (N5, mirrors learnings.py).

The richer LOCAL Mem0 backend (ollama embedder + on-disk qdrant) BEHIND the learnings interface, with TF-IDF as
the guaranteed fallback, lives ONCE in harness/metaop/memory.py (project-agnostic; store under a configurable
WORKSPACE). This shim re-exports that engine UNCHANGED and only injects the crypto WORKSPACE so the Mem0 store +
the learnings JSONL keep landing at the historical crypto path runs/autonomy (harness mem0_dir(workspace) =
workspace/mem0, so workspace = <repo>/runs/autonomy -> runs/autonomy/mem0/...).

remember WRITES (learnings JSONL + Mem0 vector store); recall READS (semantic fused with TF-IDF). The crypto loop
calls remember(...)/recall(...)/backend()/available() with NO workspace arg, so each wrapper defaults the workspace
to the crypto path. Best-effort throughout: ANY Mem0 error degrades to TF-IDF -- never breaks the loop. No emoji.
"""
from __future__ import annotations

from pathlib import Path

from harness.metaop import memory as _h  # the canonical engine
# re-export everything (so `from metaop.memory import *` and attribute access see the full surface)
from harness.metaop.memory import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[3]
# harness mem0_dir(workspace) == workspace/"mem0"; point workspace at runs/autonomy so the store lands at
# runs/autonomy/mem0/<channel>/ exactly alongside the learnings JSONL the pre-N5 crypto copy already wrote.
_WS = str(ROOT / "runs" / "autonomy")


def remember(text: str, meta: dict | None = None, thread: str = "mem", objective: str = "", cycle: int = 0,
             channel: str = "default", workspace: str | None = None) -> bool:
    return _h.remember(text, meta=meta, thread=thread, objective=objective, cycle=cycle,
                       channel=channel, workspace=workspace or _WS)


def recall(objective: str, k: int = 3, channel: str = "default", workspace: str | None = None,
           across_channels: bool = True) -> str:
    return _h.recall(objective, k=k, channel=channel, workspace=workspace or _WS,
                     across_channels=across_channels)


def backend(workspace: str | None = None, channel: str = "default") -> str:
    return _h.backend(workspace or _WS, channel)


def available(workspace: str | None = None, channel: str = "default") -> bool:
    return _h.available(workspace or _WS, channel)


def mem0_dir(workspace: str | None = None):
    return _h.mem0_dir(workspace or _WS)


if __name__ == "__main__":
    print("backend:", backend())
    remember("dollar bars clear maker costs better than time bars at 4h",
             objective="cost analysis across bar types", channel="demo")
    print(recall("which bar type clears trading costs best", k=2, channel="demo"))
