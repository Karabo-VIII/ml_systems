"""Operator LEARNINGS -- crypto-consumer SHIM over the canonical harness.metaop.learnings (G-A dedup 2026-06-07).

The persistent, channel-keyed compounding memory now lives ONCE in harness/metaop/learnings.py (project-agnostic;
storage under a configurable WORKSPACE). This shim re-exports that engine UNCHANGED and only injects the crypto
WORKSPACE so lessons keep landing at the historical path runs/autonomy/learnings/<channel>.jsonl (harness resolves
learnings_dir(workspace) = workspace/learnings, so workspace = <repo>/runs/autonomy).

reflect WRITES; plan READS -- the loop never re-learns what a prior run already settled (monotonic). The crypto loop
calls record(...)/summary_for_plan(...)/recent(...)/stats()/all_channels() with NO workspace arg, so each wrapper
defaults workspace to the crypto path. No emoji (Windows cp1252).
"""
from __future__ import annotations

from pathlib import Path

from harness.metaop import learnings as _h  # the canonical engine
# re-export everything (so `from metaop.learnings import *` and attribute access see the full surface)
from harness.metaop.learnings import *  # noqa: F401,F403

ROOT = Path(__file__).resolve().parents[3]
# harness learnings_dir(workspace) == workspace/"learnings"; point workspace at runs/autonomy so the lessons land at
# runs/autonomy/learnings/<channel>.jsonl exactly as the pre-dedup crypto copy wrote them.
_WS = str(ROOT / "runs" / "autonomy")


def record(lesson: str, thread: str, objective: str, cycle: int, channel: str = "default",
           workspace: str | None = None) -> None:
    return _h.record(lesson, thread, objective, cycle, channel=channel, workspace=workspace or _WS)


def recent(n: int = 12, channel: str = "default", workspace: str | None = None) -> list:
    return _h.recent(n, channel, workspace or _WS)


def summary_for_plan(n: int = 12, channel: str = "default", workspace: str | None = None) -> str:
    return _h.summary_for_plan(n, channel, workspace or _WS)


def all_channels(workspace: str | None = None) -> list:
    return _h.all_channels(workspace or _WS)


def stats(workspace: str | None = None) -> dict:
    return _h.stats(workspace or _WS)


if __name__ == "__main__":
    record("cost-clearing favors daily/4h", "t1", "obj A", 1, channel="plain")
    record("auditor flagged look-ahead in the entry gate", "t2", "obj B", 1, channel="expert")
    print("plain:", summary_for_plan(3, "plain"))
    print("expert:", summary_for_plan(3, "expert"))
    print("stats:", stats())
