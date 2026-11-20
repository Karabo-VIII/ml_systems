"""Harness LEARNINGS -- the persistent memory that makes experience compound across runs, by CHANNEL (lane).

Lessons persist across different objectives/threads. Each run writes to a CHANNEL (lane); by default the channel
is the run's mode, so variants keep SEPARATE improvement loops; point two runs at the same channel to POOL them.
`reflect` WRITES; `plan` READS -- the harness never re-learns what a prior run already settled (monotonic).

Project-agnostic: storage lives under the harness WORKSPACE (config.learnings_dir), not any specific repo.
Append-only JSONL at <workspace>/learnings/<channel>.jsonl. No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .config import learnings_dir


def _path(channel: str, workspace: str | None = None) -> Path:
    return learnings_dir(workspace) / f"{(channel or 'default')}.jsonl"


def record(lesson: str, thread: str, objective: str, cycle: int, channel: str = "default",
           workspace: str | None = None) -> None:
    """Append one lesson to a channel's store (called by reflect)."""
    if not lesson or not str(lesson).strip():
        return
    try:
        row = {"ts": int(time.time()), "thread": thread, "objective": str(objective)[:160],
               "cycle": cycle, "lesson": str(lesson)[:600]}
        with open(_path(channel, workspace), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # best-effort; never break the loop


def recent(n: int = 12, channel: str = "default", workspace: str | None = None) -> list:
    """Most recent N lessons from a channel (newest last). Best-effort."""
    p = _path(channel, workspace)
    if not p.exists():
        return []
    out = []
    try:
        for line in p.read_text(encoding="utf-8").strip().splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    except Exception:
        return []
    return out[-n:]


def summary_for_plan(n: int = 12, channel: str = "default", workspace: str | None = None) -> str:
    """Compact, prompt-ready digest of prior learnings from a channel, for the plan node."""
    rows = recent(n, channel, workspace)
    if not rows:
        return f"(no prior learnings yet in channel '{channel}')"
    lines = [f"- [{r.get('thread', '?')}] {r.get('lesson', '')}" for r in rows]
    return (f"PRIOR LEARNINGS (channel '{channel}', compounded across runs -- do NOT re-mine; build on them):\n"
            + "\n".join(lines))


def _collect_rows(channel: str, workspace: str | None, across_channels: bool) -> list:
    chans = all_channels(workspace) if across_channels else [channel]
    rows = []
    for c in chans:
        for r in recent(10 ** 9, c, workspace):
            r["_channel"] = c
            rows.append(r)
    return rows


def similar_for_plan(objective: str, k: int = 3, channel: str = "default", workspace: str | None = None,
                     across_channels: bool = True) -> str:
    """G-B (mem0-style): top-k TASK-SIMILAR past lessons (TF-IDF cosine over stored objective+lesson), retrieved by
    SIMILARITY to THIS objective -- NOT by recency. This is the fix for 'instances forget after compaction': a
    lesson from a SIMILAR past task resurfaces even when it is old / in another lane (across_channels). Pure-local
    (sklearn TF-IDF; no embedding model, no network, no cloud). Returns a prompt-ready string; degrades to a clear
    note if sklearn is absent or there are <2 prior rows. Best-effort -- never raises into the loop."""
    obj = str(objective or "").strip()
    if not obj:
        return "(no objective given for similarity recall)"
    try:
        rows = _collect_rows(channel, workspace, across_channels)
        if len(rows) < 2:
            return "(not enough prior cycles yet for task-similarity recall)"
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        corpus = [f"{r.get('objective', '')} {r.get('lesson', '')}" for r in rows]
        vec = TfidfVectorizer(stop_words="english", max_features=4000)
        m = vec.fit_transform(corpus + [obj])
        sims = cosine_similarity(m[-1], m[:-1]).ravel()
        order = list(sims.argsort()[::-1][:k])
        picked = [(rows[i], float(sims[i])) for i in order if sims[i] > 0.01]
        if not picked:
            return "(no task-similar prior cycles found)"
        lines = [f"- (sim={s:.2f})[{r.get('_channel', '?')}/{r.get('thread', '?')}] {r.get('lesson', '')}"
                 for r, s in picked]
        return ("TASK-SIMILAR PRIOR CYCLES (retrieved by similarity to THIS objective, not recency -- reuse these "
                "even if old/in another lane; do NOT re-derive them):\n" + "\n".join(lines))
    except Exception:
        return "(task-similarity recall unavailable)"


def all_channels(workspace: str | None = None) -> list:
    d = learnings_dir(workspace)
    return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []


def stats(workspace: str | None = None) -> dict:
    chans = all_channels(workspace)
    return {"channels": chans, "total_lessons": sum(len(recent(10 ** 9, c, workspace)) for c in chans),
            "per_channel": {c: len(recent(10 ** 9, c, workspace)) for c in chans}}


if __name__ == "__main__":
    record("cost-clearing favors approach X", "t1", "obj A", 1, channel="plain")
    print("plain:", summary_for_plan(3, "plain"))
    print("stats:", stats())
