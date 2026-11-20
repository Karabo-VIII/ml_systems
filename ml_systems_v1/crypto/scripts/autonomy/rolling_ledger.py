#!/usr/bin/env python3
"""rolling_ledger.py -- the anti-COMPACTION rolling memory: the important nuances / lessons / pivots / user-corrections
of the CURRENT chat, written-forward per turn and read-forward each turn (esp. AFTER compaction) so evolution does not
break or drift when the raw transcript is summarized away.

WHY (user mandate 2026-06-06): "instances forget, especially after compaction. I want them to remember the current chat
wholly per turn -- or at least the rolling CONSIDERATIONS, LESSONS, PIVOTS, NUANCES -- this is the whole basis of
evolution. If info is lost during rolling/continuing, the evolution breaks or goes the wrong direction. Not millions of
lines -- but you must be able to reference the important nuances, lessons, etc. as we work."

This is DISTINCT from project memory (MEMORY.md / learnings lanes = cross-session, project-level). The rolling ledger is
CURRENT-SESSION scoped: the live thread of what-matters-right-now. It COMPLEMENTS the harness compaction summary -- a
durable file the summary's lossiness cannot drop. Distil, don't dump (one line per nuance).

KINDS: CORRECTION (a user jolt / "you got X wrong") | PIVOT (a direction change) | LESSON (a durable takeaway) |
       CONSTRAINT (a standing rule/limit to honor) | DECISION (a committed choice + why) | OPEN_Q (an unresolved
       question / fork) | NUANCE (a subtle thing easy to lose).

Usage:
  python scripts/autonomy/rolling_ledger.py note CORRECTION "the .claude write prompts because the file is open in IDE"
  python scripts/autonomy/rolling_ledger.py digest            # the rolling state, grouped -- read this each turn
  python scripts/autonomy/rolling_ledger.py digest --kinds CONSTRAINT,OPEN_Q
No emoji (cp1252).
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROLL_DIR = os.path.join(ROOT, "runs", "autonomy", "rolling")
KINDS = ["CORRECTION", "PIVOT", "LESSON", "CONSTRAINT", "DECISION", "OPEN_Q", "NUANCE"]
# priority order when digesting (most evolution-critical first)
PRIORITY = {"CONSTRAINT": 0, "CORRECTION": 1, "PIVOT": 2, "OPEN_Q": 3, "DECISION": 4, "LESSON": 5, "NUANCE": 6}


def _path(session: str) -> str:
    os.makedirs(ROLL_DIR, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (session or "current"))
    return os.path.join(ROLL_DIR, f"{safe}.jsonl")


def note(kind: str, text: str, session: str = "current", ts_ms: int | None = None) -> dict:
    kind = (kind or "NUANCE").upper()
    if kind not in KINDS:
        kind = "NUANCE"
    rec = {"ts": ts_ms if ts_ms is not None else int(time.time() * 1000), "kind": kind, "text": text.strip()}
    with open(_path(session), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def load(session: str = "current") -> list:
    p = _path(session)
    if not os.path.exists(p):
        return []
    out = []
    for ln in open(p, encoding="utf-8"):
        if ln.strip():
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def digest(session: str = "current", kinds: list | None = None, limit_per_kind: int = 12) -> dict:
    rows = load(session)
    if kinds:
        want = {k.upper() for k in kinds}
        rows = [r for r in rows if r.get("kind") in want]
    grouped: dict = {}
    for r in rows:
        grouped.setdefault(r.get("kind", "NUANCE"), []).append(r.get("text", ""))
    # most-recent-N per kind, kinds in evolution-priority order
    ordered = {k: grouped[k][-limit_per_kind:] for k in sorted(grouped, key=lambda k: PRIORITY.get(k, 9))}
    return {"session": session, "n": len(rows), "by_kind": ordered}


def main():
    a = sys.argv[1:]
    session = "current"
    if "--session" in a:
        session = a[a.index("--session") + 1]
    if a and a[0] == "note" and len(a) >= 3:
        rec = note(a[1], a[2], session)
        print(f"noted [{rec['kind']}] {rec['text'][:80]}")
        return 0
    # digest (default)
    kinds = None
    if "--kinds" in a:
        kinds = a[a.index("--kinds") + 1].split(",")
    d = digest(session, kinds)
    if "--json" in a:
        print(json.dumps(d, indent=2))
        return 0
    print(f"=== ROLLING LEDGER (session={d['session']} | {d['n']} notes) -- the live state to carry each turn ===")
    if not d["by_kind"]:
        print("  (empty -- write-forward CORRECTION/PIVOT/CONSTRAINT/LESSON notes as they happen)")
    for kind, items in d["by_kind"].items():
        print(f"  {kind}:")
        for t in items:
            print(f"    - {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
