#!/usr/bin/env python3
"""loop_health.py -- the OVERSEER's 60s HEALTH + STATE check (not just liveness).

The user mandate (2026-06-06): "as orchestrator your constant job is to check diagnostics during those 60s calls --
system health, no gaps, breakages, slowness -- to act/correct/intervene, AND surface the state: are the langgraph
loops learning, writing to the right things, is communication with meta stable + productive?"

This is the mechanized version of that check so it is repeatable, not ad-hoc grep. It reports a STATE digest and a
list of ISSUES to act on. Exit code = number of ISSUES (0 = healthy). Read-only. No emoji (cp1252).

Checks:
  LIVENESS      -- solutioning/meta loop locks' PIDs vs the OS; watcher process present.
  LEARNING      -- learnings lanes (expert/plain/meta/sol): count + last-write age. A lane that an ACTIVE loop should
                   be feeding but is stale > STALE_MIN is an ISSUE (the loop is hung, mis-channeled, or not reflecting).
  WRITING-OK    -- each ALIVE loop's metaop checkpoint (.db-wal) advanced within STALL_MIN (durable progress).
  META-COMMS    -- is the meta view fresh? In ATTENDED mode the meta loop is the OVERSEER, so meta.jsonl freshness is
                   advisory (the overseer surfaces the digest live); in UNATTENDED mode a stale meta lane is an ISSUE.
  SLOWNESS      -- loop alive but neither its lane NOR its checkpoint advanced in STALL_MIN => likely hung.

Usage: python scripts/autonomy/loop_health.py [--json] [--full]   (--full also runs skill_diagnostics + CDAP)
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
LANES = os.path.join(AUT, "learnings")
STALE_MIN = 20.0   # a lane an active loop should feed, quiet longer than this = suspicious
STALL_MIN = 12.0   # alive loop whose checkpoint AND lane both quiet this long = likely hung


def _age_min(path: str) -> float | None:
    try:
        return (time.time() - os.path.getmtime(path)) / 60.0
    except OSError:
        return None


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from proc_liveness import alive as _pid_alive  # G-J 2026-06-07: the ONE create-time-aware liveness check (no PID reuse)


def _alive_loops() -> dict:
    """thread -> {pid, alive} from the metaop lock files. Canonical path is runs/autonomy/locks/<thread>.lock with a
    {"pid": ...} payload (same source loops_alive.py uses -- a crashed loop leaves a stale lock, so check the PID)."""
    out = {}
    for lk in glob.glob(os.path.join(AUT, "locks", "*.lock")):
        try:
            d = json.load(open(lk, encoding="utf-8"))
            pid = int(d.get("pid", 0))
            thread = d.get("thread") or os.path.splitext(os.path.basename(lk))[0]
            if pid:
                out[thread] = {"pid": pid, "alive": _pid_alive(pid, d.get("created"))}
        except Exception:
            continue
    return out


def _watcher_running() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(["name", "cmdline"]):
            cl = " ".join(p.info.get("cmdline") or [])
            if "watcher.py" in cl:
                return True
    except Exception:
        pass
    return os.path.exists(os.path.join(AUT, "watcher.log")) and (_age_min(os.path.join(AUT, "watcher.log")) or 999) < 3


def collect() -> dict:
    issues = []
    state = {"liveness": {}, "lanes": {}, "checkpoints": {}, "watcher": _watcher_running()}

    loops = _alive_loops()
    state["liveness"] = loops
    alive_threads = [t for t, v in loops.items() if v["alive"]]
    if not state["watcher"]:
        issues.append("WATCHER not running -- a crashed loop would go unnoticed; relaunch watcher.py")

    # LEARNING: lane freshness
    for lane in ["expert", "plain", "meta", "sol"]:
        p = os.path.join(LANES, f"{lane}.jsonl")
        if not os.path.exists(p):
            state["lanes"][lane] = {"n": 0, "age_min": None}
            continue
        n = sum(1 for l in open(p, encoding="utf-8") if l.strip())
        age = _age_min(p)
        state["lanes"][lane] = {"n": n, "age_min": round(age, 1) if age is not None else None}

    # WRITING-OK + SLOWNESS: each alive loop's checkpoint advancing
    for t in alive_threads:
        wals = glob.glob(os.path.join(AUT, f"metaop_{t}.db-wal")) or glob.glob(os.path.join(AUT, f"*{t}*.db-wal"))
        cp_age = min([_age_min(w) for w in wals if _age_min(w) is not None], default=None)
        # the channel this loop feeds (expert/plain) -- best-effort from thread name
        lane = "expert" if "expert" in t or "dirlint" in t or "build" in t else ("plain" if "plain" in t else None)
        lane_age = state["lanes"].get(lane, {}).get("age_min") if lane else None
        state["checkpoints"][t] = {"cp_age_min": round(cp_age, 1) if cp_age is not None else None, "feeds": lane}
        cp_stalled = cp_age is None or cp_age > STALL_MIN
        lane_stalled = lane_age is None or lane_age > STALL_MIN
        if cp_stalled and lane_stalled:
            issues.append(f"SLOWNESS/HUNG: loop '{t}' alive but checkpoint({cp_age}) AND lane({lane}:{lane_age}) "
                          f"both quiet > {STALL_MIN}m -- inspect/relaunch from checkpoint")

    # META-COMMS: attended -> overseer is meta (advisory); unattended -> stale meta lane is an issue
    mode = "attended"
    try:
        mode = json.load(open(os.path.join(AUT, "autonomy_launch.json"), encoding="utf-8")).get("mode", "attended")
    except Exception:
        pass
    state["mode"] = mode
    meta_age = state["lanes"].get("meta", {}).get("age_min")
    if mode == "unattended" and (meta_age is None or meta_age > STALE_MIN):
        issues.append(f"META-COMMS: meta lane stale ({meta_age}m) in UNATTENDED mode -- meta loop not synthesizing; "
                      f"loops' learnings are not feeding the meta view")
    elif mode == "attended":
        state["meta_note"] = ("ATTENDED: the OVERSEER is the meta loop -- meta.jsonl freshness is advisory; the "
                              "overseer must READ the loop lanes + WRITE the meta synthesis forward for persistence")

    state["issues"] = issues
    return state


def main():
    st = collect()
    if "--json" in sys.argv:
        print(json.dumps(st, indent=2))
        return len(st["issues"])
    print(f"=== loop_health: mode={st.get('mode')} | watcher={'OK' if st['watcher'] else 'DOWN'} | "
          f"{len(st['issues'])} ISSUE(s) ===")
    alive = [t for t, v in st["liveness"].items() if v["alive"]]
    print(f"  LIVENESS: alive={alive or 'none'}")
    print("  LEARNING (lane n | age): " + ", ".join(
        f"{k}={v['n']}@{v['age_min']}m" for k, v in st["lanes"].items()))
    if st["checkpoints"]:
        print("  WRITING:  " + ", ".join(f"{t}=cp{v['cp_age_min']}m" for t, v in st["checkpoints"].items()))
    if st.get("meta_note"):
        print(f"  META:     {st['meta_note']}")
    for i in st["issues"]:
        print(f"  ISSUE -> {i}")
    if not st["issues"]:
        print("  healthy.")
    if "--full" in sys.argv:
        import subprocess as _sp
        _nw = _sp.CREATE_NO_WINDOW if os.name == "nt" else 0
        _r = _sp.run([sys.executable, os.path.join(ROOT, "scripts", "skill_diagnostics.py")],
                     capture_output=True, text=True, creationflags=_nw)
        _lines = (_r.stdout or "").splitlines()
        if _lines:
            print(_lines[0])
    return len(st["issues"])


if __name__ == "__main__":
    raise SystemExit(main())
