#!/usr/bin/env python3
"""meta_loop.py -- the META loop as a REAL, BOUNDED, TRACKED background process (closes the 3-loop concurrency gap).

Before this, the META "loop" in attended mode was PROSE: the overseer was nominally the meta loop, but there was no
actual tracked process doing the meta job on a cadence. The gap map found only the SOLVER loop is a real process.
This module makes the META job a genuine bounded-lifetime worker the watcher + Stop-hook SEE (via track_job).

WHAT THE META JOB IS (per loop_health.py + the AUTONOMOUS_RUNNER §5 self-improving loop):
  - READ the solver loop's learning lanes (expert/plain/sol/default) -- what is the solver discovering?
  - SYNTHESIZE a meta note (lane freshness + new-since-last-tick deltas + any health issues) and WRITE it FORWARD to
    runs/autonomy/learnings/meta.jsonl (the durable meta view -- experience compounds across cycles + sessions).
  - RUN loop_health.py (the 60s diagnostics) and surface gaps/issues into the meta note.
It does NOT solve the objective (that is the solver loop) and it NEVER commits (overseer-only).

SAFETY CONTRACT (this is additive autonomy machinery -- the user has had STUCK issues):
  - TRACKED: registers runs/autonomy/locks/<job_id>.lock via track_job so the watcher monitors liveness and the
    Stop hook waits for it instead of silently dying mid-window.
  - BOUNDED: --max-iters AND --max-hours (whichever comes first) -> the process ALWAYS exits; it is re-spawned by the
    watcher/launcher if it dies, never an unbounded daemon.
  - RESUMABLE / IDEMPOTENT: state in runs/autonomy/meta_loop_state.json (last lane line-counts + iter). A relaunch
    continues from the last tick; re-running never double-counts (it diffs against the persisted counts).
  - NEVER COMMITS: no git here at all.
  - CLEAN TEARDOWN: untracks its lock in a finally block on ANY exit (normal, signal, exception) -> no orphan lock.

No emoji (Windows cp1252). Read-only w.r.t. the solver loop's lanes; append-only to meta.jsonl.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
LANES = os.path.join(AUT, "learnings")
STATE = os.path.join(AUT, "meta_loop_state.json")
LOOP_HEALTH = os.path.join(ROOT, "scripts", "autonomy", "loop_health.py")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_job import track, untrack  # the canonical lock contract (watcher + Stop hook read these)

# reuse the canonical learnings writer so meta notes land in the EXACT lane the watcher/loop_health read.
sys.path.insert(0, ROOT)
try:
    from harness.metaop import learnings as _learn
    _HAVE_LEARN = True
except Exception:
    _HAVE_LEARN = False

# the solver lanes the meta loop watches (everything EXCEPT its own meta lane -- it reflects on the SOLVER's output).
SOLVER_LANES = ["expert", "plain", "sol", "default"]


def _lane_counts() -> dict:
    """thread/lane -> line count for every solver lane (the meta loop's read-only view of solver progress)."""
    out = {}
    for lane in SOLVER_LANES:
        p = os.path.join(LANES, f"{lane}.jsonl")
        try:
            with open(p, encoding="utf-8") as fh:
                out[lane] = sum(1 for l in fh if l.strip())
        except FileNotFoundError:
            out[lane] = 0
    return out


def _load_state() -> dict:
    try:
        return json.load(open(STATE, encoding="utf-8"))
    except Exception:
        return {"iter": 0, "last_counts": {}}


def _save_state(state: dict) -> None:
    try:
        with open(STATE, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
    except Exception:
        pass


def _run_loop_health() -> tuple[int, str]:
    """Run loop_health.py --json (read-only diagnostics). Returns (n_issues, compact_digest). Bounded by timeout so
    a hung health check can never wedge the meta loop."""
    try:
        out = subprocess.run([sys.executable, LOOP_HEALTH, "--json"], cwd=ROOT,
                             capture_output=True, text=True, timeout=60,
                             creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        try:
            st = json.loads(out.stdout or "{}")
        except Exception:
            st = {}
        issues = st.get("issues", [])
        watcher = "OK" if st.get("watcher") else "DOWN"
        alive = [t for t, v in st.get("liveness", {}).items() if v.get("alive")]
        digest = f"watcher={watcher} alive_loops={alive or 'none'} issues={len(issues)}"
        if issues:
            digest += " :: " + " | ".join(str(i)[:90] for i in issues[:3])
        return len(issues), digest
    except Exception as e:
        return -1, f"loop_health unavailable: {type(e).__name__}: {e}"


def _write_meta_note(lesson: str, objective: str, cycle: int) -> None:
    """Write the meta synthesis FORWARD to runs/autonomy/learnings/meta.jsonl (durable; compounds across sessions)."""
    if _HAVE_LEARN:
        # workspace=AUT -> learnings_dir(AUT) == runs/autonomy/learnings ; channel='meta' -> meta.jsonl
        _learn.record(lesson, thread="META_LOOP", objective=objective, cycle=cycle, channel="meta", workspace=AUT)
        return
    # fallback: write the same row shape directly (never let a learnings-import failure silence the meta loop)
    try:
        os.makedirs(LANES, exist_ok=True)
        row = {"ts": int(time.time()), "thread": "META_LOOP", "objective": str(objective)[:160],
               "cycle": cycle, "lesson": str(lesson)[:600]}
        with open(os.path.join(LANES, "meta.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _one_tick(state: dict, objective: str) -> dict:
    """One META iteration: diff lane counts since last tick, run health, synthesize + write a meta note forward."""
    counts = _lane_counts()
    last = state.get("last_counts", {})
    deltas = {lane: counts.get(lane, 0) - last.get(lane, 0) for lane in SOLVER_LANES}
    new_total = sum(max(0, d) for d in deltas.values())
    n_issues, health = _run_loop_health()

    nonzero = {k: v for k, v in deltas.items() if v}
    if new_total > 0:
        synth = (f"[META tick {state.get('iter', 0) + 1}] solver lanes advanced +{new_total} since last tick "
                 f"(deltas={nonzero}); HEALTH: {health}.")
    else:
        synth = (f"[META tick {state.get('iter', 0) + 1}] no new solver lessons since last tick "
                 f"(counts={counts}); HEALTH: {health}.")
    if n_issues > 0:
        synth += f" GAPS: {n_issues} health issue(s) surfaced for the overseer to act on."

    _write_meta_note(synth, objective, state.get("iter", 0) + 1)
    state["iter"] = state.get("iter", 0) + 1
    state["last_counts"] = counts
    state["last_tick_ts"] = int(time.time())
    _save_state(state)
    return state


def main():
    ap = argparse.ArgumentParser(description="META loop -- bounded, tracked meta-synthesis background process")
    ap.add_argument("--job-id", default="meta_loop", help="track_job lock id (watcher + Stop hook see this)")
    ap.add_argument("--objective", default="improve the solving (meta lane)", help="objective context for notes")
    ap.add_argument("--max-iters", type=int, default=20, help="hard cap on iterations (bounded lifetime)")
    ap.add_argument("--max-hours", type=float, default=3.0, help="hard wall-clock cap (bounded lifetime)")
    ap.add_argument("--interval", type=float, default=60.0, help="seconds between ticks")
    a = ap.parse_args()

    os.makedirs(AUT, exist_ok=True)
    ok, msg = track(a.job_id, pid=os.getpid(), cmd=f"meta_loop.py --objective {a.objective[:40]}", kind="loop")
    if not ok:
        # another live meta loop already owns the lock -> do NOT spawn a duplicate (idempotent launch).
        print(f"[meta_loop] not starting: {msg}")
        return 0
    print(f"[meta_loop] {msg} max_iters={a.max_iters} max_hours={a.max_hours} interval={a.interval}s")

    deadline = time.time() + a.max_hours * 3600.0
    state = _load_state()
    try:
        for i in range(a.max_iters):
            if time.time() >= deadline:
                print(f"[meta_loop] EXIT reason=max_hours after {i} ticks")
                break
            state = _one_tick(state, a.objective)
            print(f"[meta_loop] tick {state['iter']} done (lanes={state['last_counts']})")
            if i < a.max_iters - 1 and time.time() + a.interval < deadline:
                time.sleep(a.interval)
            elif i < a.max_iters - 1:
                # not enough budget for another full interval -> stop cleanly rather than overshoot the deadline
                print(f"[meta_loop] EXIT reason=max_hours (no budget for another interval) after {state['iter']} ticks")
                break
        else:
            print(f"[meta_loop] EXIT reason=max_iters after {state['iter']} ticks")
    finally:
        # CLEAN TEARDOWN: always release the lock so the loop is not falsely 'alive' (watcher + Stop hook).
        untrack(a.job_id)
        print(f"[meta_loop] untracked {a.job_id}; clean exit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
