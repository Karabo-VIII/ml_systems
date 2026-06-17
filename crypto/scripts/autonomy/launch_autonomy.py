#!/usr/bin/env python3
"""launch_autonomy.py -- stand up the THREE unavoidable autonomous loops (the /orc modus operandi).

User mandate (2026-06-06): in autonomous mode we ALWAYS launch three things together:
  1. PROBLEM-SOLVER loop (expert + plain) -- the LangGraph self-evolving acting loop that solves the objective.
  2. META AGENT -- the loop that improves the solving: learnings, new questions, new ideas, frontier re-rank.
  3. PROJECT-WIDE EVOLUTION loop -- fires every 3 hours; hardens the project + framework itself.

Two modes:
  --mode attended  (default): the OVERSEER (you) is present and drives loop-1 via fast in-harness Agent workers.
     This writes the loop-1 frontier (objective + success_criteria), arms the loop-2 meta cadence, and ensures
     loop-3 (the 3-hourly evolution cycle + liveness watcher). It does NOT spawn the slow `claude -p` metaop loops.
  --mode unattended (or --spawn-loops): ALSO spawns the metaop loops (expert + plain + meta) via `claude -p`
     + the Stop-hook/driver so the loop survives context limits / session end (cross-session autonomy).

Resumable: state in runs/autonomy/autonomy_launch.json. Re-running reports + resumes rather than duplicating.
No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
MANIFEST = os.path.join(AUT, "autonomy_launch.json")
FRONTIER = os.path.join(AUT, "frontier.json")
LOCKS = os.path.join(AUT, "locks")
RUN_METAOP = os.path.join(ROOT, "scripts", "autonomy", "run_metaop.py")
WATCHER = os.path.join(ROOT, "scripts", "autonomy", "watcher.py")
META_LOOP = os.path.join(ROOT, "scripts", "autonomy", "meta_loop.py")
EVOLUTION_LOOP = os.path.join(ROOT, "scripts", "autonomy", "evolution_loop.py")
PY = sys.executable
# WINDOWLESS interpreter + flag for detached background spawns -- so no terminal window flashes (user saw one
# 2026-06-06). pythonw.exe has no console; CREATE_NO_WINDOW (0x08000000) is the correct flag (DETACHED_PROCESS shows a
# console and is mutually exclusive with CREATE_NO_WINDOW). CREATE_NEW_PROCESS_GROUP (0x200) keeps it detached.
PYW = (PY.lower().replace("python.exe", "pythonw.exe") if os.name == "nt" else PY)
if not os.path.exists(PYW):
    PYW = PY
NOWIN_FLAGS = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())[:24].strip("-") or "obj"


def _select_brain_model(forced: str | None = None) -> dict:
    """Choose the LOCAL brain model the spawned loops will run on, and EXPORT it via os.environ['OLLAMA_MODEL'] so
    every detached child (metaop / watcher / meta_loop / evolution_loop) inherits it (the Popens pass no explicit
    env -> children inherit ours). Default = the most capable PULLED ollama model that fits this GPU's VRAM
    (best_local_model / MODEL_LADDER -> qwen2.5-coder:7b on an 8GB card). An explicit --brain-model, or a
    pre-set OLLAMA_MODEL env, always wins. Never raises -- on any failure the loops fall back to the engine
    default. Returns a small dict for the launch manifest."""
    if forced:
        os.environ["OLLAMA_MODEL"] = forced
        return {"model": forced, "why": "forced via --brain-model"}
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts", "autonomy"))
        from ensure_brain import best_local_model, DEFAULT_HOST
        model, why = best_local_model(DEFAULT_HOST)
        os.environ["OLLAMA_MODEL"] = model
        return {"model": model, "why": why}
    except Exception as e:
        return {"model": os.environ.get("OLLAMA_MODEL", "(engine default)"),
                "why": f"selector unavailable ({type(e).__name__}); using env/engine default"}


def _atomic_write_json(path: str, obj) -> None:
    """Crash-safe + concurrency-safe full-file JSON write: serialize to a sibling .tmp then os.replace.

    H2 (2026-06-09): frontier.json / autonomy_launch.json are read by the live loop, the 60s watcher, and
    status.py while a relaunch (or a racing /orc invocation) rewrites them. A bare `json.dump(open(p,"w"))`
    truncates-then-writes in place, so an interrupted/concurrent writer can leave a reader a half-written
    (invalid-JSON) file -> the loop's fail-open read silently DROPS the frontier (all queued nodes lost).
    os.replace is atomic on the same filesystem (Windows + POSIX): a reader sees either the old or the new
    file, never a torn one. Mirrors scripts/autonomy/skill_library._save."""
    import time as _time, threading as _threading
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    # UNIQUE tmp per (process, thread): two racing writers (e.g. a watcher relaunch vs a manual /orc) must
    # NOT share one tmp path, or one writer's os.replace consumes the other's tmp -> FileNotFoundError.
    tmp = "%s.%d.%d.tmp" % (path, os.getpid(), _threading.get_ident())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    # os.replace is atomic (reader sees old-or-new, never a torn file -> the on-disk file is NEVER left
    # permanently corrupt, which is the data-loss bug being closed). On Windows it can transiently raise
    # PermissionError (WinError 5) if a reader currently has the TARGET open; readers here are brief, so a
    # short bounded retry clears it. NOTE: os.rename is NOT a valid fallback on Windows (cannot replace an
    # existing target -> WinError 183); retry os.replace instead.
    last = None
    for _i in range(50):  # ~0.5s worst case; the reader's open() is sub-ms
        try:
            os.replace(tmp, path)
            return
        except OSError as e:  # WinError 5 (sharing violation) / transient contention -- retry, never crash launch
            last = e
            _time.sleep(0.01)
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise last if last is not None else RuntimeError("atomic replace failed")


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from proc_liveness import alive as _alive  # G-J 2026-06-07: the ONE create-time-aware liveness check (no PID reuse)


def _watcher_running() -> bool:
    """Is a liveness/evolution watcher process alive? (checks any lock + the watcher log freshness)."""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                 "Where-Object { $_.CommandLine -like '*watcher.py*' } | Measure-Object | "
                 "Select-Object -ExpandProperty Count"],
                capture_output=True, text=True, timeout=20,
                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            return (out.stdout or "0").strip().isdigit() and int(out.stdout.strip()) > 0
        except Exception:
            return False
    return False


def _write_frontier(objective, success, anchor, window_end):
    """Loop-1 queue. Preserve existing NODES if the objective is unchanged (resume); else seed empty for the overseer."""
    existing = {}
    if os.path.exists(FRONTIER):
        try:
            existing = json.load(open(FRONTIER, encoding="utf-8"))
        except Exception:
            existing = {}
    same = existing.get("objective", "").strip()[:60] == objective.strip()[:60]
    nodes = existing.get("nodes", []) if same else []
    ledger = existing.get("overseer", {}).get("fulfillment_ledger", []) if same else []
    f = {
        "objective": objective,
        "success_criteria": success,
        "overseer": {
            "adopted_command": "launched via /orc launch_autonomy.py (3-loop autonomous gate)",
            "acceptance_test": success,
            "stop_conditions": [f"WINDOW {window_end}" if window_end else "until success_criteria VERIFIED",
                                "all nodes done + verified", "NEEDS_IRREVERSIBLE_OK"],
            "start_anchor_verified": anchor,
            "fulfillment_ledger": ledger,
            "user_proxy_notes": "OVERSEER dispatches loop-1 nodes to Agent workers (attended) or metaop (unattended); "
                                "workers never commit; RWYB every node; loop-2 meta re-ranks; loop-3 fires every 3h.",
        },
        "value_floor": 0.3,
        "budget": {"spent": existing.get("budget", {}).get("spent", 0) if same else 0,
                   "max_cycles": 16, "note": window_end or "open"},
        "nodes": nodes,
    }
    _atomic_write_json(FRONTIER, f)
    return len(nodes)


def _spawn_metaop(objective, success, thread, mode, channel, budget):
    """Spawn a durable metaop loop detached. Returns (thread, pid|None, cmd)."""
    # --fill-window (NO-IDLE-STOP): an unattended autonomy loop exists to USE the allocated window -- when its planned
    # frontier completes with budget remaining it DRAIN-REPLANS the next adjacent work instead of ending early. This is
    # the loop-level cure for the idle-stop failure (the mechanical twin of an overseer stopping with time left).
    cmd = [PYW, RUN_METAOP, "launch", "--objective", objective, "--success", success,
           "--budget", str(budget), "--parallel", "1", "--backend", "composite",
           "--mode", mode, "--learnings-channel", channel, "--thread", thread, "--durable", "--fill-window"]
    try:
        if os.name == "nt":
            p = subprocess.Popen(cmd, cwd=ROOT, creationflags=NOWIN_FLAGS,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 stdin=subprocess.DEVNULL, close_fds=True)
        else:
            p = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        return {"thread": thread, "pid": p.pid, "cmd": " ".join(cmd[2:])}
    except Exception as e:
        return {"thread": thread, "pid": None, "error": str(e)[:120], "cmd": " ".join(cmd[2:])}


def spawn_tracked_job(job_id, command):
    """P3b CONTRACT: spawn ANY long/detached background job (training, backtest, sweep) THROUGH this helper so it
    registers a runs/autonomy/locks/<job_id>.lock. Without a lock the job is INVISIBLE to the watcher AND to the
    Stop hook -> it can SILENTLY DIE mid-window (the P0 bug). Returns the child pid (or an error string).
    Equivalent CLI: `python scripts/autonomy/track_job.py run <job_id> -- <command...>`."""
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts", "autonomy"))
        from track_job import track
        if os.name == "nt":
            p = subprocess.Popen(command, cwd=ROOT, creationflags=NOWIN_FLAGS,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        else:
            p = subprocess.Popen(command, cwd=ROOT, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        track(job_id, pid=p.pid, cmd=" ".join(str(c) for c in command))
        return p.pid
    except Exception as e:
        return f"ERROR: {e}"


def _spawn_watcher():
    try:
        if os.name == "nt":
            p = subprocess.Popen([PYW, WATCHER, "--max-ticks", "180"], cwd=ROOT, creationflags=NOWIN_FLAGS,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        else:
            p = subprocess.Popen([PY, WATCHER, "--max-ticks", "180"], cwd=ROOT, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        return p.pid
    except Exception as e:
        return f"ERROR: {e}"


def _loop_lock_alive(job_id):
    """Is a tracked loop with this job_id already alive? (idempotent launch -- don't double-spawn a live loop.)"""
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts", "autonomy"))
        from track_job import alive_jobs
        return any(j["id"] == job_id for j in alive_jobs())
    except Exception:
        return False


def _spawn_loop_module(script, job_id, extra_args):
    """Spawn a SELF-TRACKING loop module (meta_loop.py / evolution_loop.py) DETACHED. The module registers its OWN
    track_job lock from inside the process (with its real PID) and untracks on clean exit -- so the watcher + Stop
    hook SEE it. We do NOT pre-register here (that would race the module's own atomic O_EXCL create). Idempotent:
    if a live loop with this job_id already holds the lock we skip. Returns {"job_id","pid"|"skipped"|"error"}."""
    if _loop_lock_alive(job_id):
        return {"job_id": job_id, "skipped": "already alive"}
    cmd = [PYW, script, "--job-id", job_id] + [str(x) for x in extra_args]
    try:
        if os.name == "nt":
            p = subprocess.Popen(cmd, cwd=ROOT, creationflags=NOWIN_FLAGS,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        else:
            p = subprocess.Popen(cmd, cwd=ROOT, start_new_session=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        return {"job_id": job_id, "pid": p.pid, "cmd": " ".join(cmd[1:])}
    except Exception as e:
        return {"job_id": job_id, "error": str(e)[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", required=True)
    ap.add_argument("--success", default="all nodes verified (RWYB), objective fulfilled not asserted")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--mode", default="attended", choices=["attended", "unattended"])
    ap.add_argument("--spawn-loops", action="store_true", help="force-spawn the metaop loops even in attended mode")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--status", action="store_true", help="report the current launch manifest and exit")
    # loop-2 (meta) + loop-3 (evolution) are REAL tracked processes (the 3-loop concurrency gap). WINDOWLESS CANONICAL
    # (2026-06-14): spawned ONLY when the user is AWAY (unattended) or explicitly opts in (--spawn-loops) -- in
    # attended mode the overseer is loops 2+3 inline and evolution_loop's Claude brain pops a "claude" terminal, so
    # attended skips them. --no-extra-loops force-disables in all modes.
    ap.add_argument("--no-extra-loops", action="store_true",
                    help="do NOT spawn the standalone meta_loop + evolution_loop processes (default: spawn ONLY "
                         "when unattended or --spawn-loops; attended skips them to avoid the claude-brain terminal popup)")
    ap.add_argument("--loop-max-hours", type=float, default=None,
                    help="bounded lifetime for the meta + evolution loops (default: --hours, else 3h)")
    ap.add_argument("--brain-model", default=None,
                    help="local ollama brain model for the loops (default: auto-select the most capable pulled "
                         "model that fits this GPU, e.g. qwen2.5-coder:7b). Exported as OLLAMA_MODEL to all children.")
    a = ap.parse_args()
    os.makedirs(AUT, exist_ok=True)

    # Select + EXPORT the local brain model BEFORE any spawn, so every detached loop inherits OLLAMA_MODEL.
    brain = _select_brain_model(a.brain_model)
    print(f"[launch] brain model: {brain['model']} ({brain['why']})")

    if a.status:
        if os.path.exists(MANIFEST):
            print(json.dumps(json.load(open(MANIFEST, encoding="utf-8")), indent=2))
        else:
            print("no autonomy launch manifest yet")
        return 0

    now = dt.datetime.now()
    anchor = now.strftime("%Y-%m-%d %H:%M:%S")
    window_end = (now + dt.timedelta(hours=a.hours)).strftime("%Y-%m-%d %H:%M:%S") if a.hours else None
    slug = _slug(a.objective)

    n_nodes = _write_frontier(a.objective, a.success, anchor, window_end)

    # ARM the autonomous Stop-hook PROMPT-FREE: runs/autonomy/AUTONOMY_ON is OUTSIDE .claude/, so writing it never
    # triggers the IDE/config-dir confirmation that .claude/autonomous_mode.json does (esp. when that file is open in
    # the editor). autonomy_loop.py honors AUTONOMY_ON as the frontier-driven arm when autonomous_mode.json is absent,
    # so we remove a stale one. The 2h/Nh window lives in the frontier stop_conditions; the overseer enforces it +
    # disarms with `rm runs/autonomy/AUTONOMY_ON`. (2026-06-06 fix: no more arming prompts.)
    try:
        am = os.path.join(ROOT, ".claude", "autonomous_mode.json")
        if os.path.exists(am):
            os.remove(am)
        with open(os.path.join(AUT, "AUTONOMY_ON"), "w", encoding="utf-8") as fh:
            fh.write(f"armed {anchor} window_end={window_end or 'open'} (prompt-free; disarm: rm this file)")
    except Exception as e:
        print(f"  (AUTONOMY_ON arm warning: {e})")

    spawn = a.spawn_loops or a.mode == "unattended"
    loops = {"problem_solver_expert": None, "problem_solver_plain": None, "meta_agent": None}
    if spawn:
        loops["problem_solver_expert"] = _spawn_metaop(a.objective, a.success, f"{slug}-expert", "expert", "expert", a.budget)
        loops["problem_solver_plain"] = _spawn_metaop(a.objective, a.success, f"{slug}-plain", "plain", "plain", a.budget)
        loops["meta_agent"] = _spawn_metaop(
            f"META: improve the solving of '{a.objective}' -- surface new questions, new ideas, fold learnings, re-rank the frontier",
            "the solver's frontier is sharper each cycle; learnings written forward", f"{slug}-meta", "expert", "meta", a.budget)

    # loop-3: the project-wide 3-hourly evolution cycle (liveness watcher with evolution-window detection)
    watcher_pid = None
    if not _watcher_running():
        watcher_pid = _spawn_watcher()

    # loop-2 META + loop-3 EVOLUTION as REAL tracked processes (closes the 3-loop concurrency gap: meta+evolution
    # were prose, only the solver was a real process). Each self-registers a track_job lock (watcher + Stop hook see
    # them), is BOUNDED, RESUMABLE, and NEVER commits. Additive -- the solver loop + Stop hook are untouched.
    loop_max_hours = a.loop_max_hours if a.loop_max_hours is not None else (a.hours if a.hours else 3.0)
    extra_loops = {"meta_loop": None, "evolution_loop": None}
    # WINDOWLESS CANONICAL (2026-06-14): the standalone meta_loop + evolution_loop run a Claude brain --
    # evolution_loop -> make_brain('auto') -> the claude-agent-SDK launches the `claude` CLI as a console GRANDCHILD
    # that the LIBRARY spawns WITHOUT CREATE_NO_WINDOW, so on Windows (default terminal = Windows Terminal) a "claude"
    # window pops and STEALS FOCUS / pauses the user's typing. We cannot patch the SDK's internal spawn. But in
    # ATTENDED mode the overseer IS loops 2+3 inline (and Agent dispatch is IN-PROCESS = no claude spawn), so these
    # background loops are REDUNDANT there. So spawn them ONLY when the user is AWAY (unattended) or explicitly opts
    # in (--spawn-loops) -- then any popup is unattended (no one is typing). `spawn` == (spawn_loops or unattended).
    # This is the canonical fix for the recurring "claude terminal pops up" report; zero capability loss attended.
    if spawn and not a.no_extra_loops:
        extra_loops["meta_loop"] = _spawn_loop_module(
            META_LOOP, f"{slug}-metaloop",
            ["--objective", f"META: improve the solving of '{a.objective}'", "--max-hours", loop_max_hours])
        extra_loops["evolution_loop"] = _spawn_loop_module(
            EVOLUTION_LOOP, f"{slug}-evoloop",
            ["--objective", f"EVOLUTION: evolve the planner for '{a.objective}'", "--max-hours", loop_max_hours])

    manifest = {
        "objective": a.objective, "success_criteria": a.success,
        "anchor_verified": anchor, "window_end": window_end, "mode": a.mode, "spawned_metaop": spawn,
        "brain_model": brain,
        "loops": {
            "1_problem_solver": {"substrate": "metaop expert+plain" if spawn else "OVERSEER->Agent dispatch (attended)",
                                 "expert": loops["problem_solver_expert"], "plain": loops["problem_solver_plain"],
                                 "frontier": FRONTIER, "nodes_pending": n_nodes},
            "2_meta_agent": {"substrate": "metaop meta channel" if spawn else "OVERSEER reflect step (attended)",
                             "detail": loops["meta_agent"], "cadence": "every node + 25/50/75%",
                             "tracked_process": extra_loops["meta_loop"],
                             "lane": "runs/autonomy/learnings/meta.jsonl",
                             "note": "REAL bounded tracked process (meta_loop.py) -- watcher + Stop hook see it"},
            "3_project_evolution": {"substrate": "watcher.py (3h evolution-window) + SELF_EVOLUTION_LEDGER.md",
                                    "watcher_pid": watcher_pid or "already running",
                                    "ledger": "docs/SELF_EVOLUTION_LEDGER.md", "cadence": "every 3h (00/03/06/09 SAST)",
                                    "tracked_process": extra_loops["evolution_loop"],
                                    "lane": "runs/autonomy/learnings/evolution.jsonl",
                                    "note": "REAL bounded tracked process (evolution_loop.py, evolve_planner) -- "
                                            "watcher + Stop hook see it"},
        },
    }
    _atomic_write_json(MANIFEST, manifest)

    print("=" * 78)
    print("AUTONOMOUS MODE LAUNCHED -- 3 loops (the unavoidable gate)")
    print("=" * 78)
    print(f"objective : {a.objective}")
    print(f"success   : {a.success}")
    print(f"anchor    : {anchor}" + (f"   window_end: {window_end}" if window_end else "   (open-ended)"))
    print(f"mode      : {a.mode}  (spawn metaop loops = {spawn})")
    print(f"brain     : {brain['model']}  ({brain['why']})")
    print("-" * 78)
    print(f"[1] PROBLEM-SOLVER (expert+plain): {'metaop spawned' if spawn else 'OVERSEER->Agent dispatch'} "
          f"| frontier {FRONTIER} ({n_nodes} nodes pending)")
    if spawn:
        for k in ("problem_solver_expert", "problem_solver_plain"):
            print(f"      {k}: {loops[k]}")
    print(f"[2] META AGENT: {'metaop meta spawned' if spawn else 'OVERSEER reflect (attended)'} "
          f"| tracked meta_loop: {extra_loops['meta_loop']}")
    if spawn:
        print(f"      {loops['meta_agent']}")
    print(f"[3] PROJECT EVOLUTION (3h): watcher {'spawned pid '+str(watcher_pid) if watcher_pid else 'already running'} "
          f"| ledger docs/SELF_EVOLUTION_LEDGER.md | tracked evolution_loop: {extra_loops['evolution_loop']}")
    print("-" * 78)
    print("OVERSEER next: dispatch loop-1 nodes -> Agent workers, judge RWYB, commit, re-rank (loop-2), "
          "honor the 3h gate (loop-3). Resume: python scripts/autonomy/launch_autonomy.py --status")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
