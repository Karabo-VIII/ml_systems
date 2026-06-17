#!/usr/bin/env python3
"""track_job.py -- the CANONICAL long-job tracker for the autonomy loop (closes audit P3b).

THE CONTRACT (mandatory): EVERY long-running / detached background job the loop launches (a training run, a
backtest sweep, a metaop loop, any Popen the overseer expects to outlive a single turn) MUST register a lock
here so the two consumers can SEE it:
  - the WATCHER (scripts/autonomy/watcher.py) -- monitors PROCESS liveness + flags death.
  - the STOP HOOK (.claude/hooks/autonomy_loop.py) -- before releasing the spin, it checks for a TRACKED LIVE
    JOB; if one is alive while the envelope is open it WAITS (does not silently die mid-window). Without a lock,
    a detached Popen is INVISIBLE to the hook -> the run can SILENTLY DIE mid-window (the P0 bug).

Lock format (compatible with metaop/manager.py + loops_alive.py + loop_health.py, which all read {"pid": ...}):
  runs/autonomy/locks/<id>.lock  ==  {"pid": <int>, "ts": <epoch>, "thread": "<id>", "cmd": "<desc>",
                                      "kind": "job", "started": "<iso>"}

API (import OR CLI):
  from track_job import track, untrack, alive_jobs, any_job_alive
  track("chess_long_train", pid=1234, cmd="az/train_robust.py --max-hours 5")   # register (pid defaults to os.getpid)
  untrack("chess_long_train")                                                    # on clean completion
  alive_jobs()   -> [{"id","pid","cmd",...}, ...]  (only locks whose PID is alive; reaps dead ones)
  any_job_alive() -> bool

CLI:
  python scripts/autonomy/track_job.py add  <id> [--pid N] [--cmd "..."]
  python scripts/autonomy/track_job.py rm   <id>
  python scripts/autonomy/track_job.py list                 # prints alive tracked jobs (reaps dead)
  python scripts/autonomy/track_job.py run  <id> -- <command...>   # spawn detached + auto-track + auto-untrack-on-exit

No emoji (Windows cp1252). Race-proof via atomic O_EXCL create (a re-track of a live id is rejected).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOCKS = os.path.join(ROOT, "runs", "autonomy", "locks")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so `import proc_liveness` works as script OR import
from proc_liveness import alive as _alive_pid, create_time as _create_time  # the ONE liveness check (G-J 2026-06-07)


def _alive(pid, created=None) -> bool:
    """Back-compat shim over the shared create-time-aware liveness check (proc_liveness). When `created` (the
    create-time captured when the lock was written) is supplied, a RECYCLED PID is rejected -- this closes the
    PID-reuse stuck-path (a stale lock whose pid got reassigned used to read as ALIVE -> eternal WAIT-MODE). Without
    `created`, a bare check (legacy behavior). All consumers share this logic now, so the old 6 duplicated ctypes
    blocks cannot drift apart."""
    return _alive_pid(pid, created)


def _lock_path(job_id: str) -> str:
    return os.path.join(LOCKS, f"{job_id}.lock")


def track(job_id: str, pid: int | None = None, cmd: str = "", kind: str = "job") -> tuple[bool, str]:
    """Register a long job. pid defaults to the current process. If a lock with this id already exists AND its
    owner PID is alive, the call is REJECTED (don't clobber a live tracked job). A stale (dead-owner) lock is
    reclaimed. Returns (ok, message)."""
    os.makedirs(LOCKS, exist_ok=True)
    if pid is None:
        pid = os.getpid()
    payload = json.dumps({
        "pid": int(pid), "ts": int(time.time()), "thread": job_id, "cmd": cmd, "kind": kind,
        "started": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "created": _create_time(pid),  # G-J: process create-time -> a reused PID won't match -> not falsely "alive"
    }).encode("utf-8")
    lf = _lock_path(job_id)
    for _ in range(2):
        try:
            fd = os.open(lf, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # atomic: fails if it already exists
            try:
                os.write(fd, payload)
            finally:
                os.close(fd)
            return True, f"tracked {job_id} pid={pid}"
        except FileExistsError:
            try:
                info = json.loads(open(lf, encoding="utf-8").read())
            except Exception:
                info = {}
            if _alive(info.get("pid", -1), info.get("created")):
                return False, f"job '{job_id}' already tracked by live PID {info.get('pid')}"
            try:
                os.unlink(lf)  # stale (dead owner) -> reclaim and retry the atomic create
            except FileNotFoundError:
                pass
    return False, f"job '{job_id}' lock contended -- try again"


def untrack(job_id: str) -> bool:
    try:
        os.unlink(_lock_path(job_id))
        return True
    except FileNotFoundError:
        return False


def alive_jobs() -> list:
    """Return [{"id","pid","cmd","kind",...}] for every lock whose owner PID is ALIVE. Reaps locks whose owner
    is dead (self-healing) so a crashed job does not falsely keep the loop in WAIT-MODE forever."""
    out = []
    try:
        names = [f for f in os.listdir(LOCKS) if f.endswith(".lock")]
    except FileNotFoundError:
        return out
    for f in names:
        lf = os.path.join(LOCKS, f)
        try:
            info = json.loads(open(lf, encoding="utf-8").read())
        except Exception:
            info = {}
        pid = info.get("pid", -1)
        if _alive(pid, info.get("created")):
            out.append({"id": f[:-5], "pid": pid, "cmd": info.get("cmd", ""),
                        "kind": info.get("kind", ""), "started": info.get("started", "")})
        else:
            try:
                os.unlink(lf)  # reap dead-owner lock
            except OSError:
                pass
    return out


def any_job_alive() -> bool:
    return bool(alive_jobs())


def _cmd_run(job_id: str, command: list) -> int:
    """Spawn `command` DETACHED, auto-track it under job_id, print the pid. The spawned process owns the lock;
    we register it AFTER spawn with the child PID. (We do NOT block on it -- that is the whole point.)"""
    if os.name == "nt":
        flags = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
        kw = dict(creationflags=flags)
    else:
        kw = dict(start_new_session=True)
    p = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    ok, msg = track(job_id, pid=p.pid, cmd=" ".join(command))
    print(f"[track_job] spawned pid={p.pid} :: {msg}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="canonical long-job tracker for the autonomy loop")
    sub = ap.add_subparsers(dest="action", required=True)
    a = sub.add_parser("add"); a.add_argument("id"); a.add_argument("--pid", type=int, default=None); a.add_argument("--cmd", default="")
    r = sub.add_parser("rm"); r.add_argument("id")
    sub.add_parser("list")
    rn = sub.add_parser("run"); rn.add_argument("id"); rn.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if args.action == "add":
        ok, msg = track(args.id, pid=args.pid, cmd=args.cmd)
        print(f"[track_job] {msg}")
        return 0 if ok else 1
    if args.action == "rm":
        print(f"[track_job] {'removed' if untrack(args.id) else 'not-found'} {args.id}")
        return 0
    if args.action == "list":
        jobs = alive_jobs()
        if not jobs:
            print("[track_job] no live tracked jobs")
        for j in jobs:
            print(f"[track_job] ALIVE id={j['id']} pid={j['pid']} started={j['started']} cmd={j['cmd']}")
        return 0
    if args.action == "run":
        command = args.command
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            print("[track_job] run: no command given"); return 2
        return _cmd_run(args.id, command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
