"""
chess_zero / watchdog.py -- OUT-OF-PROCESS liveness watchdog (audit S6).

WHY: train_robust's in-process supervise() only catches EXCEPTIONS. A deadlock, a CUDA stream
stall, a hung multiprocess pool, a segfault, or an external SIGKILL leaves the trainer 'alive'
(or dead) with GPU at ~0% and NO exception -- the in-process supervisor never fires and an
unattended multi-day run silently makes zero progress. This watchdog runs in a SEPARATE process,
polls the trainer's heartbeat file (written each phase by train_robust.write_heartbeat), and
TREE-KILLS + RESTARTS the trainer when the heartbeat goes stale beyond --max-stall-s. Because it
watches from outside, it survives a hang/segfault/kill of the trainer it is guarding.

USAGE (everything after `--` is the trainer command, launched + guarded as a subprocess):
    python projects/chess_zero/watchdog.py --ckpt-dir robust_dual --max-stall-s 900 --max-hours 8 \
        -- python -m az.train_robust --ckpt-dir robust_dual --supervise ...

The trainer's own --supervise still handles clean exception-restarts FAST; this watchdog is the
outer belt-and-suspenders for the no-exception hang. Heartbeat path mirrors train_robust:
projects/chess_zero/az/<ckpt-dir>/heartbeat.json. No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))          # selfplay/
_REPO_ROOT = os.path.dirname(_HERE)                         # games-engine root (holds az/)
_AZ = os.path.join(_REPO_ROOT, "az")


def _resolve_exe(cmd):
    """Resolve cmd[0] to an ABSOLUTE executable path. On Windows, Popen(relative_exe, cwd=...)
    fails (WinError 2) because the relative exe is not resolved against the child cwd. A bare
    name ('python') goes through PATH (shutil.which); a path ('.venv/Scripts/python.exe') is made
    absolute relative to the repo root."""
    if not cmd:
        return cmd
    exe = cmd[0]
    if (os.sep in exe) or ("/" in exe):
        if not os.path.isabs(exe):
            exe = os.path.abspath(os.path.join(_REPO_ROOT, exe))
    else:
        exe = shutil.which(exe) or exe
    return [exe] + list(cmd[1:])


def _read_heartbeat(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _launch(cmd):
    """Launch the trainer in its OWN process group/session so a tree-kill reaches its children
    (the multiprocess self-play pool) and a Ctrl-C to the watchdog doesn't fracture the tree."""
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(_resolve_exe(cmd), cwd=_REPO_ROOT, **kw)


def _tree_kill(proc) -> None:
    """Kill the trainer AND its children (the spawn-pool workers). Best-effort; never raises."""
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Out-of-process liveness watchdog for train_robust.")
    ap.add_argument("--ckpt-dir", required=True,
                    help="trainer ckpt subdir under az/ (the watchdog reads az/<ckpt-dir>/heartbeat.json)")
    ap.add_argument("--max-stall-s", type=float, default=900.0,
                    help="restart the trainer if the heartbeat is older than this (default 900s = "
                         "15min; keep it well above the slowest single iter incl. the seed eval)")
    ap.add_argument("--max-hours", type=float, default=8.0,
                    help="global wall-clock envelope; stop guarding past it (default 8h)")
    ap.add_argument("--poll-s", type=float, default=30.0, help="heartbeat poll interval (default 30s)")
    ap.add_argument("--max-restarts", type=int, default=50, help="give up after this many restarts")
    ap.add_argument("--grace-s", type=float, default=0.0,
                    help="extra grace before the FIRST heartbeat must appear (0 = use max-stall-s)")
    ap.add_argument("trainer_cmd", nargs=argparse.REMAINDER,
                    help="the trainer command, after a literal `--`")
    args = ap.parse_args(argv)

    cmd = args.trainer_cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("[watchdog] ERROR: no trainer command given (put it after `--`).")
        return 2

    hb_path = os.path.join(_AZ, args.ckpt_dir, "heartbeat.json")
    grace = args.grace_s or args.max_stall_s
    global_start = time.time()
    restarts = 0

    print(f"[watchdog] guarding heartbeat {hb_path}")
    if not os.path.isdir(os.path.dirname(hb_path)):
        print(f"[watchdog] NOTE: {os.path.dirname(hb_path)} does not exist yet. The trainer creates "
              f"it at startup; if a heartbeat NEVER appears, --ckpt-dir almost certainly does NOT "
              f"match the trainer's ckpt-dir -- the watchdog would then watch the wrong heartbeat and "
              f"false-kill a HEALTHY trainer every --max-stall-s. Make them match.")
    print(f"[watchdog] max_stall={args.max_stall_s}s poll={args.poll_s}s "
          f"max_hours={args.max_hours} max_restarts={args.max_restarts}")
    print(f"[watchdog] launching: {' '.join(cmd)}")
    proc = _launch(cmd)
    launch_t = time.time()

    try:
        while True:
            elapsed_h = (time.time() - global_start) / 3600.0
            if elapsed_h >= args.max_hours:
                print(f"[watchdog] wall-clock envelope {args.max_hours}h spent -- stopping (tree-kill).")
                _tree_kill(proc)
                return 0

            rc = proc.poll()
            if rc is not None:
                if rc == 0:
                    print("[watchdog] trainer exited cleanly (rc=0) -- done.")
                    return 0
                restarts += 1
                print(f"[watchdog] trainer EXITED rc={rc} (crash) -- restart "
                      f"{restarts}/{args.max_restarts}")
                if restarts > args.max_restarts:
                    print("[watchdog] max restarts reached -- giving up.")
                    return 1
                proc = _launch(cmd)
                launch_t = time.time()
                time.sleep(args.poll_s)
                continue

            # trainer is ALIVE -> check heartbeat staleness
            hb = _read_heartbeat(hb_path)
            ref_t = hb["t"] if (hb and "t" in hb) else launch_t
            limit = args.max_stall_s if hb else grace
            stall = time.time() - ref_t
            if stall > limit:
                if hb:
                    where = f"iter {hb.get('iter')} phase {hb.get('phase')}"
                    print(f"[watchdog] HUNG: heartbeat stale {stall:.0f}s > {limit:.0f}s ({where}) "
                          f"-- tree-killing + restarting (the in-process supervisor could not see this).")
                else:
                    # NO heartbeat ever -> almost always a --ckpt-dir MISMATCH, not a real hang.
                    # Killing here would kill-restart-loop a healthy trainer (the 2026-06-09 incident).
                    print(f"[watchdog] NO heartbeat at {hb_path} after {stall:.0f}s. This is almost "
                          f"certainly a --ckpt-dir MISMATCH (the trainer writes its heartbeat to a "
                          f"DIFFERENT dir), NOT a hang. Restarting a healthy trainer would loop it "
                          f"forever, so the watchdog is NOT killing it -- FIX --ckpt-dir to match the "
                          f"trainer, then relaunch. (Set --grace-s if the trainer is genuinely slow "
                          f"to first-heartbeat.)")
                    time.sleep(args.poll_s)
                    continue
                _tree_kill(proc)
                restarts += 1
                if restarts > args.max_restarts:
                    print("[watchdog] max restarts reached -- giving up.")
                    return 1
                proc = _launch(cmd)
                launch_t = time.time()
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("[watchdog] interrupted -- tree-killing the trainer and exiting.")
        _tree_kill(proc)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
