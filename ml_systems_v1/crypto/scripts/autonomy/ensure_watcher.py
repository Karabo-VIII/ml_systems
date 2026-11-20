#!/usr/bin/env python3
"""ensure_watcher.py -- the UNSKIPPABLE 60s-watcher gate.

User mandate 2026-06-06: "where is the 60s loop -- add it to this instance and all instances, especially in autonomous
mode -- no skipping that gate." Called by BOTH autonomous hooks (autonomous_mode_check on UserPromptSubmit, autonomy_loop
on Stop), so EVERY turn and EVERY loop continuation re-ensures it: if autonomous mode is armed and watcher.py is not
running, relaunch it FULLY DETACHED (survives the caller exiting -- the fix for the flaky self-respawn). This makes
"the 60s watcher is absolute" MECHANICAL: no instance can be in autonomous mode without the watcher alive.

Idempotent + fast (it is on the hook hot-path). No emoji (cp1252).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUT = os.path.join(ROOT, "runs", "autonomy")
SWITCH = os.path.join(AUT, "AUTONOMY_ON")
AUTO_MODE = os.path.join(ROOT, ".claude", "autonomous_mode.json")
WATCHER = os.path.join(ROOT, "scripts", "autonomy", "watcher.py")
LOG = os.path.join(AUT, "watcher.log")
SPAWN_LOCK = os.path.join(AUT, "watcher_spawn.lock")  # P3a: atomic guard so two ensure() callers can't BOTH spawn
SPAWN_LOCK_TTL = 30  # seconds: a spawn-lock older than this is stale (the spawner died before clearing it)


def _armed() -> bool:
    """Autonomous mode armed via the prompt-free switch OR the W3 authority file (autonomous:true)."""
    if os.path.exists(SWITCH):
        return True
    try:
        return bool(json.load(open(AUTO_MODE, encoding="utf-8")).get("autonomous"))
    except Exception:
        return False


def _running() -> bool:
    """True iff a watcher.py PROCESS is actually alive (psutil authoritative; log-freshness is a weak fallback)."""
    try:
        import psutil
        for p in psutil.process_iter(["cmdline"]):
            try:
                # BASENAME match -- substring "watcher.py" wrongly matches "ensure_watcher.py" (this very gate),
                # a false-positive that made the gate think the watcher was alive + never relaunch. (fix 2026-06-06)
                if any(os.path.basename(str(c)).lower() == "watcher.py" for c in (p.info.get("cmdline") or [])):
                    return True
            except Exception:
                continue
        return False  # psutil present + no match = authoritatively down
    except Exception:
        pass
    try:
        return (time.time() - os.path.getmtime(LOG)) < 90  # fallback: a tick within the last 90s
    except Exception:
        return False


def _acquire_spawn_lock() -> bool:
    """P3a: race-proof single-spawner guard. Two ensure() callers (e.g. the UserPromptSubmit hook AND the Stop
    hook firing close together) can BOTH see _running()==False and BOTH spawn -> duplicate watchers racing the same
    log/flags. The OS lets only ONE process atomically O_EXCL-create this lock, so only one wins the spawn. A stale
    lock (older than SPAWN_LOCK_TTL -- the spawner crashed before clearing it) is reclaimed. Returns True if WE own
    the spawn this call."""
    os.makedirs(AUT, exist_ok=True)
    try:
        fd = os.open(SPAWN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
        finally:
            os.close(fd)
        return True
    except FileExistsError:
        try:
            if (time.time() - os.path.getmtime(SPAWN_LOCK)) > SPAWN_LOCK_TTL:
                os.unlink(SPAWN_LOCK)  # stale -> reclaim, retry once
                fd = os.open(SPAWN_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
        except OSError:
            pass
        return False  # another ensure() is mid-spawn -> do NOT spawn a duplicate


def _release_spawn_lock():
    try:
        os.unlink(SPAWN_LOCK)
    except OSError:
        pass


def ensure() -> str:
    """If armed and the watcher is down, relaunch it fully detached. Returns a one-word status.
    P3a: guarded by an atomic spawn-lock so concurrent callers cannot launch duplicate watchers."""
    if not _armed():
        return "not-armed"
    if _running():
        return "alive"
    # P3a: only ONE caller may spawn at a time. If we don't win the lock, another ensure() is already spawning.
    if not _acquire_spawn_lock():
        return "spawn-in-progress"
    try:
        if _running():  # double-checked under the lock: a watcher may have come up since the first check
            return "alive"
        # WINDOWLESS spawn: use pythonw.exe (no console at all) so no terminal window flashes; CREATE_NO_WINDOW as a
        # belt-and-braces. DETACHED_PROCESS is REMOVED -- it is MUTUALLY EXCLUSIVE with CREATE_NO_WINDOW on Windows,
        # and combining them is what made a console window appear externally. (fix 2026-06-06: user saw the window.)
        interp = sys.executable
        if os.name == "nt":
            pyw = sys.executable.lower().replace("python.exe", "pythonw.exe")
            if os.path.exists(pyw):
                interp = pyw
        try:
            kwargs = dict(cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          close_fds=True)
            if os.name == "nt":
                kwargs["creationflags"] = 0x08000000 | 0x00000200  # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen([interp, WATCHER, "--max-ticks", "999"], **kwargs)
            return "relaunched"
        except Exception as e:
            return f"relaunch-failed: {e}"
    finally:
        _release_spawn_lock()


if __name__ == "__main__":
    print(f"[ensure_watcher] {ensure()}")
