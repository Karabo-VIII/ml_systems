#!/usr/bin/env python3
"""proc_liveness.py -- the ONE process-liveness check for the autonomy stack (closes the PID-reuse stuck-path).

WHY (2026-06-07 audit G-J): bare PID checks (OpenProcess / os.kill 0) return True for ANY process holding that PID.
On Windows PIDs recycle fast (observed live: pid 23932 was the chess train_robust job, later reassigned to
AppVShNotify). A stale lock whose PID got reused then reads as "job ALIVE" -> the Stop hook stays in WAIT-MODE
forever = a NEW stuck-path (the exact failure class we keep closing). FIX: pin liveness to (pid, create_time). A
reused PID has a DIFFERENT create_time -> detected as dead. This module is the single source every consumer imports
(track_job, watcher, loops_alive, loop_health, launch_autonomy, metaop.manager) so the 6 duplicated bare _alive
blocks can't drift apart. No emoji (Windows cp1252).
"""
from __future__ import annotations

import os

try:
    import psutil  # 7.x present in this env; cross-platform create_time + pid_exists
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False


def create_time(pid) -> float | None:
    """Process create-time (epoch seconds), or None if unknown/dead/psutil-absent. Capture this when you WRITE a
    job lock, store it, and pass it back to alive() so a recycled PID is rejected."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or not _HAVE_PSUTIL:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _bare_alive(pid) -> bool:
    """True iff SOME process holds this PID right now (no identity check). Prefer psutil; fall back to the OS
    primitive used by the legacy consumers (OpenProcess on Windows / kill(0) on POSIX) so behavior is unchanged
    when psutil is absent."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if _HAVE_PSUTIL:
        try:
            return bool(psutil.pid_exists(pid))
        except Exception:
            pass
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def alive(pid, created=None, tol: float = 2.0) -> bool:
    """True iff a process with this PID is running AND (when `created` is given) its create-time matches `created`
    within `tol` seconds. `created` is the create_time captured when the lock was written; a mismatch means the PID
    was RECYCLED to a different process -> treat as DEAD (the PID-reuse stuck-path fix). Without `created` (or with
    psutil absent) this degrades to a bare check -- no worse than the legacy behavior, never falsely killing a live
    job."""
    if not _bare_alive(pid):
        return False
    if created in (None, "", 0):
        return True
    ct = create_time(pid)
    if ct is None:
        return True  # cannot verify (psutil gone) -> do NOT falsely kill; bare check already passed
    try:
        return abs(ct - float(created)) <= tol
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    me = os.getpid()
    ct = create_time(me)
    print(f"self pid={me} create_time={ct}")
    print("alive(self)           :", alive(me))
    print("alive(self, ct)       :", alive(me, ct), "(expect True)")
    print("alive(self, ct+9999)  :", alive(me, (ct or 0) + 9999), "(expect False -> reuse detected)")
    print("alive(999999)         :", alive(999999), "(expect False)")
