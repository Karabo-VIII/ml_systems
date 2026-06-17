#!/usr/bin/env python3
"""Loop LIVENESS checker -- counts metaop loops whose PROCESS is actually alive (not just whose lock exists).

Gap fix (2026-06-06): the watcher checked lock-existence, which a CRASHED loop fools (it leaves a stale lock).
This checks the lock's PID against the OS, so a crash is detected -> the orchestrator relaunches. Reusable.

Usage: python scripts/autonomy/loops_alive.py sol-ma meta-ma   ->  prints "<n_alive> alive | dead: [...]"
Exit code = number of DEAD/missing threads (0 = all alive), so callers can `&&`/`||` on it.
"""
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from proc_liveness import alive as _alive  # G-J 2026-06-07: the ONE create-time-aware liveness check (no PID reuse)


def main():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    threads = sys.argv[1:]
    locks_dir = os.path.join(root, "runs", "autonomy", "locks")
    if not threads:
        # No args -> auto-discover every registered lock. (Gap fix 2026-06-06: a bare call
        # used to print "0 alive | dead: none" and silently mislead the watcher.)
        try:
            threads = sorted(f[:-5] for f in os.listdir(locks_dir) if f.endswith(".lock"))
        except FileNotFoundError:
            threads = []
    n_alive, dead = 0, []
    for t in threads:
        lf = os.path.join(root, "runs", "autonomy", "locks", f"{t}.lock")
        try:
            _d = json.load(open(lf, encoding="utf-8"))
            if _alive(_d.get("pid"), _d.get("created")):
                n_alive += 1
            else:
                dead.append(t)
        except Exception:
            dead.append(t)
    print(f"{n_alive} alive | dead: {dead if dead else 'none'}")
    return len(dead)


if __name__ == "__main__":
    raise SystemExit(main())
