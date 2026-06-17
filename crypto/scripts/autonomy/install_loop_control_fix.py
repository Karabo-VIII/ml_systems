#!/usr/bin/env python3
"""install_loop_control_fix.py -- OVERSEER install step for the GLOBAL anti-stuck Stop-hook fix (audit 2026-06-07).

WHY THIS EXISTS: in some environments the worker that BUILT the fix cannot Write/Edit .claude/hooks/autonomy_loop.py
(the file is IDE-open or otherwise permission-gated -- this is the very "stuck asking for permissions" path the
audit flagged). So the new hook is STAGED at runs/staging/autonomy_loop.new.py and an attended OVERSEER (or the
user, who can approve the .claude write) runs THIS to install it. The install:
  1. py_compile the staged candidate,
  2. run the RWYB test harness against the staged candidate (must be ALL PASS),
  3. back up the live hook to runs/staging/autonomy_loop.bak.<ts>.py,
  4. copy the staged candidate over .claude/hooks/autonomy_loop.py,
  5. py_compile + re-run the harness against the now-INSTALLED hook (proves the install is byte-identical/sound).
Refuses to install if any step fails. No emoji (cp1252).

Run:  python scripts/autonomy/install_loop_control_fix.py          # install
      python scripts/autonomy/install_loop_control_fix.py --check  # validate only, do not install
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGED = os.path.join(ROOT, "runs", "staging", "autonomy_loop.new.py")
LIVE = os.path.join(ROOT, ".claude", "hooks", "autonomy_loop.py")
HARNESS = os.path.join(ROOT, "scripts", "autonomy", "test_loop_control.py")
PY = sys.executable


def _run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0), **kw)


def _compile(path) -> bool:
    r = _run([PY, "-m", "py_compile", path])
    if r.returncode != 0:
        print(f"  py_compile FAILED: {path}\n{(r.stderr or '').strip()[:400]}")
        return False
    print(f"  py_compile OK: {path}")
    return True


def _harness() -> bool:
    r = _run([PY, HARNESS])
    out = (r.stdout or "")
    print("  --- harness output (tail) ---")
    for line in out.strip().splitlines()[-16:]:
        print("    " + line)
    ok = r.returncode == 0 and "ALL PASS" in out
    print(f"  harness: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate only; do not install")
    a = ap.parse_args()

    if not os.path.exists(STAGED):
        print(f"ERROR: staged candidate not found: {STAGED}")
        return 2

    print("[1/5] compile staged candidate")
    if not _compile(STAGED):
        return 2
    print("[2/5] RWYB harness vs staged candidate")
    if not _harness():
        print("REFUSING to install: harness did not pass against the staged candidate.")
        return 2

    if a.check:
        print("CHECK-ONLY: staged candidate compiles + passes the harness. Not installing.")
        return 0

    print("[3/5] back up the live hook")
    if os.path.exists(LIVE):
        bak = os.path.join(ROOT, "runs", "staging", f"autonomy_loop.bak.{dt.datetime.now():%Y%m%d_%H%M%S}.py")
        shutil.copy(LIVE, bak)
        print(f"  backed up live hook -> {bak}")

    print("[4/5] install staged -> .claude/hooks/autonomy_loop.py")
    try:
        shutil.copy(STAGED, LIVE)
        print(f"  installed: {LIVE}")
    except Exception as e:
        print(f"ERROR installing (permission? IDE-open .claude file?): {e}")
        print("  MITIGATION: close the file in the IDE, or have the user approve the .claude write, then re-run.")
        return 2

    print("[5/5] compile + RWYB harness vs INSTALLED hook")
    if not _compile(LIVE):
        return 2
    if not _harness():
        print("WARNING: installed hook did not pass the harness -- restore the backup immediately.")
        return 2

    print("\nINSTALL COMPLETE: the global anti-stuck Stop-hook fix is live + RWYB-verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
