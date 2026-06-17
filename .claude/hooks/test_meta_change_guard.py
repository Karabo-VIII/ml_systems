#!/usr/bin/env python3
"""RWYB regression test for meta_change_guard.py (the PostToolUse meta-change validation gate).

Run: python .claude/hooks/test_meta_change_guard.py  (exit 0 = all pass).
Subprocesses the guard with crafted PostToolUse inputs. Creates a temp BROKEN meta file at runtime (so no broken
syntax is committed) and confirms the guard flags it (exit 2); confirms good/non-meta edits pass (exit 0).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / ".claude" / "hooks" / "meta_change_guard.py"
BROKEN = ROOT / ".claude" / "hooks" / "_mcg_broken_tmp.py"


def run(file_path):
    payload = {"tool_name": "Edit", "tool_input": {"file_path": str(file_path)}}
    r = subprocess.run([sys.executable, str(GUARD)], input=json.dumps(payload), capture_output=True, text=True,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    return r.returncode


def main():
    ok = True
    # 1. a load-bearing meta file with a SYNTAX ERROR -> guard must flag (exit 2)
    BROKEN.write_text("def broken(:\n    pass\n", encoding="utf-8")  # invalid syntax
    try:
        rc = run(BROKEN)
        good = rc == 2
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] broken meta hook -> exit {rc} (expect 2)")
    finally:
        try:
            BROKEN.unlink()
        except OSError:
            pass
        # py_compile may leave a __pycache__ entry; harmless

    # 2. a GOOD load-bearing meta file -> pass (exit 0)
    rc = run(ROOT / ".claude" / "hooks" / "autonomy_loop.py")
    good = rc == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] good meta hook (autonomy_loop.py) -> exit {rc} (expect 0)")

    # 3. a NON-meta file (a doc) -> ignored (exit 0)
    rc = run(ROOT / "docs" / "SYSTEM_TOPOLOGY.md")
    good = rc == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] non-meta file (a doc) -> exit {rc} (expect 0)")

    # 4. permission_gate.py (has a registered selftest) -> pass (compiles + 17/17)
    rc = run(ROOT / ".claude" / "hooks" / "permission_gate.py")
    good = rc == 0
    ok = ok and good
    print(f"  [{'PASS' if good else 'FAIL'}] permission_gate.py (compile+selftest) -> exit {rc} (expect 0)")

    print("ALL PASS" if ok else "*** SOME FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
