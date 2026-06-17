#!/usr/bin/env python3
"""PostToolUse hook -- META-CHANGE VALIDATION GATE (closes weak point W2: a self-edit silently breaks the loop).

The self-evolving agent edits its own machinery (hooks, gates, the driver) WHILE that machinery is running. A
bad self-edit to e.g. autonomy_loop.py makes the Stop hook error -> fail-open -> the loop SILENTLY STOPS, with no
signal. This guard fires right AFTER an Edit/Write to a load-bearing meta file and validates it immediately:
py_compile + the component's own selftest (if known). If broken, it returns exit 2 so the harness feeds a LOUD
"you just broke X -- fix it now" message back to the model. Real-time "evolve correctly" enforcement.

Contract (Claude Code PostToolUse): stdin = {"tool_name","tool_input":{"file_path"...},...}. The tool already
ran (PostToolUse cannot un-write); exit 2 surfaces stderr to the model as a blocking-style error. Exit 0 =
silent OK. FAIL-SAFE: any error in the guard -> exit 0 (never trap the session, never false-alarm on its own bug).
No emoji (Windows cp1252).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Load-bearing meta files: a break here can disable the loop / gates / wiring. (prefix match, posix-normalized)
META_PREFIXES = (
    ".claude/hooks/",                          # shared hooks STAY at the parent root
    "crypto/scripts/autonomy/",                # crypto-owned after the 3-way split
    "crypto/scripts/autonomous_mode_check.py",
    "crypto/scripts/hooks/",
    "crypto/scripts/mandatory_gate.py",
    "crypto/src/audit/",
)
# Known selftests: file (suffix match) -> command (argv, cwd=ROOT=parent root). py_compile is always run first.
SELFTESTS = {
    ".claude/hooks/permission_gate.py":     [sys.executable, ".claude/hooks/test_permission_gate.py"],
    "crypto/scripts/mandatory_gate.py":     [sys.executable, "crypto/scripts/mandatory_gate.py"],
    "crypto/src/audit/check_dsr_holm.py":   [sys.executable, "crypto/src/audit/check_dsr_holm.py", "--selftest"],
    "crypto/src/strat/__init__.py":         [sys.executable, "crypto/src/strat/selftest_all.py"],
}


def _rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except Exception:
        return str(path).replace("\\", "/")


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    tool = data.get("tool_name", "")
    if tool not in ("Edit", "Write", "MultiEdit", "Update", "NotebookEdit"):
        sys.exit(0)
    fp = (data.get("tool_input", {}) or {}).get("file_path", "")
    if not fp:
        sys.exit(0)
    rel = _rel(fp)
    if not any(rel == p or rel.startswith(p) for p in META_PREFIXES):
        sys.exit(0)  # not a load-bearing meta file -> ignore
    if not rel.endswith(".py"):
        sys.exit(0)

    target = ROOT / rel
    if not target.exists():
        sys.exit(0)

    # 1. py_compile -- a syntax error in a hook = the hook silently no-ops
    r = subprocess.run([sys.executable, "-m", "py_compile", str(target)], capture_output=True, text=True,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    if r.returncode != 0:
        sys.stderr.write(
            f"META-CHANGE GUARD: {rel} FAILS py_compile after your edit -- it is BROKEN and the loop/gate that "
            f"depends on it will silently no-op. FIX IT NOW before relying on it.\n{(r.stderr or '').strip()[:600]}\n"
        )
        sys.exit(2)

    # 2. component selftest, if one is registered
    for suffix, cmd in SELFTESTS.items():
        if rel.endswith(suffix):
            try:
                t = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=120,
                                   creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            except Exception:
                break  # selftest harness itself failed to run -> don't false-alarm
            if t.returncode not in (0, 1):  # 0 ok, 1 = warn-only (tolerated); >=2 = real failure
                tail = ((t.stdout or "") + (t.stderr or "")).strip().splitlines()[-8:]
                sys.stderr.write(
                    f"META-CHANGE GUARD: {rel} compiled but its SELFTEST failed (exit {t.returncode}) -- your "
                    f"edit broke its behavior. FIX before relying on it.\n" + "\n".join(tail) + "\n"
                )
                sys.exit(2)
            break

    sys.exit(0)


if __name__ == "__main__":
    main()
