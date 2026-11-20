"""
CDAP pre-commit hook installer.

Drops .git/hooks/pre-commit that runs check_invariants.py before every
commit. Critical findings (exit 2) HALT the commit; warnings (exit 1)
are logged but don't block.

Usage:
    python src/audit/install_hook.py            # install
    python src/audit/install_hook.py --force    # overwrite existing
    python src/audit/install_hook.py --uninstall

Idempotent. Safe to re-run.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HOOK_PATH = PROJECT_ROOT / ".git" / "hooks" / "pre-commit"
HOOK_BACKUP = PROJECT_ROOT / ".git" / "hooks" / "pre-commit.cdap-replaced"

HOOK_SCRIPT = """#!/bin/sh
# CDAP + large-file pre-commit hook -- installed by src/audit/install_hook.py
#
# Two checks run in order:
#   1. check_no_large_files.py -- blocks files > 5MB or banned extensions
#      (parquet, pt, pkl, etc.). Bypass: SKIP_LARGE_FILES_CHECK=1
#   2. check_invariants.py -- CDAP global invariant audit
#      Bypass: SKIP_CDAP=1
#
# Either check returning rc=2 HALTS the commit. rc=1 (warn-only) allows.
#
# Provenance: large-file guard added 2026-04-30 after discovering
# 119 GB of historical chimera parquet bloat in .git/objects.

# Locate python -- prefer .venv, fall back to PATH
if [ -x .venv/Scripts/python.exe ]; then
    PYTHON=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON=python
fi

# ---- 1. Large-file guard (fastest; runs first) ----
if [ "$SKIP_LARGE_FILES_CHECK" != "1" ]; then
    "$PYTHON" src/audit/check_no_large_files.py
    RC=$?
    if [ $RC -eq 2 ]; then
        echo "" >&2
        echo "===============================================================" >&2
        echo "  LARGE/BINARY FILE STAGED -- COMMIT BLOCKED" >&2
        echo "  See src/audit/check_no_large_files.py for fix steps." >&2
        echo "===============================================================" >&2
        exit 1
    fi
else
    echo "[large-file-guard] skipped via SKIP_LARGE_FILES_CHECK=1" >&2
fi

# ---- 2. CDAP invariant audit (HARDENED 2026-05-25 trust-stack item #2) ----
if [ "$SKIP_CDAP" = "1" ]; then
    REASON_LEN=$(printf '%s' "$SKIP_CDAP_REASON" | wc -c | tr -d ' ')
    if [ -z "$SKIP_CDAP_REASON" ] || [ "$REASON_LEN" -lt 20 ]; then
        echo "" >&2
        echo "===============================================================" >&2
        echo "  CDAP SKIP REQUESTED BUT JUSTIFICATION MISSING/TOO SHORT" >&2
        echo "  Required: SKIP_CDAP_REASON='<at least 20 chars explanation>'" >&2
        echo "  Got: REASON_LEN=$REASON_LEN" >&2
        echo "  Trust-stack item #2 (2026-05-25): every bypass must be visible in git log." >&2
        echo "===============================================================" >&2
        exit 1
    fi
    if [ -f .git/COMMIT_EDITMSG ]; then
        printf '\n\n[CDAP-BYPASS] SKIP_CDAP=1 reason: %s\n' "$SKIP_CDAP_REASON" >> .git/COMMIT_EDITMSG
    fi
    echo "[CDAP] skipped via SKIP_CDAP=1; reason recorded in commit footer: $SKIP_CDAP_REASON" >&2
    exit 0
fi

"$PYTHON" src/audit/check_invariants.py --quiet
RC=$?

if [ $RC -eq 2 ]; then
    echo "" >&2
    echo "===============================================================" >&2
    echo "  CDAP CRITICAL DRIFT DETECTED -- COMMIT BLOCKED" >&2
    echo "  Fix the findings above, or pass:" >&2
    echo "    SKIP_CDAP=1 SKIP_CDAP_REASON='<>=20 chars explanation>'" >&2
    echo "===============================================================" >&2
    exit 1
fi

# rc 0 (clean) or 1 (warn-only) -> allow commit
exit 0
"""


def install(force: bool = False) -> int:
    if not (PROJECT_ROOT / ".git").exists():
        print("[install_hook] not a git repo (no .git/) at project root", file=sys.stderr)
        return 1
    HOOK_PATH.parent.mkdir(parents=True, exist_ok=True)

    if HOOK_PATH.exists() and not force:
        # Detect whether it's already our hook
        existing = HOOK_PATH.read_text(errors="replace")
        if "CDAP pre-commit hook" in existing:
            print(f"[install_hook] CDAP hook already installed at {HOOK_PATH}")
            return 0
        else:
            # Back it up so we don't clobber
            shutil.copy2(HOOK_PATH, HOOK_BACKUP)
            print(f"[install_hook] backed up existing hook -> {HOOK_BACKUP}")

    HOOK_PATH.write_text(HOOK_SCRIPT, encoding="utf-8")
    # chmod +x (Windows ignores; Linux/Mac requires)
    try:
        st = os.stat(HOOK_PATH)
        os.chmod(HOOK_PATH, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    except Exception:
        pass

    print(f"[install_hook] installed CDAP pre-commit hook -> {HOOK_PATH.relative_to(PROJECT_ROOT)}")
    print("[install_hook] runs `python src/audit/check_invariants.py` before every commit")
    print("[install_hook] bypass once via SKIP_CDAP=1 (discouraged)")
    return 0


def uninstall() -> int:
    if not HOOK_PATH.exists():
        print("[install_hook] no hook to remove")
        return 0
    text = HOOK_PATH.read_text(errors="replace")
    if "CDAP pre-commit hook" not in text:
        print(f"[install_hook] {HOOK_PATH} is NOT the CDAP hook; refusing to delete")
        return 1
    HOOK_PATH.unlink()
    if HOOK_BACKUP.exists():
        shutil.move(str(HOOK_BACKUP), str(HOOK_PATH))
        print(f"[install_hook] restored prior hook from backup")
    else:
        print("[install_hook] removed CDAP hook")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing hook")
    ap.add_argument("--uninstall", action="store_true", help="remove hook")
    args = ap.parse_args()
    if args.uninstall:
        return uninstall()
    return install(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
