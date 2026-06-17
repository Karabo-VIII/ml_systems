#!/usr/bin/env python3
"""Make ALL .claude/settings.json hook commands cwd-INDEPENDENT (absolute paths).

GLOBAL FIX for the cwd-lockout class: every PreToolUse/Stop/UserPromptSubmit/PostToolUse hook command was a
cwd-RELATIVE `python <relpath>`. When the shell cwd wedges (e.g. a top-level `cd` into a subdir), python cannot
find the script -> exit 2 -> Claude Code BLOCKS the tool -> ALL tools lock out until a restart. Resolving each
hook script via an ABSOLUTE path makes a wedged cwd completely harmless.

settings.json is gitignored (machine-local) -> an absolute machine path is appropriate AND there is no git
revert-net, so we BACK UP first. Idempotent: already-absolute commands are left untouched. No emoji (cp1252).

Run: python scripts/autonomy/patch_hook_paths.py        (apply)
     python scripts/autonomy/patch_hook_paths.py --check (report only, no write)
"""
import json
import os
import sys
from datetime import datetime

ROOT = "C:/Users/karab/Documents/coding/v4_crypto_stystem"
SETTINGS = os.path.join(ROOT, ".claude", "settings.json")
BACKUP_DIR = os.path.join(ROOT, "runs", "staging")

# relative-command -> absolute-command  (forward slashes: python accepts them on Windows; no spaces in ROOT)
RELS = [
    "python .claude/hooks/permission_gate.py",
    "python scripts/hooks/user_prompt_router.py",
    "python scripts/autonomous_mode_check.py",
    "python .claude/hooks/autonomy_loop.py",
    "python .claude/hooks/meta_change_guard.py",
]


def to_abs(cmd):
    """Map a known relative hook command to its absolute form. Idempotent / unknown -> unchanged."""
    for rel in RELS:
        if cmd.strip() == rel:
            relpath = rel[len("python "):]
            return f'python "{ROOT}/{relpath}"'
    return cmd


def walk_hooks(settings):
    changed = []
    hooks = settings.get("hooks", {})
    for event, groups in hooks.items():
        for gi, group in enumerate(groups):
            for hi, h in enumerate(group.get("hooks", [])):
                if h.get("type") == "command":
                    old = h.get("command", "")
                    new = to_abs(old)
                    if new != old:
                        h["command"] = new
                        changed.append((event, old, new))
    return changed


def main():
    check_only = "--check" in sys.argv
    with open(SETTINGS, encoding="utf-8") as fh:
        raw = fh.read()
    settings = json.loads(raw)  # validate input

    # preview on a deep copy so --check never mutates
    preview = json.loads(raw)
    changes = walk_hooks(preview)

    print(f"[patch_hook_paths] {SETTINGS}")
    if not changes:
        print("  no relative hook commands found -> already absolute (idempotent no-op). DONE.")
        return 0
    for event, old, new in changes:
        print(f"  [{event}] {old}")
        print(f"        -> {new}")

    if check_only:
        print(f"\n--check: {len(changes)} command(s) WOULD change. No write.")
        return 0

    # back up the (gitignored) original BEFORE writing -- this is the revert net
    os.makedirs(BACKUP_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = os.path.join(BACKUP_DIR, f"settings.json.prehookfix.{stamp}.bak")
    with open(backup, "w", encoding="utf-8") as fh:
        fh.write(raw)
    print(f"\n  backup -> {backup}")

    # apply for real
    walk_hooks(settings)
    out = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    json.loads(out)  # re-validate before writing
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        fh.write(out)

    # verify round-trip from disk
    with open(SETTINGS, encoding="utf-8") as fh:
        back = json.load(fh)
    remaining = walk_hooks(json.loads(json.dumps(back)))
    print(f"  wrote {SETTINGS}; applied {len(changes)} change(s); residual-relative={len(remaining)}")
    print("  DONE (activates on next session restart; this session unaffected -- hooks load at start).")
    return 0 if not remaining else 2


if __name__ == "__main__":
    sys.exit(main())
