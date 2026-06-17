#!/usr/bin/env python3
"""PreToolUse hook (H1) -- the mechanical frankenstein guard.

Denies any Edit/Write/NotebookEdit (or mutating Bash) that targets a brain file
unless the session has BRAIN_EDIT_APPROVED=1. Makes sandbox->review->push physically
enforced: an autonomous run cannot rewrite its own constitution.

Contract (Claude Code hooks): reads tool-call JSON on stdin; exit 2 = block (stderr is
fed back to Claude as the reason); exit 0 = allow. Fail-OPEN on any error so the hook
can never brick a session.

PROPOSED / STAGED. Install only after review (copy to .claude/hooks/, wire in settings.json).
No emoji anywhere (Windows cp1252).
"""
import json
import os
import re
import sys

BRAIN_PATS = [
    r"(^|/)CLAUDE\.md$",
    r"(^|/)STATE\.md$",
    r"(^|/)\.claude/skills/",
    r"(^|/)\.claude/settings[^/]*\.json$",
    r"(^|/)\.claude/hooks/",
]
MUTATING_BASH = (">", ">>", "sed -i", "tee ", "rm ", "mv ", "cp ", "Set-Content", "Out-File")


def is_brain(path):
    if not path:
        return False
    p = path.replace("\\", "/")
    return any(re.search(pat, p) for pat in BRAIN_PATS)


def main():
    # explicit, reviewed override
    if os.environ.get("BRAIN_EDIT_APPROVED") == "1":
        sys.exit(0)
    try:
        sys.stdin.reconfigure(encoding="utf-8")  # Windows defaults to cp1252; brain paths may be non-ASCII
    except Exception:
        pass
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # fail-open (never brick a session)

    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}
    hits = []

    if tool in ("Edit", "Write", "NotebookEdit"):
        fp = ti.get("file_path") or ti.get("notebook_path") or ""
        if is_brain(fp):
            hits.append(fp)
    elif tool == "Bash":
        cmd = ti.get("command", "") or ""
        if any(m in cmd for m in MUTATING_BASH):
            for tok in re.findall(r"[^\s'\"]+", cmd):
                if is_brain(tok):
                    hits.append(tok)

    if hits:
        sys.stderr.write(
            "BRAIN-GUARD: blocked a write to a brain file (%s). Per the Brain-Upgrade "
            "Guardrails, changes to CLAUDE.md / .claude/skills / settings / hooks go "
            "sandbox->review->push -- stage the change in runs/staging/ instead. To edit "
            "the brain in an explicitly reviewed session, set BRAIN_EDIT_APPROVED=1."
            % ", ".join(sorted(set(hits)))
        )
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
