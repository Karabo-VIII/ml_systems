"""PreToolUse FENCE for the CLI worker (`claude -p`).

Under `--permission-mode bypassPermissions`, Claude Code IGNORES allow/deny permission rules -- only PreToolUse
HOOKS can still block a tool call. So the no-key CLI worker (CliBrain) was UNFENCED at the CLI layer while the SDK
brain (AgentSdkBrain) had a Bash hook (2026-07-03 re-audit). This module closes that gap: CliBrain wires it as a
PreToolUse hook via `--settings`, and it BLOCKS a destructive Bash op using the SAME HARD_DENY fence as tools.py.

Protocol: Claude Code runs the hook with the tool-call JSON on stdin; a PreToolUse hook that exits with code 2
BLOCKS the call and feeds stderr back to the model. We use that (version-robust) and also emit the JSON decision.

Standalone + testable:
    echo '{"tool_name":"Bash","tool_input":{"command":"rm -rf /"}}' | python -m metaop._cli_fence   # -> exit 2
No emoji (Windows cp1252).
"""
from __future__ import annotations

import json
import sys


def decide(data: dict) -> tuple[bool, str]:
    """Return (block, reason) for a PreToolUse hook payload. Blocks a destructive Bash command (HARD_DENY) and a
    Write/Edit to a protected control-surface file (HARD_FILE_DENY). Never raises."""
    try:
        from .tools import HARD_DENY, HARD_FILE_DENY, _blocked
    except ImportError:  # pragma: no cover -- when run as a loose script with harness/ on the path
        from metaop.tools import HARD_DENY, HARD_FILE_DENY, _blocked
    name = data.get("tool_name") or data.get("toolName") or ""
    ti = data.get("tool_input") or data.get("toolInput") or {}
    if name.lower() == "bash":
        hit = _blocked(ti.get("command", "") or "", HARD_DENY)
        if hit:
            return True, f"harness CLI fence: destructive Bash op DENIED (/{hit}/) -- stage your work; a human reviews + commits."
    if name in ("Write", "Edit", "MultiEdit"):
        hit = _blocked(ti.get("file_path", "") or "", HARD_FILE_DENY)
        if hit:
            return True, f"harness CLI fence: writing a protected control-surface file DENIED (/{hit}/)."
    return False, ""


def main() -> int:
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except Exception:
        data = {}
    block, reason = decide(data)
    if block:
        # JSON decision (advanced protocol) + exit 2 (simple protocol) -- either one blocks across CLI versions.
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                                  "permissionDecisionReason": reason}}))
        sys.stderr.write(reason + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
