"""Operator TOOLS -- crypto-consumer SHIM over the canonical harness.metaop.Tools (G-A dedup 2026-06-07).

The executor (shell / python / read / write / list, each fenced + returning a structured dict) now lives ONCE in
harness/metaop/tools.py. The harness Tools accepts INJECTABLE extra deny-fences (extra_cmd_deny / extra_file_deny);
this shim injects the crypto specifics the live loop needs and was the historical behavior of this copy:

  HARD_DENY      -- a LOOP never commits/pushes/merges/rebases/tags/resets/cherry-picks (the OVERSEER commits after
                    review). Defense-in-depth, ALWAYS enforced regardless of the mutable permission policy.
  HARD_FILE_DENY -- a loop may write WORK but NEVER a CONTROL SURFACE (.claude/autonomous_mode, settings, hooks,
                    permission_policy.json, mandatory_gates.yaml, .env, .git).
  permission_policy.json -- the LIVE permission gate's cmd_deny_regex/file_deny_regex are also screened (so the loop
                    physically can't run what the gate forbids); fail-CLOSED to a built-in fence if unreadable.

The canonical harness HARD_DENY (rm -rf, force-push, dd, mkfs, fork-bomb, ...) is ALSO enforced -- the crypto Tools
gets the UNION. The build cwd is pinned to the repo ROOT (the live loop builds in the repo). No emoji (cp1252).
"""
from __future__ import annotations

import json
from pathlib import Path

from harness.metaop.tools import Tools as _HarnessTools, HARD_DENY as _HARNESS_HARD_DENY  # canonical engine

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "runs" / "autonomy" / "permission_policy.json"

# HARD fence -- ALWAYS enforced regardless of the (mutable) policy file. Loops are workers; the OVERSEER commits
# after review (user mandate: "loops must NOT commit/push/deploy; overseer commits after review"). A worker that
# commits bypasses the review gate (incident 2026-06-06: orc-upg-expert ran `git commit` itself). Defense-in-depth:
# the loop physically cannot commit/push/merge/rebase/tag even if permission_policy.json is misconfigured.
HARD_DENY = [r"\bgit\s+commit\b", r"\bgit\s+push\b", r"\bgit\s+merge\b", r"\bgit\s+rebase\b",
             r"\bgit\s+tag\b", r"\bgit\s+reset\s+--hard\b", r"\bgit\s+cherry-pick\b"]

# HARD file fence -- a loop may write WORK (code, data, reports) but NEVER a CONTROL SURFACE. The overseer/user own
# arm/disarm + permissions + hooks. Incident 2026-06-06: a loop wrote .claude/autonomous_mode.json to disarm itself
# (W3 authority bypass). Defense-in-depth, always enforced regardless of permission_policy.json.
HARD_FILE_DENY = [r"\.claude[\\/]autonomous_mode", r"\.claude[\\/]settings", r"\.claude[\\/]hooks[\\/]",
                  r"permission_policy\.json", r"mandatory_gates\.ya?ml", r"\.env\b", r"\.git[\\/]"]


def _deny() -> tuple[list, list]:
    """The LIVE permission gate's deny-lists (cmd + file). Fail-CLOSED to a built-in fence if unreadable."""
    try:
        p = json.loads(POLICY.read_text(encoding="utf-8"))
        return p.get("cmd_deny_regex", []), p.get("file_deny_regex", [])
    except Exception:
        return ([r"rm\s+-rf\s+[/~]", r"git\s+push\s+\S*\s*--force", r"git\s+reset\s+--hard", r"(^|\s)sudo\s",
                 r"(^|\s)dd\s+if=", r"mkfs", r"format\s+c:"], [r"\.claude[\\/]settings", r"\.env\b"])


class Tools(_HarnessTools):
    """Crypto-fenced Tools: canonical harness Tools + the loop-never-commits fence + control-surface fence + the
    live permission_policy.json deny-lists, all injected as extra denies. cwd defaults to the repo ROOT."""

    def __init__(self, root: Path | None = None, timeout: int = 300):
        cmd_policy, file_policy = _deny()
        # UNION: crypto HARD fences + the live policy on top of the canonical harness HARD_DENY/HARD_FILE_DENY.
        super().__init__(cwd=Path(root or ROOT), timeout=timeout,
                         extra_cmd_deny=HARD_DENY + cmd_policy,
                         extra_file_deny=HARD_FILE_DENY + file_policy)

    def run_python(self, code: str) -> dict:
        # write to runs/autonomy/_operator_snippet.py under the repo (the historical crypto scratch location)
        tmp = self.root / "runs" / "autonomy" / "_operator_snippet.py"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(code, encoding="utf-8")
        res = self.run_shell(f'python "{tmp}"')
        res["tool"] = "run_python"
        return res


if __name__ == "__main__":
    t = Tools()
    print("schema:", t.schema())
    print("safe   :", t.run_shell("python --version"))
    print("denied :", t.run_shell("git push --force origin main"))
    print("commit :", t.run_shell("git commit -m x"))
    print("ctrlsfc:", t.write_file(".claude/settings.json", "{}"))
    print("write  :", t.write_file("runs/autonomy/_tools_selftest.txt", "ok"))
    print("read   :", t.read_file("runs/autonomy/_tools_selftest.txt"))
