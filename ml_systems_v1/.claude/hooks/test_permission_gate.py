#!/usr/bin/env python3
"""RWYB regression test for permission_gate.py (the PreToolUse dynamic gate).

Run: python .claude/hooks/test_permission_gate.py
Tests are run by SUBPROCESSING the hook with each input on stdin -- so the deny-pattern strings live in THIS
file, never on a Bash command line (which the live gate would itself intercept). Exit 0 = all pass.
"""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOOK = os.path.join(ROOT, ".claude", "hooks", "permission_gate.py")

CASES = [
    ("allow: edit src code",                       {"tool_name": "Edit", "tool_input": {"file_path": "src/strat/battery.py"}}, "allow"),
    ("allow: read-only bash mentioning settings",  {"tool_name": "Bash", "tool_input": {"command": "git check-ignore .claude/settings.json.bak_2026_06_05"}}, "allow"),
    ("allow: normal git push",                     {"tool_name": "Bash", "tool_input": {"command": "git push origin wm-hardening-2026-05-29"}}, "allow"),
    ("allow: git commit",                          {"tool_name": "Bash", "tool_input": {"command": "git commit -m fix"}}, "allow"),
    ("allow: edit a skill (brain, not settings)",  {"tool_name": "Edit", "tool_input": {"file_path": ".claude/skills/trader/SKILL.md"}}, "allow"),
    ("allow: grep referencing .env in a path arg", {"tool_name": "Bash", "tool_input": {"command": "grep -rn TODO src/.envrc_notes.md"}}, "allow"),
    ("DENY: edit settings.local.json",             {"tool_name": "Edit", "tool_input": {"file_path": "C:/x/.claude/settings.local.json"}}, "deny"),
    ("DENY: edit settings.json",                   {"tool_name": "Write", "tool_input": {"file_path": ".claude/settings.json"}}, "deny"),
    ("DENY: write secrets",                        {"tool_name": "Write", "tool_input": {"file_path": "config/api_keys.yaml"}}, "deny"),
    ("DENY: write .env",                           {"tool_name": "Write", "tool_input": {"file_path": "deploy/.env.prod"}}, "deny"),
    ("DENY: rm -rf root",                          {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, "deny"),
    ("DENY: force push",                           {"tool_name": "Bash", "tool_input": {"command": "git push --force origin main"}}, "deny"),
    ("DENY: hard reset",                           {"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD~3"}}, "deny"),
    ("DENY: shell write into settings",            {"tool_name": "Bash", "tool_input": {"command": "cp foo .claude/settings.json"}}, "deny"),
    ("DENY: git restore (discards uncommitted)",   {"tool_name": "Bash", "tool_input": {"command": "git restore src/"}}, "deny"),
    ("DENY: SKIP_CDAP bypass (mandatory gate)",     {"tool_name": "Bash", "tool_input": {"command": "SKIP_CDAP=1 git commit -m x"}}, "deny"),
    ("DENY: --no-verify bypass (mandatory gate)",   {"tool_name": "Bash", "tool_input": {"command": "git commit --no-verify -m x"}}, "deny"),
    ("DENY: top-level cd wedges shared cwd",        {"tool_name": "Bash", "tool_input": {"command": "cd projects/chess_zero && python x.py"}}, "deny"),
    ("DENY: PowerShell Set-Location wedges cwd",    {"tool_name": "PowerShell", "tool_input": {"command": "Set-Location src"}}, "deny"),
    ("allow: subshell cd does NOT wedge",           {"tool_name": "Bash", "tool_input": {"command": "(cd projects/chess_zero && python x.py)"}}, "allow"),
    ("allow: git -C (no cd, no wedge)",             {"tool_name": "Bash", "tool_input": {"command": "git -C projects/chess_zero status"}}, "allow"),
    ("allow: note mentions -> settings.json (arrow, not a write)", {"tool_name": "Bash", "tool_input": {"command": "echo done -> .claude/settings.json wired"}}, "allow"),
]


def run_case(inp):
    r = subprocess.run([sys.executable, HOOK], input=json.dumps(inp), capture_output=True, text=True,
                       creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    out = (r.stdout or "").strip()
    if not out:
        return "normal-flow"
    return json.loads(out)["hookSpecificOutput"]["permissionDecision"]


def main():
    ok = True
    for name, inp, expect in CASES:
        got = run_case(inp)
        good = got == expect
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name}: got={got} expect={expect}")
    print("ALL PASS" if ok else "*** SOME FAILED ***")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
