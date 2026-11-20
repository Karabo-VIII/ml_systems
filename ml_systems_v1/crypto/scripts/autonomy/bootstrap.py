#!/usr/bin/env python3
"""Bootstrap the autonomous-agent system on a fresh clone (closes weak point W1: the wiring is gitignored).

Every hook/gate FILE is tracked, but .claude/settings.json (which WIRES them) is per-machine/gitignored -- so a
clone is inert until wired. This installer recreates the wiring from tracked templates, idempotently and
NON-destructively (it MERGES into any existing settings.json, preserving local allow-rules):
  1. ensure .claude/settings.json has the PreToolUse/UserPromptSubmit/Stop hooks (from settings.hooks.template.json)
  2. ensure permissions.defaultMode + the baseline deny-fence (union, no dups)
  3. seed runs/autonomy/permission_policy.json from its tracked template (if absent)
  4. seed runs/autonomy/frontier.json from the frontier template (if absent)
Then prints the two verification commands. Run: python scripts/autonomy/bootstrap.py [--dry-run].
No emoji (Windows cp1252). Writes settings.json directly (not via the Edit tool, which is gate-denied).
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / ".claude" / "settings.json"
HOOKS_TEMPLATE = ROOT / ".claude" / "settings.hooks.template.json"
POLICY = ROOT / "runs" / "autonomy" / "permission_policy.json"
POLICY_TEMPLATE = ROOT / "scripts" / "autonomy" / "permission_policy.template.json"
FRONTIER = ROOT / "runs" / "autonomy" / "frontier.json"
FRONTIER_TEMPLATE = ROOT / "scripts" / "autonomy" / "frontier.template.json"


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _split_py(cmd):
    """Return the script-path argument of a `python <path>` hook command (quotes stripped), else None."""
    c = (cmd or "").strip()
    if not c.lower().startswith("python "):
        return None
    rest = c[len("python "):].strip()
    if len(rest) >= 2 and rest[0] in "\"'" and rest[-1] == rest[0]:
        rest = rest[1:-1]
    return rest or None


def _resolved(cmd, root):
    """Canonical absolute script path for dedup -- an absolute and a relative form of the SAME script compare equal."""
    p = _split_py(cmd)
    if p is None:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / p
    return os.path.normcase(os.path.normpath(str(pp)))


def _absify(cmd, root):
    """Rewrite `python <relpath>` to `python "<root>/<relpath>"` (forward slashes, quoted) so the hook command is
    cwd-INDEPENDENT. Idempotent (already-absolute stays absolute). Only touches script paths that resolve under
    root AND exist -- exotic/system commands are left untouched. THIS is the cwd-lockout durability fix: the tracked
    template stays RELATIVE (portable across machines); bootstrap absifies against the clone's own root on merge."""
    p = _split_py(cmd)
    if p is None:
        return cmd
    pp = Path(p)
    absp = pp if pp.is_absolute() else (root / p)
    try:
        if not absp.exists():
            return cmd
    except OSError:
        return cmd
    return f'python "{absp.as_posix()}"'


def merge_hooks(settings, template, root):
    """Ensure every template hook is present in settings, as a cwd-INDEPENDENT absolute command, with NO duplicate
    (dedup by RESOLVED script path -- so a pre-existing relative form is recognized as the same hook, not re-added).
    Also UPGRADES any pre-existing relative known-hook command in settings to its absolute form (idempotent)."""
    changed = []
    s_hooks = settings.setdefault("hooks", {})

    # 1) upgrade any existing relative hook command -> absolute (closes the legacy-settings wedge path)
    for event, groups in s_hooks.items():
        for g in groups:
            for h in g.get("hooks", []):
                if h.get("type") == "command":
                    new = _absify(h.get("command", ""), root)
                    if new != h.get("command"):
                        changed.append(f"{event}:absify:{h.get('command')}")
                        h["command"] = new

    # 2) add any template hook whose resolved script is not already wired, as an ABSOLUTE command
    for event, groups in template.get("hooks", {}).items():
        existing = {_resolved(h.get("command", ""), root)
                    for g in s_hooks.get(event, []) for h in g.get("hooks", [])}
        existing.discard(None)
        for g in groups:
            added = False
            for h in g.get("hooks", []):
                if _resolved(h.get("command", ""), root) not in existing:
                    added = True
                    break
            if added:
                newg = json.loads(json.dumps(g))  # deep copy
                for hh in newg.get("hooks", []):
                    hh["command"] = _absify(hh.get("command", ""), root)
                    existing.add(_resolved(hh.get("command", ""), root))
                s_hooks.setdefault(event, []).append(newg)
                changed.append(f"{event}:add:{[hh.get('command') for hh in newg.get('hooks', [])]}")
    return changed


def ensure_permissions(settings, template):
    changed = []
    perms = settings.setdefault("permissions", {})
    tperms = template.get("permissions", {})
    if perms.get("defaultMode") != tperms.get("defaultMode") and tperms.get("defaultMode"):
        perms["defaultMode"] = tperms["defaultMode"]
        changed.append(f"defaultMode={perms['defaultMode']}")
    deny = perms.setdefault("deny", [])
    for d in tperms.get("deny", []):
        if d not in deny:
            deny.append(d)
            changed.append(f"deny+={d}")
    return changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    template = _load(HOOKS_TEMPLATE, None)
    if template is None:
        print(f"FATAL: missing tracked template {HOOKS_TEMPLATE}")
        return 2

    settings = _load(SETTINGS, {})
    hook_changes = merge_hooks(settings, template, ROOT)
    perm_changes = ensure_permissions(settings, template)

    seeds = []
    if not POLICY.exists() and POLICY_TEMPLATE.exists():
        seeds.append(("permission_policy.json", POLICY, POLICY_TEMPLATE))
    if not FRONTIER.exists() and FRONTIER_TEMPLATE.exists():
        seeds.append(("frontier.json", FRONTIER, FRONTIER_TEMPLATE))

    print("[bootstrap] plan:")
    print(f"  settings hooks to add : {hook_changes or 'none (already wired)'}")
    print(f"  settings perms to set : {perm_changes or 'none (already baseline)'}")
    print(f"  state files to seed   : {[s[0] for s in seeds] or 'none (already present)'}")

    if args.dry_run:
        print("[bootstrap] --dry-run: no changes written.")
        return 0

    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    for name, dst, src in seeds:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print("[bootstrap] DONE. Verify the wiring:")
    print("  python .claude/hooks/test_permission_gate.py   # all PASS")
    print("  python scripts/mandatory_gate.py               # 0 CRITICAL")
    print("  (restart the Claude Code session if the hooks were not already loaded -- hooks hot-reload, but a")
    print("   brand-new settings.json on a fresh clone is picked up at next session start.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
