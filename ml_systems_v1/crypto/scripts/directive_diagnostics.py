#!/usr/bin/env python3
"""directive_diagnostics.py -- mechanical consistency lint for the DIRECTIVES (CLAUDE.md + .claude/skills/_common/*.md).

The +k generalization of the 2026-06-06 DIRECTIVE_GAP_AUDIT: mechanize its findings so directive drift is caught by a
re-runnable check, not a one-off audit. Sibling to scripts/skill_diagnostics.py (which lints the SKILLS). Read-only.

Checks (exit code = ERROR count; WARN does not fail):
  1. DANGLING _common LINKS (ERROR) -- every [..](.claude/skills/_common/X.md) link must resolve to a real file
     (the F2 class: links to OPERATIONAL_DIRECTIVES/PROTOCOL_COMPOSITION that were archived at the reset).
  2. ARCHIVED-vs-LIVE (WARN) -- a line calling a _common file "archived" while that file STILL EXISTS live under
     _common/ is a mislabel (the F5 class: AUTONOMOUS_RUNNER listed as archived while load-bearing).
  3. ROUTER COMPLETENESS (ERROR) -- every real skill dir (minus _common) must appear in SLASH_ROUTER.md, and the
     stated skill count must match the dir count (the F3 class: missing orc/discover/narrate + wrong count).
  4. TOMBSTONE DISCIPLINE (WARN) -- a stale-as-live claim (IC-ladder / a fixed +NN.NN% headline / dossier / gold-standard)
     must sit within TOMB_WINDOW lines of an ARCHIVED / DO-NOT-FOLLOW / tombstone marker, else it reads as current.

Usage: python scripts/directive_diagnostics.py [--json]. No emoji (cp1252).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(ROOT, ".claude", "skills", "_common")
SKILLS = os.path.join(ROOT, ".claude", "skills")
CLAUDE_MD = os.path.join(ROOT, "CLAUDE.md")
ROUTER = os.path.join(COMMON, "SLASH_ROUTER.md")
TOMB_WINDOW = 6

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
COMMON_LINK_RE = re.compile(r"\.claude/skills/_common/([A-Za-z0-9_.-]+\.md)")
# the BANNED-paradigm stale-as-live markers (NOT bare percentages -- those match legit backtest numbers like +94.02%).
TOMB_CLAIM_RE = re.compile(r"IC\s*>\s*0\.0?[0-9]|ShIC\s*>|IC-ladder|Headline-tier|gold[- ]standard|dossier", re.I)
TOMB_MARK_RE = re.compile(r"ARCHIV|DO NOT FOLLOW|DO-NOT-FOLLOW|tombstone|DEPRECATED|NOT CURRENT|ghost-state", re.I)


def _directive_files():
    out = [CLAUDE_MD] if os.path.exists(CLAUDE_MD) else []
    out += sorted(glob.glob(os.path.join(COMMON, "*.md")))
    return out


def _check_dangling_common_links(path, text, findings):
    base = os.path.dirname(path)
    for m in LINK_RE.finditer(text):
        tgt = m.group(1).strip().split("#", 1)[0]
        if not tgt or tgt.startswith(("http://", "https://", "mailto:")):
            continue
        if "_common/" in tgt and tgt.endswith(".md"):
            if not os.path.exists(os.path.normpath(os.path.join(base, tgt))):
                findings.append(("ERROR", os.path.relpath(path, ROOT), f"dangling _common link -> {tgt}"))


def _check_archived_vs_live(path, text, findings):
    for ln, line in enumerate(text.splitlines(), 1):
        if "archiv" not in line.lower():
            continue
        for fn in COMMON_LINK_RE.findall(line) or re.findall(r"\b([A-Z_]{4,}\.md)\b", line):
            if os.path.exists(os.path.join(COMMON, fn)):
                # a file called "archived" that still lives in _common -> mislabel (unless the line also says it was
                # MOVED/relocated, i.e. it's an archived-spec pointer, not a status claim about the live file)
                if not re.search(r"reloc|moved|folded|fixed|harmonis|repoint", line, re.I):
                    findings.append(("WARN", os.path.relpath(path, ROOT),
                                     f"line {ln}: calls '{fn}' archived but it EXISTS live in _common/ (mislabel?)"))


def _check_router_completeness(findings):
    if not os.path.exists(ROUTER):
        findings.append(("ERROR", "SLASH_ROUTER.md", "missing"))
        return
    rt = open(ROUTER, encoding="utf-8").read()
    skill_dirs = {d for d in os.listdir(SKILLS)
                  if d != "_common" and os.path.isfile(os.path.join(SKILLS, d, "SKILL.md"))}
    for d in sorted(skill_dirs):
        if not re.search(rf"\b{re.escape(d)}\b", rt):
            findings.append(("ERROR", "SLASH_ROUTER.md", f"skill '{d}' missing from the router"))
    m = re.search(r"\b(\d{1,2})\s+(?:invokable\s+)?skills\b", rt)
    if m and int(m.group(1)) != len(skill_dirs):
        findings.append(("ERROR", "SLASH_ROUTER.md",
                         f"stated count {m.group(1)} != actual skill dirs {len(skill_dirs)}"))


def _check_tombstone(path, text, findings):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not TOMB_CLAIM_RE.search(line):
            continue
        lo, hi = max(0, i - TOMB_WINDOW), min(len(lines), i + TOMB_WINDOW + 1)
        if not any(TOMB_MARK_RE.search(lines[j]) for j in range(lo, hi)):
            findings.append(("WARN", os.path.relpath(path, ROOT),
                             f"line {i+1}: stale-as-live claim with no ARCHIVED/tombstone marker within {TOMB_WINDOW} lines"))


def main():
    findings = []
    for p in _directive_files():
        t = open(p, encoding="utf-8").read()
        _check_dangling_common_links(p, t, findings)
        _check_archived_vs_live(p, t, findings)
        _check_tombstone(p, t, findings)
    _check_router_completeness(findings)

    errors = [f for f in findings if f[0] == "ERROR"]
    warns = [f for f in findings if f[0] == "WARN"]
    if "--json" in sys.argv:
        print(json.dumps({"errors": errors, "warns": warns}, indent=2))
    else:
        print(f"=== directive_diagnostics: {len(_directive_files())} files | {len(errors)} ERROR, {len(warns)} WARN ===")
        for lvl, where, msg in errors + warns:
            print(f"  {lvl:5} [{where}] {msg}")
        if not findings:
            print("  clean.")
    return len(errors)


if __name__ == "__main__":
    raise SystemExit(main())
