#!/usr/bin/env python3
"""skill_diagnostics.py -- mechanical health check for the .claude/skills/ set (a loop-3 repeatable diagnostic).

Checks (structural, mechanizable -- the judgment-level SOTA/coverage gaps are a separate dispatched pass):
  1. frontmatter: every SKILL.md has `name:` + `description:`; name == directory name. (SKILL.md-only; ERROR.)
  2. broken links: every relative markdown link [..](target) resolves to a real file (anchors stripped). In SKILL.md
     this is ERROR; in every *.md sub-file under the skill dir (recursive) it is WARN (widened 2026-06-06).
  3. alias registry: SKILL_ALIASES.yaml canonical keys are real skills; no alias collides across two canonicals;
     no alias equals a real skill name; frontmatter `aliases` are reflected in the registry. (SKILL.md-only.)
  4. stale path refs: `src/...`, `scripts/...`, `docs/...`, `.claude/...` paths that do not exist on disk (best-effort
     -- flags likely-dead references, e.g. archived-at-reset toolchain). Scanned in SKILL.md AND every *.md sub-file
     under the skill dir (recursive); WARN at both levels (sub-file widening added 2026-06-06).

Sub-file widening (2026-06-06): checks 2 + 4 now glob ALL *.md sub-files under each skill dir, not just SKILL.md,
closing the "0 ERROR != clean" blind spot (stale-as-live refs in e.g. trader/*.md). Sub-file findings are WARN-only
and purely additive -- the SKILL.md frontmatter/alias/broken-link ERROR checks are unchanged and SKILL.md-scoped.

Exit code = number of ERROR-level findings (0 = clean). WARN findings do not fail. No emoji (cp1252).
Usage: python scripts/skill_diagnostics.py [--json]
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILLS = os.path.join(ROOT, ".claude", "skills")
ALIASES = os.path.join(SKILLS, "_common", "SKILL_ALIASES.yaml")

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
PATH_RE = re.compile(r"`((?:src|scripts|docs|config|\.claude)/[A-Za-z0-9_./-]+)`")


def _skill_dirs():
    out = []
    for d in sorted(os.listdir(SKILLS)):
        p = os.path.join(SKILLS, d, "SKILL.md")
        if d != "_common" and os.path.isfile(p):
            out.append((d, p))
    return out


def _frontmatter(text):
    fm = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            for line in text[3:end].splitlines():
                m = re.match(r"^([a-z_]+):\s*(.*)$", line)
                if m:
                    fm[m.group(1)] = m.group(2).strip()
    return fm


def _parse_aliases(path):
    """Parse the simple `canonical:\n  - alias` YAML without a yaml dep. Returns {canonical: [aliases]}."""
    out, cur = {}, None
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        if line.strip().startswith("#") or not line.strip():
            continue
        if re.match(r"^[A-Za-z0-9_]+:\s*$", line):
            cur = line.split(":")[0].strip(); out[cur] = []
        elif line.lstrip().startswith("- ") and cur:
            out[cur].append(line.lstrip()[2:].strip())
    return out


def _md_subfiles(skill_dir):
    """All *.md files under skill_dir (recursive), EXCLUDING the top-level SKILL.md.

    Returns sorted [(relpath_with_forward_slashes, abspath)]. Used to widen the broken-link + stale-path checks
    beyond SKILL.md (additive 2026-06-06)."""
    out = []
    for dirpath, _dirs, files in os.walk(skill_dir):
        for fn in files:
            if not fn.endswith(".md"):
                continue
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, skill_dir)
            if rel == "SKILL.md":
                continue
            out.append((rel.replace(os.sep, "/"), ap))
    return sorted(out)


def _scan_broken_links(text, base):
    """Yield each relative markdown-link target in `text` that does not resolve against `base` (anchors/URLs skipped)."""
    for m in LINK_RE.finditer(text):
        tgt = m.group(1).strip()
        if tgt.startswith(("http://", "https://", "#", "mailto:")):
            continue
        tgt = tgt.split("#", 1)[0]
        if not tgt:
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, tgt))):
            yield m.group(1)


def _scan_stale_paths(text):
    """Yield each `repo/path` backtick ref in `text` that does not exist under ROOT (glob/template refs skipped)."""
    for m in PATH_RE.finditer(text):
        ref = m.group(1)
        if any(g in ref for g in ("*", "{", "<")):
            continue
        if not os.path.exists(os.path.join(ROOT, ref)):
            yield ref


def main():
    as_json = "--json" in sys.argv
    findings = []  # (level, skill, msg)
    skills = _skill_dirs()
    names = {d for d, _ in skills}

    # 1. frontmatter
    texts = {}
    for d, p in skills:
        t = open(p, encoding="utf-8").read(); texts[d] = t
        fm = _frontmatter(t)
        if "name" not in fm:
            findings.append(("ERROR", d, "missing frontmatter name"))
        elif fm["name"] != d:
            findings.append(("ERROR", d, f"frontmatter name '{fm['name']}' != dir '{d}'"))
        if "description" not in fm:
            findings.append(("ERROR", d, "missing frontmatter description"))

    # 2. broken relative links in SKILL.md -> ERROR (scoped to SKILL.md; sub-files handled in block 5 as WARN).
    for d, p in skills:
        for tgt in _scan_broken_links(texts[d], os.path.dirname(p)):
            findings.append(("ERROR", d, f"broken link -> {tgt}"))

    # 3. alias registry. __builtins__ is a special list of harness-built-in command names (not a canonical mapping);
    # canonicals that are harness built-ins (update-config/schedule/...) are valid even without a project skill dir.
    reg = _parse_aliases(ALIASES)
    builtins = set(reg.pop("__builtins__", []))
    valid_canon = names | builtins
    seen_alias = {}
    for canon, al in reg.items():
        if canon not in valid_canon:
            findings.append(("ERROR", "_aliases", f"canonical '{canon}' is not a real skill dir or harness built-in"))
        for a in al:
            if a in names:
                findings.append(("WARN", "_aliases", f"alias '{a}' (->{canon}) equals a real skill name (shadowing)"))
            if a in seen_alias and seen_alias[a] != canon:
                findings.append(("ERROR", "_aliases", f"alias '{a}' maps to BOTH {seen_alias[a]} and {canon}"))
            seen_alias[a] = canon
    # frontmatter aliases reflected in registry?
    for d, p in skills:
        fm = _frontmatter(texts[d])
        if "aliases" in fm:
            declared = re.findall(r"[A-Za-z0-9_]+", fm["aliases"])
            for a in declared:
                if a == d:
                    continue
                if a not in reg.get(d, []):
                    findings.append(("WARN", d, f"frontmatter alias '{a}' not in SKILL_ALIASES.yaml[{d}]"))

    # 4. stale path refs in SKILL.md (best-effort) -> WARN (scoped to SKILL.md; sub-files handled in block 5).
    for d, p in skills:
        for ref in _scan_stale_paths(texts[d]):
            findings.append(("WARN", d, f"path ref may be stale -> {ref}"))

    # 5. sub-file scan (ADDITIVE 2026-06-06): widen the broken-link + stale-path checks (2 + 4) to ALL *.md sub-files
    # under each skill dir (recursive), not just SKILL.md. Sub-file findings are WARN-only -- they surface
    # stale-as-live refs in skill sub-docs (e.g. trader/PRE_DEPLOY_CHECKLIST.md) without ERROR-failing the gate,
    # closing the "0 ERROR != clean" blind spot. Frontmatter/alias/SKILL.md-link ERROR checks stay SKILL.md-scoped.
    for d, p in skills:
        for rel, ap in _md_subfiles(os.path.dirname(p)):
            try:
                sub = open(ap, encoding="utf-8").read()
            except OSError as e:
                findings.append(("WARN", d, f"sub-file {rel} unreadable: {e}"))
                continue
            for tgt in _scan_broken_links(sub, os.path.dirname(ap)):
                findings.append(("WARN", d, f"sub-file {rel}: broken link -> {tgt}"))
            for ref in _scan_stale_paths(sub):
                findings.append(("WARN", d, f"sub-file {rel}: path ref may be stale -> {ref}"))

    errors = [f for f in findings if f[0] == "ERROR"]
    warns = [f for f in findings if f[0] == "WARN"]
    if as_json:
        print(json.dumps({"n_skills": len(skills), "errors": errors, "warns": warns}, indent=2))
    else:
        print(f"=== skill_diagnostics: {len(skills)} skills | {len(errors)} ERROR, {len(warns)} WARN ===")
        for lvl, sk, msg in errors + warns:
            print(f"  {lvl:5} [{sk}] {msg}")
        if not findings:
            print("  clean.")
    return len(errors)


if __name__ == "__main__":
    raise SystemExit(main())
