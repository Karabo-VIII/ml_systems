#!/usr/bin/env python3
"""check_register_coverage.py -- mechanical gate for the recurring failure:
'CLAUDE.md / DIRECTIVES_REGISTER cites dead/ghost enforcer paths.'

Every rule in the DIRECTIVES_REGISTER must have an ENFORCED-BY pointer that
ACTUALLY EXISTS on disk, or be explicitly flagged a GAP (with a WARNING glyph).
Ghost enforcers (path cited but missing) are the recurring failure mode that
lets directive coverage silently rot.

CLASSIFY each rule into exactly one bucket:
  GAP              -- rule contains a WARNING glyph (U+26A0) or the word GAP
                      -> honestly declared, NOT a failure
  ENFORCED         -- has a right-arrow pointer and at least one target resolves on disk
  MISSING-ENFORCER -- has a right-arrow pointer (or a CHECK-MARK glyph) but NO target resolves
  UNCOVERED        -- no GAP/WARNING and no resolvable pointer at all

Exit code == (MISSING-ENFORCER + UNCOVERED) count.  0 = every rule is either
enforced-on-disk or honestly declared a GAP.

EMPTY-INPUT GUARD: if the register file is missing or parses to 0 rules, print
a loud ERROR and exit nonzero.

Usage:
  python scripts/autonomy/check_register_coverage.py
      -> uses default register at .claude/skills/_common/DIRECTIVES_REGISTER.md
  python scripts/autonomy/check_register_coverage.py <path/to/register.md>
  python scripts/autonomy/check_register_coverage.py --selftest

No emoji (Windows cp1252 constraint). Exit code == finding count.
"""
from __future__ import annotations

import io
import os
import re
import sys
import tempfile
from pathlib import Path

# Force UTF-8 stdout so Unicode characters in the register (arrows, glyphs)
# do not crash on Windows cp1252 terminals.  Output is still plain-text;
# the receiver (terminal or pipe) decides rendering.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

__contract__ = {
    "kind": "audit_gate",
    "inputs": ["DIRECTIVES_REGISTER.md (or path override)"],
    "outputs": ["per-rule status lines + summary; exit = MISSING-ENFORCER + UNCOVERED count"],
    "invariants": [
        "every rule must have a -> pointer resolving on disk, or a WARNING/GAP tag",
        "ghost/dead enforcer paths are flagged MISSING-ENFORCER = failure",
        "honestly-declared GAP rules are NOT failures",
        "EMPTY-INPUT GUARD: 0 rules parsed -> loud ERROR + nonzero exit",
        "read-only; deterministic; stdlib only",
    ],
}

# ---------------------------------------------------------------------------
# Repo root detection
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
# This script lives at scripts/autonomy/, so repo root is 2 levels up
_REPO_ROOT = _THIS_FILE.parent.parent.parent

# ---------------------------------------------------------------------------
# Default register path
# ---------------------------------------------------------------------------
_DEFAULT_REGISTER = _REPO_ROOT / ".claude" / "skills" / "_common" / "DIRECTIVES_REGISTER.md"

# ---------------------------------------------------------------------------
# Memory directory (for [[slug]] resolution)
# ---------------------------------------------------------------------------
_MEMORY_DIR = _REPO_ROOT / "memory"

# ---------------------------------------------------------------------------
# Glyphs (stored as escape sequences to avoid cp1252 issues in this source)
# GAP indicator: WARNING SIGN U+26A0 (displayed as a two-char UTF-8 or as the
# raw unicode scalar); CHECK MARK U+2705 (displayed as green check in some terminals)
# We match these by their unicode code point, read from the UTF-8 file.
# ---------------------------------------------------------------------------
_GAP_GLYPH    = "⚠"   # WARNING SIGN (the file has the UTF-8 encoded glyph)
_CHECK_GLYPH  = "✅"   # WHITE HEAVY CHECK MARK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Numbered-rule opening: "1." "9a." "9b." "14." etc., possibly indented
_RE_RULE_START = re.compile(r'^\s*(\d+[a-z]?)\.\s+(.+)', re.MULTILINE)
# Section headers that terminate a rule (## or #)
_RE_SECTION    = re.compile(r'^##?\s+', re.MULTILINE)
# Arrow pointer: matches ASCII '->' OR Unicode right-arrow U+2192 (as used in the real register)
_RE_ARROW      = re.compile(r'->|→')


def _parse_rules(text: str) -> list[dict]:
    """
    Parse DIRECTIVES_REGISTER.md into a list of rule dicts:
      {id, raw_text, has_gap, has_check, has_arrow, pointer_tokens}
    Rules run from the numbered line until the next numbered line or section header.
    """
    # Collect all start positions: rule starts and section headers
    events: list[tuple[int, str, str | None]] = []  # (pos, kind, id_or_None)
    for m in _RE_RULE_START.finditer(text):
        events.append((m.start(), "rule", m.group(1)))
    for m in _RE_SECTION.finditer(text):
        events.append((m.start(), "section", None))
    # Sort by position
    events.sort(key=lambda e: e[0])

    rules: list[dict] = []
    for idx, (pos, kind, rule_id) in enumerate(events):
        if kind != "rule":
            continue
        # Find end: next event after this one
        if idx + 1 < len(events):
            end_pos = events[idx + 1][0]
        else:
            end_pos = len(text)
        raw_text = text[pos:end_pos].rstrip()

        has_gap   = _GAP_GLYPH   in raw_text or re.search(r'\bGAP\b', raw_text, re.I) is not None
        has_check = _CHECK_GLYPH in raw_text

        # Extract pointer tokens after '->'
        pointer_tokens: list[str] = []
        arrow_match = _RE_ARROW.search(raw_text)
        if arrow_match:
            after_arrow = raw_text[arrow_match.end():]
            # Tokens are comma-separated or space-separated references
            # Split on comma, then tokenise each chunk
            for chunk in re.split(r'[,\n]', after_arrow):
                chunk = chunk.strip()
                if not chunk:
                    continue
                # Individual whitespace-separated tokens
                for tok in chunk.split():
                    tok = tok.strip('`.,;:()[]"\'')
                    if tok:
                        pointer_tokens.append(tok)

        # Filter to "pointer-like" tokens: contain '/', end with .py/.md/.yaml/.json,
        # start with '/', or are [[slug]] or /skillname references
        filtered: list[str] = []
        for tok in pointer_tokens:
            if (
                '/' in tok
                or tok.startswith('[[')
                or any(tok.endswith(ext) for ext in ('.py', '.md', '.yaml', '.json', '.txt'))
                or tok.startswith('watcher')  # bare filenames
                or re.match(r'^[a-z_]+\.py$', tok)
            ):
                filtered.append(tok)

        rules.append({
            "id":              rule_id,
            "raw_text":        raw_text,
            "has_gap":         has_gap,
            "has_check":       has_check,
            "has_arrow":       arrow_match is not None,
            "pointer_tokens":  filtered,
        })

    return rules


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

def _resolve_token(tok: str) -> bool:
    """
    Return True if the token resolves to an existing path on disk.

    Resolution rules (in order):
      1. [[slug]]  -> look for <slug>.md anywhere in memory dir or repo
      2. /skillname -> .claude/skills/<skillname>/ dir exists
      3. path token (contains '/' or ends .py/.md/.yaml/.json) ->
           try as-given (absolute), then relative to repo root
      4. bare *.py filename -> findable anywhere in repo (excl. .venv/.git/archive)
    """
    if tok.startswith('[[') and tok.endswith(']]'):
        slug = tok[2:-2].strip()
        # Check memory dir
        if _MEMORY_DIR.exists():
            for f in _MEMORY_DIR.rglob("*.md"):
                if slug in f.name:
                    return True
        # Broader: any .md in repo with matching name
        for f in _REPO_ROOT.rglob("*.md"):
            if slug in f.name:
                return True
        return False

    if tok.startswith('/') and '.' not in tok.split('/')[-1]:
        # Looks like a /skillname reference
        skill_dir = _REPO_ROOT / ".claude" / "skills" / tok.lstrip('/')
        return skill_dir.exists()

    # Path-like tokens
    candidate = Path(tok)
    if candidate.is_absolute():
        return candidate.exists()

    # Relative to repo root
    rel = _REPO_ROOT / tok
    if rel.exists():
        return True

    # If the token is a bare filename (no directory separator other than internal)
    # and ends with .py, search under repo (excluding heavyweight dirs)
    if re.match(r'^[a-z_][a-z_0-9]*\.py$', tok):
        for root, dirs, files in os.walk(str(_REPO_ROOT)):
            dirs[:] = [d for d in dirs if d not in {'.venv', '.git', 'archive', '__pycache__',
                                                      'node_modules', 'backups'}]
            if tok in files:
                return True
        return False

    return False


def _any_token_resolves(tokens: list[str]) -> bool:
    return any(_resolve_token(t) for t in tokens)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

STATUS_GAP      = "GAP"
STATUS_ENFORCED = "ENFORCED"
STATUS_MISSING  = "MISSING-ENFORCER"
STATUS_UNCOVERED= "UNCOVERED"


def _classify(rule: dict) -> str:
    if rule["has_gap"]:
        return STATUS_GAP
    has_pointer_claim = rule["has_arrow"] or rule["has_check"]
    tokens = rule["pointer_tokens"]
    if has_pointer_claim and tokens and _any_token_resolves(tokens):
        return STATUS_ENFORCED
    if has_pointer_claim:
        return STATUS_MISSING
    return STATUS_UNCOVERED


# ---------------------------------------------------------------------------
# Title extractor (first non-bullet text of the rule)
# ---------------------------------------------------------------------------
_RE_BOLD = re.compile(r'\*\*(.+?)\*\*')


def _extract_title(raw_text: str) -> str:
    first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    m = _RE_BOLD.search(first_line)
    if m:
        title = m.group(1)[:80]
    else:
        # Strip numbering prefix and glyphs
        cleaned = re.sub(r'^\s*\d+[a-z]?\.\s+', '', first_line)
        cleaned = cleaned.replace(_GAP_GLYPH, '').replace(_CHECK_GLYPH, '').strip()
        title = cleaned[:80]
    # Strip the arrow and everything after it so the title is a short label
    title = re.sub(r'(->|→).*', '', title).strip()
    return title


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def run_check(register_path: str) -> int:
    """
    Run the coverage check against the given register file.
    Returns the exit code (MISSING + UNCOVERED count).
    """
    if not os.path.isfile(register_path):
        print(f"ERROR: register file not found: {register_path} (check is a no-op)")
        return 1

    text = open(register_path, encoding="utf-8", errors="replace").read()
    rules = _parse_rules(text)

    if not rules:
        print(f"ERROR: 0 rules parsed from {register_path} (check is a no-op -- verify format)")
        return 1

    counts = {STATUS_GAP: 0, STATUS_ENFORCED: 0, STATUS_MISSING: 0, STATUS_UNCOVERED: 0}
    for rule in rules:
        status = _classify(rule)
        counts[status] += 1
        title = _extract_title(rule["raw_text"])
        print(f"[{status:<18}] {rule['id']:>3}  {title}")

    total = len(rules)
    e = counts[STATUS_ENFORCED]
    g = counts[STATUS_GAP]
    m = counts[STATUS_MISSING]
    u = counts[STATUS_UNCOVERED]
    exit_code = m + u
    print()
    print(f"=== check_register_coverage: {total} rules: "
          f"{e} enforced, {g} gap, {m} missing-enforcer, {u} uncovered "
          f"| exit={exit_code} ===")
    return exit_code


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== check_register_coverage selftest ===")

    # Use check_report_claims.py as the existing-path target for the ENFORCED rule
    existing_path = str(_REPO_ROOT / "src" / "audit" / "check_report_claims.py")
    if not os.path.isfile(existing_path):
        # Fallback: use this script itself
        existing_path = str(_THIS_FILE)
    print(f"  existing path for ENFORCED rule: {existing_path}")

    fixture = f"""\
# TEST REGISTER

## A. Test section

1. ✅ **Enforced rule** -- something is enforced here. [U1] -> {existing_path}
2. ⚠ **GAP rule** -- this is not yet implemented. [U2]
3. ✅ **Missing enforcer rule** -- claims to be enforced. [U3] -> does/not/exist_xyz.py
"""

    fd, tmp_path = tempfile.mkstemp(suffix=".md", prefix="test_register_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(fixture)

        text = open(tmp_path, encoding="utf-8").read()
        rules = _parse_rules(text)

        print(f"  rules parsed: {len(rules)} (want 3)")
        assert len(rules) == 3, f"FAIL: expected 3 rules, got {len(rules)}"

        statuses = [_classify(r) for r in rules]
        print(f"  statuses: {statuses}")

        n_enforced  = statuses.count(STATUS_ENFORCED)
        n_gap       = statuses.count(STATUS_GAP)
        n_missing   = statuses.count(STATUS_MISSING)
        n_uncovered = statuses.count(STATUS_UNCOVERED)

        print(f"  enforced={n_enforced} (want 1): {'OK' if n_enforced == 1 else 'FAIL'}")
        print(f"  gap={n_gap}      (want 1): {'OK' if n_gap == 1 else 'FAIL'}")
        print(f"  missing={n_missing}   (want 1): {'OK' if n_missing == 1 else 'FAIL'}")
        print(f"  uncovered={n_uncovered} (want 0): {'OK' if n_uncovered == 0 else 'FAIL'}")

        assert n_enforced  == 1, f"FAIL: enforced expected 1, got {n_enforced}"
        assert n_gap       == 1, f"FAIL: gap expected 1, got {n_gap}"
        assert n_missing   == 1, f"FAIL: missing expected 1, got {n_missing}"
        assert n_uncovered == 0, f"FAIL: uncovered expected 0, got {n_uncovered}"

        exit_would_be = n_missing + n_uncovered
        print(f"  exit-code would be {exit_would_be} (want 1): {'OK' if exit_would_be == 1 else 'FAIL'}")
        assert exit_would_be == 1, f"FAIL: exit-code expected 1, got {exit_would_be}"

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    print("ALL PASS -- check_register_coverage selftest complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        register_path = args[0]
    else:
        register_path = str(_DEFAULT_REGISTER)
    return run_check(register_path)


if __name__ == "__main__":
    # Windows cp1252 footgun (CLAUDE.md invariant): register rule titles may contain
    # non-cp1252 chars (arrows U+2192, em-dashes, etc.). Emit UTF-8 with errors=replace
    # so a rule print can NEVER crash on a Windows console/redirect. Guarded + idempotent.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
