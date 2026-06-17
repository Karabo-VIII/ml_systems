#!/usr/bin/env python3
"""check_docs_shell.py -- mechanical gate for the recurring correction:
'you gave me bash / POSIX shell, but I cannot run that on PowerShell / Windows.'

Scans *.md files for POSIX-only shell idioms inside fenced code blocks.
Only shell-ish fences (bash/sh/shell/console/zsh/unlabeled) are examined.
powershell/pwsh fences are explicitly skipped. Non-shell fences (python,
json, yaml, toml, text, md, diff) are also skipped. Prose and inline
`code` spans are never scanned -- high-precision, not cry-wolf.

Default idiom set (4 patterns, high precision):
  - POSIX for-loop:   for VAR in ...; do
  - /dev/null         literal substring
  - backtick subst:   `...`
  - tail / head cmd:  as a command (start of line or after a pipe)

Extended idioms behind --strict (off by default):
  - export VAR=
  - 2>/dev/null redirect
  - rm -rf
  - mkdir -p
  - ln -s
  - chmod

Usage:
  python scripts/check_docs_shell.py <file.md> [more.md ...]
  python scripts/check_docs_shell.py --all
  python scripts/check_docs_shell.py --all --strict
  python scripts/check_docs_shell.py --selftest

Exit code == count of findings (0 = clean).
No emoji (Windows cp1252 constraint).
"""
from __future__ import annotations

import glob
import os
import re
import sys
from pathlib import Path

__contract__ = {
    "kind": "audit_gate",
    "inputs": ["*.md files (explicit paths or --all)"],
    "outputs": ["file:line  [idiom]  <trimmed-line> per finding; exit = finding count"],
    "invariants": [
        "only shell-ish fenced code blocks are scanned (bash/sh/shell/console/zsh/unlabeled)",
        "powershell/pwsh fences are SKIPPED (they are the correct idiom)",
        "non-shell fences (python/json/yaml/toml/text/md/diff) are SKIPPED",
        "prose lines and inline `code` spans are NEVER scanned",
        "default idioms: posix-for-loop, /dev/null, backtick-subst, tail/head-cmd",
        "--strict adds: export VAR=, 2>/dev/null, rm -rf, mkdir -p, ln -s, chmod",
        "--all walks repo excluding .git/.venv/archive/node_modules/backups",
        "EMPTY-INPUT GUARD: --all with 0 .md files exits nonzero with a loud ERROR",
    ],
}

# ---------------------------------------------------------------------------
# Language-tag classification
# ---------------------------------------------------------------------------
# Tags that indicate a PowerShell fence -- skip entirely
_PWSH_TAGS = {"powershell", "pwsh"}
# Tags that indicate non-shell fences -- skip entirely
_SKIP_TAGS = {"python", "py", "json", "yaml", "toml", "text", "txt", "md",
              "diff", "patch", "sql", "r", "ruby", "go", "java", "c", "cpp",
              "rust", "html", "css", "javascript", "js", "typescript", "ts",
              "xml", "ini", "cfg", "dockerfile", "makefile", "graphql"}
# Shell-ish tags -- scan these (plus the empty/unlabeled tag)
_SHELL_TAGS = {"bash", "sh", "shell", "console", "zsh", "terminal", ""}

# ---------------------------------------------------------------------------
# Default idioms (4 core patterns)
# ---------------------------------------------------------------------------
# 1. POSIX for-loop: `for VAR in SOMETHING; do`
_RE_FOR = re.compile(r'\bfor\s+\w+\s+in\b[^\n]*;\s*do\b')
# 2. /dev/null literal
_RE_DEVNULL = re.compile(r'/dev/null')
# 3. Backtick command substitution: a backtick-delimited span
#    Match a backtick followed by non-backtick content followed by a closing backtick
_RE_BACKTICK = re.compile(r'`[^`\n]+`')
# 4. tail or head as a command: at the start of a line (ignoring leading whitespace/pipe)
_RE_TAILHEAD = re.compile(r'(?:^|(?<=\|))\s*(?:tail|head)\b', re.MULTILINE)

_DEFAULT_IDIOMS = [
    ("posix-for-loop",    _RE_FOR),
    ("/dev/null",         _RE_DEVNULL),
    ("backtick-subst",    _RE_BACKTICK),
    ("tail/head-cmd",     _RE_TAILHEAD),
]

# ---------------------------------------------------------------------------
# Extended idioms (--strict only)
# ---------------------------------------------------------------------------
_RE_EXPORT   = re.compile(r'\bexport\s+\w+=')
_RE_DEVNULL2 = re.compile(r'2>/dev/null')   # redirect specifically (not bare /dev/null)
_RE_RM_RF    = re.compile(r'\brm\s+-[a-z]*r[a-z]*f\b|\brm\s+-[a-z]*f[a-z]*r\b|\brm\s+-rf\b')
_RE_MKDIR_P  = re.compile(r'\bmkdir\s+-p\b')
_RE_LN_S     = re.compile(r'\bln\s+-s\b')
_RE_CHMOD    = re.compile(r'\bchmod\b')

_STRICT_IDIOMS = [
    ("export-VAR=",   _RE_EXPORT),
    ("2>/dev/null",   _RE_DEVNULL2),
    ("rm-rf",         _RE_RM_RF),
    ("mkdir-p",       _RE_MKDIR_P),
    ("ln-s",          _RE_LN_S),
    ("chmod",         _RE_CHMOD),
]

# ---------------------------------------------------------------------------
# Fence parser
# ---------------------------------------------------------------------------
_RE_FENCE_OPEN  = re.compile(r'^[ \t]*(`{3,}|~{3,})(\w*)\s*$', re.MULTILINE)
_RE_FENCE_CLOSE = re.compile(r'^[ \t]*(`{3,}|~{3,})\s*$',      re.MULTILINE)


def _is_shell_tag(tag: str) -> bool:
    """Return True if the fence's language tag is shell-ish and should be scanned."""
    tag = tag.strip().lower()
    if tag in _PWSH_TAGS:
        return False
    if tag in _SKIP_TAGS:
        return False
    # Unlabeled or a recognized shell tag
    return tag in _SHELL_TAGS


def _extract_shell_blocks(text: str) -> list[tuple[int, str]]:
    """
    Return list of (start_lineno_1based, block_text) for each shell-ish fence.
    start_lineno is the line number of the first line INSIDE the fence (after the opening ```).
    """
    blocks: list[tuple[int, str]] = []
    lines = text.splitlines(keepends=True)
    # Build a mapping: character offset -> 1-based line number
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    def char_to_lineno(char_pos: int) -> int:
        """Binary search for the 1-based line number for a character offset."""
        lo, hi = 0, len(offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if offsets[mid] <= char_pos:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1  # 1-based

    search_from = 0
    while True:
        m_open = _RE_FENCE_OPEN.search(text, search_from)
        if m_open is None:
            break
        fence_chars = m_open.group(1)
        tag = m_open.group(2)
        content_start = m_open.end()

        # Find the matching closing fence
        close_pat = re.compile(
            r'^[ \t]*' + re.escape(fence_chars[0]) + r'{' + str(len(fence_chars)) + r',}\s*$',
            re.MULTILINE
        )
        m_close = close_pat.search(text, content_start)
        if m_close is None:
            break  # unclosed fence -- skip rest of file

        block_text = text[content_start:m_close.start()]
        if _is_shell_tag(tag):
            start_line = char_to_lineno(content_start)
            blocks.append((start_line, block_text))

        search_from = m_close.end()

    return blocks


# ---------------------------------------------------------------------------
# Scan logic
# ---------------------------------------------------------------------------

def scan(text: str, where: str, strict: bool = False) -> list[tuple[str, int, str, str]]:
    """
    Scan text for POSIX-only shell idioms inside shell-ish fenced blocks.
    Returns list of (filepath, lineno, idiom_name, trimmed_line).
    """
    idioms = list(_DEFAULT_IDIOMS)
    if strict:
        idioms += _STRICT_IDIOMS

    findings: list[tuple[str, int, str, str]] = []
    for block_start_line, block_text in _extract_shell_blocks(text):
        block_lines = block_text.splitlines()
        for rel_idx, line in enumerate(block_lines):
            lineno = block_start_line + rel_idx
            for name, pattern in idioms:
                if pattern.search(line):
                    findings.append((where, lineno, name, line.strip()[:120]))
                    break  # one finding per line (first matched idiom wins)
    return findings


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
_EXCLUDE_DIRS = {".git", ".venv", "archive", "node_modules", "backups"}


def _walk_md(root: str) -> list[str]:
    """Walk repo for *.md files, excluding _EXCLUDE_DIRS at any level."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in-place
        dirnames[:] = [d for d in dirnames if d not in _EXCLUDE_DIRS]
        for fn in filenames:
            if fn.lower().endswith(".md"):
                found.append(os.path.join(dirpath, fn))
    return found


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = sys.argv[1:]
    strict   = "--strict"   in args
    all_mode = "--all"      in args
    args = [a for a in args if not a.startswith("--")]

    if all_mode:
        repo_root = str(Path(__file__).resolve().parent.parent)
        files = _walk_md(repo_root)
        if not files:
            print("ERROR: no .md files found in repo (check is a no-op -- something is wrong)")
            return 1
    else:
        files = args

    if not files:
        print("usage: check_docs_shell.py <file.md> [more.md ...] | --all [--strict]")
        return 0

    all_findings: list[tuple[str, int, str, str]] = []
    for f in files:
        if not os.path.isfile(f):
            print(f"  WARNING: not found -- {f}")
            continue
        try:
            text = open(f, encoding="utf-8", errors="replace").read()
        except Exception as exc:
            print(f"  (read error {f}: {exc})")
            continue
        all_findings += scan(text, f, strict=strict)

    if not all_findings:
        mode_tag = " [strict]" if strict else ""
        print(f"=== check_docs_shell: {len(files)} file(s){mode_tag} | 0 POSIX-shell findings -- clean ===")
        return 0

    mode_tag = " [strict]" if strict else ""
    print(f"=== check_docs_shell: {len(all_findings)} POSIX-shell finding(s){mode_tag} "
          f"(replace with PowerShell equivalents) ===")
    for where, ln, idiom, line in all_findings:
        print(f"  {where}:{ln}  [{idiom}]  {line}")
    return len(all_findings)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

def _selftest() -> None:
    print("=== check_docs_shell selftest ===")

    # (a) POSITIVE control: shell fence with all 4 core idioms
    positive_md = r"""
Some prose about /dev/null and for loops and `date` -- should NOT be flagged.

```bash
for f in *.md; do echo $f; done
cat x | tail -5
rm /tmp/x 2>/dev/null
result=`date`
```
"""
    findings_a = scan(positive_md, "pos.md", strict=False)
    print(f"  (a) positive-control (bash fence, 4 idioms): {len(findings_a)} findings (want >= 4)")
    for _, ln, idiom, line in findings_a:
        print(f"      line {ln}  [{idiom}]  {line!r}")
    assert len(findings_a) >= 4, f"FAIL: expected >= 4 findings, got {len(findings_a)}"

    # Verify each specific idiom is present
    found_idiom_names = {f[2] for f in findings_a}
    assert "posix-for-loop" in found_idiom_names, "FAIL: posix-for-loop not detected"
    assert "tail/head-cmd"  in found_idiom_names, "FAIL: tail/head-cmd not detected"
    assert "/dev/null"      in found_idiom_names, "FAIL: /dev/null not detected"
    assert "backtick-subst" in found_idiom_names, "FAIL: backtick-subst not detected"
    print("      all 4 core idioms detected -- OK")

    # (b) NEGATIVE control 1: powershell fence
    pwsh_md = r"""
```powershell
$null
Get-Content x -Tail 5
foreach ($f in $files) { }
```
"""
    findings_b = scan(pwsh_md, "pwsh.md", strict=False)
    print(f"  (b) negative-control-1 (pwsh fence): {len(findings_b)} findings (want 0)")
    assert len(findings_b) == 0, f"FAIL: powershell fence should produce 0 findings, got {len(findings_b)}"
    print("      pwsh fence correctly skipped -- OK")

    # (c) NEGATIVE control 2: prose + inline code spans
    prose_md = r"""
On Windows you should never use /dev/null -- use $null instead.
Also, a for loop in bash looks like `for f in *; do ...; done` but in PowerShell
you write `foreach ($f in $files) { }`.
"""
    findings_c = scan(prose_md, "prose.md", strict=False)
    print(f"  (c) negative-control-2 (prose only): {len(findings_c)} findings (want 0)")
    assert len(findings_c) == 0, f"FAIL: prose should produce 0 findings, got {len(findings_c)}"
    print("      prose never scanned -- OK")

    # (d) NEGATIVE control 3: python fence containing /dev/null in a string
    python_md = r'''
```python
outfile = open("/dev/null", "w")
for x in range(10):
    head = x
```
'''
    findings_d = scan(python_md, "py.md", strict=False)
    print(f"  (d) negative-control-3 (python fence): {len(findings_d)} findings (want 0)")
    assert len(findings_d) == 0, f"FAIL: python fence should produce 0 findings, got {len(findings_d)}"
    print("      python fence correctly skipped -- OK")

    print("ALL PASS -- check_docs_shell selftest complete.")


if __name__ == "__main__":
    # Windows cp1252 footgun (CLAUDE.md invariant): scanned doc content may contain
    # non-cp1252 chars (arrows U+2192, em-dashes, etc.). Emit UTF-8 with errors=replace
    # so a finding print can NEVER crash on a Windows console/redirect. A doc-shell gate
    # that itself dies on Windows is worthless; guarded + idempotent.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
