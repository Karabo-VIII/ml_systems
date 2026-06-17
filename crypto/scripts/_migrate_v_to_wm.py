"""One-shot migrator: rewrite `src/vN` -> `src/wm/vN` and `"src" / "vN"` ->
`"src" / "wm" / "vN"` across the whole tree.

Active versions: v0, v1, v3, v4, v6, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19.
Archived versions (v2, v5, v7) get their refs rewritten to point at
`backups/BKP_20260429_MODEL_HARMONIZATION/vN`.

Run from repo root. Idempotent: re-running is a no-op.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ACTIVE = ["v0", "v1", "v3", "v4", "v6", "v8", "v9", "v10", "v11", "v12",
          "v13", "v14", "v15", "v16", "v17", "v18", "v19"]
ARCHIVED = ["v2", "v5", "v7"]

EXTS = {".py", ".yaml", ".yml", ".md", ".txt", ".json"}

SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    "backups",  # do not touch backup contents
    "data",     # data files
    "logs", "plots", "models",
}


def should_process(p: Path) -> bool:
    if p.suffix.lower() not in EXTS:
        return False
    parts = set(p.relative_to(ROOT).parts)
    if parts & SKIP_DIRS:
        return False
    return True


def rewrite_text(text: str) -> tuple[str, int]:
    """Return (new_text, n_substitutions)."""
    n = 0

    # 1. Filesystem-style:  src/vN  ->  src/wm/vN  (active)  or
    #                              ->  backups/BKP_.../vN  (archived)
    # Lookbehind allows leading "/" (e.g. "<root>/src/wm/v0") but rejects
    # alphanumerics that would form a different word (e.g. "abcsrc/v0").
    # Lookahead matches typical path/quote/punct boundaries so we don't
    # eat into "v01" or "v0_baseline".
    # Skip rewriting if the LHS already says "src/wm/" (idempotency guard).
    for v in ACTIVE:
        pat = re.compile(r"(?<![A-Za-z0-9_])(?<!wm/)src/" + re.escape(v) + r"(?=[/\"'\s\)\]\.,;:>`]|$)")
        text, k = pat.subn(f"src/wm/{v}", text)
        n += k
    for v in ARCHIVED:
        pat = re.compile(r"(?<![A-Za-z0-9_])(?<!wm/)src/" + re.escape(v) + r"(?=[/\"'\s\)\]\.,;:>`]|$)")
        text, k = pat.subn(f"backups/BKP_20260429_MODEL_HARMONIZATION/{v}", text)
        n += k

    # 2. Path-component style:  "src" / "vN"   ->   "src" / "wm" / "vN"
    for v in ACTIVE:
        pat = re.compile(r"\"src\"\s*/\s*\"" + re.escape(v) + r"\"")
        text, k = pat.subn(f'"src" / "wm" / "{v}"', text)
        n += k

    # 3. Doc-only legacy form (some RUN.md/README.md describe the layout
    #    pre-2026 as `src/vN_training/...` without the parent dir). Rewrite
    #    those to the canonical `src/wm/vN/vN_training/...` form.
    #    Suffixes seen in tree: _training, _baseline, _meta, plus N_<digit>_training.
    for v in ACTIVE:
        pat = re.compile(
            r"(?<![A-Za-z0-9_])(?<!wm/)src/" + re.escape(v) + r"(_(?:training|baseline|meta|\d+_training))(?=[/\"'\s\)\]\.,;:>`]|$)"
        )
        text, k = pat.subn(lambda m, v=v: f"src/wm/{v}/{v}{m.group(1)}", text)
        n += k
    for v in ARCHIVED:
        pat = re.compile(
            r"(?<![A-Za-z0-9_])(?<!wm/)src/" + re.escape(v) + r"(_(?:training|baseline|meta|\d+_training))(?=[/\"'\s\)\]\.,;:>`]|$)"
        )
        text, k = pat.subn(
            lambda m, v=v: f"backups/BKP_20260429_MODEL_HARMONIZATION/{v}/{v}{m.group(1)}",
            text,
        )
        n += k

    return text, n


def main() -> int:
    changed = 0
    total_subs = 0
    files_seen = 0
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(ROOT)
        except ValueError:
            continue
        if not should_process(p):
            continue
        files_seen += 1
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        new_text, n = rewrite_text(text)
        if n == 0 or new_text == text:
            continue
        p.write_text(new_text, encoding="utf-8")
        changed += 1
        total_subs += n
        print(f"  [{n:3d}] {rel}")
    print(f"\nfiles_seen={files_seen}  files_changed={changed}  substitutions={total_subs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
