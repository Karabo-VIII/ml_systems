"""_finalize_repath.py -- JSON-aware path rewriter for the 3-way repo split finalize step.

Run by crypto/scripts/finalize_split.ps1 AFTER the parent dir has been renamed
(<OLD> -> <NEW>). It rewrites the absolute paths inside .claude/settings.json and
.claude/settings.local.json so the hook commands, allow-list, additionalDirectories,
and the encoded Claude memory-dir reference point at the new layout:

  <...>/<OLD>/<cryptodir>/...  ->  <...>/<NEW>/crypto/<cryptodir>/...   (moved-into-crypto dirs)
  <...>/<OLD>/.claude|.git|... ->  <...>/<NEW>/.claude|.git|...        (root-level, stay)
  c--...-<OLD-encoded>         ->  c--...-<NEW-encoded>                (Claude project memory dir)

It PARSES the JSON (so backslash-escaping normalises -- a raw-text replace misses the
"\\\\runs" escaped form), walks every string value, applies the replace, validates, and
writes back with a .presplit_bak backup. Idempotent. No emoji (cp1252).

Usage:
  python _finalize_repath.py --root <ml_systems-abs-path> --old v4_crypto_stystem --new ml_systems [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# Top-level dirs that MOVED INTO crypto/ during the split (so an <OLD>/<dir> absolute path
# must become <NEW>/crypto/<dir>). Root-level survivors (.claude .git .venv models games
# harness) are NOT here -- they fall through to the generic <OLD> -> <NEW> replace.
CRYPTO_SUBDIRS = [
    "src", "scripts", "runs", "data", "config", "configs", "docs", "memory", "tests",
    "tools", "workspaces", "external", "backups", "archive", "logs", "scratch", "comms",
    "deliverables", "experiments", "plots", "reports", "catboost_info",
]


def _encoded(parent_name: str) -> str:
    """Claude Code encodes the launch-cwd path as the project memory-dir name: ':' and every
    path-separator AND '_' -> '-', and the leading drive letter lowercased. Launch cwd = the
    parent root. e.g. C:\\Users\\karab\\Documents\\coding\\v4_crypto_stystem
    -> c--Users-karab-Documents-coding-v4-crypto-stystem (verified against the live dir)."""
    raw = r"C:\Users\karab\Documents\coding" + "\\" + parent_name
    enc = raw.replace(":", "-").replace("\\", "-").replace("/", "-").replace("_", "-")
    return enc[0].lower() + enc[1:]


def rewrite_text(text: str, old: str, new: str) -> str:
    """Rewrite a single (already JSON-decoded -> single-backslash) string."""
    # 1) encoded memory-dir name first (distinct hyphen token)
    text = text.replace(_encoded(old), _encoded(new))
    # 2) moved-into-crypto dirs, both separator styles, BEFORE the generic replace
    for d in CRYPTO_SUBDIRS:
        text = text.replace(f"{old}\\{d}", f"{new}\\crypto\\{d}")
        text = text.replace(f"{old}/{d}", f"{new}/crypto/{d}")
    # 3) generic catch-all for root-level survivors (.claude, .git, the dir itself)
    text = text.replace(old, new)
    return text


def _walk(obj, old, new):
    if isinstance(obj, str):
        return rewrite_text(obj, old, new)
    if isinstance(obj, list):
        return [_walk(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {_walk(k, old, new): _walk(v, old, new) for k, v in obj.items()}
    return obj


def rewrite_json_file(path: str, old: str, new: str, dry: bool) -> int:
    if not os.path.isfile(path):
        print(f"  SKIP (absent): {path}")
        return 0
    with open(path, "r", encoding="utf-8") as fh:
        original = fh.read()
    try:
        data = json.loads(original)
    except Exception as e:
        print(f"  ERROR: {path} is not valid JSON ({e}); NOT touching.")
        return 2
    updated = json.dumps(_walk(data, old, new), indent=2, ensure_ascii=False) + "\n"
    # count value-level changes by comparing the re-serialised original (apples-to-apples)
    base = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if updated == base:
        print(f"  no change: {os.path.basename(path)}")
        return 0
    n = sum(1 for a, b in zip(base.splitlines(), updated.splitlines()) if a != b)
    if dry:
        print(f"  [dry-run] {os.path.basename(path)}: {n} value-line(s) would change")
        return 0
    with open(path + ".presplit_bak", "w", encoding="utf-8") as fh:
        fh.write(original)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(updated)
    print(f"  rewrote {os.path.basename(path)} ({n} line(s)); backup -> {os.path.basename(path)}.presplit_bak")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="absolute path of the renamed parent (e.g. ...\\ml_systems)")
    ap.add_argument("--old", default="v4_crypto_stystem")
    ap.add_argument("--new", default="ml_systems")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--settings", help="override settings.json path (default <root>/.claude/settings.json)")
    a = ap.parse_args()
    s = a.settings or os.path.join(a.root, ".claude", "settings.json")
    sl = os.path.join(os.path.dirname(s), "settings.local.json")
    print(f"repath: {a.old} -> {a.new}   (encoded memory dir: {_encoded(a.old)} -> {_encoded(a.new)})")
    rc = 0
    rc |= rewrite_json_file(s, a.old, a.new, a.dry_run)
    rc |= rewrite_json_file(sl, a.old, a.new, a.dry_run)
    return rc


if __name__ == "__main__":
    sys.exit(main())
