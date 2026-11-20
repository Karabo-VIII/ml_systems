#!/usr/bin/env python3
"""Refuse commits that add large files or banned binary extensions.

Gitignored paths are skipped, so pipeline output under data/ and runs/ is
not scanned. This replaces the old CDAP audit, which stays in ml_systems_v1/
and is not invoked from this hook.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BANNED_SUFFIXES = {
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
}


def _staged_paths() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
        text=True,
    )
    return [p for p in out.split("\0") if p]


def _ignored(path: str) -> bool:
    result = subprocess.run(["git", "check-ignore", "-q", "--", path])
    return result.returncode == 0


def main() -> int:
    problems: list[str] = []
    for rel in _staged_paths():
        if _ignored(rel):
            continue
        path = Path(rel)
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in BANNED_SUFFIXES:
            problems.append(f"banned extension {suffix}: {rel}")
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"larger than 5 MB ({size} bytes): {rel}")
    if problems:
        print("pre-commit guard refused this commit:", file=sys.stderr)
        for line in problems:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
