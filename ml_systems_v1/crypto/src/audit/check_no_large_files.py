"""Pre-commit guard against large / binary tracked files.

Runs against `git diff --cached` (the staging area). Blocks commits that:
  1. Add a file > MAX_FILE_SIZE_MB (default 5 MB) without explicit allow
  2. Add any file matching BANNED_EXTENSIONS

Why this exists: the project repo accumulated 119 GB of historical
chimera parquet blobs (data/processed/chimera*/*.parquet at 1-2 GB
each) before .gitignore caught them. This hook is the second layer
of defense -- gitignore catches by path, this hook catches by
content/extension on the staged add.

Bypass once: SKIP_LARGE_FILES_CHECK=1 git commit -m "..."

Exit codes:
    0 = clean (no large or banned-ext files staged)
    2 = blocked (>=1 violation; commit halted)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

BANNED_EXTENSIONS = {
    # Tabular binaries
    ".parquet", ".feather", ".arrow",
    # Model checkpoints / pickled state
    ".pt", ".pth", ".pkl", ".pickle", ".bin",
    # NumPy / HDF5
    ".npy", ".npz", ".h5", ".hdf5",
    # Image / video / audio (rarely belong in a quant repo)
    ".mp4", ".avi", ".mov", ".webm", ".mkv",
    ".wav", ".mp3", ".ogg",
}

# Files explicitly allowed even if they hit the size or extension rule.
# Use sparingly. Add the relative path (from repo root).
EXPLICIT_ALLOWLIST = {
    # add paths here only with strong justification
}


def staged_added_files() -> list[Path]:
    """Files newly added to the staging area (status A or M with size jump)."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=AM"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"[no-large-files] git diff --cached failed: {result.stderr}", file=sys.stderr)
        return []
    return [Path(p) for p in result.stdout.strip().splitlines() if p]


def check_file(path: Path) -> tuple[bool, str]:
    """Return (is_violation, reason) for a single staged file."""
    if str(path) in EXPLICIT_ALLOWLIST:
        return False, "allowlisted"
    if not path.exists():
        # could be a delete or rename; skip
        return False, "not present"
    ext = path.suffix.lower()
    if ext in BANNED_EXTENSIONS:
        return True, f"banned extension: {ext}"
    try:
        size = path.stat().st_size
    except OSError:
        return False, "stat failed"
    if size > MAX_FILE_SIZE_BYTES:
        return True, f"size {size / 1024 / 1024:.1f} MB > {MAX_FILE_SIZE_MB} MB"
    return False, ""


def main() -> int:
    if os.environ.get("SKIP_LARGE_FILES_CHECK") == "1":
        print("[no-large-files] skipped via SKIP_LARGE_FILES_CHECK=1", file=sys.stderr)
        return 0

    files = staged_added_files()
    if not files:
        return 0

    violations = []
    for f in files:
        is_violation, reason = check_file(f)
        if is_violation:
            violations.append((f, reason))

    if not violations:
        return 0

    print("", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("  PRE-COMMIT BLOCKED -- large or banned-extension files staged", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    for f, reason in violations:
        print(f"  {f}  ({reason})", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Reasons this hook exists:", file=sys.stderr)
    print(f"    * Files > {MAX_FILE_SIZE_MB} MB bloat git history (cannot be reclaimed without history rewrite)", file=sys.stderr)
    print(f"    * Banned extensions: {sorted(BANNED_EXTENSIONS)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Fix:", file=sys.stderr)
    print("    1. git rm --cached <path>          (untrack the file)", file=sys.stderr)
    print("    2. add the path to .gitignore", file=sys.stderr)
    print("    3. retry the commit", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Bypass ONCE (discouraged; document in commit message):", file=sys.stderr)
    print("    SKIP_LARGE_FILES_CHECK=1 git commit -m \"...\"", file=sys.stderr)
    print("", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
