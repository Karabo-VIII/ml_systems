"""The artifact registry in docs/DATA_CONTRACT.md must cover every tracked file."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "docs" / "DATA_CONTRACT.md"
BEGIN = "<!-- begin:tracked-files -->"
END = "<!-- end:tracked-files -->"
ARCHIVE_PREFIX = "ml_systems_v1/"


def _repo_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


def _tracked_outside_archive(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    files = [p.decode() for p in result.stdout.split(b"\0") if p]
    return {p for p in files if not p.startswith(ARCHIVE_PREFIX)}


def _registry_paths(text: str) -> list[str]:
    if BEGIN not in text or END not in text:
        raise AssertionError(
            f"docs/DATA_CONTRACT.md must contain {BEGIN} and {END} around the tracked-file table"
        )
    block = text.split(BEGIN, 1)[1].split(END, 1)[0]
    paths: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        path = cells[0].strip().strip("`")
        if not path or path == "path" or set(path) <= {"-"}:
            continue
        paths.append(path)
    return paths


def test_tracked_files_match_artifact_registry():
    root = _repo_root()
    text = CONTRACT.read_text(encoding="utf-8")
    listed = _registry_paths(text)
    assert listed, "tracked-file registry table is empty"
    duplicates = sorted({p for p in listed if listed.count(p) > 1})
    assert not duplicates, f"duplicate registry paths: {duplicates}"

    tracked = _tracked_outside_archive(root)
    unlisted = sorted(tracked - set(listed))
    assert not unlisted, (
        "tracked files missing from the artifact registry in docs/DATA_CONTRACT.md "
        f"(add a row in the same commit): {unlisted}"
    )

    missing = sorted(p for p in listed if not (root / p).exists())
    assert not missing, f"registry lists files that no longer exist: {missing}"
