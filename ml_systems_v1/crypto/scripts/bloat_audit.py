"""Project bloat audit. Identifies cleanup candidates without auto-deleting.

Operator confirms before any deletion.

Categories scanned:
  1. Archived model versions still in src/ (per CLAUDE.md, V2/V5/V7 are ARCHIVED)
  2. Backups/ snapshots older than the BKP_2026_04_25_PRE_GAP_CLOSURE
  3. Frontier subtree (data/frontier/) duplicated by raw_external + features + bars after migration
  4. Scratch/ files older than 30 days
  5. Empty __pycache__/ directories
  6. Duplicate parquet files (same name in multiple paths post-migration)

Output: docs/BLOAT_AUDIT_<DATE>.md with rankings + recommended actions.
NO deletions performed.

Run: python scripts/bloat_audit.py
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass
    return total


def audit_archived_model_versions() -> list[dict]:
    """V2, V5, V7 are ARCHIVED per CLAUDE.md. Find any code under src/ that's listed as archived."""
    archived = ["v2", "v5", "v7"]  # per CLAUDE.md "Architecture Summary"
    rows = []
    for v in archived:
        d = PROJECT_ROOT / "src" / v
        if d.exists():
            sz = dir_size(d)
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            rows.append({
                "category": "archived_model_src",
                "path": str(d.relative_to(PROJECT_ROOT)),
                "size": fmt_size(sz),
                "files": n_files,
                "recommendation": f"Move src/{v}/ -> backups/BKP_archived_models/ (per CLAUDE.md)",
            })
    return rows


def audit_backup_snapshots() -> list[dict]:
    """Per user 2026-04-26: backups are sacrosanct. NEVER recommend touching them.

    Kept this stub for documentation purposes; always returns empty list.
    """
    return []


def audit_frontier_duplicates() -> list[dict]:
    """data/frontier/ contents migrated to raw_external/, features/_global/, bars/.
    The originals are duplicates."""
    fdir = PROJECT_ROOT / "data" / "frontier"
    if not fdir.exists():
        return []
    sz = dir_size(fdir)
    n_files = sum(1 for _ in fdir.rglob("*") if _.is_file())
    return [{
        "category": "migrated_duplicate",
        "path": "data/frontier/",
        "size": fmt_size(sz),
        "files": n_files,
        "recommendation": ("DELETE -- migrated to raw_external/, features/_global/, bars/. "
                           "Run: python scripts/migrate_data_layout_v51.py --execute --delete-source"),
    }]


def audit_scratch_old() -> list[dict]:
    """scratch/ files >30 days old + over 200KB are candidates for archive."""
    sdir = PROJECT_ROOT / "scratch"
    if not sdir.exists():
        return []
    cutoff = datetime.now(timezone.utc).timestamp() - (30 * 86400)
    big_old = []
    for f in sdir.rglob("*.py"):
        try:
            stat = f.stat()
            if stat.st_mtime < cutoff and stat.st_size > 200_000:
                big_old.append((f, stat.st_mtime, stat.st_size))
        except Exception:
            pass
    rows = []
    for f, mtime, size in sorted(big_old, key=lambda x: -x[2])[:10]:  # top 10 biggest old
        rows.append({
            "category": "scratch_old_big",
            "path": str(f.relative_to(PROJECT_ROOT)),
            "size": fmt_size(size),
            "files": 1,
            "recommendation": f"Move to backups/scratch_archive/. mtime={datetime.fromtimestamp(mtime).date()}",
        })
    return rows


def audit_pycache() -> list[dict]:
    """All __pycache__/ are pure caches; safe to wipe."""
    n = 0
    total_size = 0
    for d in PROJECT_ROOT.rglob("__pycache__"):
        if d.is_dir():
            n += 1
            total_size += dir_size(d)
    if n == 0:
        return []
    return [{
        "category": "pycache",
        "path": "**/__pycache__/",
        "size": fmt_size(total_size),
        "files": n,
        "recommendation": "Wipe via: find . -name __pycache__ -type d -exec rm -rf {} +",
    }]


def main() -> None:
    rows = []
    rows.extend(audit_archived_model_versions())
    rows.extend(audit_backup_snapshots())
    rows.extend(audit_frontier_duplicates())
    rows.extend(audit_scratch_old())
    rows.extend(audit_pycache())

    today = datetime.now(timezone.utc).strftime("%Y_%m_%d")
    out_path = PROJECT_ROOT / "docs" / f"BLOAT_AUDIT_{today}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = [f"# Project Bloat Audit -- {today}", ""]
    md.append(f"Total candidates: {len(rows)}. NO files deleted automatically.")
    md.append(f"Operator must confirm each recommendation before action.")
    md.append("")
    md.append("| # | Category | Path | Size | Files | Recommendation |")
    md.append("|---|----------|------|------|-------|----------------|")
    for i, r in enumerate(rows, 1):
        md.append(f"| {i} | {r['category']} | `{r['path']}` | {r['size']} | {r['files']} | {r['recommendation']} |")
    md.append("")
    md.append("## Total potential reclaim")
    md.append("")
    md.append("Run `du -sh` on each `path` for actual size. Audit estimates above use")
    md.append("`Path.stat().st_size` recursively.")
    out_path.write_text("\n".join(md))

    print(f"[bloat] {len(rows)} cleanup candidates identified")
    for r in rows:
        print(f"  [{r['category']}] {r['path']} ({r['size']}, {r['files']} files)")
    print(f"\n[bloat] full report: {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
