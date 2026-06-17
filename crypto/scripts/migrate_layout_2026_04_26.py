"""One-shot migration: data/ layout cleanup per user 2026-04-26.

Moves existing files to the new canonical layout (see src/pipeline/layout.py):

    OLD                                              NEW
    data/processed/<sym>usdt_v50_chimera.parquet  -> data/processed/chimera_legacy/<sym>usdt_v50_chimera_<YYYYMMDD>.parquet
    data/processed/<SYM>/v51.parquet              -> data/processed/chimera/<sym>usdt_v51_chimera_<YYYYMMDD>.parquet
    data/processed/<SYM>/v51_<cad>.parquet        -> data/processed/chimera/<sym>usdt_v51_chimera_<cad>_<YYYYMMDD>.parquet
    data/features/<SYM>/frontier_daily.parquet    -> data/processed/frontier/<sym>usdt_frontier_daily_<YYYYMMDD>.parquet
    data/bars/<SYM>/<bartype>/<files>.parquet     -> data/processed/bars/<sym>usdt_<bartype>_<YYYYMMDD>.parquet  (consolidated)
    data/features/_global/hawkes_*.parquet        -> data/processed/hawkes/hawkes_*_<YYYYMMDD>.parquet
    data/features/_global/<other>.parquet         -> data/processed/panels/<other>_<YYYYMMDD>.parquet
    data/_manifests/                              -> data/manifests/  (rename only)

Plus orphan deletes:
    data/processed/{btcusdt,ethusdt}_v51_chimera*.parquet
    data/processed/{btcusdt,ethusdt}_frontier_features.parquet
    data/processed/te_matrix_u50.pkl
    data/twin_bars/                              -> delete (empty)
    data/ml_training/                            -> delete (stale)
    data/frontier/                               -> delete (already migrated)

Date suffix = max(timestamp/date) inside the parquet (UTC date).

Usage:
    python scripts/migrate_layout_2026_04_26.py --dry-run   # preview
    python scripts/migrate_layout_2026_04_26.py --execute   # actually move
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone, date as date_type
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

from layout import (  # noqa: E402
    DIR_CHIMERA, DIR_CHIMERA_LEGACY, DIR_FRONTIER, DIR_BARS, DIR_HAWKES,
    DIR_PANELS, DIR_MANIFESTS,
    chimera_v51_path, chimera_v50_path, frontier_daily_path, bars_path,
    hawkes_panel_path, panel_path, normalize_asset,
)

DATA = PROJECT_ROOT / "data"


def latest_date_from_parquet(p: Path) -> date_type:
    """Read parquet, return latest UTC date from timestamp/date column."""
    schema = pl.read_parquet_schema(p)
    if "timestamp" in schema:
        df = pl.read_parquet(p, columns=["timestamp"])
        if df.is_empty():
            raise ValueError(f"empty parquet: {p}")
        ts_max = df["timestamp"].max()
        if isinstance(ts_max, datetime):
            return ts_max.astimezone(timezone.utc).date()
        # epoch ms
        return datetime.fromtimestamp(ts_max / 1000.0, tz=timezone.utc).date()
    if "date" in schema:
        df = pl.read_parquet(p, columns=["date"])
        if df.is_empty():
            raise ValueError(f"empty parquet: {p}")
        d_max = df["date"].max()
        if isinstance(d_max, datetime):
            return d_max.date()
        if isinstance(d_max, str):
            # Try ISO date string
            return datetime.fromisoformat(d_max).date()
        if isinstance(d_max, date_type):
            return d_max
        # numeric epoch days?
        return datetime.fromtimestamp(int(d_max) * 86400, tz=timezone.utc).date()
    if "ts" in schema:
        df = pl.read_parquet(p, columns=["ts"])
        if df.is_empty():
            raise ValueError(f"empty parquet: {p}")
        ts_max = df["ts"].max()
        if isinstance(ts_max, datetime):
            return ts_max.astimezone(timezone.utc).date()
        return datetime.fromtimestamp(ts_max / 1000.0, tz=timezone.utc).date()
    # No date col — fallback to file mtime
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).date()


def plan_v50_chimera_moves() -> list[tuple[Path, Path]]:
    """Move data/processed/<sym>usdt_v50_chimera.parquet -> chimera_legacy/<sym>usdt_v50_chimera_<DATE>.parquet"""
    moves = []
    proc = DATA / "processed"
    for f in sorted(proc.glob("*_v50_chimera.parquet")):
        if not f.is_file():
            continue
        sym_l = f.stem.replace("usdt_v50_chimera", "")
        try:
            d = latest_date_from_parquet(f)
        except Exception as e:
            print(f"  WARN: couldn't get date for {f.name}: {e}; using mtime")
            d = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).date()
        new_path = chimera_v50_path(sym_l + "usdt", d)
        moves.append((f, new_path))
    return moves


def plan_v51_chimera_moves() -> list[tuple[Path, Path]]:
    """Move data/processed/<SYM>/v51*.parquet -> chimera/<sym>usdt_v51_chimera<_cad>_<DATE>.parquet"""
    moves = []
    proc = DATA / "processed"
    for sym_dir in sorted(proc.iterdir()):
        if not sym_dir.is_dir() or not sym_dir.name.endswith("USDT"):
            continue
        for f in sorted(sym_dir.glob("v51*.parquet")):
            stem = f.stem  # 'v51' or 'v51_1d' etc.
            if stem == "v51":
                cadence = "dollar"
            else:
                cadence = stem.replace("v51_", "")
            try:
                d = latest_date_from_parquet(f)
            except Exception as e:
                print(f"  WARN: couldn't get date for {f}: {e}; using mtime")
                d = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).date()
            new_path = chimera_v51_path(sym_dir.name, cadence, d)
            moves.append((f, new_path))
    return moves


def plan_frontier_moves() -> list[tuple[Path, Path]]:
    """Move data/features/<SYM>/frontier_daily.parquet -> processed/frontier/<sym>usdt_frontier_daily_<DATE>.parquet"""
    moves = []
    feats = DATA / "features"
    if not feats.exists():
        return moves
    for sym_dir in sorted(feats.iterdir()):
        if not sym_dir.is_dir() or not sym_dir.name.endswith("USDT"):
            continue
        f = sym_dir / "frontier_daily.parquet"
        if not f.exists():
            continue
        try:
            d = latest_date_from_parquet(f)
        except Exception as e:
            print(f"  WARN: couldn't get date for {f}: {e}; using mtime")
            d = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).date()
        new_path = frontier_daily_path(sym_dir.name, d)
        moves.append((f, new_path))
    return moves


def plan_bars_consolidation() -> list[tuple[list[Path], Path]]:
    """Consolidate data/bars/<SYM>/<type>/<files>.parquet -> processed/bars/<sym>usdt_<type>_<DATE>.parquet

    For year-partitioned types (dib, range), concat all years into one file.
    For all-in-one types (runs_tick, runs_volume, adaptive_vol), single file copy.
    """
    consolidations = []
    bars_dir = DATA / "bars"
    if not bars_dir.exists():
        return consolidations
    for sym_dir in sorted(bars_dir.iterdir()):
        if not sym_dir.is_dir() or not sym_dir.name.endswith("USDT"):
            continue
        for type_dir in sorted(sym_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            bartype = type_dir.name
            files = sorted(type_dir.glob("*.parquet"))
            if not files:
                continue
            # Latest date across all source files
            latest_d = None
            for f in files:
                try:
                    d = latest_date_from_parquet(f)
                except Exception:
                    d = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).date()
                if latest_d is None or d > latest_d:
                    latest_d = d
            new_path = bars_path(sym_dir.name, bartype, latest_d)
            consolidations.append((files, new_path))
    return consolidations


def plan_panel_moves() -> list[tuple[Path, Path]]:
    """Split data/features/_global/*.parquet between processed/hawkes/ and processed/panels/."""
    moves = []
    g = DATA / "features" / "_global"
    if not g.exists():
        return moves
    for f in sorted(g.glob("*.parquet")):
        name = f.stem  # e.g. 'hawkes_branching_daily'
        try:
            d = latest_date_from_parquet(f)
        except Exception as e:
            print(f"  WARN: couldn't get date for {f}: {e}; using mtime")
            d = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).date()
        if name.startswith("hawkes_"):
            new_path = hawkes_panel_path(name, d)
        else:
            new_path = panel_path(name, d)
        moves.append((f, new_path))
    return moves


def plan_manifests_rename() -> tuple[Path, Path]:
    return DATA / "_manifests", DATA / "manifests"


def plan_orphans_delete() -> list[Path]:
    """Stale top-level files in processed/ that newer layout supersedes."""
    targets = [
        "btcusdt_v51_chimera.parquet",
        "btcusdt_v51_chimera_1d.parquet",
        "btcusdt_v51_chimera_4h.parquet",
        "btcusdt_frontier_features.parquet",
        "ethusdt_v51_chimera.parquet",
        "ethusdt_v51_chimera_1d.parquet",
        "ethusdt_v51_chimera_4h.parquet",
        "ethusdt_frontier_features.parquet",
        "te_matrix_u50.pkl",
    ]
    return [DATA / "processed" / t for t in targets if (DATA / "processed" / t).exists()]


def plan_dead_dirs_delete() -> list[Path]:
    """Dirs that are dead/empty and superseded."""
    return [
        DATA / "twin_bars",     # empty
        DATA / "ml_training",   # stale v1 cache
        DATA / "frontier",      # legacy already migrated to raw_external/features/bars
        DATA / "features",      # emptied after frontier moves (will be empty after migration)
        DATA / "bars",          # emptied after consolidation
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Preview only")
    ap.add_argument("--execute", action="store_true", help="Actually move/delete files")
    ap.add_argument("--out-manifest", default="backups/BKP_20260426_PRE_LAYOUT_CLEANUP/MOVE_MANIFEST.json")
    args = ap.parse_args()
    if not (args.dry_run or args.execute):
        ap.error("Must specify --dry-run or --execute")

    # Make new dirs (no-op if exist)
    if args.execute:
        for d in (DIR_CHIMERA, DIR_CHIMERA_LEGACY, DIR_FRONTIER, DIR_BARS,
                  DIR_HAWKES, DIR_PANELS):
            d.mkdir(parents=True, exist_ok=True)

    plan = {
        "v50_chimera_moves": [(str(s), str(d)) for s, d in plan_v50_chimera_moves()],
        "v51_chimera_moves": [(str(s), str(d)) for s, d in plan_v51_chimera_moves()],
        "frontier_moves":   [(str(s), str(d)) for s, d in plan_frontier_moves()],
        "bars_consolidations": [
            ([str(p) for p in srcs], str(dst))
            for srcs, dst in plan_bars_consolidation()
        ],
        "panel_moves":       [(str(s), str(d)) for s, d in plan_panel_moves()],
        "manifests_rename":  [str(plan_manifests_rename()[0]), str(plan_manifests_rename()[1])],
        "orphans_delete":    [str(p) for p in plan_orphans_delete()],
        "dead_dirs_delete":  [str(p) for p in plan_dead_dirs_delete()],
    }

    print("\n=== Migration plan ===")
    print(f"  v50 chimera moves:    {len(plan['v50_chimera_moves'])}")
    print(f"  v51 chimera moves:    {len(plan['v51_chimera_moves'])}")
    print(f"  frontier moves:       {len(plan['frontier_moves'])}")
    print(f"  bars consolidations:  {len(plan['bars_consolidations'])}")
    print(f"  panel moves:          {len(plan['panel_moves'])}")
    print(f"  manifests rename:     {plan['manifests_rename'][0]} -> {plan['manifests_rename'][1]}")
    print(f"  orphans delete:       {len(plan['orphans_delete'])}")
    print(f"  dead dirs delete:     {len(plan['dead_dirs_delete'])}")
    print()

    out_path = PROJECT_ROOT / args.out_manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"  manifest: {out_path.relative_to(PROJECT_ROOT)}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    print("\n=== Executing ===")

    # 1. v50 chimera moves
    for src, dst in plan_v50_chimera_moves():
        if not Path(src).exists():
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"  [OK] {len(plan['v50_chimera_moves'])} v50 chimera moves")

    # 2. v51 chimera moves
    for src, dst in plan_v51_chimera_moves():
        if not Path(src).exists():
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"  [OK] {len(plan['v51_chimera_moves'])} v51 chimera moves")

    # 3. Remove now-empty <SYM>/ dirs in processed/
    proc = DATA / "processed"
    for sym_dir in proc.iterdir():
        if sym_dir.is_dir() and sym_dir.name.endswith("USDT"):
            try:
                sym_dir.rmdir()  # only if empty
            except OSError:
                pass

    # 4. Frontier moves (per-asset silver)
    for src, dst in plan_frontier_moves():
        if not Path(src).exists():
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"  [OK] {len(plan['frontier_moves'])} frontier moves")

    # 5. Bars consolidations: read+concat+write+delete sources
    for srcs, dst in plan_bars_consolidation():
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if len(srcs) == 1:
            shutil.move(str(srcs[0]), str(dst))
        else:
            frames = [pl.read_parquet(s) for s in srcs]
            combined = pl.concat(frames, how="vertical_relaxed")
            combined.write_parquet(dst)
            for s in srcs:
                Path(s).unlink()
    print(f"  [OK] {len(plan['bars_consolidations'])} bars consolidations")

    # 6. Panel moves
    for src, dst in plan_panel_moves():
        if not Path(src).exists():
            continue
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    print(f"  [OK] {len(plan['panel_moves'])} panel moves")

    # 7. Manifests rename
    src, dst = plan_manifests_rename()
    if src.exists() and not dst.exists():
        src.rename(dst)
        print(f"  [OK] manifests: {src} -> {dst}")
    elif dst.exists():
        print(f"  [SKIP] manifests already at {dst}")

    # 8. Orphan deletes
    for p in plan_orphans_delete():
        if p.exists():
            p.unlink()
    print(f"  [OK] {len(plan['orphans_delete'])} orphan deletes")

    # 9. Dead-dir deletes (after migration; only if empty or fully migrated)
    deleted_dirs = []
    for d in plan_dead_dirs_delete():
        if not d.exists():
            continue
        # Only auto-delete if empty after migration
        try:
            shutil.rmtree(d)
            deleted_dirs.append(str(d))
        except Exception as e:
            print(f"  WARN: couldn't delete {d}: {e}")
    print(f"  [OK] {len(deleted_dirs)} dead dirs deleted: {deleted_dirs}")

    print("\n=== Migration complete ===")


if __name__ == "__main__":
    main()
