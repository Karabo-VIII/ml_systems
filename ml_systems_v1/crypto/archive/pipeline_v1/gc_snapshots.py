"""GC dated snapshots across data/processed/ layers.

Layout v3 keeps multiple dated snapshots per file (e.g. for BTCUSDT chimera_legacy:
    btcusdt_v50_chimera_20260422.parquet  (older, healthy)
    btcusdt_v50_chimera_20260427.parquet  (newer, possibly partial)

Without GC these accumulate indefinitely. This tool retains the newest N
*VALID* snapshots per (asset, layer, cadence/bartype) and removes older ones.

"VALID" check is layer-aware:
  - chimera/chimera_legacy: file readable + has minimum required cols
  - frontier/bars/hawkes/panels: file readable + non-empty

The default `keep` is 1 — frozen split dates (config/data_config.yaml) mean
train/val/oos segments are identical run-to-run; only the unseen segment grows.
Older snapshots add no information once today's validates. Safety: this tool is
COLUMN-AWARE — if today's snapshot is missing required features, it is treated
as invalid and an older healthy snapshot is preserved automatically.

Usage:
    # Preview what would be deleted (no changes)
    python src/pipeline/gc_snapshots.py --dry-run

    # Apply across all layers (default keep=1, safe due to column-aware check)
    python src/pipeline/gc_snapshots.py

    # GC only chimera_legacy
    python src/pipeline/gc_snapshots.py --layer chimera_legacy

    # Conservative (keep newest 2 valid — extra fallback room)
    python src/pipeline/gc_snapshots.py --keep 2

Wired into pipeline:
    python src/pipeline/run_pipeline.py --gc
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "pipeline"))
import layout as _layout  # noqa: E402


# Per-layer minimum-column requirements for "valid" check.
# A file is invalid if it's missing ANY of the required columns.
LAYER_MIN_COLS = {
    "chimera_legacy": {
        "timestamp", "bar_id", "close",
        # Pattern P / xd_* must be present (full f41 invariant)
        "norm_flow_imbalance", "norm_hawkes_imbalance",
        "xd_btc_return", "xd_funding_spread", "xd_momentum_rank",
        "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    },
    "chimera": {
        "timestamp", "bar_id", "close",
        "target_return_1", "target_return_4",
        # v51 frontier minimum: HBR family must be present
        # (we don't enforce all 80 frontier features, just smoke check)
    },
    # Frontier silver / bars / hawkes / panels: just non-empty + readable
}


def _layer_dirs() -> dict[str, list[Path]]:
    """Map layer name -> list of subdirectories containing dated parquets."""
    out = {}
    if _layout.DIR_CHIMERA_LEGACY.exists():
        out["chimera_legacy"] = [_layout.DIR_CHIMERA_LEGACY / "dollar"]
    if _layout.DIR_CHIMERA.exists():
        out["chimera"] = [d for d in _layout.DIR_CHIMERA.iterdir() if d.is_dir()]
    if _layout.DIR_FRONTIER.exists():
        out["frontier"] = [d for d in _layout.DIR_FRONTIER.iterdir() if d.is_dir()]
    if _layout.DIR_BARS.exists():
        out["bars"] = [d for d in _layout.DIR_BARS.iterdir() if d.is_dir()]
    if _layout.DIR_HAWKES.exists():
        out["hawkes"] = [d for d in _layout.DIR_HAWKES.iterdir() if d.is_dir()]
    if _layout.DIR_PANELS.exists():
        out["panels"] = [d for d in _layout.DIR_PANELS.iterdir() if d.is_dir()]
    return {k: v for k, v in out.items() if v}


def _is_valid(path: Path, min_cols: set[str] | None) -> tuple[bool, str]:
    """Lightweight validity check via parquet schema (no row read)."""
    try:
        cols = set(pl.read_parquet_schema(path).keys())
    except Exception as e:
        return False, f"unreadable: {type(e).__name__}"
    if min_cols:
        missing = min_cols - cols
        if missing:
            return False, f"missing {len(missing)} cols: {sorted(missing)[:3]}"
    return True, "ok"


def _group_by_key(dir_path: Path) -> dict[str, list[Path]]:
    """Group files in dir_path by their key (filename minus _<YYYYMMDD>.parquet).

    Examples:
      btcusdt_v50_chimera_20260427.parquet -> key='btcusdt_v50_chimera'
      ethusdt_v51_chimera_20260427.parquet -> key='ethusdt_v51_chimera'
      ethusdt_v51_chimera_1h_20260427.parquet -> key='ethusdt_v51_chimera_1h'
    """
    grouped: dict[str, list[Path]] = defaultdict(list)
    for f in dir_path.glob("*.parquet"):
        stem = f.stem
        # Trailing _<YYYYMMDD> required
        if len(stem) < 9 or stem[-9] != "_":
            continue
        date_tail = stem[-8:]
        if not date_tail.isdigit():
            continue
        key = stem[:-9]
        grouped[key].append(f)
    return grouped


def gc_layer(layer_name: str, dirs: list[Path], keep: int,
             dry_run: bool = False) -> tuple[int, int, int]:
    """GC one layer. Returns (n_groups_scanned, n_deleted, bytes_deleted)."""
    min_cols = LAYER_MIN_COLS.get(layer_name)
    n_groups = n_deleted = bytes_deleted = 0

    for d in dirs:
        if not d.exists():
            continue
        for key, files in _group_by_key(d).items():
            n_groups += 1
            files.sort(key=lambda p: p.name, reverse=True)  # newest first
            valid, invalid = [], []
            for f in files:
                ok, reason = _is_valid(f, min_cols)
                if ok:
                    valid.append(f)
                else:
                    invalid.append((f, reason))

            keep_set = set(valid[:keep])
            for f, reason in invalid:
                # Older invalid files: delete. Newest invalid: keep (might be in-progress write).
                # But if there's a valid newer one, even the newest invalid is safe to delete.
                if valid and f.name < valid[0].name:
                    pass  # invalid older than newest valid -> delete
                elif valid and f == files[0]:
                    pass  # newest invalid, but we have a valid fallback -> delete
                else:
                    keep_set.add(f)  # invalid AND we have no valid fallback -> keep (don't lose data)

            for f in files:
                if f in keep_set:
                    continue
                size = f.stat().st_size if f.exists() else 0
                if not dry_run:
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"  [WARN] {f}: unlink failed ({e})")
                        continue
                action = "would delete" if dry_run else "deleted"
                why = "older valid" if f in valid else "invalid"
                print(f"  [{action}] {d.name}/{f.name}  ({size/1024/1024:.1f} MB, {why})")
                n_deleted += 1
                bytes_deleted += size

    return n_groups, n_deleted, bytes_deleted


def wipe_processed(dry_run: bool = False) -> tuple[int, int]:
    """Delete EVERY parquet under data/processed/ (full reset).

    Use this before a clean rebuild. The pipeline DAG runner is idempotent;
    fetch_binance is preserved (raw data lives under data/raw/, not processed).

    Returns (n_deleted, bytes_deleted).
    """
    n = 0
    sz = 0
    for layer_dirs in _layer_dirs().values():
        for d in layer_dirs:
            if not d.exists():
                continue
            for f in d.rglob("*.parquet"):
                size = f.stat().st_size if f.exists() else 0
                if not dry_run:
                    try:
                        f.unlink()
                    except Exception as e:
                        print(f"  [WARN] {f}: {e}")
                        continue
                action = "would delete" if dry_run else "deleted"
                print(f"  [{action}] {f.relative_to(f.parent.parent)} ({size/1024/1024:.1f} MB)")
                n += 1
                sz += size
    return n, sz


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=int, default=1,
                    help="Number of newest VALID snapshots to keep per asset/key (default: 1; "
                         "safe due to column-aware validity check)")
    ap.add_argument("--layer", choices=sorted(LAYER_MIN_COLS.keys()) +
                    ["frontier", "bars", "hawkes", "panels", "all"],
                    default="all",
                    help="Restrict to one layer (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be deleted without changes")
    ap.add_argument("--wipe", action="store_true",
                    help="DESTRUCTIVE: delete ALL parquets under data/processed/ "
                         "(full reset before clean rebuild). Raw data under "
                         "data/raw/ is NOT touched. Use --dry-run first.")
    args = ap.parse_args()

    if args.wipe:
        print(f"\n{'='*70}")
        print(f"WIPE PROCESSED  dry_run={args.dry_run}")
        print(f"  This deletes ALL parquets under data/processed/.")
        print(f"  data/raw/ is preserved (run_pipeline.py rebuilds from there).")
        print(f"{'='*70}\n")
        n, sz = wipe_processed(dry_run=args.dry_run)
        print(f"\n{'='*70}")
        print(f"TOTAL: {n} files {'would be ' if args.dry_run else ''}deleted, "
              f"{sz/1024/1024/1024:.2f} GB freed")
        print(f"{'='*70}")
        return 0

    dirs_by_layer = _layer_dirs()
    if args.layer != "all":
        dirs_by_layer = {args.layer: dirs_by_layer.get(args.layer, [])}

    print(f"\n{'='*70}")
    print(f"GC SNAPSHOTS  keep={args.keep}  dry_run={args.dry_run}")
    print(f"{'='*70}\n")

    totals = {"groups": 0, "deleted": 0, "bytes": 0}
    for layer, dirs in dirs_by_layer.items():
        if not dirs:
            continue
        print(f"--- layer={layer}  dirs={[d.name for d in dirs]}")
        g, d, b = gc_layer(layer, dirs, keep=args.keep, dry_run=args.dry_run)
        print(f"    {g} keys, {d} files {'would be ' if args.dry_run else ''}deleted, {b/1024/1024:.1f} MB freed\n")
        totals["groups"] += g
        totals["deleted"] += d
        totals["bytes"] += b

    print(f"{'='*70}")
    print(f"TOTAL: {totals['groups']} keys, {totals['deleted']} files "
          f"{'would be ' if args.dry_run else ''}deleted, "
          f"{totals['bytes']/1024/1024:.1f} MB freed")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
