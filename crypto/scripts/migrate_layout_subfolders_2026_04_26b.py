"""Migration: cadence/type subfolders within each processed/<layer>/ (2026-04-26 v3).

Reorganizes flat dated files into homogeneous cadence/type subfolders:

    OLD (flat, dated)                                            NEW (subfolder + dated)
    processed/chimera/btcusdt_v51_chimera_<DATE>.parquet      -> processed/chimera/dollar/...
    processed/chimera/btcusdt_v51_chimera_1d_<DATE>.parquet   -> processed/chimera/1d/...
    processed/chimera/btcusdt_v51_chimera_4h_<DATE>.parquet   -> processed/chimera/4h/...
    processed/chimera/btcusdt_v51_chimera_1h_<DATE>.parquet   -> processed/chimera/1h/...
    processed/chimera/btcusdt_v51_chimera_15m_<DATE>.parquet  -> processed/chimera/15m/...

    processed/chimera_legacy/btcusdt_v50_chimera_<DATE>.parquet -> processed/chimera_legacy/dollar/...
    processed/frontier/btcusdt_frontier_daily_<DATE>.parquet    -> processed/frontier/daily/...
    processed/bars/btcusdt_<bartype>_<DATE>.parquet             -> processed/bars/<bartype>/...
    processed/hawkes/<panel>_<DATE>.parquet                      -> processed/hawkes/daily/...
    processed/panels/<panel>_<DATE>.parquet                      -> processed/panels/daily/...

Idempotent: skips files already in the right subfolder.

Usage:
    python scripts/migrate_layout_subfolders_2026_04_26b.py --dry-run
    python scripts/migrate_layout_subfolders_2026_04_26b.py --execute
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "pipeline"))

import layout as _layout

DATA = PROJECT_ROOT / "data"
PROC = DATA / "processed"

CADENCE_PATTERNS = {
    "1d":  re.compile(r"^([a-z0-9]+)_v51_chimera_1d_(\d{8})\.parquet$"),
    "4h":  re.compile(r"^([a-z0-9]+)_v51_chimera_4h_(\d{8})\.parquet$"),
    "1h":  re.compile(r"^([a-z0-9]+)_v51_chimera_1h_(\d{8})\.parquet$"),
    "15m": re.compile(r"^([a-z0-9]+)_v51_chimera_15m_(\d{8})\.parquet$"),
    # dollar (no cad tag) MUST be checked AFTER cadence patterns above
    "dollar": re.compile(r"^([a-z0-9]+)_v51_chimera_(\d{8})\.parquet$"),
}

V50_PATTERN = re.compile(r"^([a-z0-9]+)_v50_chimera_(\d{8})\.parquet$")
FRONTIER_PATTERN = re.compile(r"^([a-z0-9]+)_frontier_daily_(\d{8})\.parquet$")

BAR_PATTERNS = {
    "dib":          re.compile(r"^([a-z0-9]+)_dib_(\d{8})\.parquet$"),
    "runs_tick":    re.compile(r"^([a-z0-9]+)_runs_tick_(\d{8})\.parquet$"),
    "runs_volume":  re.compile(r"^([a-z0-9]+)_runs_volume_(\d{8})\.parquet$"),
    "range":        re.compile(r"^([a-z0-9]+)_range_(\d{8})\.parquet$"),
    "adaptive_vol": re.compile(r"^([a-z0-9]+)_adaptive_vol_(\d{8})\.parquet$"),
}


def plan_chimera_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_CHIMERA
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        # Check cadence patterns first (longer/more specific), then dollar
        matched = False
        for cad in ("1d", "4h", "1h", "15m"):
            if CADENCE_PATTERNS[cad].match(f.name):
                moves.append((f, _layout.chimera_dir(cad) / f.name, f"chimera/{cad}"))
                matched = True
                break
        if not matched and CADENCE_PATTERNS["dollar"].match(f.name):
            moves.append((f, _layout.chimera_dir("dollar") / f.name, "chimera/dollar"))
    return moves


def plan_chimera_legacy_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_CHIMERA_LEGACY
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        if V50_PATTERN.match(f.name):
            moves.append((f, _layout.chimera_legacy_dir() / f.name, "chimera_legacy/dollar"))
    return moves


def plan_frontier_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_FRONTIER
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        if FRONTIER_PATTERN.match(f.name):
            moves.append((f, _layout.frontier_dir() / f.name, "frontier/daily"))
    return moves


def plan_bars_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_BARS
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        # Check most-specific patterns first (runs_tick before "runs", etc.)
        # Order matters: runs_tick / runs_volume share "runs_" prefix
        for bartype in ("runs_tick", "runs_volume", "adaptive_vol", "dib", "range"):
            if BAR_PATTERNS[bartype].match(f.name):
                moves.append((f, _layout.bars_dir(bartype) / f.name, f"bars/{bartype}"))
                break
    return moves


def plan_hawkes_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_HAWKES
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        moves.append((f, _layout.hawkes_dir() / f.name, "hawkes/daily"))
    return moves


def plan_panel_moves() -> list[tuple[Path, Path, str]]:
    moves = []
    src_dir = _layout.DIR_PANELS
    if not src_dir.exists():
        return moves
    for f in sorted(src_dir.iterdir()):
        if not f.is_file() or f.suffix != ".parquet":
            continue
        moves.append((f, _layout.panels_dir() / f.name, "panels/daily"))
    return moves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--out-manifest",
                    default="backups/BKP_20260426_PRE_LAYOUT_CLEANUP/MOVE_MANIFEST_subfolders.json")
    args = ap.parse_args()
    if not (args.dry_run or args.execute):
        ap.error("specify --dry-run or --execute")

    chimera_moves = plan_chimera_moves()
    legacy_moves = plan_chimera_legacy_moves()
    frontier_moves = plan_frontier_moves()
    bars_moves = plan_bars_moves()
    hawkes_moves = plan_hawkes_moves()
    panel_moves = plan_panel_moves()

    plan = {
        "chimera": [(str(s), str(d), tag) for s, d, tag in chimera_moves],
        "chimera_legacy": [(str(s), str(d), tag) for s, d, tag in legacy_moves],
        "frontier": [(str(s), str(d), tag) for s, d, tag in frontier_moves],
        "bars": [(str(s), str(d), tag) for s, d, tag in bars_moves],
        "hawkes": [(str(s), str(d), tag) for s, d, tag in hawkes_moves],
        "panels": [(str(s), str(d), tag) for s, d, tag in panel_moves],
    }

    print("\n=== Subfolder migration plan ===")
    for k, v in plan.items():
        print(f"  {k}: {len(v)} files")
    total = sum(len(v) for v in plan.values())
    print(f"  TOTAL: {total} files\n")

    out_path = PROJECT_ROOT / args.out_manifest
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(plan, indent=2))
    print(f"  manifest: {out_path.relative_to(PROJECT_ROOT)}")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return

    print("\n=== Executing ===")
    n_done = 0
    for layer_name, group in (
        ("chimera", chimera_moves), ("chimera_legacy", legacy_moves),
        ("frontier", frontier_moves), ("bars", bars_moves),
        ("hawkes", hawkes_moves), ("panels", panel_moves),
    ):
        for src, dst, tag in group:
            if not src.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # Already migrated; skip
                continue
            shutil.move(str(src), str(dst))
            n_done += 1
        print(f"  [OK] {layer_name}: {len(group)} moves")

    print(f"\n=== Migration complete: {n_done}/{total} files moved ===")


if __name__ == "__main__":
    main()
