"""One-shot migration: data/frontier/ -> new SOTA layout (data/raw_external/, data/features/, data/bars/).

Idempotent: safe to re-run; only moves files that haven't been moved yet.
Does NOT delete the source files until --commit is passed.

Usage:
  python scripts/migrate_data_layout_v51.py --plan        # show what would move
  python scripts/migrate_data_layout_v51.py --execute     # do the moves (copies)
  python scripts/migrate_data_layout_v51.py --execute --delete-source  # also delete after copy
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FRONTIER = DATA / "frontier"


# Plan: source-glob -> destination-rule
# destination-rule is either a fixed Path or a callable(src) -> Path
MIGRATIONS = [
    # raw_external (non-Binance bronze)
    {
        "src_dir": FRONTIER / "etf",
        "dst_dir": DATA / "raw_external" / "farside",
        "patterns": ["*.parquet"],
    },
    {
        "src_dir": FRONTIER / "defillama",
        "dst_dir": DATA / "raw_external" / "defillama",
        "patterns": ["*.parquet"],
    },
    {
        "src_dir": FRONTIER / "funding",
        "dst_dir": DATA / "raw_external" / "binance_futures_panels",
        "patterns": ["*.parquet"],
    },
    {
        "src_dir": FRONTIER / "dvol",
        "dst_dir": DATA / "raw_external" / "deribit",
        "patterns": ["*.parquet"],
    },
    {
        "src_dir": FRONTIER / "spreads",
        "dst_dir": DATA / "raw_external" / "coinbase_okx_bybit",
        "patterns": ["*.parquet"],
    },
    {
        "src_dir": FRONTIER / "social",
        "dst_dir": DATA / "raw_external" / "wikipedia",
        "patterns": ["*.parquet"],
    },
    # features/_global (per-source panels with asset col)
    {
        "src_dir": FRONTIER / "metrics",
        "dst_dir": DATA / "features" / "_global",
        "patterns": ["s3_features_long.parquet", "s3_metrics_panel.parquet"],
    },
    {
        "src_dir": FRONTIER / "basis",
        "dst_dir": DATA / "features" / "_global",
        "patterns": ["basis_features_long.parquet", "spot_klines_daily.parquet"],
    },
    {
        "src_dir": FRONTIER / "liquidations",
        "dst_dir": DATA / "features" / "_global",
        "patterns": ["liq_features_long.parquet", "liq_daily_approx.parquet"],
    },
    {
        "src_dir": FRONTIER / "whale",
        "dst_dir": DATA / "features" / "_global",
        "patterns": ["whale_activity_daily.parquet"],
    },
    {
        "src_dir": FRONTIER / "hawkes_enh",
        "dst_dir": DATA / "features" / "_global",
        "patterns": ["hawkes_branching_daily.parquet", "hawkes_enh_daily.parquet"],
    },
    # bars (per-asset bar fabric); files like BTCUSDT_dib_2025.parquet
    # We extract asset from filename and route per-asset
]


def migrate_per_asset_bars(plan_only: bool, delete_source: bool) -> list[tuple[Path, Path, str]]:
    """Per-asset bar fabric: route by exact filename pattern per source dir.

    Filename conventions vary:
      data/frontier/dib/BTCUSDT_dib_2025.parquet            -> bars/BTCUSDT/dib/2025.parquet
      data/frontier/range_bars/BTCUSDT_range_2025.parquet   -> bars/BTCUSDT/range/2025.parquet
      data/frontier/runs_bars/BTCUSDT_tick_runs.parquet     -> bars/BTCUSDT/runs_tick/all.parquet
      data/frontier/runs_bars/BTCUSDT_vol_runs.parquet      -> bars/BTCUSDT/runs_volume/all.parquet
      data/frontier/adaptive_bars/BTCUSDT_adaptive_vol.parquet -> bars/BTCUSDT/adaptive_vol/all.parquet
    """
    moves = []

    # DIB: <SYM>_dib_<YEAR>.parquet
    for fp in (FRONTIER / "dib").glob("*.parquet") if (FRONTIER / "dib").exists() else []:
        parts = fp.stem.split("_")
        if len(parts) >= 3 and parts[1] == "dib":
            sym, year = parts[0], parts[2]
            moves.append((fp, DATA / "bars" / sym / "dib" / f"{year}.parquet", "dib_bar"))

    # range: <SYM>_range_<YEAR>.parquet
    for fp in (FRONTIER / "range_bars").glob("*.parquet") if (FRONTIER / "range_bars").exists() else []:
        parts = fp.stem.split("_")
        if len(parts) >= 3 and parts[1] == "range":
            sym, year = parts[0], parts[2]
            moves.append((fp, DATA / "bars" / sym / "range" / f"{year}.parquet", "range_bar"))

    # runs: <SYM>_<tick|vol>_runs.parquet -> runs_tick / runs_volume
    runs_map = {"tick": "runs_tick", "vol": "runs_volume"}
    for fp in (FRONTIER / "runs_bars").glob("*.parquet") if (FRONTIER / "runs_bars").exists() else []:
        parts = fp.stem.split("_")
        if len(parts) >= 3 and parts[2] == "runs" and parts[1] in runs_map:
            sym = parts[0]
            bar_type = runs_map[parts[1]]
            moves.append((fp, DATA / "bars" / sym / bar_type / "all.parquet", "runs_bar"))

    # adaptive_vol: <SYM>_adaptive_vol.parquet -> bars/<SYM>/adaptive_vol/all.parquet
    for fp in (FRONTIER / "adaptive_bars").glob("*.parquet") if (FRONTIER / "adaptive_bars").exists() else []:
        parts = fp.stem.split("_")
        if len(parts) >= 3 and parts[1] == "adaptive" and parts[2] == "vol":
            sym = parts[0]
            moves.append((fp, DATA / "bars" / sym / "adaptive_vol" / "all.parquet", "adaptive_bar"))

    return moves


def execute_moves(moves: list[tuple[Path, Path, str]], plan_only: bool, delete_source: bool) -> dict:
    """Execute all moves; return summary."""
    n_planned = n_skipped = n_done = n_err = 0
    for src, dst, label in moves:
        if not src.exists():
            n_skipped += 1
            continue
        if dst.exists():
            # Already moved
            n_skipped += 1
            continue
        if plan_only:
            print(f"  [plan] {label} {src.relative_to(DATA)} -> {dst.relative_to(DATA)}")
            n_planned += 1
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_done += 1
            if delete_source:
                src.unlink()
        except Exception as e:
            print(f"  [ERR] {src} -> {dst}: {e}")
            n_err += 1
    return {"planned": n_planned, "skipped": n_skipped, "done": n_done, "err": n_err}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="Print plan, no changes.")
    ap.add_argument("--execute", action="store_true", help="Execute the moves (copies).")
    ap.add_argument("--delete-source", action="store_true",
                    help="After copy, delete source. Only with --execute.")
    args = ap.parse_args()

    if not (args.plan or args.execute):
        ap.error("must pass --plan or --execute")

    moves = []
    # Build flat moves list from MIGRATIONS specs
    for spec in MIGRATIONS:
        src_dir = spec["src_dir"]
        dst_dir = spec["dst_dir"]
        if not src_dir.exists():
            continue
        for pattern in spec["patterns"]:
            for fp in src_dir.glob(pattern):
                if fp.is_file():
                    dst = dst_dir / fp.name
                    moves.append((fp, dst, src_dir.name))

    moves.extend(migrate_per_asset_bars(args.plan, args.delete_source))

    print(f"Total moves to evaluate: {len(moves)}")
    summary = execute_moves(moves, plan_only=args.plan, delete_source=args.delete_source)
    print(f"Result: {summary}")
    if args.execute and not args.delete_source:
        print(f"\nNote: source files preserved. Re-run with --delete-source to remove duplicates.")


if __name__ == "__main__":
    main()
