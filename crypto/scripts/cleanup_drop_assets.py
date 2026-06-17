"""
Purge DROP-tier assets from the project.

Reads the latest screener CSV (logs/universe_screen_*.csv) to identify
liquidity_tier == DROP / DROP_NO_DATA assets, then plans + executes
deletion across:

  RAW:          data/raw/<SYM>USDT/                   (heaviest)
  Silver:       data/processed/frontier/daily/<sym>usdt_frontier_daily_*.parquet
  Bars:         data/processed/bars/<bartype>/<sym>usdt_<bartype>_*.parquet
  Chimera v50:  data/processed/chimera_legacy/dollar/<sym>usdt_v50_chimera_*.parquet
  Chimera v51:  data/processed/chimera/{dollar,1d,4h,1h,15m}/<sym>usdt_v51_chimera*_*.parquet
  Manifests:    data/manifests/v51_<SYMBOL>.json

Default: dry-run. Pass --apply to execute deletion. Always prints the
complete plan and total bytes to be reclaimed first.

Universe yaml updates are NOT done here — handled by a sibling step
that moves DROP names into the excluded_assets list with rationale.

Usage:
    python scripts/cleanup_drop_assets.py             # dry-run
    python scripts/cleanup_drop_assets.py --apply     # execute
    python scripts/cleanup_drop_assets.py --csv logs/universe_screen_X.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def _latest_csv() -> Path | None:
    cands = sorted(LOG_DIR.glob("universe_screen_*.csv"))
    return cands[-1] if cands else None


def load_drop_assets(csv_path: Path) -> Set[str]:
    """Return the set of SYMBOLS marked DROP / DROP_NO_DATA in the screener."""
    drop = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sym = row["asset"].strip().upper()
            tier = (row.get("tier") or "").strip().upper()
            if tier in ("DROP", "DROP_NO_DATA"):
                drop.add(sym)
    return drop


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def collect_paths_for_asset(sym: str) -> List[Path]:
    """Every project path tied to this asset, ranked by importance."""
    sym_l = sym.lower().replace("USDT", "")             # e.g. "aliceusdt"
    sym_upper = sym.upper()
    paths: List[Path] = []

    # RAW (largest disk usage; aggTrades + funding + metrics)
    raw_dir = PROJECT_ROOT / "data" / "raw" / sym_upper
    if raw_dir.exists():
        paths.append(raw_dir)

    # Frontier silver
    silver_glob = (PROJECT_ROOT / "data" / "processed" / "frontier" / "daily").glob(
        f"{sym_l}_frontier_daily_*.parquet"
    )
    paths.extend(silver_glob)

    # Bars (5 bartypes)
    bars_root = PROJECT_ROOT / "data" / "processed" / "bars"
    if bars_root.exists():
        for bartype_dir in bars_root.iterdir():
            if not bartype_dir.is_dir():
                continue
            paths.extend(bartype_dir.glob(f"{sym_l}_*.parquet"))

    # Chimera v50 (legacy)
    chim_legacy = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    paths.extend(chim_legacy.glob(f"{sym_l}_v50_chimera*.parquet"))

    # Chimera v51 (5 cadences)
    chim_v51_root = PROJECT_ROOT / "data" / "processed" / "chimera"
    if chim_v51_root.exists():
        for cad_dir in chim_v51_root.iterdir():
            if cad_dir.is_dir():
                paths.extend(cad_dir.glob(f"{sym_l}_v51_chimera*.parquet"))

    # Manifests
    manifests = PROJECT_ROOT / "data" / "manifests"
    paths.extend(manifests.glob(f"v51_{sym_upper}.json"))

    return paths


def _path_size(p: Path) -> int:
    try:
        if p.is_file():
            return p.stat().st_size
        if p.is_dir():
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None,
                    help="screener csv path (default: latest in logs/)")
    ap.add_argument("--apply", action="store_true",
                    help="execute deletion (default: dry-run)")
    ap.add_argument("--log-decision", default=None,
                    help="path to a JSON log of what was deleted (default: "
                         "logs/cleanup_drop_assets_<DATE>.json)")
    args = ap.parse_args()

    csv_path = Path(args.csv) if args.csv else _latest_csv()
    if csv_path is None or not csv_path.exists():
        print("[error] no screener csv found in logs/. "
              "Run scripts/screen_universe_by_liquidity.py first.", file=sys.stderr)
        return 1

    drop_assets = sorted(load_drop_assets(csv_path))
    print(f"Screener CSV:   {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"DROP assets:    {len(drop_assets)}")
    if not drop_assets:
        print("Nothing to clean up.")
        return 0
    print(f"  {', '.join(drop_assets)}")
    print()

    plan: Dict[str, List[Path]] = {}
    sizes: Dict[str, int] = {}
    grand_total = 0
    for sym in drop_assets:
        paths = collect_paths_for_asset(sym)
        plan[sym] = paths
        sizes[sym] = sum(_path_size(p) for p in paths)
        grand_total += sizes[sym]

    print(f"{'asset':<14} {'paths':>5}  {'size':>10}")
    print("-" * 40)
    for sym in drop_assets:
        n = len(plan[sym])
        print(f"{sym:<14} {n:>5}  {_human_bytes(sizes[sym]):>10}")
    print("-" * 40)
    print(f"{'TOTAL':<14} {sum(len(p) for p in plan.values()):>5}  "
          f"{_human_bytes(grand_total):>10}")
    print()

    # Write a JSON record so we can document EXACTLY what was deleted
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_path = (Path(args.log_decision) if args.log_decision
                else LOG_DIR / f"cleanup_drop_assets_{today}.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts_utc":          datetime.now(timezone.utc).isoformat(),
        "csv":             str(csv_path.relative_to(PROJECT_ROOT)),
        "drop_assets":     drop_assets,
        "applied":         args.apply,
        "total_bytes":     grand_total,
        "per_asset":       {
            sym: {
                "n_paths": len(plan[sym]),
                "size_bytes": sizes[sym],
                "paths": [str(p.relative_to(PROJECT_ROOT)) for p in plan[sym]],
            }
            for sym in drop_assets
        },
    }
    import json
    log_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Decision log written: {log_path.relative_to(PROJECT_ROOT)}")
    print()

    if not args.apply:
        print("DRY-RUN -- no files deleted. Re-run with --apply to execute.")
        return 0

    print("EXECUTING DELETIONS...")
    n_deleted = 0
    n_failed = 0
    for sym in drop_assets:
        for p in plan[sym]:
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink(missing_ok=True)
                n_deleted += 1
            except Exception as e:
                print(f"  [FAIL] {p}: {type(e).__name__}: {e}")
                n_failed += 1
    print(f"\nDeleted {n_deleted} paths; {n_failed} failed.")
    print(f"Reclaimed: {_human_bytes(grand_total)}")
    return 0 if n_failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
