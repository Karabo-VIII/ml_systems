"""Delete raw + processed + snapshot data for non-liquid / broken assets.

Source of truth: src/strategy/universe.py UNIVERSE_EXCLUDE_LIQUIDITY_2026_04_23
plus the 14 pending-fetch drops from config update.

Prints what it will delete with --dry (default), deletes with --execute.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "strategy"))
from universe import UNIVERSE_EXCLUDE_LIQUIDITY_2026_04_23 as FETCHED_EXCLUDE

# Also drop these (pending-fetch, predicted unusable; might or might not be present)
PENDING_EXCLUDE = {"D", "XUSD", "XAUT"}

DROP_ALL = FETCHED_EXCLUDE | PENDING_EXCLUDE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    args = ap.parse_args()

    print(f"Drop set: {len(DROP_ALL)} assets")
    print(f"  Fetched exclude: {sorted(FETCHED_EXCLUDE)}")
    print(f"  Pending exclude: {sorted(PENDING_EXCLUDE)}")

    targets = []
    for base in sorted(DROP_ALL):
        # Raw data dir (e.g. data/raw/BARDUSDT/)
        raw_dir = ROOT / "data" / "raw" / f"{base}USDT"
        if raw_dir.exists():
            size_mb = sum(f.stat().st_size for f in raw_dir.rglob("*") if f.is_file()) / 1e6
            targets.append(("raw_dir", raw_dir, size_mb))

        # Processed chimera file
        chimera = ROOT / "data" / "processed" / f"{base.lower()}usdt_v50_chimera.parquet"
        if chimera.exists():
            size_mb = chimera.stat().st_size / 1e6
            targets.append(("chimera", chimera, size_mb))

        # DIB bars (only for BTC/ETH/SOL/BNB/XRP/DOGE/ADA/AVAX/LINK/LTC so probably none)
        dib = ROOT / "data" / "frontier" / "dib" / f"{base}USDT_dib_2025.parquet"
        if dib.exists():
            size_mb = dib.stat().st_size / 1e6
            targets.append(("dib", dib, size_mb))

        # Seed snapshot (typically none for these excluded assets)
        seed = ROOT / "logs" / "paper_trader_v2" / "seeds" / f"pt_{base.lower()}"
        if seed.exists():
            size_mb = sum(f.stat().st_size for f in seed.rglob("*") if f.is_file()) / 1e6
            targets.append(("seed_dir", seed, size_mb))

    if not targets:
        print("\nNothing to delete.")
        return

    total_mb = sum(s for _, _, s in targets)
    print(f"\n{len(targets)} targets, {total_mb:.1f} MB total:")
    print("-" * 70)
    for kind, path, size_mb in targets:
        print(f"  {kind:<10} {size_mb:>7.1f} MB  {path.relative_to(ROOT)}")

    if not args.execute:
        print("\n(dry-run; nothing deleted; re-run with --execute)")
        return

    print("\nDeleting...")
    for kind, path, _ in targets:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"  [ok] {path.relative_to(ROOT)}")
        except Exception as e:
            print(f"  [ERR] {path.relative_to(ROOT)}: {e}")
    print(f"\nDone. Freed ~{total_mb:.1f} MB.")


if __name__ == "__main__":
    main()
