"""Update config/data_config_top100.yaml to exclude non-liquid / broken assets.

Removes assets from the fetch list that either:
    - Have broken chimera data (NOM, ONT, STO)
    - Were too new at fetch time (<180 days) and dropped off (KITE de-listed)
    - Have <$3M/day 30d volume (SPK, SAPIEN, GUN, GTC, QI, API3, W, TWT, KERNEL, KAT, etc)
    - Pending-fetch, predicted unusable (D=single letter, XUSD=stablecoin, XAUT=gold)

Preserves all deployable assets + pending-fetch high-liquidity candidates.

Usage:
    python scripts/update_data_config.py          # update in place
    python scripts/update_data_config.py --dry    # show diff only
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "data_config_top100.yaml"
BACKUP = ROOT / "config" / "data_config_top100.backup_2026_04_23.yaml"

# Drop set (base asset names, not pairs)
DROP_ASSETS = {
    # Fetched and confirmed non-liquid / broken (from U100 liquidity audit)
    "API3", "BARD", "GTC", "GUN", "G", "KAT", "KERNEL", "KITE", "NIGHT",
    "NOM", "ONT", "QI", "SAPIEN", "SKY", "SPK", "STO", "TWT", "U", "W",
    # Pending-fetch, predicted unusable
    "D",       # single-letter ticker, high risk
    "XUSD",    # stablecoin (no crypto alpha ranker value)
    "XAUT",    # gold-pegged (not crypto alpha)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="Show diff only, don't write")
    args = ap.parse_args()

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    original_assets = dict(cfg["assets"])
    n_before = len(original_assets)

    kept = {}
    dropped = []
    for pair, params in original_assets.items():
        base = pair.split("/")[0].upper()
        if base in DROP_ASSETS:
            dropped.append(pair)
        else:
            kept[pair] = params

    print(f"Assets before: {n_before}")
    print(f"Assets to DROP ({len(dropped)}): {sorted(dropped)}")
    print(f"Assets to KEEP: {len(kept)}")

    if args.dry:
        print("\n(dry-run; config unchanged)")
        return

    # Backup original
    shutil.copyfile(CONFIG, BACKUP)
    print(f"\n[backup] {CONFIG} -> {BACKUP}")

    cfg["assets"] = kept
    with open(CONFIG, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"[write]  {CONFIG} updated: {n_before} -> {len(kept)} assets")


if __name__ == "__main__":
    main()
