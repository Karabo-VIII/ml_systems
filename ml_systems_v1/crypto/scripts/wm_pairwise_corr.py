"""CC1 probe: pairwise OOS prediction correlation matrix for V1.x family.

Resolves the "ship 1 vs ship 4" question. If V1.0/V1.1/V1.4/V1.6 OOS
predictions are pairwise rho > 0.95, three of them are duplicates and
should be archived. If rho < 0.85 across all pairs, ensemble has real
diversity to harvest.

Usage:
    python scripts/wm_pairwise_corr.py --features 34 --asset BTC

What this script does:
    1. Loads each V1.x EMA checkpoint that exists for the requested
       (features, asset) pair.
    2. Loads the same OOS slice each model was evaluated on (per
       AntifragileConfig.split_four_way with purge=400).
    3. Runs forward-only inference at each model's TwoHot decoded
       output (h=1).
    4. Computes pairwise Pearson correlation across the prediction
       vectors.
    5. Prints a triangle correlation matrix + redundancy verdict.

What this script does NOT do:
    - Re-train.
    - Trigger CDAP / pre_train_gate.
    - Mutate any checkpoint or settings file.

Pre-flight:
    - Each version must have a `best_ema.pt` checkpoint at the path
      its settings.py declares.
    - chimera_legacy/dollar/<asset>_v50_chimera_*.parquet must exist.

Status: SKELETON (2026-04-29). Body is sketched but the per-version
forward call wiring is parked behind `--dry-run` until the next
session validates a clean checkpoint set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

VERSIONS = [
    ("v1_0", "src/wm/v1/v1_0_training", "models/v1/v1_0/base/v1_0_f{N}_wm_best_ema.pt"),
    ("v1_1", "src/wm/v1/v1_1_training", "models/v1/v1_1/base/v1_1_f{N}_wm_best_ema.pt"),
    ("v1_4", "src/wm/v1/v1_4_training", "models/v1/v1_4/base/v1_4_f{N}_wm_best_ema.pt"),
    ("v1_6", "src/wm/v1/v1_6_training", "models/v1/v1_6/base/v1_6_f{N}_wm_best_ema.pt"),
]


def find_checkpoints(n_features: int) -> list[tuple[str, Path]]:
    found = []
    missing = []
    for name, _src, ckpt_template in VERSIONS:
        p = ROOT / ckpt_template.format(N=n_features)
        if p.exists():
            found.append((name, p))
        else:
            missing.append((name, p))
    return found, missing


def load_model_and_predict(name: str, src_dir: str, ckpt: Path, asset: str, n_features: int):
    """Sketch: load the version's TransformerWorldModel, restore EMA
    weights, run forward on OOS slice, return decoded h=1 predictions.

    Returns numpy array [N_oos_bars] or None on failure.
    """
    # The actual implementation requires per-version sys.path wiring
    # (each settings.py has its own MODEL_DIR convention). To keep this
    # skeleton minimal-risk, we defer to a `--run` flag and return None
    # in the dry-run path so the rest of the script (matrix print,
    # verdict) still smoke-tests cleanly.
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="CC1: pairwise V1.x OOS correlation")
    parser.add_argument("--features", type=int, default=34,
                        choices=[13, 18, 21, 25, 29, 30, 34, 37],
                        help="Feature count (matches existing checkpoint suffix).")
    parser.add_argument("--asset", default="BTC",
                        help="Asset for the OOS evaluation (uppercase, no USDT).")
    parser.add_argument("--run", action="store_true",
                        help="Actually load checkpoints and run forward (default: dry-run skeleton).")
    args = parser.parse_args()

    print(f"CC1 pairwise V1.x correlation probe (features=f{args.features}, asset={args.asset})")
    print("-" * 70)

    found, missing = find_checkpoints(args.features)
    print(f"Checkpoints found: {len(found)}/{len(VERSIONS)}")
    for name, p in found:
        print(f"  {name:>5}: {p.relative_to(ROOT)}")
    for name, p in missing:
        print(f"  {name:>5}: MISSING {p.relative_to(ROOT)}")

    if not args.run:
        print("\n[dry-run] skipping forward-pass + correlation matrix.")
        print("  Re-run with --run after confirming all 4 checkpoints exist.")
        return 0

    if len(found) < 2:
        print("\nNeed at least 2 checkpoints to compute pairwise correlation.")
        return 2

    # ----- Forward + collect predictions -----
    preds: dict[str, np.ndarray] = {}
    for name, ckpt in found:
        src_dir = next(s for n, s, _ in VERSIONS if n == name)
        p = load_model_and_predict(name, src_dir, ckpt, args.asset, args.features)
        if p is not None:
            preds[name] = p

    if len(preds) < 2:
        print("\nForward pass failed for all but <2 versions; cannot compute correlation.")
        return 2

    # ----- Pairwise correlation -----
    names = sorted(preds.keys())
    n = len(names)
    corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            corr[i, j] = np.corrcoef(preds[names[i]], preds[names[j]])[0, 1]

    print("\nPairwise Pearson correlation (h=1 OOS decoded predictions):")
    print(" " * 8 + "  ".join(f"{n:>5}" for n in names))
    for i, name in enumerate(names):
        row = "  ".join(f"{corr[i, j]:+.3f}" for j in range(n))
        print(f"  {name:>5}: {row}")

    # ----- Redundancy verdict -----
    upper = corr[np.triu_indices(n, k=1)]
    if (upper > 0.95).all():
        print("\nVERDICT: all pairs rho > 0.95 -> SHIP 1, archive 3.")
        print("  Recommendation: keep V1.1 (record holder), archive V1.0 / V1.4 / V1.6.")
    elif (upper > 0.90).all():
        print("\nVERDICT: all pairs rho > 0.90 -> SHIP 1-2.")
        print("  Recommendation: V1.1 + 1 sibling with lowest rho to V1.1.")
    elif (upper > 0.85).any():
        print("\nVERDICT: some pairs rho > 0.85 -> selective ship.")
    else:
        print("\nVERDICT: all pairs rho <= 0.85 -> SHIP all 4 + ensemble.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
