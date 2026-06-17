"""
V11-V14 SOTA-upgraded retraining launcher.

Runs the 4 fixed world models (V11 VIB+strong-disc, V12 strong-VIB, V13
hard-top-k VSN, V14 strong-VIB) sequentially or specified subset.

Because all four share the 4060 8GB GPU, serial is the only safe option.
Estimated wall-clock:
  V11: 3-5 hours (50 epochs)
  V12: 3-5 hours
  V13: 3-5 hours
  V14: 4-6 hours (diffusion)
Total sequential: ~15-20 hours.

Usage:
    python scripts/retrain_v11_v14_sota.py                     # all 4
    python scripts/retrain_v11_v14_sota.py --only 11,13        # subset
    python scripts/retrain_v11_v14_sota.py --clean-stale       # wipe old ckpts first
    python scripts/retrain_v11_v14_sota.py --dry-run           # show commands

Output:
    logs/v11/v11_train_<timestamp>.log (per version)
    models/v11/base/v11_f34_wm_latest.pt (rolling)

Monitor progress via:
    python scripts/monitor_v10_v14_training.py --watch
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

VERSIONS = {
    "11": {
        "train_script": "src/wm/v11/v11_training/train_world_model.py",
        "name": "V11 WaveNet+MoE+Disc (+VIB +DISC=0.3)",
    },
    "12": {
        "train_script": "src/wm/v12/v12_training/train_world_model.py",
        "name": "V12 Cross-Attn (strong VIB 0.15)",
    },
    "13": {
        "train_script": "src/wm/v13/v13_training/train_world_model.py",
        "name": "V13 TFT (hard top-k VSN=8)",
    },
    "14": {
        "train_script": "src/wm/v14/v14_training/train_world_model.py",
        "name": "V14 Diffusion (strong VIB 0.10)",
    },
}


def clean_stale_checkpoints(version: str):
    base = MODELS_DIR / f"v{version}" / "base"
    if not base.exists():
        return
    for p in base.glob("*.pt"):
        print(f"  removing stale {p.name}")
        p.unlink()


def run_version(version: str, features: int = 34, dry_run: bool = False) -> int:
    cfg = VERSIONS[version]
    log_dir = LOGS_DIR / f"v{version}" / f"v{version}"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"v{version}_sota_retrain_{ts}.log"

    script = PROJECT_ROOT / cfg["train_script"]
    cmd = [sys.executable, str(script), "--features", str(features)]

    print(f"\n{'=' * 80}")
    print(f"  {cfg['name']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log:     {log_file}")
    print(f"{'=' * 80}")

    if dry_run:
        return 0

    t0 = time.time()
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                              cwd=str(PROJECT_ROOT),
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dt = time.time() - t0
    print(f"  V{version} finished in {dt/60:.1f}min, exit={proc.returncode}")
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="11,12,13,14",
                    help="Comma-separated version numbers")
    ap.add_argument("--features", type=int, default=34)
    ap.add_argument("--clean-stale", action="store_true",
                    help="Wipe old checkpoints before training")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    targets = [v.strip() for v in args.only.split(",") if v.strip() in VERSIONS]
    if not targets:
        print("No valid versions specified")
        sys.exit(1)

    print(f"SOTA retrain plan: {targets}")
    if args.clean_stale:
        print("\nCleaning stale checkpoints...")
        for v in targets:
            clean_stale_checkpoints(v)

    results = {}
    for v in targets:
        try:
            rc = run_version(v, args.features, args.dry_run)
            results[v] = rc
        except KeyboardInterrupt:
            print(f"\nInterrupted during V{v}")
            results[v] = -1
            break
        except Exception as e:
            print(f"V{v} exception: {e}")
            results[v] = -2

    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    for v, rc in results.items():
        status = "OK" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  V{v}: {status}")


if __name__ == "__main__":
    main()
