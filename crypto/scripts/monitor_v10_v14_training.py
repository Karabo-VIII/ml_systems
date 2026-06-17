"""
Monitor V10-V14 training progress.

Reads latest checkpoints from models/v{10-14}/base/ and reports:
- Current epoch
- Train/val loss
- IC, ShIC at last checkpoint
- ShIC decline count (gate status)
- VIB KL and other SOTA-upgrade specific metrics

Usage:
    python scripts/monitor_v10_v14_training.py              # single snapshot
    python scripts/monitor_v10_v14_training.py --watch      # refresh every 30s
    python scripts/monitor_v10_v14_training.py --version 11 # only V11
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = PROJECT_ROOT / "models"

VERSIONS = ["v11", "v12", "v13", "v14"]


def read_checkpoint_meta(ckpt_path: Path) -> Optional[Dict]:
    if not ckpt_path.exists():
        return None
    try:
        c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return {"error": str(e)[:80]}
    if not isinstance(c, dict):
        return {"error": "not a dict"}
    return {
        "epoch": c.get("epoch"),
        "val_loss": c.get("val_loss"),
        "train_loss": c.get("train_loss"),
        "best_shic": c.get("best_shic"),
        "shic_decline_count": c.get("shic_decline_count"),
        "patience": c.get("patience"),
        "ic": c.get("ic"),
        "shic": c.get("shic"),
        "vib_kl": c.get("vib_kl"),
        "n_features": c.get("n_features"),
    }


def snapshot(version: str) -> Dict:
    base = MODEL_ROOT / version / "base"
    out = {"version": version, "base_dir": str(base), "exists": base.exists()}
    if not base.exists():
        return out

    latest = base / f"{version}_f34_wm_latest.pt"
    best_ema = base / f"{version}_f34_wm_best_ema.pt"
    out["latest"] = read_checkpoint_meta(latest)
    out["best_ema"] = read_checkpoint_meta(best_ema)

    # Latest epoch checkpoint
    epoch_ckpts = sorted(base.glob(f"{version}_f34_wm_epoch_*.pt"),
                          key=lambda p: int(p.stem.split("_")[-1]))
    if epoch_ckpts:
        out["n_epoch_ckpts"] = len(epoch_ckpts)
        out["latest_epoch_file"] = epoch_ckpts[-1].name
        out["ckpt_mtime"] = epoch_ckpts[-1].stat().st_mtime

    return out


def format_snapshot(snap: Dict) -> str:
    v = snap["version"]
    if not snap.get("exists"):
        return f"{v}: NO MODEL DIR"
    latest = snap.get("latest") or {}
    if not latest:
        return f"{v}: no checkpoint yet"
    if "error" in latest:
        return f"{v}: ckpt error: {latest['error']}"

    ep = latest.get("epoch", "?")
    vl = latest.get("val_loss")
    vl_s = f"{vl:.4f}" if vl is not None else "n/a"
    bs = latest.get("best_shic")
    bs_s = f"{bs:.4f}" if bs is not None else "n/a"
    sdc = latest.get("shic_decline_count", 0)
    gate = "FAIL" if bs is not None and bs == 0.0 and ep and ep > 10 else "OK"
    if ep and ep < 10:
        gate = "warm"
    n_ckpts = snap.get("n_epoch_ckpts", 0)
    return (f"{v}: ep={ep:>3} val_loss={vl_s:>7} best_shic={bs_s:>7} "
            f"decline={sdc} gate={gate:<4} ckpts={n_ckpts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="Refresh every 30s")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--version", type=int, default=None,
                    help="Only report specific version (11/12/13/14)")
    args = ap.parse_args()

    targets = VERSIONS
    if args.version is not None:
        targets = [f"v{args.version}"]

    while True:
        print("=" * 88)
        print(f"  V10-V14 TRAINING MONITOR — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 88)
        for v in targets:
            snap = snapshot(v)
            print(format_snapshot(snap))

        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
