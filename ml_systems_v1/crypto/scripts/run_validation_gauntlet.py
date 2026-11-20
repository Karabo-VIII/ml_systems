"""Serial 12-epoch validation gauntlet for the WM cohort (2026-05-10).

Runs each version with --max-epochs 12 --seed 42, captures stdout to per-version
log files. Designed for completion-throughput maximization on a single GPU:
each subprocess inherits the env (PYTORCH_CUDA_ALLOC_CONF set in trainers).

Versions covered (this session's wired changes only):
  V4  — Mamba+RSSM + forecast head + ATME=0.20  (regression-low risk; recon decoder anchors)
  V3  — WaveNet+GRU+RSSM + forecast head        (recon decoder anchors)
  V8  — NeuralODE+RSSM + forecast head          (recon decoder + bf16 CUDA-only)
  V13 — TFT + quantile pinball loss             (head shape changed 255->3; fresh ckpt)

Deferred to next session (committed in cohort 76a21c4 but never run at f29):
  V14 — Diffusion + score-head LayerNorm (needs --max-epochs added)
  V11 — WaveNet+MoE+Disc + warmup gate (needs --max-epochs added)
  V6  — JEPA first-time launch (needs --max-epochs added)

After completion, run:
  python scripts/parse_validation_gauntlet.py
to tier-classify each version's Ep 10 ShIC + IC into Filter/Sizer/Trader/Headline.

Usage:
  python scripts/run_validation_gauntlet.py [--max-epochs 12] [--seed 42] [--features 29]
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# (version, trainer relative path, log dir relative path) tuples
VERSIONS = [
    ("V4",  "src/wm/v4/v4_training/train_world_model.py",   "logs/v4/v4"),
    ("V3",  "src/wm/v3/v3_training/train_world_model.py",   "logs/v3"),
    ("V8",  "src/wm/v8/v8_training/train_world_model.py",   "logs/v8"),
    ("V13", "src/wm/v13/v13_training/train_world_model.py", "logs/v13"),
    ("V14", "src/wm/v14/v14_training/train_world_model.py", "logs/v14"),
    ("V11", "src/wm/v11/v11_training/train_world_model.py", "logs/v11"),
    ("V6",  "src/wm/v6/v6_training/train_world_model.py",   "logs/v6"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-epochs", type=int, default=12,
                    help="Epochs per version (12 hits the Ep 10 ShIC checkpoint)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--features", type=int, default=29)
    ap.add_argument("--versions", type=str, default="",
                    help="Comma-separated subset, e.g. 'V4,V13'. Empty = all.")
    args = ap.parse_args()

    selected = set([v.strip() for v in args.versions.split(",") if v.strip()])
    pairs = [v for v in VERSIONS if not selected or v[0] in selected]

    print("=" * 70)
    print(f"WM VALIDATION GAUNTLET — {len(pairs)} versions × {args.max_epochs} epochs")
    print(f"  seed={args.seed}  features={args.features}")
    print("=" * 70)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_log = PROJECT_ROOT / f"logs/validation_gauntlet_{timestamp}.summary"
    summary_log.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for ver, train_rel, log_dir in pairs:
        log_path = PROJECT_ROOT / log_dir / f"{ver.lower()}_validate_gauntlet_{timestamp}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / train_rel),
            "--features", str(args.features),
            "--max-epochs", str(args.max_epochs),
            "--seed", str(args.seed),
        ]
        print(f"\n[{ver}] running -> {log_path}")
        print(f"  cmd: {' '.join(cmd)}")
        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        with open(log_path, "w", encoding="utf-8") as logf:
            ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        dt = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL (exit {ret.returncode})"
        print(f"  [{ver}] {status} in {dt/60:.1f} min")
        results.append((ver, ret.returncode, dt, str(log_path)))

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION GAUNTLET SUMMARY")
    print("-" * 70)
    with open(summary_log, "w", encoding="utf-8") as f:
        for ver, rc, dt, log in results:
            line = f"  {ver:5s} : exit={rc:3d}  time={dt/60:6.1f}min  log={log}"
            print(line)
            f.write(line + "\n")
    print("=" * 70)
    print(f"Summary saved: {summary_log}")

    # Per-version Ep 10 metric extraction
    print("\nPer-version IC / ShIC at Ep 10 (read from logs):")
    for ver, rc, _, log_path in results:
        log_p = Path(log_path)
        if not log_p.exists():
            print(f"  {ver}: log missing")
            continue
        # Read all VAL/Ep lines and find Ep 10 + ShIC
        try:
            with open(log_p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Replace tqdm carriage-return overwrites with newlines for easier search
            content = content.replace("\r", "\n")
            ep10_lines = [ln for ln in content.split("\n")
                          if ("Ep  10" in ln or "Epoch  10" in ln or "[NEW BEST SHUFFLED IC]" in ln
                              or "ShIC" in ln or "[GATE" in ln or "GATE PASS" in ln or "GATE FAIL" in ln)]
            # Print last 5 relevant lines
            for ln in ep10_lines[-5:]:
                print(f"  {ver}: {ln.strip()[:140]}")
        except Exception as e:
            print(f"  {ver}: read error {e}")


if __name__ == "__main__":
    main()
