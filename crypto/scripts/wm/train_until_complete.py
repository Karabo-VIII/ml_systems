"""scripts/wm/train_until_complete.py -- crash-resilient WM training driver.

WHY THIS EXISTS (2026-06-12): a V1.1 `vsn_fr` run crashed at epoch 31 step 0 with a NATIVE C++ abort
(`c10/util/AbortHandler.h` -> std::terminate), AFTER saving a strong epoch-30 checkpoint (ShIC 0.0278,
IC1 0.0568, GATE PASS). A native terminate CANNOT be caught by a Python try/except -- it kills the process.
But the trainer checkpoints EVERY epoch and AUTO-RESUMES from `<tag>_wm_latest.pt` (train_world_model.py
loads it unconditionally and sets start_epoch = ckpt['epoch']; `--force` only bypasses run_all_training's
skip-if-complete check, it does NOT reset). So the correct fix for a transient native abort is an EXTERNAL
auto-resume loop: re-launch the SAME command (which resumes from latest.pt) until it exits 0.

It also sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces 8GB-card fragmentation-OOM, a common
cause of transient native aborts) unless the caller already set it.

USAGE (PowerShell) -- resume the crashed vsn_fr run from epoch 30 with auto-retry:
    python scripts/wm/train_until_complete.py --model v1_1 --features 41 --run-tag vsn_fr --vsn --forward-regime
The baseline (no levers):
    python scripts/wm/train_until_complete.py --model v1_1 --features 41 --run-tag baseline
Dry-run (print the exact command + env, launch nothing):
    python scripts/wm/train_until_complete.py --model v1_1 --features 41 --run-tag vsn_fr --vsn --forward-regime --dry-run

rc==0 from run_all_training means "trained to completion OR nothing left to do" (with --force it always
runs the trainer, which exits 0 once start_epoch >= WM_TOTAL_EPOCHS). Any non-zero exit = a crash -> resume.

__contract__ = {
    "kind": "wm_train_driver",
    "inputs": ["--model", "--features", "--run-tag", "--vsn", "--forward-regime", "--max-retries"],
    "outputs": ["resumed training to completion via run_all_training (--force resumes from latest.pt)"],
    "invariants": ["never passes a reset flag; --force only bypasses skip-if-complete; trainer resumes from latest.pt",
                   "retries ONLY on non-zero exit (a crash); rc==0 = done",
                   "env levers passed via os.environ (V1_VSN / V1_FORWARD_REGIME), not bash inline-prefix (PowerShell-safe)"],
}
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Per-model env levers (the world-class flags). Extend here as other versions gain levers.
VSN_ENV = {"v1_1": "V1_VSN", "v3": "V3_VSN", "v4": "V4_VSN", "v6": "V6_VSN", "v8": "V8_VSN", "v13": "V13_VSN"}
FR_ENV = {"v1_1": "V1_FORWARD_REGIME", "v3": "V3_FORWARD_REGIME", "v4": "V4_FORWARD_REGIME",
          "v6": "V6_FORWARD_REGIME", "v8": "V8_FORWARD_REGIME", "v13": "V13_FORWARD_REGIME"}


def free_vram_mib():
    """Free GPU VRAM in MiB via nvidia-smi, or None if unavailable. The 8GB card is shared with the games
    AlphaZero training + Ollama -- launching WM training without headroom is what causes the native abort."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_vram(min_free_mib: int, max_wait_sec: int, poll_sec: int) -> bool:
    """Block until free VRAM >= min_free_mib (or timeout). Returns True if headroom is available.
    On a shared 8GB box this lets the driver start the WM run the moment the GPU frees up -- no babysitting."""
    free = free_vram_mib()
    if free is None:
        print("[VRAM] nvidia-smi unavailable -- skipping the VRAM gate (launching blind).", flush=True)
        return True
    if free >= min_free_mib:
        print(f"[VRAM] {free} MiB free >= {min_free_mib} MiB required -- headroom OK, launching.", flush=True)
        return True
    waited = 0
    print(f"[VRAM] only {free} MiB free (< {min_free_mib} required) -- GPU is contended (games AZ / Ollama). "
          f"Waiting up to {max_wait_sec}s for headroom...", flush=True)
    while waited < max_wait_sec:
        time.sleep(poll_sec)
        waited += poll_sec
        free = free_vram_mib()
        if free is not None and free >= min_free_mib:
            print(f"[VRAM] {free} MiB free now -- headroom available after {waited}s, launching.", flush=True)
            return True
    print(f"[VRAM] still only {free} MiB free after {max_wait_sec}s -- NOT launching into an oversubscribed GPU "
          f"(would OOM -> native abort). Free VRAM first (pause the games training / stop Ollama) then re-run.",
          flush=True)
    return False


def build_command(model: str, features: int, run_tag: str) -> list:
    # --force = bypass run_all_training's skip-if-complete; the TRAINER still resumes from latest.pt.
    # NO reset/fresh flag is ever passed -> a re-run continues from the last saved epoch.
    cmd = [sys.executable, str(ROOT / "src" / "run_all_training.py"),
           "--features", str(features), "--model", model, "--force", "--only-base"]
    if run_tag:
        cmd += ["--run-tag", run_tag]
    return cmd


def build_env(model: str, vsn: bool, forward_regime: bool) -> dict:
    env = dict(os.environ)
    if vsn:
        key = VSN_ENV.get(model)
        if not key:
            print(f"[WARN] no VSN env var registered for model {model}; --vsn ignored", flush=True)
        else:
            env[key] = "1"
    if forward_regime:
        key = FR_ENV.get(model)
        if not key:
            print(f"[WARN] no FORWARD_REGIME env var registered for model {model}; --forward-regime ignored", flush=True)
        else:
            env[key] = "1"
    # anti-fragmentation on the 8GB card -- a common transient-native-abort cause. Don't override a caller's setting.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(prog="train_until_complete")
    ap.add_argument("--model", required=True, help="e.g. v1_1, v3, v4, v6, v8, v13")
    ap.add_argument("--features", type=int, required=True)
    ap.add_argument("--run-tag", default=None)
    ap.add_argument("--vsn", action="store_true", help="set the model's V*_VSN=1 lever")
    ap.add_argument("--forward-regime", action="store_true", help="set the model's V*_FORWARD_REGIME=1 lever")
    ap.add_argument("--max-retries", type=int, default=6, help="auto-resume attempts after a crash (default 6)")
    ap.add_argument("--backoff-sec", type=int, default=15, help="seconds to wait before a resume (lets the GPU settle)")
    ap.add_argument("--min-free-mib", type=int, default=3500,
                    help="required free VRAM before (re)launching -- V1.1 needs up to 3.3GB (default 3500)")
    ap.add_argument("--max-vram-wait", type=int, default=0,
                    help="max seconds to WAIT for VRAM headroom each attempt (0 = don't wait, refuse if contended)")
    ap.add_argument("--vram-poll", type=int, default=30, help="VRAM re-check interval while waiting (default 30s)")
    ap.add_argument("--dry-run", action="store_true", help="print the command + the levers, launch nothing")
    a = ap.parse_args(argv)

    cmd = build_command(a.model, a.features, a.run_tag)
    env = build_env(a.model, a.vsn, a.forward_regime)
    levers = {k: env.get(k) for k in (VSN_ENV.get(a.model), FR_ENV.get(a.model), "PYTORCH_CUDA_ALLOC_CONF") if k}

    print("=" * 78)
    print("WM TRAIN-UNTIL-COMPLETE (crash-resilient auto-resume)")
    print(f"  model={a.model} features={a.features} run-tag={a.run_tag}")
    print(f"  command : {' '.join(cmd)}")
    print(f"  levers  : {levers}")
    print(f"  retries : up to {a.max_retries} (resume from <tag>_wm_latest.pt on any non-zero exit)")
    print("  note    : --force bypasses skip-if-complete ONLY; the trainer resumes from latest.pt (no reset).")
    print("=" * 78, flush=True)

    if a.dry_run:
        print("[DRY-RUN] launching nothing.")
        return 0

    for attempt in range(1, a.max_retries + 1):
        if not wait_for_vram(a.min_free_mib, a.max_vram_wait, a.vram_poll):
            print("[ABORT] insufficient VRAM headroom -- not launching. Free the GPU (pause games AZ / stop Ollama) "
                  "or pass --max-vram-wait <sec> to wait for it.", flush=True)
            return 3
        print(f"\n[ATTEMPT {attempt}/{a.max_retries}] launching (resumes from latest.pt if present)...", flush=True)
        rc = subprocess.run(cmd, env=env, cwd=str(ROOT),
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).returncode
        if rc == 0:
            print(f"\n[DONE] run_all_training exited 0 on attempt {attempt} -- training complete (or nothing left).", flush=True)
            return 0
        print(f"\n[CRASH] exit code {rc} on attempt {attempt} (likely the transient native abort). "
              f"Resuming from latest.pt in {a.backoff_sec}s...", flush=True)
        time.sleep(a.backoff_sec)

    print(f"\n[EXHAUSTED] {a.max_retries} attempts all crashed -- this is NOT a transient abort; "
          f"investigate the trainer (check VRAM with nvidia-smi, reduce WM_BATCH_SIZE, inspect the last traceback).",
          flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())
