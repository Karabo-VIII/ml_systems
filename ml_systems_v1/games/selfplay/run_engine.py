"""
chess_zero.run_engine -- ONE-COMMAND launcher for the chess AlphaZero engine.

Wraps `play.py learn-watch` so a user can start training + watching the agent
improve without memorising a dozen flags. Pick a recipe, run this file.

Usage:
    python projects/chess_zero/run_engine.py                    # standard recipe
    python projects/chess_zero/run_engine.py --recipe quick     # 15-min smoke
    python projects/chess_zero/run_engine.py --recipe overnight # 8-hour run
    python projects/chess_zero/run_engine.py --list             # show recipes
    python projects/chess_zero/run_engine.py --doctor           # preflight check
    python projects/chess_zero/run_engine.py --dry-run          # print command only

Must be run from the repo root with PYTHONPATH set to the repo root, or via the
venv python at .venv/Scripts/python.exe (Windows).
"""
from __future__ import annotations

__contract__ = {
    "kind": "entrypoint",
    "inputs": [
        "--recipe {quick,standard,overnight,ceiling}",
        "--list",
        "--doctor",
        "--dry-run",
        "--ckpt-dir override (optional)",
    ],
    "outputs": [
        "subprocess: python projects/chess_zero/play.py learn-watch ...",
        "live.html path printed on completion",
    ],
    "invariants": [
        "always runs --doctor before any real launch; exits 2 on hard-fail",
        "never launches a real training run under --list / --doctor / --dry-run",
        "no emoji characters in any print (Windows cp1252 safety)",
        "lock file at <ckpt-dir>/.train.lock is auto-cleared when owner pid is dead",
    ],
}

import argparse
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Paths (all absolute so this script can be called from any cwd)
# ---------------------------------------------------------------------------
_THIS_FILE   = os.path.abspath(__file__)
_PKG_DIR     = os.path.dirname(_THIS_FILE)                  # selfplay/  (holds play.py)
_REPO_ROOT   = os.path.dirname(_PKG_DIR)                    # games-engine root (holds az/)
_AZ_DIR      = os.path.join(_REPO_ROOT, "az")
_VENV_PYTHON = os.path.join(_REPO_ROOT, ".venv", "Scripts", "python.exe")
_PLAY_PY     = os.path.join(_PKG_DIR, "play.py")
_BOOTSTRAP_DIR = os.path.join(_AZ_DIR, "bootstrap_checkpoints")

# ---------------------------------------------------------------------------
# Recipe definitions
# ---------------------------------------------------------------------------
# Each recipe maps to exactly the flags passed to `play.py learn-watch`.
# Boolean flags (store_true in play.py) use value True; omit = not passed.
_BASE_STANDARD = {
    "--max-hours":        2.0,
    "--games-per-iter":   64,
    "--train-steps":      500,
    "--selfplay-sims":    64,
    "--eval-games":       8,
    "--train-opponent":   "mix",
    "--anchor-kl":        1.0,
    "--curriculum":       True,   # boolean flag
    "--auto-balance":     True,   # boolean flag
    "--selfplay-workers": 16,
    "--watch":            "classical",
    "--viz":              True,   # boolean flag
    "--temperature":      1.0,
    # OPENING DIVERSITY (2026-06-09): each self-play game starts from a distinct sound
    # opening so the net stops reinforcing one rote line / bad learned habits from a
    # fixed start. Eval stays on startpos so the strength curve is unaffected.
    "--opening-mode":     "mixed",
    "--opening-plies":    4,
}

RECIPES = {
    "quick": {
        **_BASE_STANDARD,
        "--max-hours":       0.25,
        "--games-per-iter":  24,
        "--train-steps":     100,
        "--selfplay-sims":   48,
        "--eval-games":      4,
        "_ckpt_dir":         "robust_dual",
    },
    "standard": {
        **_BASE_STANDARD,
        "_ckpt_dir":         "robust_dual",
    },
    "overnight": {
        **_BASE_STANDARD,
        "--max-hours":       8.0,
        "_ckpt_dir":         "robust_dual",
    },
    "ceiling": {
        # filled in at runtime (may fall back to standard's ckpt-dir)
        "_uses_bootstrap_d4": True,
        **_BASE_STANDARD,
        "_ckpt_dir":         "__ceiling_resolved_at_runtime__",
    },
}

# Human-readable summary for --list
_RECIPE_SUMMARY = {
    "quick":     "15-min smoke run   | max-hours 0.25 | games/iter 24  | steps 100",
    "standard":  "2-hour solid run   | max-hours 2.0  | games/iter 64  | steps 500",
    "overnight": "8-hour deep run    | max-hours 8.0  | games/iter 64  | steps 500",
    "ceiling":   "strongest seed run | max-hours 2.0  | seeds from bootstrap_d4 if available",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ceiling_ckpt_dir() -> tuple[str, bool]:
    """Return (ckpt_dir_name, is_fallback).

    Ceiling recipe tries to use az/bootstrap_d4 as the seed seed into az/robust_d4.
    If bootstrap_d4 does not exist or has no .pt checkpoint, falls back to robust_dual
    and returns is_fallback=True.
    """
    bootstrap_d4 = os.path.join(_AZ_DIR, "bootstrap_d4")
    has_d4 = False
    if os.path.isdir(bootstrap_d4):
        import glob
        has_d4 = bool(glob.glob(os.path.join(bootstrap_d4, "*.pt")))
    if has_d4:
        return "robust_d4", False
    return "robust_dual", True


def _seed_ceiling_from_d4() -> bool:
    """Seed az/robust_d4/ with bootstrap_d4's net as net_iter0 + latest pointer, so the ceiling
    recipe RESUMES the STRONGER depth-4 imitation base (bootstrap_d4/net_bootstrap.pt) instead of
    play.py's default depth-1 bootstrap. Idempotent + non-clobbering: skips if robust_d4 already
    has a net (an in-progress run wins). Returns True iff it seeded. Mirrors play.py's
    _seed_refine_from_bootstrap, sourced from bootstrap_d4."""
    import glob
    src = os.path.join(_AZ_DIR, "bootstrap_d4", "net_bootstrap.pt")
    if not os.path.exists(src):
        return False
    dst_dir = os.path.join(_AZ_DIR, "robust_d4")
    os.makedirs(dst_dir, exist_ok=True)
    if glob.glob(os.path.join(dst_dir, "net_iter*.pt")):
        return False  # already has progress -- never clobber a live run
    import torch  # lazy: --list/--doctor/--dry-run never pay the import
    ck = dict(torch.load(src, map_location="cpu", weights_only=False))
    ck["iter"] = 0  # so train_robust resumes at iter 1 from the depth-4 bootstrap weights
    dst = os.path.join(dst_dir, "net_iter0.pt")
    tmp = dst + ".tmp"
    torch.save(ck, tmp)
    os.replace(tmp, dst)
    ptr = os.path.join(dst_dir, "latest.json.tmp.pt")
    torch.save({"iter": 0, "path": "net_iter0.pt"}, ptr)
    os.replace(ptr, os.path.join(dst_dir, "latest.pt"))
    return True


def _pid_alive(pid: int) -> bool:
    """Return True if process pid is currently running (Windows + POSIX)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        SYNCHRONIZE = 0x00100000
        handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        result = ctypes.windll.kernel32.WaitForSingleObject(handle, 0)        # type: ignore[attr-defined]
        ctypes.windll.kernel32.CloseHandle(handle)                             # type: ignore[attr-defined]
        return result != 0  # WAIT_OBJECT_0 (0) means exited; non-zero = still alive
    else:
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def _check_lock(ckpt_dir_name: str) -> tuple[bool, str]:
    """Check the instance lock for a ckpt dir.

    Returns (ok: bool, message: str).
    ok=True  -> no problem (either no lock, or stale lock was cleared).
    ok=False -> a live trainer is already running; cannot proceed.
    """
    ckpt_dir = os.path.join(_AZ_DIR, ckpt_dir_name)
    lock_path = os.path.join(ckpt_dir, ".train.lock")
    if not os.path.exists(lock_path):
        return True, ""
    try:
        with open(lock_path, "r", encoding="ascii", errors="replace") as f:
            raw = f.read().strip()
        pid = int(raw) if raw.isdigit() else -1
    except Exception:
        pid = -1

    if _pid_alive(pid):
        return False, f"a trainer (pid {pid}) is already running on {ckpt_dir_name}"
    # stale lock: owner dead -- auto-clear
    try:
        os.remove(lock_path)
        return True, f"cleared stale lock (dead pid {pid})"
    except OSError as e:
        return False, f"could not clear stale lock: {e}"


def _has_seed_checkpoint(ckpt_dir_name: str) -> bool:
    """Return True if either the bootstrap_checkpoints dir has a .pt OR the named
    ckpt dir already has a net_iter*.pt / latest.pt (so training can resume/start).
    """
    import glob
    # Bootstrap seed (the canonical first-run seed)
    if glob.glob(os.path.join(_BOOTSTRAP_DIR, "*.pt")):
        return True
    # The target ckpt dir already has progress
    ckpt_dir = os.path.join(_AZ_DIR, ckpt_dir_name)
    if os.path.isdir(ckpt_dir):
        if glob.glob(os.path.join(ckpt_dir, "net_iter*.pt")):
            return True
        if os.path.exists(os.path.join(ckpt_dir, "latest.pt")):
            return True
    return False


def _free_gb(path: str) -> float:
    """Return free disk space in GB for the volume containing `path`."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return float("inf")


# ---------------------------------------------------------------------------
# Doctor (preflight)
# ---------------------------------------------------------------------------

def run_doctor(ckpt_dir_name: str) -> int:
    """Run all preflight checks. Prints [OK]/[WARN]/[FAIL] lines.
    Returns 0 if all hard checks pass, 2 if any hard-fail.
    """
    failures = 0

    # 1. venv python
    if os.path.isfile(_VENV_PYTHON):
        print(f"[OK]   venv python: {_VENV_PYTHON}")
    else:
        print(f"[FAIL] venv python not found at {_VENV_PYTHON}")
        failures += 1

    # 2. torch + cuda
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "PYTHONPATH": _REPO_ROOT},
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        cuda_out = result.stdout.strip()
        if cuda_out == "cuda":
            print("[OK]   torch available; CUDA detected")
        elif cuda_out == "cpu":
            print("[WARN] torch available but CUDA not detected -- will run on CPU (slow)")
        else:
            print(f"[FAIL] torch check failed: {result.stderr.strip()[:120]}")
            failures += 1
    except Exception as e:
        print(f"[FAIL] torch import check error: {e}")
        failures += 1

    # 3. python-chess
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import chess; print(chess.__version__)"],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONPATH": _REPO_ROOT},
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        if result.returncode == 0:
            print(f"[OK]   python-chess {result.stdout.strip()}")
        else:
            print(f"[FAIL] python-chess not installed: {result.stderr.strip()[:120]}")
            failures += 1
    except Exception as e:
        print(f"[FAIL] python-chess check error: {e}")
        failures += 1

    # 4. bootstrap seed checkpoint
    if _has_seed_checkpoint(ckpt_dir_name):
        print(f"[OK]   bootstrap seed checkpoint found (bootstrap_checkpoints/ or {ckpt_dir_name}/)")
    else:
        print(
            f"[FAIL] no seed checkpoint found in bootstrap_checkpoints/ or {ckpt_dir_name}/.\n"
            f"       Run first:  python -m az.bootstrap_supervised"
        )
        failures += 1

    # 5. instance lock check
    ok, msg = _check_lock(ckpt_dir_name)
    if ok:
        if msg:
            print(f"[OK]   {msg}")
        else:
            print(f"[OK]   no lock on {ckpt_dir_name} (clear to run)")
    else:
        print(f"[FAIL] {msg}")
        failures += 1

    # 6. disk space
    free_gb = _free_gb(_AZ_DIR)
    if free_gb >= 5.0:
        print(f"[OK]   free disk: {free_gb:.1f} GB")
    elif free_gb >= 2.0:
        print(f"[WARN] free disk low: {free_gb:.1f} GB (recommend >= 5 GB)")
    else:
        print(f"[FAIL] free disk critically low: {free_gb:.1f} GB (need >= 2 GB)")
        failures += 1

    if failures == 0:
        print("\n[OK] all preflight checks passed -- ready to launch")
    else:
        print(f"\n[FAIL] {failures} check(s) failed -- fix the above before running")

    return 0 if failures == 0 else 2


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_command(recipe_name: str, ckpt_dir_name: str) -> list[str]:
    """Assemble the full subprocess command list for the given recipe."""
    recipe = dict(RECIPES[recipe_name])
    # pop internal keys
    recipe.pop("_ckpt_dir", None)
    recipe.pop("_uses_bootstrap_d4", None)

    python = _VENV_PYTHON if os.path.isfile(_VENV_PYTHON) else sys.executable
    cmd = [python, _PLAY_PY, "learn-watch", "--ckpt-dir", ckpt_dir_name]

    for flag, value in recipe.items():
        if isinstance(value, bool):
            if value:
                cmd.append(flag)            # e.g. --curriculum
            # False booleans are simply not added
        else:
            cmd.extend([flag, str(value)])

    return cmd


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def print_recipe_list() -> None:
    """Print the recipes table."""
    print("Available recipes (pass with --recipe NAME):\n")
    width = max(len(n) for n in RECIPES)
    for name, summary in _RECIPE_SUMMARY.items():
        default = "  [DEFAULT]" if name == "standard" else ""
        print(f"  {name:<{width}}  {summary}{default}")
    print()
    print("All recipes use --watch classical --viz --curriculum --auto-balance --anchor-kl 1.0 "
          "--opening-mode mixed")
    print("  --ckpt-dir default: robust_dual (quick/standard/overnight)")
    print("  ceiling:            robust_d4 (seeded from bootstrap_d4 if present, else robust_dual)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="run_engine.py",
        description=(
            "One-command launcher for the chess AlphaZero engine. "
            "Wraps play.py learn-watch per a named recipe."
        ),
    )
    ap.add_argument(
        "--recipe",
        choices=list(RECIPES.keys()),
        default="standard",
        help="Training recipe (default: standard)",
    )
    ap.add_argument(
        "--ckpt-dir",
        default=None,
        help="Override the recipe's default ckpt-dir (subdir under projects/chess_zero/az/)",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="Print available recipes and exit",
    )
    ap.add_argument(
        "--doctor",
        action="store_true",
        help="Run preflight checks and exit (0=all pass, 2=hard-fail)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the assembled play.py command and exit without launching",
    )

    args = ap.parse_args()

    # --list
    if args.list:
        print_recipe_list()
        sys.exit(0)

    # Resolve ckpt-dir
    if args.ckpt_dir:
        ckpt_dir_name = args.ckpt_dir
        ceiling_fallback = False
    elif args.recipe == "ceiling":
        ckpt_dir_name, ceiling_fallback = _resolve_ceiling_ckpt_dir()
        if ceiling_fallback:
            print(
                "[ceiling] bootstrap_d4 not ready (no .pt in az/bootstrap_d4/). "
                "Falling back to robust_dual.\n"
                "To unlock ceiling mode: run bootstrap training into az/bootstrap_d4/ first."
            )
        else:
            print(f"[ceiling] will seed {ckpt_dir_name} from bootstrap_d4 (stronger depth-4 base) at launch")
    else:
        ckpt_dir_name = RECIPES[args.recipe]["_ckpt_dir"]

    # --doctor
    if args.doctor:
        rc = run_doctor(ckpt_dir_name)
        sys.exit(rc)

    # Build the command
    cmd = build_command(args.recipe, ckpt_dir_name)

    # --dry-run
    if args.dry_run:
        print("Assembled command:")
        print("  " + " ".join(cmd))
        sys.exit(0)

    # Default action: run doctor first, then launch
    print(f"[run_engine] recipe={args.recipe}  ckpt-dir={ckpt_dir_name}")
    print("[run_engine] running preflight checks...\n")
    rc = run_doctor(ckpt_dir_name)
    if rc != 0:
        print("\n[run_engine] preflight FAILED -- not launching. Fix the issues above.")
        sys.exit(2)

    # CEILING: seed robust_d4 from the stronger depth-4 bootstrap just before launch (real launch
    # only -- never under --list/--doctor/--dry-run, which exit above). Non-clobbering.
    if args.recipe == "ceiling" and not args.ckpt_dir and ckpt_dir_name == "robust_d4":
        if _seed_ceiling_from_d4():
            print("[ceiling] seeded robust_d4/net_iter0 from bootstrap_d4/net_bootstrap.pt (depth-4 base)")
        else:
            print("[ceiling] robust_d4 already has progress -- resuming it (no re-seed)")

    print(f"\n[run_engine] launching: {' '.join(cmd)}\n")
    live_html = os.path.join(_AZ_DIR, ckpt_dir_name, "live.html")

    try:
        subprocess.run(
            cmd,
            cwd=_REPO_ROOT,
            env={**os.environ, "PYTHONPATH": _REPO_ROOT},
        )
    except KeyboardInterrupt:
        print("\n[run_engine] interrupted by user.")
    finally:
        print(f"\n[run_engine] done. Live board was at: {live_html}")


if __name__ == "__main__":
    main()
