#!/usr/bin/env python
"""
Version Training Runner (V2-V14, V22-V25)
==========================================

Runs world model training for all variants of a given version serially
with preflight checks. Skips variants that fail preflight.

Usage:
    python src/run_version_training.py --version 2 --features 13
    python src/run_version_training.py --version 4 --features 13
    python src/run_version_training.py --version 9 --features 37
    python src/run_version_training.py --version 3 --features 13 --only base
    python src/run_version_training.py --version 4 --features 13 --dry-run
    python src/run_version_training.py --version 25 --features 29

For V1 (which has different variant structure), use:
    python src/wm/v1/run_training.py --features 13

Notes:
- V22, V23, V24, V25 are base-only (no FiLM/snapshot/NCL variants).
  --only is ignored for these.
- Path layout post-2026-04-29 harmonization is `src/wm/v{N}/`.
- Universe selection is done in each version's settings.py or via the
  chimera_legacy dataset path; this wrapper does NOT forward --universe.
  To train V25 on u87, edit `src/wm/v25/v25_training/settings.py` (or
  symlink the relevant chimera_legacy dir) before invoking.
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# V2-V9 variant naming convention:
#   v{N}_training/    -> base world model
#   v{N}_1_training/  -> FiLM adapter variant
#   v{N}_2_training/  -> Snapshot ensemble variant
#   v{N}_3_training/  -> NCL diversity variant
VARIANT_LABELS = {
    "base": ("v{ver}_training", "Base World Model"),
    "film": ("v{ver}_1_training", "FiLM Adapter"),
    "snapshot": ("v{ver}_2_training", "Snapshot Ensemble"),
    "ncl": ("v{ver}_3_training", "NCL Diversity"),
}

# V2-V14 supported feature counts. V25 also accepts 25, 29, 121 (handled per-version below).
SUPPORTED_FEATURES = [13, 18, 25, 29, 30, 34, 37, 41, 121]

# Versions that are base-only (no FiLM/snapshot/NCL variants).
BASE_ONLY_VERSIONS = {22, 23, 24, 25}

# Path-layout convention (post-2026-04-29 harmonization): src/wm/v{N}/
WM_ROOT_NAME = "wm"


def preflight_check(variant_dir: Path, n_features: int) -> str:
    """Run preflight checks. Returns error string or empty string."""
    errors = []

    # 1. Check directory exists
    if not variant_dir.exists():
        return f"Directory not found: {variant_dir}"

    # 2. Check train script exists
    train_script = variant_dir / "train_world_model.py"
    if not train_script.exists():
        return f"Training script not found"

    # 3. Check feature count is supported
    if n_features not in SUPPORTED_FEATURES:
        return f"f{n_features} not supported (options: {SUPPORTED_FEATURES})"

    # 4. Check settings.py compiles
    settings_path = variant_dir / "settings.py"
    if settings_path.exists():
        try:
            import py_compile
            py_compile.compile(str(settings_path), doraise=True)
        except py_compile.PyCompileError as e:
            return f"settings.py compile error: {e}"

    # 5. Check train_world_model.py compiles
    try:
        import py_compile
        py_compile.compile(str(train_script), doraise=True)
    except py_compile.PyCompileError as e:
        return f"train_world_model.py compile error: {e}"

    # 6. Quick settings validation (import and check invariants)
    try:
        saved_path = sys.path[:]
        saved_modules = {}
        for mod_name in ["settings", "components", "world_model"]:
            if mod_name in sys.modules:
                saved_modules[mod_name] = sys.modules.pop(mod_name)

        sys.path.insert(0, str(variant_dir))
        try:
            spec_obj = importlib.util.spec_from_file_location("settings", settings_path)
            settings_mod = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(settings_mod)

            checks = {
                "DIRECT_RETURN_WEIGHT": (getattr(settings_mod, "DIRECT_RETURN_WEIGHT", None), 3.0),
                "WM_STEPS_PER_EPOCH": (getattr(settings_mod, "WM_STEPS_PER_EPOCH", None), 2000),
                "BIN_MIN": (getattr(settings_mod, "BIN_MIN", None), -1.0),
                "BIN_MAX": (getattr(settings_mod, "BIN_MAX", None), 1.0),
                "NUM_BINS": (getattr(settings_mod, "NUM_BINS", None), 255),
                "WM_BATCH_SIZE": (getattr(settings_mod, "WM_BATCH_SIZE", None), 32),
            }
            for name, (actual, expected) in checks.items():
                if actual is not None and actual != expected:
                    errors.append(f"{name}={actual} (expected {expected})")
        finally:
            sys.path[:] = saved_path
            for mod_name in ["settings", "components", "world_model"]:
                sys.modules.pop(mod_name, None)
            for mod_name, mod in saved_modules.items():
                sys.modules[mod_name] = mod
    except Exception as e:
        errors.append(f"Settings import failed: {e}")

    return "; ".join(errors)


def run_variant(variant_dir: Path, n_features: int,
                dry_run: bool = False, clean: bool = False) -> bool:
    """Run training for one variant. Returns True if successful."""
    train_script = variant_dir / "train_world_model.py"

    cmd = [sys.executable, str(train_script), "--features", str(n_features)]
    if clean:
        cmd.append("--clean")

    print(f"\n  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return True

    # Set env vars for stable CUDA memory + utf-8 logs.
    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True prevents fragmentation-
    # induced OOM on long runs (caught V25 OOM at Ep 3 / patch_embed spectral
    # norm clone, 2026-05-09).
    env = dict(__import__("os").environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            timeout=86400,  # 24h max
        )
        elapsed = time.time() - start
        hours = elapsed / 3600
        if result.returncode == 0:
            print(f"\n  [OK] {variant_dir.name} completed in {hours:.1f}h")
            return True
        else:
            print(f"\n  [FAIL] {variant_dir.name} exited with code {result.returncode} after {hours:.1f}h")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {variant_dir.name} exceeded 24h limit")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {variant_dir.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Version Training Runner (V2-V9) -- serial training with preflight"
    )
    parser.add_argument("--version", type=int, required=True,
                        choices=list(range(2, 15)) + [22, 23, 24, 25],
                        help="Model version (2-14, 22-25)")
    parser.add_argument("--features", type=int, required=True,
                        choices=SUPPORTED_FEATURES,
                        help="Feature count for all variants")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=list(VARIANT_LABELS.keys()),
                        help="Only train specific variants (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight only, don't actually train")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-train even if checkpoint exists")
    parser.add_argument("--clean", action="store_true",
                        help="Use stripped clean model (V3/V6/V9 only)")
    args = parser.parse_args()

    ver = args.version
    n_features = args.features

    # V22-V25 are base-only; ignore --only and force "base"
    if ver in BASE_ONLY_VERSIONS:
        if args.only and args.only != ["base"]:
            print(f"  [info] V{ver} is base-only; ignoring --only={args.only}",
                  flush=True)
        variant_keys = ["base"]
    else:
        variant_keys = args.only or list(VARIANT_LABELS.keys())

    # Post-2026-04-29 model layer harmonization: src/wm/v{N}/
    ver_dir = PROJECT_ROOT / "src" / WM_ROOT_NAME / f"v{ver}"
    if not ver_dir.exists():
        # Defensive: check pre-harmonization path; print clear migration hint
        legacy = PROJECT_ROOT / "src" / f"v{ver}"
        if legacy.exists():
            print(f"Found legacy path {legacy}; this layout is deprecated. "
                   f"Move to src/wm/v{ver}/ per CLAUDE.md MODEL_LAYER.md.",
                   flush=True)
        else:
            print(f"Version directory not found: {ver_dir}", flush=True)
        sys.exit(1)

    print("=" * 70)
    print(f"  V{ver} TRAINING -- f{n_features}")
    print("=" * 70)
    print(f"  Variants: {', '.join(variant_keys)}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    # Build variant list
    variants = []
    for key in variant_keys:
        dir_template, label = VARIANT_LABELS[key]
        dir_name = dir_template.format(ver=ver)
        variant_dir = ver_dir / dir_name
        variants.append((key, variant_dir, f"V{ver} {label}"))

    # Phase 1: Preflight all variants
    print("\n--- PREFLIGHT CHECKS ---")
    ready = []
    skipped = []
    already_done = []
    for key, variant_dir, label in variants:
        error = preflight_check(variant_dir, n_features)
        if error:
            print(f"  [SKIP] {label}: {error}")
            skipped.append((key, error))
            continue

        # Check for existing checkpoint (resume logic)
        if not args.fresh:
            # Checkpoint dir: models/v{ver}/v{ver}_{variant_idx}/base/
            dir_name = variant_dir.name  # e.g., v4_training or v4_1_training
            model_key = dir_name.replace("_training", "")  # e.g., v4 or v4_1
            ckpt_name = f"{model_key}_f{n_features}_wm_best_ema.pt"
            model_dir = PROJECT_ROOT / "models" / f"v{ver}" / model_key / "base"
            if (model_dir / ckpt_name).exists():
                print(f"  [DONE] {label}: checkpoint exists ({ckpt_name})")
                already_done.append((key, variant_dir, label))
                continue

        print(f"  [ OK ] {label}")
        ready.append((key, variant_dir, label))

    if not ready and not already_done:
        print("\nNo variants passed preflight. Nothing to train.")
        sys.exit(1)

    if already_done:
        print(f"\n  Already trained: {len(already_done)} -- use --fresh to re-train")
    print(f"  Ready: {len(ready)}/{len(variants)} variants")
    if skipped:
        print(f"  Skipped: {', '.join(k for k, _ in skipped)}")

    if not ready:
        print("\nAll variants already trained. Nothing to do.")
        print("Use --fresh to force re-training.")
        sys.exit(0)

    # Phase 2: Train sequentially
    print("\n--- TRAINING ---")
    results = {}
    for key, variant_dir, label in ready:
        print(f"\n{'=' * 70}")
        print(f"  STARTING: {label} (f{n_features})")
        print(f"{'=' * 70}")
        success = run_variant(variant_dir, n_features, dry_run=args.dry_run,
                              clean=args.clean)
        results[key] = success

    # Phase 3: Summary
    print(f"\n{'=' * 70}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    done_keys = [k for k, _, _ in already_done]
    for key, _, label in variants:
        if key in results:
            status = "PASS" if results[key] else "FAIL"
        elif key in done_keys:
            status = "DONE"
        else:
            status = "SKIP"
        print(f"  {label:40s} [{status}]")

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\n  Passed: {passed}, Failed: {failed}, Skipped: {len(skipped)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
