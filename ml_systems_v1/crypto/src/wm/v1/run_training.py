#!/usr/bin/env python
"""
V1 Family Training Runner
==========================

Runs world model training for all V1 variants (V1.0, V1.1, V1.4, V1.6)
serially with preflight checks. Skips variants that fail preflight.

Usage:
    python src/wm/v1/run_training.py --features 13
    python src/wm/v1/run_training.py --features 25
    python src/wm/v1/run_training.py --features 34    # all base + SOTA
    python src/wm/v1/run_training.py --features 41    # full (base + SOTA + XD)
    python src/wm/v1/run_training.py --features 13 --only v1_0 v1_1
    python src/wm/v1/run_training.py --features 13 --dry-run
"""
import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
V1_DIR = PROJECT_ROOT / "src" / "wm" / "v1"

# V1 variants in training order (simplest first)
V1_VARIANTS = {
    "v1_0": {
        "dir": "v1_0_training",
        "name": "V1.0 (Baseline Transformer)",
        "features_flag": False,  # V1.0 is hardcoded to f13
        "supported_features": [13],
    },
    "v1_1": {
        "dir": "v1_1_training",
        "name": "V1.1 (XD Anti-Memorization)",
        "features_flag": True,
        "supported_features": [13, 18, 21, 25, 30, 34, 37, 41],
    },
    "v1_4": {
        "dir": "v1_4_training",
        "name": "V1.4 (FeatureAttentionBlock)",
        "features_flag": True,
        "supported_features": [13, 18, 21, 25, 30, 34, 37, 41],
    },
    "v1_6": {
        "dir": "v1_6_training",
        "name": "V1.6 (All Techniques)",
        "features_flag": True,
        "supported_features": [13, 18, 21, 25, 30, 34, 37, 41],
    },
}


def preflight_check(variant_key: str, spec: dict, n_features: int) -> str:
    """Run preflight checks for a variant. Returns error string or empty string."""
    errors = []
    variant_dir = V1_DIR / spec["dir"]

    # 1. Check directory exists
    if not variant_dir.exists():
        return f"Directory not found: {variant_dir}"

    # 2. Check train script exists
    train_script = variant_dir / "train_world_model.py"
    if not train_script.exists():
        return f"Training script not found: {train_script}"

    # 3. Check feature count is supported
    if n_features not in spec["supported_features"]:
        return f"f{n_features} not supported (options: {spec['supported_features']})"

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

            # Check critical invariants
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


def run_variant(variant_key: str, spec: dict, n_features: int,
                dry_run: bool = False) -> bool:
    """Run training for one variant. Returns True if successful."""
    train_script = V1_DIR / spec["dir"] / "train_world_model.py"

    cmd = [sys.executable, str(train_script)]
    if spec["features_flag"] and n_features != 37:  # 37 is default
        cmd.extend(["--features", str(n_features)])

    print(f"\n  Command: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return True

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=86400,  # 24h max
        )
        elapsed = time.time() - start
        hours = elapsed / 3600
        if result.returncode == 0:
            print(f"\n  [OK] {spec['name']} completed in {hours:.1f}h")
            return True
        else:
            print(f"\n  [FAIL] {spec['name']} exited with code {result.returncode} after {hours:.1f}h")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {spec['name']} exceeded 24h limit")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {spec['name']}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="V1 Family Training Runner -- serial training with preflight checks"
    )
    parser.add_argument("--features", type=int, required=True,
                        choices=[13, 18, 21, 25, 30, 34, 37, 41],
                        help="Feature count for all variants")
    parser.add_argument("--only", nargs="+", default=None,
                        choices=list(V1_VARIANTS.keys()),
                        help="Only train specific variants (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preflight only, don't actually train")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-train even if checkpoint exists")
    args = parser.parse_args()

    n_features = args.features
    variants = args.only or list(V1_VARIANTS.keys())

    print("=" * 70)
    print(f"  V1 FAMILY TRAINING -- f{n_features}")
    print("=" * 70)
    print(f"  Variants: {', '.join(variants)}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    # Phase 1: Preflight all variants
    print("\n--- PREFLIGHT CHECKS ---")
    ready = []
    skipped = []
    already_done = []
    for key in variants:
        spec = V1_VARIANTS[key]
        error = preflight_check(key, spec, n_features)
        if error:
            print(f"  [SKIP] {spec['name']}: {error}")
            skipped.append((key, error))
            continue

        # Check for existing checkpoint (resume logic)
        if not args.fresh:
            ckpt_dir = PROJECT_ROOT / "models" / "v1" / key.replace("_", "/", 0)
            # Checkpoint pattern: v1_0_f13_wm_best_ema.pt
            ckpt_name = f"{key}_f{n_features}_wm_best_ema.pt"
            # Search in base/ subdirectory and parent
            ckpt_found = False
            for search_dir in [ckpt_dir / "base", ckpt_dir]:
                if (search_dir / ckpt_name).exists():
                    ckpt_found = True
                    break
            if ckpt_found:
                print(f"  [DONE] {spec['name']}: checkpoint exists ({ckpt_name})")
                already_done.append(key)
                continue

        print(f"  [ OK ] {spec['name']}")
        ready.append(key)

    if not ready and not already_done:
        print("\nNo variants passed preflight. Nothing to train.")
        sys.exit(1)

    if already_done:
        print(f"\n  Already trained: {len(already_done)} "
              f"({', '.join(already_done)}) -- use --fresh to re-train")
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
    for key in ready:
        spec = V1_VARIANTS[key]
        print(f"\n{'=' * 70}")
        print(f"  STARTING: {spec['name']} (f{n_features})")
        print(f"{'=' * 70}")
        success = run_variant(key, spec, n_features, dry_run=args.dry_run)
        results[key] = success

    # Phase 3: Summary
    print(f"\n{'=' * 70}")
    print("  TRAINING SUMMARY")
    print(f"{'=' * 70}")
    for key in variants:
        spec = V1_VARIANTS[key]
        if key in results:
            status = "PASS" if results[key] else "FAIL"
        elif key in already_done:
            status = "DONE"
        else:
            status = "SKIP"
        print(f"  {spec['name']:40s} [{status}]")

    passed = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    print(f"\n  Passed: {passed}, Failed: {failed}, Skipped: {len(skipped)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
