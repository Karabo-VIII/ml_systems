"""
Unified V1 Family Validation Script
====================================
Auto-discovers all trained V1 variant models (V1, V1.1, V1.2, V1.3, V1.4)
and validates each using its own validate_world.py with the correct flags.

Checkpoint config (features, RevIN) is parsed from the filename — no need
to open any checkpoint files.

Usage:
    python src/validate_v1_all.py                          # validate all best_ema
    python src/validate_v1_all.py --include-latest          # also validate latest
    python src/validate_v1_all.py --robust --horizon 1      # robust validation
    python src/validate_v1_all.py --variants v1,v1_1        # specific variants only
    python src/validate_v1_all.py --dry-run                 # show what would run
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All V1-family variant directories and their model base dirs
V1_VARIANTS = {
    "v1_0": {"src": "src/wm/v1/v1_0_training", "models": "models/wm/v1/v1_0/base", "logs": "logs/v1/v1_0"},
    "v1_1": {"src": "src/wm/v1/v1_1_training", "models": "models/wm/v1/v1_1/base", "logs": "logs/v1/v1_1"},
    "v1_2": {"src": "src/wm/v1/v1_2_training", "models": "models/wm/v1/v1_2/base", "logs": "logs/v1/v1_2"},
    "v1_3": {"src": "src/wm/v1/v1_3_training", "models": "models/wm/v1/v1_3/base", "logs": "logs/v1/v1_3"},
    "v1_4": {"src": "src/wm/v1/v1_4_training", "models": "models/wm/v1/v1_4/base", "logs": "logs/v1/v1_4"},
    "v1_5": {"src": "src/wm/v1/v1_5_training", "models": "models/wm/v1/v1_5/base", "logs": "logs/v1/v1_5"},
    "v1_6": {"src": "src/wm/v1/v1_6_training", "models": "models/wm/v1/v1_6/base", "logs": "logs/v1/v1_6"},
    "v1_7": {"src": "src/wm/v1/v1_7_training", "models": "models/wm/v1/v1_7/base", "logs": "logs/v1/v1_7"},
}

# Also check models/wm/v1/v1/ root (older V1 checkpoints stored there)
V1_EXTRA_DIRS = {
    "v1_0": "models/wm/v1/v1_0",
}

# Checkpoint types to look for
CKPT_TYPES = ["best_ema", "latest"]

# Timeout per validation subprocess (seconds)
VALIDATION_TIMEOUT = 1800  # 30 minutes


# ---------------------------------------------------------------------------
# Checkpoint Discovery & Parsing
# ---------------------------------------------------------------------------

def parse_checkpoint_filename(filename: str) -> dict | None:
    """Parse a V1-family checkpoint filename into config.

    Returns dict with keys: variant, n_features, use_revin, ckpt_type
    or None if the filename doesn't match expected patterns.

    Handles both old and new naming conventions:
      Old: v1_1_f13_norevin_wm_best_ema.pt  (_norevin suffix = no RevIN)
      New: v1_1_f13_wm_best_ema.pt           (no suffix = no RevIN, default)
      New: v1_1_f18_revin_wm_best_ema.pt     (_revin suffix = RevIN enabled)

    Examples:
        v1_0_f13_wm_best_ema.pt        -> variant=v1_0, features=13, revin=False, type=best_ema
        v1_1_f13_norevin_wm_best_ema.pt -> variant=v1_1, features=13, revin=False, type=best_ema (old)
        v1_1_f18_wm_best_ema.pt        -> variant=v1_1, features=18, revin=False, type=best_ema (new default)
        v1_1_f18_revin_wm_best_ema.pt  -> variant=v1_1, features=18, revin=True, type=best_ema (new opt-in)
    """
    # All V1 family (v1_0, v1_1, ..., v1_5): v{N}_{M}_f{F}[_revin]_wm_{type}.pt
    # Matches both old (_norevin) and new (_revin / no suffix) conventions
    m = re.match(
        r"^v(\d+_\d+)_f(\d+)(_norevin|_revin)?_wm_(best_ema|latest)\.pt$",
        filename,
    )
    if m:
        variant = f"v{m.group(1)}"
        n_features = int(m.group(2))
        revin_tag = m.group(3)
        # _revin = RevIN ON, _norevin or no tag = RevIN OFF
        use_revin = (revin_tag == "_revin")
        ckpt_type = m.group(4)
        return {
            "variant": variant,
            "n_features": n_features,
            "use_revin": use_revin,
            "ckpt_type": ckpt_type,
        }

    return None


def discover_checkpoints(variants: list[str] | None = None,
                         include_latest: bool = False) -> list[dict]:
    """Scan model directories for V1-family checkpoints.

    Returns list of dicts, each with: variant, n_features, use_revin, ckpt_type, path
    """
    target_types = {"best_ema"}
    if include_latest:
        target_types.add("latest")

    found = []
    seen_keys = set()  # (variant, n_features, use_revin, ckpt_type) for dedup

    # Scan configured base dirs (preferred — newer checkpoints)
    for variant_key, cfg in V1_VARIANTS.items():
        if variants and variant_key not in variants:
            continue

        model_dir = PROJECT_ROOT / cfg["models"]
        if not model_dir.exists():
            continue

        for pt_file in sorted(model_dir.glob("*.pt")):
            parsed = parse_checkpoint_filename(pt_file.name)
            if parsed and parsed["ckpt_type"] in target_types:
                key = (parsed["variant"], parsed["n_features"],
                       parsed["use_revin"], parsed["ckpt_type"])
                if key not in seen_keys:
                    parsed["path"] = pt_file
                    found.append(parsed)
                    seen_keys.add(key)

    # Also check V1 root dir (older checkpoints — only if not already found in base/)
    if variants is None or "v1" in variants:
        v1_root = PROJECT_ROOT / V1_EXTRA_DIRS.get("v1", "")
        if v1_root.exists() and v1_root != PROJECT_ROOT / V1_VARIANTS["v1"]["models"]:
            for pt_file in sorted(v1_root.glob("*.pt")):
                parsed = parse_checkpoint_filename(pt_file.name)
                if parsed and parsed["ckpt_type"] in target_types:
                    key = (parsed["variant"], parsed["n_features"],
                           parsed["use_revin"], parsed["ckpt_type"])
                    if key not in seen_keys:
                        parsed["path"] = pt_file
                        found.append(parsed)
                        seen_keys.add(key)

    return found


# ---------------------------------------------------------------------------
# Subprocess Validation
# ---------------------------------------------------------------------------

def build_command(ckpt: dict) -> list[str]:
    """Build the subprocess command for a given checkpoint."""
    variant = ckpt["variant"]
    cfg = V1_VARIANTS[variant]
    script = PROJECT_ROOT / cfg["src"] / "validate_world.py"
    model_path = ckpt["path"]

    cmd = [sys.executable, str(script), "--model", str(model_path)]

    # V1.0 base doesn't accept --features or --revin
    if variant != "v1_0":
        cmd.extend(["--features", str(ckpt["n_features"])])
        if ckpt["use_revin"]:
            cmd.append("--revin")

    return cmd


def build_robust_command(ckpt: dict, horizon: int) -> list[str]:
    """Build the subprocess command for robust validation."""
    cmd = build_command(ckpt)
    cmd.extend(["--robust", "--horizon", str(horizon)])
    return cmd


def run_validation(ckpt: dict, robust: bool = False,
                   horizon: int = 1) -> dict:
    """Run validation for a single checkpoint. Returns result dict."""
    label = format_label(ckpt)

    if robust:
        cmd = build_robust_command(ckpt, horizon)
    else:
        cmd = build_command(ckpt)

    print(f"\n{'='*70}")
    print(f"  VALIDATING: {label}")
    print(f"  Command: {' '.join(cmd[-6:])}")  # show last 6 args for brevity
    print(f"{'='*70}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=VALIDATION_TIMEOUT,
        )
        elapsed = time.time() - start
        success = result.returncode == 0

        # Print subprocess output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            # Filter out common warnings
            for line in result.stderr.splitlines():
                if "UserWarning" not in line and "FutureWarning" not in line:
                    print(f"  [stderr] {line}")

        return {
            "label": label,
            "ckpt": ckpt,
            "success": success,
            "returncode": result.returncode,
            "elapsed": elapsed,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  [TIMEOUT] Validation exceeded {VALIDATION_TIMEOUT}s limit")
        return {
            "label": label,
            "ckpt": ckpt,
            "success": False,
            "returncode": -1,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": "TIMEOUT",
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {e}")
        return {
            "label": label,
            "ckpt": ckpt,
            "success": False,
            "returncode": -1,
            "elapsed": elapsed,
            "stdout": "",
            "stderr": str(e),
        }


# ---------------------------------------------------------------------------
# JSON Result Collection
# ---------------------------------------------------------------------------

def find_latest_json(log_dir: Path, prefix: str = "validation_",
                     after: datetime | None = None) -> Path | None:
    """Find the most recently created validation JSON in a log directory."""
    if not log_dir.exists():
        return None

    json_files = sorted(
        log_dir.glob(f"{prefix}*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )

    for jf in json_files:
        if after is not None:
            mtime = datetime.fromtimestamp(jf.stat().st_mtime)
            if mtime < after:
                continue
        return jf

    return None


def extract_metrics_from_json(json_path: Path) -> dict | None:
    """Extract key metrics from a validation JSON file."""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    gate_passed = data.get("gate_passed", None)
    results = data.get("results", {})

    if not results:
        return None

    # Aggregate across assets
    all_ic1, all_ic4, all_ic16, all_ic64 = [], [], [], []
    all_recon = []

    for asset_name, asset_data in results.items():
        returns = asset_data.get("returns", {})
        if "1" in returns:
            all_ic1.append(returns["1"].get("ic", 0))
        if "4" in returns:
            all_ic4.append(returns["4"].get("ic", 0))
        if "16" in returns:
            all_ic16.append(returns["16"].get("ic", 0))
        if "64" in returns:
            all_ic64.append(returns["64"].get("ic", 0))
        if "rec_mse" in asset_data:
            all_recon.append(asset_data["rec_mse"])

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "gate_passed": gate_passed,
        "ic_1": mean(all_ic1),
        "ic_4": mean(all_ic4),
        "ic_16": mean(all_ic16),
        "ic_64": mean(all_ic64),
        "recon_mse": mean(all_recon),
        "n_assets": len(results),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def format_label(ckpt: dict) -> str:
    """Format a human-readable label for a checkpoint."""
    variant = ckpt["variant"].upper().replace("_", ".")
    feat = f"f{ckpt['n_features']}"
    revin = "RevIN" if ckpt["use_revin"] else "noRevIN"
    ckpt_type = ckpt["ckpt_type"]
    return f"{variant} {feat} {revin} ({ckpt_type})"


def print_discovery_table(checkpoints: list[dict]):
    """Print discovered checkpoints."""
    print(f"\n{'='*70}")
    print(f"  DISCOVERED CHECKPOINTS ({len(checkpoints)} found)")
    print(f"{'='*70}")

    for i, ckpt in enumerate(checkpoints, 1):
        label = format_label(ckpt)
        path = ckpt["path"].relative_to(PROJECT_ROOT)
        print(f"  {i:2d}. {label:<35} {path}")

    print()


def print_comparison_table(run_results: list[dict]):
    """Print a comparison table of all validation results."""
    print(f"\n{'='*90}")
    print(f"  V1 FAMILY VALIDATION COMPARISON")
    print(f"{'='*90}")

    # Header
    header = (
        f"  {'Model':<30} | {'Feat':>4} | {'RevIN':>5} | "
        f"{'IC_1':>6} | {'IC_4':>6} | {'IC_16':>6} | {'IC_64':>6} | "
        f"{'Recon':>6} | {'Gate':>6} | {'Time':>5}"
    )
    print(header)
    print(f"  {'-'*28}-+-{'-'*4}-+-{'-'*5}-+-"
          f"{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
          f"{'-'*6}-+-{'-'*6}-+-{'-'*5}")

    for rr in run_results:
        ckpt = rr["ckpt"]
        variant = ckpt["variant"].upper().replace("_", ".")
        feat_str = str(ckpt["n_features"])
        revin_str = "Yes" if ckpt["use_revin"] else "No"
        ckpt_type = ckpt["ckpt_type"]
        label = f"{variant} {ckpt_type}"
        if ckpt["n_features"] != 18 or not ckpt["use_revin"]:
            tags = []
            if ckpt["n_features"] != 18:
                tags.append(f"f{ckpt['n_features']}")
            if not ckpt["use_revin"]:
                tags.append("noRV")
            label += f" ({','.join(tags)})"

        elapsed_min = rr["elapsed"] / 60

        metrics = rr.get("metrics")
        if metrics:
            gate_str = "PASS" if metrics["gate_passed"] else "FAIL"
            row = (
                f"  {label:<30} | {feat_str:>4} | {revin_str:>5} | "
                f"{metrics['ic_1']:>6.4f} | {metrics['ic_4']:>6.4f} | "
                f"{metrics['ic_16']:>6.4f} | {metrics['ic_64']:>6.4f} | "
                f"{metrics['recon_mse']:>6.4f} | {gate_str:>6} | {elapsed_min:>4.1f}m"
            )
        elif not rr["success"]:
            row = (
                f"  {label:<30} | {feat_str:>4} | {revin_str:>5} | "
                f"{'--':>6} | {'--':>6} | {'--':>6} | {'--':>6} | "
                f"{'--':>6} | {'ERROR':>6} | {elapsed_min:>4.1f}m"
            )
        else:
            row = (
                f"  {label:<30} | {feat_str:>4} | {revin_str:>5} | "
                f"{'??':>6} | {'??':>6} | {'??':>6} | {'??':>6} | "
                f"{'??':>6} | {'??':>6} | {elapsed_min:>4.1f}m"
            )
        print(row)

    print(f"  {'='*88}")

    # Summary
    passed = sum(1 for r in run_results if r.get("metrics", {}).get("gate_passed"))
    failed = sum(1 for r in run_results if r.get("metrics") and not r["metrics"]["gate_passed"])
    errors = sum(1 for r in run_results if not r["success"])
    no_json = sum(1 for r in run_results if r["success"] and not r.get("metrics"))
    total_time = sum(r["elapsed"] for r in run_results)

    print(f"  {passed} passed, {failed} failed, {errors} errors, {no_json} no-JSON")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unified V1 Family Validation: auto-discover and validate all V1 variant models"
    )
    parser.add_argument("--include-latest", action="store_true",
                        help="Also validate *_wm_latest.pt checkpoints")
    parser.add_argument("--robust", action="store_true",
                        help="Run robust validation (overfitting detection)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Horizon for robust validation (1,4,16,64)")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated list of variants to validate (e.g. v1,v1_1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be validated without running")
    args = parser.parse_args()

    variants = None
    if args.variants:
        variants = [v.strip() for v in args.variants.split(",")]
        unknown = [v for v in variants if v not in V1_VARIANTS]
        if unknown:
            print(f"  [ERROR] Unknown variants: {unknown}")
            print(f"  Available: {list(V1_VARIANTS.keys())}")
            sys.exit(1)

    # Discover checkpoints
    checkpoints = discover_checkpoints(
        variants=variants,
        include_latest=args.include_latest,
    )

    if not checkpoints:
        print("\n  No V1-family checkpoints found.")
        print("  Expected locations:")
        for key, cfg in V1_VARIANTS.items():
            if variants and key not in variants:
                continue
            print(f"    {cfg['models']}/")
        sys.exit(0)

    print_discovery_table(checkpoints)

    if args.dry_run:
        print("  DRY RUN -- commands that would execute:\n")
        for ckpt in checkpoints:
            if args.robust:
                cmd = build_robust_command(ckpt, args.horizon)
            else:
                cmd = build_command(ckpt)
            label = format_label(ckpt)
            # Show command relative to project root
            cmd_short = [str(Path(c).relative_to(PROJECT_ROOT)) if PROJECT_ROOT.as_posix() in c else c for c in cmd]
            print(f"  {label}")
            print(f"    {' '.join(cmd_short)}\n")
        sys.exit(0)

    # Run validations sequentially
    run_results = []
    for ckpt in checkpoints:
        before = datetime.now()
        result = run_validation(ckpt, robust=args.robust, horizon=args.horizon)

        # Try to collect JSON metrics
        variant = ckpt["variant"]
        log_dir = PROJECT_ROOT / V1_VARIANTS[variant]["logs"]
        prefix = "robust_validation_" if args.robust else "validation_"
        json_path = find_latest_json(log_dir, prefix=prefix, after=before)

        if json_path and not args.robust:
            metrics = extract_metrics_from_json(json_path)
            result["metrics"] = metrics
            result["json_path"] = json_path
        else:
            result["metrics"] = None
            result["json_path"] = None

        run_results.append(result)

    # Print comparison
    if not args.robust:
        print_comparison_table(run_results)
    else:
        # For robust validation, just summarize pass/fail
        print(f"\n{'='*60}")
        print(f"  ROBUST VALIDATION RESULTS (horizon={args.horizon})")
        print(f"{'='*60}")
        for rr in run_results:
            status = "PASS" if rr["success"] else "FAIL"
            print(f"  [{status}] {rr['label']} ({rr['elapsed']/60:.1f}m)")
        print()


if __name__ == "__main__":
    main()
