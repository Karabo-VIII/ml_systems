"""
End-to-End Pipeline Inspector -- Single PASS/FAIL verdict.

Runs stage inspectors in sequence, adds cross-stage consistency
checks, and returns exit code 0 (PASS) or 1 (FAIL).

Stages:
  1. Raw data inspection (inspect_raw_data.py)
  2. Dataset inspection (inspect_dataset.py) -- base + cross-asset features
  3. Cross-stage consistency checks

Usage:
    python src/pipeline/inspect_pipeline.py           # Informational only
    python src/pipeline/inspect_pipeline.py --strict   # CI gate mode (exit code)
"""
import sys
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PIPELINE_DIR = PROJECT_ROOT / "src" / "pipeline"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def run_inspector(script_name, strict=False):
    """Run an inspector script and capture exit code."""
    script_path = PIPELINE_DIR / script_name
    if not script_path.exists():
        print(f"  [SKIP] {script_name} not found")
        return -1

    cmd = [sys.executable, str(script_path)]
    if strict:
        cmd.append("--strict")
    result = subprocess.run(cmd, timeout=600)
    return result.returncode


def check_cross_stage_consistency():
    """Verify schema consistency across all chimera files."""
    try:
        import polars as pl
    except ImportError:
        print("  [SKIP] polars not available for cross-stage checks")
        return []

    # post-2026-04-26: legacy v50 chimeras live under processed/chimera_legacy/
    legacy_dir = PROCESSED_DIR / "chimera_legacy"
    chimera_files = sorted(legacy_dir.glob("*_v50_chimera*.parquet")) if legacy_dir.exists() else []
    issues = []

    if not chimera_files:
        issues.append(f"No chimera parquet files found in {legacy_dir}")
        return issues

    for f in chimera_files:
        try:
            df_schema = pl.read_parquet(f, n_rows=0)
        except Exception as e:
            issues.append(f"{f.name}: Cannot read parquet: {e}")
            continue

        symbol = f.stem.split("_")[0].upper()
        n_cols = len(df_schema.columns)

        # Should have at least 48 columns (11 OHLCV + 21 base features + 6 XD + 10 targets)
        if n_cols < 48:
            issues.append(f"{symbol}: Only {n_cols} columns (expected >= 48)")

        # Check XD features present (V0/V1 need these)
        xd_present = [c for c in df_schema.columns if c.startswith("xd_")]
        if len(xd_present) < 7:  # 34 base + 7 cross-asset (xd_) per STATE.md schema
            issues.append(f"{symbol}: Only {len(xd_present)}/7 XD features present")

        # Check core features present
        core_features = [
            "norm_deviation", "norm_fd_close", "norm_vpin",
            "norm_flow_imbalance", "norm_vol_cluster",
        ]
        missing_core = [c for c in core_features if c not in df_schema.columns]
        if missing_core:
            issues.append(f"{symbol}: Missing core features: {missing_core}")

        # Check targets present
        core_targets = [
            "target_return_1", "target_return_4",
            "target_return_16", "target_return_64",
        ]
        missing_targets = [t for t in core_targets if t not in df_schema.columns]
        if missing_targets:
            issues.append(f"{symbol}: Missing targets: {missing_targets}")

    # Check that all 10 expected assets have files
    expected_assets = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
    ]
    found_assets = [f.stem.split("_")[0].upper() for f in chimera_files]
    missing_assets = [a for a in expected_assets if a not in found_assets]
    if missing_assets:
        issues.append(f"Missing chimera files for assets: {missing_assets}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Inspector")
    parser.add_argument("--strict", action="store_true",
                        help="CI gate mode: return exit code 1 on any failure")
    args = parser.parse_args()

    print("=" * 70)
    print("  END-TO-END PIPELINE INSPECTION")
    print("=" * 70)

    all_pass = True
    stage_results = {}

    # Stage 1: Raw data
    print(f"\n--- STAGE 1: Raw Data ---")
    rc = run_inspector("inspect_raw_data.py", strict=args.strict)
    stage_results["raw_data"] = rc
    if rc != 0 and rc != -1:
        all_pass = False
        print(f"  [FAIL] Raw data inspection failed (exit code {rc})")
    elif rc == 0:
        print(f"  [OK] Raw data inspection passed")

    # Stage 2: Dataset (base + cross-asset features)
    print(f"\n--- STAGE 2: Dataset (Base + Cross-Asset Features) ---")
    rc = run_inspector("inspect_dataset.py", strict=args.strict)
    stage_results["dataset"] = rc
    if rc != 0 and rc != -1:
        all_pass = False
        print(f"  [FAIL] Dataset inspection failed (exit code {rc})")
    elif rc == 0:
        print(f"  [OK] Dataset inspection passed")

    # Stage 3: Cross-stage consistency
    print(f"\n--- STAGE 3: Cross-Stage Consistency ---")
    issues = check_cross_stage_consistency()
    if issues:
        for issue in issues:
            print(f"  [ISSUE] {issue}")
        if args.strict:
            all_pass = False
    else:
        print(f"  [OK] All chimera files have consistent schema")

    # Final verdict
    print(f"\n{'='*70}")
    if all_pass:
        print(f"  [PASS] All pipeline stages passed inspection")
    else:
        print(f"  [FAIL] Pipeline inspection detected issues")
    print(f"{'='*70}")

    if args.strict:
        sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
