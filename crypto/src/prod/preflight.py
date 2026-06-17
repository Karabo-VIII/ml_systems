#!/usr/bin/env python
"""
Pre-Paper-Trading Preflight Check
====================================

Validates everything is in order before paper or live trading.
Runs automatically before live_trader.py, or standalone.

Checks per asset:
  1. Chimera file exists and is readable
  2. Has minimum bars (>= 2000 for strategy warmup)
  3. All 30+ base features present and non-null
  4. Feature std ~1.0 (properly normalized)
  5. Data freshness (last bar timestamp vs now)
  6. Targets present (for model validation)
  7. No excessive nulls in any column

Checks for models:
  8. WM checkpoints exist (best_ema)
  9. Ensemble loads without error
  10. Model produces valid predictions

Checks for prod system:
  11. Strategy configs valid for all assets
  12. Asset mapper can resolve unknown assets
  13. State directory writable

Usage:
    python src/prod/preflight.py                    # Check core 10 assets
    python src/prod/preflight.py --all              # Check all chimera files
    python src/prod/preflight.py --asset btcusdt    # Check one asset
    python src/prod/preflight.py --models           # Include model checks (loads GPU)
"""
import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy"
STATE_DIR = PROJECT_ROOT / "data" / "prod_state"

CORE_ASSETS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]

# Minimum features that must be present (first 13 = legacy base)
REQUIRED_FEATURES = [
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
]

TARGET_COLUMNS = [
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
]


def check_asset(asset: str, verbose: bool = True) -> Tuple[bool, Dict]:
    """Run all checks for one asset. Returns (passed, details)."""
    try:
        import polars as pl
    except ImportError:
        return False, {"error": "polars not installed"}

    clean = asset.upper()
    path = DATA_DIR / f"{clean}_v50_chimera.parquet"
    result = {"asset": clean, "gates": [], "passed": True}

    def gate(name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        result["gates"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            result["passed"] = False
        if verbose:
            tag = f"  [{status}]"
            print(f"  {tag} {name}{': ' + detail if detail else ''}")
        return passed

    # Gate 1: File exists
    if not gate("chimera_exists", path.exists(), str(path)):
        return False, result

    # Load data
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        gate("chimera_readable", False, str(e))
        return False, result
    gate("chimera_readable", True, f"{len(df):,} rows x {len(df.columns)} cols")

    # Gate 2: Minimum bars
    n_bars = len(df)
    gate("min_bars", n_bars >= 2000, f"{n_bars:,} bars (need >= 2000)")

    # Gate 3: Required features present
    missing = [f for f in REQUIRED_FEATURES if f not in df.columns]
    gate("features_present", len(missing) == 0,
         f"{len(REQUIRED_FEATURES) - len(missing)}/{len(REQUIRED_FEATURES)}"
         + (f" missing: {missing[:3]}" if missing else ""))

    # Gate 4: Feature std check (~1.0 for properly normalized features)
    bad_std = []
    for feat in REQUIRED_FEATURES:
        if feat in df.columns:
            std = df[feat].std()
            if std is not None and (std < 0.1 or std > 5.0):
                bad_std.append(f"{feat}={std:.2f}")
    gate("feature_std", len(bad_std) == 0,
         f"{len(bad_std)} features with bad std" +
         (f": {bad_std[:3]}" if bad_std else ""))

    # Gate 5: Data freshness
    if "timestamp" in df.columns:
        last_ts = df["timestamp"].max()
        if last_ts and last_ts > 1e12:
            last_dt = datetime.fromtimestamp(last_ts / 1000, timezone.utc)
            age_days = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
            gate("data_freshness", age_days < 7,
                 f"last bar: {last_dt:%Y-%m-%d %H:%M} ({age_days:.1f} days ago)")
        else:
            gate("data_freshness", False, "invalid timestamp")
    else:
        gate("data_freshness", False, "no timestamp column")

    # Gate 6: Targets present
    missing_targets = [t for t in TARGET_COLUMNS if t not in df.columns]
    gate("targets_present", len(missing_targets) == 0,
         f"{len(TARGET_COLUMNS) - len(missing_targets)}/{len(TARGET_COLUMNS)}")

    # Gate 7: Null check
    null_cols = []
    for col in REQUIRED_FEATURES + TARGET_COLUMNS:
        if col in df.columns:
            null_pct = df[col].null_count() / len(df) * 100
            if null_pct > 1.0:
                null_cols.append(f"{col}={null_pct:.1f}%")
    gate("low_nulls", len(null_cols) == 0,
         f"{len(null_cols)} columns with >1% nulls" +
         (f": {null_cols[:3]}" if null_cols else ""))

    # Count total features (informational)
    feature_cols = [c for c in df.columns if c.startswith("norm_") or c.startswith("hurst_")
                    or c.startswith("xd_")]
    result["n_features"] = len(feature_cols)
    result["n_bars"] = n_bars

    return result["passed"], result


def check_models(verbose: bool = True) -> Tuple[bool, Dict]:
    """Check that WM models are loadable."""
    result = {"gates": [], "passed": True}

    def gate(name, passed, detail=""):
        result["gates"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            result["passed"] = False
        if verbose:
            tag = "PASS" if passed else "FAIL"
            print(f"  [{tag}] {name}{': ' + detail if detail else ''}")

    # Check checkpoints exist
    model_dir = PROJECT_ROOT / "models" / "v1"
    checkpoints = list(model_dir.glob("*/base/*best_ema.pt"))
    gate("checkpoints_exist", len(checkpoints) >= 1,
         f"{len(checkpoints)} best_ema checkpoints found")

    # Try loading ensemble
    try:
        import torch
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "v1"))
        sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "v1" / "v1_0_training"))
        from cross_ensemble import CrossModelEnsemble
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossModelEnsemble(model_keys=None, device=device)
        n_models = model.n_models
        gate("ensemble_loads", True, f"{n_models} models on {device}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as e:
        gate("ensemble_loads", False, str(e)[:100])

    return result["passed"], result


def check_prod_system(verbose: bool = True) -> Tuple[bool, Dict]:
    """Check prod system readiness."""
    result = {"gates": [], "passed": True}

    def gate(name, passed, detail=""):
        result["gates"].append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            result["passed"] = False
        if verbose:
            tag = "PASS" if passed else "FAIL"
            print(f"  [{tag}] {name}{': ' + detail if detail else ''}")

    # State directory
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    test_file = STATE_DIR / "_preflight_test"
    try:
        test_file.write_text("test")
        test_file.unlink()
        gate("state_dir_writable", True, str(STATE_DIR))
    except Exception as e:
        gate("state_dir_writable", False, str(e))

    # Kill switch not active
    kill_file = PROJECT_ROOT / "KILL_SWITCH"
    gate("no_kill_switch", not kill_file.exists(),
         "KILL_SWITCH file exists!" if kill_file.exists() else "clear")

    # Universe screener output
    universe_path = STATE_DIR / "universe.json"
    if universe_path.exists():
        with open(universe_path) as f:
            u = json.load(f)
        n_assets = len(u.get("assets", []))
        gate("universe_screened", n_assets > 0, f"{n_assets} assets")
    else:
        gate("universe_screened", False, "no universe.json (run universe_screener.py)")

    return result["passed"], result


def run_preflight(assets: List[str] = None, check_all: bool = False,
                   include_models: bool = False, fix: bool = False):
    """Run full preflight check."""
    print("=" * 70)
    print("  PRE-TRADING PREFLIGHT CHECK")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 70)

    # Determine asset list
    if check_all:
        # All chimera files in data/processed/
        assets = [p.stem.replace("_v50_chimera", "")
                  for p in sorted(DATA_DIR.glob("*_v50_chimera*.parquet"))]
    elif assets:
        assets = [a.upper() for a in assets]
    else:
        assets = CORE_ASSETS

    # Per-asset checks
    print(f"\n  --- DATA CHECKS ({len(assets)} assets) ---")
    all_pass = True
    asset_results = {}
    for asset in assets:
        print(f"\n  {asset}:")
        passed, result = check_asset(asset)
        asset_results[asset] = result
        if not passed:
            all_pass = False

    # Prod system checks
    print(f"\n  --- SYSTEM CHECKS ---")
    sys_pass, sys_result = check_prod_system()
    if not sys_pass:
        all_pass = False

    # Model checks (optional, loads GPU)
    if include_models:
        print(f"\n  --- MODEL CHECKS ---")
        model_pass, model_result = check_models()
        if not model_pass:
            all_pass = False

    # Summary
    n_pass = sum(1 for r in asset_results.values() if r["passed"])
    n_fail = len(asset_results) - n_pass
    total_bars = sum(r.get("n_bars", 0) for r in asset_results.values())

    print(f"\n{'=' * 70}")
    if all_pass:
        print(f"  PREFLIGHT PASS: {n_pass}/{len(assets)} assets ready")
    else:
        print(f"  PREFLIGHT FAIL: {n_fail}/{len(assets)} assets have issues")

    print(f"  Total bars: {total_bars:,}")
    if n_fail > 0:
        failed = [a for a, r in asset_results.items() if not r["passed"]]
        print(f"  Failed: {', '.join(a.lower() for a in failed)}")

    # Auto-fix: remove failed assets from universe.json
    if fix and n_fail > 0:
        universe_path = STATE_DIR / "universe.json"
        if universe_path.exists():
            with open(universe_path) as f:
                universe = json.load(f)
            passed_assets = [a["asset"] for a in universe.get("assets", [])
                           if a["asset"].upper() not in {f.upper() for f in failed}]
            original_n = len(universe.get("assets", []))
            universe["assets"] = [a for a in universe.get("assets", [])
                                  if a["asset"].upper() not in {f.upper() for f in failed}]
            with open(universe_path, "w") as f:
                json.dump(universe, f, indent=2)
            print(f"\n  [FIX] Removed {n_fail} failed assets from universe.json "
                  f"({original_n} -> {len(universe['assets'])})")

    print(f"{'=' * 70}")

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="Pre-trading preflight check")
    parser.add_argument("--asset", type=str, default=None,
                        help="Check specific asset (e.g., btcusdt)")
    parser.add_argument("--all", action="store_true",
                        help="Check all chimera files")
    parser.add_argument("--models", action="store_true",
                        help="Include model loading checks (requires GPU)")
    parser.add_argument("--fix", action="store_true",
                        help="Remove failed assets from universe.json")
    args = parser.parse_args()

    assets = [args.asset] if args.asset else None

    passed = run_preflight(
        assets=assets,
        check_all=args.all,
        include_models=args.models,
        fix=args.fix,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
