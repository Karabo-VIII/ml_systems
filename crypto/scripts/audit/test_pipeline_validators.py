"""Meta-validator smoke test — validators-of-validators.

Purpose
=======
Catches the class of bugs where a validator itself is silently broken:
  - 513a889 (2026-05-22): validate_chimera freshness check broken for
    bar-level chimera (per-day collapse). The validator returned `ok`
    on data that should have flagged stale, because the freshness check's
    own date-grouping logic was buggy.

This module exercises each validator with synthetic-but-realistic input
and asserts the output shape + sensitivity (does the validator flag a
SYNTHETIC LEAK we know it should catch?).

Invocation
==========
    python scripts/audit/test_pipeline_validators.py            # run all, exit 0/1/2
    python scripts/audit/test_pipeline_validators.py --verbose  # show every test

Exit codes:
  0  all validators pass meta-test (correctly detect synthetic issue)
  1  warnings (validator behavior is unclear but not broken)
  2  CRITICAL — a validator FAILS to detect a synthetic issue it should catch

Wired into:
  - `src/audit/check_invariants.py` (suggested addition: run as part of CDAP gate)
  - manual pre-release sanity check
"""
from __future__ import annotations

# CDAP contract
__contract__ = {
    "kind": "meta_validator",
    "stage": "pipeline_validator_smoke",
    "inputs": {
        "synthetic_chimera_with_known_leak": True,
        "real_feature_registry": "config/feature_registry.yaml",
    },
    "outputs": {
        "exit_code": "0=pass, 1=warn, 2=CRITICAL — a validator missed a known leak",
    },
    "invariants": {
        "deterministic": True,
        "no_network": True,
        "no_data_write": True,
    },
    "rationale": "Closes pipeline validator A+ gap. Catches the validate_chimera-self-bug class (513a889) by exercising each validator against synthetic input with a known leak/issue.",
}

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Ensure src is importable.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _make_synthetic_chimera_with_known_leak(n: int = 500, seed: int = 42) -> pl.DataFrame:
    """Build a synthetic chimera DataFrame containing:
      - A clean noise feature (should be flagged ok by all checks)
      - A LINEAR-leak feature: f_leak_linear = next-day target * 0.9 + noise
        → check_lookahead_correlation MUST flag this (|corr| ≈ 0.9)
      - A NON-LINEAR-leak feature: f_leak_mi = |next-day target| * sin(t)
        → check_lookahead_mutual_info MUST flag this (high MI, low Pearson)
      - A reserved-name violation: target_return_99 (should fail naming lint)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    target_return_1 = rng.normal(0, 0.03, n)
    f_clean = rng.normal(0, 1, n)
    f_leak_linear = target_return_1 * 0.9 + rng.normal(0, 0.005, n)
    f_leak_mi = np.abs(target_return_1) * np.sin(t * 0.1) * 30.0 + rng.normal(0, 0.01, n)
    # bm_ret_5d label-named column (should be exempt from leak checks)
    bm_ret_5d_15 = rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=n),
        "asset": ["BTC"] * n,
        "ts": (pd.date_range("2023-01-01", periods=n).astype("int64") // 10**6).astype(int),
        "target_return_1": target_return_1,
        "f_clean": f_clean,
        "f_leak_linear": f_leak_linear,
        "f_leak_mi": f_leak_mi,
        "bm_ret_5d_15": bm_ret_5d_15,
    })
    return pl.from_pandas(df)


def test_lookahead_correlation_catches_linear_leak(verbose: bool = False) -> tuple[str, str]:
    """check_lookahead_correlation must flag f_leak_linear (|corr| ≈ 0.9)."""
    from pipeline.validate_chimera import check_lookahead_correlation
    df = _make_synthetic_chimera_with_known_leak()
    results = check_lookahead_correlation(df)
    fails = [r for r in results if r.severity == "fail"]
    flagged_features = [r.metrics.get("feature") for r in fails if r.metrics]
    if "f_leak_linear" in flagged_features:
        return ("test_lookahead_correlation", "ok")
    return ("test_lookahead_correlation",
            f"fail — f_leak_linear (|corr|≈0.9) NOT flagged. Flagged: {flagged_features}")


def test_lookahead_mutual_info_catches_nonlinear_leak(verbose: bool = False) -> tuple[str, str]:
    """check_lookahead_mutual_info must flag f_leak_mi (high MI, low Pearson)."""
    from pipeline.validate_chimera import check_lookahead_mutual_info
    df = _make_synthetic_chimera_with_known_leak()
    results = check_lookahead_mutual_info(df)
    fails = [r for r in results if r.severity == "fail"]
    flagged_features = [r.metrics.get("feature") for r in fails if r.metrics]
    if "f_leak_mi" in flagged_features:
        return ("test_lookahead_mutual_info", "ok")
    return ("test_lookahead_mutual_info",
            f"fail — f_leak_mi (high MI, low Pearson) NOT flagged. Flagged: {flagged_features}")


def test_lookahead_checks_exempt_label_columns(verbose: bool = False) -> tuple[str, str]:
    """Both Pearson and MI checks must NOT flag bm_ret_5d_15 (label-pattern exempt)."""
    from pipeline.validate_chimera import check_lookahead_correlation, check_lookahead_mutual_info
    df = _make_synthetic_chimera_with_known_leak()
    for fn in (check_lookahead_correlation, check_lookahead_mutual_info):
        results = fn(df)
        fails = [r for r in results if r.severity == "fail"]
        flagged_features = [r.metrics.get("feature") for r in fails if r.metrics]
        if "bm_ret_5d_15" in flagged_features:
            return ("test_label_exempt",
                    f"fail — {fn.__name__} flagged bm_ret_5d_15 which should be exempt")
    return ("test_label_exempt", "ok")


def test_lookahead_checks_pass_clean_feature(verbose: bool = False) -> tuple[str, str]:
    """Both checks must NOT flag f_clean (pure noise)."""
    from pipeline.validate_chimera import check_lookahead_correlation, check_lookahead_mutual_info
    df = _make_synthetic_chimera_with_known_leak()
    for fn in (check_lookahead_correlation, check_lookahead_mutual_info):
        results = fn(df)
        fails = [r for r in results if r.severity == "fail"]
        flagged_features = [r.metrics.get("feature") for r in fails if r.metrics]
        if "f_clean" in flagged_features:
            return ("test_clean_feature_unflagged",
                    f"fail — {fn.__name__} flagged f_clean (pure noise, false positive)")
    return ("test_clean_feature_unflagged", "ok")


def test_registry_contract_C1_detects_missing_schema_version(verbose: bool = False) -> tuple[str, str]:
    """registry_contract_test C1 must FAIL if schema_version is absent."""
    from pipeline.registry_contract_test import C1_schema_version
    # Synthetic registry without schema_version
    reg_no_sv = {"version": "1.0", "sources": {}, "chimera_v51": {"sources_to_join": []}}
    results = C1_schema_version(reg_no_sv)
    if any(r.severity == "fail" for r in results):
        return ("test_C1_missing_schema_version", "ok")
    return ("test_C1_missing_schema_version",
            f"fail — C1 did NOT flag missing schema_version. Returned: {results}")


def test_registry_contract_C3_detects_reserved_name(verbose: bool = False) -> tuple[str, str]:
    """registry_contract_test C3 must FAIL if a feature collides with reserved label tokens."""
    from pipeline.registry_contract_test import C3_reserved_name_guard
    reg_with_leak_feature = {
        "sources": {
            "evil_source": {
                "prefix": "evil_",
                "features": ["target_return_99", "harmless"],
            }
        },
        "chimera_v51": {"sources_to_join": ["evil_source"]},
    }
    results = C3_reserved_name_guard(reg_with_leak_feature)
    if any(r.severity == "fail" for r in results):
        return ("test_C3_reserved_name", "ok")
    return ("test_C3_reserved_name",
            f"fail — C3 did NOT flag target_return_99 as reserved-name violation. {results}")


ALL_TESTS = [
    test_lookahead_correlation_catches_linear_leak,
    test_lookahead_mutual_info_catches_nonlinear_leak,
    test_lookahead_checks_exempt_label_columns,
    test_lookahead_checks_pass_clean_feature,
    test_registry_contract_C1_detects_missing_schema_version,
    test_registry_contract_C3_detects_reserved_name,
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Pipeline-validator meta-test (validators-of-validators)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    n_pass = 0
    n_fail = 0
    failures: list[str] = []
    for tfn in ALL_TESTS:
        try:
            name, status = tfn(verbose=args.verbose)
        except Exception as e:
            n_fail += 1
            failures.append(f"  [EXCEPTION] {tfn.__name__}: {type(e).__name__}: {e}")
            print(f"[EXCEPTION] {tfn.__name__}: {type(e).__name__}: {e}")
            continue
        if status == "ok":
            n_pass += 1
            if args.verbose:
                print(f"[OK]   {name}")
        else:
            n_fail += 1
            failures.append(f"  [{name}] {status}")
            print(f"[FAIL] {name}: {status}")

    print(f"\n[test_pipeline_validators] SUMMARY: {n_pass} pass / {n_fail} fail")
    if failures and not args.verbose:
        print("\nFailures:")
        for f in failures:
            print(f)
    if n_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
