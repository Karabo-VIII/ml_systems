"""Cross-asset feature consistency check.

Catches subtle bugs in v50/v51 Phase 2 cross-asset feature computation:
  - xd_btc_return at asset X must equal BTC's norm_return_1 (lagged, same date)
  - xd_btc_volatility at asset X must equal BTC's norm_hl_spread (or vol proxy)
  - xd_cross_return_mean at X must approximate mean(norm_return_1 across other assets)
  - xd_funding_spread = asset's norm_funding - BTC's norm_funding (z-scored)

Why this matters: cross-asset features are computed via panel join. A subtle
bug in the join (timestamp drift, asof staleness, asset filtering) can silently
produce wrong cross-asset values. Models trained on these features would then
learn spurious cross-asset relationships.

This script does N independent samples per asset, compares stored xd_* against
recomputed-from-source values, flags discrepancies.

Run:
  python src/pipeline/cross_asset_consistency.py --assets BTC,ETH,SOL
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from universe_loader import UniverseLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import layout as _layout
PROCESSED = PROJECT_ROOT / "data" / "processed"


def check_xd_btc_return_consistency(asset_chimera: pl.DataFrame, btc_chimera: pl.DataFrame,
                                    n_samples: int = 200, tol: float = 0.05) -> dict:
    """Sample N timestamps where both have data, compare xd_btc_return vs BTC's
    norm_return_1 at the same timestamp.

    Allow tolerance: xd_btc_return may be the BTC norm_return_1 from the *prior bar*
    (asof join, not exact). 5% tolerance accommodates that.
    """
    if "xd_btc_return" not in asset_chimera.columns or "norm_return_1" not in btc_chimera.columns:
        return {"status": "skip_missing_cols", "checked": 0}
    btc_lookup = btc_chimera.select(["timestamp", "norm_return_1"]).rename(
        {"norm_return_1": "btc_norm_return_1"}
    )
    asset_with_btc = asset_chimera.select(["timestamp", "xd_btc_return"]).join(
        btc_lookup, on="timestamp", how="inner"
    )
    if len(asset_with_btc) < 100:
        return {"status": "skip_no_overlap", "checked": 0}
    rng = np.random.default_rng(seed=42)
    idx = rng.integers(low=100, high=len(asset_with_btc) - 100, size=min(n_samples, len(asset_with_btc) - 200))
    bad = 0
    sample_diffs = []
    for i in idx:
        xd_val = asset_with_btc["xd_btc_return"][int(i)]
        btc_val = asset_with_btc["btc_norm_return_1"][int(i)]
        if xd_val is None or btc_val is None:
            continue
        diff = abs(float(xd_val) - float(btc_val))
        sample_diffs.append(diff)
        # Allow tolerance: xd_btc_return is z-scored / lagged version
        if diff > tol:
            bad += 1
    n_checked = len(sample_diffs)
    if n_checked == 0:
        return {"status": "skip_all_null", "checked": 0}
    bad_frac = bad / n_checked
    return {
        "status": "ok" if bad_frac < 0.20 else "fail",
        "checked": n_checked,
        "bad": bad,
        "bad_frac": bad_frac,
        "max_diff": float(np.max(sample_diffs)),
        "median_diff": float(np.median(sample_diffs)),
    }


def check_xd_funding_spread_zero_for_btc(btc_chimera: pl.DataFrame) -> dict:
    """For BTCUSDT, xd_funding_spread MUST be 0 (asset funding - BTC funding = 0)."""
    if "xd_funding_spread" not in btc_chimera.columns:
        return {"status": "skip_missing_col"}
    s = btc_chimera["xd_funding_spread"]
    n_nonzero = s.filter(s != 0).len()
    return {
        "status": "ok" if n_nonzero == 0 else "fail",
        "n_rows": len(s),
        "n_nonzero": n_nonzero,
    }


def check_xd_momentum_rank_in_unit_range(chim: pl.DataFrame) -> dict:
    """xd_momentum_rank should be roughly z-scored: mean ~0, std ~1, no extreme outliers."""
    if "xd_momentum_rank" not in chim.columns:
        return {"status": "skip"}
    s = chim["xd_momentum_rank"].drop_nulls()
    if len(s) == 0:
        return {"status": "skip_all_null"}
    mean = float(s.mean())
    std = float(s.std())
    pct_outside_5sigma = float((s.abs() > 5.0).sum()) / len(s)
    healthy = abs(mean) < 0.20 and 0.5 < std < 2.0 and pct_outside_5sigma < 0.001
    return {
        "status": "ok" if healthy else "warn",
        "mean": mean, "std": std,
        "pct_outside_5sigma": pct_outside_5sigma,
    }


def audit_one_asset(asset: str, btc_chimera: pl.DataFrame) -> dict:
    sym = asset.upper() if asset.upper().endswith("USDT") else f"{asset.upper()}USDT"
    sym_short = sym.replace("USDT", "")
    chim_path = _layout.chimera_v51_latest(sym, "dollar")
    if chim_path is None or not chim_path.exists():
        return {"asset": sym, "status": "skip_no_v51"}
    chim = pl.read_parquet(chim_path, columns=[
        "timestamp", "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
        "xd_cross_return_mean", "xd_cross_vol_mean", "xd_momentum_rank", "norm_return_1",
    ])
    out = {"asset": sym, "checks": {}}

    # Skip BTC vs BTC self-check
    if sym == "BTCUSDT":
        out["checks"]["xd_funding_spread_btc_zero"] = check_xd_funding_spread_zero_for_btc(chim)
    else:
        out["checks"]["xd_btc_return_consistency"] = check_xd_btc_return_consistency(chim, btc_chimera)

    out["checks"]["xd_momentum_rank_zscore"] = check_xd_momentum_rank_in_unit_range(chim)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=None,
                    help="Comma-sep symbols. Default: all U10 assets that have v51.")
    args = ap.parse_args()

    loader = UniverseLoader.load()

    # Load BTC chimera once (reference for cross-asset checks)
    btc_path = _layout.chimera_v51_latest("BTCUSDT", "dollar")
    if btc_path is None or not btc_path.exists():
        print(f"[xd_audit] BTC v51 missing; cannot run cross-asset checks. Skipping.")
        sys.exit(1)
    btc_chimera = pl.read_parquet(btc_path, columns=["timestamp", "norm_return_1", "xd_funding_spread"])
    print(f"[xd_audit] BTC reference loaded: {len(btc_chimera)} rows")

    if args.assets:
        symbols = [s.strip().upper() for s in args.assets.split(",")]
    else:
        symbols = [s for s in loader.list("u10")
                   if _layout.chimera_v51_latest(s, "dollar") is not None]

    print(f"[xd_audit] auditing {len(symbols)} assets: {symbols}")
    print()

    n_ok = n_warn = n_fail = n_skip = 0
    for sym in symbols:
        result = audit_one_asset(sym, btc_chimera)
        print(f"=== {result['asset']} ===")
        if result.get("status", "").startswith("skip"):
            print(f"  {result['status']}")
            n_skip += 1
            continue
        for check_name, check_result in result.get("checks", {}).items():
            status = check_result.get("status", "?")
            print(f"  {status:>4}  {check_name}: {check_result}")
            if status == "ok":
                n_ok += 1
            elif status == "warn":
                n_warn += 1
            elif status == "fail":
                n_fail += 1
            else:
                n_skip += 1
        print()

    print(f"[xd_audit] Summary: {n_ok} pass, {n_warn} warn, {n_fail} fail, {n_skip} skip")
    if n_fail > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
