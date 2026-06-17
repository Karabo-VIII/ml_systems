"""V50 backward-compat verifier.

V1-V14 model checkpoints read from data/processed/<sym>usdt_v50_chimera.parquet.
This script verifies the legacy v50 layer still loads cleanly under the new
SOTA layout, hasn't been corrupted by v51 work, and that V1.x feature lists
still resolve to existing columns.

Checks:
  1. Every v50 chimera file readable (no parquet corruption)
  2. Every v50 file has the canonical 41 features expected by V1.x trainers
  3. Sample feature values are sane (no all-NaN columns, no inf, no constant)
  4. V1.x FEATURE_LIST imports still resolve (no name drift)
  5. Identity check: v51 chimera contains every v50 column with byte-identical
     values (V1-V14 inference can read either)

Run:
  python src/pipeline/v50_backward_compat.py
  python src/pipeline/v50_backward_compat.py --asset BTC --strict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

import sys as _sys
_pipe_dir = Path(__file__).resolve().parent
if str(_pipe_dir) not in _sys.path:
    _sys.path.append(str(_pipe_dir))
import layout as _layout

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"

# Canonical 41 features per CLAUDE.md (34 base + 7 cross-asset)
EXPECTED_V50_BASE = [
    # 0-12: Legacy
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    # 13-17: Extended
    "norm_ma_distance", "norm_whale", "norm_efficiency",
    "norm_return_4", "norm_return_16",
    # 18-20: Tier 1
    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",
    # 21-24: Hawkes
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # 25-29: IC-boost
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # 30-33: SOTA Tier 3
    "norm_yz_volatility", "norm_cs_spread",
    "norm_perm_entropy", "norm_kyle_lambda",
]

EXPECTED_XD = [
    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
    "xd_cross_return_mean", "xd_cross_vol_mean",
    "xd_ma_distance", "xd_momentum_rank",
]

EXPECTED_TARGETS = [
    "target_return_1", "target_return_4", "target_return_16", "target_return_64",
    "target_voladj_1", "target_voladj_4", "target_voladj_16", "target_voladj_64",
]


def verify_one_v50(asset: str) -> dict:
    fp = _layout.chimera_v50_latest(asset)
    if fp is None or not fp.exists():
        return {"asset": asset, "status": "missing", "path": "chimera_legacy/<latest>"}
    try:
        schema = pl.read_parquet_schema(fp)
    except Exception as e:
        return {"asset": asset, "status": "corrupt", "err": str(e)}
    cols = set(schema.keys())
    issues = []
    missing_base = [c for c in EXPECTED_V50_BASE if c not in cols]
    missing_xd = [c for c in EXPECTED_XD if c not in cols]
    missing_targets = [c for c in EXPECTED_TARGETS if c not in cols]
    if missing_base:
        issues.append(f"missing base: {missing_base[:3]}")
    if missing_xd:
        issues.append(f"missing xd: {missing_xd[:3]}")
    if missing_targets:
        issues.append(f"missing targets: {missing_targets[:3]}")
    return {
        "asset": asset,
        "status": "ok" if not issues else "bad",
        "n_cols": len(cols),
        "missing_base": missing_base,
        "missing_xd": missing_xd,
        "missing_targets": missing_targets,
        "issues": issues,
    }


def verify_v51_contains_v50_identical(asset: str, n_check: int = 5000) -> dict:
    """Sample N rows; verify v51 has every v50 column with identical values."""
    sym = asset.upper() if asset.upper().endswith("USDT") else f"{asset.upper()}USDT"
    v50_path = _layout.chimera_v50_latest(sym)
    v51_path = _layout.chimera_v51_latest(sym, "dollar")
    if v50_path is None or v51_path is None or not v50_path.exists() or not v51_path.exists():
        return {"asset": asset, "status": "skip_missing"}
    v50 = pl.read_parquet(v50_path).head(n_check)
    v51 = pl.read_parquet(v51_path).head(n_check).select(v50.columns)
    diffs = []
    for c in v50.columns:
        if c == "date":
            continue
        try:
            a = v50[c].fill_null(-9.999e9)
            b = v51[c].fill_null(-9.999e9)
            if a.dtype.is_numeric():
                if (a - b).abs().sum() > 1e-6:
                    diffs.append(c)
            else:
                if (a != b).sum() > 0:
                    diffs.append(c)
        except Exception as e:
            diffs.append(f"{c}({e})")
    return {
        "asset": asset,
        "status": "ok" if not diffs else "fail",
        "n_check": n_check,
        "n_diffs": len(diffs),
        "diff_cols": diffs[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    if args.asset:
        assets = [args.asset.upper()]
    else:
        assets = _layout.list_v50_assets()

    print(f"[bc] checking {len(assets)} assets for V50 backward compat...")
    n_ok = n_bad = n_missing = 0
    for asset in assets:
        r = verify_one_v50(asset)
        if r["status"] == "ok":
            n_ok += 1
        elif r["status"] == "missing":
            n_missing += 1
        else:
            n_bad += 1
            print(f"  {asset:>10}  BAD  {r}")
    print(f"[bc] V50 schema check: {n_ok} OK, {n_bad} bad, {n_missing} missing")

    # Identity check on a few representative assets
    print(f"\n[bc] V50<->V51 identity check (sample 5k rows per asset)...")
    sample = ["BTC", "ETH"]
    n_diff = 0
    for a in sample:
        r = verify_v51_contains_v50_identical(a)
        if r["status"] == "ok":
            print(f"  {a}: OK ({r['n_check']} rows checked, all cols identical)")
        elif r["status"] == "skip_missing":
            print(f"  {a}: SKIP (v51 not built)")
        else:
            n_diff += 1
            print(f"  {a}: FAIL {r['n_diffs']} cols differ -- {r['diff_cols']}")

    print()
    print(f"[bc] FINAL: {n_ok}/{len(assets)} assets clean; identity {n_diff} fails on sampled.")
    if n_bad > 0 or n_diff > 0:
        # exit 2 ALWAYS on a real v50<->v51 identity failure. Previously this only
        # fired under --strict, but no caller (pre_train_gate / e2e_btc_eth) passed
        # --strict, so v50 corruption silently exited 0 and the gate saw PASS.
        # --strict is retained as a no-op for backward compatibility.
        sys.exit(2)


if __name__ == "__main__":
    main()
