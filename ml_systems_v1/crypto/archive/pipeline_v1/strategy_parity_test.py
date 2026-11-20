"""Strategy parity test — verify chimera v51 produces same feature values as
the original frontier sources for any strategy that needs them.

For each migration candidate (etf_flow_overlay, stable_flow_overlay, ...),
this script:
  1. Loads features the same way the original strategy does (raw frontier).
  2. Loads features from chimera v51 (with prefix lookup).
  3. Verifies they match within numerical tolerance for overlapping (asset, date).

If parity holds: the strategy can be migrated by a 5-line change (replace
data load path with v51 read).

Usage:
  python src/pipeline/strategy_parity_test.py            # all checks
  python src/pipeline/strategy_parity_test.py --asset BTC
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

import layout as _layout

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA = PROJECT_ROOT / "data"
RAW_EXTERNAL = DATA / "raw_external"
# Panels post-2026-04-26: silver multi-asset wide panels live in processed/panels/
# (hawkes_* go to processed/hawkes/ but parity test reads only the wide panels here)


@dataclass
class ParityCase:
    name: str
    raw_path: Path
    raw_features: list[str]          # column names in raw file
    v51_features: list[str]          # column names in chimera v51 (with prefix)
    raw_layout: str                  # 'global' or 'per_asset'
    raw_asset_col: str | None = None


CASES = [
    ParityCase(
        name="etf_flow_features",
        raw_path=RAW_EXTERNAL / "farside" / "etf_flow_features.parquet",
        raw_features=["btc_etf_total_z30", "btc_etf_inflow_shock",
                      "eth_etf_total_z30", "any_inflow_shock"],
        v51_features=["etf_btc_etf_total_z30", "etf_btc_etf_inflow_shock",
                      "etf_eth_etf_total_z30", "etf_any_inflow_shock"],
        raw_layout="global",
    ),
    ParityCase(
        name="stable_flow_features",
        raw_path=RAW_EXTERNAL / "defillama" / "stable_flow_features.parquet",
        raw_features=["total_zscore_30d", "stable_shock", "compound_shock"],
        v51_features=["stbl_total_zscore_30d", "stbl_stable_shock", "stbl_compound_shock"],
        raw_layout="global",
    ),
    ParityCase(
        name="hawkes_branching",
        raw_path=_layout.hawkes_panel_latest("hawkes_branching_daily")
                 or (_layout.hawkes_dir() / "hawkes_branching_daily.parquet"),
        raw_features=["eta_total", "eta_imbalance"],
        v51_features=["hbr_eta_total", "hbr_eta_imbalance"],
        raw_layout="per_asset",
        raw_asset_col="asset",
    ),
    ParityCase(
        name="liq_features",
        raw_path=_layout.panel_latest("liq_features_long")
                 or (_layout.panels_dir() / "liq_features_long.parquet"),
        raw_features=["liq_total_usd", "liq_long_z30", "liq_capitulation"],
        v51_features=["liq_total_usd", "liq_long_z30", "liq_capitulation"],
        raw_layout="per_asset",
        raw_asset_col="asset",
    ),
    ParityCase(
        name="s3_features",
        raw_path=_layout.panel_latest("s3_features_long")
                 or (_layout.panels_dir() / "s3_features_long.parquet"),
        raw_features=["top_pos_lsr_z", "smart_vs_retail", "smart_bullish"],
        v51_features=["s3_top_pos_lsr_z", "s3_smart_vs_retail", "s3_smart_bullish"],
        raw_layout="per_asset",
        raw_asset_col="asset",
    ),
]


def normalize_date(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    if col not in df.columns:
        return df
    if df.schema[col] != pl.Date:
        df = df.with_columns(pl.col(col).cast(pl.Date))
    return df


def test_one_case(case: ParityCase, asset: str) -> dict:
    """Compare feature values between raw frontier and v51 chimera."""
    asset_u = asset.upper()
    sym = asset_u if asset_u.endswith("USDT") else f"{asset_u}USDT"
    sym_short = sym.replace("USDT", "")
    v51_path = _layout.chimera_v51_latest(sym, "dollar")
    if v51_path is None or not v51_path.exists():
        return {"case": case.name, "asset": sym_short, "status": "skip_no_v51"}
    if not case.raw_path.exists():
        return {"case": case.name, "asset": asset_u, "status": "skip_no_raw"}

    # Load raw
    raw = pl.read_parquet(case.raw_path)
    raw = normalize_date(raw)
    if case.raw_layout == "per_asset":
        raw = raw.filter(pl.col(case.raw_asset_col).str.to_uppercase() == sym_short)
        raw = raw.select(["date"] + case.raw_features).unique("date").sort("date")
    elif case.raw_layout == "global":
        raw = raw.select(["date"] + case.raw_features).unique("date").sort("date")

    if len(raw) == 0:
        return {"case": case.name, "asset": sym_short, "status": "skip_empty_raw"}

    # Load v51 chimera, derive date, deduplicate
    v51_chim = pl.read_parquet(
        v51_path,
        columns=["timestamp"] + case.v51_features,
    )
    v51_chim = v51_chim.with_columns(
        pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date")
    )
    v51_daily = v51_chim.sort("timestamp").group_by("date").last().drop("timestamp")
    v51_daily = v51_daily.sort("date")

    # Inner-join both
    rename_raw = {raw_c: f"raw__{raw_c}" for raw_c in case.raw_features}
    rename_v51 = {v51_c: f"v51__{v51_c}" for v51_c in case.v51_features}
    j = raw.rename(rename_raw).join(v51_daily.rename(rename_v51), on="date", how="inner")
    if len(j) == 0:
        return {"case": case.name, "asset": sym_short, "status": "no_overlap"}

    # Compare each pair
    metrics = {}
    fails = []
    for raw_c, v51_c in zip(case.raw_features, case.v51_features):
        rA = j[f"raw__{raw_c}"].fill_null(np.nan).to_numpy()
        rB = j[f"v51__{v51_c}"].fill_null(np.nan).to_numpy()
        # mask non-NaN in both
        mask = np.isfinite(rA) & np.isfinite(rB)
        if mask.sum() == 0:
            metrics[raw_c] = {"compared": 0}
            continue
        diffs = np.abs(rA[mask] - rB[mask])
        max_diff = float(diffs.max())
        n = int(mask.sum())
        metrics[raw_c] = {
            "compared": n,
            "max_abs_diff": max_diff,
            "match": bool(max_diff < 1e-6),
        }
        if max_diff >= 1e-6:
            fails.append(f"{raw_c}<->{v51_c}: max_diff={max_diff:.2e} on n={n}")

    return {
        "case": case.name,
        "asset": sym_short,
        "status": "ok" if not fails else "mismatch",
        "n_overlap_dates": len(j),
        "metrics": metrics,
        "fails": fails,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None)
    args = parser.parse_args()

    if args.asset:
        assets = [args.asset.upper()]
    else:
        # All assets that have v51 in new layout
        assets = [a.replace("USDT", "") for a in _layout.list_v51_assets()][:3]

    results = []
    n_ok = n_skip = n_mismatch = 0
    for asset in assets:
        for case in CASES:
            r = test_one_case(case, asset)
            results.append(r)
            if r["status"] == "ok":
                n_ok += 1
                fc = list(r["metrics"].keys())
                ns = ", ".join(f"{c}({r['metrics'][c].get('compared', 0)}d)" for c in fc[:3])
                print(f"[ OK ] {asset:>5} {case.name:<22} {r['n_overlap_dates']:>5} dates  feats: {ns}")
            elif r["status"] == "mismatch":
                n_mismatch += 1
                print(f"[FAIL] {asset:>5} {case.name:<22} {r['n_overlap_dates']:>5} dates  -- {r['fails']}")
            else:
                n_skip += 1
                print(f"[skip] {asset:>5} {case.name:<22} -- {r['status']}")

    print()
    print(f"Summary: {n_ok} ok, {n_skip} skip, {n_mismatch} mismatch (of {len(results)})")
    if n_mismatch > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
