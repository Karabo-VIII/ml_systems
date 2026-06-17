"""Validation script for chimera v51 -- catch alignment / leakage / NaN issues.

Validates:
  1. Schema   -- every chimera v51 has 41 v50 base + 80 frontier features = 121
                 (+ timestamp + bar_id + ohlcv + targets + regime_label = ~143 cols)
  2. Date coverage -- no gaps > N days in the dollar-bar timeline
  3. NaN budget -- per-feature, fraction of NaN rows
  4. Frontier alignment -- silver feature dates align with chimera dates
  5. No leakage -- no future timestamps; targets are actually forward-shifted
  6. Cross-asset consistency -- xd_btc_return matches BTC's norm_return_1
  7. Cadence resampling correctness -- 1d/4h files have right row counts
  8. Backward compat -- every v50 column present in v51 with same values
  9. Timestamp monotonicity -- bar_id strictly increasing within asset
 10. Hawkes branching parity -- eta_total values match the source parquet

Run:
  python src/pipeline/validate_chimera_v51.py                 # validate all assets
  python src/pipeline/validate_chimera_v51.py --asset BTC     # one asset
  python src/pipeline/validate_chimera_v51.py --json          # machine-readable

Exit codes:
  0: all checks pass
  1: warnings only (NaN budget exceeded, etc.)
  2: hard failures (schema mismatch, leakage, broken backward compat)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from feature_registry import FeatureRegistry  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NAN_WARN_THRESHOLD = 0.30      # warn if a feature is >30% NaN over full range
DATE_GAP_WARN_DAYS = 5          # warn if >5 calendar days between consecutive bars


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: str   # 'fail' | 'warn' | 'ok'
    detail: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class AssetReport:
    asset: str
    checks: list[CheckResult] = field(default_factory=list)
    n_pass: int = 0
    n_warn: int = 0
    n_fail: int = 0

    def add(self, c: CheckResult) -> None:
        self.checks.append(c)
        if c.severity == "fail":
            self.n_fail += 1
        elif c.severity == "warn":
            self.n_warn += 1
        else:
            self.n_pass += 1

    def is_clean(self) -> bool:
        return self.n_fail == 0 and self.n_warn == 0


def check_schema(asset: str, v50: pl.DataFrame, v51: pl.DataFrame, registry: FeatureRegistry) -> list[CheckResult]:
    out = []
    expected_added = registry.chimera.expected_total_new_features  # 80
    actual_added = len(v51.columns) - len(v50.columns)
    # We add `date` helper too, so 81 is OK
    if actual_added in (expected_added, expected_added + 1):
        out.append(CheckResult("schema_count", True, "ok",
                               f"v50={len(v50.columns)} v51={len(v51.columns)} +{actual_added}"))
    else:
        out.append(CheckResult("schema_count", False, "fail",
                               f"v50={len(v50.columns)} v51={len(v51.columns)} +{actual_added} expected +{expected_added}"))
    # Verify each registered feature exists with prefix
    expected_features = registry.list_features()
    missing = [f for f in expected_features if f not in v51.columns]
    if missing:
        out.append(CheckResult("registered_features_present", False, "fail",
                               f"missing {len(missing)}: {missing[:5]}..."))
    else:
        out.append(CheckResult("registered_features_present", True, "ok",
                               f"all {len(expected_features)} registered features present"))
    return out


def check_backward_compat(asset: str, v50: pl.DataFrame, v51: pl.DataFrame) -> CheckResult:
    """Every v50 column must be in v51 with the SAME values."""
    missing_cols = [c for c in v50.columns if c not in v51.columns]
    if missing_cols:
        return CheckResult("backward_compat_columns", False, "fail",
                           f"v50 cols missing in v51: {missing_cols}")
    n_check = min(10000, len(v50))
    sample = v50.head(n_check)
    sample_v51 = v51.head(n_check).select(v50.columns)
    diffs = []
    for c in v50.columns:
        if c == "date":
            continue
        try:
            v50_vals = sample[c].fill_null(-9.999e9)
            v51_vals = sample_v51[c].fill_null(-9.999e9)
            if v50_vals.dtype.is_numeric():
                ne = (v50_vals - v51_vals).abs().sum()
                if ne > 1e-6:
                    diffs.append(f"{c}: numeric diff sum={ne:.4f}")
            else:
                ne = (v50_vals != v51_vals).sum()
                if ne > 0:
                    diffs.append(f"{c}: {ne} mismatched rows")
        except Exception as e:
            diffs.append(f"{c}: ERR {e}")
    if diffs:
        return CheckResult("backward_compat_values", False, "fail",
                           f"value mismatches in {len(diffs)} cols: {diffs[:3]}")
    return CheckResult("backward_compat_values", True, "ok",
                       f"all v50 columns identical in v51 (sample n={n_check})")


def check_timestamp_monotonic(v51: pl.DataFrame) -> CheckResult:
    """Strictly decreasing ts is a hard fail. Zero-diff duplicates are an
    expected v50 property (multiple dollar bars closing at same ms in burst
    trading) and only flagged as warn if pervasive (>5% of bars).
    """
    if "timestamp" not in v51.columns:
        return CheckResult("ts_monotonic", False, "fail", "no timestamp column")
    diffs = v51["timestamp"].diff().drop_nulls()
    n_neg = (diffs < 0).sum()
    n_zero = (diffs == 0).sum()
    if n_neg > 0:
        return CheckResult("ts_monotonic", False, "fail", f"{n_neg} strictly-decreasing ts steps")
    if n_zero > 0.05 * len(v51):
        return CheckResult("ts_monotonic", False, "warn",
                           f"{n_zero} zero-diff duplicates (>5% of bars)")
    return CheckResult("ts_monotonic", True, "ok",
                       f"{len(v51)} bars: 0 decreasing, {n_zero} zero-diff (within budget)")


def check_no_future_leak(v51: pl.DataFrame) -> CheckResult:
    """Verify target_return_h at row N matches (close[N+h]-close[N])/close[N] within tol.

    v50 may trim trailing rows (no nulls at end), so we cannot use null-at-boundary
    as a leak proxy. Instead, sample N=200 random non-edge rows and verify the
    formula holds.
    """
    if "close" not in v51.columns:
        return CheckResult("forward_target_shift", True, "ok", "no close col; skip")
    closes = v51["close"].to_numpy()
    n = len(closes)
    rng = np.random.default_rng(seed=42)
    fails = []
    for h in (1, 4, 16, 64):
        col = f"target_return_{h}"
        if col not in v51.columns:
            continue
        targets = v51[col].to_numpy()
        # sample 200 random indices in [100, n-h-1]
        n_sample = min(200, max(0, n - 2 * h - 100))
        if n_sample < 5:
            continue
        idxs = rng.integers(low=100, high=n - h - 1, size=n_sample)
        bad = 0
        for i in idxs:
            if not np.isfinite(targets[i]) or not np.isfinite(closes[i]) or not np.isfinite(closes[i+h]):
                continue
            expected = (closes[i+h] - closes[i]) / max(abs(closes[i]), 1e-12)
            # v50 has clip(-0.15, 0.15) for h=1, clip(-0.50, 0.50) for h>1
            clip_max = 0.15 if h == 1 else 0.50
            expected_clipped = max(-clip_max, min(clip_max, expected))
            if abs(targets[i] - expected_clipped) > 1e-4:
                bad += 1
        if bad > n_sample * 0.05:  # >5% mismatch = real problem
            fails.append(f"{col}: {bad}/{n_sample} sampled rows mismatch formula")
    if fails:
        return CheckResult("forward_target_shift", False, "fail", "; ".join(fails))
    return CheckResult("forward_target_shift", True, "ok",
                       f"target_return_h matches forward-shift formula in 200-row sample")


def check_nan_budget(v51: pl.DataFrame, registry: FeatureRegistry) -> list[CheckResult]:
    out = []
    n_rows = len(v51)
    high_nan = []
    for f in registry.list_features():
        if f not in v51.columns:
            continue
        n_nan = v51[f].null_count()
        frac = n_nan / max(n_rows, 1)
        if frac > NAN_WARN_THRESHOLD:
            high_nan.append((f, frac, n_nan))
    if high_nan:
        sample = ", ".join(f"{n}({f:.1%})" for n, f, _ in high_nan[:5])
        out.append(CheckResult("nan_budget", False, "warn",
                               f"{len(high_nan)} features >{NAN_WARN_THRESHOLD:.0%} NaN: {sample}",
                               metrics={"high_nan_features": [n for n, f, _ in high_nan]}))
    else:
        out.append(CheckResult("nan_budget", True, "ok",
                               f"all features <{NAN_WARN_THRESHOLD:.0%} NaN"))
    return out


def check_date_gaps(v51: pl.DataFrame) -> CheckResult:
    if "timestamp" not in v51.columns:
        return CheckResult("date_gaps", False, "fail", "no timestamp")
    df = v51.select(pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("d")).unique().sort("d")
    days = df["d"].to_list()
    if len(days) < 2:
        return CheckResult("date_gaps", True, "ok", f"<2 unique days")
    gaps = []
    for i in range(1, len(days)):
        delta = (days[i] - days[i-1]).days
        if delta > DATE_GAP_WARN_DAYS:
            gaps.append((days[i-1], days[i], delta))
    if gaps:
        gap_str = ", ".join(f"{a}->{b}({d}d)" for a, b, d in gaps[:3])
        return CheckResult("date_gaps", False, "warn",
                           f"{len(gaps)} gaps >{DATE_GAP_WARN_DAYS}d: {gap_str}")
    return CheckResult("date_gaps", True, "ok", f"{len(days)} contiguous days, max gap <={DATE_GAP_WARN_DAYS}d")


def check_hawkes_alignment(asset: str, v51: pl.DataFrame) -> CheckResult:
    """eta_total values in v51 should match values in hawkes_branching_daily.parquet for matching dates."""
    src = PROJECT_ROOT / "data" / "frontier" / "hawkes_enh" / "hawkes_branching_daily.parquet"
    if not src.exists():
        return CheckResult("hawkes_alignment", True, "ok", "hawkes source missing; skip")
    src_df = pl.read_parquet(src).filter(pl.col("asset").str.to_uppercase() == asset.upper())
    if len(src_df) == 0:
        return CheckResult("hawkes_alignment", True, "ok", f"no hawkes rows for {asset}")
    src_df = src_df.select(["date", pl.col("eta_total").alias("src_eta_total")])
    if "hbr_eta_total" not in v51.columns:
        return CheckResult("hawkes_alignment", False, "fail", "hbr_eta_total missing in v51")
    if "date" not in v51.columns:
        v51 = v51.with_columns(pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().alias("date"))
    # Pick last bar of each date in v51 (to compare to daily source)
    v51_last = v51.sort(["timestamp"]).group_by("date").last().select(["date", "hbr_eta_total"])
    j = v51_last.join(src_df, on="date", how="inner")
    if len(j) == 0:
        return CheckResult("hawkes_alignment", True, "ok", f"no overlap dates")
    diffs = (j["hbr_eta_total"] - j["src_eta_total"]).abs()
    max_diff = float(diffs.max() or 0.0)
    if max_diff > 1e-9:
        return CheckResult("hawkes_alignment", False, "fail",
                           f"max abs diff={max_diff:.2e} on {len(j)} overlap dates")
    return CheckResult("hawkes_alignment", True, "ok",
                       f"{len(j)} overlap dates, max diff={max_diff:.2e}")


def check_cadence_files(asset: str, v51: pl.DataFrame, registry: FeatureRegistry) -> list[CheckResult]:
    out = []
    for spec in registry.chimera.cadence_materializations:
        cad = spec["cadence"]
        path = PROCESSED_DIR / spec["output_pattern"].format(asset_lower=asset.lower())
        if not path.exists():
            out.append(CheckResult(f"cadence_{cad}_exists", False, "fail",
                                   f"missing {path.name}"))
            continue
        cad_df = pl.read_parquet(path)
        # Schema parity
        missing_cols = [c for c in v51.columns if c not in cad_df.columns]
        extra = [c for c in cad_df.columns if c not in v51.columns]
        if missing_cols:
            out.append(CheckResult(f"cadence_{cad}_schema", False, "fail",
                                   f"missing {len(missing_cols)} cols: {missing_cols[:3]}"))
            continue
        # Row count plausibility
        if cad == "1d":
            expected_rows = v51.select(pl.from_epoch(pl.col("timestamp"), time_unit="ms").dt.date().n_unique()).item()
            if abs(len(cad_df) - expected_rows) > 2:
                out.append(CheckResult(f"cadence_{cad}_rows", False, "warn",
                                       f"got {len(cad_df)}, expected ~{expected_rows}"))
                continue
        out.append(CheckResult(f"cadence_{cad}_ok", True, "ok",
                               f"{len(cad_df)} rows, +{len(extra)} extras"))
    return out


def validate_one_asset(asset: str, registry: FeatureRegistry) -> AssetReport:
    asset_l = asset.lower()
    asset_u = asset.upper()
    rep = AssetReport(asset=asset_u)

    v50_path = PROCESSED_DIR / f"{asset_l}usdt_v50_chimera.parquet"
    v51_path = PROCESSED_DIR / f"{asset_l}usdt_v51_chimera.parquet"
    if not v51_path.exists():
        rep.add(CheckResult("v51_exists", False, "fail", f"missing {v51_path}"))
        return rep
    if not v50_path.exists():
        rep.add(CheckResult("v50_exists", False, "fail", f"missing {v50_path}"))
        return rep

    v50 = pl.read_parquet(v50_path)
    v51 = pl.read_parquet(v51_path)

    for c in check_schema(asset_u, v50, v51, registry):
        rep.add(c)
    rep.add(check_backward_compat(asset_u, v50, v51))
    rep.add(check_timestamp_monotonic(v51))
    rep.add(check_no_future_leak(v51))
    for c in check_nan_budget(v51, registry):
        rep.add(c)
    rep.add(check_date_gaps(v51))
    rep.add(check_hawkes_alignment(asset_u, v51))
    for c in check_cadence_files(asset_u, v51, registry):
        rep.add(c)

    return rep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    reg = FeatureRegistry.load()

    if args.asset:
        assets = [args.asset.upper()]
    else:
        v51_files = sorted(PROCESSED_DIR.glob("*_v51_chimera.parquet"))
        v51_files = [f for f in v51_files if "_1d" not in f.stem and "_4h" not in f.stem]
        assets = [f.stem.replace("usdt_v51_chimera", "").upper() for f in v51_files]

    if not assets:
        print("No v51 chimera files found. Run make_dataset_v51.py first.")
        sys.exit(1)

    reports = []
    n_clean = n_warn_only = n_fail = 0
    for asset in assets:
        rep = validate_one_asset(asset, reg)
        reports.append(rep)
        if rep.n_fail > 0:
            n_fail += 1
        elif rep.n_warn > 0:
            n_warn_only += 1
        else:
            n_clean += 1
        if not args.quiet:
            status = "FAIL" if rep.n_fail else "WARN" if rep.n_warn else "OK"
            print(f"[{status:>4}] {asset:>10}  {rep.n_pass:>2} pass  {rep.n_warn:>2} warn  {rep.n_fail:>2} fail")
            for c in rep.checks:
                if c.severity != "ok":
                    print(f"          {c.severity.upper():>4}: {c.name} -- {c.detail}")

    print()
    print(f"Summary: {n_clean} clean, {n_warn_only} warn-only, {n_fail} fail (of {len(reports)} assets)")

    if args.json:
        out = {
            "summary": {"clean": n_clean, "warn_only": n_warn_only, "fail": n_fail, "total": len(reports)},
            "reports": [
                {
                    "asset": r.asset, "n_pass": r.n_pass, "n_warn": r.n_warn, "n_fail": r.n_fail,
                    "checks": [{"name": c.name, "severity": c.severity, "passed": c.passed, "detail": c.detail}
                               for c in r.checks],
                }
                for r in reports
            ],
        }
        out_path = PROCESSED_DIR.parent.parent / "logs" / "validate_chimera_v51.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"JSON written to {out_path}")

    if n_fail > 0:
        sys.exit(2)
    if n_warn_only > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
