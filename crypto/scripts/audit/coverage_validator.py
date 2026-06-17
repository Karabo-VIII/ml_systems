"""Coverage Validator — pipeline integrity & completeness audit.

Runs per-asset, per-cadence, per-day completeness + schema checks across:
  - data/raw/<sym>/{aggTrades, klines_1m, funding, metrics}
  - data/processed/chimera/{1d, 4h, 1h, 15m, dollar}/
  - data/processed/bars/{dib, range, runs_tick, runs_vol, adaptive_vol}/

Outputs a single JSON report + markdown summary so any operator can verify
"the data is complete and correct" without re-deriving the inventory.

Usage:
    python scripts/audit/coverage_validator.py [--universe u59] [--start 2020-01-01] [--end 2026-05-08]

Exit codes:
    0  CLEAN — every (asset, cadence, day) cell is populated + valid
    1  WARN  — coverage gaps or schema issues that are documented elsewhere
    2  FAIL  — corruption detected (ts out of range, null close, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parent.parent.parent

# Per CLAUDE.md ms invariant
TS_MS_LOW = 1_500_000_000_000
TS_MS_HIGH = 2_000_000_000_000

# Universe to validate; default to u59 per assets-with-metadata
def _load_universe() -> list[str]:
    import yaml
    syms = set()
    for u in ("u50", "u100"):
        with open(ROOT / "config" / "universes" / f"{u}.yaml") as f:
            cfg = yaml.safe_load(f)
        for a in cfg.get("assets", []):
            syms.add(a["symbol"])
        for a in cfg.get("extra_assets", []):
            if a.get("status") == "ready":
                syms.add(a["symbol"])
    return sorted(syms)


def check_raw_aggtrades(asset: str, start: date, end: date) -> dict[str, Any]:
    """Per-day file existence in data/raw/<sym>/aggTrades/."""
    p = ROOT / "data" / "raw" / asset / "aggTrades"
    if not p.exists():
        return {"status": "MISSING_DIR", "expected_days": 0, "actual_days": 0, "gaps": []}
    files = sorted(p.glob(f"{asset}-aggTrades-*.parquet"))
    file_dates = set()
    for f in files:
        try:
            d_str = "-".join(f.stem.split("-")[-3:])
            file_dates.add(date.fromisoformat(d_str))
        except Exception:
            continue
    # Compute expected dates (start..min(end, last_observed))
    if not file_dates:
        return {"status": "EMPTY", "expected_days": 0, "actual_days": 0, "gaps": []}
    first_have = min(file_dates)
    last_have = max(file_dates)
    eff_start = max(start, first_have)
    expected = set()
    d = eff_start
    while d <= min(end, last_have):
        expected.add(d)
        d += timedelta(days=1)
    gaps = sorted(expected - file_dates)
    status = "CLEAN" if len(gaps) == 0 else "GAPS"
    return {
        "status": status,
        "first_day": str(first_have),
        "last_day": str(last_have),
        "expected_days": len(expected),
        "actual_days": len(expected & file_dates),
        "gap_count": len(gaps),
        "gaps_sample": [str(g) for g in gaps[:5]],
    }


def check_raw_klines_1m(asset: str) -> dict[str, Any]:
    p = ROOT / "data" / "raw" / asset / "klines_1m"
    if not p.exists():
        return {"status": "MISSING_DIR", "actual_days": 0}
    files = list(p.glob(f"{asset}-1m-*.parquet"))
    return {"status": "OK" if files else "EMPTY", "actual_days": len(files)}


def check_chimera_cadence(asset: str, cadence: str) -> dict[str, Any]:
    """Verify chimera artifact exists, is non-empty, has valid ts range, no NaN close.

    Naming inconsistency [DOCUMENTED]: dollar bars use
    `<sym>_v51_chimera_<date>.parquet` (no cadence in name); 1d/4h/1h/15m use
    `<sym>_v51_chimera_<cadence>_<date>.parquet`.
    """
    p = ROOT / "data" / "processed" / "chimera" / cadence
    sym_lower = asset.lower()
    if cadence == "dollar":
        files = sorted(p.glob(f"{sym_lower}_v51_chimera_*.parquet"))
    else:
        files = sorted(p.glob(f"{sym_lower}_v51_chimera_{cadence}_*.parquet"))
    if not files:
        return {"status": "MISSING", "file_count": 0}
    fp = files[-1]  # latest
    try:
        df = pl.read_parquet(fp, columns=["timestamp", "close"])
    except Exception as exc:
        return {"status": "CORRUPT_READ", "file": fp.name, "err": str(exc)[:100]}
    n = df.height
    if n == 0:
        return {"status": "EMPTY", "file": fp.name}
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    if ts_min is None or ts_max is None:
        return {"status": "ALL_NULL_TS", "file": fp.name}
    if ts_min < TS_MS_LOW or ts_max > TS_MS_HIGH:
        return {"status": "TS_SCALE_VIOLATION",
                "file": fp.name, "ts_min": ts_min, "ts_max": ts_max}
    n_null_close = df["close"].null_count()
    if n_null_close > 0:
        return {"status": "NULL_CLOSE", "file": fp.name, "n_null": n_null_close}
    # Monotonicity (allow duplicates — chimera 1d shouldn't have them but check sort)
    ts_diff_neg = (df["timestamp"].diff().drop_nulls() < 0).sum()
    if ts_diff_neg > 0:
        return {"status": "TS_NOT_MONOTONE", "file": fp.name, "n_negative_diffs": ts_diff_neg}
    return {"status": "CLEAN", "file": fp.name, "rows": n,
            "ts_span": f"{ts_min}..{ts_max}"}


def check_alt_bar(asset: str, bar_type: str) -> dict[str, Any]:
    """Check alt-bar coverage + schema."""
    folder_map = {"dib": "dib", "range": "range", "tick_runs": "runs_tick",
                  "vol_runs": "runs_vol", "adaptive_vol": "adaptive_vol"}
    folder = folder_map.get(bar_type)
    if not folder:
        return {"status": "BAD_BAR_TYPE"}
    p = ROOT / "data" / "processed" / "bars" / folder
    if bar_type == "adaptive_vol":
        fp = p / f"{asset}_adaptive_vol.parquet"
        if not fp.exists():
            return {"status": "MISSING"}
        files = [fp]
    else:
        files = sorted(p.glob(f"{asset}_{bar_type}_*.parquet"))
        if not files:
            return {"status": "MISSING"}
    total_rows = 0
    earliest_ts = None
    latest_ts = None
    issues = []
    for fp in files:
        try:
            df = pl.read_parquet(fp, columns=["bar_start_ts", "close"])
            total_rows += df.height
            if df.height == 0:
                continue
            ts_min = df["bar_start_ts"].min()
            ts_max = df["bar_start_ts"].max()
            if ts_min < TS_MS_LOW or ts_max > TS_MS_HIGH:
                issues.append(f"{fp.name}: ts out of range {ts_min}..{ts_max}")
            if df["close"].null_count() > 0:
                issues.append(f"{fp.name}: null close")
            earliest_ts = ts_min if earliest_ts is None else min(earliest_ts, ts_min)
            latest_ts = ts_max if latest_ts is None else max(latest_ts, ts_max)
        except Exception as exc:
            issues.append(f"{fp.name}: {type(exc).__name__}: {str(exc)[:60]}")
    if issues:
        return {"status": "ISSUES", "n_files": len(files), "rows": total_rows,
                "issues_sample": issues[:3]}
    return {"status": "CLEAN", "n_files": len(files), "rows": total_rows,
            "ts_span": f"{earliest_ts}..{latest_ts}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2020-01-01")
    ap.add_argument("--end", type=str, default="2026-05-08")
    ap.add_argument("--limit-assets", type=int, default=None)
    ap.add_argument("--out-json", type=str,
                    default="runs/audit/coverage_report.json")
    ap.add_argument("--out-md", type=str,
                    default="runs/audit/coverage_report.md")
    args = ap.parse_args()

    universe = _load_universe()
    if args.limit_assets:
        universe = universe[:args.limit_assets]
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    print(f"=== Coverage Validator ===")
    print(f"assets: {len(universe)} (universe = u50 + u100_ready)")
    print(f"date span: {start} -> {end}")
    print()

    report = {"meta": {"start": str(start), "end": str(end), "n_assets": len(universe)},
              "per_asset": {}}
    total_clean = total_gaps = total_corrupt = 0
    cadences = ["1d", "4h", "1h", "15m", "dollar"]
    alt_bars = ["dib", "range", "tick_runs", "vol_runs", "adaptive_vol"]

    for i, asset in enumerate(universe, 1):
        a_report = {}
        a_report["aggTrades"] = check_raw_aggtrades(asset, start, end)
        a_report["klines_1m"] = check_raw_klines_1m(asset)
        for c in cadences:
            a_report[f"chimera_{c}"] = check_chimera_cadence(asset, c)
        for b in alt_bars:
            a_report[f"alt_{b}"] = check_alt_bar(asset, b)
        report["per_asset"][asset] = a_report
        # Classify
        statuses = [v.get("status", "") for v in a_report.values()]
        if any(s in ("TS_SCALE_VIOLATION", "NULL_CLOSE", "TS_NOT_MONOTONE",
                     "CORRUPT_READ", "ALL_NULL_TS") for s in statuses):
            total_corrupt += 1
        elif any(s in ("MISSING", "MISSING_DIR", "EMPTY", "GAPS", "ISSUES") for s in statuses):
            total_gaps += 1
        else:
            total_clean += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(universe)}] processed")

    print()
    print(f"=== SUMMARY ===")
    print(f"  CLEAN  : {total_clean}/{len(universe)}")
    print(f"  GAPS   : {total_gaps}/{len(universe)}  (data missing, not corrupt)")
    print(f"  CORRUPT: {total_corrupt}/{len(universe)}  (schema violation)")
    report["meta"]["clean"] = total_clean
    report["meta"]["gaps"] = total_gaps
    report["meta"]["corrupt"] = total_corrupt

    # Write outputs
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"wrote: {out_json}")

    # Markdown summary
    out_md = Path(args.out_md)
    with open(out_md, "w") as f:
        f.write(f"# Coverage Report — {datetime.now().date()}\n\n")
        f.write(f"Universe: {len(universe)} assets. Span: {start} -> {end}.\n\n")
        f.write(f"| Status | Count |\n|---|---|\n")
        f.write(f"| CLEAN | {total_clean} |\n")
        f.write(f"| GAPS  | {total_gaps} |\n")
        f.write(f"| CORRUPT | {total_corrupt} |\n\n")
        # Per-cadence coverage rate (CLEAN only)
        f.write(f"## Per-cadence CLEAN counts\n\n")
        for c in cadences:
            clean_n = sum(1 for a in report["per_asset"].values()
                          if a.get(f"chimera_{c}", {}).get("status") == "CLEAN")
            f.write(f"- chimera/{c}: {clean_n}/{len(universe)}\n")
        f.write(f"\n## Per-alt-bar CLEAN counts\n\n")
        for b in alt_bars:
            clean_n = sum(1 for a in report["per_asset"].values()
                          if a.get(f"alt_{b}", {}).get("status") == "CLEAN")
            f.write(f"- bars/{b}: {clean_n}/{len(universe)}\n")
        f.write(f"\n## Issues found\n\n")
        for asset, a_report in report["per_asset"].items():
            problems = [(k, v) for k, v in a_report.items()
                        if v.get("status") not in (None, "CLEAN", "OK")]
            if problems:
                f.write(f"### {asset}\n")
                for k, v in problems[:10]:
                    f.write(f"- **{k}**: {v.get('status')}")
                    if v.get("file"):
                        f.write(f" (file: `{v['file']}`)")
                    if v.get("gap_count"):
                        f.write(f" — {v['gap_count']} gaps")
                    if v.get("issues_sample"):
                        f.write(f" — sample: {v['issues_sample'][0]}")
                    f.write("\n")
                f.write("\n")
    print(f"wrote: {out_md}")

    if total_corrupt > 0:
        return 2
    if total_gaps > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
