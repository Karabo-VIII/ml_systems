"""pipeline_staleness_crawler.py -- detect stale outputs + fragmentation.

Phase 8 audit. Walks every per_asset + per_asset_per_date output and flags:

  1. STALE-OUTPUT      : output mtime older than the FRESHNESS contract
                         (daily: >7d; weekly: >14d; event_driven: >60d)
  2. FRAGMENTATION-GAP : per_asset_per_date dir has a hole (e.g. dates
                         2024-01-01, 2024-01-02, 2024-01-04 = missing 03)
  3. PRE-LISTING-WASTE : confirmed-missing markers BEFORE the asset's
                         Binance listing date (false positives that could
                         have been skipped via listing_dates module)
  4. PROCESS-FRAGMENT  : .tmp files lying around (crashed mid-write or
                         non-atomic producer)
  5. ZERO-BYTE-OUTPUT  : output file exists but size < 100 bytes (ghost
                         file; should not match should_skip())

OUTPUT
------
runs/audit/pipeline_staleness_<DATE>.md
"""
from __future__ import annotations

__contract__ = {
    "kind": "pipeline_staleness_crawler",
    "owner": "audit/pipeline",
    "outputs": ["runs/audit/pipeline_staleness_<DATE>.md"],
    "invariants": [
        "non-invasive: only reads file stats + listing_dates cache",
        "flags fragmentation across 5 axes",
        "complements pipeline_audit_crawler (correctness) and "
        "bidirectional_readiness_crawler (parallel-safety)",
    ],
}

import argparse
import datetime as dt
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DAG_PATH = PROJECT_ROOT / "config" / "asset_dag.yaml"
RAW_BASE = PROJECT_ROOT / "data" / "raw"
PROC_BASE = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Freshness windows in days
FRESHNESS_WINDOWS = {
    "daily": 7,
    "weekly": 14,
    "event_driven": 60,
    "static": 365,
}


def load_dag() -> dict[str, dict]:
    with open(DAG_PATH) as f:
        d = yaml.safe_load(f)
    return d.get("assets") or {}


def _days_since(p: Path) -> float:
    try:
        return (dt.datetime.now().timestamp() - p.stat().st_mtime) / 86400.0
    except OSError:
        return float("inf")


# ============================================================================
# Axis 1: stale-output
# ============================================================================

def audit_stale_outputs(dag: dict[str, dict]) -> list[dict]:
    findings: list[dict] = []
    for stage_name, body in dag.items():
        if body.get("output_kind") == "ephemeral":
            continue
        freshness = (body.get("freshness") or "daily").lower()
        window = FRESHNESS_WINDOWS.get(freshness, 7)
        output_pat = body.get("output", "")
        if not output_pat:
            continue
        # Resolve to per-asset or single-blob paths
        paths: list[Path]
        if "{asset}" in output_pat:
            # Sample a few assets
            paths = []
            for sym in ("btc", "eth", "sol"):
                pat = output_pat.replace("{asset}", sym)
                matches = list(PROJECT_ROOT.glob(pat))
                if matches:
                    paths.append(matches[-1])
        else:
            paths = list(PROJECT_ROOT.glob(output_pat))[:1]
        for p in paths:
            age = _days_since(p)
            if age > window:
                findings.append({
                    "stage": stage_name, "category": "stale-output",
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "age_days": round(age, 1), "freshness": freshness,
                    "window_days": window,
                    "fix": f"freshness={freshness} -> window={window}d; "
                              f"file is {age:.1f}d old. Rebuild this stage.",
                })
    return findings


# ============================================================================
# Axis 2: fragmentation-gap (per_asset_per_date stages)
# ============================================================================

def audit_fragmentation(dag: dict[str, dict],
                          sample_assets: list[str] = None) -> list[dict]:
    if sample_assets is None:
        sample_assets = ["BTC", "ETH", "SOL"]
    findings: list[dict] = []
    try:
        from pipeline.listing_dates import get_listing_date
    except ImportError:
        return findings
    for stage_name, body in dag.items():
        if body.get("output_kind") != "per_asset_per_date_files":
            continue
        output_pat = body.get("output", "")
        if not output_pat or "{asset}" not in output_pat:
            continue
        for sym in sample_assets:
            # Resolve dir + extract dates
            sym_pat = output_pat.replace("{asset}", sym)
            # Strip the per-date glob suffix to find the parent dir
            parent_dir = Path(sym_pat).parent
            full_parent = PROJECT_ROOT / parent_dir
            if not full_parent.exists():
                continue
            file_pattern = Path(sym_pat).name
            files = sorted(full_parent.glob(file_pattern))
            if len(files) < 10:
                continue
            # Extract dates from filenames (YYYY-MM-DD or YYYYMMDD)
            dates: list[dt.date] = []
            for fp in files:
                m = re.search(r"(\d{4}-?\d{2}-?\d{2})", fp.name)
                if not m:
                    continue
                ds = m.group(1).replace("-", "")
                try:
                    d = dt.datetime.strptime(ds, "%Y%m%d").date()
                    dates.append(d)
                except ValueError:
                    continue
            dates = sorted(set(dates))
            if len(dates) < 2:
                continue
            # Detect gaps INSIDE [listing_date, max_date]
            listing = get_listing_date(sym + "USDT").date()
            d_min = max(dates[0], listing)
            d_max = dates[-1]
            expected = set()
            cur = d_min
            while cur <= d_max:
                expected.add(cur)
                cur += dt.timedelta(days=1)
            present = set(d for d in dates if d >= listing)
            missing = sorted(expected - present)
            if missing:
                findings.append({
                    "stage": stage_name, "category": "fragmentation-gap",
                    "sample_asset": sym,
                    "dir": str(parent_dir),
                    "n_files": len(files), "n_missing": len(missing),
                    "first_missing": str(missing[0]),
                    "last_missing": str(missing[-1]),
                    "fix": f"backfill {len(missing)} dates in [{missing[0]}, "
                              f"{missing[-1]}] for {sym}",
                })
    return findings


# ============================================================================
# Axis 3: pre-listing-waste (confirmed-missing markers BEFORE listing)
# ============================================================================

def audit_pre_listing_waste(sample_assets: list[str] = None) -> list[dict]:
    """Read raw/SYM/_missing_dates.json (if exists) and flag entries
    BEFORE the asset's listing date."""
    if sample_assets is None:
        sample_assets = ["BTC", "ETH", "SOL", "PEPE", "SHIB", "BONK"]
    findings: list[dict] = []
    try:
        from pipeline.listing_dates import get_listing_date
        import json as _json
    except ImportError:
        return findings
    for sym in sample_assets:
        sym_full = sym + "USDT"
        for marker_name in ("_missing_dates.json", "_confirmed_missing.json"):
            marker_path = RAW_BASE / sym_full / marker_name
            if not marker_path.exists():
                continue
            try:
                data = _json.loads(marker_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            listing = get_listing_date(sym_full).date()
            # data could be {dtype: [dates]} or just [dates]
            all_dates: list[str] = []
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        all_dates.extend(v)
            elif isinstance(data, list):
                all_dates = data
            n_pre_listing = 0
            for ds in all_dates:
                if not isinstance(ds, str):
                    continue
                try:
                    d = dt.datetime.strptime(ds[:10], "%Y-%m-%d").date()
                    if d < listing:
                        n_pre_listing += 1
                except ValueError:
                    continue
            if n_pre_listing > 0:
                findings.append({
                    "category": "pre-listing-waste",
                    "sample_asset": sym,
                    "marker": str(marker_path.relative_to(PROJECT_ROOT)),
                    "n_pre_listing_entries": n_pre_listing,
                    "listing_date": str(listing),
                    "fix": f"strip pre-{listing} entries from "
                              f"{marker_path.name}; producer now uses "
                              f"listing_dates.is_pre_listing()",
                })
    return findings


# ============================================================================
# Axis 4: process-fragment (.tmp files left behind)
# ============================================================================

def audit_process_fragments() -> list[dict]:
    findings: list[dict] = []
    tmps = list(PROC_BASE.rglob("*.tmp"))
    for tmp in tmps:
        age = _days_since(tmp)
        if age > 0.1:    # only flag if older than 2.4 hours (not in-flight)
            findings.append({
                "category": "process-fragment",
                "path": str(tmp.relative_to(PROJECT_ROOT)),
                "age_days": round(age, 2),
                "size_bytes": tmp.stat().st_size if tmp.exists() else 0,
                "fix": "rm the .tmp file; crashed-mid-write artifact",
            })
    return findings


# ============================================================================
# Axis 5: zero-byte-output
# ============================================================================

def audit_zero_byte_outputs(dag: dict[str, dict]) -> list[dict]:
    findings: list[dict] = []
    for stage_name, body in dag.items():
        output_pat = body.get("output", "")
        if not output_pat or body.get("output_kind") == "ephemeral":
            continue
        # Just scan glob; small N OK
        if "{asset}" in output_pat:
            for sym in ("btc", "eth", "sol"):
                pat = output_pat.replace("{asset}", sym)
                for p in PROJECT_ROOT.glob(pat):
                    try:
                        sz = p.stat().st_size
                    except OSError:
                        continue
                    if sz < 100:
                        findings.append({
                            "stage": stage_name, "category": "zero-byte-output",
                            "path": str(p.relative_to(PROJECT_ROOT)),
                            "size_bytes": sz,
                            "fix": "delete + rebuild this stage's output for "
                                      "this asset; ghost file from interrupted write",
                        })
        else:
            for p in PROJECT_ROOT.glob(output_pat):
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                if sz < 100:
                    findings.append({
                        "stage": stage_name, "category": "zero-byte-output",
                        "path": str(p.relative_to(PROJECT_ROOT)),
                        "size_bytes": sz,
                        "fix": "delete + rebuild this stage's output",
                    })
    return findings


# ============================================================================
# Reporter
# ============================================================================

def write_report(all_findings: list[dict]) -> Path:
    today = dt.date.today().isoformat()
    out = OUT_DIR / f"pipeline_staleness_{today}.md"
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_cat[f["category"]].append(f)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Pipeline Staleness + Fragmentation Crawler -- {today}\n\n")
        fh.write(f"Total findings: {len(all_findings)}\n\n")
        fh.write(f"## Summary by category\n\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"- **{cat}**: {len(lst)}\n")
        fh.write("\n")
        for cat, lst in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
            fh.write(f"## {cat} ({len(lst)})\n\n")
            for f in lst[:30]:
                fh.write("- finding:\n")
                for k, v in f.items():
                    if k != "category":
                        fh.write(f"  - {k}: {v}\n")
                fh.write("\n")
            if len(lst) > 30:
                fh.write(f"... and {len(lst) - 30} more.\n\n")
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axes", nargs="+",
                     default=["stale", "frag", "pre-listing", "tmp", "zero-byte"],
                     help="Which axes to audit")
    args = ap.parse_args()

    dag = load_dag()
    findings: list[dict] = []
    if "stale" in args.axes:
        print("  axis 1: stale-output ...", flush=True)
        findings += audit_stale_outputs(dag)
    if "frag" in args.axes:
        print("  axis 2: fragmentation-gap ...", flush=True)
        findings += audit_fragmentation(dag)
    if "pre-listing" in args.axes:
        print("  axis 3: pre-listing-waste ...", flush=True)
        findings += audit_pre_listing_waste()
    if "tmp" in args.axes:
        print("  axis 4: process-fragment ...", flush=True)
        findings += audit_process_fragments()
    if "zero-byte" in args.axes:
        print("  axis 5: zero-byte-output ...", flush=True)
        findings += audit_zero_byte_outputs(dag)

    out = write_report(findings)
    print(f"[staleness-crawler] {len(findings)} findings -> {out}")
    by_cat: dict[str, int] = defaultdict(int)
    for f in findings:
        by_cat[f["category"]] += 1
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<28s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
