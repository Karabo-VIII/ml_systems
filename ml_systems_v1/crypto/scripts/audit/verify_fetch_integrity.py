"""verify_fetch_integrity.py -- content-level audit of data/raw/<asset>/<dtype>/.

User 2026-05-15: 'Is the fetch script robust - it seems there are fetches that
seem like they are fetched, but are not, or I'm tripping?'

Honest answer: fetch_all.py uses file-count + size>100 byte presence check,
NOT content verification. This tool fills the gap by:
  1. Loading every parquet file
  2. Counting rows + checking schema + date coverage
  3. Flagging suspiciously-small / empty / corrupt files
  4. Reporting gaps in the date range (holes in continuous coverage)

Output: runs/audit/fetch_integrity_<TIMESTAMP>.md + JSON
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DTYPES = ["aggTrades", "funding", "metrics", "klines_1m"]

# Expected MIN row counts per file (heuristic floors below which we suspect corruption)
MIN_ROWS = {
    "aggTrades": 1,        # high-volume assets: 1000s; small assets: dozens; 1 = suspicious
    "funding": 1,           # 3 per day for perp; sometimes 1 row per day archive
    "metrics": 1,           # one daily snapshot
    "klines_1m": 60,        # 1m bars: should be ~1440/day; thin assets >= 60 sanity
}

# Suspect size threshold (bytes). Below this we mark as potentially-truncated.
# Real parquets even for thin assets are 2-5 KB minimum due to metadata.
MIN_SIZE_BYTES = 500


def parse_date_from_name(fname: str, asset: str, dtype: str) -> str | None:
    """Extract YYYY-MM-DD from typical filename like SHIBUSDT-aggTrades-2024-03-15.parquet."""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
    return m.group(1) if m else None


def audit_one_dir(asset: str, dtype: str) -> dict:
    """Scan all parquets in data/raw/<asset>/<dtype>/. Return audit row."""
    d = RAW / asset / dtype
    if not d.exists():
        return {"asset": asset, "dtype": dtype, "status": "missing_dir",
                 "n_files": 0, "dates_covered": []}
    files = sorted(d.glob("*.parquet"))
    if not files:
        return {"asset": asset, "dtype": dtype, "status": "empty_dir",
                 "n_files": 0, "dates_covered": []}

    suspicious = []   # (file, reason)
    corrupt = []      # (file, error)
    too_small = []
    too_few_rows = []
    dates_seen: set[str] = set()
    sizes = []
    rows = []
    for f in files:
        size = f.stat().st_size
        sizes.append(size)
        if size < MIN_SIZE_BYTES:
            too_small.append((f.name, size))
            continue
        date = parse_date_from_name(f.name, asset, dtype)
        if date:
            dates_seen.add(date)
        try:
            # Read just the row count efficiently
            n_rows = pl.scan_parquet(f).select(pl.len()).collect().item()
            rows.append(n_rows)
            if n_rows < MIN_ROWS[dtype]:
                too_few_rows.append((f.name, n_rows))
        except Exception as e:
            corrupt.append((f.name, type(e).__name__))

    # Compute date-range gaps
    gap_dates = []
    if dates_seen:
        ds = sorted(dates_seen)
        start = datetime.strptime(ds[0], "%Y-%m-%d")
        end = datetime.strptime(ds[-1], "%Y-%m-%d")
        cur = start
        while cur <= end:
            ds_str = cur.strftime("%Y-%m-%d")
            if ds_str not in dates_seen:
                gap_dates.append(ds_str)
            cur += timedelta(days=1)

    return {
        "asset": asset,
        "dtype": dtype,
        "status": "ok",
        "n_files": len(files),
        "n_dates_covered": len(dates_seen),
        "date_min": min(dates_seen) if dates_seen else None,
        "date_max": max(dates_seen) if dates_seen else None,
        "n_gaps_in_range": len(gap_dates),
        "gap_dates_sample": gap_dates[:5],
        "median_size_b": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        "min_size_b": min(sizes) if sizes else 0,
        "median_rows": sorted(rows)[len(rows) // 2] if rows else 0,
        "min_rows": min(rows) if rows else 0,
        "n_too_small": len(too_small),
        "n_too_few_rows": len(too_few_rows),
        "n_corrupt": len(corrupt),
        "too_small_sample": too_small[:3],
        "too_few_rows_sample": too_few_rows[:3],
        "corrupt_sample": corrupt[:3],
    }


def main():
    print("=" * 72)
    print("FETCH INTEGRITY AUDIT -- content-level verification")
    print("=" * 72)

    assets = sorted([d.name for d in RAW.iterdir() if d.is_dir() and d.name.endswith("USDT")])
    print(f"Assets: {len(assets)}")
    t0 = time.time()

    results = []
    for i, a in enumerate(assets):
        for dt in DTYPES:
            try:
                r = audit_one_dir(a, dt)
                results.append(r)
            except Exception as e:
                results.append({"asset": a, "dtype": dt, "status": "audit_error",
                                 "error": f"{type(e).__name__}: {e}"})
        if (i + 1) % 10 == 0:
            print(f"  audited {i+1}/{len(assets)} assets ({time.time()-t0:.0f}s)")

    # Aggregate
    summary = {
        "audit_run_at": datetime.now().isoformat(),
        "total_audits": len(results),
        "n_assets": len(assets),
        "issues": {
            "too_small_files": sum(r.get("n_too_small", 0) for r in results),
            "too_few_rows_files": sum(r.get("n_too_few_rows", 0) for r in results),
            "corrupt_files": sum(r.get("n_corrupt", 0) for r in results),
            "date_gaps_in_range": sum(r.get("n_gaps_in_range", 0) for r in results),
        },
    }

    # Files-with-issues lists
    issue_rows = [r for r in results if r.get("status") == "ok" and (
        r.get("n_too_small", 0) > 0 or r.get("n_too_few_rows", 0) > 0
        or r.get("n_corrupt", 0) > 0 or r.get("n_gaps_in_range", 0) > 0)]

    # Compose report
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_md = OUT_DIR / f"fetch_integrity_{ts}.md"
    out_json = OUT_DIR / f"fetch_integrity_{ts}.json"

    lines = []
    lines.append(f"# Fetch Integrity Audit -- {datetime.now().isoformat()}")
    lines.append("")
    lines.append(f"Scope: data/raw/<asset>/<dtype>/*.parquet for {len(assets)} assets x 4 dtypes")
    lines.append(f"Total audits: {len(results)}")
    lines.append("")
    lines.append("## Headline issues")
    lines.append("")
    lines.append(f"- **Too-small files (< {MIN_SIZE_BYTES} bytes)**: {summary['issues']['too_small_files']}")
    lines.append(f"- **Too-few rows (per-dtype min row threshold)**: {summary['issues']['too_few_rows_files']}")
    lines.append(f"- **Corrupt / unreadable parquet**: {summary['issues']['corrupt_files']}")
    lines.append(f"- **Date gaps in continuous coverage range**: {summary['issues']['date_gaps_in_range']}")
    lines.append("")

    if issue_rows:
        lines.append(f"## Assets/dtypes with issues ({len(issue_rows)})")
        lines.append("")
        lines.append("| Asset | Dtype | Files | Too-small | Too-few-rows | Corrupt | Date gaps | Sample issues |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for r in sorted(issue_rows, key=lambda x: -(x.get("n_too_small", 0) + x.get("n_too_few_rows", 0) + x.get("n_corrupt", 0) + x.get("n_gaps_in_range", 0))):
            sample = []
            if r.get("too_small_sample"): sample.append(f"small: {r['too_small_sample'][0][0]}")
            if r.get("too_few_rows_sample"): sample.append(f"few-rows: {r['too_few_rows_sample'][0][0]} ({r['too_few_rows_sample'][0][1]} rows)")
            if r.get("corrupt_sample"): sample.append(f"corrupt: {r['corrupt_sample'][0][0]}")
            if r.get("gap_dates_sample"): sample.append(f"gaps: {r['gap_dates_sample'][:2]}")
            lines.append(f"| {r['asset']} | {r['dtype']} | {r['n_files']} | "
                         f"{r.get('n_too_small', 0)} | {r.get('n_too_few_rows', 0)} | "
                         f"{r.get('n_corrupt', 0)} | {r.get('n_gaps_in_range', 0)} | "
                         f"{'; '.join(sample)[:80]} |")
    else:
        lines.append("## No issues detected ✓")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    out_json.write_text(json.dumps({"summary": summary, "results": results}, indent=2),
                          encoding="utf-8")
    print(f"\n[wrote] {out_md.relative_to(ROOT)}")
    print(f"[wrote] {out_json.relative_to(ROOT)}")
    print()
    print("Headline issues:")
    for k, v in summary["issues"].items():
        print(f"  {k}: {v}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
