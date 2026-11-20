"""listing_date_consumer_crawler.py -- audit pre-listing-skip adoption.

Phase 8 audit. Walks every DAG producer; for date-iterating stages,
verifies they consume pipeline.listing_dates (pre-listing skip) to avoid:
  - "confirmed-missing" markers for pre-listing dates
  - wasted fetch attempts (404s on Vision archive)
  - polluted audit findings

CRITERIA
========
A producer is a CANDIDATE for pre-listing skip if it:
  - iterates a date range OR (sym, date) pairs
  - per-asset OR per-asset-per-date output_kind
  - has CLI args that suggest date iteration (--days, --start, --end)

OUTPUT
------
runs/audit/listing_date_consumer_<DATE>.md -- per-producer scorecard
"""
from __future__ import annotations

__contract__ = {
    "kind": "listing_date_consumer_crawler",
    "owner": "audit/pipeline",
    "outputs": ["runs/audit/listing_date_consumer_<DATE>.md"],
    "invariants": [
        "complements pipeline.listing_dates centralization",
        "flags producers iterating dates without pre-listing skip",
    ],
}

import argparse
import datetime as dt
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = PROJECT_ROOT / "config" / "asset_dag.yaml"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dag() -> dict[str, dict]:
    with open(DAG_PATH) as f:
        d = yaml.safe_load(f)
    return d.get("assets") or {}


def audit_producer(producer_path: Path) -> dict:
    if not producer_path.exists():
        return {"missing": True}
    try:
        text = producer_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"read_error": True}
    # Date-iteration markers
    iter_markers = [
        "for date in ", "for d in ", "for day in ", "for dt_ in ",
        "for date_range", "for dt in ", "timedelta(days=", "date_range(",
    ]
    iterates_dates = any(m in text for m in iter_markers)
    # Pre-listing skip markers (centralized helper OR equivalent local check)
    consumes_listing = (
        "from pipeline.listing_dates" in text or
        "listing_dates" in text or
        "is_pre_listing" in text or
        "_resolve_launch_date" in text or
        "filter_pre_listing" in text or
        "launch_date" in text
    )
    return {
        "iterates_dates": iterates_dates,
        "consumes_listing": consumes_listing,
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    args = ap.parse_args()

    dag = load_dag()
    scorecards: dict[str, dict] = {}
    for stage_name, body in dag.items():
        producer = body.get("producer")
        if not producer or body.get("output_kind") == "ephemeral":
            continue
        sc = audit_producer(PROJECT_ROOT / producer)
        sc["producer"] = producer
        sc["per_asset"] = body.get("per_asset", False)
        sc["output_kind"] = body.get("output_kind", "?")
        scorecards[stage_name] = sc

    today = dt.date.today().isoformat()
    out = OUT_DIR / f"listing_date_consumer_{today}.md"
    n_needs = 0
    n_ok = 0
    n_na = 0
    findings = []
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Listing-Date Consumer Crawler -- {today}\n\n")
        fh.write(f"Total producers audited: {len(scorecards)}\n\n")
        fh.write(f"## Per-producer scorecard\n\n")
        fh.write(f"| stage | iterates-dates | consumes-listing | verdict |\n")
        fh.write(f"|---|---|---|---|\n")
        for stage_name, sc in sorted(scorecards.items()):
            if sc.get("missing") or sc.get("read_error"):
                fh.write(f"| {stage_name} | - | - | UNREADABLE |\n")
                continue
            if not sc["iterates_dates"]:
                fh.write(f"| {stage_name} | . | . | N/A |\n")
                n_na += 1
                continue
            if sc["consumes_listing"]:
                fh.write(f"| {stage_name} | Y | Y | OK |\n")
                n_ok += 1
            else:
                fh.write(f"| {stage_name} | Y | . | NEEDS-RETROFIT |\n")
                n_needs += 1
                findings.append({
                    "stage": stage_name, "producer": sc["producer"],
                    "fix": "from pipeline.listing_dates import is_pre_listing; "
                              "skip pre-listing dates in iteration loops",
                })
        fh.write(f"\n## Summary\n\n")
        fh.write(f"- N/A (no date iteration): {n_na}\n")
        fh.write(f"- OK (date-iterating + consumes listing_dates): {n_ok}\n")
        fh.write(f"- NEEDS-RETROFIT (date-iterating WITHOUT listing_dates): {n_needs}\n\n")
        if findings:
            fh.write(f"## Retrofit queue\n\n")
            for f in findings:
                fh.write(f"- **{f['stage']}** ({f['producer']})\n")
                fh.write(f"  - {f['fix']}\n\n")
    print(f"[listing-consumer-crawler] {len(scorecards)} producers audited -> {out}")
    print(f"  OK:              {n_ok}")
    print(f"  NEEDS-RETROFIT:  {n_needs}")
    print(f"  N/A:             {n_na}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
