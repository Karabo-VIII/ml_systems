"""bidirectional_readiness_crawler.py -- validate --reverse safety per producer.

Phase 7 audit. For each per-asset/per-(asset,date) producer in the DAG:
  1. Check if it accepts --reverse / -r
  2. Verify it has the skip-if-exists guard (so two terminals don't double-write)
  3. Verify it has atomic-write (so half-files don't trick the skip gate)
  4. Verify it uses iter_assets() / iter_dates() helpers OR equivalent reverse logic

Stages with `single_file` output_kind that AGGREGATE multiple per-asset inputs
are NOT candidates (e.g. basis_features_long, lob_proxy_daily) - they'd race
write the same file. The crawler skips those.

OUTPUT
------
runs/audit/bidirectional_readiness_<DATE>.md -- per-producer scorecard
"""
from __future__ import annotations

__contract__ = {
    "kind": "bidirectional_readiness_crawler",
    "owner": "audit/pipeline",
    "outputs": ["runs/audit/bidirectional_readiness_<DATE>.md"],
    "invariants": [
        "per-asset producers should accept --reverse for meet-in-middle",
        "atomic-write contract MUST be in place before --reverse is safe",
        "single-file aggregation producers are NOT candidates",
    ],
}

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAG_PATH = PROJECT_ROOT / "config" / "asset_dag.yaml"
OUT_DIR = PROJECT_ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dag() -> dict[str, dict]:
    with open(DAG_PATH) as f:
        d = yaml.safe_load(f)
    return d.get("assets") or {}


def is_candidate(body: dict) -> bool:
    """A stage is a bidirectional candidate if it produces per-asset or
    per-(asset, date) files (not aggregated single blobs)."""
    output_kind = body.get("output_kind", "")
    return output_kind in ("per_asset_files", "per_asset_per_date_files")


def audit_producer(producer_path: Path) -> dict[str, Any]:
    """Return per-producer scorecard."""
    if not producer_path.exists():
        return {"missing": True}
    try:
        text = producer_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"read_error": True}
    return {
        "accepts_reverse": "--reverse" in text or '"-r"' in text or "'-r'" in text,
        "uses_framework": "from pipeline.bidirectional" in text or
                            "iter_assets" in text or "iter_dates" in text,
        "has_skip_exists": "exists()" in text and (".unlink" not in text or
                              "should_skip" in text),
        "has_atomic_write": "atomic_write_parquet" in text or
                              (".rename(" in text and ("tmp" in text or "_tmp" in text)),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                     help="Require ALL 4 axes; default = 2 of 4 OK")
    args = ap.parse_args()

    dag = load_dag()
    candidates = {n: b for n, b in dag.items() if is_candidate(b)}
    non_candidates = {n: b for n, b in dag.items() if not is_candidate(b)
                         and b.get("output_kind") not in (None, "ephemeral")}
    print(f"[bidi-crawler] {len(candidates)} candidates, "
          f"{len(non_candidates)} non-candidates (single-blob aggregators)")

    scorecards: dict[str, dict] = {}
    for stage_name, body in candidates.items():
        producer = body.get("producer")
        if not producer:
            continue
        sc = audit_producer(PROJECT_ROOT / producer)
        sc["producer"] = producer
        scorecards[stage_name] = sc

    today = dt.date.today().isoformat()
    out = OUT_DIR / f"bidirectional_readiness_{today}.md"
    n_ready = 0
    n_partial = 0
    n_unready = 0
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"# Bidirectional Readiness Crawler -- {today}\n\n")
        fh.write(f"Candidates: {len(candidates)} (per-asset / per-asset-per-date)\n")
        fh.write(f"Non-candidates: {len(non_candidates)} (single-blob aggregators; "
                  f"NOT eligible for bidirectional)\n\n")
        fh.write(f"## Per-producer scorecard (4 axes)\n\n")
        fh.write(f"| stage | accepts-r | framework-used | skip-exists | atomic-write | verdict |\n")
        fh.write(f"|---|---|---|---|---|---|\n")
        for stage_name, sc in sorted(scorecards.items()):
            if sc.get("missing") or sc.get("read_error"):
                fh.write(f"| {stage_name} | - | - | - | - | UNREADABLE |\n")
                continue
            n_yes = sum(1 for k in ("accepts_reverse", "uses_framework",
                                       "has_skip_exists", "has_atomic_write")
                          if sc.get(k))
            if n_yes >= 4:
                verdict = "READY"; n_ready += 1
            elif n_yes >= 2:
                verdict = "PARTIAL"; n_partial += 1
            else:
                verdict = "UNREADY"; n_unready += 1
            cells = ["Y" if sc.get(k) else "."
                       for k in ("accepts_reverse", "uses_framework",
                                  "has_skip_exists", "has_atomic_write")]
            fh.write(f"| {stage_name} | {cells[0]} | {cells[1]} | "
                      f"{cells[2]} | {cells[3]} | {verdict} |\n")
        fh.write(f"\n## Summary\n\n")
        fh.write(f"- READY (4/4 axes): {n_ready}\n")
        fh.write(f"- PARTIAL (2-3/4): {n_partial}\n")
        fh.write(f"- UNREADY (0-1/4): {n_unready}\n\n")
        fh.write(f"## Non-candidate stages (single-blob aggregators)\n\n")
        fh.write(f"These produce ONE output file from multiple per-asset inputs; "
                  f"two terminals writing the same file = race. NOT eligible:\n")
        for n in sorted(non_candidates):
            fh.write(f"- {n} (output_kind={non_candidates[n].get('output_kind')})\n")
    print(f"[bidi-crawler] {len(scorecards)} producers audited -> {out}")
    print(f"  READY:    {n_ready}")
    print(f"  PARTIAL:  {n_partial}")
    print(f"  UNREADY:  {n_unready}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
