#!/usr/bin/env python3
"""campaign_monitor -- one-view status + stall-detector for an autonomous campaign (SOTA harness: save-state + intervene-midflight).

Reads a campaign LEDGER + scans its dir for on-disk artifacts, surfacing freshness and STALL signals
(zero-byte or stale outputs == the workflow-stalled failure mode we hit before, where workers wrote
partials to disk but the orchestration returned empty). A resuming instance runs this FIRST to recover
the thread without re-reading a whole transcript. No emoji (Windows cp1252). No heavy deps.

Usage:
  python crypto/scripts/autonomy/campaign_monitor.py [CAMPAIGN_DIR] [--stale-min N]
  (default CAMPAIGN_DIR = crypto/runs/engines/V1.1_finetf_exits ; --stale-min default 25)
"""
import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "runs" / "engines" / "V1.1_finetf_exits"


def _age_str(secs):
    m = secs / 60.0
    if m < 60:
        return f"{m:5.1f}m"
    return f"{m/60:5.1f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign_dir", nargs="?", default=str(DEFAULT_DIR))
    ap.add_argument("--stale-min", type=float, default=25.0, help="flag artifacts older than this (min) as possibly-stalled")
    args = ap.parse_args()

    root = Path(args.campaign_dir)
    now = time.time()
    print(f"=== CAMPAIGN MONITOR: {root} ===")
    print(f"wall-clock now (epoch): {now:.0f}  |  stale threshold: {args.stale_min:.0f}m")
    if not root.exists():
        print("  (campaign dir does not exist yet -- nothing launched, or wrong path)")
        return

    # 1) ledger head (wall-clock + decision log tail)
    ledger = root / "CAMPAIGN_LEDGER.md"
    if ledger.exists():
        lines = ledger.read_text(encoding="utf-8", errors="replace").splitlines()
        log_i = next((i for i, l in enumerate(lines) if l.startswith("## DECISION")), None)
        print("\n-- LEDGER decision log (tail) --")
        for l in (lines[log_i + 1:] if log_i is not None else [])[-8:]:
            if l.strip():
                print("  " + l)
    else:
        print("\n  (no CAMPAIGN_LEDGER.md -- state-saving not initialized)")

    # 2) artifact scan: freshness + STALL flags
    arts = sorted([p for p in root.rglob("*") if p.is_file() and p.name != "CAMPAIGN_LEDGER.md"],
                  key=lambda p: p.stat().st_mtime)
    print(f"\n-- ARTIFACTS ({len(arts)}) --")
    stalls = []
    for p in arts:
        st = p.stat()
        age = now - st.st_mtime
        flag = ""
        if st.st_size == 0:
            flag = "  <== ZERO-BYTE (STALL/abort signal)"
            stalls.append(p)
        elif age > args.stale_min * 60:
            flag = "  <== STALE (no write in %s)" % _age_str(age)
        print(f"  {age and _age_str(age)}  {st.st_size:>9d}B  {p.relative_to(root)}{flag}")

    print("\n-- VERDICT --")
    if not arts:
        print("  No artifacts yet -- workflow just launched or has not checkpointed. Re-check shortly.")
    elif stalls:
        print(f"  {len(stalls)} ZERO-BYTE artifact(s) -- likely a stalled stage. Salvage the non-empty partials,")
        print("  then TaskStop + resume the workflow (Workflow resumeFromRunId) from the last good stage.")
    else:
        newest = now - arts[-1].stat().st_mtime
        print(f"  Healthy: newest write {_age_str(newest)} ago. Workflow appears to be progressing.")


if __name__ == "__main__":
    main()
