#!/usr/bin/env python3
"""check_report_claims.py -- the mechanical gate for the user's #1 most-repeated correction: HONEST REPORTING.

"no lies to get the true picture of performance" was steered 8+ times in one stretch, and every layer of the WEALTH-BOT
TRUST STACK in CLAUDE.md exists because a silent inflated number shipped (+468% V2, +501.2% pre-MtM, the +120% NAV
ceiling, the btc_ret_same_day t+1 leak). Prose reminders drift; this is the mechanism: any PERFORMANCE number in a
report/markdown artifact must carry a CLAIM-TAG (VERIFIED / REPORTED / INFERRED) or a reconciliation note nearby, else
it is flagged. Wire as a CDAP sub-check (commit-time) and/or a PreToolUse guard on Write/Edit to report *.md.

A "performance number" = a number within TAG_WINDOW lines of a performance keyword (return/ROI/Sharpe/compound/drawdown/
profit/gain/equity/NAV/alpha/win-rate/CAGR). A bare number in prose is NOT flagged (avoids the cry-wolf trap). Read-only.
Exit code = number of UNTAGGED performance claims (0 = clean). No emoji (cp1252).

Usage:
  python src/audit/check_report_claims.py <file.md> [<file2.md> ...]   # check given files
  python src/audit/check_report_claims.py --staged                      # check git-staged *.md report artifacts
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

__contract__ = {
    "kind": "audit_gate",
    "inputs": ["report/markdown files (or --staged)"],
    "outputs": ["list of untagged performance claims; exit = count"],
    "invariants": [
        "a performance number (near a perf keyword) must have a claim-tag (VERIFIED/REPORTED/INFERRED) or "
        "reconciliation within TAG_WINDOW lines, else flagged",
        "bare numbers in prose are NOT flagged (no cry-wolf); read-only; deterministic",
    ],
}

PERF_KEYWORD = re.compile(
    r"\b(return|returns|ROI|sharpe|sortino|compound|compounded|CAGR|draw.?down|profit|PnL|P&L|gain|equity|"
    r"NAV|alpha|win.?rate|hit.?rate|annuali[sz]ed|net\s+of)\b", re.I)
# a number that looks like a performance figure: percentage, multiplier (3.2x), sharpe-like (1.45), bps
PERF_NUMBER = re.compile(r"([+-]?\d{1,3}(?:\.\d+)?\s*%|[+-]?\d+(?:\.\d+)?\s*x\b|\b\d+(?:\.\d+)?\s*bps\b|"
                         r"sharpe[^0-9]{0,6}\d+(?:\.\d+)?)", re.I)
TAG = re.compile(r"\b(VERIFIED|REPORTED|INFERRED)\b|reconcil|backtest\s*==\s*sim|RWYB|sim.?match|claim.?tag", re.I)
TAG_WINDOW = 2  # lines


def scan(text: str, where: str = "") -> list:
    lines = text.splitlines()
    findings = []
    for i, line in enumerate(lines):
        if not PERF_NUMBER.search(line):
            continue
        # is it a PERFORMANCE number (near a perf keyword within the window)?
        lo, hi = max(0, i - TAG_WINDOW), min(len(lines), i + TAG_WINDOW + 1)
        ctx = "\n".join(lines[lo:hi])
        if not PERF_KEYWORD.search(ctx):
            continue  # a number, but not in a performance context -> not our concern
        if TAG.search(ctx):
            continue  # tagged / reconciled -> ok
        num = PERF_NUMBER.search(line).group(0).strip()
        findings.append((where, i + 1, num, line.strip()[:100]))
    return findings


def _staged_md():
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout
        return [f for f in out.splitlines() if f.endswith(".md") and ("report" in f.lower() or "runs/" in f or "docs/" in f)]
    except Exception:
        return []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    files = _staged_md() if "--staged" in sys.argv else args
    if not files:
        print("usage: check_report_claims.py <file.md ...> | --staged"); return 0
    all_findings = []
    for f in files:
        if not os.path.exists(f):
            continue
        try:
            all_findings += scan(open(f, encoding="utf-8", errors="replace").read(), f)
        except Exception as e:
            print(f"  (read error {f}: {e})")
    if not all_findings:
        print(f"=== check_report_claims: {len(files)} file(s) | 0 untagged performance claims -- clean ===")
        return 0
    print(f"=== check_report_claims: {len(all_findings)} UNTAGGED performance claim(s) "
          f"(tag with VERIFIED/REPORTED/INFERRED or reconcile) ===")
    for where, ln, num, line in all_findings:
        # cp1252-safe: doc lines may carry non-cp1252 chars (arrows, middots) that crash Windows stdout
        print(f"  {where}:{ln}  [{num}]  {line}".encode("ascii", "replace").decode("ascii"))
    return len(all_findings)


def _selftest():
    print("=== check_report_claims selftest ===")
    bad = "The strategy delivered a compound return of +94% over the period."
    good = "The strategy delivered a compound return of +94% (VERIFIED: backtest==simulator, RWYB)."
    neutral = "We trained for 94% of the epochs and used 8 workers."
    fb = scan(bad, "bad.md"); fg = scan(good, "good.md"); fn = scan(neutral, "neutral.md")
    print(f"  untagged perf claim -> {len(fb)} flagged (want 1):", fb[0][2] if fb else None)
    print(f"  VERIFIED+reconciled -> {len(fg)} flagged (want 0)")
    print(f"  non-perf number     -> {len(fn)} flagged (want 0)")
    assert len(fb) == 1 and len(fg) == 0 and len(fn) == 0
    print("  ALL PASS -- untagged perf numbers flagged; tagged + non-perf pass.")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        raise SystemExit(main())
