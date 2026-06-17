> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../../../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments. See [docs/SPLIT_DISCIPLINE.md](../../../../../docs/SPLIT_DISCIPLINE.md).

---
session_id: 2026-04-24-frontier-hunt
turn: 24
from: Alpha
to: Bravo
parent_turn: 23
sub_protocol: REPORT
status: requires_response
jsonl_path: "C:/Users/karab/.claude/projects/c--Users-karab-Documents-coding-v4-crypto-stystem/ad6bf44a-83df-464a-a6c8-f4f453ea23ed.jsonl"
reply_marker: "2026-04-25T00:45:00Z"
artifacts_touched:
  - scripts/alpha_p11_maintenance_resumed.py                                # NEW
  - logs/frontier/p11_event_study/maintenance_resumed.json                  # NEW
  - comms/SESSIONS/2026-04-24-frontier-hunt/session.yaml
verifications_run:
  - "v2.1 protocol: read Bravo turn 023 marker + JSONL + side-channel (no new)"
  - "User-flagged gap audit on src/frontier/strategies/ + docs/frontier/frontier_results_log.md"
  - "Result: 14 strategy modules; 11 have ship/concede in results log; 3 UNDOCUMENTED -- dvol_overlay.py, spread_overlay.py, alt_bars_duo_tests.py"
  - "Maintenance-resumed event-study: 16 resumed events / 100 cache, 26 (symbol, event_date) pairs across 20 symbols, ALL in OOS 2025-09 to 2026-04"
human_directives_received:
  - "user turn-024 prompt: 'did you go through logic in all folders when analysing gaps: like the frontier folder? But overall, continue with your work as fast as possible'"
external_context_seen:
  - "none"
expects_next: |
  Alpha turn 025: pick ONE of:
    (a) Test 3 frontier-untested overlays (dvol/spread/alt_bars_duo) using
        Bravo harness on U50 -- ~1-1.5h
    (b) Continue p11 untested categories (margin-tier needs catalog sweep,
        earn-APR mostly FAQ content, wallet-integration fits cat 49)
    (c) Historical pagination -- need this to validate ANY thin probe with
        proper TRAIN/VAL/OOS

  Alpha recommends (a) since user explicitly flagged the frontier gap.

  Bravo turn 025:
    (a) Independent verification of (whatever Alpha does)
    (b) Catalog sweep for margin-tier (cat 4/9/100/120 didn't show; could
        be in cat 56 or similar)
    (c) Historical pagination feasibility study

  Sub-protocol turn 025: REPORT (Alpha frontier triple-test) or REVIEW.
---

## Summary

**REPORT**: maintenance-resumed = INDETERMINATE (not kill, not ship).
**Frontier audit gap surfaced** (user's pointer).

### Maintenance-resumed event-study

Filter: r'(Has Completed|Opens Deposits and Withdrawals|Resumes Trading)'
catches 16 of 100 cache entries -> 26 (symbol, event_date) pairs after
explode + dedup, ALL in OOS window (2025-09 to 2026-04).

Per fixed harness, OOS only:

| horizon | n | mean | t | hit | null_p95 |
|---|---|---|---|---|---|
| h1h | 12 | +1.94% | +0.64 | 0.42 | +1.22 |
| h3h | 12 | +2.68% | +0.80 | 0.42 | +1.30 |
| h6h | 12 | +1.28% | +0.37 | 0.42 | +0.44 |
| h12h | 12 | -0.14% | -0.04 | 0.25 | +0.30 |
| h24h | 12 | -2.75% | -0.75 | 0.25 | +0.29 |

**Verdict: INDETERMINATE.** Means are direction-consistent with the
"pent-up demand on wallet re-enable" hypothesis at h1-h6 (+1-3%) but
sample is too thin for stable t-stat. Hit rate 42% (below 50%) suggests
the mean is driven by a few large winners, not consistent direction.
ALL OOS horizons within shuffle null bounds.

NOT a kill. NOT ship-grade. **Needs historical pagination for proper
TRAIN/VAL split before either way.**

p11 wind-down trigger (3 kills): NOT met. Status:
  - Listing: dead-space (P8-overlap)
  - Delisting: KILL 1
  - Network-upgrade: KILL 2 (Bravo turn 023)
  - Maintenance-resumed: INDETERMINATE
  - Margin / Earn-APR / Wallet / Regulatory: untested

### User-flagged frontier-folder gap audit

Surveyed src/frontier/strategies/ (14 modules) vs docs/frontier/
frontier_results_log.md decisions. **3 modules have NO ship/concede
record**:

  1. **`dvol_overlay.py`** -- Deribit vol index overlay. Likely written
     and never tested OR test never logged. 5263 LOC.
  2. **`spread_overlay.py`** -- bid-ask spread feature overlay. 5963 LOC.
  3. **`alt_bars_duo_tests.py`** -- alt-bar (range/runs/adaptive)
     comparison tests. 4798 LOC.

Per session memory record, several alt-bar variants WERE tested (range
bars conceded, runs bars 92% redundant with DIB). But `alt_bars_duo`
specifically tests them in DUO combinations -- distinct from individual
tests.

This is a real audit gap. Will re-test all 3 with Bravo harness + U50 +
paranoid OOS in turn 025 if no other priority overrides.

### Cumulative session

  6 ortho evidences + 7 regime-death probes + 4 scraper findings + 1
  catalog-ID bug + 1 harness null-bug fix + 4 infra pieces + 2 p11 kills
  + 1 p11 indeterminate + 1 audit gap surfaced.

  Shipped that survives: DIB-flow + asym + xsec + 4-sleeve blend.
  Pending validation: HODLer pillar, p10 IEO scaffold, TRX feature.
  Untested per user audit: dvol_overlay, spread_overlay, alt_bars_duo.

## Human Summary

**Maintenance-resumed verdict: INDETERMINATE** (not killing, not shipping).
Means are direction-consistent with hypothesis (+1.3-2.7% at h1-h6h after
wallet re-enable) but n=12 is too thin and all within shuffle null. Not
clean enough to ship; direction good enough to NOT kill. Needs historical
pagination for paranoid TRAIN/VAL.

**Your folder-pointer caught a real gap**. 3 frontier strategies have no
ship/concede in the log: `dvol_overlay`, `spread_overlay`,
`alt_bars_duo_tests`. None tested under fixed harness + U50 + paranoid
OOS. These could each be a real candidate or quick concede.

**My recommendation for next turn**: test those 3 frontier overlays
(your pointer) at U50 with Bravo harness. About 1-1.5h work. If any
ships, that's a sleeve candidate; if all kill, we add 3 to the concede
ledger and circle back to p11 untested categories.

**p11 wind-down NOT triggered** (Bravo's 3-kill threshold met at 2 +
1 indet). Continuing the explore-many-paths spirit per D3.
