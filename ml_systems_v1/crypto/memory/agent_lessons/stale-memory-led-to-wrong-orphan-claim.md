# Stale memory led to wrong "orphan" claim

**Date observed**: 2026-05-13
**Severity**: high (user-facing wrong fact in strategic synthesis)
**Frequency**: one-off but pattern-recurring

## Context

While producing `docs/META_ROI_SYNTHESIS_2026_05_13.md`, I claimed
`v6_frontier` and `v7_frontier` were "ORPHANED in production_blends.yaml."
This claim was a major architectural finding that led to a recommendation
("REWIRE v6/v7_frontier as highest-EV next move").

## What went wrong

The claim was wrong. Commit `2345a48` on 2026-05-12 had already added
`V6_FRONTIER_v2026_05` to `production_blends.yaml`. My synthesis ran ONE DAY
STALE.

User had to point this out before I caught it. I then re-read the yaml,
confirmed the wire was in place, and pivoted the work to wiring whale + liq
overlays (the actual remaining orphans).

## Root cause

I trusted a memory memo (`memory/meta_layer_protocol_2026_05_11.md`) as
authoritative for current state. The memo was true when written; it became
stale 1 day later when the rewire commit landed. I didn't run `git log` or
re-read the yaml before claiming state.

## How to apply

1. Always run `git log --since="24 hours ago"` at session start (and before
   any state claim) — codified in `agent_protocols/cross_instance_awareness.md`
2. Treat memory memos as REPORTED (with potential staleness), not VERIFIED —
   codified in `agent_protocols/calibrated_uncertainty.md`
3. For load-bearing claims (claims that drive recommendations), re-verify the
   underlying file BEFORE citing

## Related

- `memory/agent_protocols/cross_instance_awareness.md` — the protocol that
  would have caught this
- `memory/agent_protocols/calibrated_uncertainty.md` — VERIFIED vs REPORTED
  tagging that would have flagged "STALE-REPORTED" on a 2-day-old claim
- `docs/META_ROI_SYNTHESIS_2026_05_13.md` — the affected synthesis
