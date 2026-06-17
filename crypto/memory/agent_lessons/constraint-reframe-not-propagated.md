# Constraint reframe not propagated to all artifacts

**Date observed**: 2026-05-13
**Severity**: medium (silent misalignment between docs)
**Frequency**: pattern-recurring (every time user reframes target)

## Context

Earlier in the session, I framed the user's 0.75-3%/day ROI target as
"SELECTIVE ENTRY (80-150 conviction days per year)" — a deliberate
reinterpretation to make the target feasible.

Later, the user clarified: 1-10% daily moves are abundant — 100% of 2025-2026
days have ≥1 u100 asset moving ≥5%, avg 20 assets/day. Empirical evidence
showed the opportunity is 365 days/year, not 80-150.

## What went wrong

After the reframe, multiple in-flight artifacts retained the OLD framing:
- `docs/META_ROI_SYNTHESIS_2026_05_13.md` §1 still said "80-150 conviction days"
- `memory/meta_roi_synthesis_2026_05_13.md` still said "SELECTIVE ENTRY"
- V7_FRONTIER sleeve weights were sized for "sparse trigger" without considering dense-opportunity sizing
- F1-F8 target framing was still per-event PnL, not per-(asset, day) pick-from-20

I updated SOME of these (docs/memo) after user asked for status, but not all
(V7 sizing, F1-F8 framing). The dissonance between old and new framing
persisted silently.

## Root cause

No formal mechanism for "constraint changed → sweep all affected artifacts."
I updated reactively (user-prompted) instead of proactively (auto-sweep on
constraint change).

The lazy re-evaluation pattern is the bug.

## How to apply

1. When user changes ANY constraint, scope, target, or framing, run the
   constraint propagation sweep IMMEDIATELY (codified in
   `agent_protocols/constraint_propagation.md`)
2. Build affected-decision registry first; then update each entry
3. Report the sweep result back to user, even when they didn't ask
4. The CURRENT_PLAN.md must record both the CURRENT constraint and the
   last-sweep date

## Affected artifacts (full list from today)

| Artifact | Old framing | New framing |
|---|---|---|
| META_ROI_SYNTHESIS §1 | "80-150 days" | "supersedes — 365 days opportunity" |
| meta_roi_synthesis memory | "SELECTIVE ENTRY" | "§1 SUPERSEDED" |
| V7_FRONTIER caveats | "sparse trigger" | needs note on dense-opportunity context |
| F1-F8 MA specialist target framing | per-event PnL ranking | "pick from 20 daily 5%+ movers" framing in F7-Phase2 |
| Deploy unit definition | "Sharpe" | "% of daily 5%+ movers correctly pre-called AND profitably exited" |

The first two I updated. The last three I haven't formally updated yet (this
lesson is being written to capture the pattern; the actual updates go in
CURRENT_PLAN.md).

## Related

- `agent_protocols/constraint_propagation.md` — the protocol that prevents this
- `CURRENT_PLAN.md` — the persistent artifact that tracks current-constraint state
- `docs/PROJECT_MISSION_2026_05_13.md` — captures the new framing
