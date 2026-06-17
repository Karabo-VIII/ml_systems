# Constraint Propagation Protocol

> When the user changes any constraint, scope, target, or invariant, ALL
> in-flight decisions must be re-evaluated. Failing to do this caused today's
> "80-150 conviction days/year" framing to persist after the user clarified
> the opportunity is 365 days/year.

## Trigger

User changes any of:

- **Targets** (ROI floor, Sharpe minimum, drawdown ceiling, win-rate)
- **Scope** (universe size, time period, asset bucket, regime exclusion)
- **Invariants** (project rules, hard constraints, mission framing)
- **Operating principles** (priority order, deployment strategy, capital allocation rule)
- **Vocabulary / framing** ("daily floor" → "selective entry"; "selective entry" → "every day has opportunity")

## Steps (run IMMEDIATELY after constraint change)

### 1. Restate the constraint change in writing
Single sentence:
> "User changed: <constraint name> from <old value/framing> to <new value/framing> as of <timestamp>."

### 2. Build the affected-decision registry
List every in-flight decision, plan, framing, or commit that referenced the old
constraint. Sources:
- Current TodoWrite items
- Active CURRENT_PLAN.md entries
- Recent (last 5) user-facing reports in this session
- Docs/memos written this session
- Active YAML configs / blends / sleeve weights

### 3. Per-entry re-evaluation
For each affected item:
- Is the conclusion still valid under the new constraint? YES / NO / NEEDS-RE-CHECK
- If NO: what change is needed?
- If NEEDS-RE-CHECK: what verification will resolve it?

### 4. Update the artifacts
Write the updates BEFORE claiming the constraint change is handled. Specifically:
- Update CURRENT_PLAN.md
- Update affected docs/memos (or mark them SUPERSEDED with a forward pointer)
- Update TodoWrite items
- Update sleeve weights / configs if the constraint affects sizing

### 5. Report to user
Single block:
> "Constraint propagation sweep:
>   - Affected items: N
>   - Updated artifacts: list (file:section)
>   - Items now invalidated: list with SUPERSEDED tags
>   - In-flight work re-targeted: list"

## Failure mode this prevents

**2026-05-13 today**: user reframed 0.75-3%/day from "selective entry on 80-150
days/year" to "every day has 20+ ≥5% movers — bottleneck is signal quality
not opportunity density." Several artifacts silently became misaligned:

| Artifact | Old framing | New framing required |
|---|---|---|
| `docs/META_ROI_SYNTHESIS_2026_05_13.md` §1 | "80-150 days" | superseded — opportunity is 365 days |
| `memory/meta_roi_synthesis_2026_05_13.md` | "SELECTIVE ENTRY (80-150)" | now reads as "but §1 SUPERSEDED" |
| V7_FRONTIER sleeve weights | sized for "selective" | might need re-weight given dense opportunity |
| MA specialist Phase 2-5 training target | per-event PnL | now also per-(asset, day) pick-from-20 framing |

I did SOME of these manually after the user pointed it out. A formal protocol
would have caught all of them in a single sweep.

## Example (verbatim 2026-05-13 sweep — done retroactively)

**Constraint change**: User clarified 1-10% daily moves are abundant (per-asset-day
hit rate 80% / 48% / 27% for >=1% / >=3% / >=5%), NOT rare events.

**Affected-decision registry**:
1. META_ROI_SYNTHESIS §1 framing — UPDATED with "supersedes" pointer
2. memory/meta_roi_synthesis_2026_05_13.md — UPDATED with same
3. PROJECT_MISSION_2026_05_13.md — INCORPORATES new framing
4. Deploy-unit definition — INCORPORATES "% of daily 5%+ movers correctly pre-called"
5. F1-F8 MA specialist target framing — needs F7 target to align with "pick from 20" framing

Re-evaluation results:
- (1) (2) (3) (4): updated to new framing. DONE.
- (5): still aligned (MA specialist produces per-(asset, day) ranks). NO change needed.

## Integration with CURRENT_PLAN.md

CURRENT_PLAN.md must record both the CURRENT constraint set AND the date of
last constraint sweep. When a constraint changes, the date updates after the
sweep completes.

## Anti-patterns

- "I'll re-evaluate as I go" → NO, do the sweep IMMEDIATELY when constraint
  changes. Lazy re-evaluation leaves silent misalignment.
- "Only the obvious items need re-checking" → NO, every item in the registry
  must be touched at least with a "still valid" tag.
- "The user can tell me what else to update" → NO, that defeats the
  purpose of propagation.
