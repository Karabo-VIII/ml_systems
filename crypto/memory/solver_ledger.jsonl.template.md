# solver_ledger.jsonl — schema spec

> **Status**: schema documentation. The actual ledger lives at `memory/solver_ledger.jsonl` (append-only, one JSON object per line).
> **Used by**: `.claude/skills/_common/SOLVER.md` step 8 (LEDGER_WRITE) — the goal-decomposition + falsifier-first protocol.
> **Created**: 2026-05-22 from the dialectic round on `/solver` vs preamble.

## Schema (one JSON object per line)

```json
{
  "ts": "ISO-8601 with TZ (e.g., 2026-05-22T19:00:00+02:00)",
  "session_id": "<short descriptor — e.g., solver_north_star_1pct_2026_05_22>",
  "goal_id": "<short id — reused across sessions for the same goal>",
  "goal_tuple": {
    "metric": "<measurable metric name, e.g., daily_ROI_pct_compounded_LO_spot_lev1>",
    "current": "<number — best honest measurement of current value>",
    "target": "<number — user-mandated target>",
    "gap": "<number — multiplier from current to target>",
    "constraints": ["<list of binding constraints>"]
  },
  "paths": [
    {
      "name": "<short_id, snake_case>",
      "mechanism": "<one sentence>",
      "expected_lift_pct": "<number — best-case daily ROI delta>",
      "prerequisite": "<what must be true / done first>",
      "cost_hr": "<wall-clock + compute estimate>",
      "falsifier": "<the cheapest experiment that would KILL this path>",
      "prior": "<P(path closes goal | path is true), 0-1>",
      "posterior": "<P after evidence; null on first write>",
      "falsifier_fired": "<true|false|null>",
      "references": ["<fix-log entries, prior ledger rounds, paper:dois>"],
      "evidence_pointers": ["<files / runs touched during evaluation>"]
    }
  ],
  "decision": "<DISPATCH_TOP_K | ITERATE | ESCALATE | TERMINATE>",
  "dispatched_paths": ["<list of path names that went to maxx workers>"],
  "queued_paths": ["<list of path names deferred for next round>"],
  "next_action": "<one sentence>",
  "composes_with": ["<dialectic round id>", "<maxx run id>", "<docs/PROVENANCE doc>"]
}
```

## Field-level invariants

- `ts`: ISO-8601 with timezone. No naive timestamps.
- `session_id`: must be unique per solver invocation but recur for the same logical goal so cross-session aggregation works.
- `goal_id`: persistent across sessions. New goal → new goal_id. Refining the same goal → same goal_id.
- `goal_tuple.current` and `.target`: numeric, same units. `.gap` = `.target / .current` for ratio goals or `.target - .current` for delta goals (specify in `.metric`).
- `paths[]`: minimum 7 entries (per SOLVER.md step 2). Fewer → refuse the round.
- `paths[].prior`: 0 < prior < 1; > 0.5 requires ≥2 evidence anchors in `references[]`.
- `paths[].posterior`: null until POSTERIOR_UPDATE (step 7) runs.
- `paths[].falsifier_fired`: tri-state (null on first write, true after refutation, false after confirmation).
- `decision`: one of the 4 enum values; no free-form.
- `composes_with`: cross-references to existing ledger entries — keeps the calibration graph traversable.

## Append-only discipline

- NEVER edit an existing line. Outcomes go in NEW lines that reference the prior `session_id` + `goal_id`.
- POSTERIOR updates are append-rounds — a new entry with the same `goal_id`, updated `paths[].posterior`, and `composes_with` pointing to the prior round.

## Reading the ledger

For aggregate analysis across sessions, parse with polars or jq:

```bash
# Last 5 entries
tail -n 5 memory/solver_ledger.jsonl

# Goals where any path crossed posterior >= 0.5
jq -c 'select(.paths[].posterior >= 0.5) | {ts, goal_id, decision}' memory/solver_ledger.jsonl

# All paths with falsifier_fired=true (rejection history)
jq -c '.paths[] | select(.falsifier_fired == true) | {name, falsifier, references}' memory/solver_ledger.jsonl
```

## Bootstrapping the file

On first solver invocation: create `memory/solver_ledger.jsonl` as an empty file, then append. Do NOT pre-populate.

## Audit cadence

- After 5 SOLVER runs: audit prior-vs-realized correlation. r < 0.3 → priors are biased; review.
- After 10 SOLVER runs: audit whether SOLVER decisions outperformed a vanilla oracle baseline on the same goals (the original dialectic round's 1-hr A/B confirming experiment).
- Annually: prune entries whose `goal_id` is no longer active (move to `memory/solver_ledger_archive_YYYY.jsonl`).
