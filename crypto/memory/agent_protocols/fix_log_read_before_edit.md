# Fix-Log Read Before Edit

> Codifies CLAUDE.md §9 ("Fix Log Protocol") as a mandatory pre-edit gate.

## Trigger

Before editing ANY of:

- Model version files (`src/wm/v*/`)
- Strategy sleeve files (`src/strategy/sleeves/`, `src/strategy/gen*/`)
- Pipeline producers (`src/pipeline/`)
- Cost model / risk model / position sizer
- Training scripts (`train_world_model.py` and variants)

## Steps

### 1. Identify the file's fix-log entry
| File group | Fix log location |
|---|---|
| `src/wm/v{N}/v{N}_training/...` | `memory/fix_logs/v{N}_M.md` (per major.minor) |
| `src/strategy/sleeves/*` | `memory/fix_logs/INDEX.md` (look for sleeve-specific patterns) |
| `src/pipeline/*` | `memory/fix_logs/INDEX.md` (look for Pattern P et al.) |
| Cross-cutting | `memory/fix_logs/INDEX.md` (Pattern A-Z entries) |

### 2. Read the relevant entry
Look for:
- Bugs previously fixed in this exact area (don't reintroduce)
- Anti-patterns flagged
- Settings constants that must stay synchronized
- Schema invariants

### 3. Cross-check the proposed edit against fix-log entries
For each fix-log finding:
- Does my edit re-introduce a fixed bug? → STOP, re-plan
- Does my edit violate a documented anti-pattern? → STOP, re-plan
- Does my edit affect a synchronized constant? → propagate the change to all
  versions that share the constant (CLAUDE.md §10)

### 4. Proceed with edit
Only after step 3 passes. The edit's commit message must reference the
fix-log entry if the bug type is recurring.

### 5. If a NEW bug emerges from the edit
Append to the appropriate fix-log (per CLAUDE.md §9): date, severity,
file:line, what was wrong, what was fixed. Cross-version pattern → add to
Cross-Cutting Bug Patterns in `memory/fix_logs/INDEX.md`.

## Why this matters

Per CLAUDE.md §9 and §13 ("RED TEAM Audit Protocol"), fix logs are project
memory. They document specific anti-patterns that have ALREADY cost
debugging time. Editing without reading them is gambling that you'll
independently re-derive the same lesson.

In the past 2 months, fix-log patterns A-P have prevented at least 10
re-introduction-of-fixed-bug events. The protocol is cheap (2 minutes per
edit) and the prevention is real.

## Failure mode this prevents

Without the protocol:
- V1.1 fix (Bin range [-1, 1]) is at risk of being reset when someone edits
  V1.1 settings without reading `memory/fix_logs/v1_1.md`
- Pipeline us-scale fix (Pattern P, today) could be reintroduced by
  someone reverting `_aggtrades_utils.py` without context
- ATME batch-level vs per-sample bug (pattern in `auditor/SKILL.md`) could
  reappear in new model variants

## Anti-patterns

- "I'll read the fix log if I hit a bug" → NO, read BEFORE edit. The point
  is to PREVENT bug re-introduction.
- "This is a small change, fix log doesn't apply" → NO, small changes
  cause most reintroductions.
- "Fix log is too long to read entirely" → read the INDEX.md and the
  specific entries for files you're touching. ~5 minutes max.

## Quick command

```bash
# Before editing src/wm/v1/v1_1_training/settings.py:
cat memory/fix_logs/v1_1.md
cat memory/fix_logs/INDEX.md | head -40

# Before editing a sleeve:
grep -i "sleeve\|blend\|runner" memory/fix_logs/INDEX.md
```

## Integration with pre_action_debate.md

`pre_action_debate.md` step 2 ("hidden constraint") explicitly asks "what
fix-log entry might this violate?" — this protocol is the answer mechanism.
Read fix-log BEFORE debate; cite specific entries in the debate output.
