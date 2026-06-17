# Agent Lessons

> Methodology lessons learned across sessions about HOW the agent works,
> not WHAT the project does. Project-state lessons go in `memory/*.md`;
> agent-methodology lessons go HERE.

## Purpose

Reflexion-style accumulating lessons about Claude's own working patterns:
- "When I X, Y went wrong" — failure modes to avoid
- "When I tried Z, it worked better" — improvements to repeat
- "I tend to under-estimate W" — calibration lessons

This is the place to record meta-learning. Future Claude instances read these
to avoid repeating methodology errors.

## What goes here

- Decomposition errors ("I broke task into wrong axes; should have been per-asset not per-sleeve")
- Tool usage patterns ("Grep beats Glob when looking for X")
- Sonnet integration failures (see also `agent_protocols/sonnet_integration_safety.md`)
- Time-estimation calibration ("estimated 30min; took 3hr because of Y")
- User-feedback synthesis ("user repeatedly corrects pattern X; here's why")
- Protocol-application failures ("I had pre_action_debate but skipped it because <reason>")

## What does NOT go here

- Project state (which sleeves are shipped, current Sharpe, etc.) → `memory/*.md`
- Domain knowledge (how dollar bars work, what RSSM does) → `memory/*.md`
- Code-bug fix logs → `memory/fix_logs/`
- Per-skill operational rules → `.claude/skills/<skill>/SKILL.md`

## Format

Each lesson is a single file under this folder:

```markdown
# <lesson title>

**Date observed**: YYYY-MM-DD
**Severity**: high | medium | low
**Frequency**: one-off | recurring | systematic

## Context
What was happening when the lesson surfaced.

## What went wrong / right
The observation.

## Root cause
Why this happened. Underlying pattern.

## How to apply
Specific change in future sessions. Concrete, not "do better."

## Related
Links to protocols, fix logs, memos.
```

## Maintenance

- Add a lesson when a methodology error repeats OR when a non-obvious
  improvement is validated
- Cap each lesson at ~150 lines
- Review monthly: deprecate (don't delete) lessons that no longer apply
- INDEX.md kept current as lessons accumulate (>5 lessons)

## First seed lessons (2026-05-13)

These document failures from today's session that this protocol bundle is
intended to prevent in future sessions:

1. [stale-memory-led-to-wrong-orphan-claim.md](stale-memory-led-to-wrong-orphan-claim.md)
2. [sonnet-scout-hallucinated-xsec-orphan.md](sonnet-scout-hallucinated-xsec-orphan.md)
3. [constraint-reframe-not-propagated.md](constraint-reframe-not-propagated.md)
