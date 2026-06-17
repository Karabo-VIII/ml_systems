---
name: author-skill
description: How to author a new SKILL.md so this harness gains a reusable capability. Use when the user asks to add a skill, teach a procedure, capture a workflow, or write a SKILL.md.
---
# Authoring a skill

A skill is a markdown file the harness can SELECT and inject when relevant. Create either layout:
```
skills/<name>/SKILL.md      (preferred -- can bundle reference files + scripts beside it)
skills/<name>.md            (flat, single file)
```

## Frontmatter (required)
```
---
name: <kebab-case, [a-z0-9-]>
description: <ONE line. State WHAT it does AND WHEN to use it, in THIRD PERSON. This is the selector signal --
             it decides whether the skill is chosen, so put the trigger words in it. Keep under ~1 sentence.>
---
```
Good description: `"Summarize a CSV -- row/col counts, dtypes, per-column stats. Use for CSV/tabular/dataframe tasks."`
Bad: `"I can help with data."` (no triggers, first person, vague).

## Body (the instructions, loaded only when the skill is selected)
- Keep it tight (a few hundred lines max). The body is the Tier-1 payload -- it costs tokens only when selected.
- Give concrete steps, the key API/commands, constraints, and a worked example.
- For deterministic/large work, prefer a bundled SCRIPT the worker RUNS (only its output costs context) over prose.

## After authoring
Nothing to restart -- the next run with `--skills-dir ./skills` re-scans the dir and the new skill is selectable.
To verify it is picked up: `python -c "from metaop.skills import select; print([s['name'] for s in select('<a task that should match>', 'skills', k=3)])"`.
