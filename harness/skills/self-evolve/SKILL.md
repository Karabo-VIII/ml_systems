---
name: self-evolve
description: How this harness augments and improves ITSELF from within its own loop -- grow the skill library, evolve the planner, consolidate memory. Use when the user asks to "augment yourself", "evolve", "improve the harness", "get better", or to add a new capability/skill.
---
# Self-augmentation playbook (the harness improving itself)

You are an autonomous build loop that can improve its OWN capabilities. Three levers, all already built in:

## 1. Grow the skill library (accumulate what you verify)
Run any build with `--harvest` and a `--skills-dir`. Whenever a build node passes the MECHANICAL verifier
(`verify_cmd` exit 0 = ground truth), a `SKILL.md` for that capability is authored into the skills dir automatically,
and becomes selectable on future runs. So: **build it, verify it mechanically, and it is harvested as a reusable skill.**
```
python -m metaop.manager launch --objective "..." --skills-dir ./skills --harvest --backend cli --durable
```

## 2. Author a new skill by hand (when asked for a specific capability)
Write `./skills/<name>/SKILL.md` with frontmatter (`name`, a third-person `description` that says WHAT it does AND
WHEN to use it) + a concise body. See the `author-skill` skill for the exact format. New skills are picked up on the
next run with zero restart.

## 3. Evolve the planner (improve HOW you decompose problems)
```
python -m metaop.manager evolve --backend cli --generations 3 --pop-size 4   # one evolution run
```
This optimizes the planner prompt against the honest solve-rate fitness (it cannot be faked) and, on a real
improvement over baseline, installs it as the champion (`champion.json`). The next `launch`/`resume` auto-applies it
via `apply_champion`. An elitism floor guarantees a worse planner is never installed.

## 3b. Continuous self-evolution (hands-off)
```
python -m metaop.manager improve --backend cli --rounds 20        # or --max-minutes 60
```
The daemon repeats step 3 round after round on its own, each round seeding from the best-so-far champion, so the
planner gets monotonically better with no further input. Bounded + resumable (the champion persists). Use this when
the user asks to "keep evolving" / "run continuously" / "improve in the background".

## The discipline that makes self-improvement safe
- **Verify before you trust.** A capability counts as "yours" only when a real command (`verify_cmd`) says so. Never
  harvest or claim an unverified result.
- **Monotonic.** Every verified capability is harvested once and reused -- never re-discover what you already proved.
- **Reflexion.** When a build is refuted, write down WHY it failed + what to try differently (the reflect step does
  this); the next attempt is a directed retry, not a fresh guess.
