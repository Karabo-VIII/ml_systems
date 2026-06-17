# skills/ — the harness's skill library

Drop-in skills the harness SELECTS by relevance and injects at plan time (progressive disclosure: the manifest of
names+descriptions is always cheap; a skill's body is loaded only when selected). Works on ANY model, with a
mechanical selector sized for a small local one. See `../SKILLS.md` for the full mechanism.

## Layout
```
skills/<name>/SKILL.md      # preferred (can bundle reference files + scripts)
skills/<name>.md            # flat single-file
```
Each `SKILL.md`: YAML frontmatter (`name`, a third-person `description` stating WHAT + WHEN) + a markdown body.

## Use it
```
python -m metaop.manager launch --objective "..." --skills-dir ./skills --backend cli
python -m metaop.manager launch --objective "..." --skills-dir ./skills --harvest   # also GROW this library from verified builds
```

## Starter skills shipped here
The full build-and-improve loop: plan -> research -> build -> red-team -> debug -> reflect -> evolve.
- **plan-first** — decompose a task into ordered VERIFIABLE sub-goals before building (think-before-build).
- **research-web** — use the `web_search`/`fetch_url` tools to ground a load-bearing external fact instead of guessing.
- **write-verifiable** — build an artifact + a mechanical test that proves it (the core honest-build discipline).
- **red-team** — adversarially attack your own artifact (edge cases / failure modes) before finalizing.
- **debug-failure** — fix a failing test/verify_cmd from the real error (rejection-as-gradient).
- **reflect** — write a one-line post-mortem on a failure and carry it forward (Reflexion; the learning half of debug).
- **author-skill** — how to write a new SKILL.md (so the harness can add capabilities).
- **self-evolve** — how the harness augments itself (grow skills, evolve planner, consolidate). The one the loop
  selects when you ask it to "augment yourself".

Add your own domain skills beside these; replace these if you like. New skills are picked up on the next run, no
restart. Verify a skill is selectable:
`python -c "from metaop.skills import select; print([s['name'] for s in select('<task>', 'skills', k=3)])"`
