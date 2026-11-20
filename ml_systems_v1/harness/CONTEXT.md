# Project context (template — edit this for YOUR project)

This is the `CLAUDE.md` equivalent: always-relevant project context injected into the harness at plan time via
`--context CONTEXT.md`. Keep it SHORT and high-signal — every line is a tax on every plan call (especially for a
small local model). Put only what the planner must always know. Replace the placeholders below.

## What this project is
<one or two lines: the goal, the domain, what "done" looks like>

## Invariants (hard rules — do not violate)
- Build verifiable artifacts: every claim is backed by a command that exits 0.
- Prefer the standard library; add a dependency only when it clearly pays for itself.
- <your rule, e.g. "no network calls", "Python 3.11+", "no emoji in output">

## Conventions
- <where code/artifacts go, naming, style — e.g. "tests live next to the module as verify_*.py">

## Do-not-repeat (the dead list — things already tried and refuted)
- <append refuted approaches here so the planner never re-mines them>
