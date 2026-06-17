# Skill -> shared-loop routing (the same engine, your expertise as the lens)

> **Added 2026-06-08.** Binds every expert skill (`audit`, `trader`, `discover`, `decide`, `architect`, `pipeline`,
> `research`, `trainer`, `validator`, `narrate`). The autonomy ENGINE (the metaop LangGraph loop:
> plan -> dispatch -> judge -> reflect -> route -> replan, with the mechanical verifier and the `fill_window`
> no-idle-stop gate) is SHARED. A skill does not re-implement it; a skill is the **mandate + the expert LENS** the
> loop plans/dispatches/judges through.

## When to route (vs solve in-turn)

- **Duration / autonomy mandate present** ("for N hours", `/loop`, "go agentic", a window budget, "close all gaps"):
  HAND the mandate to the shared loop so it uses the WHOLE window instead of stopping when you finish the obvious
  list. This is the mechanical cure for the idle-stop failure ([[feedback-deliver-end-to-end-nothing-pending]] clause
  5 -- "blocked" needs a falsifiable proof).
- **One-shot, immediately answerable** (a lookup, a single-file edit, a direct question): solve in-turn as normal.
  Do NOT pay for a multi-cycle loop on a trivial ask.

## How to route (the command)

```
python scripts/autonomy/skill_route.py <skill> "<one-line objective>" "<verifiable success criteria>" \
    [--backend cli|ollama|cascade|mock] [--budget N] [--parallel 1] [--no-fill-window]
```

`skill_route.py` reads `.claude/skills/<skill>/SKILL.md` and injects THAT skill's expertise as the brain's planning
lens (`domain` + `set_plan_instruction`), then runs the shared `build(...)` graph with `fill_window=True` and
`expert_mode` (so dispatch workers also attach the matching `.claude/agents/expert-<skill>` persona). The SAME
objective therefore decomposes DIFFERENTLY per skill:
- **audit** -> adversarial red-team nodes (gradient-flow / leakage / invariant / cross-version / repro checks);
- **trader** -> sizing / risk / execution / portfolio / sleeve-lifecycle nodes.

Inspect the injected lens without running: `python scripts/autonomy/skill_route.py <skill> --show-lens`.

## What stays SHARED vs what changes

| Shared (the engine)                         | Per-skill (the lens)                                  |
|---|---|
| plan/dispatch/judge/reflect/route/replan    | `domain` (expert identity) + `plan_instruction` (decompose THROUGH this skill) |
| mechanical verifier (verify_cmd = truth)    | which protocols the plan node applies (from SKILL.md) |
| `fill_window` no-idle-stop + budget bound   | which `.claude/agents/expert-<skill>` persona workers attach |
| learnings channel + replanner recovery      | the learnings `channel` (= the skill name, so lanes don't pool) |

Provenance: 2026-06-08 -- the user asked for the same prompt to be decomposed/solved differently by auditor vs trader
on a shared loop. Mechanism: `scripts/autonomy/skill_route.py`; RWYB: `scripts/autonomy/_test_skill_route.py`
(deterministic lens-divergence + e2e routed run + empirical ollama decomposition-divergence, ALL PASS).
