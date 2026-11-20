# DIRECTIVES REGISTER — the canonical record of user mandates (mined from the build session, 2026-06-06)

> **Why this exists (user mandate, 2026-06-06):** *"go through this chat history thoroughly to figure out what I was
> asking of the instances as we were building the langgraph and autonomous mode … I won't be able to remember
> everything."* This is the durable, single-source list of every standing directive, so no instance re-asks or
> forgets. Mined from all 56 user turns of the founding session (transcript `6560cf55…jsonl`). **Status tags:**
> ✅ captured (where) · ⚠️ GAP (not yet done — actionable). Keep this updated whenever the user gives a new standing
> directive.

## A. The autonomous operating model (the "brain")
1. ✅ **End-to-end autonomous workflows as a FRAMEWORK** (instances decide direction, act agentically, self-improve;
   the Claude brain is fixed, the autonomy is in the LOOP around it). [U8/U9/U25/U26] → `crypto/docs/AUTONOMY_FRAMEWORK.md`,
   `crypto/scripts/autonomy/metaop/` (real LangGraph), `/orc` skill.
2. ✅ **The 3 loops, always launched together** = 4 concurrent roles: orchestration (the overseer = YOU), the
   langgraph SOLUTIONING loop (does the work, runs long), the meta agent, the project-wide monitor/evolution.
   [U12/U22/U24/U41] → `/orc` SKILL.md, [[feedback-orc-4-roles-dont-self-execute]].
3. ✅ **Orchestrate, do NOT self-execute** — the langgraph loop solves; the overseer launches/monitors/judges/commits.
   [U22/U24] → `/orc`, OVERSEER.md.
4. ✅ **Expert + plain variants**, each its own improvement loop/lessons (separable or pooled to one meta lane). [U36]
   → metaop `--mode expert|plain --learnings-channel`.
5. ✅ **Autonomous mode is the DEFAULT modus operandi**; a summary is NOT a stop; run to the end of the allocated
   window (Claude's "summarize-then-stop" weakness is defeated by the Stop hook). [U5/U12] → autonomy_loop.py Stop hook.
6. ✅ **Respect the time**: use the whole allocated window; when the primary objective is met with time left, GENERATE
   + solve ADJACENT problems (don't idle-stop early). [U21] → [[feedback-respect-time-solve-adjacent]], AUTONOMOUS_RUNNER §3.
7. ✅ **The ABSOLUTE 60s watcher** — every /orc run has `watcher.py` running; FOLLOW it (re-check at the 60s cadence;
   never park-and-hang — a loop can be alive-but-hung). [U40] → watcher.py (self-respawn + stall-detection), `/orc`.
8. ✅ **Meta = 60s DUAL-VIEW**: the project-level meta audit rides the 60s wakeup (moved off the 3h gate), attached to
   the meta agent, which holds BOTH the project view and the running-loops view. [U40/U42 + 2026-06-06] → `/orc`, watcher.
9. ✅ **Per-minute gap-find-and-solve / correct-as-you-go**: as the loops work, fix weaknesses in the apparatus/brain
   the moment you find them (meta-authorized). [U12/U42] → the autonomous-mode hook, this session's correct-as-you-go.
9a. ✅ **The 60s cycle is a HEALTH + STATE check, not just liveness** [2026-06-06]: the overseer's CONSTANT 60s job is
    diagnostics → ACT/CORRECT/INTERVENE on gaps/breakages/slowness (`crypto/scripts/autonomy/loop_health.py` + CDAP +
    `skill_diagnostics`) AND verify the loops are PRODUCTIVE (learning = lanes growing, writing to the RIGHT
    lanes/checkpoints, loop↔meta comms stable). Attended mode = the overseer IS the meta loop → READ the loops'
    learning lanes + WRITE the meta synthesis FORWARD to `meta.jsonl`. → `/orc` 60s HEALTH+STATE clause, `loop_health.py`,
    [[feedback-60s-cycle-is-health-state-check]].
9b. ✅ **Idle → SURFACE THE META DIGEST** (never silent, never busywork) [2026-06-06]: in a non-busy stretch, surface
    what META is working on — NOW + new ideas + new questions + EV-frontier — from the real meta lane, so the user can
    steer. → `/orc` IDLE PROTOCOL, [[feedback-idle-surface-the-meta-digest]].
9c. ✅ **FRAME BROADLY (breadth+depth) + CARRY THE ROLLING STATE** [2026-06-06]: a single task must auto-expand into
    DEPTH **and** BREADTH questions with the standing lenses always at the back of the mind (setup-not-candle,
    compound-not-IC, trader/institution mindset, crypto nature, archetype-fit, explore-all-dims) → a solution PATH
    possibly different+better than asked; NEVER declare 'impossible' without the anti-impossible rail (validate numbers
    + re-frame first). And REMEMBER the rolling state across compaction (CONSTRAINTs/CORRECTIONs/PIVOTs/OPEN_Qs). →
    `crypto/scripts/autonomy/problem_framing.py` + `crypto/scripts/autonomy/rolling_ledger.py`, `/orc` FRAME-BROADLY clause,
    [[feedback-frame-broadly-and-roll-memory]].
9d. ✅ **BE RESOURCEFUL — decompose the IDEAL + reverse-engineer; don't collapse the framing or quit on one test**
    [2026-06-06]: LLMs force a rigid framing, prove it fails, give up; counter with spirit-first, enumerate-framings,
    constraint-as-enabler, and decompose-the-ideal (the best-achievable WITHIN constraints = the oracle; reverse-engineer
    toward it — e.g. best adaptive-MA-per-move IS the oracle, NOT per-candle). The **self-evolution loop reflects on
    COGNITION (how Claude fails), not just artifacts**, and EXTENDS the failure-mode registry. →
    `crypto/scripts/autonomy/resourcefulness.py`, `/orc` BE-RESOURCEFUL clause, [[feedback-be-resourceful-decompose-the-ideal]].
9e. ✅ **END EVERY TURN WITH THE NEXT VALUE-ADDING ITEM** [2026-06-06]: close every turn by surfacing THE top
    value-adding next item (top of the EV-frontier: roadmap EV-rank + hypothesis_register open frontier + rolling OPEN_Qs
    + broad lenses like trading-mindset engines / crypto research) so the user steers without asking. One line "Next
    highest-value: …" + alternatives. NOT a stop. → `/orc` END-EVERY-TURN clause, [[feedback-end-turn-with-next-value-item]].
10. ✅ **Self-evolution dedicated time** — across the life of the project, run a self-evolution/improvement cycle
    (close gaps, push to SOTA, fold learnings). [U43] → `crypto/docs/SELF_EVOLUTION_LEDGER.md` (now continuous via loop-2 60s).
11. ✅ **Resumability** — survive interruptions / subscription limits / restarts; loops are durable + independent.
    [U6/U17/U18/U45] → `resume_all.py`, per-thread SqliteSaver checkpoints.
12. ✅ **Lessons persist across objectives/threads** — "everything is part of the same project." [U34] → learnings lanes.
13. ✅ **Clean stop / auto-cleanup on kill** [U34] → BUILT: `run_metaop.py stop --thread X` = kill the process TREE +
    release the lease + reap stale locks + PRESERVE the durable checkpoint (RWYB-verified 2026-06-06). USE IT instead
    of manual `Stop-Process` + `rm lock`. (The command existed; the only gap was the overseer not using it.)
14. ✅ **Persistent loop instance (live → learn → rebirth), not a fresh `claude -p` per NODE** [U34/U42] → BUILT +
    RWYB-verified 2026-06-06: `PersistentCliBrain` (`--backend persistent`) keeps ONE warm claude session across nodes
    via `claude -p --resume <session_id>` (captured from `--output-format json`); context CARRIES (2-node continuity
    smoke PASS); rebirth on context-limit; graceful CliBrain fallback. Harvested to the skill library. The last
    unbuilt user-asked component — now done.
15. ✅ **Agent-cap discipline**: ≤2 Opus + ≤9 Sonnet concurrent (+1 meta); serial chains unbounded. [U42] → CLAUDE.md.

## B. Guardrails, governance & permissions
16. ✅ **No frankenstein — sandbox → review → push**: build in the working tree, the OVERSEER reviews, then commit
    (git is the revert net). [U3/U9] → workers never commit; overseer commits after RWYB-judging.
17. ✅ **Push reviewed commits to the remote** (permission granted U11) → DONE 2026-06-06 (pushed 47 commits to
    origin/wm-hardening-2026-05-29). Standing practice: the overseer pushes after the review+commit step (sandbox →
    review → commit → PUSH), not just local commits.
18. ✅ **All permissions allowed (git is the revert net); NO approval prompts** — file-edits + commands auto-approved.
    [U11/U12] → settings `defaultMode: bypassPermissions` + `permission_gate.py` (+ the 2026-06-06 fixes: loop-children
    fenced from commit/control-surfaces but auto-allowed for work; F4 over-block scoped).
19. ✅ **Loops NEVER commit/push/deploy or touch control surfaces** (arm/disarm, permissions, hooks, settings) — the
    OVERSEER owns those. [U3 governance] → metaop tools.py HARD fences + permission_gate METAOP_LOOP fences.
20. ✅ **Mandate = make the project ROBUST (the FOUNDATION), not chase strat/code** — set the foundation so that when
    the user says "1-5%/d or per move," it gets DONE. [U11] → [[feedback-foundation-not-solve]].
21. ✅ **Elevate every directive to SOTA** (a directive is a FLOOR not a ceiling). [U2/U26] → [[feedback-elevate-directives-to-sota]].
22. ✅ **Red-audit the project's BRAIN (skills/directives/orchestration), propose upgrades, apply the loop to the brain
    itself** — instances skip gates / forget instructions / think narrowly / mis-align objectives. [U3] → the skill +
    directive gap audits (`SKILL_GAP_AUDIT.md`, `DIRECTIVE_GAP_AUDIT.md`), `skill_diagnostics.py`.
23. ✅ **Reproducible + harmonious components** — know the weakpoints / chokeholds / contention; harmonize parallel
    skills/instances. [U15] → `crypto/docs/SYSTEM_TOPOLOGY.md`.
24. ✅ **Verify before claiming / RWYB / no false victory**; trust-but-verify a 2nd voice on load-bearing calls. [U9] →
    RWYB everywhere, the de-biased judge upgrades.

## C. Anti-drift / objective-fulfilment (the user's core fear)
25. ✅ **Guarantee the problem is SOLVED on return** — if "develop a system per instrument at 1-5% ROI per fire" with
    allocated time, ensure it's done; no narrow-mindedness, no drift. [U9] → OVERSEER objective-guardian + verify-gate.
26. ✅ **Objective re-anchor every cycle; honest-failure (no silent target reframing); IDLE-STOP vs never-idle-stop
    reconciled by timed-vs-untimed.** [U7/U12/U21] → AUTONOMOUS_RUNNER, the Stop hook.
27. ✅ **A higher-level META directive/lens** for the project + surrounding areas; add the missing lenses. [U2] → this
    register + OVERSEER.md + the SOTA-upgrades section.

## D. The strategy / project work (what the loops SOLVE)
28. ✅ **North star**: 2-5%+ NET per move/per-trade; hold hours-to-<7d; also 1d/3d/7d ROI. Treat as what an ORACLE
    attains → decompose its DNA, diffuse the noise, copy the realizable part. [U9/U37/U39] → `PROJECT_NORTH_STAR`, oracle-decomp.
29. ✅ **The unit of trading is the SETUP across a MOVE (multi-candle); ENTRY signals only — EXIT is a separate
    decomposable domain (trailing/fixed/vol); PER-SETUP not per-candle.** [post-compaction] → MEMORY founding framing,
    narrate, trader exit-policy ownership.
30. ✅ **Adaptive MA system** — per rolling window compute per-asset salient variables (vol/regime/cluster) to ADAPT MA
    configs to the current regime; regime is NOT the anchor; capture moves dynamically; MA is just the instrument;
    explore ALL cadences/chart-types. [U37/U39] → the MA-oracle decomposition (closed at daily/4h; pivots to sub-bar).
31. ✅ **Crypto is its OWN market** (24/7, perp funding, liquidation reflexivity, BTC-beta, fragmentation) — the engines
    must encode this. [post-compaction] → `crypto/src/narrate/crypto_context.py`.
32. ✅ **Strategy-archetype MASTER MAP** (scalp/HFT/intraday/swing/trend/mean-rev/breakout) — pick the right MODE; don't
    repeat the per-candle mistake. [post-compaction] → `crypto/docs/MARKET_STRATEGY_ARCHETYPES.md`.
33. ✅ **Descriptive market-intelligence FOUNDATION** — engines per chart type that narrate the WHAT of (asset, period,
    chart); decompose ALL chimera; wire our trained artifacts + a downloaded non-crypto TS foundation model (MOMENT),
    validated against what we know; max coverage + explainer. The foundation BEFORE any strat. A skill paired with
    `trader`. [U46] → `crypto/src/narrate/`, `/narrate` skill.
34. ✅ **Research the market case BEFORE strategies** (opportunity premise; don't find the edge yet). [U20] →
    `crypto/docs/MARKET_RESEARCH_2026_06_05.md`.
35. ✅ **Market Wizards series** — distil lessons, fan out the stories, one PDF. [U43] → `deliverables/Market_Wizards_Distilled.pdf`.

## E. Open gaps to action (consolidated)
- ✅ #13 clean-stop (already built — verified + use it) · ✅ #17 push to remote (done) · ⚠️ **#14 persistent loop
  instance (live → learn → die → rebirth, not a fresh `claude -p` per node)** — THE one genuinely-unbuilt user-asked
  item, and the highest-value remaining framework optimization (per-node spawning is the main latency/quota cost,
  U34/U42). Everything else in the register is captured. Update tags to ✅ as built.
