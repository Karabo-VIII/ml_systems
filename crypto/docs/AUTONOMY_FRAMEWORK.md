# Autonomy Framework — SOTA self-directing, self-improving agent harness (2026-06-05)

> The production realization of Lilian Weng's *LLM-Powered Autonomous Agents* (Planning · Memory · Tool-use ·
> Reflection), with the one addition that makes it actually work in practice and which the 2023 post assumes
> but does not build: a **mechanical control loop** the model cannot talk itself out of.

## 0. The principle (why prose autonomy fails — proven, not asserted)
A Claude instance is **reactive and turn-based**: it answers, then halts. "Be autonomous / keep working" is
*prose*, and prose is a suggestion the model follows inconsistently — the 2026-06-05 brain-audit's core finding,
and demonstrated live: the `autonomous_mode.json` mandate was injected into ~40 consecutive turns and the
instance still stopped 4 times. **Therefore autonomy is not built in the model; it is built in the LOOP around
it.** The model supplies *judgment per cycle*; the harness owns *control*. That split is the entire design.

## 1. Architecture (Weng's components, made mechanical)
| Weng component | This harness | Mechanism |
|---|---|---|
| **Planning — decomposition** | `frontier.json` = an EV-ranked queue of nodes (the n±k lattice) | model decomposes; the queue persists it OUTSIDE context |
| **Planning — reflection** | a mandatory VERIFY node after each build node | "let Claude verify its work" (Boris Cherny) — no false victory |
| **Memory — short-term** | the live context window | per-cycle working memory |
| **Memory — long-term** | `frontier.json` + `memory/` + git history | read-forward at cycle start, write-forward at cycle end — survives context resets |
| **Tool use** | the Workflow tool / subagents / all Claude Code tools | the execution substrate |
| **The loop itself** | **`.claude/hooks/autonomy_loop.py` (Stop hook) + `scripts/autonomy/autonomy_driver.py`** | the keep-going engine prose cannot override |

- **In-session loop** = the **Stop hook**: when the model finishes, it returns `decision:block` + the next node,
  so the harness re-invokes instead of stopping. Fenced 6 ways (anti-infinite via `stop_hook_active`, master
  switch, budget, value-floor, hard ceiling, fail-open). RWYB-tested.
- **Cross-session loop** = the **driver**: spawns fresh `claude -p` sessions per node so the loop survives
  context limits / crashes / session end (state lives in the repo). Schedule via cron for "works while you sleep".

## 1.5 Who calls the loop — the OVERSEER (the meta layer that stands in for you)
**Question: what calls the loop — normal Opus, or something else? Answer: a normal Opus instance running the
OVERSEER ROLE.** Same brain, distinct job. The role is the meta layer. Two tiers, never collapsed:

- **Tier 0 — the Overseer (your stand-in / "human intelligence layer").** When you give a command, the instance
  you're talking to *adopts* it and owns it end-to-end. It does NOT execute. It: (a) sharpens your command into a
  verifiable objective + success_criteria — the act of *standing in for you*, doing what you'd do if you watched
  every cycle; (b) owns the EV frontier; (c) **dispatches** execution to workers; (d) **judges** what returns,
  adversarially, refusing false victory; (e) detects drift/narrowness and course-corrects; (f) decides "truly
  fulfilled?" and only then reports done. Kept alive by the Stop hook in-session; re-spawned by the driver
  cross-session. Full role spec: [`.claude/skills/_common/OVERSEER.md`](../.claude/skills/_common/OVERSEER.md).
- **Tier 1 — Execution.** The workers — sub-agents (Agent tool), Workflows, the autonomy loop — that build/run/
  test/learn/pivot and report structured results UP. Ephemeral; they don't own the objective or declare it done.

**Why separate them:** conflating planner+executor in one context is what causes drift — execution detail
floods the context the meta judgment needs. Keeping the Overseer's context for *judgment only* (it dispatches,
it doesn't build) is the structural fix. This is the orchestrator-worker / "objective-guardian" pattern, and it
is exactly: *"the meta layer is not responsible for execution, but acts as a human intelligence layer to make
sure all work is done."* The Overseer makes the calls (it is your proxy) and escalates to the real you ONLY for
genuinely irreversible real-world actions — everything git can revert, it just does.

## 2. The anti-DRIFT system (your core concern: "ensure the problem is solved on return")
Drift = the agent slowly optimizes something *other than your actual goal*. Defeated mechanically:
1. **Objective re-anchor every cycle.** The Stop hook injects the OBJECTIVE + SUCCESS_CRITERIA into every cycle
   and forces step (a): "re-state the objective in one line + confirm this node serves it." A node that doesn't
   serve the objective is dropped. (Counters Weng's *long-horizon planning* limitation.)
2. **EV tied to the REAL goal, not a proxy.** `value_floor` + `ev` are defined against `success_criteria`
   (e.g. "1-5% ROI per fire, robust on UNSEEN"), so the argmax can't wander to a high-activity / low-value proxy.
   (This is the hardest part — a bad objective function yields confident wrong autonomy. The objective is the input you must get right.)
3. **Verify-gate (no false victory).** Every build node is followed by a VERIFY node; "solved" requires the
   success_criteria to be *verified*, not asserted. (This caught 5 real errors in this session's package.)
4. **Whiplash → mandatory consensus.** Any cycle that REVERSES a prior conclusion or declares a vein dead/proven
   does not stand on one agent's word — it convenes a ≥2-skill panel (`decide` + `audit`) before it becomes canon.
   Stops a single drifting cycle from rewriting the plan.
5. **Provenance-anchored memory.** Every write-forward carries git SHA + run-output + a real `date` — so a later
   cycle can't be poisoned by a fabricated prior result (the 2026-06-03 fabricated-timestamps scar).

## 3. The anti-NARROW-MINDEDNESS system (breadth, not tunnel-vision)
Narrow-mindedness = the agent grinds one path and never sees the better one. Defeated structurally:
1. **The n±k lattice is mandatory.** Every objective seeds not just the primary path (+n) but **−k FALSIFIERS**
   ("is the method/apparatus even sound?" — often the highest-value node) and **+k GENERALIZATIONS** ("what is
   the general class / a higher-leverage framing?"). The frontier template ships with an n2 falsifier + n3
   diverge node by default. The Stop hook's per-cycle protocol *requires* pushing a −k and a +k neighbor.
2. **Value-needle / pivot-up rule.** If 3 cycles in a row only move low-value nodes, the loop must pivot UP an
   order (to a +k generalization or a −k reframe) or stop — it cannot keep mining a dead vein (the over-mining trap).
3. **Completeness-critic node.** Periodically a node asks "what modality/approach did we NOT try? what claim is
   unverified?" — what it finds becomes new frontier nodes. (Tree-of-Thoughts breadth, made explicit.)
4. **Multi-modal fan-out for discovery.** Discovery nodes route to a parallel scout fan-out (different search
   angles), each blind to the others — one angle can't monopolize.

## 4. Skills/agents PER LOOP (the routing — and why it's not "always spawn a swarm")
The loop routes each node by KIND (this is the `AF_BL_RF` routing function, now driving the loop):
| Node kind | Pattern | Why |
|---|---|---|
| `build` (coupled coding/apparatus) | **single-agent** (the cycle itself) | multi-agent is *worse* on tightly-coupled work + 15x cost |
| `verify` | **adversarial panel** (≥2 skeptics prompted to REFUTE; majority kills) | confidence; catches false victory |
| `diverge` / discovery | **fan-out scouts** (≤9 parallel, different angles) | breadth; ~90% better on breadth-first |
| `decide` (promote/deploy/fork) | **`decide` BULL/BEAR/NULL**, default NULL under ambiguity | asymmetric loss; auditable |
| reversal / whiplash | **mandatory ≥2-skill consensus** | a lone agent is unreliable on reversals |
Drift/over-spawn guard: a `build` node must NOT fan out (cost trap); a `verify`/`reversal` node MUST.

## 5. The self-improving loop (compounding across cycles AND sessions — Weng's reflection + Algorithm Distillation)
READ-FORWARD memory/dead-list/reusable-assets at every cycle start (never re-mine a refuted vein, never rebuild
an existing tool); WRITE-FORWARD every learning as it happens; fold user/panel feedback into the operating model
immediately. The durable memory IS the improving agent — any one session is just its current step. MONOTONIC:
refuted veins stay closed, validated tools accumulate, the honest-number ledger tightens.

### 5a. Harvest protocol -- the reusable-asset register (MECHANICAL, not aspirational)

The "reusable-asset register" mentioned in AUTONOMOUS_RUNNER.md §5 is implemented at:
  - Registry:   runs/autonomy/skill_library/INDEX.json  (JSON, atomic writes)
  - Manager:    scripts/autonomy/skill_library.py       (Python API + CLI)
  - Seed:       scripts/autonomy/_seed_skill_library.py (re-run after new assets land)

**harvest node KIND** -- after any CONFIRM / verified-build step, abstract the artifact into
a named reusable asset and register it immediately (the WRITE-FORWARD contract):

```python
from scripts.autonomy.skill_library import register
register(
    name="my_tool",              # short unique id
    kind="tool",                 # tool | probe | harness | engine | gate | dataset
    path="src/.../my_tool.py",   # repo-relative
    entrypoint="main_fn",
    signature="(df: pd.DataFrame, ...) -> dict",
    summary="1-2 sentence plain-English description",
    tested_on="ASSET CADENCE; RWYB date",
    provenance_sha=git_sha,      # subprocess.check_output(["git","rev-parse","HEAD"])
    tags=["tag1", "tag2"],
)
```

**read-forward at every cycle start** -- call digest() before mapping the n+-k lattice so that
reuse-before-build is checked MECHANICALLY (not from memory):

```python
from scripts.autonomy.skill_library import digest
print(digest(n=10))          # or: digest(query="oracle validation")
```

CLI equivalents:
```
python scripts/autonomy/skill_library.py digest
python scripts/autonomy/skill_library.py list
python scripts/autonomy/skill_library.py search <query>
```

The harvest discipline closes the gap between "experience compounds" (AUTONOMOUS_RUNNER §5 prose)
and "an agent actually reuses the probe it built last session" (mechanical). A CONFIRM without a
harvest call leaves the asset invisible to the next cycle -- that is a monotonicity violation.

## 6. The "solved on return" contract (and its honest limits)
**Guarantee:** given a well-formed objective + success_criteria + budget, the loop runs to one of:
`SOLVED` (success_criteria *verified*) · `BUDGET_SPENT` · `GENUINELY_BLOCKED` (needs an external state/your
decision — parked with a wake-condition) · `REFUTED` (objective shown infeasible, with the falsifier). It will
not silently stop, drift to a proxy, declare false victory, or tunnel — those are each fenced above.
**Honest limits (no fantasy):** (a) per-cycle judgment quality is still the model's — the loop guarantees
*persistence + breadth + verification*, not genius; (b) a bad objective function yields confident wrong work —
**the objective is the one thing only you can get right**; (c) cost scales with cycles (the budget cap bounds
it); (d) trust-critical writes still route through the human-review fence (sandbox→review→push) — autonomy
operates *inside* a mechanical fence, by design, for real-capital safety.

## 7. How to run it
```
cp scripts/autonomy/frontier.template.json runs/autonomy/frontier.json   # then edit: objective, success_criteria, seed nodes, budget
touch runs/autonomy/AUTONOMY_ON                                          # arm the loop (the master switch)
# in-session: the Stop hook now keeps the session going node-by-node until a real stop-condition.
# cross-session/durable: python scripts/autonomy/autonomy_driver.py --max 40   (or schedule via cron)
rm runs/autonomy/AUTONOMY_ON                                            # disarm anytime
```
Wiring: `settings.json` gets a `Stop` hook → `python .claude/hooks/autonomy_loop.py` (added during integration).

**Worked example (your own goal, demonstrated 2026-06-05).** Seed the frontier with the objective + its
*verifiable* success criteria, and the n±k breadth nodes:
```json
{ "objective": "Develop a SOL spot system aiming for 1-5% ROI per fire, robust on UNSEEN.",
  "success_criteria": "ship-tier SOL setup: 10/10 seeds positive UNSEEN, p05>0, maxDD<30%, beats a cost-matched
                       random-entry null on EVERY window + a beta-matched static, per-fire ROI in [1%,5%] net of
                       taker cost, mechanism+falsifier verified",
  "value_floor": 0.25, "budget": {"spent":0,"max_cycles":30},
  "nodes": [
    {"id":"s1","ev":0.9,"kind":"build",  "task":"characterize SOL ground-zero: per-fire move dist + cadence break-even (is 1-5%/fire harvestable after cost?)"},
    {"id":"s2","ev":0.85,"kind":"verify","task":"-k FALSIFIER (do FIRST): is the apparatus sound? taker cost / working DSR gate / random-entry null / bear holdout"},
    {"id":"s3","ev":0.5, "kind":"diverge","task":"+k GENERALIZE: map the avenue (chart x timeframe x conditioner x exit) before tunneling on one path"},
    {"id":"s4","ev":0.2, "kind":"decide", "task":"route to decide BULL/BEAR/NULL once s1-s2 land"} ] }
```
The driver dry-run picks `s1` (top-EV), injects objective + success_criteria + the anti-drift protocol, and the
loop proceeds: build → verify-the-apparatus-first → generalize-for-breadth → decide — exactly the discipline that
stops narrow-mindedness and drift. It runs until `success_criteria` is *verified* (SOLVED), budget spent, or a
genuine block. **That is "give it the problem + allocate time + it's solved (or honestly bounded) on return."**

## 8. Map of Weng's 4 limitations → the mechanical defeat
| Weng limitation | Defeat in this harness |
|---|---|
| Finite context (info competes for bandwidth) | externalize plan+memory to `frontier.json`/`memory/`; each cycle loads only what it needs |
| Long-horizon planning (can't adjust on errors) | the re-rankable frontier + pivot-up rule + park-on-block; replanning is the default, not the exception |
| NL-interface unreliability (skips instructions) | **mechanical hooks** (Stop/PreToolUse) instead of prose — the whole point |
| Weak trial-and-error robustness | the verify-gate + adversarial panel + monotonic memory (learns from each refutation, never re-pays) |

**Bottom line:** this is the agent you imagined — self-directing (EV frontier), self-improving (compounding
memory), breadth-seeking (n±k + completeness-critic), drift-resistant (objective re-anchor + verify-gate), and
mechanically persistent (the Stop hook) — built entirely in the scaffolding you control, not the brain you can't.
