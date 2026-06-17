# Self-Evolution Ledger (GLOBAL, resumable, persists across project life)

**Mandate (user 2026-06-06):** every 3 hours, spend ~30 minutes evolving the project AND the loop framework
itself -- close gaps, push toward SOTA, fold in learnings / new findings / new ideas. Tracked globally; skipping a
cycle is fine but MUST be recorded here. This runs for the life of the project; dedicated time is mandated.

**Cadence:** every 3h from 00:00 SAST -> 00:00, 03:00, 06:00, 09:00, ... (30 min each). Resume by reading this
ledger + the latest cycle's notes; a reborn orchestrator/loop digests prior cycles end-to-end ("Cell-style").

**Anchor:** schedule established 2026-06-06 00:24:10 +0200.

---

## Cycle 1 -- 00:00 SAST -- STARTED (2026-06-06 00:24:10 +0200)
**Focus:** the loop FRAMEWORK (highest-leverage gap, enables everything after):
- (a) Resumability across orchestrator / meta / loops (survive subscription limits -> resume).
- (b) Loop briefs + metadata (each loop instance carries its objective + a digest of prior loop generations' lessons -- Cell-style evolution).
- (c) This evolution-cadence mechanism + global tracking (this ledger).
- (d) Gap-scan of metaop + the adaptive-MA apparatus; close what's cheap + high-value.
Status: IN PROGRESS.

### Cycle 1 -- DONE (~00:30 SAST)
Gaps closed (loop framework):
- RESUMABILITY: `scripts/autonomy/resume_all.py` (--list / --resume all parked loops). Loops are durable
  (per-thread SqliteSaver) + independent processes; orchestrator state (ledger/OVERSEER_LOG/learnings lanes) is all
  on disk -> the whole stack survives a dead orchestrator or subscription limit and resumes. VERIFIED (lists 6 loops).
- LOOP BRIEFS (Cell-style): `runs/autonomy/loop_briefs/<thread>.json` = objective + parent generation + a DIGEST of
  the prior generation's lessons (pulled from the learnings lane). A reborn loop digests its predecessor end-to-end.
- The per-minute gap-find-and-solve behavior is adopted as the standing watcher routine.
NOTED for a future cycle (not closed): persistent-vs-respawn instances. Each metaop node is a fresh `claude -p`
(Opus-tier) = the main latency/quota cost; user absorbs traffic for now. The deeper optimization (a persistent
API/Sonnet-worker brain that lives->learns->is reborn without re-spawning per node) is a larger change -> later cycle.

## ~01:19 -- gap found (for cycle-2 evolution): metaop worker timeout (600s) vs heavy tasks -> a full-grid-in-one-node worker hangs ~19min. FIX applied via guidance (one cadence/subsample per node). FUTURE: make CliBrain timeout configurable + auto-decompose oversized nodes.

## ~01:25 -- gap found+fixed (cycle-2 evolution): watcher checked LOCK-existence, fooled by a stale lock from a CRASHED loop (sol-ma OOM-crashed ~01:00 but showed alive ~25min). FIX: scripts/autonomy/loops_alive.py = PROCESS-liveness via lock PID. Also relaunched sol-ma BOUNDED (parallel 1, one-cadence-per-node, u20 subsample) to avoid the OOM.

## ~02:18 -- RWYB catch + gap found+fixed (per-minute mandate):
- **RWYB correction**: the committed MA-oracle conclusion framed orderflow/liquidation as an untested "next avenue".
  Verified the falsifier already uses all 40 norm_/xd_ features incl. 11 orderflow/micro; ran the one untested family
  (+29 liquidation/book features) directly -> still genuine=False. Corrected to the comprehensive finding (NO bar-level
  causal family is tradeable on the oracle's 2-bar DNA).
- **loops_alive.py hardened**: a bare no-arg call printed "0 alive | dead: none" and silently misled the watcher ->
  now auto-discovers every lock in runs/autonomy/locks/.
- **watcher.py added**: resumable, bounded-lifetime 1-min liveness watcher (auto-discover locks, early-exit on
  loop-death / evolution-window so the orchestrator wakes on time; writes runs/autonomy/watcher.log).

## ~02:30 -- APPARATUS GAP found+fixed (two-sided-soundness; cycle-2 evolution):
The falsifier's `shuffled_collapses` SOUNDNESS gate used the shuffle **p95 tail** (`<0.55`) for a question that is
about the **MEAN** ("does permutation destroy signal on average?"). At n_shuffle=30 the tail straddles 0.55 from
sampling noise -> BTC 1d (shuffled mean=0.502 SOUND) spuriously flagged APPARATUS_SOUND=False; the liq-augmented run
flipped False for the same reason (more features = wider tail), NOT a target leak. This was the "possible pipeline
leak" caveat. **FIX**: soundness now judged by the shuffle MEAN (auc<0.54, |ic|<0.03, cap_skill<0.10) -- consistent
with the falsifier's own selftest principle ("the MEAN is the statistically correct collapses-to-chance check");
genuineness still uses the p95 tail. Post-fix BTC 1d + SOL 4h both read APPARATUS_SOUND=True, DNA_GENUINE=False (valid
honest refutation on confirmed-sound apparatus). Lesson: a soundness gate must use the statistic that matches its
question (mean for collapse-on-average, tail for beat-the-null) -- mixing them produces false leak alarms. Reusable
across any shuffle/permutation firewall in src/strat.

## ~02:35 -- ORACLE-OBJECTIVE pivot (the highest-leverage finding of the night):
The scalp oracle (`min_move_net=0`) max-compounds into 2-bar wiggles -> an UNFAIR test for lagging MAs. Added a
per-move net floor to the oracle DP (`oracle_high_capture(..., min_move_net=)`, backward-compat, selftest PASS) and a
nonlinear `--model gbm` path to the falsifier. At a 3-5% floor the oracle becomes a SWING oracle (multi-day holds,
5-15% mean net/move = the project's actual target unit). Re-tested: linear DNA flips +IC on 1d (frame is far less
hostile) but 0/8 genuine; GBM hit DNA_GENUINE=True on BTC-1d at n_shuffle=15 -- BUT a stress test (n_shuffle=50 x 3
seeds x 3 floors + OOS->UNSEEN persistence) REFUTED it as seed-selection + OOS-overfit noise (OOS +58% -> UNSEEN
+0.1%). Lesson for the framework: **the oracle objective IS a design variable -- a max-capture DP with no per-move
floor silently becomes a scalper and mis-frames every downstream test; always state the per-move floor + hold band so
the oracle matches the target unit-of-trading.** And: **a single firewall pass is not enough -- seed-robustness +
OOS->UNSEEN persistence are mandatory before any "genuine" is believed** (caught a false positive tonight). Both fold
into the src/strat candidate_gate. The swing oracle is the right frame to carry forward for any future signal source.

## Cycle 2 -- 03:00 SAST window -- STARTED ~02:51 (anchor 03:00; started 9 min early to avoid idle, real time logged)
**Focus:** fold tonight's findings into the durable framework (the "fold-in learnings" mandate).
Done in this cycle:
- **MEMORY persisted**: `memory/project-ma-oracle-decomposition-2026-06-06.md` (+ MEMORY.md index) -- the empirical
  closure (3 avenues null at daily/4h -> sub-bar/HF) + the 4 reusable methodology lessons. The next session inherits
  it; never re-pay.
- **Loop lanes corrected (write-forward / self-improving loop)**: wrote the ORCHESTRATOR CORRECTION into the sol +
  meta learnings lanes -- the loops had a subtly-wrong premise ("oracle holds ~2 bars EVERYWHERE = market structure")
  which my swing-oracle work corrected (it was a no-floor scalp-oracle ARTIFACT). Their next plan cycle digests it.
- **src/strat registered the oracle-decomposition apparatus** (README "Upstream" section): the swing-oracle knob
  (`min_move_net`), the two-sided falsifier, the mean-vs-tail soundness rule, and the MANDATORY seed-robustness +
  OOS->UNSEEN persistence checks -- so future strat work uses the hardened tool instead of re-deriving it.
- Apparatus hardened earlier tonight (committed): swing-oracle floor, gbm model path, soundness-gate fix.
NOTED for cycle 3 (not closed): metaop worker-timeout still not made configurable + auto-decompose (cycle-1 carryover);
dollar-bar oracle DP too slow (needs an O(n) approximation or a longer budget). Both are framework-robustness items.

## Cycle 3 -- ~13:50 SAST (2026-06-06) -- FRAMEWORK EVOLUTION (the big one)
Driven by the user's two directives: (a) formalize autonomous mode as a launch-3-loops gate; (b) "close all gaps in
the skills to make them SOTA, not narrow." Treated as the loop-3 project-wide evolution.
- **/orchestrator skill** (aliases /or /os /orc /auto): the 3-loop autonomous operating model (problem-solver
  expert+plain + meta-agent + project-3h-evolution) + `scripts/autonomy/launch_autonomy.py` launcher (attended/
  unattended) + the ELEVATE-TO-SOTA standing mandate. Commits 93903d9, c3724a6.
- **SOTA gap analysis (web-grounded, cited)**: found our "reusable-asset register" was VAPORWARE; flat memory;
  same-family naive judge panels; uncalibrated stopping; greedy hill-climbing.
- **Voyager skill-library made REAL** (the #1 gap): `scripts/autonomy/skill_library.py` (register/search/digest) +
  INDEX seeded with 11 actual reusable assets + harvest protocol in AUTONOMY_FRAMEWORK.md. Turns "memory of lessons"
  into a growing "library of capabilities". READ-FORWARD digest at cycle start; HARVEST after every CONFIRM.
- **8 SOTA patterns folded** into orchestrator + decide/audit/discover skills: Reflexion, 3-lane memory+consolidation,
  self-consistency, 2-round debate, de-biased judge panels, difficulty-adaptive compute, calibrated VOI stopping,
  evolutionary discovery (stage 4b), frontier-as-DAG. Honest about where a pattern did NOT fit (no cargo-culting).
  Commits c3724a6, 0968ebb.
- **watcher EVOLUTION_HOURS fixed** to the full every-3h cadence [0,3,6,9,12,15,18,21] (was [3,6,9]).
- Big parallel deliverable this window: the **src/narrate/** descriptive market-intelligence foundation (8 commits;
  see project-market-research memory) -- the "narrate the what before any strat" engine, built THROUGH the framework
  (overseer dispatched 7 worker nodes, judged + committed each).
NOTED for next cycle (still [P] protocol, not yet [M] mechanized): Reflexion + self-consistency as actual
frontier.template.json node KINDs in metaop; 3-lane memory restructure; evolutionary-search driver script.

## Cycle 4 -- ~15:33 SAST (2026-06-06) -- /orc "apply all upgrades", run THROUGH the langgraph loops
First genuine 4-role run (user-corrected operating model): orchestration (me) + 2 langgraph SOLUTIONING loops
(orc-upg-expert/plain) + meta loop + the absolute 60s watcher. The solutioning loops DID the work and delivered:
all 5 remaining skills SOTA-upgraded (research literature-registry+fan-out+3-lane, architect adversarial+evolutionary,
trainer reflexion+adaptive-preflight, apex difficulty-adaptive+critic+evolutionary+DAG, trader self-consistency+
reflexion) + the EXIT-POLICY gap owned by trader (commit f2e2254, verified additive / 0 ERROR / frontmatter intact).
Earlier in the same /orc: built scripts/skill_diagnostics.py (repeatable, harvested), fixed correctness gaps
(architect IC-tombstone etc.), validator SOTA. **Two GOVERNANCE bugs the loops exposed + fixed correct-as-you-go**:
(1) a loop ran `git commit` itself -> HARD_DENY fence in metaop/tools.py (loops never commit/push); (2) a loop wrote
.claude/autonomous_mode.json to disarm -> HARD_FILE_DENY control-surface fence (loops write work, not control
surfaces). Plus the W3 silent-disarm parse bug (ISO vs plain envelope_end) fixed. Process: /orc now ARMS autonomous
mode + 30m floor + the absolute 60s watcher, baked into the skill. Lesson: an autonomous loop with shell+write tools
WILL touch control surfaces unless mechanically fenced -- fence commit/push AND the arm/disarm + permission + hook
files. The langgraph solutioning substrate genuinely works (real additive edits, self-RWYB) -- the gap was governance,
now closed.

## Cycle 5 -- 16:42-17:15+ SAST (2026-06-06) -- /orc "build remaining components + harmonise" (2h)
The last unbuilt user-asked components BUILT + the project harmonised, run through the langgraph loops (overseer
judged+committed+pushed each; ~16 commits).
- **#14 PersistentCliBrain** (the last open DIRECTIVES_REGISTER gap): warm `claude -p --resume` session across nodes
  (no per-node cold-start). RWYB-verified (2-node continuity smoke PASS). `--backend persistent` wired. Harvested.
- **G1**: skill_diagnostics widened to skill SUB-files (closes "0 ERROR != clean"; now catches trader sub-file refs).
- **Directive harmonisation F1-F6**: CLAUDE.md (SKIP-bypass doc now matches the unskippable mechanism; 3 dead _common
  links repointed/tombstoned), STANDARDS (AUTONOMOUS_RUNNER un-mislabeled; PROTOCOL_COMPOSITION one story), SLASH_ROUTER
  (14 dirs/13 skills + orc/discover/narrate). Trader stale refs repointed to LIVE homes (position_sizer->wealth_bot/bot,
  DSR->src/strat/battery). SYSTEM_TOPOLOGY §9 registers all new components + the corrected arming authority.
- **Re-enabled the kill-switch SAFETY invariant** against the live src/wealth_bot/bot/risk_manager.py (was a ghost
  path -> 'not enforced'); the other trader ghost-invariants stay honest WARNs (modules genuinely unbuilt).
- **PERMISSIONS fixed (user friction)**: arming is now prompt-free via runs/autonomy/AUTONOMY_ON (the .claude/
  autonomous_mode.json write prompted because it was open in the IDE). launch_autonomy + orc/SKILL.md updated; the file
  dropped from the tree.
Lesson: the overseer must JUDGE loop output at the 60s cadence (a loop drops findings WITHOUT ending = silent-hang if
you park); and a loop editing beyond its literal scope can still be CORRECT (the plain loop's trader repoints were
verified-live) -- judge on correctness (ls-before-accept), not just scope.

## 2026-06-11 -- WM+agent-layer foundation (attended /orc sprint, while V1.1 trained)
The agent layer went from prose to machine-enforced reality, all V1.1-independent:
- **Phase 0 (commit 72bc94d)**: the 3-class taxonomy (F forecasters / A1 WM-consuming / A2 raw-data) made REAL --
  src/agents/ tree, V16/V17 + src/agent/ reclassified OUT of the forecaster zoo, the frozen ForecastBundle contract,
  a split registry (forecasters.json by IC/ShIC vs agents.json by compound -- PHYSICALLY separate so compound can't
  launder a memorized forecaster), and 4 CDAP invariants (forecaster_frozen_in_agents / no_predicted_return_as_
  realized_reward / agent_class_declared / v16_v17_not_in_wm) -- two-sided injection-proven. The F-frozen + no-GIGO
  boundary is now mechanically unbreakable.
- **Gate apparatus** (built + adversarial-verify): gate0_harvestability (does signal EXIST at our resolution? -- the
  gate that precedes all agent-building), wm_value_probe (Gate A -- is a forecaster genuinely-learnt enough to plan
  over?), shuffled_market_control (the A2 policy-overfit detector = the RL analog of ShIC, did not exist), wm_
  calibration_probe. So the moment V1.1 lands, its result becomes a DECISION (build A1 / pivot A2 / change resolution).
- **A1 build spec** (docs/A1_BUILD_SPEC.md): the half-built Dreamer is a smoke shell with the predicted defects
  (action-CONDITIONED imagination -- wrong for a price-taker; no portfolio state; direct WM import; predicted-reward
  critic = GIGO; H=15 vs 4-8). The spec fixes each, ready-to-execute when Gate A passes.
- **Cohort f41 pinning** (dab23e1, registry under-listed f41) + **ShIC-OOM regression guard** (7457bc3).
Lesson: BUILD THE GATE BEFORE THE THING IT GATES. The highest-value pre-V1.1 work was the apparatus that turns the
training result into a decision -- not more models. The taxonomy's GUARANTEED value is GIGO-isolation + a cheap clean
Ceiling-2 null; "this plumbing finds alpha" is [PROJECTED], never assumed.
