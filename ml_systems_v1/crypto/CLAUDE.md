# V4 Crypto System — Invariants

> **🟢 MARKET-DECOMPOSITION HARNESS = THE CANONICAL RESEARCH + WORKING-MODEL LAYER (2026-06-09, BINDING).**
> The project has ONE repeatable, market-agnostic working model — **research/decomposition → mining → engine →
> strat → bot → execution → deployment** — and ONE consolidated research layer. Any market analysis, decomposition,
> mining, or strategy work MUST consult this FIRST (do not re-mine a refuted vein, do not scatter findings):
> - **The working model (the 7 gated stages):** [`docs/SOLUTIONING_PIPELINE.md`](docs/SOLUTIONING_PIPELINE.md).
>   Tool: `python -m framework.pipeline {init|record|gate|advance|status|registry|doctor} <market> <instrument>`
>   ([`src/framework/pipeline.py`](src/framework/pipeline.py)); market-agnostic via the `MarketAdapter` contract
>   ([`src/framework/adapter.py`](src/framework/adapter.py) + [`src/framework/crypto_adapter.py`](src/framework/crypto_adapter.py)).
>   SOTA-fortified (machine-checked gates + reproducibility lineage + run-registry + crash-safe store); it self-tests
>   (`python -m framework.selftest`, 14/14) and is self-policed by CDAP (`check_invariants.check_framework_store()`).
> - **The consolidated research layer (STAGE 00):** [`docs/MARKET_FRAMEWORK/`](docs/MARKET_FRAMEWORK/) — the
>   **01_DEAD_LIST** (63 refuted theories — read before proposing ANYTHING), 02_APPROACH_LEDGER, 03_METHODOLOGY,
>   04_MARKET_MODEL, 05_OPEN_THREADS — plus the companion theory docs
>   ([CRYPTO_MARKET_UNDERSTANDING](docs/CRYPTO_MARKET_UNDERSTANDING.md),
>   [CHIMERA_FEATURE_DICTIONARY](docs/CHIMERA_FEATURE_DICTIONARY.md), [STRATEGY_PLAYBOOK](docs/STRATEGY_PLAYBOOK.md)).
> - **The decomposer/viewer:** `python src/mining/decompose.py --asset <SYM> --cadence <TF>` (descriptive — view all
>   chimera feature behaviours for a chosen asset/period/cadence; NOT a signal hunt).
> - **The store (single source of truth, never disparate):** [`workspaces/`](workspaces/) +
>   [`workspaces/REGISTRY.md`](workspaces/REGISTRY.md). Record artifacts in the workspace stage; advance only when the
>   gate passes; `doctor` keeps the store honest. Audit details: [`docs/PIPELINE_SOTA_AUDIT_2026_06_09.md`](docs/PIPELINE_SOTA_AUDIT_2026_06_09.md).
>
> Provenance: 2026-06-09 user mandate to build a repeatable working model + consolidate every prior idea/approach into
> one place so the next instance never re-mines a dead vein or scatters findings. STATE.md §"Working model" mirrors
> current stage-state.

> **🟢 SLASH-COMMAND FUZZY ROUTER (2026-05-22)**. When the user types `/oarcle` /
> `/auditr` / `/dialetic` (typo), apply the closest-neighbor protocol in
> [`.claude/skills/_common/SLASH_ROUTER.md`](.claude/skills/_common/SLASH_ROUTER.md).
> Auto-invoke at similarity ≥ 0.90 with a one-line correction note; ask-once
> default-yes at 0.70-0.90; explicit ask at 0.55-0.70; treat as plain text at
> < 0.55. Matcher primitive: `python scripts/fuzzy_slash_match.py <token>`.

> **🟢 WALL-CLOCK GROUNDING (2026-05-22; r2 2026-06-03 + ELAPSED-TIME clause)**. Every directive that makes a
> wall-clock claim (date, ETA, "last modified", "X days ago", **elapsed / "X hours in" / "time left"**) MUST ground
> against verified time per
> [`.claude/skills/_common/WALL_CLOCK.md`](.claude/skills/_common/WALL_CLOCK.md).
> Tag claims VERIFIED / REPORTED / INFERRED. **You have NO internal clock — elapsed time is unknowable by feel and
> MUST be measured: elapsed = (fresh `date`) − (VERIFIED start); show the subtraction. In autonomous/timed/`/loop`
> runs, record the VERIFIED start once and re-`date` before every progress/elapsed claim and at the start of each
> cycle; every learnings-ledger timestamp is a real `date` reading (write "C7" with no time rather than invent one).**
> The 2026-05-21 → 2026-05-22 session-mid-rollover is the canonical date-drift trigger; the 2026-06-03 "claimed ~5h
> in, was ~1h12m + fabricated a ledger of invented timestamps" incident is the canonical ELAPSED-time trigger.

> **🟢 AUTONOMOUS-RUNNER PATTERN (2026-05-22, REWRITTEN 2026-05-30 r7 — n±k + build→run→learn→pivot; r8 2026-06-02 + §5 self-improving loop)**.
> When a directive receives an autonomy mandate ("go for 3 hours", "no consult", "agentic work",
> `/loop`, `/schedule`), follow
> [`.claude/skills/_common/AUTONOMOUS_RUNNER.md`](.claude/skills/_common/AUTONOMOUS_RUNNER.md):
> **turn-1 maps the n±k OBJECTIVE NEIGHBORHOOD** (primary n, plus foundational −k = is the
> method/data/spec sound? + derived +k = adjacent solutions / the general class, each with an
> OPPORTUNITY+ and a FALSIFIER− valence) + runs **GOAL_BOUNDS pre-flight** (budget, per-node value
> floor, wall-clock anchor, stop conditions). Then execute the **build→run→learn→pivot cycle** over
> an **EV-ranked frontier**: learning expands the lattice, the frontier re-ranks every cycle, pivot
> on diminishing-returns/refutation. **Value-needle test** each cycle; destructive ops STILL require
> user OK. **IDLE-STOP (supersedes the old "time-utilization mandate"): "use the remaining time" is
> NOT an objective — park blocked nodes with a wake-condition, and STOP rather than tick once the
> frontier is empty or below the value floor. Honest early-stop > busywork (burning the clock = the
> over-mining trap).** honest-failure clause (no silent target reframing); self-summon `decide`/oracle
> at 25/50/75% + on whiplash/plateau; one PushNotification per state-change only. **§5 SELF-IMPROVING LOOP
> (added 2026-06-02): experience compounds across cycles AND sessions — READ-FORWARD prior memory /
> dead-list / reusable-asset register at every start (seed the lattice from accumulated knowledge, never
> re-mine a REFUTED vein), WRITE-FORWARD every LEARN as it happens, fold user/oracle FEEDBACK into the
> operating model immediately + persist it; the durable memory IS the improving agent — never re-pay for a
> lesson already learned (MONOTONIC).** (The old goal-tree + PROTOCOL_COMPOSITION ceremony is archived; the
> lattice subsumes it.)

> **🟢 PROTOCOL_COMPOSITION (folded into AUTONOMOUS_RUNNER 2026-05-30)**. The cross-skill
> summoning contract is now part of [`AUTONOMOUS_RUNNER.md`](.claude/skills/_common/AUTONOMOUS_RUNNER.md)
> §4: the goal-tree schema is subsumed by the n±k objective-neighborhood lattice; signal-triggered
> self-summons survive (posterior plateau / whiplash / goal-stretch → `decide` or oracle; new-domain →
> `research`); the value-needle test runs every cycle. (Old standalone spec archived.)

> **🔴 UNIVERSAL PRE-DELIVERY SELF-AUDIT + MAXX PER-COMMIT GATE (2026-05-26, BINDING for all agents/skills)**. User mandate verbatim: *"add the self-audit clause always (for all agents and directives). The only thing with MAXX (and probably should extend to other instances) where they have per-commit audit gate so that anything is fixed before committing (be very careful here because an instance might not commit but might deliver work)"*. **Two-layer contract**: (1) Every agent in every skill MUST run a `pre_delivery_self_audit` block before returning ANY deliverable (commit OR non-commit: analysis, scout findings, recommendations, dialectic positions, etc.) covering claim-tagging / look-ahead / harness compliance / gate-spec consistency / repro block / asymmetric loss / synthesis-vs-real-data caveats. Spec at [`.claude/skills/_common/STANDARDS.md`](.claude/skills/_common/STANDARDS.md) §6 (self-audit; relocated there from the reset-archived OPERATIONAL_DIRECTIVES — harmonised 2026-06-06 per DIRECTIVE_GAP_AUDIT F2). (2) MAXX (and any coordinator skill) MUST run a `per_commit_audit_gate` BEFORE committing worker-derived work to master: verify all upstream workers reported self-audit PASS, run Pattern S/T grep, run repro-block check; if any fail, do NOT commit (dispatch focused auditor on worker output OR fix inline OR return to user). The per-commit audit gate is now realized as the OVERSEER-commits-after-review model + the loop-commit fences (the `maxx` skill was consolidated into `apex` 2026-05-28); the two-layer rule binds inline above regardless of its spec home. Provenance: 3 gap-window incidents in INST-F96BE75A 2026-05-25/26 (R46 E46_1/E46_3 inflated by close-only breach; R51 E51_1 inflated by gap-down fallback miss + gate-spec drift; R54 A54_2 PSEUDO-VB forward-close leak) — in each case META reported wrong numbers to user before post-hoc auditor caught the defect. Two-layer contract closes both worker-commit step AND META-commit step. Full background at `docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md` §F.8 lesson #6a.

> **🔴 LAYER-2 INVARIANT (2026-05-19, REVISED 2026-05-20 PM): 2ND-PASS RED-TEAM AUDIT — RUN INTERNALLY, DO NOT DISPLAY.**
>
> **Rule A (RUN INTERNALLY)**: For every substantive response (claims, numbers, methodology, code changes, analysis), mentally run the 2nd-pass red-team check before answering. Look for: empirical errors, magnitude inflation, methodological gaps, stale numbers, look-ahead, leverage drift, multiple-comparisons. The check still happens — silently.
>
> **Rule B (DO NOT OUTPUT)**: Do NOT include a visible `## 🔴 RED-TEAM 2nd PASS` section in the response. The user finds the section padding-heavy. Trust the cognitive check; don't display it.
>
> **Rule C (ACTION ON CRITICAL ONLY)**: If the internal check surfaces a 🔴 CRITICAL flag (claim invalid / number wrong / methodology broken), fix it inline in the response or flag it briefly in a single sentence ("Caveat: number X depends on Y — verify before deploy"). Don't list 🟡 MEDIUM / 🟢 LOW flags unless the user asks. Don't pad with self-doubt.
>
> **Rule D (ON-DEMAND DISPLAY)**: If the user asks "what could be wrong?" or "give me the red-team take" or similar, THEN display the full audit. Otherwise keep it internal.
>
> Composes with: CDAP, DOUBLE_AUDIT_PROTOCOL, LAYER-1 RWYB. Provenance: 2026-05-19 mandate was for catching real bugs (V2 +468% inflation). The cognitive discipline stays; the visible section was generating padding flags ("LOW: vibes estimate") that didn't change user actions. User directive 2026-05-20 PM: "remove the 2nd pass audit, it's annoying — do it internally."

> **🟢 NORTH STAR (2026-05-17 PERMANENT)**: Mission + ROI target + empirical premise documented at **[PROJECT_NORTH_STAR.md](PROJECT_NORTH_STAR.md)**. READ FIRST every session — stops the loop of re-establishing goals. External/published benchmarks are NOT our concern; our empirical opportunity premise (95%+ of 2025-2026 days have ≥1 asset moving ≥5%; 20 movers/day avg) is what binds us. +1-5%/d daily ROI is the user-mandated target under LO+spot+lev=1; methodology is the open variable. Current deployable: **(archived 2026-06-04 — no verified active alpha post-reset; the prior "REGIME_ROUTER +20.25%" figure was apparatus-inflated. Re-establish under the current sound apparatus before citing, per MEMORY.md + the A/B/C fork.)**
>
> **🔴 WEALTH-BOT TRUST STACK (2026-05-25 PERMANENT, defence-in-depth for real-capital safety)**: every ship-tier wealth-bot candidate audit JSON MUST conform to the canonical claim contract in [`src/wealth_bot/framework/claim_contract.py`](src/wealth_bot/framework/claim_contract.py) (required fields: `per_trade_returns_sorted_desc`, `top_3_pct_of_compound`, `jackknife: {K=0..K=5}`, `combined_K2_plus_S9_pct`, `mechanism_falsifier_check.verified_by`, `sample_size_discipline.passes_stressed_gate`). CDAP [`src/audit/check_wealth_bot_claims.py`](src/audit/check_wealth_bot_claims.py) halts commit (exit 2) on violation. **Sample-size discipline applied to `min(baseline_compound, combined_K2_plus_S9)`, NOT baseline alone.** Mechanism-claim falsifier required when `top_3_pct_of_compound > 70% AT n_unseen < 30`. Provenance: 2026-05-25 INST-A's P4_route_basis_pos_only mechanism claim ("filter strips top-tail trades") was empirically FALSE — filter kept top 3 and dropped diversifying ones. **6-layer defence-in-depth**: (1) CDAP `check_wealth_bot_claims.py` pre-commit exit 2, (2) `claim_contract.py` required fields at write-time, (3) auditor brief items 8-12 in [`runs/coordination/AUDITOR_FINDINGS_2026_05_25.md`](runs/coordination/AUDITOR_FINDINGS_2026_05_25.md), (4) memory patterns Q + R in [`memory/fix_logs/INDEX.md`](memory/fix_logs/INDEX.md) + `feedback_mechanism_verification.md`, (5) the turn-end self-check (PROTOCOL_COMPOSITION's TURN_END_CHECKLIST was archived 2026-05-30; its rules folded into [`AUTONOMOUS_RUNNER.md`](.claude/skills/_common/AUTONOMOUS_RUNNER.md)), (6) cross-instance handshake script-prefix register in [`runs/coordination/HANDSHAKE_2026_05_25.md`](runs/coordination/HANDSHAKE_2026_05_25.md). Framework rule SR1.3 binding in [`docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md`](docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md). Real-capital safety: silent inflated headlines previously shipped through three artifacts (leaderboard, learnings, closure decision) before audit caught it; trust stack ensures next time the commit physically cannot land.
>
> **🟢 OBJECTIVE FUNCTION (2026-05-24 user mandate, PERMANENT)**: **Optimize for WEALTH (held-out compound return), NOT Sharpe or any other metric.** Pure returns and pure ROI is the goal. Strategy must be ROBUST (10/10 seeds positive on UNSEEN, block-bootstrap p05 > 0, max DD < 30%). Sharpe is nice-to-have, not the target — a bot with Sharpe 1.5 / compound +70% beats a bot with Sharpe 2.0 / compound +50%. Full spec at [PROJECT_NORTH_STAR.md §3.1](PROJECT_NORTH_STAR.md). Provenance: 2026-05-24 INST-C clarification after the multi-cadence GATED bot (Sharpe 1.92 / compound +50%) was initially over-recommended vs 4h SOTA (Sharpe 1.45 / compound +70%). Verbatim: *"We're optimising for wealth, not sharpe or any of the other metrics. We want to build pure returns and pure ROI, nothing more, nothing less. The strategy itself has to be robust, and if it has good sharpe that's good, but we not building for that necessarily."*
>
> **🟢 WEALTH-BOT DEVELOPMENT FRAMEWORK (2026-05-25 user mandate, PERMANENT — r5.2 with TI×ASSET-fixity + Phase 4 within-TI Regime Composition + Closed-Dossier protocol + SR1/SR1.1/SR1.2 + Gold-Standard Dossier + 80% canonical exhaustion floor)**: Any new wealth-bot follows the **2-phase methodology + Edge-Pushing Protocol**. **Phase 1** = robust discovery (pick Instrument + Indicator + Approach, audit, ship verified baseline at 10/10 seeds + 4-window-positive + p05>0 + maxDD<30%). **Phase 2** = oracle-augmented refinement under the **imagine-frame** (*"imagine Phase 1 was done by another instance, not you"*): mine trade-decision context with PRE-REGISTERED thresholds + asymmetric loss (false-positive > false-negative); HONEST validation (fit TRAIN+VAL only, test UNSEEN); cross-window persistence test. **Edge-Pushing Protocol** addresses discipline-without-momentum: every baseline ships with stretch-target + pace-conversion (gap to 1-5%/d OR 1-5%/3d OR 3-5%/week); 3 consecutive Phase 2 NULL rounds (with sample growth) = SATURATION → 1-quarter ban + mandatory scope expansion (new Instrument/Indicator/Approach/Cadence); 30/60/90-day no-improvement triggers spawn expansion workers; Stretch-Goal Worker after 2 NULLs proposes top-3 expansion directions; Cross-Bot Leaderboard auto-deploys leader monthly; Failure-Mode Catalog records every refuted hypothesis to prevent re-mining; Sample-Size Discipline tightens thresholds when n<20; Wall-Clock Budgets enforce phase bounds. Full spec at [docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md](docs/WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md). Supporting artifacts: `runs/oracle/PHASE2_CALIBRATION_LEDGER.md`, `WEALTH_BOT_LEADERBOARD.md`, `WEALTH_BOT_FAILURE_CATALOG.md`, `WEALTH_BOT_WALLCLOCK_LEDGER.md`, `scripts/wealth_bot/_pace_conversion.py`. **Phase 3** = expansion (new cadence / chart-type / indicator variant / approach / instrument) when Phase 1 baseline misses all floor bands OR after 2 NULL Phase 2 rounds. **Non-linear phase bouncing** explicitly allowed: knowledge in any phase can demand back-jumps. **Reproducibility binding**: every framework run records git SHA + chimera SHA + seeds + config snapshot in output JSON; bit-exact replay verifiable. **TI×Asset dossier**: each instance maintains own dossier at `docs/dossiers/<TI>_<ASSET>__inst<X>.md`; instances are competing collaborators, learn from each other via dossier read-only sharing. **Target bands are FLOORS, not ceilings**: ≥1%/d AND ≥2%/3d AND ≥3%/week are LOWER BOUNDS — exceeding upper is good, no ceiling, keep pushing. **Layered Strategy Decomposition (r4)**: every strategy is 7 independent layers (L0 time-frame / L1 signal / L2 capture / L3 cost / L4 conditioning / L5 sizing / L6 risk) with separate KPIs — capture_rate = realized/available_move within signal-valid window is the L2 KPI, cost-FREE and capital-FREE. Compound % is the OUTPUT, not a tuning target. Every audit emits a `layer_kpis` block. **Oracle-as-Parent**: oracle owns canonical layer KPI definitions + cross-bot artifacts (`runs/oracle/{LEADERBOARD,FAILURE_CATALOG,CALIBRATION_LEDGER,LAYER_DECOMPOSITION_TEMPLATE,CROSS_LAYER_HANDOFFS}.md`). Static and ML are CHILDREN that emit per-layer audits + post cross-talk handoffs when one surfaces a pattern the other should test. Provenance: 2026-05-25 INST r1 — *"phase 1 is always as is...phase 2 is augmentation"*; r2 — *"add everything to the framework"*; r3 — *"target bands should not be the limiters"*; r4 — *"I need to know the quality of strategies at a deeper level, not just what ROI...signal, signal quality, and signal decay are one dimension and should not necessarily intertwine with position sizing"* + *"oracle acting as parent, and STATIC and ML as children — they need to speak to each other"*. Four failure modes the framework closes: (1) Phase-1-only complacency, (2) Oracle-only overconfidence, (3) Discipline-without-momentum, (4) Layer-conflation (single compound % hides 7 layers). **r5 corrections (2026-05-25)**: **(TI, ASSET) is the closed problem space** — a (PEPE, MA/EMA) dossier cannot mix MACD/RSI/Bollinger etc. Each TI gets its own dossier; cross-TI competition only at the meta-leaderboard after each closes. **Phase 3 expansion is WITHIN-TI only** (param/cadence/chart-type/filter/approach/instrument-variant); new indicator family = new dossier, never a Phase 3 axis. **Phase 4 — Within-TI Regime Composition** routes between same-TI configs by detected regime (self-regime vs external vs hybrid). **No-cross-pollination rule**: cross-talk handoffs are same-(TI,ASSET) Static↔ML only; cross-TI insights become NOTES in the new dossier when started. **Capture Rate Level 1 ceiling = within-family best** (not cross-indicator). **Closed-Dossier protocol**: after ≥6 within-TI expansion axes OR saturation, dossier closes read-only; ship candidates to cross-TI leaderboard; next instance starts a new (TI, ASSET). Worked example: (PEPE, MA/EMA) closed by F96BE75A 2026-05-25 with R12 perp +48.9% UNSEEN shipped pending 4 pre-live verification items. **§Standard Dossier Report Format (r5 SR1-SR1.2)**: every dossier renders the §1A Manifest Dimension Status Grid (the holes-detector, BREADTH) + §1A Honest Sub-Dimension Exhaustion % column (the depth-detector, DEPTH) + §1B Top-5 Highest-EV Untested Sub-Axes (decision support). §1A status legend: SHIPPED/REFUTED/INCONCLUSIVE/PARTIAL/NOT EXPLORED/N/A. SR1.1 honest exhaustion % = within-axis sub-variant coverage (variants tested / variants in scope). 3 aggregate views: simple average, weighted by historical alpha source, weighted by EV-of-remainder. Closure requires ALL dims status != NOT EXPLORED **AND** (honest exhaustion ≥ **80%** OR all floors MET). User mandate r5.1 verbatim: *"I want > 80 - 90% to consider exhaustion... canonically that should be the coverage."* The infinite search space caveat is acknowledged; 80% is the canonical operational bar. **§Gold-Standard Dossier (r5.2 2026-05-25)**: a (TI, ASSET) designated as gold standard (the canonical reference all others benchmark against) — termination shortcut DENIED, multi-sprint commitment required to hit 80% genuinely, documentation premium applies. (PEPE, MA/EMA) is the FIRST gold-standard by user mandate verbatim — *"PEPE × EMA/MA is going to be our gold standard for all work going forward, so we cannot take it for granted"*. Status: **(archived 2026-06-04 — all dossiers archived at the reset; docs/dossiers/ does not exist; the "ACTIVE 28% exhaustion" + §1B queue are ghost-state. Re-establish under the current apparatus before citing.)** All other-TI dossiers (RSI, MACD, Bollinger) deferred until gold-standard hits 80% OR user authorizes parallel start. Without SR1.1, "REFUTED" can hide "we tested 3 of 20 sub-variants" — that's partial-sample wearing a confident label. (PEPE, MA/EMA) honest exhaustion ≈ 28% mean depth despite 14/14 dims TOUCHED — proves SR1.1 catches what SR1 still hides. Replace "EXHAUSTED" language with "TOUCHED at X% sub-dim coverage". Provenance: 2026-05-25 INST r5 verbatim — *"when it comes to an indicator, you're bound by that, it's not optional... we never cross contaminate a single indicator and its asset"* + *"§E — All 14 manifest dimensions (status grid). It had holes."*
>
> **🔴 LAYER-1 INVARIANT (2026-05-17, user-mandated): RUN-WHAT-YOU-BUILD (RWYB).** Every code change MUST be run/tested with real data before commit. No silent failures allowed. Today's session exposed: Phase 2 silent-drop-72/87 (exit 0 despite 17% coverage), --passthrough leaking 25 GB to wrong dir, numba cache `<dynamic>` errors, runs_bars 'volume' typo silently skipped builds. Document run command + result in commit body. NON-NEGOTIABLE. Full rule: [memory/feedback_run_what_you_build.md](memory/feedback_run_what_you_build.md). Composes with: CDAP, DOUBLE_AUDIT_PROTOCOL, feedback_empirical_before_fix, agent_protocols/test_first.
>
> **🔴 GATE: read [docs/BROWSER_DIRECTIVE.md](docs/BROWSER_DIRECTIVE.md) FIRST.**
> Every prompt in this project carries the `@browser` routing tag. The tag binds you to the directive's continuity gates (A1-A3), solutioning gates (B1-B11, no silent failures), character gates (C1-C5), and response modes (G1-G4). Each rule has provenance from a real bug shipped earlier this project; per-rule bug histories are in [docs/BROWSER_DIRECTIVE_PROVENANCE.md](docs/BROWSER_DIRECTIVE_PROVENANCE.md). Violating a B-series rule is grounds for the user to halt and re-prompt citing the rule number. Compose with — don't replace — CDAP, DOUBLE_AUDIT_PROTOCOL, per-file `__contract__`, and auto-memory.
>
> **🔴 PROJECT INVARIANT (2026-05-14, user directive "110% capability unlocked"): Web tools (WebSearch + WebFetch) are FIRST-CLASS — behave identically to out-of-the-box Opus.** At session start, BEFORE any other work, run `ToolSearch("select:WebSearch,WebFetch")` to load tool schemas. Then use proactively: every factual claim about a technique, every API spec verification, every 2024-2026 SOTA citation, every market-state check. Failure to bootstrap is a Layer-2 violation. Enforcement in `.claude/skills/LAYER2_UNCONSTRAINED.md` (top + §"Amplified Capabilities → Research") and at the top of every `.claude/skills/*/SKILL.md`.
>
> Current project state (versions / paths / tables / roadmap) lives in [STATE.md](STATE.md). Load on demand, not every turn.

## ⭐ INDISPUTABLE OPERATING LENS — 🔴 ARCHIVED 2026-06-04 (post-reset; DO NOT FOLLOW AS CURRENT)

> **ARCHIVED 2026-06-04.** The WM "IC > 0.10 / ShIC > 0.05 = PRIMARY TARGET / Headline-tier / WM signal IS the
> alpha" paradigm in this entire section is **REPLACED** by the post-reset founding framing (MEMORY.md):
> **IC / per-bar predictability is BANNED as a primary metric**; the unit of trading is a **SETUP across a
> MULTI-CANDLE MOVE**; optimize for robust held-out **COMPOUND return**. IC h=1 survives ONLY as a within-WM
> diagnostic gate (>0.015), never an objective. The IC-ladder text below is kept for historical reference only —
> it is NOT current marching orders. (Surgical tombstone per 2026-06-05 brain-audit finding A1 + 2-skill consensus;
> the full slim constitution is staged at runs/staging/brain_upgrade/02 for a user-present session.)

This project is **building SOTA world-class world-models with real capital at stake**. NOT a research project, NOT a science exercise, NOT an exploration of techniques. Every architectural choice is judged by ONE question:

> **Does this push the WM signal into the agent-teaching tier (IC > 0.10 / ShIC > 0.05) where the WM signal IS the alpha — not just a position-sizing input to rule-based strategies?**

Implications that override prior conservative defaults:

1. **A SHIP-tier model (IC ≈ 0.06 / ShIC ≈ 0.03) is a stepping stone, not the goal.** Anything settling at SHIP-tier without a documented push toward Headline-tier is incomplete.
2. **Every architecture in `src/wm/v*/` must have a Headline-target upgrade plan.** Per-version specs in [docs/WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md](docs/WM_HEADLINE_UPGRADE_PLAN_2026_04_30.md).
3. **GPU budget allocation: 30% must go to aspirational runs** (V20+ proposals OR existing-architecture super-tier upgrades), not 100% to conservative SHIP-cohort retrains.
4. **The merged scoresheet's D1 has a super-tier**: IC ≥ 0.10 / ShIC ≥ 0.05 = +5 bonus, breaking the /100 ceiling for Headline models. See [docs/WM_SCORESHEET_MERGED_2026_04_29.md](docs/WM_SCORESHEET_MERGED_2026_04_29.md) D1 entry.
5. **A version with no Headline upgrade hypothesis fails D10 by default**, regardless of its current SHIP-tier numbers.

This lens DOES NOT replace anti-fragility (ShIC > IC*0.5 ratio, walk-forward, DSR). It ADDS a numerical floor any production WM must clear.

The architectural ladder (🔴 ARCHIVED 2026-06-04 — DO NOT FOLLOW AS CURRENT; IC-as-primary is BANNED post-reset, see §header above + MEMORY.md; kept for historical reference only):

```
Filter      IC > 0.015, ShIC > 0.015     gate-only on rule strategies
Sizer       IC > 0.030, ShIC > 0.020     multiplier on rule strategies
Trader      IC > 0.050, ShIC > 0.030     standalone w/ quarter-Kelly  [V1.x is here]
Headline    IC > 0.10,  ShIC > 0.05      WM signal IS the alpha       [PRIMARY TARGET]
Ambitious   IC > 0.13,  ShIC > 0.065     pushing daily-bar info ceiling
Capacity    IC > 0.20,  ShIC > 0.10      tick-level HF stat-arb       [requires V20]
```

**Why "Headline" is top-of-class for our regime**: IC > 0.10 on dollar bars across f34 features is at the upper end of published quant-fund crypto WM benchmarks. To exceed 0.13 reliably, the bottleneck shifts from architecture to representation (tick-level features, sub-second LOB, MEV-aware bundle timing). V20 (tick-level Performer/Hyena) targets the Capacity tier; Headline is the right ceiling for daily/dollar-bar architectures (V1.x family, V3-V8, V11-V14). **"Top of class" with respect to the daily-bar regime is Headline (IC > 0.10); pursuing IC > 0.13 in the same regime is increasingly noise-fitting**.

Provenance: user mandate 2026-04-30 — *"Remember, we are building SOTA world class models with real capital, not just research and fun projects."*

## Anti-Fragile Training Philosophy

The system prioritizes robustness over raw accuracy:
- **Shuffled IC > Contiguous IC** is the critical test. If shuffled IC = 0, the model has memorized temporal patterns, not learned signal.
- **ShIC gap uses h=1 only** — both contiguous IC and ShIC measured at horizon 1 for apples-to-apples.
- **No ShIC LR reduction** — LR follows warmup schedule only. ShIC decline triggers early stop (SHUFFLED_IC_PATIENCE), not LR halving. LR reduction locks in memorized weights and prevents recovery.
- **No focal/smoothing on return TwoHot** — Plain `bucketer.compute_loss(logits, targets)`. Focal gamma upweights temporally-clustered tail returns; label smoothing learns temporal return shape. Both accelerate memorization.
- **ShIC checked every 10 epochs** — More frequent checking (5) triggered premature early stopping before generalization.
- Walk-forward CV with purge gaps prevents temporal leakage
- Regime-balanced sampling ensures performance across trending/mean-reverting/volatile markets
- Temporal jitter, mixup, and noise augmentation prevent overfitting

## Critical Invariants

- **No emoji characters** in any Python print statements (Windows cp1252 crashes)
- Timestamps are always 13-digit milliseconds (range [1.5e12, 2.0e12])
- bar_id must be globally unique per asset (no collisions)
- All 41 pipeline features must be non-null with std approximately 1.0 (34 base + 7 cross-asset)
- Target tail: <10 zero values in last 100 rows (guards against fill_null corruption; threshold 10 separates legitimate altcoin zeros from 50+ fill_null corruption)
- Hardware: RTX 4060 (8GB VRAM), Windows 11 — use mixed precision, respect memory limits
- **Target default: raw returns** (`target_return_*`). Voladj targets create vol shortcut (V1.0 OOS: voladj IC=0.10 but raw IC=0.017). Use `target_prefix="target_return"` everywhere.
- Regime labels: precomputed SMA-200 based (pipeline), fallback to return-based (in get_loss) for old datasets
- **RevIN disabled by default** in all V1-V9 models. RevIN causes temporal memorization (ShIC=-0.001 with RevIN vs ShIC=0.028 without). Use `--revin` to opt-in for experiments only.
- **torch.compile DISABLED for V1.1** — causes NaN collapse at epochs 3-5 with f13. V1.0 works fine; V1.6 untested.
- **Data split: 50/20/20/10** (train/val/oos/unseen). All training scripts use `split_four_way()`. Unseen segment NEVER touched during development — reserved for backtesting only. Purge gap: 400 bars between segments to prevent normalization leakage.
- **Strategy data access (v51 v2 onward)**: read via `pipeline.chimera_loader.ChimeraLoader.load(sym, cadence)`, NOT direct `pl.read_parquet`.
- **Pre-train CI gate**: run `python src/pipeline/pre_train_gate.py --asset <SYM>` before any model training. Composes 5 validators (data_health, chimera_v51, xd_consistency, e2e, split). Exit 2 = hard fail; exit 1 = warns; exit 0 = clean. Wired into `src/run_all_training.py` (post-2026-04-29) — runs by default before preflight; `--skip-gate` to bypass (NOT RECOMMENDED). `--auto-refresh` calls `src/pipeline/refresh.py --target chimera_v51 --scope u50` first.
- **Model layer location**: All world-model versions under `src/wm/v*/` (not `src/v*/`). Archived versions (v2/v5/v7) in `backups/BKP_20260429_MODEL_HARMONIZATION/`. See `docs/MODEL_LAYER.md`.
- **Universe specs are declarative** at `config/universes/{u10,u50,u100}.yaml`; `is_u10/is_u50/is_u100` and `asset_dna` columns inline in v51 chimera (no code lookups needed).
- **V51 v2 fixes vs V50** (per docs/V50_TO_V51_FIXES.md): tick_seq tiebreaker, returns_clean (no silent fill_null(0)), target_return_<h>_raw uncapped alongside clipped, no trailing-row trim, manifest+lineage in `data/manifests/v51_<SYM>.json`.
- **Pipeline framework (post-2026-05-01)**: New pipeline producers MUST use the framework primitives in `src/pipeline/{parquet_io,dispatch,cli}.py`. `parquet_io.atomic_write_parquet(df, path, required_cols=...)` for the G-AUDIT-020 atomic-tmp-rename contract; `dispatch.run_per_task(tasks, worker_fn, workers, mode)` for parallel dispatch with per-task error capture and sys.exit(2)-on-zero-ok; `cli.add_standard_args(ap)` + `cli.resolve_assets(args)` for the canonical `--workers --force --universe --assets --dry-run` surface. See `docs/PIPELINE_FRAMEWORK_2026_05_01.md` for the full pattern + producer template.
- **refresh.py orchestrator (simplified 2026-05-21)**: thin caller over self-sufficient producers. Each producer owns its own asset filtering (`--assets` plural), skip-existing, and worker parallelism per the canonical CLI contract enforced by `cli_assets_support` / `cli_force_support` / `cli_universe_support` invariants. refresh.py adds: DAG order, gate validation, STUB detection, memory_exclusive serialization, failure logging, live heartbeat. CLI: `--target X [--assets ... --universe ... --force --workers N --parallel]`, `--all`, `--exclude STAGE`, `--no-deps`, `--status`, `--list`, `--live`, `--failures`, `--attach/--wait-active/--ignore-active` for active-run handling. `--scope` is a back-compat alias for `--universe`. Pre-refactor refresh.py (1824 lines, content-hash cache + per-asset fanout) archived at `backups/refresh_v1_pre_simplify_2026_05_21/`.

Current versions / current model state in [STATE.md](STATE.md).

## Backtest Simulator Invariants (MANDATORY — 2026-04-22 fix)

Strategy simulators (`short_term_speculator_v2.run_v2`, `short_term_speculator_v3.run_v3`, `trend_follow_backtest.run_one_window_tf`) use **MtM-only accounting**. DO NOT re-introduce the double-count bug fixed on 2026-04-22.

Rules:
1. **Per-bar PnL**: each bar's contribution = sum over held positions of `weights[j] * ret_matrix[t, j] * direction` for positions whose `entry_t < t`. Positions with `entry_t == t` are SKIPPED.
2. **On exit**: capture exit-bar MtM explicitly (`pnl_bar += w * ret_matrix[t, j]`), then subtract `(cost - entry_side) * w`. Do NOT add cumulative `ret_from_entry` to the bar PnL stream — already MtM'd. The `trade_pnl = (ret - cost) * w` in `trade_log` is for attribution only, not for the pnl_bar stream.
3. **On entry**: charge `per_side * w` via `pnl_bar -= w * side_cost`. Exit charges `(round_trip - per_side) * w`. Total round-trip = `2*per_side + residual`.
4. **Flush at window end**: MtM already captured returns through we-1; charge only `-(cost - entry_side) * w`.
5. **Reconciliation gate**: every new simulator includes a probe confirming `sum(pnl_stream) ≈ sum(trade_log.pnl) within 0.1%`. See `scripts/probe_simulator_fix.py`.

Pre-fix symptom: equity inflation of `(2N-1)/N` per N-bar hold, compounded to 5-7x over 13 months. Post-fix: `prod_meta_combined @ maker, 13mo, $1000 = +94.02%` (was claimed +501.2% pre-fix). See `memory/simulator_bug_fix_2026_04_22.md`.

## MakerCostModel Invariants

- **p_fill = 0.80 DEFAULT is optimistic.** Empirical OHLC replay (`src/analysis/execution_sim.py`, 2026-04-22) found actual p_fill = 0.21-0.40 across buckets. Calibration yaml at `config/maker_cost_calibration.yaml`. Real live equity expected 50-75% of fixed-backtest equity.
- `adverse_selection = 0.3` default — calibration shows filled trades are dip-biased winners; actual may be lower. Recalibrate with real live fill data.
- Every deploy plan must budget for `p_fill_live ∈ [0.25, 0.50]` in sizing, not the model default.

## Cross-Version Training Invariants (ALL V1-V9)

These parameters are **load-bearing** — validated through V1 experimentation. They MUST be identical across ALL versions regardless of architecture, unless an experiment proves otherwise with quantitative evidence. Deviations are the #1 source of silent training failures.

**Settings.py invariants (MUST match across all versions):**

| Parameter | Required Value | Why |
|-----------|---------------|-----|
| WM_STEPS_PER_EPOCH | 2000 | <2000 causes ShIC checks to fire before model learns |
| DIVERSITY_STEPS_PER_EPOCH | 2000 | Same issue for NCL diversity training |
| DIRECT_RETURN_WEIGHT | 3.0 | Huber dominance regularizes against TwoHot temporal memorization |
| WM_BATCH_SIZE | 32 | Larger batches accelerate memorization |
| BIN_MIN / BIN_MAX | -1.0 / 1.0 | [-5,5] creates wrong bins for raw returns |
| NUM_BINS | 255 | Schema compatibility with all checkpoints |
| ACTIVE_HORIZONS | [1, 4, 16, 64] | Removing h16/h64 causes ShIC decay |
| TWOHOT_FOCAL_GAMMA | 0.0 (disabled) | Focal upweights temporally-clustered tails = memorization |
| TEMPORAL_CTX_DROP | 0.15 (RSSM, per-sample) / 0.0 (JEPA) | Per-sample ATME (V1.6-class) |
| target_prefix | "target_return" | Voladj creates vol shortcut |

**train_world_model.py invariants (MUST be present in all versions):**

| Feature | Why |
|---------|-----|
| `strict=False` on model `load_state_dict()` | Schema compatibility when architecture changes |
| `strict=False` on ema_model `load_state_dict()` | Same |
| `shic_decline_count` persisted in save/load | Without it, restarts reset counter = infinite ShIC declines |
| `n_features` saved in checkpoint | Collision guard depends on it |
| Checkpoint collision guard in `load_latest()` | Prevents loading f13 checkpoint for f37 model |
| 6-tuple return from `load_latest()` | (epoch, val_loss, patience, gate, best_shic, shic_decline_count) |

**Pre-first-training audit:** Before training ANY version for the first time, grep its settings.py and train_world_model.py against this table. Old code that was never fixed is the most common failure mode.

## Code Change Verification (MANDATORY)

### Per-File Thoroughness Rule

**Multi-file changes are the #1 source of silent bugs in this codebase.** When a change touches N files, each file must be verified individually.

- **Read before writing**: Before editing any file, read the relevant section of THAT file. Do not assume it matches a sibling file you already read.
- **Verify after writing**: After editing a file, confirm the edit landed correctly in context. `py_compile` every modified file individually.
- **No sed-and-pray**: When using sed/regex to batch-apply changes, grep the result in EVERY target file to confirm the substitution matched.
- **One file, one mental context**: When propagating a fix from V1 to V4, mentally re-enter V4's context. Check imports / variable names / surrounding compatibility.
- **Log what you skipped**: If a file was intentionally NOT changed, explicitly note why. Silence about a file is indistinguishable from forgetting it.

After ANY non-trivial code change, complete ALL applicable checks before declaring "done":

**1. Caller/Callee Audit** — When changing a function signature, return values, or class interface: `grep -r "function_name(" src/` to find ALL callers. Update every caller. Verify return value unpacking matches new return order.

**2. Cross-Version Propagation** — When changing code in one version: `grep -r "pattern" src/wm/v*/` to find ALL sibling implementations (V1.1-V1.5, V2-V9). Apply same fix to every version sharing the pattern.

**3. Schema Compatibility** — When adding/removing/renaming class attributes or checkpoint keys: find every `load_state_dict()` call loading this class and verify `strict=False`. Find every `__init__` constructing this class and verify new params have defaults. Check both training scripts (resume) and validation scripts (inference).

**4. Scope Verification** — When copy-pasting code between contexts: verify all variable references resolve in target context (`flat_dim` vs `self.flat_dim`). `py_compile` every modified file.

**5. Smoke Test Depth** — Tests must exercise the SPECIFIC changed code path. If you changed `dream_step`, the test must CALL `dream_step` and verify output shapes. If you changed `get_loss` returns, the test must UNPACK the return values. "Starts without crash" at import/load is NOT a valid test. Minimum: run through at least one full batch of the changed workflow.

**6. Comment & Doc Sync** — When changing logic, counts, schemas, or feature lists: update ALL comments, docstrings, banners, and `--help` text referencing old values. `grep -rn "old_value" src/` to find stale references.

**7. MEMORY.md Sync** — When changes affect project state tracked in MEMORY.md: update the relevant section. New "Do Not Reintroduce" bug → add to Bugs Fixed list. Completed backlog item → mark done or remove. New fix log → add to `memory/fix_logs/INDEX.md`. New Claude instances inherit MEMORY.md as first context.

**8. CLAUDE.md / STATE.md Sync** — When changes affect project-wide invariants or conventions: update CLAUDE.md (invariants) or STATE.md (current state) as appropriate. CLAUDE.md is the source of truth that ALL instances read. STATE.md is the on-demand current-state reference.

**9. Fix Log Protocol** — Before editing ANY model version: read its fix log at `memory/fix_logs/v{N}_{M}.md` to avoid reintroducing fixed bugs. After fixing a NEW bug, append: date, severity, file:line, what was wrong, what was fixed. Cross-version pattern → add to Cross-Cutting Bug Patterns in `memory/fix_logs/INDEX.md`.

**10. Settings Constant Sync** — When changing a constant in one version's settings.py: check if the same constant exists in ALL other versions' settings.py. `grep -rn "CONSTANT_NAME" src/wm/v*/v*_training/settings.py`. Constants that MUST be identical across ALL versions: BIN_MIN, BIN_MAX, NUM_BINS, ACTIVE_HORIZONS, REWARD_HORIZONS, target_prefix, SHUFFLED_IC_PATIENCE, SHUFFLED_IC_CHECK_INTERVAL, WM_STEPS_PER_EPOCH (2000), DIVERSITY_STEPS_PER_EPOCH (2000), DIRECT_RETURN_WEIGHT (3.0), WM_BATCH_SIZE (32), TWOHOT_FOCAL_GAMMA (0.0). Version-specific (OK to differ): D_MODEL, N_HEADS, N_LAYERS, LR, WM_WEIGHT_DECAY, WM_DROPOUT, architecture-specific params.

**11. Pre-First-Training Audit** — Before training ANY version for the first time: grep its `settings.py` against the Cross-Version Training Invariants table. Grep its `train_world_model.py` for: `strict=False`, `shic_decline_count`, `n_features`, collision guard. Check `memory/fix_logs/INDEX.md` "Versions NOT YET audited" list.

**12. Empirical Probe for Numerical Issues** — When fixing NaN, inf, crash, or numerical instability: write a standalone probe script: 200-300 real-data steps through the actual model under AMP. **STRESS PROBE at B=32 (full batch), not just B=4.** Track `h_seq.abs().max()` every step. If growing unboundedly, add post-block/post-stack RMSNorm. ONLY commit a fix after the probe confirms the source AND confirms the fix resolves it. Cost of not probing: V3 had 3 failed fix attempts (6 GPU-hours wasted). Probe cost: 60 seconds.

**Never declare "all pass" from a test that didn't reach the changed code.**

**13. RED TEAM Audit Protocol (MANDATORY)** — Per [docs/DOUBLE_AUDIT_PROTOCOL.md](docs/DOUBLE_AUDIT_PROTOCOL.md), every non-trivial change goes through TWO ADVERSARIAL audit stages. **Stance: every change is presumed broken until proven otherwise.** The reviewer's job is to ATTACK the diff.
- **Stage 1 (during build)**: caller-search, py_compile, smoke-import after each step. Apply to changes touching ≥3 files OR files with ≥3 external importers.
- **Stage 2 (RED TEAM — before commit)**: re-import canary modules, run `validate_all_models.py --quick`, scan `docs/GAPS.md`, **spawn a Sonnet RED TEAM agent on high-stakes diffs** (>10 files, strategy/training/cost-model code, pipeline DAG). **VERIFY every agent claim against actual code** — agents hallucinate even on adversarial prompts.
- **Halt the commit** on any 🔴 CRITICAL finding. File 🟠 HIGH/🟡 MEDIUM in GAPS.md.

**14. Project-specific anti-patterns to scan for** (per [DOUBLE_AUDIT_PROTOCOL.md](docs/DOUBLE_AUDIT_PROTOCOL.md)):
- Walk-forward purge gap = 0 (G-AUDIT-002 caught in xsec ranker)
- MtM double-count (G-PIPE bug, fixed 2026-04-22)
- Look-ahead via full-history standardization (G-AUDIT-011 caught in BOCPD)
- Cost-model bypass (G-AUDIT-010 — gen3 engines hardcode fees)
- Stale imports after file moves (G-AUDIT-001 caught in strat_profiles)
- Same-day publication race in panel features (G-AUDIT-008 in ETF overlay)
- Inline gitignore comments (G-PIPE-005)
- Output-path drift after layout changes (G-PIPE-001/002)
- Capture-output buffering on long-running stages (heartbeat invisibility)
- Two-orchestrator capital silo (G-STR-009)

**15. Contract-Driven Audit Protocol (CDAP) — pre-commit gate** (per [DOUBLE_AUDIT_PROTOCOL.md](docs/DOUBLE_AUDIT_PROTOCOL.md) §"Contract-Driven Audit Protocol"):
- Every commit runs `python src/audit/check_invariants.py` via the pre-commit hook (install via `python src/audit/install_hook.py`).
- Validates `config/_invariants.yaml` against the current tree:
  - Cross-version constants identical across all `src/wm/v*/v*_training/settings.py`
  - Walk-forward purge gap presence
  - Simulator MtM-no-double-count regression guard
  - DAG ordering (chimera_legacy depends only on fetch_binance; chimera_v51 depends on both v50 and frontier)
  - CLI universe support (`--universe u10/u50/u100`) on multi-asset pipeline scripts
  - Atomic-write contract on silver/gold producers
  - No emoji in print/log statements
  - No inline `.gitignore` comments
- Exit codes: `0` clean, `1` warnings (allowed), `2` CRITICAL drift (HALT COMMIT).
- **No bypass — CDAP is mechanically UNSKIPPABLE** (`config/mandatory_gates.yaml` blocks the skip env-var + `--no-verify`; loop/`claude -p` contexts additionally cannot commit at all). Fix the finding's root cause; there is no skip. (Harmonised 2026-06-06 per `DIRECTIVE_GAP_AUDIT` F1: the old documented one-shot bypass is physically blocked — the directive now matches the mechanism.)
- New components MUST declare a top-of-file `__contract__` dict (kind, inputs, outputs, invariants). Discovered via AST by `src/audit/contract_loader.py` (no import-time side effects).
- Add new invariants to `config/_invariants.yaml` whenever a bug is fixed that originated from cross-file value drift, output-path drift, or an unstated contract.

## Expert Coordination Protocol

When multiple experts work on a task:
1. Pipeline changes may affect model inputs → notify architect
2. Architecture changes need training config updates → notify trainer
3. Trained models need evaluation → invoke validator
4. Failed validation means retraining → loop back to trainer
5. Passed validation enables risk assessment → invoke trader

## Agent Model (HARD LIMITS)

Opus 4.7 is the primary brain. **Updated 2026-05-22 17:46 SAST by user mandate** (supersedes the earlier 3/10/13 cap on the same day): the cap is a **PARALLEL constraint** (concurrent active at any moment), NOT a session-total constraint:

- **≤2 Opus sub-agents** in parallel (load-bearing decisions: position-workers, domain-workers)
- **≤9 Sonnet sub-agents** in parallel (scans, literature, look-ups)
- **+1 META** (the Opus parent / coordinator itself)
- **Total ≤12 concurrent at any moment**
- **Serial chains unbounded** (don't count against the parallel cap) — over a session you can spawn dozens of agents as long as ≤12 are concurrent at any one time
- These limits apply **across ALL directives** (/maxx, /apex, /un, /dialect, /oracle, /research, /auditor — any session)
- **Sub-agent model is EXPLICIT every time** (`model: "opus"` or `model: "sonnet"` — omitted parameter inherits parent, verified gotcha from 2026-05-22 session)
- **No Haiku agents** (still excluded)
- **Skill fallback**: Any `/command` that cannot be classified to a registered skill should be treated as `/normal` (Opus 4.7 direct processing)
- **Prefer direct tools** (Grep/Glob/Read) over spawning agents for simple lookups

## Validation Gates

| Gate | Threshold |
|------|-----------|
| Reconstruction MSE | < 0.12 |
| Information Coefficient (h=1 only) | > 0.015 |
| KL Divergence | 0.01 - 15.0 |
| Shuffled IC / Contiguous IC | > 0.3 |
| Val/Train Loss Ratio | < 2.0 |
