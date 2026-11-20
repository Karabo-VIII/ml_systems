# Wealth-Bot Development Framework — Two-Phase Methodology

> **Status**: canonical, 2026-05-26 (v8.4 — §SM11 SETUP-CHASE DOCTRINE: entry detection and move riding are ORTHOGONAL optimization problems with independent KPIs; Phase 2.5 EXIT MECHANISM EXPLORATION mandatory between Phase 2 and Phase 3; canonical exit library (R1-R3, A1-A3, H1-H3, M1-M3); cross-product measurement table (entry × exit); per-(entry,exit) pair L2 capture rate + mover-day capture as first-class outputs; EXIT_BAKEOFF.md deliverable; dossier §13 compliance checklist added. v8.3 — §SM10 CANONICAL LIBRARY STACK MANDATE: pandas-ta + CanonicalHarness adopted; Pattern U (inline indicator) + Pattern U2 (inline simulator) added to auditor grep checklist; inline simulator code BANNED in new scripts; R12 POC validates harness UNSEEN delta=0.0000pp vs post-fix verified. v8.2 — §SM9.1 CONDITIONAL MAX-HOLD EXTENSION (PRE-REGISTERED): default 3d cap unchanged; IF at cap AND continuation signal positive (3 conditions) → EXTEND to ≤7d; 7d is HARD CAP; asymmetric — WINNERS only; losers cut at 3d; cadence-equivalent ceiling table added; empirical anchor (median_hold_of_winning_trades) is DIAGNOSTIC not prescriptive; R12/R23a rarely fires given 1.3d avg hold; falsifier encoded; §F.10 item 11 added. v8.1 — §SM8 CADENCE-AWARE DD-HALT (PRE-REGISTERED, R57b1 provenance); §SM9 3-DAY MAX-HOLD SWEET-SPOT MANDATE; §SM2 cadence priority hierarchy updated (1d DEPRIORITIZED); §F.10 item 10 added. r8 — §STRATIFIED MINING MANDATE: every dossier MUST produce a cadence × regime × approach TOP-3 TABLE; §F.11 REGIME-ROUTED DEPLOY PROTOCOL added. r7 — L5 split into L5a (Sizing Rule, capital-FREE) + L5b (Portfolio Allocation, capital-AWARE); L6 ratio-vs-dollar clarification; new §QUALITY-DIAGNOSTIC (Q1-Q10) framework orthogonal to L0-L6; §RE-MINING NECESSITY matrix. r6 — adds ORACLE/STATIC/ML 3-way split per user mandate 2026-05-26 ~04:35 SAST + session learnings F96BE75A consolidated; supersedes r5.2). Binding for any new ML or Static wealth-bot.
> **Provenance**: 2026-05-25 user mandates — *"phase 1 is always as is: find robust ML and Static strat... And then phase 2 is augmentation: use the oracle framework to augment phase 1 using knowledge from the oracle to tell us and mine how we get better results."* + *"add everything to the framework to ensure we get the output we want: outdoing work done in the past with rigor and determination and pushing to the edge."* 2026-05-26 r6 mandate: *"update the framework docs based on your learnings (oracle, static rules, ML rules) so that we have a robust framework"*.
> **Composes with**: [PROJECT_NORTH_STAR.md](../PROJECT_NORTH_STAR.md) §3.1 (WEALTH objective), [src/wealth_bot/README.md](../src/wealth_bot/README.md), [CLAUDE.md](../CLAUDE.md).
> **Child spec files**: [docs/wealth_bot_static_rules_spec.md](wealth_bot_static_rules_spec.md) | [docs/wealth_bot_ml_rules_spec.md](wealth_bot_ml_rules_spec.md)

---

## TABLE OF CONTENTS

> Scope tags used throughout: `[ORACLE]` = parent framework (L0-L6 KPIs, gates, leaderboard, arbitration). `[STATIC]` = rule-based / no-ML child. `[ML]` = learned-parameter child (LGBM, DL, RL). `[ALL]` = applies to every bot regardless of type.

### Part I — Shared Foundation `[ALL]`
- [Why this framework exists](#why-this-framework-exists)
- [§(TI, ASSET) — the Closed Problem Space](#ti-asset--the-closed-problem-space)
- [Phase 1 — Robust Discovery](#phase-1--robust-discovery)
- [Phase 2 — Oracle-Augmented Refinement](#phase-2--oracle-augmented-refinement)
- [Phase 2.5 — Exit Mechanism Exploration](#phase-25--exit-mechanism-exploration-mandatory-v84)
- [§Phase 3 — Expansion Hunt](#phase-3--expansion-hunt)
- [§Phase 4 — Within-TI Regime Composition](#phase-4--within-ti-regime-composition)
- [§Reproducibility](#reproducibility)
- [§TI x Asset Dossier convention](#ti--asset-dossier-convention)
- [§Parallel-Instance Coordination](#parallel-instance-coordination)
- [§No-Cross-Pollination Rule](#no-cross-pollination-rule)
- [§Sample-Size Discipline](#sample-size-discipline)
- [§Wall-Clock Budget Discipline](#wall-clock-budget-discipline)
- [Anti-patterns](#anti-patterns)
- [Recurring cadence](#recurring-cadence)

### Part II — ORACLE Rules `[ORACLE]`
- [§STRATIFIED MINING MANDATE (v8.1/r8)](#stratified-mining-mandate-r8)
  - [SM2 — Cadence priority hierarchy (v8.1)](#sm2--cadence-priority-hierarchy-v81-update)
  - [SM2b — Regime axis](#sm2b--regime-axis--required-stratified-output)
  - [SM8 — Cadence-Aware DD-Halt (v8.1, PRE-REGISTERED)](#sm8--cadence-aware-dd-halt-calibration-v81-pre-registered)
  - [SM9 — 3-Day Max-Hold Mandate (v8.1)](#sm9--3-day-max-hold-sweet-spot-mandate-v81)
  - [SM9.1 — Conditional Max-Hold Extension (v8.2, PRE-REGISTERED)](#sm91--conditional-max-hold-extension-v82-pre-registered)
  - [SM10 — Canonical Library Stack Mandate (v8.3)](#sm10--canonical-library-stack-mandate-v83-all)
  - [SM11 — Setup-Chase Doctrine (v8.4)](#sm11--setup-chase-doctrine-v84-all)
- [§QUALITY-DIAGNOSTIC Framework (Q1-Q10)](#quality-diagnostic-framework-q1-q10)
- [§RE-MINING NECESSITY](#re-mining-necessity)
- [§ORACLE-RULES — Parent Framework](#oracle-rules--parent-framework)
  - [OR1 Gate Spec (G1-G6)](#or1-gate-spec-g1-g6)
  - [OR2 Capture-Rate KPI Hierarchy (L0-L6)](#or2-capture-rate-kpi-hierarchy-l0-l6)
  - [OR3 Calibration Ledger Discipline](#or3-calibration-ledger-discipline)
  - [OR4 Cross-Bot Leaderboard](#or4-cross-bot-leaderboard)
  - [OR5 Failure Catalog](#or5-failure-catalog)
  - [OR6 Ship-vs-Refute Discipline](#or6-ship-vs-refute-discipline)
  - [OR7 Asymmetric Loss + No-Mid-Stream-Gate-Switch](#or7-asymmetric-loss--no-mid-stream-gate-switch)
- [§Layered Strategy Decomposition](#layered-strategy-decomposition)
- [§Oracle-as-Parent / Static-ML-as-Children Architecture](#oracle-as-parent--static-ml-as-children-architecture)
- [§Edge-Pushing Protocol](#edge-pushing-protocol)
- [§Standard Dossier Report Format](#standard-dossier-report-format)
- [§Gold-Standard Dossier](#gold-standard-dossier)
- [§Anti-Advocacy Protocol](#anti-advocacy-protocol)

### Part III — STATIC-RULES Child Framework `[STATIC]`
- [§STATIC-RULES — Rule-Based Bot Framework](#static-rules--rule-based-bot-framework)
  - [ST1 Indicator-Cross Protocol](#st1-indicator-cross-protocol)
  - [ST2 Filter Combination Tests](#st2-filter-combination-tests)
  - [ST3 Exit Policy Variants](#st3-exit-policy-variants)
  - [ST4 Sizing Rules](#st4-sizing-rules)
  - [ST5 Position Structure Variants](#st5-position-structure-variants)
  - [ST6 Pattern S/T Compliance (mandatory)](#st6-pattern-st-compliance-mandatory)
  - [ST7 Failure-Mode Catalog — Static-specific](#st7-failure-mode-catalog--static-specific)

### Part IV — ML-RULES Child Framework `[ML]`
- [§ML-RULES — Learned-Parameter Bot Framework](#ml-rules--learned-parameter-bot-framework)
  - [ML1 Training Discipline](#ml1-training-discipline)
  - [ML2 Conformal Abstention Gating](#ml2-conformal-abstention-gating)
  - [ML3 LGBM-Gated Bot Patterns](#ml3-lgbm-gated-bot-patterns)
  - [ML4 Transformer / DL Bot Patterns](#ml4-transformer--dl-bot-patterns)
  - [ML5 RL Bot Patterns](#ml5-rl-bot-patterns)
  - [ML6 ML-Specific Look-Ahead Risks](#ml6-ml-specific-look-ahead-risks)
  - [ML7 Leak Exposure Hierarchy (LGBM amplification)](#ml7-leak-exposure-hierarchy-lgbm-amplification)

### Part V — Session Learnings Addendum `[ALL]`
- [§F96BE75A EXPERIENTIAL ADDENDUM](#f96be75a-experiential-addendum)
  - [§F.11 Regime-Routed Deploy Protocol (r8)](#f11-regime-routed-deploy-protocol-r8)

---

## Why this framework exists `[ALL]`

Four failure modes the framework explicitly closes:

1. **Phase-1-only complacency**: pick (Instrument, Indicator, Approach), audit, ship. Misses structural defects buried in trade-decision context. Bot has compound % but suboptimal exits / over-eager fires / blind to regime / missing easy refinements.
2. **Oracle-only overconfidence**: mine for patterns, propose refinements, ship. Small-sample noise gets ship-promoted as alpha. 2026-05-25 evidence: oracle round mined 9 UNSEEN trades, F1 gate claimed +28pp; honest TRAIN+VAL fit collapsed it.
3. **Discipline-without-momentum** (added 2026-05-25 r2): rigorous null-detection that never escalates. The framework ships NULL twice, the third round mines the same saturated tuple, and 6 months pass without outdoing the baseline. Discipline becomes a comfort zone. The Edge-Pushing Protocol (§below) closes this third failure mode.
4. **Layer-conflation** (added 2026-05-25 r4): reporting compound % as the headline KPI collapses 7 independent layers (time-frame / signal / capture / cost / conditioning / sizing / risk) into one scalar — you cannot diagnose which layer is broken. The Layered Strategy Decomposition (§below) gives each layer its own KPI so refinements can target the actual bottleneck.

The framework requires FOUR properties simultaneously:
- a robust baseline (Phase 1 discovery + audit gates)
- honest refinement (Phase 2 imagine-frame + pre-registration + asymmetric loss)
- continuous expansion (Edge-Pushing Protocol: stretch targets + saturation escape + scope expansion)
- layered visibility (per-layer KPIs surfaced in every audit, capture rate not collapsed into compound)

Without all four: inflated claims, stale baselines, stalled progress, or undiagnosable bottlenecks.

---

## §STRATIFIED MINING MANDATE (r8) `[ALL]` (2026-05-26 user mandate)

> **Provenance**: 2026-05-26 user verbatim — *"Does the framework cover the ff: different time frames, different regimes (I now want top strat candidates from across these when we do our mining. Top 3 candidates each). Bake this into what the static and ML strats will produce. And obviously the exit mechanism and other dimensions will be exercised for each."*
>
> **What this adds**: prior framework explored cadence and regimes as individual dimensions but did NOT require stratified output. Mining produced a single ranked list. r8 requires that every dossier produce a **cadence × regime × approach TOP-3 TABLE** as a mandatory closure deliverable. The G1-G6 gate spec is UNCHANGED — stratification adds COVERAGE requirements, not new gates. A candidate that passes G1-G6 ships into its appropriate cell; one that does not is REFUTED in its cell (NULL).
>
> **Applies to**: ALL (TI, ASSET) dossiers starting after r8. Existing dossiers: document gap as "not yet stratified per r8" with queued action (see back-fill mandate below).

### SM1 — Cadence axis (L0) — REQUIRED coverage

Every (TI, ASSET) dossier MUST mine and report across all applicable cadences before closure. NULL is explicit and honest — it means data was unavailable or trade-count was insufficient for that cell.

| Cadence tier | Cadences | Expected UNSEEN n (rough) | Ship gate |
|---|---|---|---|
| Sub-hour fast | 15m, 30m | 150-300 / 40-80 | Standard G3 at n >= 20 |
| Intraday | 1h | 40-80 | Standard G3 at n >= 20 |
| Swing canonical | 4h | 10-30 | SS2 small-n override at n < 20 |
| Daily | 1d | 3-8 | INCONCLUSIVE by default; queue for natural-n |

Per §F.4 (timeframe coverage mandate, binding since r6): 4h-only mining is a method-artifact ceiling, not a real signal ceiling. Sub-day cadences shift the MDE 4-16x lower.

**Coverage rule**: ALL four cadence tiers must appear in the STRATIFIED_TOP3.md grid, with explicit NULL for tiers where data was unavailable. Dossiers that close with a single cadence documented MUST note this as a coverage gap in §1A row B.

### SM2 — Cadence priority hierarchy (v8.1 update)

**Default mining cadence set: {15m, 30m, 1h, 4h}**. These four cadences are the PRIMARY targets for all new (TI, ASSET) dossier mining, per the user mandate 2026-05-26 ~07:00 SAST: short-term focus, ≤1d, get in and out, daily top-asset rotation makes >3d holds structurally risky.

**Priority order**: `15m > 30m > 1h > 4h > [1d DEPRIORITIZED]`

| Cadence | Priority | Mining default | Notes |
|---|---|---|---|
| 15m | 1 (highest) | MINE FIRST | Fast-cycle signal; >150 UNSEEN trades expected |
| 30m | 2 | MINE SECOND | Balance between n and noise |
| 1h | 3 | MINE THIRD | Strong sub-day n; regime-routed deploy favors this tier |
| 4h | 4 | MINE FOURTH | Canonical swing; lowest n at PEPE-class assets |
| 1d | DEPRIORITIZED | QUEUE LAST | Mine only after sub-day cells exhausted; structural incompatibility with cross-only signals documented in R57b2 |

**1d rationale for DEPRIORITIZED status**: R57b2 (commit 89cb3f2) found that 1d cadence at PEPE produces 3-23 total trades over 3 years for cross-only signals — structurally insufficient for bootstrap gates. For non-cross signals (breakout, mean-revert from extreme), 1d may still produce signal but is lower priority for short-term mandate. **1d is NOT removed** — Phase 3 expansion may include it; it is simply QUEUE_LAST.

**Regime-Routed Deploy implication** (§F.11): the regime router SHOULD weight cadence allocation toward sub-day cells. When a `btc_bull` or `high_vol` regime cell has both a 1h and a 4h candidate passing gates, the 1h candidate gets allocation priority (faster rotation exposure).

---

### SM2b — Regime axis — REQUIRED stratified output

Four independent regime axes are tracked. Regimes are detected using PAST-ONLY data (all definitions use only bars[0..t] — INFERRED tag on any claim citing these).

**Trend regime** (SMA-based):
- `trending_up`: SMA-50 slope > 0 AND close > SMA-200 at bar t (both computed on bars[0..t])
- `trending_down`: SMA-50 slope < 0 AND close < SMA-200 at bar t
- `chop`: neither trending_up nor trending_down

**Volatility regime** (rolling 30-day return std percentile, computed on bars[0..t]):
- `low_vol`: rolling std < 33rd percentile of the asset's training-window distribution
- `med_vol`: 33rd to 67th percentile
- `high_vol`: > 67th percentile

**Funding regime** (perp only; spot = N/A):
- `funding_pos`: 8h funding rate > +0.01%
- `funding_neutral`: -0.01% to +0.01%
- `funding_neg`: < -0.01%

**BTC-macro regime** (BTC 30d return, computed at bar t using bars[0..t]):
- `btc_bull`: BTC 30d return > +10%
- `btc_bear`: BTC 30d return < -10%
- `btc_sideways`: between -10% and +10%

**Usage in stratified table**: each trade-fire is tagged with its regime context at the signal-fire bar. Cells in STRATIFIED_TOP3.md group trade-fires by their primary regime combination. Cells with fewer than 3 viable candidates are marked NULL (expected for sparse combinations).

### SM3 — Approach axis — BOTH children required

The stratified table has two sub-tables: one for STATIC approach and one for ML approach. Each sub-table has the same cadence × regime cell structure.

**STATIC sub-table**: populated by rule-based bots only. Each cell's top-3 candidates are ranked by Q_total (primary), UNSEEN compound % (secondary), own-bootstrap p05 (tiebreaker).

**ML sub-table**: populated by learned-parameter bots (LGBM, DL, RL) only. Same ranking. ML-specific note: see SM5 for single-model vs separate-model architecture choice.

### SM4 — STRATIFIED_TOP3.md output format

Every (TI, ASSET) dossier produces a `STRATIFIED_TOP3.md` file at `docs/dossiers/stratified/<TI>_<ASSET>__STRATIFIED_TOP3.md`. Format:

```markdown
# <TI> × <ASSET> — Stratified Top-3 Table
> Framework r8. Produced at dossier closure (or partial-closure checkpoint).
> Primary sort: Q_total. Secondary: UNSEEN%. Tiebreaker: own-bootstrap p05.
> NULL = fewer than 3 viable candidates cleared G1-G6 for this cell.

## STATIC approach

| Cell (cadence × trend × vol) | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| 4h × trending_up × med_vol | candidate_id \| UNSEEN% \| n_trades \| Q_total \| deploy_tier | ... | ... |
| 4h × trending_up × high_vol | NULL | NULL | NULL |
| 4h × chop × low_vol | candidate_id \| ... | NULL | NULL |
| 1h × trending_up × med_vol | ... | ... | ... |
| 1h × chop × high_vol | ... | ... | ... |
| ... | | | |

BTC-macro regime overlay (add to cell label when relevant — e.g., "4h × trending_up × med_vol × btc_bull"):
> For brevity, BTC-macro regime is reported as a SEPARATE axis table below if it materially
> splits outcomes; otherwise merged into the primary cell label.

## ML approach

| Cell (cadence × trend × vol) | Rank 1 | Rank 2 | Rank 3 |
|---|---|---|---|
| ... | ... | ... | ... |

## Coverage summary

| Cadence | Cells populated | Cells NULL | Coverage % |
|---|---|---|---|
| 15m | — | — | — |
| 30m | — | — | — |
| 1h | — | — | — |
| 4h | — | — | — |
| 1d | — | — | — |
```

**Each candidate entry** in a cell contains:
- `candidate_id`: canonical round + config label (e.g., `R12_WMA_10_30_whale_kelly1.0`)
- `UNSEEN_pct`: UNSEEN compound % (H4-realistic if applicable)
- `n_trades`: number of UNSEEN trades for that cell's regime slice
- `Q_total`: Q_total_measured_pct from quality_diagnostic_score_card
- `deploy_tier`: HIGH / MEDIUM / LOW (from Q_total)

### SM5 — ML regime-model architecture decision

ML bots face an architecture choice for producing regime-stratified output: (a) a SINGLE model with regime-conditional thresholds, or (b) SEPARATE models trained on each regime cell.

**Default pattern: single model + regime-conditional thresholds** (REQUIRED starting point):
- Train one LGBM/DL model on the full TRAIN+VAL window, with regime-label as a feature.
- Calibrate abstention threshold separately per regime cell (on VAL subset for that regime).
- Report per-regime AUC and hit-rate separately.
- Advantage: more training data per model; no small-sample overfitting in sparse regime cells; single deployment artifact.

**Escalation to separate models** (allowed only under this condition):
- Single-model AUC degrades by > 0.10 when evaluated on a specific regime cell vs the full-window AUC.
- AND that regime cell has n_train >= 100 (enough data to train a separate model).
- Document the escalation decision in the dossier with the AUC evidence cited (INFERRED without actual AUC run; VERIFIED with run).

**Regime-conditional threshold table** (mandatory in ML candidates targeting stratified output):

```yaml
regime_conditional_thresholds:
  # Abstention threshold calibrated per regime on VAL subset only
  trending_up_med_vol: 0.62
  trending_up_high_vol: 0.70   # stricter in high-vol (more abstention)
  chop_low_vol: 0.55
  # ... one entry per populated regime cell
  default_fallback: 0.60       # used when regime not determinable at bar t
```

### SM6 — Dossier closure checklist additions (r8)

Add to the dossier-closure checklist in §TI x Asset Dossier convention:

```
[ ] STRATIFIED_TOP3.md produced at docs/dossiers/stratified/<TI>_<ASSET>__STRATIFIED_TOP3.md
[ ] All 4 cadence tiers represented (or explicit NULL with data-unavailable reason)
[ ] All 3 trend regimes covered in table rows (or NULL for sparsely-populated cells)
[ ] STATIC sub-table populated (or "approach not pursued — document why")
[ ] ML sub-table populated (or "approach not pursued — document why")
[ ] SM5 ML architecture decision documented (single-model vs separate; evidence cited)
[ ] Per-cell n_trades reported (guard against sub-10 trade cells claiming ranking)
[ ] NULL cells labeled with reason: "n < 3 viable candidates" or "data unavailable for cadence"
```

### SM7 — Back-fill mandate for existing dossiers

**New dossiers** (any (TI, ASSET) started after r8): MUST produce STRATIFIED_TOP3.md at closure.

**(PEPE, MA/EMA) gold-standard dossier** (back-fill queue):
- R57+ task: retroactively populate STRATIFIED_TOP3.md for existing SHIPPED and INCONCLUSIVE candidates.
- Available from existing artifacts: R12 perp (WMA 10/30 + whale + Kelly 1.0 at 4h), R23a (mean-revert at extreme MA distance at 4h). Both are 4h candidates — cadence coverage is partial.
- Sub-day cadence cells (15m, 30m, 1h) will be NULL until Phase-0 cadence sweep (queued per §F.4) completes.
- Document as: "STRATIFIED_TOP3.md: partial — 4h cells populated from R12/R23a; sub-day cells queued for R57+ sweep."
- This is a SEPARATE round from the current deepening track. Do NOT block current rounds on back-fill.

---

### SM8 — Cadence-Aware DD-Halt Calibration (v8.1, PRE-REGISTERED) `[ALL]`

> **Provenance**: R57b1 finding, commit 89cb3f2, 2026-05-26. 1h variants showed genuine UNSEEN signal (+25-29% top candidates) but ALL failed G2 because the per-window DD halt of 25% was calibrated for 4h cadence (TRAIN window n=21-46 trades). At 1h, TRAIN windows contain only 7-10 trades after whale filter — the DD halt fires on the 2nd or 3rd losing trade BEFORE the strategy has had statistical time to recover.
>
> **CRITICAL DISCIPLINE**: This is a PRE-REGISTERED framework-level calibration update. The rationale, falsifier, and provenance are all documented BEFORE any candidate is re-run under the new spec. Per §F.8 lesson #4, this is NOT silent mid-stream gate switching — it is an explicit cadence-aware calibration with a documented n-count trigger condition.
>
> **Candidates already REFUTED under v8 are NOT automatically promoted under v8.1.** They MUST be re-run as a new round (R57b1b for 1h candidates). v8.1 opens a NEW gate spec; it does not retroactively reverse v8 decisions.

#### Per-window DD halt thresholds (v8.1 canonical)

| Window trade count (n_trades_in_window) | DD halt threshold | Rationale |
|---|---|---|
| n >= 20 | **25%** (unchanged from v8) | At n>=20 the variance of cumulative compound is low enough that a 25% drawdown is statistically meaningful signal — not just 1-2 bad trades |
| n < 20 | **40%** (relaxed) | At n<20 the variance of cumulative compound is dominated by 1-2 individual trades. A 25% halt fires on noise, not signal. 40% provides room for 3-4 bad trades before the halt engages, allowing the strategy's expected positive mean to manifest |

**Implementation rule**: `n_trades_in_window` is measured as the NUMBER OF SIGNAL FIRES (entry events) in that walk-forward window's TRAIN segment, after all filters applied. It is NOT bar count. Computed on TRAIN segment only (not VAL or OOS/UNSEEN).

**Example at 1h cadence**:
- WMA(10,30) + whale filter at 1h PEPE: ~7-10 fires per 18-month TRAIN window
- 7-10 < 20 → apply 40% per-window DD halt
- With 3-4 consecutive losers (expected at n=7-10): 40% halt absorbs them; 25% halt would have fired on trade 2-3

**Example at 4h cadence**:
- WMA(10,30) + whale filter at 4h PEPE: ~21-46 fires per TRAIN window
- 21-46 >= 20 → apply 25% per-window DD halt (unchanged)

#### Falsifier (mandatory — encoded at registration)

If a candidate passes the v8.1 gate (clears G2 under 40% threshold at n<20) BUT subsequently fails on:
- **G6 cross-window persistence**: compound positive in all 4 windows on the UNSEEN regime-filtered slice
- **G3 own-bootstrap p05 > 0**: block bootstrap on UNSEEN compound

Then the 40% relaxation was OVER-PERMISSIVE for that candidate. Conclusion: re-tighten the threshold back toward 35% and re-run. Do NOT simply accept a G2-pass that fails G3/G6 as a framework loophole — the falsifier is binding.

#### Interaction with SM2 cadence priority

At 15m cadence: TRAIN windows are expected to have >50 fires even with aggressive filters — n>=20 almost always satisfied; 25% halt applies. At 30m: depends on filter; likely n>=20 at most filter combinations. At 1h: the n<20 condition is triggered for aggressive filters (whale + bd combo). Compute n BEFORE applying the threshold, per the table above.

---

### SM9 — 3-Day Max-Hold Sweet-Spot Mandate (v8.1) `[ALL]`

> **Provenance**: user mandate 2026-05-26 ~07:00 SAST verbatim: *"trade short-term ideally and capture setups that make us money in the short term. Max hold will be per strat, per exit, but 3d is the sweet spot. The reason for this is that daily the top assets change — so we want to get in, make money, and get out."*
>
> **This is a HARDENING** (more restrictive than current state), not a relaxation. It imposes a structural upper bound on hold duration across all candidates.

#### Default max-hold cap

**3 trading days per position** = sweet-spot maximum. Cadence-equivalent bars:

| Cadence | 3-day bar equivalent | Max-hold bars (default) |
|---|---|---|
| 15m | 3 × 24h × 4 bars/h | 288 bars |
| 30m | 3 × 24h × 2 bars/h | 144 bars |
| 1h | 3 × 24h × 1 bar/h | 72 bars |
| 4h | 3 × 24h / 4h | 18 bars |
| 1d | 3 calendar days | 3 bars |

**Note**: crypto trades 24/7, so "3 trading days" = 3 calendar days = 72 hours continuously.

#### Structural requirement at framework level

Every candidate strategy MUST include a max-hold guard as an explicit component of its exit policy. Options:
1. The exit rule naturally caps hold time (e.g., opposite-cross at 1h with mean hold of 8h — already within 3d; document and keep)
2. Add explicit `max_hold_bars: <N>` guard as a tiebreaker exit (fires when the signal-flip exit has not triggered within N bars)

**This is mandatory** — bots with no max-hold logic are structurally exposed to asset-rotation drag in crypto memecoin space.

#### Extension to ≤7d (see SM9.1 — Conditional Max-Hold Extension)

The 3d cap is the DEFAULT sweet-spot. Framework v8.2 adds a conditional extension rule (§SM9.1) allowing extension to 7d ONLY for WINNING positions with an active continuation signal. Losers exit at 3d always. See §SM9.1 for the 3-condition check, 7d hard cap, cadence-equivalent bars, and falsifier.

#### Per-strategy override (beyond 7d — requires explicit evidence)

Candidates that demonstrate explicit EDGE beyond 7 days (i.e., beyond the SM9.1 conditional ceiling) may override with documented evidence:

```yaml
# In candidate YAML config:
risk:
  max_hold_bars: 36          # v8.1 override: 9 days at 4h cadence
  max_hold_override_rationale: "EMA(5,200) trend-following shows +EV at bar 24-36 on UNSEEN; measured hold-time decay curve in R_XX audit JSON"
  max_hold_override_evidence_artifact: "runs/audit/R_XX/hold_time_edge.json"
```

Without documented evidence, the **default 3d cap applies and is non-negotiable**.

#### Existing candidates (v8.1 compliance check)

- **R12 perp (WMA 10/30 + whale + Kelly 1.0 at 4h)**: avg hold ~32h = 1.3 days. ALREADY within 3d cap. No change needed. Add `max_hold_bars: 18` as documentation-only guard (does not fire in practice).
- **R23a (EMA30_dist>1% + whale at 4h)**: hold-time NOT recorded in current artifact. QUEUE: measure avg hold from per-trade log; if avg > 72h, add explicit override or refit exit with max-hold guard. Add `max_hold_bars: 18` as structural guard regardless.

#### Rationale

In crypto memecoin space, the top-10 by volume changes daily. An asset that was top-5 on Monday may rank #50 by Thursday. Holding a position beyond 3 days:
1. Increases probability of being locked in a declining asset while the capital could be rotating to better movers
2. Reduces signal freshness — whale flow signals at bar t are 72h stale at the exit
3. Increases exposure to overnight regime changes (BTC macro, funding flips)

3 days = sufficient for momentum continuation capture while maintaining rotation flexibility.

---

### SM9.1 — Conditional Max-Hold Extension (v8.2, PRE-REGISTERED) `[ALL]`

> **Provenance**: user mandate 2026-05-26 ~09:30 SAST verbatim — *"should we extend the holding period to < 7d if the move is still ongoing? Or should we make it per time frame — to say what is the equivalent at that point to hold for on average? Because I know in smaller time frames move can or don't last depending?"* META resolution: HYBRID. Baseline 3d (SM9 unchanged) + conditional extension to ≤7d on continuation signal + per-cadence-empirical-anchor as diagnostic evidence (not prescriptive override).
>
> **CRITICAL DISCIPLINE**: This is a PRE-REGISTERED rule, encoded BEFORE any candidate is re-run under it. The 3d default (SM9) is UNCHANGED — SM9.1 is a CONDITIONAL RELAXATION only for winning positions with active continuation signal. Asymmetric by design: losers are cut at 3d (no extension). Winners may extend to 7d only if ALL 3 continuation conditions hold.
>
> **Candidates already compliant under SM9 are NOT automatically re-run.** The extension rule is a forward deployment guard that rarely fires for R12 (avg hold 1.3d). New candidates must include the extension rule as a documented structural field.

#### Default max-hold cap (unchanged from SM9)

**3 calendar days** = sweet-spot maximum. SM9 cadence-equivalent bars apply unchanged (see §SM9 table above).

#### Conditional extension: up to 7 calendar days

**IF** at the 3d cap the following continuation signal is positive, EXTEND hold up to 7d.

**ALL THREE conditions must hold simultaneously**:
1. **Primary entry signal still active**: strategy's indicator cross / threshold condition is still firing (e.g., for MA-cross strategy: `fast_ma > slow_ma` still true at bar t=18; for distance strategy: close still > slow_ma × (1 + threshold_pct)).
2. **Winning position gate** (`close_current > entry_price × 1.005`): the position must be at least +0.5% above entry. NEVER extend losers past 3d cap — losers exit at 3d HARD.
3. **No opposite-direction signal fired**: no opposite cross (fast_ma < slow_ma) AND no whale-flip (whale_net < 0 if whale was the filter) during the hold period.

#### 7-day HARD CAP

At 7d (cadence-equivalent bar count): **HARD EXIT regardless of signal state.** No further extension. No override. The 7d cap is architectural.

#### Cadence-equivalent bar counts (v8.2 canonical)

| Cadence | 3d cap (default — SM9) | 7d hard cap (SM9.1 extension ceiling) |
|---|---:|---:|
| 4h | 18 bars | 42 bars |
| 1h | 72 bars | 168 bars |
| 30m | 144 bars | 336 bars |
| 15m | 288 bars | 672 bars |
| 1d | 3 bars | 7 bars |

**Derivation** (verifiable): 3d × 24h ÷ cadence_hours = 3d bars. 7d × 24h ÷ cadence_hours = 7d bars. Crypto trades 24/7 — no session gaps.

#### Asymmetric loss discipline (binding)

- **Losers**: cut at 3d cap. NEVER extended. Extending a loser adds dead-capital exposure and asset-rotation drag.
- **Winners**: MAY extend to 7d ONLY on continuation signal (all 3 conditions above).
- This is "let winners run, cut losers fast" — the canonical asymmetric loss principle.

#### Empirical anchor (DIAGNOSTIC, not prescriptive)

Every (cadence, signal-family) dossier MUST track `median_hold_of_winning_trades` in its per-layer audit block. This is a DIAGNOSTIC field — it describes observed behavior; it does NOT override the 3d-to-7d extension rule.

| Median hold vs caps | Interpretation |
|---|---|
| `median_hold > 3d` | Extension rule is binding — trades routinely need the 3d-to-7d window; document and verify it adds compound vs 3d forced exit |
| `1d <= median_hold <= 3d` | 3d cap is non-binding for most trades; extension rarely fires |
| `median_hold < 1d` | Neither cap is binding; extension fires near-zero; document as compliance-only guard |

**R12 anchor** (INFERRED from avg hold 1.3d): `median_hold ≈ 1.3d`. Extension fires for positions still open at bar 18 (3d). Prior estimate: <3% of R12 trades reach bar 18 open AND winning AND with continuation signal — extension is a forward-deployment safeguard, not a live behavioral change for R12.

#### YAML config block (mandatory structural field for all candidates)

Add to the `risk:` block in every candidate YAML config:

```yaml
risk:
  max_hold_bars: 18          # SM9 3d default (4h cadence)
  # SM9.1 CONDITIONAL EXTENSION (v8.2, PRE-REGISTERED)
  max_hold_extension_bars: 42        # SM9.1 7d hard cap (4h cadence = 42 bars)
  extension_continuation_rule:
    condition_1_signal_still_active: true   # fast_ma > slow_ma (or equiv) at bar 18
    condition_2_winner_gate_pct: 0.005      # close > entry_price × 1.005 required
    condition_3_no_opposite_signal: true    # no opposite cross / filter flip
    # All 3 must hold simultaneously. If ANY fails: exit at max_hold_bars (3d cap).
    losers_cut_at_3d: true                  # BINDING: winners-only extension; losers exit at 3d ALWAYS
  max_hold_override_rationale: null  # null = using SM9/SM9.1 default; populate only if abandoning 7d ceiling
```

#### Falsifier (encoded at pre-registration — binding)

If the extension rule produces candidates that:
- **Pass G2** (10/10 seeds positive on UNSEEN) under v8.2 extension, BUT
- **Fail G6** (cross-window persistence: compound positive in all 4 windows on regime-filtered UNSEEN slice) at rates higher than the same candidates under the 3d-only rule

Then the 7d extension was OVER-PERMISSIVE — the extra hold captured noise, not continuation alpha. Conclusion: re-tighten the `condition_2_winner_gate_pct` from 0.005 to 0.010 (require +1.0% before extending) and re-run. Do NOT accept a G2-pass-G6-fail under extension as a framework loophole.

#### Existing candidates (v8.2 compliance)

- **R12 perp (WMA 10/30 + whale + Kelly 1.0 at 4h)**: avg hold ~1.3d. Extension rarely fires (prior <3%). Add `max_hold_extension_bars: 42` and `extension_continuation_rule:` block as documentation guard. Behavior unchanged in practice.
- **R23a (EMA30_dist>1% + whale at 4h)**: hold-time NOT yet measured. Add `max_hold_extension_bars: 42` and extension block structurally. Queue: once hold-time measured from per-trade log, verify whether extension is binding or documentation-only.

---

### SM10 — Canonical Library Stack Mandate (v8.3, 2026-05-26) `[ALL]`

> **Provenance**: 2026-05-26 ~14:00 SAST user directive verbatim -- *"Fix this indicator thing. I know you can literally get it off the shelf."* Root cause: 50+ R-round scripts each contained an INLINE simulator + INLINE indicator computation. Each is a new bug surface. v8.2 pre-delivery self-audit + per-commit gate are bandaids; the root cause is re-implementation. Solution: adopt off-the-shelf library stack + canonical harness so future workers CANNOT write inline simulators.

#### SM10.1 -- pandas-ta as canonical indicator library

- **pandas-ta** is the canonical indicator library for all new wealth-bot scripts.
- Already present in `requirements.txt`. Install: `pip install pandas-ta`.
- Do NOT write inline `closes.rolling(N).mean()` or `wma()` functions (Pattern U).
- Import convention: `import pandas_ta as ta` at top of any new script.
- Past-only convention (close-of-bar signal, fill at next-open -- the standard case):
  `wma_fast = ta.wma(df["close"], length=10)` -- no shift needed; indicator at bar t
  uses closes up to t, fill at opens[t+1] is past-only by construction.
- Strict prior-bar convention (mid-bar evaluation or extra conservative):
  `sma = ta.sma(df["close"], length=10).shift(1)` -- shift(1) makes bar-t indicator use only closes[t-1] and prior.
- The harness helper functions `wma_past_only()`, `sma_past_only()`, `ema_past_only()` wrap
  pandas-ta with explicit `shift` parameter and numpy fallback for offline environments.

#### SM10.2 -- CanonicalHarness mandate

**ALL new wealth-bot scripts MUST import `src/wealth_bot/harness.py::CanonicalHarness`.**
Inline simulator code is BANNED in any script written after 2026-05-26 (framework v8.3).

```python
from wealth_bot.harness import CanonicalHarness, StrategySpec, WindowSpec
from wealth_bot.harness import wma_past_only, sma_past_only, ema_past_only
```

The harness makes the following bug classes STRUCTURALLY IMPOSSIBLE:

| Bug Pattern | Pre-v8.3 Surface | Harness Enforcement |
|-------------|-----------------|---------------------|
| Pattern S -- trail-stop via `max(low, trail)` | Inline exit loop | Breach check via `lows[j] <= trail_level` only |
| Pattern T -- same-bar close fill (`entry_p = closes[i]`) | Inline `entry_p = closes[i]` | API only exposes `opens[i+1]` for entry |
| Pattern U -- inline indicator computation | `wma()` / `rolling().mean()` functions | Must use `wma_past_only()` or `past_only_indicator()` |
| Pattern U2 -- no-library indicator | Any custom EWMA/SMA/WMA | pandas-ta canonical; numpy fallback only in library |
| MFE-lock look-ahead | Inline `closes[exit]` for unrealized fill | Harness resolves fill as `opens[exit+1]` |
| Unlabelled tail-flush | Silent residual at data end | `tail_flush=True` flag in every trade dict |
| Missing repro block | Omitted `build_repro_block()` | Auto-attached in `CanonicalResults.repro` |
| Unlabelled all-4-positive | Caller computes inconsistently | `CanonicalResults.all_4_positive` auto-computed |

#### SM10.3 -- Migration plan

- **Grandfathered (archive)**: scripts written before 2026-05-26 are ARCHIVED AS-IS. Do not modify them to use the harness retroactively unless re-running the round for a new purpose.
- **New scripts**: any new R-round or research script MUST use CanonicalHarness. The pre-commit CDAP (`check_invariants.py`) will include Pattern U grep on new files.
- **POC migration**: `scripts/wealth_bot/r12_canonical_harness_poc.py` demonstrates R12 migrated to the canonical harness. UNSEEN result verified to EXACTLY match the post-fix VERIFIED R12 (+39.65%, delta=0.0000pp).
- **Migration backlog**: tracked in `src/wealth_bot/harness.py::MIGRATION_BACKLOG` (15 scripts queued). Priority: scripts still in active development first.

#### SM10.4 -- Falsifier

If a legitimate research need requires going outside the harness API (e.g. a non-standard exit policy not covered by `exit_policy` in StrategySpec), that is EVIDENCE the harness API needs extension, NOT a permission to bypass it. Steps:
1. File a harness extension PR adding the new `exit_policy` variant.
2. Run the new variant through the harness.
3. NEVER write a standalone inline simulator.

Pattern U (new) is now part of the auditor grep checklist in `docs/DOUBLE_AUDIT_PROTOCOL.md`:
- **Pattern U**: inline indicator function (def wma / rolling().mean()) in a NEW script (post-v8.3).
- **Pattern U2**: inline simulator loop in a NEW script (post-v8.3).
Both are 🔴 CRITICAL in pre-commit RED-team audit.

---

### SM11 — Setup-Chase Doctrine (v8.4, 2026-05-26) `[ALL]`

> **Provenance**: 2026-05-26 ~16:00 SAST user mandate verbatim: *"the framework should have exit mechanism as an exploration feature, right? ... do we develop models that are good at entry, ... but then we use the exit mechanism as the variable to control what happens during the move and when we exit. Effectively saying: get me good exits when we know there is a quality setup forming, and then once we're in, use the mechanical/variable stop conditions we have to actually ride the wave of the move?"*
>
> **Empirical grounding (VERIFIED from this session)**: PEPE mover-day capture analysis (R57a, commit 2f8c3c4): UP-day HIT rate = 84.1% — entry detection is working. Capture-of-UP-move = 26.1% — exit leaves 74% of available UP-day move on the table. L2 capture analysis (R57a 4h R12 trades): mean capture = -0.325 (exit fires before peak on most trades). Conclusion: ENTRY is NOT the bottleneck. EXIT is the L2 bottleneck.

#### SM11.1 — Two orthogonal optimization problems

Strategies decompose into TWO INDEPENDENT optimization problems that MUST be solved separately:

| Problem | Layers | Primary KPI | Target |
|---|---|---|---|
| **SETUP DETECTION** | L1 (signal) + L4 (conditioning) | TOPQ_DAYS HIT rate — % of top-25% daily moves where we are positioned | HIT rate >= 70% (signal is catching quality setups) |
| **MOVE RIDING** | L2 (capture) | Capture-of-available-move on TOPQ_DAYS (weighted mean) | Capture rate >= 50% (exit is extracting the setup) |

**Failure to separate these problems is the #1 cause of ineffective Phase 2 mining.** When both are conflated into a single compound % headline, it is impossible to diagnose whether a refinement is improving the signal or improving the exit.

#### SM11.2 — Solving them separately

- **Phase 1** validates SETUP DETECTION. Entry quality is the Phase 1 claim. Phase 1 uses the CANONICAL BASELINE EXIT (opposite-signal for cross strategies, fixed-N for distance/threshold strategies). This is INTENTIONAL — Phase 1 does not claim to have an optimal exit; it claims to have a quality entry signal.
- **Phase 2.5** (new mandatory phase — see §F.12) explores EXIT MECHANISMS as the top-level variable. Once Phase 1 has validated entry quality, Phase 2.5 holds entry fixed and sweeps the canonical exit library across 12 mechanisms.

#### SM11.3 — Cross-product table mandate

Every dossier MUST emit a cross-product table:

```
rows = entry baselines (1-3 Phase 1 validated candidates)
cols = exit mechanisms (canonical library R1-R3, A1-A3, H1-H3, M1-M3 — see §F.12)
cells = compound + L2 capture rate + Q-DIAGNOSTIC stability composite + mover-day capture
```

This table is the Phase 2.5 deliverable. It surfaces which exit mechanism best exploits each validated entry signal. Without this table, the dossier's L2 layer is UNOPTIMIZED.

#### SM11.4 — Falsifier (encoded at registration)

If ALL exit mechanisms produce L2 capture rate indistinguishable from the baseline (within +/-5pp across the cross-product sweep), the conclusion is that the SIGNAL has a structural exit-ceiling: the indicator fires at a point where no mechanical exit can extract materially more of the move. In that case, changing the indicator family at Phase 1 (new (TI, ASSET) dossier) is the correct action, NOT further exit tuning. Do not re-test the same exit variants with tighter parameters.

#### SM11.5 — Interaction with existing phases

- SM11 does NOT change Phase 1 (entry validation) or Phase 2 (oracle refinement).
- SM11 INSERTS Phase 2.5 between Phase 2 and Phase 3 as a mandatory checkpoint.
- Phase 3 expansion axes (cadence, chart-type, filter, etc.) benefit from a known-optimal exit from Phase 2.5 — they compose cleanly.
- Phase 4 regime composition uses the per-regime best (entry, exit) pair from the Phase 2.5 cross-product table.

---

## §QUALITY-DIAGNOSTIC Framework (Q1-Q10) `[ALL]` (r7, 2026-05-26 user mandate)

> **Provenance**: 2026-05-26 user verbatim — *"We need to know the quality of the bots we are building (not that we don't and can't deploy models that generate wealth for us even if they show jagged behaviour, but we need to definitely be sure)."* + *"Keep the L framework above as framework, but add a 2nd layer to it that will be able to diagnose exactly the quality of the model (vs its peers and oracle, say)."*
>
> **Relationship to L0-L6**: the L0-L6 framework gives STRUCTURAL decomposition (what each layer does, what fraction of the move was captured). The Q1-Q10 framework gives DIAGNOSTIC quality scoring — a second, orthogonal lens. L0-L6 is the architecture; Q1-Q10 is the quality grade. They compose, not replace.
>
> **MANDATORY at ship-time**: every candidate that clears Phase 1 / Phase 2 gates MUST include a `quality_diagnostic_score_card` block (template in §F.6 extended below). Q-scoring is NOT optional.

### Q1 — Edge Provenance

Where does the edge come from? Can the operator explain the mechanism in 1-2 sentences with a citable anchor?

| Score | Meaning |
|---|---|
| 0 | Mystery: no mechanism articulated; "it worked in backtest" |
| 1 | Hand-wave: vague narrative ("momentum tends to continue") with no specifics |
| 2 | Rule-based mechanistic explanation (specific conditions articulated; no formal literature) |
| 3 | Mechanistic + literature or empirical anchor (e.g., whale-net momentum cited to on-chain flow literature, OR empirical null-falsifier run confirming the mechanism holds) |

Reference examples: R12 perp Q1=2 (WMA cross + whale_net momentum; mechanistic but no formal literature anchor). R23a Q1=2 (mean-revert from extreme MA distance + whale confirm; mechanistic with no literature).

### Q2 — Robustness

Walk-forward stability (per-window IC variance), regime-conditional performance, and sensitivity to parameter perturbation (±10% on fast/slow period, ±25% on filter threshold).

| Score | Meaning |
|---|---|
| 0 | Single-window positive only; collapses on out-of-sample window OR parameter ±10% |
| 1 | Passes 2 of 4 windows; parameter perturbation degrades by > 30pp |
| 2 | All-4-windows positive; perturbation degrades < 20pp; one regime significantly outperforms |
| 3 | All-4-windows positive; perturbation-stable (< 10pp delta at ±10%); consistent across bull/chop/bear regimes |

Reference: R12 Q2 likely 2 (R51b cadence sweep showed sub-day cadences REFUTED; 4h flanking pairs (8,24) and (12,36) both worse — confirms Pareto-optimal in cadence dimension, but single-cadence concentration reduces score).

### Q3 — Capacity and Scaling

How does edge degrade with position size? Test at R10k (sub-bp impact), R250k (~5bp), R1M (~25bp impact estimate). Report UNSEEN-equivalent compound at each deploy size.

| Score | Meaning |
|---|---|
| 0 | Edge collapses at R250k (compound < 50% of R10k value) |
| 1 | Edge stable to R100k; degrades significantly at R250k |
| 2 | Edge stable to R250k; degrades at R1M+ |
| 3 | Edge stable to R1M+ (institutional scale); no slippage-driven degradation at target deploy scale |

**Status note**: Q3 REQUIRES NEW MINING for any candidate not yet measured with a capacity sweep. Retroactive scoring from existing artifacts is NOT possible. See §RE-MINING NECESSITY.

### Q4 — Look-Ahead / Leakage Integrity

Simulator audit score per Auditor hardened checklist (Pattern S / Pattern T compliance, FM-PSEUDO-VB-FORWARD-CLOSE absence, H4-realistic fill verified).

| Score | Meaning |
|---|---|
| 0 | Multiple unresolved leaks found; no fix applied |
| 1 | One known leak with partial fix; H4-realistic correction not applied |
| 2 | All known leaks fixed; H4-realistic fill applied; one residual caveat documented |
| 3 | Clean across all auditor passes (22+); zero residual caveats; Pattern S/T compliance confirmed |

Reference: R12 Q4=2 (bespoke sim has `entry_p = closes[i]` — bar-close-fill leak — but H4-realistic correction applied and net behavior documented; residual caveat acknowledged).

### Q5 — Statistical Confidence

Own-bootstrap CI width, jackknife K=2 + K=3 collapse magnitudes, and top-N-trade concentration (top-3 % of compound).

| Score | Meaning |
|---|---|
| 0 | top-3 > 70% of compound AND/OR jackknife K=2 collapses > 80% |
| 1 | top-3 > 50%; jackknife K=2 collapses 40-80%; n_unseen < 20 |
| 2 | top-3 40-50%; jackknife K=2 stable within 30%; n_unseen 20-40 |
| 3 | top-3 < 40%; jackknife K=2 collapses < 20%; n_unseen >= 40 |

Reference: R23a Q5=1 (top-3 = 52.7% of compound; jackknife K=2 → +1.54% from +60.31%, an 97% collapse).

### Q6 — Peer Comparison (within-TI)

Rank versus sibling configs in the same TI family using L1 capture rate from `capture_rate_decomposer.py` L1 mode. Best-of-TI = oracle within-family ceiling.

| Score | Meaning |
|---|---|
| 0 | Below 25th percentile of within-TI configs on L1 capture rate |
| 1 | 25th-50th percentile |
| 2 | 50th-75th percentile |
| 3 | Top decile (within-TI oracle ceiling or near it) |

**Status note**: Q6 REQUIRES the L1 within-family sweep mode of `capture_rate_decomposer.py` which is queued but not yet built. Score as `NOT_YET_MEASURED` until tooling exists. See §RE-MINING NECESSITY.

### Q7 — Oracle Comparison (vs frontier first-principles)

Distance from what a vanilla Opus oracle would propose given the asset's distributional signature. Single `/oracle` pass per candidate.

| Score | Meaning |
|---|---|
| 0 | Oracle says "this is the wrong tool for this asset / regime"; recommends a fundamentally different approach |
| 1 | Oracle confirms direction but identifies a significantly better alternative with evidence |
| 2 | Oracle confirms approach; suggests incremental improvements; no better first-principles alternative |
| 3 | Oracle confirms approach as optimal and has no better suggestion given current tooling |

**Status note**: Q7 requires a dedicated oracle pass per candidate. One pass = one Q7 score. See §RE-MINING NECESSITY.

### Q8 — Failure-Mode Predictability

Have all expected failure modes been documented IN ADVANCE in the candidate's `strengths_weaknesses_expected_failures` block (§F.6)? Can the operator predict WHICH category of loss the bot will exhibit next?

| Score | Meaning |
|---|---|
| 0 | No failure-mode documentation exists |
| 1 | Generic failure modes listed (e.g., "bear market bad") with no specifics |
| 2 | Most failure modes predicted in advance with specific falsifier conditions |
| 3 | All failure modes predicted in advance with citable falsifier per mode; monitoring triggers defined |

Reference: R12 Q8 likely 2 (regime shift + whale-net disruption FMs documented; MEV/adversarial FM listed; no quantitative falsifier for each).

### Q9 — Reproducibility

Repro block completeness: env + canonical_seeds + git_sha + chimera_mtime + schema_version + claim_contract version. Bit-exact replay verifiable.

| Score | Meaning |
|---|---|
| 0 | Missing multiple repro fields; replay not verifiable |
| 1 | Most fields present but chimera_sha or seeds missing; replay approximate |
| 2 | All fields present; replay not smoke-tested |
| 3 | Full repro block present AND smoke-tested (bit-exact replay confirmed) |

### Q10 — Operational Maturity

Pre-live verification items closed, paper-trade journal age, cross-instance validation depth, audit history (how many dispatches has this candidate survived?).

| Score | Meaning |
|---|---|
| 0 | Fresh candidate; no live or paper-trade history; 0-1 auditor dispatches |
| 1 | 1-5 auditor dispatches; some pre-live items open; no paper-trade history |
| 2 | 5-15 auditor dispatches; most pre-live items closed; 1-5 paper-trade fires logged |
| 3 | N >= 20 paper-trade fires; cross-instance validated; multi-auditor survived; all pre-live items closed |

Reference: R12 Q10=1 (5 pre-live items pending as of INST-F96BE75A closure; 17 auditor dispatches but pre-live items not closed; no paper-trade history yet).

### Q-score aggregation and deploy tiers

```
Q_total = sum(Q1..Q10) / 30 × 100%
```

For candidates with `NOT_YET_MEASURED` dimensions:

```
Q_total_measured = sum(measured Qs only) / (3 × count_measured) × 100%
```

Report both. Tier classification uses `Q_total_measured` when any Q is unmeasured, with a mandatory "measured-only caveat" note.

| Q_total | Deploy tier | Constraint |
|---|---|---|
| >= 70% | HIGH QUALITY | Deploy with standard discipline (full Phase 1/2 gates + standard Kelly) |
| 50-69% | MEDIUM QUALITY | Deploy at half-Kelly or smaller; explicit weakness profile required; monitoring at 2x cadence |
| < 50% | LOW QUALITY | Deploy only with quarter-Kelly max + heavy monitoring + documented "even-if-jagged-still-wealth-positive" rationale |

### Per-candidate Q-score-card template (mandatory ship-template field, extends §F.6)

Add this block to every SHIPPED candidate's config YAML AND dossier entry:

```yaml
quality_diagnostic_score_card:
  # Q1-Q10 scores: integer 0-3, or NOT_YET_MEASURED (with mining plan noted)
  Q1_edge_provenance: 2 / 3        # mechanistic + cited; no formal literature
  Q2_robustness: 2 / 3             # all-4-windows; single-cadence concentration
  Q3_capacity_scaling: NOT_YET_MEASURED   # requires capacity sweep at R10k/R250k/R1M
  Q4_look_ahead_integrity: 2 / 3   # bespoke sim leak + H4-realistic correction applied
  Q5_statistical_confidence: 2 / 3 # top-3% and jackknife K=2 stable within 30%
  Q6_peer_comparison: NOT_YET_MEASURED    # requires L1 within-TI sweep (decomposer L1 mode not yet built)
  Q7_oracle_comparison: NOT_YET_MEASURED  # requires dedicated oracle pass
  Q8_failure_mode_predictability: 2 / 3   # most FMs in strengths_weaknesses block
  Q9_reproducibility: 3 / 3
  Q10_operational_maturity: 1 / 3  # pre-live items pending; no paper-trade history

  # Aggregate score (measured dimensions only):
  Q_measured_dimensions: [Q1, Q2, Q4, Q5, Q8, Q9, Q10]   # 7 of 10 scored
  Q_total_measured_pct: 67%         # (2+2+2+2+2+3+1) / (3*7) = 14/21 = 67%
  Q_total_full_pct: NN%             # fill when Q3/Q6/Q7 measured
  deploy_quality_tier: MEDIUM       # based on Q_total_measured_pct = 67%
  jagged_behavior_acknowledgment: null   # required if tier = LOW; explain why deploy proceeds despite gaps
  unmeasured_mining_plan:
    Q3_capacity_scaling: "Run capacity sweep at R10k / R250k / R1M using simulate_perp with scaled slippage model. Queue for post-pre-live-verification phase."
    Q6_peer_comparison: "Run capture_rate_decomposer.py L1 mode once tooling is built (Phase 0 instrumentation queue)."
    Q7_oracle_comparison: "Dispatch one /oracle pass with R12 perp trade log + PEPE 4h distributional signature as context."
```

---

## §RE-MINING NECESSITY `[ALL]` (r7, 2026-05-26)

> **Premise**: approximately 70-80% of Q-DIAGNOSTIC dimensions can be scored RETROACTIVELY from existing artifacts. NEW mining is required for only 2 of 10 dimensions; a single oracle pass covers 1 more. Future dossiers should populate ALL 10 Qs DURING the dossier lifecycle, not retroactively.

### Retroactive scoring (no new mining required)

These Q dimensions score from existing artifacts in the candidate's dossier, audit JSON, and commit history:

| Q | Source artifact |
|---|---|
| Q1 — Edge Provenance | Candidate's `strengths_weaknesses_expected_failures` block; auditor findings |
| Q2 — Robustness | Walk-forward audit JSON; per-window compound table; parameter-perturbation sweep (if run) |
| Q4 — Look-Ahead Integrity | Auditor findings (Pattern S/T compliance, H4-realistic fill delta); simulator commit history |
| Q5 — Statistical Confidence | Bootstrap JSON; jackknife K=2/K=3 runs; top-N trade attribution block |
| Q8 — Failure-Mode Predictability | `strengths_weaknesses_expected_failures` block; monitoring triggers |
| Q9 — Reproducibility | Repro block in audit JSON; smoke-test log |
| Q10 — Operational Maturity | Auditor dispatch count in calibration ledger; pre-live verification checklist; paper-trade log |

### New mining required

| Q | What is needed | Estimated cost |
|---|---|---|
| Q3 — Capacity Scaling | Run `simulate_perp` at 3 wallet sizes (R10k, R250k, R1M) with a scaled slippage model (`slippage_bps = base + impact_bps(size)`) | 1-2h per candidate |
| Q6 — Peer Comparison | Run `capture_rate_decomposer.py` in L1 within-TI sweep mode (sweeps ALL (fast,slow,filter) combos in the same TI family for each signal-fire bar; tooling currently QUEUED) | Requires tooling build first (~2-3h), then 30min per candidate |

### Single oracle pass required

| Q | What is needed |
|---|---|
| Q7 — Oracle Comparison | One `/oracle` pass per candidate: provide the candidate's config summary + asset distributional signature (PEPE 4h stats: skew, tail, autocorrelation) + trade log summary. Oracle rates whether the approach is first-principles optimal. 15-20min per candidate. |

### Dossier-lifecycle integration mandate (future dossiers)

Future (TI, ASSET) dossiers MUST populate Q1-Q10 at these lifecycle points, not retroactively:

| Q | When to populate |
|---|---|
| Q1 | At Phase 1 baseline ship (edge mechanism must be stated before gates close) |
| Q2 | At Phase 1 completion (walk-forward and perturbation sweep are Phase 1 gates) |
| Q3 | After capacity sweep (run once baseline clears G1-G4; before leaderboard entry) |
| Q4 | At every audit dispatch (auditor updates Q4 score if new leaks found or resolved) |
| Q5 | At Phase 1 completion (bootstrap + jackknife are Phase 1 auditor checks) |
| Q6 | After L1 sweep mode is built (populate for all prior candidates once tooling exists) |
| Q7 | After Phase 2 round 1 (oracle pass naturally happens; log Q7 in calibration ledger) |
| Q8 | At Phase 2 completion (all FMs should be documented before ship-gate adjudication) |
| Q9 | At every commit (repro block maintained continuously) |
| Q10 | Incrementally (updates after each auditor dispatch and each paper-trade fire) |

---

## §ORACLE-RULES — Parent Framework `[ORACLE]`

> Full detail for each sub-section lives in the corresponding named section further in this document. This section is the **consolidated quick-reference** — a new instance reads here first to understand what the Oracle owns, then follows the cross-reference links for implementation detail.
>
> **Child spec files**: `[STATIC]` detail at [docs/wealth_bot_static_rules_spec.md](wealth_bot_static_rules_spec.md). `[ML]` detail at [docs/wealth_bot_ml_rules_spec.md](wealth_bot_ml_rules_spec.md).

### OR1 Gate Spec (G1-G6) `[ORACLE]`

The Oracle **pre-registers and enforces** the ship gate for EVERY candidate — Static or ML. Gates are locked at dossier start and CANNOT be changed mid-stream (see OR7).

| Gate | Rule | Applies to |
|---|---|---|
| G1 | All-4-windows compound > 0 (TRAIN / VAL / OOS / UNSEEN) | ALL candidates |
| G2 | 10/10 seeds positive on UNSEEN | ML ensemble candidates |
| G3 | Block-bootstrap UNSEEN p05 > 0 | ALL candidates at n >= 20 |
| G4 | Max DD < 30% | ALL candidates |
| G5 | UNSEEN compound > baseline by ≥ +10pp (Phase 2 refinement) OR absolute floor met (Phase 1) | Phase-2 refinements |
| G6 | Mechanism-falsifier check (SR1.3) when `top_3_pct_of_compound > 70% AND n_unseen < 30` | ALL ship candidates |

**G2 signed tu_ratio**: G2 UNSEEN positivity check counts the SIGN of UNSEEN compound; a negative mean with positive variance is a G2 FAIL regardless of bootstrap width. All-4-positive is enforced simultaneously — not per-window separately (Auditor-19-CRIT-2 lesson, 2026-05-26).

**Small-n gate override** (R48b power-analysis lesson, binding for all future dossiers): the `delta_p05 > 0 vs baseline` gate is structurally unbeatable at n < 30 in compound-pp metric. At n < 30 replace G3 with: (a) per-trade Information Ratio vs baseline, OR (b) absolute-floor + own-bootstrap p05 + cross-window persistence in 2/3 sub-windows. Pre-register WHICH override at dossier start — do NOT switch mid-dossier.

### OR2 Capture-Rate KPI Hierarchy (L0-L6) `[ORACLE]`

The Oracle owns the canonical L0-L6 KPI definitions. Both children MUST use these exact definitions. Full detail: [§Layered Strategy Decomposition](#layered-strategy-decomposition) and [§Oracle-as-Parent / Static-ML-as-Children Architecture](#oracle-as-parent--static-ml-as-children-architecture).

**Key invariant**: capture_rate is CAPITAL-FREE and COST-FREE. It measures what fraction of the signal-valid move was realized. Compound % is the output, not the KPI.

**Tooling (R50 built)**: `scripts/wealth_bot/capture_rate_decomposer.py` — accepts trade log + chimera + signal-validity rule. Run on every candidate before ship-gate adjudication. All future ship decisions MUST include verified L2 capture rate (not just compound proxy).

### OR3 Calibration Ledger Discipline `[ORACLE]`

`runs/oracle/PHASE2_CALIBRATION_LEDGER.md` is the append-only per-round ledger. Oracle owns it; both children write to it.

Rules:
- Every Phase 2 round adds an entry: date, baseline, candidates, gate result, ship/null.
- Healthy ship rate: 20-40% of rounds. Outside this band: diagnose (always-null = over-strict; always-ship = rubber-stamp).
- A round that changes the gate spec MUST be labeled as an explicit re-adjudication round (not a normal Phase 2 entry) — prevents retroactive gate-relaxation masquerading as normal progress.

### OR4 Cross-Bot Leaderboard `[ORACLE]`

`runs/oracle/WEALTH_BOT_LEADERBOARD.md` ranks ALL shipped candidates across both children. Oracle owns and maintains.

Ranking columns (mandatory): UNSEEN compound (H4-realistic) / UNSEEN pace / max DD / L2 capture rate / n_unseen / robustness gates passed. No candidate enters the leaderboard without a verified layer_kpis block.

Auto-deploy the leader monthly. Demote bots at bottom-half for 2 consecutive months. New bot must beat the median to deploy.

### OR5 Failure Catalog `[ORACLE]`

`runs/oracle/WEALTH_BOT_FAILURE_CATALOG.md` is the append-only refuted-hypothesis catalog. Oracle owns it; both children read before every Phase 2 round.

**Entries are tagged by**: (TI, ASSET, approach, target-layer, date-refuted). Before any Phase 2 mining query, check if the same hypothesis exists in the catalog under the same (TI, ASSET, approach). If yes: skip the query, cite the catalog entry in the calibration ledger — prevents re-mining dead ends.

**Static-specific catalog entries from F96BE75A session** (see §STATIC-RULES ST7 for full catalog):
- FM-ATR-TIGHT-REGIME: tight ATR trailing stops are intrinsically regime-fragile (3 data points; MATURE failure mode).
- FM-PSEUDO-VB-FORWARD-CLOSE: synthetic-bar group-aggregate close assignment leaks future close to earlier bars in the group.

### OR6 Ship-vs-Refute Discipline `[ORACLE]`

The Oracle adjudicates ship-vs-refute when children disagree or when a candidate is borderline.

**Asymmetric loss (binding)**:
- False-positive (ship overfit): real capital deployed on inflated claims. PRIORITY 1 harm.
- False-negative (miss real alpha): opportunity cost of 1 month until re-validation. PRIORITY 2 harm (recoverable).
- Default verdict when borderline: **NULL** + queue re-validation with larger UNSEEN sample.

**Null IS a valid artifact**: every null result is logged in the calibration ledger with mining output. Future instances inherit the mining — preventing re-mining the same dead end.

### OR7 Asymmetric Loss + No-Mid-Stream-Gate-Switch `[ORACLE]`

**Binding rule (Auditor-19-CRIT-2 lesson)**: ship gates are PRE-REGISTERED at dossier start and LOCKED for the dossier lifetime. Any gate change requires:
1. An explicit Oracle-arbitrated re-adjudication round (labeled as such in the calibration ledger).
2. Retroactive re-adjudication of ALL prior REFUTED candidates under the new gate.
3. A commit message and dossier annotation explaining why the gate was wrong.

Mid-stream gate-tightening disguised as "more rigorous methodology" is a discipline mistake — it converts a wealth-optimization problem into a stat-arb-comparison problem and retroactively inflates the refuted count (making past work look more "rigorous" than it was).

---

## §STATIC-RULES — Rule-Based Bot Framework `[STATIC]`

> Scope: pure if-then logic with NO learned parameters. Includes all MA/EMA/SMA cross families, filter combinations, exit policies derived from deterministic rules, and sizing rules not conditioned on ML predictions.
> Full detail and worked examples: [docs/wealth_bot_static_rules_spec.md](wealth_bot_static_rules_spec.md).

### ST1 Indicator-Cross Protocol `[STATIC]`

**Signal generation rule (MA/EMA/SMA cross family)**:
- Fire condition: `fast_ma crosses above slow_ma` (long bias). Directional constraint: LONG only under current framework (no shorting per CLAUDE.md lev=1 and user mandate).
- Signal-validity window: from cross-bar through `(a) opposite cross, (b) filter fails, OR (c) fwd_bars limit` — whichever fires first.
- Decay definition must be pre-registered at dossier start. Do NOT switch decay definition post-baseline.

**Config sweep discipline**:
- Phase 1 grid: cover the full (fast, slow) parameter space appropriate to the cadence. At 4h: {5,7,9,10,12,15,20,26,30,40,50} for fast; {15,21,26,30,40,50,100,120,200} for slow — giving ~500 combinations minimum.
- Selection criterion: pick best-in-train compound AND verify all-4-windows positive before calling SHIPPED.
- Param-selection bias guard: the selected (fast, slow) pair must NOT be re-tuned in Phase 2. If Phase 2 mining suggests a param change, that's a new Phase 1 candidate — run it through Phase 1 gates.

**Within-TI oracle ceiling**: L1 capture rate is computed against ALL in-TI configs (same indicator family), NOT against configs from other TI families. (PEPE, MA/EMA) ceiling is the best of SMA/EMA/WMA × all (fast,slow) × all filters — it does not include MACD or RSI.

### ST2 Filter Combination Tests `[STATIC]`

Filters are L4 conditioning additions applied to the signal-fire bar. Each filter must be tested INDEPENDENTLY first (L4 additive, one-at-a-time) before testing combinations.

**Tested filter families in (PEPE, MA/EMA)** (canonical reference for future dossiers):
- `whale_net > 0` (L4): net whale buy pressure positive. Canonical winner for PEPE 4h.
- `bd_bgf > threshold` (L4): buy depth at best bid. Tested, underperformed whale standalone.
- `fund_rate < threshold` (L4): funding rate. Tested, structurally thinned UNSEEN.
- `premium > threshold` (L4): spot-perp basis. Tested, structurally thinned UNSEEN.
- `xd_btc_return > 0` (L4): BTC regime cross-asset. Persistent in TRAIN, direction-flipped OOS — REFUTED for cross-window persistence gate.

**Combination discipline**:
- OR-combination tested at R18: all 6 filter-OR-agg variants REFUTED — OR-agg floods with noise signals.
- AND-combination narrows n below reliable Phase 2 mining threshold for most combos.
- Preferred pattern: SINGLE dominant filter (whale_net) + parameter tuning of that filter's threshold.

**Filter aggregation anti-pattern**: do NOT aggregate filters with OR logic expecting compound benefit. OR logic lowers selectivity, not raises it — the resulting signal-set is dominated by single-filter fires.

### ST3 Exit Policy Variants `[STATIC]`

Exit policy is an L2 / L3 layer, independent of signal (L1). Each exit variant tested against the same entry rule.

**Tested families in (PEPE, MA/EMA)** (from R7 + R45/R46 + R51 — 21+ variants across exit policy dimension):
- Opposite-cross exit (baseline): flip when fast_ma crosses below slow_ma. This is the canonical baseline for static bots.
- Fixed fwd_bars (time-stop): exit after N bars regardless of signal state. Tested; underperformed opposite-cross.
- MFE trailing stop (exit when MFE gives back X%): REFUTED at R7. Reappeared in R45/R46/R51 — see Pattern S/T below.
- ATR trailing stop (ATR multiple below high-water): REFUTED (R46 post-Pattern-S-fix) and cataloged as FM-ATR-TIGHT-REGIME.
- Ratchet stop (step-level ratchet): REFUTED (R46 post-Pattern-S-fix).
- Chandelier exit (ATR below highest-high): REFUTED (R45).
- Partial take + trail remainder: REFUTED (position structure axis R15).

**Canonical implementation of trailing exits (Pattern S compliance — MANDATORY)**:

Every trailing/chandelier/MFE-lock exit implementation MUST use this guard:

```python
# CORRECT: intra-bar detection + gap-down fill guard
if lows[j] <= trail_level:
    if highs[j] >= trail_level:
        exit_price = trail_level   # mid-bar fill reachable
    else:
        exit_price = closes[j]     # gap-down: price was below trail at open; fill at close (conservative)
    # execute exit
```

Close-only detection (`closes[j] <= trail_level`) is FORBIDDEN — it inflates TRAIN compound 100-1000x on gap-prone bars (Pattern S, 4 catches across R45/R46/R51 this session).

### ST4 Sizing Rules `[STATIC]`

Sizing is L5 — independent of signal (L1) and exit (L2). Each sizing variant is capital-relative (fraction of equity).

**Tested in (PEPE, MA/EMA) at R8** (19 variants across Kelly, vol-target, anti-martingale, equity-curve):
- Kelly 1.0 (full Kelly): canonical winner. Empirically: PEPE fat-tail signal favors aggressive sizing because the signal fires infrequently and the valid windows are high-EV.
- Kelly 0.25, 0.50, 0.75: all underperformed full Kelly (lower compound, lower per-trade exposure on high-EV fires).
- Vol-targeting (constant-vol): underperformed — dynamic sizing penalizes the largest-move opportunities.
- Anti-martingale (double after win): thinned UNSEEN sample via consecutive-loss stop; REFUTED.
- Equity-curve Kelly (reduce after DD): over-triggered on PEPE's volatile equity curve; REFUTED.
- CRRA Kelly (risk-aversion): R30 G1 — partial pass under absolute gate; failed cross-window multi-window.

**Sizing anti-pattern**: do NOT use per-trade conviction weighting from ML features in a static bot. If ML conviction is available, that's an ML child, not a static bot. Static sizing should be a fixed fraction (Kelly 1.0 is canonical for this (TI, ASSET)).

**Deploy constraint (§F.2)**: wallet size R10k-R250k. At this scale, constant-bps slippage model is accurate. No impact term needed.

### ST5 Position Structure Variants `[STATIC]`

Position structure is a distinct L5 sub-dimension from sizing. Tested at R15 (9 variants).

**Variants refuted** (PEPE fat-tail signal is concentrated; diversification hurt):
- Scale-in (2 or 3 tranches into signal window): diluted entry at better prices but missed the bulk of the move.
- Scale-out (partial take at MFE): locked partial gains but missed continuation.
- Single position with trailing (combination): trail was the bottleneck (see ST3 and Pattern S).

**Canonical winner**: single full-position, full-Kelly, entered at signal-fire bar close, exited on opposite cross. No complex position mechanics.

### ST6 Pattern S/T Compliance (mandatory) `[STATIC]`

**Pattern S — Close-only trail breach inflation**: applicable to ALL static bots with trailing / chandelier / MFE-lock exits. Full spec in `memory/fix_logs/INDEX.md` under Pattern S.

Required pre-commit check for any exit-policy script: grep for `closes[j] <= trail_level` without accompanying `highs[j] >=` guard. The pre-commit hook (queued for build) will reject any function body containing close-only trail detection without the intra-bar + gap-down guard.

**Pattern T — Harness inheritance regression**: when writing a new exit-policy script, ALWAYS copy the harness from `scripts/oracle/r46_run_all_depth.py` lines 203-285 (the post-Pattern-S-fix canonical implementation). Do NOT copy from a predecessor that had only a partial fix. Commit body must include a line-by-line diff against r46 lines 203-285 as proof of full compliance.

**TRAIN/UNSEEN ratio canary**: if TRAIN compound / UNSEEN compound > 8x, auto-flag for harness audit BEFORE bootstrap validation. A ratio > 8x is a near-certain sign of Pattern S or a related sim-bug — do not attribute to "overfitting" without first ruling out harness.

### ST7 Failure-Mode Catalog — Static-specific `[STATIC]`

These are STATIC-only failure modes from the (PEPE, MA/EMA) F96BE75A session. Encoded here to prevent re-testing known dead-ends in future Static dossiers on similar assets.

**FM-ATR-TIGHT-REGIME** (MATURE, 3 data points confirmed 2026-05-25 to 2026-05-26):
- Mechanism: tight ATR trailing stops (<= 1.0x ATR multiplier) generate catastrophic whipsaw losses in sideways/declining regimes because the same mechanism (rapid re-entries riding trend) produces round-trip losses when the trend is absent. The failure is STRUCTURAL, not tunable.
- Confirmed instances: C8-R39-E39_1 trailing-0.75x (TRAIN -17.7%), R46-E46_1 trail-2xATR(7) post-fix (TRAIN +2.82% / UNSEEN +22.79%), R51-E51_1 atr-1.0x post-fix (TRAIN -30.99% / UNSEEN +38.52%).
- Ruling: ATR exit family is EXHAUSTED for (PEPE, MA/EMA). Do NOT propose tighter-ATR variants. Alternative exit families (trailing-percent-of-MFE, time-stop-conditional-on-volatility, partial-scale-out tiers) are the next axes.

**FM-PSEUDO-VB-FORWARD-CLOSE** (look-ahead pattern, Auditor 22 MED-1, 2026-05-26):
- Mechanism: synthetic bar implementations that assign the group-aggregate close back to all member bars leak the group's terminal close to earlier bars in the group. A subsequent `.shift(1)` does NOT fix this because the leaked value persists.
- Fix: use `closes[t]` (current bar close) or `cummax(closes within group up to t)` as the within-group representative — never `closes[last_idx_in_group]`.
- Reference: `scripts/wealth_bot/r54_depth_deepening.py:998-1020`.

**FM-WHALE-NET-OR-AGG** (filter aggregation anti-pattern, R18):
- OR-combining whale_net with bd/fund/premium produces signal-set dominated by single-filter fires. Net effect: compound drops vs standalone whale_net. OR aggregation of L4 conditioning filters always tested and refuted for this (TI, ASSET).

---

## §ML-RULES — Learned-Parameter Bot Framework `[ML]`

> Scope: any bot with at least one component that learns parameters from training data. Includes LGBM-gated (signal-picker using chimera features), transformer / DL-based gating, RL-based sizing, and conformal-prediction abstention. Does NOT cover pure if-then rule bots (those are STATIC).
> Full detail and worked examples: [docs/wealth_bot_ml_rules_spec.md](wealth_bot_ml_rules_spec.md).

### ML1 Training Discipline `[ML]`

**Feature engineering**:
- All features must be computable from TRAIN+VAL data only — no lookahead. Standardization/normalization computed on TRAIN, applied to VAL/OOS/UNSEEN (never fit on combined window including test).
- All chimera features used as ML inputs must be in `layer_kpis.L1_signal.chimera_features` of the audit JSON.
- Time-based features (hour-of-day, day-of-week) are allowed but tag as L0 context, not L1 signal.

**Train/val/oos/unseen split discipline**:
- Fit on TRAIN+VAL only. NEVER include OOS or UNSEEN in any fit-time operation.
- OOS is the honest "early-stage validation" window — closer to TRAIN than UNSEEN. A model that passes OOS but fails UNSEEN is a PARTIAL PASS (not a ship).
- UNSEEN is the final holdout. Touch it exactly ONCE per candidate.

**Seed discipline**:
- LGBM ensemble: 10 seeds. Median UNSEEN compound is the headline metric. NOT the best seed.
- Report: seed-0 to seed-9 UNSEEN compounds + median + 10th-percentile. A candidate that passes on median but fails on 10th-percentile is fragile.

### ML2 Conformal Abstention Gating `[ML]`

Conformal prediction sets calibrated confidence intervals around LGBM predictions. When the conformal interval is too wide (low-confidence prediction), abstain — do not fire the signal.

**Threshold discipline**: abstention threshold must be pre-registered on TRAIN+VAL before any evaluation on OOS/UNSEEN. Post-hoc threshold tuning on OOS/UNSEEN is a Phase 2 violation.

**Effect on n_unseen**: conformal abstention typically reduces n_unseen. At n < 20, the ship gate immediately tightens per §Sample-Size Discipline. Factor in the expected abstention rate when sizing the chimera window.

### ML3 LGBM-Gated Bot Patterns `[ML]`

**Architecture**: the LGBM picker reads chimera features at the signal-fire bar and outputs a binary (fire / abstain) or a probability. The underlying static signal (e.g., EMA cross) still fires first — the LGBM gate FILTERS, it does not replace, the static signal.

**Tested variants in (PEPE, MA/EMA) context**:
- 1-strat LGBM-gated (EMA 7/15 + whale + LGBM gate): SHIPPED at +51.86% UNSEEN (H4 realistic with next-bar-open fill: +36.63%). Rank 1 on leaderboard pre-closure.
- 2-strat ortho LGBM-gated (gating across 2 strategy signals): REFUTED at -28.76% UNSEEN H4-realistic. The picker over-fit to bar-close-fill edge; see ML7 Leak Exposure Hierarchy.

**Look-ahead risk specific to LGBM gating** (see also ML6):
- LGBM pickers trained on per-bar chimera features can amplify simulator look-ahead leaks because the picker learns to predict which bars have post-bar favorable fills — a signal not available in live trading. 2-strat ortho showed 6x larger bar-close-fill premium than 1-strat.
- All LGBM-gated bots must be evaluated under H4-realistic fill (next-bar-open + latency + slippage). The bar-close-fill premium is the ML-specific leak exposure.

**Post-training audit**: every LGBM model must include feature importance + SHAP values in its audit JSON. Features with near-zero importance on OOS/UNSEEN but high importance on TRAIN are overfit indicators — flag before ship-gate adjudication.

### ML4 Transformer / DL Bot Patterns `[ML]`

> Status: NOT YET TESTED in (PEPE, MA/EMA) dossier. Queued as Phase 3 expansion axis after LGBM-gated family is exhausted. These rules are STRUCTURAL — derived from the broader V4 Crypto System world-model architecture principles.

**Binding rules (carry over from CLAUDE.md WM invariants)**:
- No RevIN by default (RevIN causes temporal memorization).
- RMSNorm, AdamW betas=(0.9, 0.95).
- ShIC / IC > 0.3 is the gate for WM-layer predictions used as L1 signal.
- Shuffled IC = 0 means memorization, not learning — REJECT the DL signal if ShIC ~ 0.

**Look-ahead risks specific to DL**:
- Sequence-level batch normalization computed across the full training window leaks future statistics. Use layer norm or RMSNorm (per-sample, no global stats).
- Target horizons: fwd_bars=7 means the model is trained on returns from bar t to t+7. This is fine when the model's PREDICTIONS are for t+1 only — but feature engineering (e.g., "average return over last 7 bars") requires the trailing window to be computed from bars T-7 to T only, never T to T+7.

### ML5 RL Bot Patterns `[ML]`

> Status: NOT YET TESTED. Queued for post-DL expansion. Structural rules only.

**Binding rules**:
- Environment rewards must be MtM-only (no double-count per CLAUDE.md Backtest Simulator Invariants).
- Action space: LONG / FLAT only (no short positions, consistent with lev=1 constraint).
- Evaluation window must be OOS/UNSEEN-only — RL agents overfit extremely aggressively to training-period market regimes.
- State representation: chimera features at bar t only (no future leakage into state).

### ML6 ML-Specific Look-Ahead Risks `[ML]`

These look-ahead risks are MORE SEVERE in ML bots than static bots because learned models can extract and amplify subtle data leaks that deterministic rules cannot.

**Risk ML6-A — Feature normalization with future data**: z-scoring / min-max scaling a chimera feature over the FULL dataset (train+val+oos+unseen combined) before splitting. Fix: compute normalization stats on TRAIN only; apply transform to all windows.

**Risk ML6-B — Target horizon leaking**: using `fwd_ret_7` (return bars t to t+7) as a feature for bar t's model decision. This is a direct look-ahead. Fix: use only `past_ret_7` (return bars t-7 to t) as a feature.

**Risk ML6-C — Bar-close-fill premium in LGBM gating** (from R17/R17b, confirmed 2026-05-25): the simulator's `evaluate_actions()` filling at same-bar close leaks bar-close price to the picker's training signal. The LGBM learns to pick bars where the fill happened at a favorable close relative to the next-bar open. Fix: next-bar-open fill in the simulator before any LGBM training. All existing LGBM-gated candidates must be re-run under H4-realistic fill before leaderboard ranking.

**Risk ML6-D — Synthetic bar look-ahead (FM-PSEUDO-VB-FORWARD-CLOSE)**: group-aggregate close assignment in synthetic bars. Affects BOTH static and ML, but ML models can learn to exploit the forward-close signal in features derived from synthetic bars. Fix: same as ST7 FM-PSEUDO-VB-FORWARD-CLOSE.

**Risk ML6-E — Cross-window standardization in panel features**: when chimera features are computed across multiple assets simultaneously (e.g., cross-sectional z-score of volume), the cross-section at bar t includes bars from assets that may not be contemporaneous. Fix: verify all cross-sectional operations are same-timestamp-only joins before ML training.

### ML7 Leak Exposure Hierarchy (LGBM amplification) `[ML]`

Empirical finding from R17b (2026-05-25, INST-A): LGBM pickers have HIGHER leak exposure than single-strategy bots because the picker can compose multiple bar-close-fill edges to amplify the look-ahead.

**Observed leak magnitudes** (H4-realistic delta from optimistic baseline):
- 1-strat static bot (EMA 7/15 + whale): -13.57pp bar-close-fill premium.
- 1-strat LGBM-gated bot: -15.22pp (1.5x static).
- 2-strat ortho LGBM-gated bot: -84.22pp (6x static). Refuted under H4-realistic.

**Implication**: multi-strategy LGBM pickers carry dramatically elevated leak exposure. Any LGBM bot covering N > 1 strategies must be re-validated under H4-realistic fill — do not assume static's ~10pp correction is a valid upper bound for ML.

**Pre-training checklist for LGBM bots** (mandatory before any training run):
1. Verify simulator uses next-bar-open fill (not same-bar-close fill).
2. Verify all chimera features are computed without future data in their rolling windows.
3. Verify target label uses close[t+1]/close[t] - 1 (or equivalent open[t+1]-based return), NOT close[t+fwd_bars]/close[t] used as a feature.
4. Run a forward-leak sanity check: train on TRAIN, predict on TRAIN, compute AUC. If AUC > 0.90 on TRAIN but < 0.55 on OOS, training data has look-ahead contamination.

---

## §(TI, ASSET) — the Closed Problem Space `[ALL]` (binding — 2026-05-25 r5, user mandate)

> Provenance: 2026-05-25 verbatim — *"when it comes to an indicator, you're bound by that, it's not optional. So you have to solve within that TI, not outside of it... MACD will get its turn, RSI will get its turn, etc. But we never cross contaminate a single indicator (and all the possible config) and its asset."*

A wealth-bot problem is defined by the **(TI, ASSET) tuple** — the technical-indicator family AND the asset. This tuple is **FIXED for the duration of the dossier**. All work — Phase 1 discovery, Phase 2 oracle refinement, Phase 3 within-TI expansion, Phase 4 within-TI regime composition — stays inside this closed universe.

### What "TI" means (binding scope)

- **MA/EMA family** = {SMA, EMA, WMA} (3 classic moving averages). DEMA / TEMA / KAMA / HMA / ZLEMA / ALMA are **DISTINCT TI's** — each becomes its own (TI, ASSET) cycle.
- **MACD** = own cycle. **RSI** = own cycle. **Bollinger** = own cycle. Etc.
- Within a TI, all CONFIGS (parameters, signal-rule variants — cross / state / bounce / slope-change, filter combinations) are in-scope.

### What CANNOT happen within a single (TI, ASSET) dossier

- **No mixing indicator families**: a (PEPE, MA/EMA) dossier cannot include MACD or RSI logic. Each gets its own cycle.
- **No cross-pollination of indicator math**: a Phase 4 regime router routes BETWEEN MA/EMA configs only, never to a non-MA indicator.
- **No meta-ensembling across TIs**: if (PEPE, MA/EMA) ships +50% UNSEEN and (PEPE, RSI) ships +40% UNSEEN later, they are TWO SEPARATE candidates on the cross-TI leaderboard. They are NOT combined into one bot.

### What CAN expand within a (TI, ASSET) dossier (Phase 3 axes)

WITHIN-TI only:
- **Param expansion**: broader (fast, slow) grid, additional signal-rule variants (cross / state / slope / bounce)
- **Filter expansion**: new chimera-based filters tested against the SAME TI's signal
- **Cadence expansion**: 1m, 5m, 1h, 1d ensembles of the SAME TI rule
- **Chart-type expansion**: HA, Renko, dollar-bars of the SAME TI rule
- **Conditioning expansion**: regime-conditional fires WITHIN the same TI rule
- **Instrument-variant expansion**: spot vs perp vs basket OF THE SAME ASSET (e.g., PEPEUSDT spot, 1000PEPEUSDT perp)
- **Approach expansion within TI**: Static + LGBM-gated + DL-gated, all reading the SAME TI signal

**NOT a Phase 3 axis**: new indicator family. That's a NEW (TI, ASSET) cycle.

### Closed-Dossier protocol

A (TI, ASSET) dossier DECLARES COMPLETE when ALL of:
- ≥6 Phase 3 expansion axes from the in-scope list have been explored (REFUTED or SHIPPED)
- Calibration ledger has ≥2 SHIPPED candidates OR has 6+ consecutive NULL/REFUTED rounds (saturation)
- The cross-instance handoff section is written naming the next-TI candidate

**Sample completion**: instance F96BE75A declared `(PEPE, MA/EMA)` COMPLETE 2026-05-25 after 10 explored axes (8 refuted, 1 shipped, 2 side-findings queued). Dossier: [`docs/PEPE_MA_EMA_INST_F96BE75A.md`](../docs/PEPE_MA_EMA_INST_F96BE75A.md) §EXPLORATION COMPLETION DECLARATION.

After a dossier closes:
- Its primary deploy candidate enters the cross-TI leaderboard
- A new instance may start a NEW (TI, ASSET) cycle (e.g., `(PEPE, RSI)`, `(PEPE, MACD)`)
- The closed dossier remains read-only canonical reference; NEVER reopened (start a new dossier rev if material new data arrives)

### Cross-TI competition (the META-leaderboard layer)

After ≥2 (TI, ASSET) dossiers close, they compete on the same UNSEEN window for capital allocation. Comparison metrics: composed compound, capture rate (3 levels), robustness gates. No cross-TI ensembling — capital ROUTES to the winning TI, or splits between non-overlapping TIs as separate sleeves.

---

## §Layered Strategy Decomposition `[ORACLE]` (binding — 2026-05-25 r4, user mandate)

> Provenance: 2026-05-25 verbatim — *"I need to know the quality of strategies at a deeper level, not just what ROI they would have done (because this is usually based on position sizing, which deserves to be a dimension). Signal, signal quality, and signal decay are one dimension and should not necessarily intertwine with something like position sizing or mechanical exit conditions. We need to know from % to % how much of a move we are capturing (capture rate), not necessarily that if we put x capital we would get y return."*

A strategy is NOT a single black-box monolith. It is **7 independent layers**, each with its own KPI. The framework REQUIRES that every audit / paper-trade report emit per-layer KPIs, not just composed compound.

### The 7 layers

| Layer | Purpose | KPI | Capital-aware? | Independent of |
|---|---|---|---|---|
| **L0 — Time Resolution & Frame** | Cadence, chart-type, window selection | bar-count, frame-fidelity | No | Everything downstream |
| **L1 — SIGNAL** | Rule, quality score, decay function — "is now valid?" | hit-rate × signal-window EV (cost-FREE) | No | Capital, costs |
| **L2 — CAPTURE** | Entry timing, exit timing, in-trade behavior | **CAPTURE RATE = realized_move / available_move_within_signal_valid_window** (cost-FREE, capital-FREE) | No | Sizing, costs |
| **L3 — COST** | Fees, slippage, funding | gross→net friction drag | No | Signal, sizing |
| **L4 — CONDITIONING** | Regime envelope (BTC bull/chop/bear, vol regime, time-of-day) | regime-correct fire-rate | No | Signal rule itself |
| **L5a — SIZING RULE** | Kelly fraction, conditional sizing logic, sizing methodology — "what fraction of equity per fire?" | contribution-per-unit-risk (capital-FREE: expressed as fraction, not $) | **No** (MODEL QUALITY layer) | Signal, capture, cost |
| **L5b — PORTFOLIO ALLOCATION** | Dollar deployment per wallet (R10k-R250k), cross-wallet orchestration, concentration in $-terms | position_$ / wallet_$ at deployment | **Yes** (PORTFOLIO QUESTION; stripped from model quality scoring) | L5a sizing rule |
| **L6 — RISK** | DD trip (ratio-based), circuit breaker (ratio-based), blackouts | stress-period survival; DD% ratio thresholds are capital-FREE; equity-curve dollar thresholds are capital-AWARE (deploy-ops, not model quality) | **Ratio KPIs: No** / **Dollar KPIs: Yes** | Everything upstream (kill-switch on top) |

> **r7 clarification on L5 split**: L5a is part of MODEL QUALITY — it answers "given a signal fires, what fraction of equity is the correct bet size?" (Kelly, vol-target, conviction-weighted). L5b is the PORTFOLIO QUESTION — it answers "given R250k wallet and this bot, how much dollar enters each trade?" L5b is stripped from all Q-DIAGNOSTIC scoring.
>
> **r7 clarification on L6 capital-boundary**: report BOTH the ratio threshold ("-25% circuit breaker") AND its dollar equivalent ("≈ -R25k on a R100k wallet") as two separate fields. The ratio is the model-quality invariant; the dollar is the deploy-ops note. Model quality audit scores on the ratio only.

### Composition formula (the OUTPUT, not a tuning parameter)

```
gross_return_per_signal  = capture_rate × available_move    # L1, L2
net_return_per_signal    = gross_return × (1 − 2×cost − slippage)  # L3
fraction_bet             = kelly_fraction (or vol-target etc.)      # L5a (capital-FREE)
sized_$_return_per_signal = net_return × fraction_bet × portfolio_$ # L5a × L5b (L5b injects $)
portfolio_compound        = ∏(1 + sized_$_return_per_signal) − 1   # outer compounding
                            × kill_switch_factor (L6 ratio-gate)
```

Optimizing compound directly = optimizing a 7-D function via a single scalar. You can't tell whether your gain came from a better SIGNAL (sustainable), better CAPTURE (sustainable), lower COST (venue-dependent, finite), better SIZING (Kelly is concave, finite headroom), better REGIME GATE (sustainable), or DEFENSIVE drawdown noise reduction (defensive only, not offensive).

### Capture Rate — the L2 KPI definition (with the 3-level hierarchy)

Capture rate is the headline KPI for L2. **There are THREE levels of capture rate**, each diagnosing a different defect within the (TI, ASSET) dossier:

```
Level 3 — REALIZED CAPTURE (what we actually got, vs our config's signal-valid window)
  signal_valid_start = bar where rule + filter first co-fired (the entry moment)
  signal_valid_end   = bar where (rule flips back) OR (filter fails) OR (decay-triggered)
  available_L3       = max(close[t] for t in [signal_valid_start, signal_valid_end])
                       − close[signal_valid_start]
  realized           = close[exit_bar] − close[entry_bar]
  capture_L3         = realized / available_L3
  Diagnoses          = is our EXIT/timing optimal given our chosen config?

Level 2 — OUR-CONFIG CEILING (signal-valid window vs OUR config's best-possible capture)
  Same numerator as L3 (we held this config; what could have been captured given the window)
  available_L2       = available_L3 (same)
  capture_L2         = realized / available_L2
  When L2 ≈ L3 (current bot): L2 is not the bottleneck

Level 1 — WITHIN-TI ORACLE CEILING (vs best-of-all-CONFIGS-IN-SAME-TI)
  For each fired bar, sweep ALL (fast, slow, filter) combos WITHIN THE SAME TI
  (e.g., for MA/EMA: all SMA/EMA/WMA × all (fast, slow) × all filters)
  available_L1       = max over all in-TI configs of (their realized move starting from this bar
                       through their own signal-valid window)
  capture_L1         = our_realized / available_L1
  Diagnoses          = did we pick the wrong CONFIG WITHIN THE SAME TI? (different (fast, slow)
                       or different filter — but still SAME indicator family)

Level 0 — ASSET PHYSICS CEILING (only for reporting context; never achievable)
  available_L0       = perfect foresight buy-the-low-sell-the-high
  Always ≤ 100%; included only as denominator-of-denominators reference.
```

**Interpretation thresholds**:
- L3 ≥ 0.80: bot's exit geometry is near-optimal for its chosen config
- L1 ≥ 0.50: our config is roughly best-in-family — minor config-tweak headroom
- L1 < 0.30: WE PICKED THE WRONG CONFIG within the TI. Phase 3 within-TI param/filter expansion should fire.

**Crucial r5 clarification**: Level 1 sweeps ONLY WITHIN-TI configs. It does NOT include MACD / RSI / Bollinger / Donchian / etc. Those are DIFFERENT (TI, ASSET) cycles entirely. Level 1 = best-in-family, NOT best-across-all-indicators.

Reference implementation: [`scripts/wealth_bot/_capture_rate.py`](../scripts/wealth_bot/_capture_rate.py) (L3 implemented; L1 within-family sweep queued for r5 build).

**Interpretation**:
- Capture rate ≥ 0.80 → execution is near-optimal; refinements should target L1 (more/better signals) or L4 (filter regime)
- Capture rate 0.40–0.80 → L2 has headroom; trailing-stop / signal-flip-exit / MFE-aware exits could lift
- Capture rate < 0.40 → L2 is the bottleneck; deferred-entry or wrong fwd_bars likely
- Capture rate < 0 → trade entered AFTER best moment (chasing); signal-decay handling broken

### Per-layer audit report template (binding for every audit)

Every audit JSON / markdown report must emit a `layer_kpis` block:

```yaml
layer_kpis:
  L0_time_resolution:
    cadence: "4h"
    chart_type: "OHLCV"
    bars_train: 2250
    bars_val: 1824
    bars_oos: 1746
    bars_unseen: 846

  L1_signal:
    hit_rate: 0.67           # 6/9 UNSEEN winners
    signal_window_mean_ev_pct: 12.4    # cost-FREE EV per valid window
    decay_definition: "fwd_bars=7 fixed"  # or "signal_flips" / "filter_fails"

  L2_capture:
    capture_rate_mean: 0.44   # MEAN realized/available within valid windows
    capture_rate_median: 0.51
    capture_rate_min: -0.12
    capture_rate_max: 0.89
    interpretation: "L2 has headroom — capture < 0.80"

  L3_cost:
    cost_per_side_pct: 0.22
    avg_slippage_bps: 5
    round_trip_drag_pct: 0.44
    funding_drag_pct: 0.0

  L4_conditioning:
    regime_correct_fires: 7
    regime_total_fires: 9
    regime_hit_rate: 0.78
    regime_definition: "BTC 30d return sign"

  L5a_sizing_rule:
    # MODEL QUALITY layer (capital-FREE). Scoring target for Q-DIAGNOSTIC.
    kelly_fraction: 0.25                  # or "full_kelly" / "vol_target_2pct" etc.
    sizing_methodology: "fixed_kelly"     # fixed_kelly | vol_target | anti_martingale | equity_curve | conviction_weighted
    conditional_sizing: null              # e.g. "half_kelly in bear regime"
    contribution_per_unit_risk: null      # composed_return / max_DD_pct (capital-FREE ratio)

  L5b_portfolio_allocation:
    # PORTFOLIO QUESTION (capital-AWARE). Stripped from model-quality scoring.
    # This layer is for deploy-ops documentation only, NOT for Q-DIAGNOSTIC.
    deploy_wallet_size_R: null            # e.g. 100000.0  (R ZAR equivalent)
    position_$_per_fire: null            # deploy_wallet_size_R * kelly_fraction
    wallet_count_target: null            # how many parallel wallets run this bot
    portfolio_$_start: 5000.0
    portfolio_$_end: 5703.06

  L6_risk:
    # ratio-based KPIs = capital-FREE (model quality scoring target)
    # dollar-equivalent KPIs = capital-AWARE (deploy-ops note only)
    max_dd_pct: -12.52                    # capital-FREE ratio — model quality KPI
    dd_trip_threshold_pct: -25.0          # capital-FREE circuit breaker ratio
    dd_trip_threshold_dollar_R: null      # capital-AWARE equiv; e.g. -25000 at R100k wallet
    dd_trips_fired: 0
    consec_loss_max: 2
    consec_loss_circuit_threshold: 10
    blackout_days: []

composed_compound_pct: 51.86    # the OUTPUT, not a tuning target
```

### Per-layer refinement targeting (binding for Phase 2 oracle queries)

Every candidate refinement surfaced by Phase 2 mining must be TAGGED with its target layer:

```
F1 gate_on_heated_entry          → targets L1 (signal quality)
F2 conditional_softer_whale      → targets L1 (signal envelope)
F3 mfe_trailing_stop             → targets L2 (capture / exit)
F4 reentry_during_continuation   → targets L1 + L2 (re-fire + capture)
F5 lgbm_anti_momentum_override   → targets L1 (signal quality)
regime_gate_BTC_bull_only        → targets L4 (conditioning)
half_kelly_to_full_kelly         → targets L5a (sizing rule — capital-FREE)
reduce_wallet_from_R250k_to_R50k → targets L5b (portfolio allocation — capital-AWARE; not a model-quality refinement)
DD_trip_lowered_to_15%           → targets L6 (risk; ratio-based = model quality)
halt_at_minus_25k_dollars        → targets L6 (risk; dollar threshold = deploy-ops only)
```

When validating, refinements targeting DIFFERENT layers can compose. Refinements targeting the SAME layer are mutually exclusive (pick one). The miner SHOULD prefer cross-layer-orthogonal refinements when ranking.

---

## §Oracle-as-Parent / Static-ML-as-Children Architecture `[ORACLE]` (binding — 2026-05-25 r4, user mandate)

> Provenance: 2026-05-25 verbatim — *"with oracle acting as parent, and STATIC and ML as children of parent — they need to speak to each other and inform each other of opportunities, gaps, etc."*

### Roles

```
                    ┌─────────────────────────────────────┐
                    │           ORACLE (PARENT)            │
                    │  - DIMENSION_SURFACE preamble        │
                    │  - Layer-decomposition arbiter       │
                    │  - Cross-bot failure catalog         │
                    │  - Cross-bot leaderboard             │
                    │  - Phase 2 calibration ledger        │
                    │  - Capture-rate computation canonical│
                    │  - Imagine-frame mandate enforcement │
                    └────────────────┬────────────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
        ┌─────────▼──────────┐              ┌──────────▼──────────┐
        │  STATIC (CHILD)    │◄────────────►│    ML (CHILD)        │
        │                    │  CROSS-TALK  │                      │
        │ - Pure rule bots   │  via shared  │ - LGBM-gated bots    │
        │ - L1-L6 per-layer  │  oracle      │ - DL bots (V20 etc)  │
        │   reports          │  artifacts   │ - L1-L6 per-layer    │
        │                    │              │   reports            │
        └────────────────────┘              └──────────────────────┘
```

### Parent (Oracle) responsibilities

The Oracle is **NOT a third bot** — it's the META-framework that BOTH children obey. The Oracle:

1. **Owns the canonical layer-KPI definition** (§Layered Strategy Decomposition above). Both children MUST emit `layer_kpis` block in their audit JSONs using oracle-canonical definitions.
2. **Owns the shared artifacts**:
   - `runs/oracle/WEALTH_BOT_LEADERBOARD.md` — cross-bot ranking (capture rate + composed compound, both surfaced)
   - `runs/oracle/WEALTH_BOT_FAILURE_CATALOG.md` — every refuted refinement, tagged by target layer
   - `runs/oracle/PHASE2_CALIBRATION_LEDGER.md` — every Phase 2 round across both children
   - `runs/oracle/CROSS_LAYER_HANDOFFS.md` (new) — append-only cross-talk log
   - `runs/oracle/LAYER_DECOMPOSITION_TEMPLATE.md` (new) — canonical per-layer audit template
3. **Arbitrates conflicts**: when Static and ML produce different conclusions about the same layer, oracle round records both and runs a dialectic (BULL=Static, BEAR=ML, NULL=both-wrong) before promoting either.
4. **Enforces DIMENSION_SURFACE preamble** on every L2+ decision: 3-check prelude (convention / scope / falsifier) per `.claude/skills/_common/DIMENSION_SURFACE.md`.

### Children (Static + ML) responsibilities

Each child:

1. **Emits a layer_kpis block** in every audit JSON. Both children use the SAME oracle-canonical KPI definitions. Comparable across children.
2. **Posts cross-talk to oracle's `CROSS_LAYER_HANDOFFS.md`** when its Phase 2 mining surfaces a structural pattern that the OTHER child should test. Format:

```
## [HH:MM TZ] STATIC -> ML: <pattern>
Layer: L2 capture
Observation: Static bot's capture rate is 44% on PEPE 4h EMA 7/15 + whale.
              Trailing-stop at MFE-5%/give-2% would lift to estimated 58%
              (TRAIN+VAL fit, UNSEEN refuted in R1 honest validation).
Hypothesis for ML side: LGBM-gated variant with the SAME exit rule may
              capture differently because the LGBM filters out fires
              where capture would be < 30%. Test on ML side.
Falsifier: if ML capture rate is statistically indistinguishable from
           static after exit harmonization, the pattern is L1-driven, not L2.
Artifact: scripts/wealth_bot/_capture_rate.py output JSON
---
```

3. **Reads the OTHER child's CROSS_LAYER_HANDOFFS posts** before launching any Phase 2 round. Cross-pollination of insights is mandatory, not optional.
4. **Reports refuted refinements TO oracle's failure catalog** with target-layer tag. Prevents re-mining dead ends across children.

### Cross-talk mechanisms (concrete protocols)

| Scenario | Static action | ML action | Oracle arbitrates? |
|---|---|---|---|
| Static finds L4 conditioning insight (e.g., "all winning fires are in BTC bull regime") | Posts handoff to `CROSS_LAYER_HANDOFFS.md` tagged L4 | Reads handoff, runs L4 conditioning test on ML side, reports back | NO unless conflict |
| Static finds L2 capture pattern (trailing stop works on static) | Posts handoff | Tests with LGBM-gated entries (subset of static fires) | NO |
| ML finds new chimera feature with high AUC for winners | Reads handoff | Posts handoff | Static incorporates as L1 filter candidate |
| Both children claim same metric improvement | Both post handoff with their numbers | Same | YES — oracle runs dialectic (BULL/BEAR/NULL on which is the real driver) |
| Either child violates layer-conflation (e.g., reports only compound, no per-layer KPIs) | Audit fails the oracle's pre-ship gate | Same | YES — oracle blocks the ship; refuse to add to leaderboard |

### Required oracle scripts (queued for build)

| Script | Purpose |
|---|---|
| `scripts/oracle/capture_rate_decomposer.py` | Canonical capture-rate calc given (entry, exit, signal_valid_window, price_path) |
| `scripts/oracle/layer_kpi_emitter.py` | Common helper both children call — emits the `layer_kpis` YAML block in audit JSONs |
| `scripts/oracle/cross_framework_propagator.py` | Reads CROSS_LAYER_HANDOFFS.md, surfaces unread items to the active child instance |
| `scripts/oracle/layer_dialectic_arbiter.py` | When children conflict, runs BULL/BEAR/NULL on the disputed claim with the oracle's pre-registration discipline |

### Migration path (binding — existing audits MUST add layer_kpis)

Existing audit artifacts that do NOT have a `layer_kpis` block are LEGACY. Re-deriving them is queued:
- `runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/data/static_rule_1strat/data/audit_ensemble.json` — needs layer_kpis retrofit
- `runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/data/static_2strat_ortho/data/audit_ensemble.json` — same
- All future audits (Phase 1, 2, 3) MUST emit layer_kpis at creation time.

No new wealth-bot ships to the leaderboard without layer_kpis. The framework's pre-ship gate enforces this.

---

## Phase 1 — Robust Discovery `[ALL]`

> **SM11 exit-assumption (v8.4 binding)**: Phase 1 uses the CANONICAL BASELINE EXIT — opposite-signal for cross strategies (e.g., fast_ma crosses below slow_ma), fixed-N-bar hold for distance/threshold strategies. This is INTENTIONAL. Phase 1 validates ENTRY quality, not exit quality. The baseline exit is not expected to be optimal; it exists to produce a reproducible, auditable compound metric. Phase 2.5 will explore exit mechanisms once entry quality is confirmed.

**Input**: (Instrument, Indicator, Approach ∈ {Static, ML})

**Procedure**:

1. **Pick instrument + indicator**. Currently PEPE 4h × EMA/MA cross. Future: extend to multi-asset, additional indicators.
2. **Pick approach**:
   - **Static** = MA-cross + whale/depth/premium filter, no ML gate.
   - **ML** = MA-cross + LGBM signal-picker reading chimera features as the gate.
3. **Build config YAML** at `src/wealth_bot/configs/<name>.yaml` declaring: strategy spec, chimera feature inputs, fwd_bars, cost, Kelly fraction, risk gates, window splits.
4. **Audit**: 10-seed `train_and_audit.py --ablation full` (LGBM ensemble + threshold pathway).
5. **Robustness gates** (binding):
   - All 4 windows (TRAIN, VAL, OOS, UNSEEN) compound > 0 at the ensemble level
   - 10/10 seeds positive on UNSEEN
   - Block-bootstrap UNSEEN p05 > 0
   - Max DD < 30%
6. **Output**: config + audit JSON + paper-trade journal at `runs/audit/<run_dir>/journals/DEPLOY_*.jsonl`.

**Example (this session)**: 1-strat EMA 7/15 + whale_net>0, LGBM-gated, UNSEEN +51.86% / 10/10 seeds / p05 +50.66% / max DD −12.5%. Configured at [pepe_ema_bot_static_1strat.yaml](../src/wealth_bot/configs/pepe_ema_bot_static_1strat.yaml).

---

## Phase 2 — Oracle-Augmented Refinement `[ALL]`

**Input**: Phase 1 baseline (verified robust).

**Procedure** (apply each step in order; stop at first hard refute):

### Step 2.1 — Trade-decision-context mining

For every fire in the baseline (across all windows: TRAIN+VAL+OOS+UNSEEN):
- Pre-fire (T−12 to T−1): chimera feature values at decision time
- In-hold (T to T+fwd_bars): MFE, MAE, intra-hold bar of peak/trough
- Post-exit (T+fwd_bars to T+fwd_bars+12): did the move continue?
- Outcome: WIN / LOSS, fwd_ret pct

Mine for:
- Feature distribution shift winners vs losers (AUC, Cliff's delta)
- Missed-alpha catalog: high-fwd_ret bars where the bot DIDN'T fire — disambiguate why (MA failed, filter failed, LGBM rejected, in-trade lockout)
- Near-miss entries: filter-failed bars with positive fwd_ret
- In-hold structural patterns (e.g., consistent peak-by-bar-3, decay-by-bar-7)
- Cluster fires by feature signature, look for WR-extreme clusters

**Reference implementations**:
- [scripts/oracle/wealth_bot_decision_miner.py](../scripts/oracle/wealth_bot_decision_miner.py)
- [scripts/oracle/wealth_bot_persistent_pattern_miner.py](../scripts/oracle/wealth_bot_persistent_pattern_miner.py)

### Step 2.2 — Hypothesis generation

The miner outputs ≤5 refinement hypotheses, each as: **name + mechanism + expected effect**.

Refinement axes (precedent — extend as new ones emerge):
- F1 **Entry-condition gate**: block fires when T−1 features satisfy "heated" conditions
- F2 **Conditional softer filter**: admit fires that normally fail (e.g., whale<0) when alternate condition holds (e.g., EMA sustained + BTC up)
- F3 **Trailing-stop / MFE-aware exit**: exit early when MFE peaks and gives back ≥X%
- F4 **In-trade re-entry**: allow re-fire within hold window when conviction renews
- F5 **Anti-momentum override**: bypass LGBM rejection after consecutive winning fires

**Anti-pattern**: pre-defining the refinement set. The mining must SURFACE the hypothesis, not validate a pre-chosen one. Pre-defined hypotheses bias the miner into confirmation.

### Step 2.3 — Honest validation (the discipline)

**Binding rule**: never tune on UNSEEN. Period.

For each refinement:

1. **Threshold/parameter grid**: define a sensible search space (5-8 cells per axis).
2. **Fit on TRAIN+VAL only** (or TRAIN+VAL+OOS for the larger sample size when the refinement is conservative — block-only rules). Select the parameter tuple maximizing fit-window compound.
3. **Apply chosen parameters to UNSEEN**. Report compound, n_trades, WR, max_dd, Sharpe.
4. **Compare to baseline**. If UNSEEN gain >= +10pp AND fit-window compound NOT worse than baseline by more than 10pp, refinement SURVIVES. Otherwise REFUTED or OVERFIT.

**Falsifier (write before running)**: "Refinement fails if its TRAIN+VAL compound is lower than baseline, OR UNSEEN compound is within ±5pp of baseline (no real edge)."

### Step 2.4 — Cross-window persistence test

For each candidate persistent feature/pattern (NOT per-refinement):

1. Bootstrap (200 resamples) AUC of winner-vs-loser distribution per pre-UNSEEN window (TRAIN, VAL, OOS) separately.
2. **Persistence gate**: AUC > 0.60 in ALL THREE windows with SAME direction, AND 90% CI lower bound > 0.55 in at least 2 of 3.
3. If no feature passes: ship baseline (the candidate refinements were noise).
4. If any feature passes: derive a gate (threshold) from TRAIN+VAL+OOS combined, evaluate on UNSEEN.

**Reference implementation**: [scripts/oracle/wealth_bot_persistent_pattern_miner.py](../scripts/oracle/wealth_bot_persistent_pattern_miner.py).

### Step 2.5 — Ship or null

- **Ship modified bot** if Step 2.3 OR Step 2.4 produces a refinement that survives honest validation. Update the YAML config; re-run the audit; write the new paper-trade journal. The bot becomes the new baseline.
- **Ship null result** if all refinements fail. The baseline stands. The null result is ITSELF a valuable artifact — record what was mined, what was refuted, why.

In both cases, save:
- Mining JSON: `runs/audit/<run>/data/oracle_*.json`
- Validation JSON: `runs/audit/<run>/data/honest_refinement_validator.json`
- Markdown report: `runs/audit/<run>/ORACLE_*.md`

---

## Phase 2.5 — Exit Mechanism Exploration (MANDATORY, v8.4) `[ALL]`

> **Provenance**: 2026-05-26 ~16:00 SAST user mandate. §SM11 SETUP-CHASE DOCTRINE (above). Empirical anchor: PEPE UP-day HIT rate 84.1% (entry works); capture-of-UP-move 26.1% (exit wastes 74%). This phase is inserted between Phase 2 (oracle refinement) and Phase 3 (expansion) as a MANDATORY checkpoint.

**Input**: Phase 1 validated entry baselines (typically 1-3 candidates from Phase 1 / Phase 2).

**Goal**: For each Phase 1 validated entry signal, identify the highest-capture-rate exit using a cross-product sweep of the canonical exit library.

### Exit mechanism library (canonical, harness-compliant)

All exit mechanisms are implemented in `src/wealth_bot/harness.py` via the `exit_policy` field of `StrategySpec`. They are harness-first (Pattern S/U compliant) — no inline implementation in round scripts.

| Family | ID | Description |
|---|---|---|
| Reactive | R1 | Opposite-signal exit (canonical Phase 1 baseline — e.g., fast_ma crosses below slow_ma) |
| Reactive | R2 | Fixed-N-bar hold (current R23a baseline for distance strategies) |
| Reactive | R3 | Time-cap exit: 3d or 7d hard exit regardless of signal state (SM9/SM9.1) |
| Adaptive | A1 | Trailing ATR, k in {1.5, 2.0, 2.5} (Pattern S compliant) |
| Adaptive | A2 | Chandelier exit: N-bar high minus k×ATR, N in {7, 14}, k in {2.0, 3.0} (Pattern S) |
| Adaptive | A3 | MFE-lock-in: when unrealized gain >= X%, lock Y% of MFE as hard floor |
| Hybrid | H1 | Opposite-signal OR trailing-ATR (whichever fires first) |
| Hybrid | H2 | MFE-lock + opposite-signal fallback |
| Hybrid | H3 | Sigmoid scale-out: partial at t+N, full at t+M or signal flip |
| Momentum-following | M1 | Rate-of-Change deceleration exit (ROC turns negative after being positive) |
| Momentum-following | M2 | Parabolic SAR |
| Momentum-following | M3 | Volume-flow reversal (sustained negative on-bar volume shift) |

**Harness compliance mandatory for all mechanisms**:
- Pattern S: intra-bar breach via `lows[j] <= trail_level` + gap-down fallback to `closes[j]`.
- Pattern U: all indicators via `wma_past_only()`, `sma_past_only()`, `ema_past_only()` or pandas-ta.
- Repro block in every output JSON.
- Per-trade timestamp logging (required for mover-day capture analysis).

### Cross-product measurement per (entry, exit) pair

For every (entry_baseline, exit_mechanism) pair, compute and report ALL of:

| Metric | Definition | Source |
|---|---|---|
| G1-G6 verdicts | Standard ship gates | parent doc §OR1 |
| L2 capture rate | realized/available per signal-valid window (capital-FREE, cost-FREE) | `capture_rate_decomposer.py` |
| Mover-day capture-of-UP-move | % of available UP-day move captured (from TOPQ_DAYS analysis) | `pepe_mover_day_capture_analysis.py` or equiv |
| Jackknife K=2, K=3 stability | Drop 2 or 3 trades; % compound retained | `claim_contract.py` |
| Max DD + duration | Standard risk metrics | harness output |
| Top-3 trade concentration | top_3_pct_of_compound | `claim_contract.py` |
| Stability composite | Q2 robustness + Q5 confidence score | see Q-DIAGNOSTIC |

### Output deliverable

**Primary output**: `docs/dossiers/stratified/<TI>_<asset>__EXIT_BAKEOFF.md`

Format:

```markdown
# <TI> x <ASSET> — Exit Mechanism Bakeoff
> Phase 2.5. Framework v8.4. Produced at <date>.
> Entry baselines: <list>. Exit mechanisms: R1-R3, A1-A3, H1-H3, M1-M3.
> Rows = entry baselines. Cols = exit mechanisms. Primary sort: L2 capture rate.

## Cross-product table

| Entry baseline | Exit | UNSEEN% | L2 capture | TOPQ capture | JK K=2 | Top-3% | Stability |
|---|---|---|---|---|---|---|---|
| R12_WMA_10_30_whale | R1 (opposite-signal) | +39.65% | -0.325 | 26.1% | ... | ... | ... |
| R12_WMA_10_30_whale | A1 (trail_ATR_2.0) | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... |

## Top-3 exit mechanisms per entry (ranked by L2 capture rate)

### Entry: <baseline_id>
1. <exit_id> — L2 capture: X%, UNSEEN: Y% — mechanism note
2. ...
3. ...

## Winning (entry, exit) combos
<list recommended combos for Phase 3 and leaderboard entry>

## Falsifier outcome
<did any exit mechanism beat baseline by >5pp L2 capture? If not: signal has structural exit-ceiling — document and escalate to new TI family.>
```

**Secondary artifact**: per-(entry, exit) pair full Q-DIAGNOSTIC block in `runs/audit/<run>/data/exit_bakeoff_*.json`.

### Phase 2.5 procedure

1. **Pre-register**: at Phase 2.5 launch, record entry baselines + exit library in `PHASE2_PREREGISTRATION.md`. Exit library is FIXED — adding new exit variants mid-sweep is a discipline violation.
2. **Run cross-product**: one CanonicalHarness run per (entry, exit) pair on TRAIN+VAL+OOS+UNSEEN. UNSEEN is the final evaluation; no parameter fitting on UNSEEN.
3. **Measure**: compute all 7 metrics in the table above. `capture_rate_decomposer.py` must be run (Phase 2.5 does NOT use compound% as a proxy for capture rate).
4. **Rank**: sort by L2 capture rate (primary), stability composite (secondary).
5. **Apply falsifier**: if 0 of 12 exit mechanisms improve L2 capture above R1 baseline by more than 5pp — log as STRUCTURAL EXIT-CEILING. Escalate signal to change indicator family (new TI) rather than continuing to tune exits.
6. **Document winners**: top-3 exit mechanisms per entry baseline go into EXIT_BAKEOFF.md.
7. **Feed forward**: winning (entry, exit) combos from Phase 2.5 become the STANDARD CONFIGS for Phase 3 expansion and Phase 4 regime composition.

### Falsifier (encoded at registration — binding)

If zero exit mechanisms improve L2 capture above the R1 baseline by more than 5pp (after full cross-product sweep), the signal has a structural exit-ceiling. This means the MA-cross signal fires at a point where the move has already peaked — no mechanical exit can materially improve extraction once the signal fires. Conclusion: change indicator family at Phase 1 (start a new (TI, ASSET) dossier), NOT more exit tuning.

### Wall-clock budget

Phase 2.5 budget: 2-4 hours wall-clock (12 mechanisms × 1-3 entry baselines = 12-36 harness runs at ~5 min each + analysis). Over-budget triggers split-into-batches (run R1-R3 batch first; if clear winner emerges, skip remaining batches). Harness parallelism: up to 4 runs concurrently on RTX 4060 hardware.

---

## Worked example (2026-05-25 session)

| Phase | What happened | Outcome |
|---|---|---|
| **Phase 1** | Trained 10-seed LGBM ensemble on EMA 7/15 + whale config. 4-window all-positive at ensemble level, 10/10 seeds positive on UNSEEN, p05 +50.66%. | SHIP. +51.86% UNSEEN compound. |
| **Phase 2.1** | Decision-context mining on 9 UNSEEN trades. Found 4 losers cluster on heated-entry features (`liq_short_z30`, `norm_ma_distance`, `wh_whale_net_usd`). | 5 refinement hypotheses (F1-F5) generated. |
| **Phase 2.2** | F1 claimed +28pp UNSEEN uplift on UNSEEN-fit thresholds. | Hypothesis recorded as "candidate". |
| **Phase 2.3** | Honest TRAIN+VAL fit of F1, F3, F1+F3 stack. Best TRAIN+VAL compound for any F1 cell was +96% (vs baseline TRAIN+VAL +203%). UNSEEN with chosen thresholds: +51.18% (F1), +49.89% (F3) — both WORSE than +51.86% baseline. | REFUTED. |
| **Phase 2.4** | Large-sample persistence mining on 111 fires across TRAIN+VAL+OOS. NO feature passed the cross-window AUC>0.60 gate. The TRAIN signal `xd_btc_return` (AUC 0.86) FLIPPED direction on OOS (AUC 0.31). | REFUTED. |
| **Phase 2.5** | Ship null. Baseline +51.86% stands. | Documented as canonical. |

**Lesson**: the oracle methodology IS sound; it correctly surfaced a SUGGESTIVE 4-loser pattern. Honest validation refuted it as small-sample artifact. Without Phase 2.3 + 2.4, this session would have shipped an overfit +28pp claim. With them, we ship honest +51.86%.

---

## §Edge-Pushing Protocol `[ORACLE]` (binding — 2026-05-25 r2)

Discipline prevents shipping noise. Momentum forces continuous expansion past the current ceiling. Both are required.

### EP1 — Stretch target on every baseline (mandatory)

**CRITICAL FRAMING (2026-05-25 user mandate)**: target bands are **LOWER-BOUND aim levels**, NOT ceilings or limiters. Hitting the lower bound = minimum acceptable. Exceeding the upper bound = good, KEEP PUSHING — there is no ceiling above which we declare victory and stop. The user's project objective is to extract maximum wealth/ROI; the bands define the floor of "deployable", not the goal.

Every Phase 1 baseline ships WITH a documented stretch target:

```
Current baseline:        +X% UNSEEN compound over D days = Y%/week pace
Project floor bands:     ≥1%/d  AND/OR  ≥2%/3d  AND/OR  ≥3%/week  (per PROJECT_NORTH_STAR.md)
Project upper aim:       ≥5% in each band (still a floor, not a ceiling)
Gap to floor:            Z pp  (lower_bound − current_pace)  — must be ≤ 0 to ship
Gap to upper aim:        W pp  (upper_bound − current_pace) — keep pushing past
Stretch goal next round: explicit numeric (e.g., "next baseline must improve UNSEEN by ≥20pp OR ship null with documented saturation")
```

No baseline ships without the stretch target in writing. The comfort-zone failure mode begins when teams forget that the band is a FLOOR, not a destination.

### EP2 — Pace conversion (mandatory in every report)

Every UNSEEN compound number is converted to pace estimates AT THE POINT OF REPORTING. Stops the user from having to do mental math to check if the bot has cleared the floor:

```
UNSEEN compound +51.86% / 141 days =
  +0.30%/d    (floor ≥1.0%/d   → MISSED FLOOR by 0.70pp; aim ≥5%/d, gap 4.70pp)
  +0.89%/3d   (floor ≥2.0%/3d  → MISSED FLOOR by 1.11pp; aim ≥5%/3d, gap 4.11pp)
  +2.10%/week (floor ≥3.0%/week → MISSED FLOOR by 0.90pp; aim ≥5%/week, gap 2.90pp)
```

Status labels (no "in-band victory" — floors are minimums):
- `MISSED FLOOR` — below lower bound
- `BORDERLINE FLOOR` — within 0.5pp of lower bound
- `FLOOR MET (push to stretch)` — at/above lower bound, push to upper
- `FLOOR + STRETCH MET (push higher)` — at/above upper bound, but no ceiling — keep pushing

Reference implementation at [`scripts/wealth_bot/_pace_conversion.py`](../scripts/wealth_bot/_pace_conversion.py) — every bot's audit report must include this table.

### EP3 — Saturation Detection + Escape (the loop-break)

A (TI, ASSET) dossier is declared **SATURATED** when ANY of the following triggers:
- 3 consecutive Phase 2 rounds return NULL on within-TI refinement attempts (no candidate survives honest validation), AND sample size has grown ≥ 50% between rounds, OR
- ≥6 Phase 3 within-TI expansion axes from §P3.2 have been explored without lifting the baseline by ≥10pp UNSEEN, OR
- Cross-window persistence test has not surfaced any AUC > 0.60 persistent feature in any of the 3 rounds

When SATURATED, the framework declares the **(TI, ASSET) dossier CLOSED** per §Closed-Dossier protocol. The next sprint MUST escalate to a NEW (TI, ASSET) cycle — NOT another Phase 3 axis on the same dossier:

| New dossier axis | Examples |
|---|---|
| **Different TI on same asset** | (PEPE, MA/EMA) closes → start (PEPE, RSI), (PEPE, MACD), (PEPE, Bollinger), (PEPE, Donchian) |
| **Same TI on different asset** | (PEPE, MA/EMA) closes → start (SOL, MA/EMA), (BTC, MA/EMA) — separate dossiers |
| **Different TI on different asset** | (PEPE, MA/EMA) closes → start (SOL, RSI) — fully separate dossier |

Closed dossiers' primary deploy candidates enter the cross-TI leaderboard (§EP6). Each dossier remains read-only canonical reference; never re-opened (start a new dossier rev if material new data arrives).

### EP4 — Edge-Pushing Cadence (no-comfort-zone clause)

Every wealth-bot is on an explicit expansion clock:

| Trigger | Action |
|---|---|
| 30 days since last shipped improvement | Spawn ≥1 expansion-direction worker (new Instrument/Indicator/Cadence) |
| 60 days since last shipped improvement | Saturation review — even without 3-NULL cascade, audit whether the tuple is the bottleneck |
| 90 days since last shipped improvement | MANDATORY scope expansion. Default to L3 user-confirmation gate (capital at stake of staleness). |

Track per-bot at top of paper-trade journal: `last_improvement_date: YYYY-MM-DD`.

### EP5 — Stretch-Goal Worker (the momentum lever)

When Phase 2 ships NULL but the stretch target is unmet, spawn a STRETCH-GOAL worker AS THE NEXT TURN. Brief:

> "The current baseline is +X% UNSEEN (Y%/week pace) but the project target is Z%/week. The honest refinement loop has been exhausted on this tuple. Propose the TOP-3 highest-EV expansion directions (new Instrument / new Indicator / new Cadence / new Approach). For each: state the mechanism by which it could close the gap, 1-sentence falsifier, and rough wall-clock cost to test."

Worker returns 3 ranked expansion proposals. META picks 1 to dispatch as the next Phase 1 round (NEW baseline, new (I,Indi,A) tuple). Default cadence: stretch-goal worker every 2 NULL Phase 2 rounds.

### EP6 — Cross-Bot Benchmarking (when ≥2 bots exist)

When ≥2 wealth-bots are deployed simultaneously:
- Same UNSEEN window, side-by-side compound + pace + DD
- Ranked leaderboard in `runs/oracle/WEALTH_BOT_LEADERBOARD.md`
- Auto-deploy the leader monthly; demote bots that fall to bottom-half for 2 consecutive months
- When a new bot enters, it must beat the median to deploy

Prevents bot-portfolio bloat (every bot earns its slot).

### EP7 — Failure-Mode Catalog (append-only)

`runs/oracle/WEALTH_BOT_FAILURE_CATALOG.md` records every refuted hypothesis with date + mechanism + refute evidence. Before any new Phase 2 round, the META consults the catalog — prevents re-mining the same dead ends. Catalog entries are tagged by (Instrument, Indicator, Approach, refinement-family) so they retire when scope expands to a new tuple.

This session seeds: F1 gate_on_heated_entry (REFUTED), F3 mfe_trailing_stop (REFUTED), conviction-weighted Kelly (REFUTED), fwd_bars>7 (REFUTED), per-regime picker (UNDERPERFORMS), threshold>0 (REFUTED).

---

## §Phase 3 — Expansion Hunt `[ALL]` (binding — 2026-05-25 r3)

After Phase 1 (robust discovery) and Phase 2 (oracle-augmented refinement), Phase 3 is **EXPANSION** — try things outside the established (Instrument, Indicator, Approach) tuple. This is where ceiling-breaking lift comes from, not refinement.

### P3.1 — When Phase 3 fires

- After 2 consecutive Phase 2 NULL rounds (saturated baseline), OR
- When the Phase 1 baseline misses ALL floor bands (1%/d, 2%/3d, 3%/week) by ≥0.5pp, OR
- When user explicitly authorizes expansion, OR
- Every 60-90 days regardless of state (Edge-Pushing Cadence EP4)

### P3.2 — Expansion axes (WITHIN-TI ONLY, in order of typical EV)

**CRITICAL (r5 correction, user mandate)**: Phase 3 expansion is **WITHIN-TI only**. Changing the indicator family is NOT a Phase 3 axis — it's a SEPARATE (TI, ASSET) dossier per §(TI, ASSET) Closed Problem Space.

1. **Within-TI param expansion**: broader (fast, slow) grid; additional signal-rule variants (cross / state / slope / bounce / MA-of-MA). Still SAME indicator family.
2. **New cadence**: 1m, 5m, 15m, 1h, 1d ensembles of the SAME TI rule.
3. **New chart-type representation**: HA, Renko, Range bars, Volume bars, dollar bars, tick bars — applied to the SAME TI rule.
4. **Within-TI filter expansion**: new chimera-based filters (whale, bd, fund, premium, lob, hbr, te, rv, xrel, liq, composite) tested against the SAME TI signal.
5. **New chimera feature subset**: V50 → V51 feature deltas — for filtering / gating, not for changing the TI math.
6. **New approach**: Static + LGBM-gated + DL-gated — all reading the SAME TI signal.
7. **Instrument-variant expansion**: spot vs perp vs basket OF THE SAME asset (e.g., PEPEUSDT spot, 1000PEPEUSDT perp). NOT a new asset.
8. **Exit-policy axis**: opposite-cross, trailing stop, MFE-lock, partial-take, regime-change exit — all paired with the SAME TI's entry rule.
9. **Sizing-methodology axis**: Kelly variants, vol-targeting, anti-martingale, equity-curve trading — independent of TI signal.

**OUT OF SCOPE (new (TI, ASSET) dossier required)**:
- Switching to MACD / RSI / Bollinger / ADX / KAMA / HMA / ZLEMA / DEMA / TEMA / Hull / Donchian / Keltner / Ichimoku / Supertrend
- Switching to a different asset (e.g., PEPE → SOL); ASSET is fixed within the dossier alongside TI

### P3.3 — Phase 3 procedure

1. **Stretch-Goal Worker** proposes 3-5 ranked expansion proposals (mechanism + falsifier + wall-clock cost).
2. **META picks 1-2** to test in parallel (parallel-cap permitting).
3. Each pick → fresh Phase 1 discovery loop on the new tuple. Old tuple's baseline KEEPS for benchmarking (Cross-Bot Leaderboard).
4. If new tuple's Phase 1 produces a baseline beating the old, it gets a Phase 2 round.
5. Phase 3 is COMPLETE when either:
   - A new tuple's Phase 2 ships a refinement beating ALL prior baselines, OR
   - 3 expansion attempts all fail to beat the prior champion (declare CURRENT-FRAME SATURATED, escalate to user for new axis)

### P3.4 — Non-Linear Phase Bouncing (binding)

**The 3 phases are NOT a linear pipeline.** Knowledge discovered in any phase can demand back-jumps:

- **Phase 2 → Phase 1**: persistent-pattern test surfaces a structural insight that would have changed Phase 1 audit gates (e.g., "all winning fires require BTC bull regime — Phase 1 should screen on regime FIRST").
- **Phase 3 → Phase 1**: a new tuple's expansion reveals the old tuple's robustness gates were wrong (e.g., "fwd_bars should be 3 not 7 for short-cadence variants").
- **Phase 3 → Phase 2**: an expansion experiment surfaces a refinement applicable to the OLD tuple too (e.g., "trailing-stop with the new chart type also works on the original 4h").
- **Phase 2 → Phase 3**: 3 NULL Phase 2 rounds with sample growth (saturation) FORCES Phase 3 jump.

When a back-jump occurs:
- Document the trigger insight in the TI×Asset Dossier (§Reproducibility)
- Mark the prior phase's state in the calibration ledger as "superseded by back-jump <date>"
- The new phase runs with the imagine-frame + pre-registration as if starting fresh — back-jumps don't bypass discipline

The framework is bounce-driven, not linear. The dossier preserves the trail.

---

## §Reproducibility `[ALL]` (binding — 2026-05-25 r3, user mandate)

> *"given these are frameworks: static, ML, and oracle, they need to be reproducible, including their result set."*

Every framework run produces an artifact set that, given the same chimera + config + seeds, produces BIT-FOR-BIT IDENTICAL outputs.

### REP1 — Determinism mandate

Every Phase 1 / Phase 2 / Phase 3 script:
- Sets explicit seeds for ALL randomness consumers (numpy, LGBM bagging/feature, train/test split, bootstrap).
- Records seed list at top of output JSON.
- Records framework version (`git rev-parse HEAD`) at output time.
- Records the chimera parquet checksum/mtime at output time.
- Pins library versions in a `requirements.txt` snapshot in `runs/oracle/repro_env_<date>.txt`.

### REP2 — Re-run command in every report

Every audit report (markdown + JSON) carries a `## Reproduce` section with:

```
## Reproduce this result

Source revision:  <git SHA at output time>
Chimera file:     data/processed/chimera/4h/pepeusdt_v51_chimera_4h_20260522.parquet
Chimera mtime:    2026-05-24 04:17 SAST
Chimera SHA256:   <sha256 of parquet>
Seeds:            [0,1,2,3,4,5,6,7,8,9]
Library env:      runs/oracle/repro_env_2026-05-25.txt (or sibling)

Command:
  python scripts/wealth_bot/<script>.py --config <yaml> --seeds 0-9
Expected output: matches runs/audit/<run>/data/<file>.json byte-for-byte

If output differs: chimera was rebuilt OR framework code changed since this report.
Investigate which; do not silently update the baseline.
```

### REP3 — Output JSON schema (binding minimum)

Every audit JSON contains:
- `wall_clock`: ISO timestamp
- `git_sha`: revision at run time
- `chimera_file`: path
- `chimera_sha256`: hex digest
- `seeds`: list
- `config_yaml_path`: source config
- `config_snapshot`: full inlined YAML contents (so config drift is detectable)
- `result_set`: the actual metrics

The `config_snapshot` makes the JSON self-contained: anyone can re-derive the run without external state.

### REP4 — Bit-exact replay verification

Before any L2/L3 ship, run the replay command on a clean checkout, diff the new JSON against the committed JSON. Bit-equal = ship. Diff = INVESTIGATE BEFORE SHIPPING (the chimera or the code changed; figure out which).

Helper utility to build: `scripts/wealth_bot/_repro_helper.py` — wraps the above into a single `verify_reproducibility(audit_json_path)` function. Queued for next session.

### REP5 — Result-set reproducibility (not just code)

The framework's result set IS PART of the framework. Reproducibility means:
- Same input data + same seeds + same config + same code → identical output JSON
- The paper-trade journals (`runs/audit/<run>/journals/DEPLOY_*.jsonl`) are also reproducible artifacts. They must include the chimera SHA and seeds in their header.

---

## §TI x Asset Dossier convention `[ALL]` (binding — 2026-05-25 r3, user mandate)

> *"everything should be documented relative to the TI x [ASSET] — there should already be a PEPE document. Each instance is to maintain their own document."*

Every (TI, Asset) pair has a canonical dossier markdown file. Each parallel instance maintains its OWN dossier under its own filename — instances are "competing collaborators" who can learn from each other.

### TID1 — Dossier path convention

```
docs/dossiers/<TI>_<ASSET>__<instance_tag>.md
```

Example: `docs/dossiers/EMAMA_PEPE__instA.md`, `docs/dossiers/EMAMA_PEPE__instB.md`.

Pre-existing aggregate dossier (per user note): `runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/` contains the multi-instance audit trail. New per-instance dossiers go in `docs/dossiers/`.

### TID2 — Dossier required sections

```
# <TI> × <ASSET> Dossier — Instance <tag>

> Owner instance: <tag>
> Started: <date>
> Last update: <date>
> Live? YES (chimera+seeds reproducible) / NO

## §1 Current canonical baseline (Phase 1)
- Config path
- UNSEEN compound + pace + gap-to-floor table
- Robustness numbers (10/10 seeds, p05, max_dd)

## §2 Phase 2 history
- All rounds with shipped/null verdict
- Cross-reference PHASE2_CALIBRATION_LEDGER.md

## §3 Phase 3 expansion attempts
- Each new tuple tried, result, why it superseded or was discarded

## §4 Refuted hypotheses (cross-reference WEALTH_BOT_FAILURE_CATALOG.md)

## §5 Non-linear back-jumps
- When/why phase bouncing occurred, what was learned

## §6 Cross-instance observations
- What we learned from the OTHER instance's dossier
- Where we agreed/disagreed/competed

## §7 Reproducibility manifest
- Git SHA, chimera SHA, seed list, library env file path
- One-line replay command
```

### TID3 — Cross-instance learning (no-conflict rule)

The user mandate: *"at any time I'll never have more than 2 separate instances working on the same problem — even if I do, never worry about getting in each other's way. You guys can learn from each other and infer from each other — you're competing collaborators."*

- Each instance writes ONLY to its own dossier file. No edit-collisions.
- Each instance may READ the other instance's dossier (and SHOULD, before any L2 decision).
- A "cross-instance observation" entry in §6 cites the other instance's dossier file:line.
- If two instances reach contradictory conclusions, the dossiers DOCUMENT the contradiction — neither auto-wins. User resolves at next checkpoint.

---

## §Parallel-Instance Coordination `[ALL]` (binding — 2026-05-25 r3)

Replaces / refines the MAXX-MULTI-DIALECT section for the wealth_bot context.

### PIC1 — Handshake file (when 2 instances run simultaneously)

```
runs/coordination/HANDSHAKE_<YYYY_MM_DD>.md
```

Append-only, instance-tagged. Each instance posts at session-start + every 30 min:

```
## [HH:MM TZ] INST-<X>: <topic>
Phase: <1/2/3>  Round: <ID>
Working tuple: (<Instrument>, <Indicator>, <Approach>)
Posterior on current claim: P(BULL)=X.X P(BEAR)=Y.Y P(NULL)=Z.Z
Next experiment: <1 sentence>
Dossier link: docs/dossiers/<TI>_<ASSET>__inst<X>.md#section
---
```

### PIC2 — File-ownership boundaries

- Each instance OWNS its dossier file. No other instance edits it.
- Shared artifacts (`CLAUDE.md`, `WEALTH_BOT_DEVELOPMENT_FRAMEWORK.md`, calibration ledger, leaderboard, failure catalog) are APPEND-ONLY where possible. Where they require modification (e.g., framework rev), the instance making the change records it in the handshake.
- New scripts (`scripts/wealth_bot/*.py`, `scripts/oracle/*.py`) can be created freely; commit them per the canonical commit-per-version rule.

### PIC3 — Competing collaborators (the productive disagreement)

The two instances may produce DIFFERENT canonical baselines for the same (TI, Asset). That's OK — the leaderboard ranks them on the SAME UNSEEN window; the user picks the winner. Until then both ship to the leaderboard with full reproducibility manifests.

---

## §Phase 4 — Within-TI Regime Composition `[ORACLE]` (binding — 2026-05-25 r5, user mandate)

> Provenance: 2026-05-25 verbatim — *"when I say different regimes, you have to find a strat within the TI that works more robustly in that different regime (self-regime is what I think is best, but I might be wrong)."*

When Phase 1-3 within-TI work has surfaced ≥2 candidate configs that win in DIFFERENT regimes (e.g., EMA(7,15) + whale wins trending regimes; EMA(20,50) + bd_imb wins chop regimes), Phase 4 composes them into a regime-routed bot. **All composed configs are SAME-TI**. Never cross-TI.

### P4.1 — When Phase 4 fires

- ≥2 within-TI configs each have non-overlapping winning regimes
- Phase 3 expansion has surfaced regime-conditional signals (Phase 3 axes E or I)
- The (TI, ASSET) dossier has not yet closed AND single-config winners miss the FLOOR target

### P4.2 — Regime detector candidates (test all 3, pick by capture rate)

1. **Self-regime** (indicator's own internal state defines the regime)
   - Within MA/EMA: `|fast_ma − slow_ma| / close` = trend-strength; high = trending, low = chop
   - Within MACD: signal-line distance, histogram size
   - Within RSI: where in the 30-70 band
   - Pro: comes "for free" from the indicator math; no external feature
   - Con: may correlate too strongly with the signal itself (regime + signal entangled)

2. **External regime** (asset / market state, independent of TI)
   - BTC 30d return sign (bull / chop / bear)
   - Asset volatility regime (rolling std percentile)
   - Time-of-day / day-of-week
   - Microstructure (whale intensity, basis sign, funding extremes)
   - Pro: structurally independent of the TI
   - Con: extra feature engineering; may be noisier than self-regime on short samples

3. **Hybrid regime** (self + external composite)
   - Self-regime AS PRIMARY axis, external AS SECONDARY confirmation (e.g., self-trending AND BTC bull)
   - Pro: usually strongest empirically (combines orthogonal signals)
   - Con: more search dimensions; higher overfit risk on short UNSEEN

### P4.3 — Procedure

1. Catalog all SHIPPED + INCONCLUSIVE within-TI configs from the dossier's Phase 1-3 ledger
2. For each config, compute its WIN regimes (regimes where capture_L3 > 0.50 AND realized > 0)
3. For each of the 3 regime-detector families, train a router on TRAIN+VAL+OOS that dispatches the highest-expected-capture config per detected regime
4. Apply to UNSEEN; report composed compound + per-regime capture rates
5. Robustness gates apply: composed bot must pass 10-seed positive + 4-window positive + p05 > 0
6. SHIP composed IF composed UNSEEN compound ≥ best single-config UNSEEN compound + 10pp

### P4.4 — Phase 4 anti-patterns

- **Crossing TI families**: NEVER. Phase 4 routes BETWEEN MA/EMA configs only (when in PEPE × MA/EMA dossier). To route between MA/EMA AND MACD requires both having closed dossiers AND meta-leaderboard capital allocation, NOT Phase 4.
- **Regime detector overfit**: pre-register the regime threshold sweep on TRAIN+VAL only; UNSEEN is final holdout.
- **Composition complexity creep**: max 3 within-TI configs in the composition. >3 = search-space explosion AND maintenance cost.

### P4.5 — Reference implementation queued

- `scripts/wealth_bot/_regime_router.py` — within-TI regime-routing engine (queued)
- `scripts/oracle/within_ti_regime_miner.py` — catalogs each config's winning regimes (queued)
- Both build on `src/wealth_bot/framework/repro.py` for reproducibility

---

## §Standard Dossier Report Format `[ORACLE]` (binding — 2026-05-25 r5, user mandate)

> Provenance: 2026-05-25 verbatim — *"check how the other instance reported, and how you reported, and combine the reporting into one standard reporting framework. What caught my attention that side was the §E All 14 manifest dimensions (status grid). It had holes, and so I asked for further exploration."*

Every (TI, ASSET) dossier MUST render the following sections in order. Skipping any section is a discipline gap.

### SR1 — Manifest Dimension Status Grid (NEW, mandatory — closes the "holes hidden" defect)

Every dossier renders an EXPLICIT STATUS GRID covering all 14 manifest dimensions (A–N from §Exploration Manifest), Phase-3 expansion axes, and Phase 4 composition. Status legend:

- `[SHIPPED]` — explored, candidate(s) shipped to leaderboard
- `[REFUTED]` — explored, all candidates failed honest validation
- `[INCONCLUSIVE]` — explored, results sample-size-bound; queued
- `[PARTIAL]` — partially explored (some variants tested, others not)
- `[NOT EXPLORED]` — untouched; HOLE in the dossier
- `[N/A]` — dimension not applicable to this (TI, ASSET) tuple (must justify)

### SR1.1 — Honest Sub-Dimension Exhaustion % (NEW, mandatory — closes the "depth-collapse" defect)

> Provenance: 2026-05-25 user mandate verbatim — *"I should have said 'all 14 dimensions TOUCHED, with ~25-35% sub-dimension coverage per dimension' — not 'EXHAUSTED.'"* (from instance F96BE75A's reply table) + *"And yes, integrate the honest exhaustion"*.

SR1 (status flag) is the BREADTH metric — "did the axis get touched at all?". SR1.1 is the DEPTH metric — "within each touched axis, what % of in-scope sub-variants got tested?". Without SR1.1, the round log can carry "REFUTED" on an axis where only 3 of 20 possible sub-variants were tested — that's a partial sample wearing a confident label.

Every dossier's §1A status grid carries 5 additional columns per dimension:

| Column | Definition |
|---|---|
| **Variants tested** | Concrete list of variants/configs actually run for this dim |
| **Untested family-level** | Sub-families from the manifest that received ZERO testing |
| **Within-tested-family depth** | For the families that WERE touched: how deeply (parameter sweep size, threshold grid breadth) |
| **Honest %** | Numeric estimate of in-scope sub-variant coverage (0-100%) |
| **Highest-EV untested** | Single highest-prior untested sub-axis with mechanism note + cost-to-test |

**Three aggregation views REQUIRED in the dossier**:
1. Simple average across all dims (gives the literal coverage %)
2. Weighted by "where alpha historically came from" (which dims drove SHIPPED candidates)
3. Weighted by "where untested sub-axis could plausibly add lift" (forward-looking EV-weighted)

**Closure threshold (updated with SR1.1, raised 2026-05-25 r5.1 per user mandate)**: a dossier may NOT declare CLOSED if EITHER:
- Any dim is `[NOT EXPLORED]` (SR1 breadth gap) — **OR**
- Aggregate honest exhaustion % < **80%** AND any floor MISSED (SR1.1 depth gap)

**Canonical exhaustion floor: 80%** (target band 80-90%). User mandate verbatim: *"I want > 80 - 90% to consider exhaustion (Not that there are infinite ideas means we'll never reach that, but canonically that should be the coverage)"*. The search space is theoretically infinite — 80-90% is the canonical operational bar that distinguishes "thoroughly swept" from "surface-swept".

A dossier with all dims TOUCHED but ~28% mean depth + missed floors is **NOT exhausted by an order of magnitude**. It's surface-swept. Reopen with the top-5 untested sub-axes per §1B until exhaustion ≥ 80% OR all floors MET.

**Honest reframe (binding language)**: replace "EXPLORED" / "EXHAUSTED" with "TOUCHED at X% sub-dim coverage". The narrative shifts from "we've tried everything" to "we've tried X% of the search space; here's the highest-EV remainder".

### SR1.3 — Mechanism-Verification Rule (BINDING 2026-05-25, closes the "mechanism-claim-false" defect)

**Rule**: every dossier ship-candidate that includes a mechanism explanation (text of shape *"filter strips X"* / *"this works because Y"* / *"the reason it's robust is Z"*) MUST be paired with an empirical trade-level falsifier check BEFORE the candidate is promoted to SHIP, INCONCLUSIVE-near-SHIP, or featured in the leaderboard.

**Falsifier check** (canonical fields in audit JSON, enforced by `src/wealth_bot/framework/claim_contract.py`):

```json
{
  "per_trade_returns_sorted_desc": [...],
  "top_3_pct_of_compound": <float>,
  "jackknife": {"K=0": ..., "K=1": ..., "K=2": ..., "K=3": ..., "K=5": ...},
  "combined_K2_plus_S9_pct": <float>,
  "mechanism_claim": "<text>",
  "mechanism_falsifier_check": {
    "what_filter_keeps": [<trade indices kept>],
    "what_filter_drops": [<trade indices dropped>],
    "verified_by": "<auditor-identity + timestamp>"
  }
}
```

**Binding gate**: if `top_3_pct_of_compound > 70%` AND `n_unseen < 30` AND `mechanism_falsifier_check.verified_by` is unset, the candidate CANNOT be shipped, and the CDAP pre-commit (`src/audit/check_wealth_bot_claims.py`) will FAIL with exit 2.

**Sample-size discipline (revised)**: the n<20 → +25pp escalated ship threshold is compared against `min(baseline_compound, combined_K2_plus_S9_compound)`, NOT against `baseline_compound` alone. A candidate that clears baseline-vs-threshold but fails stressed-vs-threshold is INCONCLUSIVE not SHIP.

**Provenance**: 2026-05-25 INST-A session — P4_route_basis_pos_only was nearly promoted as "first to pass combined K=2 + S9 stress at +2.25%" with the mechanism claim that the basis≥0 filter strips top-tail-dependent trades. RED-team audit traced the trade list and found the filter KEPT the same top 3 trades as ABC_AND (+24.71/+23.57/+21.63%) and DROPPED 10 diversifying smaller trades — the OPPOSITE of the claim. The +2.25% stress-pass was one-trade-away-from-collapse (K=3 → −5%). Pattern R + Q codified in `memory/fix_logs/INDEX.md`. Real-capital downside: silent inflated headline shipped through three artifacts (leaderboard, learnings, closure) before audit caught it.

**Layered enforcement** (defence-in-depth, all 6 layers):

1. **CDAP pre-commit** (`src/audit/check_wealth_bot_claims.py`) — physical block on commit
2. **Audit JSON contract** (`src/wealth_bot/framework/claim_contract.py`) — required fields at write-time
3. **Auditor brief items 8-12** (`runs/coordination/AUDITOR_FINDINGS_<date>.md`) — RED-team checks per dispatch
4. **Memory patterns Q + R** (`memory/fix_logs/INDEX.md`) — knowledge transfer to future instances
5. **TURN_END_CHECKLIST item 4** (`.claude/skills/_common/PROTOCOL_COMPOSITION.md`) — META discipline at decision-points
6. **Cross-instance handshake** (`runs/coordination/HANDSHAKE_<date>.md`) — multi-instance coordination

A new wealth-bot dossier instance encountering ANY of these gates must address the finding in-session before reporting. CRIT halt → fix-before-next-round. HIGH/MED → fix within session.

### SR1.2 — §1B Highest-EV Untested Sub-Axes (NEW, mandatory companion to §1A)

For every dossier, render a ranked table of the TOP-5 highest-EV untested sub-axes that emerge from the SR1.1 column. Each row: rank / sub-axis / mechanism / dim / cost-to-test. This is the queue for "go deeper instead of starting a new TI cycle".

The §1B table is the DECISION SUPPORT for the user choice between:
- (a) Deepen the current (TI, ASSET) via top-5 untested sub-axes
- (b) Close + escalate to a new (TI, ASSET) dossier per EP3

Example status grid format:

| Dim | Name | Status | Round(s) | Best result | Notes |
|---|---|---|---|---|---|
| A | Bar generation (chart types) | REFUTED | R9, R9b | dollar-bars baseline | HA/Renko/tick/volume/range all underperform |
| B | Time frame | REFUTED | R11 | 4h dominant | multi-cadence ensemble killed by 1d drag |
| C | Indicator config | SHIPPED | R12 | WMA(10,30) | 510-config sweep + iteration |
| D | Filter family | SHIPPED | R12 | whale_net>0 | bd / fund / premium tested |
| E | Exit policy | REFUTED | R7 | opposite-cross baseline | 21 variants tested |
| F | Position structure | **NOT EXPLORED** | — | — | **HOLE** |
| G | Sizing methodology | REFUTED | R8 | Kelly 1.0 perp | 19 variants |
| H | Execution / fill realism | **NOT EXPLORED** | — | — | **HOLE** |
| I | Risk-mgmt triggers | **NOT EXPLORED** | — | — | **HOLE** |
| J | Goal function | METHODOLOGICAL | R14 | G_compound confirms | 5 metrics tested |
| K | Instrument variant | SHIPPED | R12 | 1000PEPEUSDT perp | spot also live |
| L | Phase-specific inputs | N/A | — | — | |
| M | Signal aggregation | **NOT EXPLORED** | — | — | **HOLE** |
| N | Filter aggregation | **NOT EXPLORED** | — | — | **HOLE** |
| P3 | Phase 3 expansion axes (within-TI only) | SHIPPED-via-R12 | R5-R14 | 10 axes covered | full enumeration above |
| P4 | Phase 4 within-TI regime composition | **NOT EXPLORED** | — | — | **HOLE** |

**Binding rule (anti-hole)**: a dossier CANNOT declare COMPLETE if any dimension is `[NOT EXPLORED]` AND any (TI, ASSET) target floor remains MISSED. The status grid surfaces the gap; closure is gated on filling it.

Holes are acceptable ONLY when:
- Every floor is MET (≥1%/d AND ≥2%/3d AND ≥3%/week) — no expansion needed, OR
- The dimension is `[N/A]` with justification cited

### SR2 — Standard sections (in order)

Every dossier renders these sections, in this order:

1. **§Header** — TI, ASSET, owner instance, status (ACTIVE / CLOSED), framework rev
2. **§Scope** — in-scope / out-of-scope / windows / targets
3. **§Exploration Manifest** — the 14 sub-dimensions (A-N) + Phase 3 axes + Phase 4
4. **§Status Grid** — SR1 above (the holes-detector)
5. **§Frameworks in scope** — Static / ML / Oracle roles
6. **§Phase model** — non-linear bouncing diagram
7. **§Round log** — append-only R1...Rn with status
8. **§Per-Layer KPI Report (L0-L6)** — current canonical baseline's layer_kpis block (per LAYER_DECOMPOSITION_TEMPLATE.md)
9. **§Capture Rate Hierarchy (3 levels)** — L1/L2/L3 capture rates with interpretation
10. **§Pace-to-Floor Gaps** — current pace + gap to each floor (per `_pace_conversion.py`)
11. **§Calibration Ledger Summary** — per-round Phase 2 ledger entries summarized
12. **§Refuted Hypotheses** — failure catalog tagged by target layer
13. **§Side-Findings Queued** — INCONCLUSIVE results blocked on sample-size
14. **§Deploy Decision (terminal, when closing)** — primary + fallback + caveats
15. **§Pre-Live Verification List** — items that must clear before capital allocation
16. **§Reproducibility Manifest** — git SHA, chimera SHA, seeds, replay command
17. **§Cross-Instance Handoff** — recommended next (TI, ASSET) + queued items
18. **§Framework Alignment Notes** — any retrofits / conformance gaps closed
19. **§Honest Final Verdict** — Q&A summary (did it deploy / floor met / search exhausted)

### SR3 — Why the Status Grid is the headline-most-important section

The status grid is what separates "complete" from "looks complete":
- A dossier with 10 axes refuted + 5 holes is NOT complete; it's premature-closure
- A dossier with all 14 dimensions explored (any status, including REFUTED) IS complete
- Floor-met OR full-coverage are the only two valid closure conditions

Without the grid, the round log can carry an impressive count of explored axes while still leaving structural holes. The grid forces the holes to be NAMED before "complete" is claimed.

### SR4 — Retroactive application

Existing dossiers without an SR1 status grid MUST add one before they can be cited as "complete". The (PEPE, MA/EMA) dossier was instance F96BE75A's reopen trigger (2026-05-25 user mandate): the 5 holes (F, H, I, M, N + Phase 4) were named via user prompt, not surfaced by the dossier itself. R15-R20+ rounds are now exploring those holes per user mandate.

### SR5 — Template

The dossier starter at [`docs/dossiers/_TEMPLATE__inst_starter.md`](../docs/dossiers/_TEMPLATE__inst_starter.md) MUST embed the SR1 status grid skeleton with all 14 dimensions seeded as `[NOT EXPLORED]`. As rounds complete, the status flips. New dossiers START with the grid intact.

---

## §Gold-Standard Dossier `[ORACLE]` (binding — 2026-05-25 r5.2, user mandate)

> Provenance: 2026-05-25 user mandate verbatim — *"PEPE × EMA/MA is going to be our gold standard for all work going forward, so we cannot take it for granted"*.

A **gold-standard dossier** is the canonical reference (TI, ASSET) that all other dossiers benchmark against. The first wealth-bot dossier — (PEPE, MA/EMA) — is designated as the gold standard by user mandate. Implications:

### GS1 — No closure shortcut

Gold-standard dossiers MUST meet the FULL closure threshold (SR1 breadth = all dims status != NOT EXPLORED, AND SR1.1 depth = honest exhaustion ≥ 80%). The "all floors MET" escape hatch DOES still apply, but gold-standard dossiers are expected to keep deepening even after floors are met — because they set the canonical bar.

**Termination tag** (`TERMINATED — budget-bounded`) is NOT permitted for gold-standard dossiers. Only `CLOSED — EXHAUSTED` or `ACTIVE — DEEPENING IN PROGRESS`.

### GS2 — Multi-sprint commitment

Gold-standard deepening is multi-sprint by definition. Typical pattern: §1B top-5 untested sub-axes queued sequentially over weeks/months, each round producing R21 → R22 → ... entries in the calibration ledger. The dossier remains ACTIVE — DEEPENING IN PROGRESS until the 80% exhaustion bar is genuinely cleared.

### GS3 — Cross-TI benchmarking anchor

When OTHER (TI, ASSET) dossiers close (e.g., (PEPE, RSI), (PEPE, MACD)), they are compared against the gold-standard on the same UNSEEN window. The gold-standard's numbers — particularly its honest exhaustion %, capture rate hierarchy, and within-TI ceiling — become the canonical comparison reference. If a new-TI dossier closes at ≥ gold-standard depth but FAR below gold-standard compound, the asset may be the constraint; if at ≥ gold-standard depth AND higher compound, the prior gold standard is supplanted.

### GS4 — Documentation premium

Gold-standard dossiers carry richer documentation:
- Every refuted hypothesis explained at MECHANISM level (not just "REFUTED at G5")
- Every untested sub-axis has prior probability + cost estimate
- §1B top-5 untested ranked + re-ranked at each completion of a sub-axis
- All reproducibility manifests fully complete (seeds, chimera SHA, lib env, replay command tested)

The gold-standard sets the bar for what "done" looks like across the project.

### GS5 — Current gold-standard designation

| Tuple | Status | Honest exhaustion | Target |
|---|---|---:|---:|
| (PEPE, MA/EMA) | **ACTIVE — DEEPENING (r5.2 reopen 2026-05-25)** | ~28% | ≥80% |

Reopen trigger: user mandate 2026-05-25 — *"because PEPE × EMA/MA is going to be our gold standard, we cannot take it for granted"*. Path-(a) chosen over path-(b) (terminate + escalate). §1B top-5 untested sub-axes are the immediate queue:

1. Distance-from-MA signal (Dim C, 1-2h)
2. hbr_eta_buy filter (Dim D, 30 min)
3. HMM regime detector for Phase 4 (Phase 4, 2-3h)
4. Imbalance bars (Lopez de Prado) (Dim A, 2-3h)
5. Filter cascade whale → bd_bgf → fund_rate (Dim N, 30 min)

Subsequent rounds will surface new sub-axes as each tier exhausts. Estimated 3-5 weeks of deepening to genuinely clear 80%.

---

## §No-Cross-Pollination Rule `[ALL]` (binding — 2026-05-25 r5, user mandate)

> Provenance: 2026-05-25 verbatim — *"we never cross contaminate a single indicator (and all the possible config) and its asset."*

Within a (TI, ASSET) dossier:

| What | Allowed | Not allowed |
|---|---|---|
| Different parameter values (e.g., EMA(7,15) vs EMA(20,50)) | ✓ within-TI | — |
| Different signal-rule variants (cross, state, bounce) | ✓ within-TI | — |
| Different filter combinations | ✓ within-TI | — |
| Different cadences of SAME TI rule | ✓ within-TI | — |
| Different chart-types of SAME TI rule | ✓ within-TI | — |
| Different instrument variants (spot, perp) of SAME asset | ✓ within-TI | — |
| Mixing TI families (e.g., EMA + RSI in same bot) | — | ✗ separate (TI, ASSET) dossier |
| Phase 4 routing between EMA-config and MACD-config | — | ✗ violation; close MA/EMA dossier first |
| Cross-talk handoffs across TI families | — | ✗ within-TI children only (Static ↔ ML in SAME dossier) |
| Meta-leaderboard ranking across TIs | ✓ after dossiers close | only as separate sleeves; NEVER ensembled into one bot |

**Cross-talk on CROSS_LAYER_HANDOFFS.md**: HANDOFFS are SAME (TI, ASSET) cross-child only (Static ↔ ML within (PEPE, MA/EMA), etc.). NEVER cross-TI. If an insight from (PEPE, MA/EMA) might apply to (PEPE, RSI), it becomes a NOTE in the NEW (PEPE, RSI) dossier when started — not a handoff in the old dossier.

---

## §Sample-Size Discipline `[ALL]` (binding — 2026-05-25 r2)

Small-sample noise is the dominant Phase 2 failure mode. Rules:

### SS1 — UNSEEN-trade-count threshold for honest mining

Phase 2 mining produces RELIABLE candidates only at:
- ≥ 20 UNSEEN trades (current 1-strat has 9 — sub-threshold)
- ≥ 12 losing trades (need power to distinguish loser-cluster patterns from noise)

Below threshold: any single-feature claim has effect-size CI too wide to validate. The framework still RUNS Phase 2 (the structure has value) but DEFAULTS to NULL more aggressively, and explicitly notes the sample-size limitation in the calibration ledger.

### SS2 — Power-aware ship thresholds

When UNSEEN n < 20:
- Ship requires honest UNSEEN gain ≥ +25pp (vs +10pp at n ≥ 30) — tighter to compensate for higher false-positive risk
- Cross-window persistence AUC > 0.65 (vs > 0.60) in all 3 pre-UNSEEN windows
- Bootstrap p05 of refined-bot UNSEEN compound > baseline UNSEEN compound (not just > 0)

### SS3 — Wait-or-Refine choice

When Phase 2 returns NULL at n < 20:
- Option A — WAIT: collect 10-20 more UNSEEN trades via paper-trade, re-run Phase 2 next month. Cheaper, lower-cost-of-being-wrong.
- Option B — REFINE NOW with looser thresholds: ONLY if the user explicitly authorizes accepting higher false-positive risk.
- Default: Option A.

---

## §Wall-Clock Budget Discipline `[ALL]` (binding — 2026-05-25 r2)

Phase work has explicit time bounds. Beyond bound = scope drift.

| Phase | Wall-clock budget | When over-budget |
|---|---|---|
| Phase 1 — Robust Discovery | ≤ 1 dev-day per (Instrument, Indicator, Approach) tuple | Halt, audit what's blocking, write what was tried, ship null or escalate |
| Phase 2 — Oracle Mining | ≤ 4 hours wall-clock | Halt at 4h, ship whatever's been honest-validated, defer rest |
| Phase 2 — Honest Validation | ≤ 1 hour | If a refinement's validation takes >1h to compute, the refinement is too complex — refactor or refute |
| Stretch-Goal Worker | ≤ 30 min | Worker returns 3 ranked proposals or ship null with documented exhaustion |
| Saturation Declaration | ≤ 15 min | Triggered after 3-NULL cascade, no debate, ship the declaration |

Track wall-clock per task. Over-budget tasks add a row to `runs/oracle/WEALTH_BOT_WALLCLOCK_LEDGER.md` with reason — pattern of over-runs surfaces process bugs.

---

## §Anti-Advocacy Protocol `[ORACLE]` (binding — 2026-05-25)

The Phase-1 author has an inherent advocacy bias toward defending their baseline in Phase 2. Five mechanisms close the trap, exploiting the parallel-agent capability already in the system (no need for different sessions / instances).

### M1 — The imagine-frame (PRIMARY mechanism, zero protocol cost)

When entering Phase 2, adopt this cognitive reframe **verbatim**:

> *"Imagine Phase 1 was done by a different instance, not you. You are receiving their bot as a stranger receives a candidate. You owe them no defense. Your only loyalty is to the data. Find what they missed."*

This is one rule. No infrastructure. No worker spawns. The reframe flips the agent's default posture from "validate my work" to "stress-test someone else's work" — which is exactly the posture Phase 2 needs.

**Why this is enough for L1/most L2 rounds**: advocacy bias is a posture problem, not a structural one. Posture changes with reframing. Engineering the structural fix (separate sessions, parallel-workers) is overkill when the cognitive fix suffices.

**Provenance**: 2026-05-25 user mandate verbatim: *"and when you get to phase 2, it's simple: imagine the phase 1 was done by another instance, not you. This should remove any limiters introduced by bias."*

### M1-ESCALATION — Parallel-challenge worker pipeline (use only when L3 / capital-at-stake)

Reserved for cases where the candidate refinement, if shipped, materially changes deployed capital allocation AND the META has high prior conviction in the baseline (advocacy bias likely strong). Then spawn two Opus workers with opposite incentives:

- **Worker A — Challenger**. Brief begins: *"The +X% baseline is the FLOOR, NOT the ceiling. Find what breaks the ceiling higher. There IS a higher ceiling — find it."* Mines refinements, returns candidates only.
- **Worker B — Refuter**. Brief begins: *"You receive candidate refinements. You do NOT know what mining produced them. Default verdict: REFUTE."* Honest-validates blindly.

This is the structural belt-and-suspenders for L3 decisions. Default for L1/L2 is M1 alone.

### M2 — Pre-registration before spawning

Before Worker A spawns, write `runs/audit/<run>/PHASE2_PREREGISTRATION.md` (immutable). Contains:
- Baseline state (compound, n_trades, robustness numbers)
- Worker A's frame (verbatim challenge string)
- Worker B's frame (verbatim refuter string)
- Pre-registered ship thresholds (numeric, all-must-hold)
- Pre-registered refute thresholds (numeric, any-trips)
- Pre-registered mining query list (fixed, Worker A may not add/skip)
- Asymmetric loss statement
- Calibration ledger row to be appended

Pre-registration prevents post-hoc threshold tuning. Any deviation requires explicit user override and goes in the calibration ledger as a process violation.

### M3 — Asymmetric loss function (explicit)

- **False-positive (ship overfit)**: real capital + audit-discipline erosion → PRIORITY 1
- **False-negative (miss real alpha)**: opportunity over 1 month until re-validation → PRIORITY 2 (recoverable)
- When ship vs null is borderline, **default NULL** + queue re-validation next round (with more UNSEEN paper-trade samples). The conservative bias is intentional.

### M4 — Pre-registered mining queries

Worker A receives a fixed list of mining angles, may NOT add or skip. Standard 3-angle template:
- LOSER-side: characterize losing-trade structure across TRAIN+VAL+OOS+UNSEEN, propose block-only filter
- MISSED-MOVE side: bars with fwd_ret ≥ +5% where bot didn't fire, propose admit-only filter
- EXIT-side: 7-bar in-hold path patterns, propose exit modifier

Fixed queries prevent cherry-picking — agent can't skip the angle that would surface defects.

### M5 — Project-level calibration ledger

`runs/oracle/PHASE2_CALIBRATION_LEDGER.md` — append-only. Each Phase 2 round adds: date, baseline, candidates surfaced, validated, shipped (Y/N), reason. Healthy rate: 20-40% of rounds ship modifications. Outside that band, investigate discipline:
- Always-null → over-strict gates / advocacy bias dominant
- Always-ship → rubber-stamp / asymmetric loss not honored

### Worked example (2026-05-25 R2)

See `runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/PHASE2_PREREGISTRATION.md` + the Worker A/B outputs + the calibration ledger entry. The R1 round was META-briefed and proven vulnerable; R2 was the first to use the full parallel-challenge protocol.

---

## Anti-patterns (red flags) `[ALL]`

1. **Oracle-only confidence**: "the mining said +28pp, ship it." Without Step 2.3 you ship overfit noise.
2. **Phase-1-only complacency**: "10/10 seeds positive on UNSEEN, ship." Without Phase 2 you miss structural defects (missed-alpha / late-entries / inefficient exits) that pure compound % doesn't expose.
3. **Tuning on UNSEEN**: pulling UNSEEN into ANY threshold search collapses the test. The seductive form: "let me see how the refinement does on UNSEEN first, then tune."
4. **Composite scoring across windows**: e.g., "average compound across TRAIN+VAL+OOS+UNSEEN". UNSEEN must be a final holdout, not a fit input.
5. **Per-refinement re-mining**: re-running the miner with the refinement applied creates a recursion where each iteration finds a new "defect" that's actually the prior round's induced curvature. Mine once, validate, ship — don't iterate.
6. **Pre-defined refinement set**: the miner must SURFACE hypotheses. If you write "I'll test F1=tighter_whale_thresh first", you've biased the miner.
7. **Skipping null reports**: a refuted refinement IS a valuable artifact. Document and save it. Future instances need to see what was tried and refuted.

---

## Recurring cadence `[ALL]`

- **Quarterly** or after any data-pipeline change (chimera rebuild, new feature family): re-run Phase 1 audit on the deployed bot. Verify the baseline still passes the robustness gates.
- **Monthly** (or after collecting ≥20 new UNSEEN paper-trade fires): re-run Phase 2 oracle mining with the expanded sample size. Patterns that were small-sample noise at n=9 may surface as real alpha at n=50.
- **On structural finding**: any time a refinement survives honest validation, the bot becomes the new baseline; restart the cycle.

---

## Cross-references

- WEALTH objective: [PROJECT_NORTH_STAR.md §3.1](../PROJECT_NORTH_STAR.md)
- Framework code: [src/wealth_bot/framework/](../src/wealth_bot/framework/)
- Oracle primitives: [scripts/oracle/](../scripts/oracle/)
- This session's artifacts:
  - Decision-context mining: [runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/ORACLE_TRADE_DECISION_AUDIT.md](../runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/ORACLE_TRADE_DECISION_AUDIT.md)
  - Honest validation: [runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/HONEST_REFINEMENT_RESULTS.md](../runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/HONEST_REFINEMENT_RESULTS.md)
  - Cross-window persistence: [runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/PERSISTENT_PATTERN_MINING.md](../runs/audit/AUTONOMOUS_MAXX_PEPE_BOT_2026_05_24/PERSISTENT_PATTERN_MINING.md)

---

## §F96BE75A EXPERIENTIAL ADDENDUM `[ALL]` (2026-05-26, INST-F96BE75A)

> **Provenance**: This section is written by INST-F96BE75A after 50+ rounds (R0-R50) on the (PEPE, MA/EMA) gold-standard dossier. The user explicitly mandated *"add your input as someone who has worked extensively and knows strengths and weaknesses and can add input from experience"* at 2026-05-26 ~01:00 SAST. This addendum amends the framework with empirically-grounded clarifications.

### §F.1 Framework intent (canonical re-statement)

The framework exists to **prevent lazy answers given to satisfy the user**. It is not bureaucracy. Every gate (Phase 1 / Phase 2 / Edge-Pushing / Layered KPIs) forces an instance to **earn** its claims with rigor. The framework's value is not closure speed; it is preventing a lazy "yes" that ships a fragile candidate.

**Mandated discipline at ship-time**: after a candidate clears all gates, the instance MUST also articulate the strategy's **strengths, weaknesses, and expected failure modes** in plain language. The user's verbatim mandate (2026-05-26): *"even if a bot is building wealth, we can know what to expect"*. A ship-grade candidate without an explicit weakness profile is incomplete documentation.

**The framework does NOT replace the operator's judgment.** When evidence is ambiguous, the framework forces explicit pre-registration + asymmetric loss (default NULL on borderline). When evidence is clear, the framework gets out of the way. If you find yourself running the full 12-step decision-flow on a trivial decision, that's a tiering error — promote/demote tiers per `_common/PROTOCOL_COMPOSITION.md`.

### §F.2 Deploy model (binding — wallet-size constraint, 2026-05-26 user mandate)

Per user verbatim (2026-05-26): *"wallets will be kept at R10,000 - R250,000 per wallet, per strat if we grow our wealth multi-fold to be small players that win."*

**Deploy unit = ONE STRAT per WALLET, with capital R10k-R250k.** Wealth growth comes from **wallet-count multiplication**, NOT from per-wallet leverage. This has framework consequences:

- **NO leverage > 1.0** (already enforced via CLAUDE.md lev=1 invariant — this is wholly consistent)
- **NO multi-strat-per-wallet ensembling** at the bot level (each wallet = one strat = one (TI, ASSET) candidate). The cross-strat orchestration happens at the **wallet-portfolio layer**, OUTSIDE the bot.
- **NO need for capital-efficiency optimization within a single bot** — the bot solves its own (TI, ASSET) problem; capital deployment is upstream of the bot.
- **Slippage modeling**: at R250k max per wallet, slippage on PEPE-class memecoins is sub-bp on perp taker and sub-5bp on spot. Bots can model `slippage = constant_bps` (no impact term) without inflating ROI claims. At R1M+ wallet sizes, impact modeling becomes required — but R250k cap puts us safely below.
- **Sample-size mandate is REAL**: each wallet's bot has its own n. Pooling wallet streams into one large-n bootstrap is a **violation of independence** unless the bots run different strats and different assets. Honest combined n = per-wallet n.

This deploy model **invalidates** the prior framing of "multi-asset diversification as a bot feature". Diversification is a wallet-orchestration concern, not a bot concern. A bot solves ONE (TI, ASSET) problem; the operator allocates capital across many bots.

### §F.3 Setups > Regimes (re-canonicalized)

Per user verbatim (2026-05-26): *"I chase setups, nothing else."*

A **SETUP** is the unit of trading: `(asset, indicator, config, filter, exit_policy)`. The regime is **metadata about the setup** — not a separate construct to compose with. Implications:

- Phase 4 "regime composition" rounds should be re-framed: regime-conditional configs are **distinct setups**, not the same setup with a different regime hat. Per-regime entries become separate rows in the calibration ledger.
- Regime-detection mining (HMM / BOCPD / K-means clusters) is REFUTED as a wealth source in the (PEPE, MA/EMA) dossier (R20). The pattern likely repeats in other (TI, ASSET) dossiers because **regime detection is value-additive only if the regimes have ORTHOGONAL alpha signatures within the same TI** — which empirically they don't for moving-average families.
- New (TI, ASSET) dossiers should run a setup-discovery pass FIRST, THEN evaluate regime-conditional variants ONLY IF Phase 1 baseline shows clear per-regime alpha differentials. Save the Phase 4 round if it's predictable-NULL.

### §F.4 Timeframe coverage mandate (binding — surfaces a real gap, 2026-05-26)

**The (PEPE, MA/EMA) gold-standard dossier under INST-F96BE75A ran almost exclusively at 4h cadence.** Per user clarification (2026-05-26): *"I thought... that all time frames and time scales were considered. Guess that did not happen."* This is a real framework-coherence gap.

**MANDATE for ALL future dossiers**: each (TI, ASSET) dossier must explicitly cover the cadence dimension as a Phase 3 axis with the following minimum coverage:
- Each of {1m, 5m, 15m, 30m, 1h, 4h, 1d} must be EXPLORED (not assumed-similar to 4h)
- Each cadence-level should produce: a baseline + 1-2 within-cadence variants + an honest UNSEEN compound figure
- The dossier's §1A status grid MUST include a per-cadence row (currently the L0/B dim row is too coarse — it conflates all sub-day with all super-day)
- If a cadence is genuinely intractable (e.g., chimera unavailable), it's marked `STRUCTURAL_NOT_AVAILABLE` with the specific blocker

**Empirical justification**: at the n=18 trade-count constraint in 4h-only PEPE (R48b power analysis), the ship gate is mathematically unbeatable. Switching to 1h (n=72 expected) or 15m (n=288 expected) shifts the gate-MDE 4-16x lower per `delta_p05` scaling — turning the discovered ROI ceiling into a method-artifact ceiling. Sub-cadence coverage is the cheapest hole to fill before declaring closure.

**Doc consequence**: `docs/dossiers/_TEMPLATE__inst_starter.md` § Time-frame coverage matrix should be added as a template section. Existing dossiers should backfill.

### §F.5 Alt-data backlog (curated, 2026-05-26 user-selected)

The user explicitly approved 2 alt-data sources for the queue, rejecting others:

1. **Stablecoin printing → memecoin rotation lag**: USDC/USDT supply expansion events lead memecoin volume by 6-12h on average. This is a TIER-2 alt-data source (not in chimera; requires DeFiLlama or DEX bridge data + lag computation). Add as: `whale_net_xchain_stable_supply_lag` derived feature; first-test as L4 conditioning filter.

2. **Etherscan top-100 wallet net-flow (per-address granularity)**: current `whale_net` is hourly aggregate. Per-address top-K decomposition yields finer signal. **Caveat (user-flagged)**: top wallets include rug-pullers / market makers / known-malicious actors. **Mandatory rug-pull filter**: before using a top-K wallet's signal, the address must pass `(a) age > 90 days, (b) no historical net-flow inversion > $5M, (c) not on community blocklists (e.g., chainabuse.com)`. Without this filter, top-wallet signal is dominated by adversarial actors and reverses sign on regime change.

**Rejected** (NOT in queue): Twitter/Discord sentiment, BTC ETF flow, on-chain accumulation pattern detection beyond whale_net. These were proposed but user vetoed for scope.

**Alt-data implementation order**: stablecoin printing first (data more tractable; chainalysis APIs available), then top-100 with rug-pull filter (filter logic is the load-bearing piece). Both go into Phase 3 L4 conditioning axis of relevant future dossiers.

### §F.6 Strengths / Weaknesses / Expected-failure mandate (binding ship-template)

Per §F.1, every SHIPPED candidate must include a `strengths_weaknesses_expected_failures` block in its config YAML AND its dossier entry. Template:

```yaml
strengths_weaknesses_expected_failures:
  strengths:
    - "What edge does the bot exploit? (mechanism, not just metric)"
    - "What conditions amplify the bot's edge? (vol regime / asset state / market regime)"
    - "What's the bot's robustness profile? (DD ceiling, recovery time, max consec losses)"
  weaknesses:
    - "Under what conditions does the bot underperform baseline?"
    - "What's the worst-case trade scenario (largest single-trade loss)?"
    - "What hidden assumptions does the backtest model that live trading may break?"
  expected_failure_modes:
    - "Regime shift: <specific description>"
    - "Cost regime: <e.g., funding spike > N bps>"
    - "Liquidity: <e.g., venue depletion event>"
    - "Adversarial: <e.g., whale collusion, MEV sandwich>"
  monitoring_triggers:
    - "If <metric> > threshold for N consecutive bars, pause and alert"
    - "If realized DD > X% from peak in M trades, halt and audit"
```

This is NOT about discouraging deployment — it's about **knowing what to expect** before capital is at stake. R12 perp's weakness profile (added retroactively to its YAML): n=18 UNSEEN sample (low statistical confidence), 32h avg hold (slow capital turnover), single-asset concentration (no portfolio diversification within bot), whale_net filter sensitivity (rug-pull risk if top wallets shift).

**r7 extension (mandatory)**: every SHIPPED candidate's YAML must also include a `quality_diagnostic_score_card` block (per §QUALITY-DIAGNOSTIC Q1-Q10 template above). The strengths_weaknesses_expected_failures block and the Q-score-card are COMPANION fields — one gives narrative, the other gives scored diagnostic. Neither replaces the other.

### §F.7 L2 capture-rate instrumentation gap (CRIT-impact framework gap)

The framework defines L2 capture rate as the canonical capital-stripped KPI. **The decomposer was never built.** Per the Sonnet scout report (2026-05-26): `scripts/wealth_bot/_capture_rate.py` exists but L1 within-family sweep is queued — never implemented. NO candidate in any dossier (R12, C2-C8) has a VERIFIED L2 capture rate. All ship decisions to date use compound % as a proxy.

**Action**: build `scripts/wealth_bot/capture_rate_decomposer.py` as Phase 0 instrumentation for ALL future dossiers. Should accept: trade log + chimera + signal-validity rule → returns L1/L2/L3 capture rates per trade with summary statistics. Block all new dossier-closure decisions on the instrumentation being live.

### §F.8 Operational learnings (from 50+ rounds, INST-F96BE75A)

Recurring patterns this instance hit + how to avoid them in future instances:

1. **Trailing-stop / chandelier exit policies are EXTREMELY sensitive to gap-down fill modeling.** Close-only breach detection (`closes[j] <= trail_level`) silently inflates TRAIN compound 100-1000x on gap-prone bars. ALL trailing-style exit code MUST use `lows[j] <= trail_level` for intra-bar detection + `closes[j]` fallback for gap-downs past trail. See commit bed5d2c (R45 chandelier) and Auditor 17 MED-1/MED-2 (R46 trailing-ATR + ratchet-stop) for both fix instances. Add to fix_logs/INDEX.md as **Pattern S: close-only trail breach inflation**.

2. **TRAIN/UNSEEN compound-ratio > 8x is a strong overfit / sim-bug canary.** R12 baseline ratio = 3.9x; R30 G1 honest = 4.3x; R46 E46_1 was 29.7x (smoking gun for harness bug — pre-fix). Any future audit should auto-flag candidates with this ratio for closer investigation BEFORE bootstrap validation.

3. **At n<30, all bootstrap CIs are wide.** The R40b delta-p05>0 gate is structurally unbeatable when both sides are at n=18-25 (compound variance grows multiplicatively with n in compound-pp metric). For small-n adjudication, prefer:
   - **Per-trade Information Ratio** (CLT-scaling, SE ∝ 1/√n)
   - **Cross-window persistence** (split UNSEEN into 3 sub-windows; require ≥2 of 3 compound > 0)
   - **Absolute floor + own-bootstrap p05** (avoid the delta-vs-baseline trap)
   - See R48b power analysis (`runs/audit/.../r48b_n18_power_analysis.json`) for the math; Oracle cross-check (commit 1c9eacb) for the gate-spec critique.

4. **Mid-stream gate-tightening is a discipline mistake.** R12 shipped on absolute floor. Subsequent rounds adopted delta-p05>0 vs baseline — converting wealth-optimization into stat-arb-comparison. **NEVER tighten a ship gate mid-stream within a single dossier.** Either lock the gate at dossier start OR explicitly mark a re-adjudication round when changing gates (e.g., R49b absolute-gate re-adjudication: 1/7 of historical REFUTEDs flipped to PARTIALLY_PASS once the original gate was honored).

5. **Synthesis bootstrap (two-class binary) underestimates true MDE.** Real PEPE returns have positive skew (+15-30% best trades) and fatter left tail than {r_win, r_loss} two-class assumes. Real MDE is ~1.5-2x the estimated MDE. Document this caveat on any bootstrap-based decision at n<30; prefer empirical per-trade-log bootstrap when per-trade data is available.

6. **Permanent auditor cadence works.** Across 17 dispatches in INST-F96BE75A, the auditor caught: R46 trailing-stop harness CRIT-impact bug (would have shipped 2 false SHIP_CANDIDATEs); R47 fabricated sensitivity sweep (would have provided false-confirmation); R44 missing evidence_class tags; multiple methodology drift instances. **Auditor-after-every-commit cadence should be the framework default**, not an opt-in.

6a. **GAP-WINDOW addendum (added 2026-05-26 r7 per user procedural question)**: the current auditor cadence is **"ONE AUDITOR PER COMMIT/BATCH, POST-HOC"** — NOT per-worker, NOT always-on. When N workers dispatch in parallel and each auto-commits its output, the auditor reviews the AGGREGATE master state AFTER all workers have committed. Between worker-commit and auditor-review there is a **GAP WINDOW (5-15 min)** during which META may have already synthesized + reported to the user using inflated/buggy worker numbers. **3 confirmed gap-window incidents in INST-F96BE75A this session**: (i) R46 E46_1/E46_3 SHIP_CANDIDATEs reported as "validated by R47 bootstrap" before Auditor 17 caught the close-only breach harness bug; (ii) R51 E51_1 reported as "REFUTED per R52" before Auditor 19 caught CRIT-1 (gap-down fallback) + CRIT-2 (mid-stream gate switching) — the validator missed the underlying cause; (iii) R54 A54_2 PSEUDO-VB reported without flagging the forward-close look-ahead until Auditor 22 surfaced it.

**Four options to close the gap-window** (ranked by recommended adoption order):

- **Option C — Mandate workers SELF-AUDIT before commit (RECOMMENDED IMMEDIATELY, FREE)**: every worker brief includes a `pre_commit_self_audit:` block requiring (a) Pattern S compliance grep on own harness (`lows[j] <=` count should equal `highs[j] >=` count in trailing exits), (b) G2 all-4-positive check on own gate (no `abs(train/unseen)` collapsing negative-TRAIN), (c) repro-block presence check (env + canonical_seeds + git_sha + chimera_mtime), (d) SIGNED tu_ratio verification. If any check fails, worker fixes BEFORE committing. Catches ~70% of patterns the post-hoc auditor finds at ~5% extra worker time.
- **Option D — CI-style pre-commit hook (RECOMMENDED OVER NEXT SESSION, ~30 min build)**: extend existing CDAP (`src/audit/check_invariants.py`) with Pattern S/T grep + gate-spec consistency check + repro-block-presence check. Would have caught R51's missing gap-down fallback BEFORE the commit landed. One-time cost; saves all subsequent gap-window incidents in this session pattern.
- **Option A — Per-worker auditor pairing (ESCALATED 2026-05-26 r7b: now MANDATORY for MAXX commits per user mandate)**: workers DO NOT auto-commit; return output to META; META runs `per_commit_audit_gate` (per OPERATIONAL_DIRECTIVES.md §UNIVERSAL PRE-DELIVERY SELF-AUDIT Layer 2) BEFORE committing. Verifies all upstream workers reported `pre_delivery_self_audit: PASS`, runs Pattern S/T grep on staged files, verifies repro-block presence. If any check fails: META does NOT commit; dispatches focused auditor on worker output OR fixes inline OR returns to user. **NEW r7b update**: user verbatim 2026-05-26 ~05:15 SAST: *"per-commit audit gate so that anything is fixed before committing"*. Was previously "selective for SHIP_CANDIDATE rounds"; now BINDING for ALL MAXX commits. The cost (~2× audit dispatches) is now considered acceptable given 3 gap-window incidents this session.
- **Option B — Always-on auditor watcher tab (MULTI-DIALECT mode only)**: separate Claude Code tab whose only job is auditing master commits. MULTI-tab coordination via `runs/coordination/HANDSHAKE_<date>.md`. Cost: extra Opus budget for a watcher tab; gain: real-time gap-window closure. Overkill for solo-dialect; appropriate for high-stakes capital-allocation sprints.

**Binding for r7+**: every worker brief MUST include Option C `pre_commit_self_audit:` block. Option D should be built in the next L0 session as a discrete one-off task. Option A is reserved for L2-tier SHIP_CANDIDATE validation rounds. Option B is reserved for MULTI-DIALECT mode.

7. **Other-instance work happens in parallel.** Across my 5-hour session, INST-fdbdb2bb made multiple parallel commits including a dossier closure that I initially mis-interpreted as applying to my track. **Future instances must check git log for parallel-instance activity at session start AND at every 30-min checkpoint**; assume the master branch is concurrently modified.

### §F.9 The North-Star ROI gap (mechanical accounting)

**Current SOLE SHIPPED**: R12 perp UNSEEN +48.90% over 14 months = **+0.12%/d**. Target band: **+1-5%/d**.

The 8-40x gap is NOT closeable by depth-mining a single (PEPE, MA/EMA) bot. Per §F.2 deploy model, the path is:

- **N wallets × N strats × per-wallet ROI** = portfolio ROI
- If R12 (or comparable) ships at +0.12%/d per wallet, and the operator runs **8-40 such wallets across uncorrelated (TI, ASSET) tuples**, portfolio ROI = +1-5%/d.
- This is the **wallet-multiplication path** to the target, consistent with user's "small players that win" framing.

**Framework consequence**: dossier-closure cadence and new-TI dossier start-up must accelerate. The bottleneck is **dossier throughput**, not depth-of-mining per dossier. Closing PEPE/MA/EMA at 60.9% with R12 + a documented C2 PARTIALLY_PASS may be the wealth-optimal call IF it frees the operator to start (PEPE/RSI), (PEPE/MACD), (PEPE/Bollinger), (ETH/MA/EMA), (SOL/MA/EMA) in parallel. The 80% canonical depth is a quality-of-documentation gate, not a deploy-prerequisite gate. **Recommend amending r5.2 §Gold-Standard Dossier to allow CLOSE-WITH-CAVEAT at 60-79% if the closing instance documents the residual sub-axis gaps as "QUEUED-for-natural-n-rev"**.

### §F.10 Recommendations summary (for next instance / framework rev)

1. **Build `capture_rate_decomposer.py` BEFORE starting any new dossier** (Phase 0 instrumentation gap).
2. **Add cadence-coverage rows to §1A status grid template** (close the timeframe gap surfaced by user).
3. **Codify the rug-pull filter for top-100 wallet alt-data** before any per-address signal use.
4. **Amend r5.2 §Gold-Standard closure to allow 60-79% close-with-caveat** if cross-instance coordination supports new-dossier momentum.
5. **Make auditor-after-every-commit the framework default**, not an opt-in.
6. **Add Pattern S (close-only trail breach) to `memory/fix_logs/INDEX.md`** as cross-cutting bug pattern.
7. **Promote `strengths_weaknesses_expected_failures` block to mandatory ship-template field** (§F.6).
8. **Populate Q1-Q10 DURING every dossier lifecycle** — not retroactively. Q3 and Q6 require new mining; Q7 requires one oracle pass. See §RE-MINING NECESSITY. Do NOT ship a candidate without at least Q1/Q2/Q4/Q5/Q8/Q9/Q10 scored (the 7 retroactive dimensions). Q3/Q6/Q7 as `NOT_YET_MEASURED` with mining plans are acceptable at ship-time if capacity sweep and L1 tooling are unavailable, but these must be queued.
9. **Produce STRATIFIED_TOP3.md at every dossier closure** per §STRATIFIED MINING MANDATE (r8). For existing dossiers, document the gap with queued back-fill.
10. **v8.1 amendments (2026-05-26 ~07:00 SAST user mandate)**: three PRE-REGISTERED additions encoded in §SM8-SM9 and §SM2 cadence-priority update: (a) cadence-aware DD-halt (40% at n<20 trades/window, 25% at n>=20) motivated by R57b1 finding that 1h TRAIN windows at PEPE had 7-10 trades — 25% DD halt fired on 2nd-3rd losing trade before recovery; (b) 3-day max-hold sweet-spot mandate — every strategy must include max-hold guard; `max_hold_bars` field added to canonical YAML; R12 perp ALREADY compliant (avg 1.3d hold); R23a needs hold-time measurement; (c) cadence priority hierarchy — default mining set {15m, 30m, 1h, 4h}; 1d marked DEPRIORITIZED (QUEUE_LAST). Provenance: R57b1 commit 89cb3f2. Candidates refuted under v8 must be RE-RUN as new rounds (not automatically promoted).
11. **v8.2 amendment — SM9.1 CONDITIONAL MAX-HOLD EXTENSION (2026-05-26 ~09:30 SAST, PRE-REGISTERED)**: hybrid resolution to user question on whether to extend beyond 3d if move is ongoing. Baseline 3d cap (SM9) UNCHANGED. Conditional extension to ≤7d permitted ONLY when ALL THREE hold: (1) primary entry signal still active; (2) position winning (close > entry × 1.005); (3) no opposite-direction signal fired. 7d is HARD CAP — no further extension. Asymmetric: losers cut at 3d (never extended); winners may extend to 7d only on continuation signal. Per-cadence empirical anchor (median_hold_of_winning_trades) is DIAGNOSTIC, not a gate override. R12 avg hold 1.3d → prior <3% of trades reach extension gate; rule is a forward-deployment safeguard. Cadence bar counts: 4h=42, 1h=168, 30m=336, 15m=672, 1d=7. Falsifier: if extension produces G2-pass/G6-fail at higher rate than 3d-only rule, tighten winner gate from 0.5% to 1.0% and re-run. Files updated: §SM9.1 (this doc), ST3.1 (static spec), ML3.1 (ML spec), LAYER_DECOMPOSITION_TEMPLATE.md (L6), CROSS_LAYER_HANDOFFS.md, dossier template §12, R12 + R23a YAML configs.
12. **v8.3 amendment — SM10 CANONICAL LIBRARY STACK MANDATE (2026-05-26 ~14:00 SAST)**: root-cause fix for Pattern S/T/U bug class. Every R-round with inline simulator + inline indicator is a new bug surface; v8.2 audit gates are bandaids. Solution: (a) pandas-ta adopted as canonical indicator library (already in requirements.txt); (b) `src/wealth_bot/harness.py` built as canonical backtest module -- CanonicalHarness class makes Pattern S (trail breach), Pattern T (same-bar close fill), Pattern U (inline indicator), MFE-lock, missing repro block, and unlabelled all-4-positive STRUCTURALLY IMPOSSIBLE at API level; (c) R12 POC (`scripts/wealth_bot/r12_canonical_harness_poc.py`) verified harness matches post-fix R12 UNSEEN=+39.65% with delta=0.0000pp; (d) Pattern U + Pattern U2 added to auditor RED-team grep checklist; (e) MIGRATION_BACKLOG of 15 scripts tracked in harness.py; (f) falsifier: if harness limits a legitimate research need, extend the harness API, NEVER bypass it.

13. **v8.4 amendment — SM11 SETUP-CHASE DOCTRINE + Phase 2.5 EXIT MECHANISM EXPLORATION (2026-05-26 ~16:00 SAST, PRE-REGISTERED)**: empirical grounding: PEPE mover-day HIT rate 84.1% (entry works); capture-of-UP-move 26.1% (exit leaves 74% on the table); L2 mean capture R12 = -0.325 (fires before peak). SM11 formalizes ENTRY and EXIT as two ORTHOGONAL optimization problems with independent KPIs (TOPQ_DAYS HIT rate >= 70% for setup detection; TOPQ capture >= 50% for move riding). Phase 2.5 inserts between Phase 2 and Phase 3 as a MANDATORY cross-product sweep: rows = entry baselines, cols = 12 canonical exit mechanisms (R1-R3, A1-A3, H1-H3, M1-M3), cells = compound + L2 capture + mover-day capture + jackknife + stability composite. Output: `docs/dossiers/stratified/<TI>_<ASSET>__EXIT_BAKEOFF.md`. Phase 1 binding note: Phase 1 baseline exit = opposite-signal or fixed-N by construction; this is intentional — Phase 1 validates entry, not exit. Falsifier: if all 12 exit mechanisms are within 5pp L2 capture of baseline, signal has a structural exit-ceiling (fires too late); new TI family required. PEPE (MA/EMA) retroactive augmentation: 24 harness runs (R12 + R23a entry × 12 exits), ~2h wall-clock. ATR k<=1.0 remains BANNED per FM-ATR-TIGHT-REGIME. Files updated: §SM11, §F.12, §Phase 2.5 (all in parent doc); ST13 + ST3 (static spec); ML3/ML13 (ML spec); dossier template §13; LAYER_DECOMPOSITION_TEMPLATE.md L2 first-class note; CROSS_LAYER_HANDOFFS.md v8.4 entry.

These are framework-coherence upgrades. Per the framework's own anti-pattern §("Skipping null reports"), the gaps documented here are themselves valuable artifacts — future instances inherit them.

---

### §F.11 Regime-Routed Deploy Protocol (r8) `[ALL]`

> **Provenance**: 2026-05-26 user mandate (r8). Enables regime-aware capital routing from the stratified top-3 tables produced per §STRATIFIED MINING MANDATE.

#### What changes vs single-config deploy

Prior framework shipped ONE config per (TI, ASSET) dossier (e.g., R12 perp WMA 10/30 + whale + Kelly 1.0 at 4h). This config ran regardless of market regime. The stratified top-3 table (§SM4) unlocks a different deploy model: when a regime detector identifies the current state, capital routes to the top candidate FOR THAT CELL.

Example routing logic (INFERRED — not yet empirically validated):
- BTC enters `btc_bear` regime: portfolio routes to top-3 candidates from "4h × btc_bear" cells (which may favor lower-exposure or mean-reversion bots over trend-following).
- PEPE enters `high_vol` regime: portfolio routes to top-3 from "1h × high_vol" cells (faster strategies that trade more frequently and capture intraday swings rather than multi-day trends).
- PEPE enters `chop` regime: trend-following bots (top of "4h × trending_up" cells) are DE-WEIGHTED; chop-specific configs take capital allocation.

This is fundamentally different from Phase 4 within-TI regime composition (which routes between configs within ONE (TI, ASSET) dossier on one wallet). The regime-routed deploy protocol operates ACROSS dossiers at the portfolio-orchestration layer.

#### Deploy implications

1. **Per-wallet isolation maintained**: regime routing does NOT break the one-strat-per-wallet rule (§F.2). Each wallet still runs ONE config. Routing means WHICH wallet gets capital allocated at any moment.

2. **Regime detector is shared infrastructure**: the regime detector (SMA-50 slope, vol percentile, BTC 30d return) is computed once per bar and published to all wallets. Each wallet subscribes to its relevant cell tag and accepts allocation when its cell is "active."

3. **Fallback config**: each (TI, ASSET) dossier must designate a FALLBACK config (typically the all-regime winner from Phase 1 — e.g., R12 perp) for periods when the regime detector is ambiguous or the specific cell has no SHIPPED candidates. Fallback runs at reduced allocation (half-Kelly default).

4. **Cross-dossier routing** (future capability, not yet implemented):
   - When multiple (TI, ASSET) dossiers each have candidates in the same regime cell, the meta-leaderboard allocates across them by Q_total ranking.
   - Example: (PEPE, MA/EMA) "4h × trending_up" cell vs (PEPE, RSI) "4h × trending_up" cell — the higher Q_total candidate gets primary allocation.
   - This requires the meta-leaderboard to be regime-cell-indexed (queued for meta-leaderboard v2).

5. **Regime detection latency**: all regime proxies use PAST-ONLY data (bars[0..t]) with no look-ahead. Regime classification at bar t is available at the close of bar t. Entry on next bar's open is consistent with H4-realistic fill model.

#### Regime router script (queued)

```
scripts/wealth_bot/_regime_router_v2.py
```

Extends `_regime_router.py` (§P4.5) from within-TI Phase 4 scope to cross-dossier portfolio scope. Inputs: list of STRATIFIED_TOP3.md files + live chimera bar. Output: per-wallet allocation weight for the current regime cell. Queued for implementation when 2+ dossiers have STRATIFIED_TOP3.md populated.

#### Pre-conditions for regime-routed deploy

Before routing capital via regime cells, ALL of the following must be true:
- `[ ]` At least 2 distinct (TI, ASSET) dossiers have STRATIFIED_TOP3.md with populated cells in the same regime.
- `[ ]` Each active cell candidate has passed G1-G6 gates on its regime-filtered UNSEEN slice (not just full-UNSEEN G1-G6 — regime-filtered slice must independently pass G1 all-4-windows-positive).
- `[ ]` Fallback config designated for each (TI, ASSET) and verified passing G1-G6 on full UNSEEN.
- `[ ]` `_regime_router_v2.py` built + smoke-tested on historical regime transitions.
- `[ ]` Paper-trade simulation of regime-switch event (at least 1 historical BTC-regime-change and 1 vol-regime-change tested).

Until pre-conditions met: deploy the fallback config (full-UNSEEN G1-G6 ship candidate) without regime routing.

---

### §F.12 Phase 2.5 Exit Mechanism Exploration — Operational Addendum `[ALL]` (v8.4)

> **Provenance**: 2026-05-26 ~16:00 SAST user mandate + empirical anchor from this session: PEPE UP-day HIT rate 84.1%, capture-of-UP-move 26.1%. Entry detection works; exit is the bottleneck. Canonical protocol at §Phase 2.5 above.

#### When Phase 2.5 fires

Phase 2.5 fires AUTOMATICALLY after Phase 2 completes (ship OR null), before Phase 3 expansion. There is no opt-out. Rationale: without a validated exit, Phase 3 expansion axes (cadence, filter, chart-type) all produce compound figures that conflate entry quality with exit quality — making it impossible to isolate which axis drove improvement.

**Exception**: Phase 2.5 MAY be deferred (not skipped) when:
- UNSEEN n < 12 (insufficient power to measure L2 capture rate reliably); in this case, flag as "EXIT_BAKEOFF DEFERRED — n < 12" in the dossier and proceed to Phase 3. Revisit at Phase 3 completion when n is larger.
- The prior Phase 1 already ran a partial exit sweep (document which mechanisms were covered; run only the remainder).

#### PEPE (MA/EMA) retroactive augmentation

The (PEPE, MA/EMA) gold-standard dossier can be retroactively augmented with Phase 2.5 without reopening Phase 1 or Phase 2. Steps:

1. **Entry baselines** (from existing Phase 1 / Phase 2 outcomes):
   - R12: WMA(10,30) + whale_net + Kelly 1.0 at 4h — SHIPPED, n=18 UNSEEN
   - R23a: EMA30_dist>1% + whale at 4h — INCONCLUSIVE, hold-time unmeasured
   These two are the Phase 2.5 entry candidates.

2. **Run cross-product** on existing UNSEEN window (same chimera, same split, no new data required). All 12 exit mechanisms against both entry baselines = 24 harness runs. At ~5 min each: ~2h wall-clock.

3. **Capture-rate gap to close**: current R12 L2 mean = -0.325 (fires before peak; exit is net negative capture). Target: identify exit mechanism achieving L2 capture rate >= 0.20 (a 0.525pp improvement from baseline). Even A3 MFE-lock-in at moderate lock % should outperform R1 by this margin empirically.

4. **Output**: `docs/dossiers/stratified/EMAMA_PEPE__EXIT_BAKEOFF.md` + audit JSONs.

5. **Dossier update**: add §13 EXIT_BAKEOFF compliance checklist (see template update below) and mark Dim E in §1A status grid as [PARTIAL → IN_PROGRESS_2.5] while Phase 2.5 runs, then [SHIPPED] for winning combos and [REFUTED] for non-winners after.

#### Prior (PEPE, MA/EMA) exit testing (do not re-test)

Per the existing failure catalog (ST7):
- FM-ATR-TIGHT-REGIME: ATR <= 1.0x MATURE FAILURE — confirmed 3 data points. Do NOT propose ATR k<= 1.0 variants. Use A1 with k >= 1.5 only.
- Chandelier REFUTED (R45): re-test A2 only with updated CanonicalHarness (Pattern S compliant) — prior R45 result may have been inflated by Pattern S breach; harness-compliant re-test is valid.
- MFE trailing (R7 REFUTED, R45/R46 post-Pattern-S): re-test A3 MFE-lock-in — note this is conceptually DIFFERENT from the R7 "exit when MFE gives back X%". A3 locks in a floor (ratchet up), not a trailing exit down.

#### Integration with §SM10 CanonicalHarness

ALL Phase 2.5 exit mechanisms MUST be implemented via CanonicalHarness `exit_policy` field. If a new exit mechanism is not yet in the harness API (e.g., M1 ROC-deceleration), EXTEND the harness first (§SM10.4 falsifier rule), then run Phase 2.5. No standalone inline simulators permitted.
