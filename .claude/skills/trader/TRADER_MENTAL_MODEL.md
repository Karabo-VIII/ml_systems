# Trader's Mental Model + Failure Catalog

> Numbers don't trade — people (or, here, the trader skill) do. This file captures the *judgment* layer: how to think when the data is ambiguous, what behavioral failures to guard against, and which patterns of trader failure have been observed in this project.

## Core mental models

### 1. The asymmetry of bets

Every trade has an upside cap and a downside cap. The trader's job is to seek positive expectancy under bounded downside. NOT to maximize expectancy.

The cost of a perfectly-sized winning trade (we left return on the table by being defensive): bounded, ~25% of potential.
The cost of a perfectly-sized losing trade in fat-tail regime (we sized for Kelly's σ² but actual realized was 3σ): unbounded, can be 100%.

Implication: when in doubt about distribution, size for the worse case. The expected-utility math is asymmetric.

### 2. Base rate discipline

For every new sleeve, ask: what fraction of sleeves with these properties have historically worked?

Base rates in this project:
- Sleeves promoted through INCUBATION -> PAPER on backtest alone: ~30% survive to LIVE_SMALL.
- Sleeves in PAPER that diverge from backtest expectation: ~70% never recover.
- Memecoin sleeves with concentrated top-3 trade attribution: ~20% verified by mechanism falsifier (per Pattern Q).
- ML-trained sleeves with single-seed claims: ~10% survive multi-seed verification (LSTM/DQN refuted).

Base rates frame the question. A sleeve looking great in backtest is the prior; the base rate is the posterior dampener.

### 3. The value-needle test (per AUTONOMOUS_RUNNER.md §2)

Every turn in autonomous mode asks: did this move the primary needle?

For trader-skill turns: every action either (a) reduces capital-at-risk, (b) increases verified edge, or (c) compresses time-to-deploy. If it does none of those, it's busy-work — stop.

### 4. Regret minimization (Howard Marks)

Two regrets to balance:
- Regret of acting and being wrong (e.g., deployed a sleeve that lost).
- Regret of not acting and missing the opportunity (e.g., didn't deploy a sleeve that worked).

In this project, regret of (1) > regret of (2). Capital lost is unrecoverable; an opportunity missed is recoverable next sleeve. This biases toward delay-and-verify over speed-and-ship.

Exception: when the opportunity is decaying (e.g., regime shift, capacity ceiling). Then regret of (2) catches up.

### 5. Bet sizing as the dominant variable

Edgington-Kelly observation: in a portfolio of many edges, sizing dominates selection. A mediocre signal sized correctly outperforms a great signal sized poorly.

For this project: spend more time on `RISK_PLAYBOOK.md` thresholds and `SIZING_THEORY.md` decision tree than on hunting for new signals.

### 6. The Lindy effect for sleeves

A sleeve's expected remaining useful life is proportional to its current age. A sleeve that worked for 6 months has higher expected remaining life than one that worked for 2 weeks.

Implication: in capital allocation across sleeves, weight by *time-tested*, not *recent-best-performer*.

### 7. Survive first, profit second

The first job is to be around tomorrow. Drawdowns compound — a -20% requires +25% to recover, -50% requires +100%. Avoiding ruin is the precondition for any return target.

This is why DD thresholds in `RISK_PLAYBOOK.md` are the binding constraint, and compound % is the *output*, not the *target*.

---

## Behavioral guardrails

These are failure patterns that trip even experienced traders. Embedded as guardrails to check against.

### Revenge trading

Pattern: sleeve hits -10% DD. Trader doubles sizing to "make it back."
Guardrail: any sizing INCREASE after a DD requires a fresh deploy claim. Asymmetric sizing rule from `RISK_PLAYBOOK.md` makes this hard to fall into.

### FOMO

Pattern: an asset is pumping; trader deploys a sleeve to chase it.
Guardrail: all new sleeves go through INCUBATION -> PAPER -> LIVE_SMALL. No fast-track. Catalyst-driven plays are doable in PAPER only.

### Anchoring on entry price

Pattern: position is -5% from entry; trader holds because "it'll come back."
Guardrail: max-hold rules in sleeve YAML. Position exits at time T regardless of P&L vs entry.

### Sunk cost fallacy

Pattern: months of effort on a sleeve that's underperforming; trader keeps trying variants.
Guardrail: per WEALTH_BOT_DEVELOPMENT_FRAMEWORK r5, 3 consecutive Phase 2 NULL rounds = SATURATION → 1-quarter ban + mandatory scope expansion. Hard rule.

### Confirmation bias on mechanism

Pattern: sleeve works, trader assumes the stated mechanism. Doesn't verify trade-level which trades drove the return.
Guardrail: `mechanism_falsifier_check` in claim contract v1.2. P4_route_basis incident proves this is non-optional.

### Hindsight bias on regime

Pattern: regime shift is obvious in retrospect. Trader assumes future regime shifts will be equally detectable.
Guardrail: regime detection is post-hoc per gap G1; treat regime calls with high uncertainty.

### Overconfidence after a win streak

Pattern: 5 consecutive winning trades; trader sizes up.
Guardrail: sizing UP rate from `RISK_PLAYBOOK.md` is +10-25% per stage transition (slow). Cannot up-size on win streak alone.

### Underconfidence after a loss streak

Pattern: 5 consecutive losing trades; trader halves AGAIN beyond what playbook dictates.
Guardrail: `RISK_PLAYBOOK.md` has specific thresholds. Don't go below playbook minimum without explicit reason.

### Selection bias in backtest review

Pattern: trader runs 20 backtests, picks the best one, deploys.
Guardrail: DSR > 0.95 AND CSCV PBO < 0.50 required when sweep > 20. Claim contract enforces.

### Look-ahead via "obvious" features

Pattern: feature includes a same-bar value that wasn't available at decision time.
Guardrail: PRE_DEPLOY_CHECKLIST item 3 grep + look-ahead audit.

---

## Failure catalog (trader-specific patterns, observed in this project)

### Pattern T1: Stride-staleness inflation (Pattern N)

Pre-2026-04-14: all backtests used WM predictions stale up to 95 bars. Headline Sharpe numbers were inflated. Fix: stride=1 in gen_preds.
Detection: ANY pre-2026-04-14 audit JSON is suspect. PRE_DEPLOY_CHECKLIST item 4.

### Pattern T2: MtM double-count (Pattern simulator-2026-04-22)

Pre-2026-04-22: simulators added cumulative ret_from_entry to MtM stream, double-counting position contribution. Pre-fix +501% became truth +94%.
Detection: PRE_DEPLOY_CHECKLIST item 2 reconciliation gate.

### Pattern T3: Mechanism claim falsified by data (Pattern Q)

2026-05-25: P4_route_basis_pos_only claimed "filter strips top-tail trades." Empirically, filter KEPT top-3 and DROPPED diversifying ones. Sleeve still shipped with wrong story.
Detection: mechanism_falsifier_check in claim contract; PRE_DEPLOY_CHECKLIST item 7.

### Pattern T4: Single-seed init-luck (Pattern multi-seed)

2026-05-24: LSTM +44.6%, DQN +40.9% — refuted by N=10 seed audit. Median was near zero.
Detection: PRE_DEPLOY_CHECKLIST item 8.

### Pattern T5: Look-ahead via future-return K-selection

Multiple sleeves selected top-K trades using future-return data, baking look-ahead into the claim.
Detection: PRE_DEPLOY_CHECKLIST item 3; K-selection must use signal columns only, not target columns.

### Pattern T6: Gap-down fallback miss / forward-close leak

2026-05-25 R51 E51_1 + R54 A54_2 — close-only execution missed gap-down events; PSEUDO-VB forward-close leak. Three gap-window incidents in INST-F96BE75A 2026-05-25/26.
Detection: PRE_DEPLOY_CHECKLIST item 3 (look-ahead audit covers close-bar timing); UNIVERSAL PRE-DELIVERY SELF-AUDIT (CLAUDE.md).

### Pattern T7: H6 catalyst plays without downside protection

H23 trail-stop on H6 entries refuted (-35.5%). Mechanism: catalyst plays move so fast that trail-stops trigger AFTER the catalyst-decay drawdown is locked in.
Detection: do not deploy trail-stops on H6-class sleeves. RISK_PLAYBOOK.md explicit rule.

### Pattern T8: Bear-regime gating block (correct behavior)

W25/W26 3-MA survivors REFUTED in bear regime (commit 665183e 2026-05-27). Regime gating blocked entries — the SYSTEM worked.
This is a NON-failure pattern. Recognize when the gating system is doing its job.

### Pattern T9: Paper-trade decay refutation (correct behavior)

H18 paper-trade 4 phases REFUTED (commit fd0a870 4ba027b 2026-05-27). Paper-trade caught what backtest missed.
This is what paper-trade is FOR. Don't shortcut it.

---

## Decision quality vs outcome quality

A good decision can have a bad outcome (good sleeve, drew down). A bad decision can have a good outcome (sleeve worked despite look-ahead bug).

Evaluate decisions by their **process**, not their outcome. The pre-deploy checklist, the lifecycle gates, the claim contract — these are decision-quality checks.

In retrospect, when reviewing wins and losses:
- Won + good process: stay the course.
- Won + bad process: revise process before more capital is at risk. Got lucky.
- Lost + good process: stay the course. Expected drawdown.
- Lost + bad process: revise process AND apologize for the loss. Avoidable.

---

## When to consult other skills

This trader skill is the trader. But trading is a multi-domain discipline:

- **audit** when a sleeve underperforms and you need a RED-team review of what the trader missed.
- **decide** when the strategic question is bigger than tactics ("should we abandon WM-based signals and go pure price-action?").
- **decide** when two equally-defensible positions exist (e.g., scale H18 vs hold; pursue PEPE vs diversify).
- **/research** when a literature-grounded answer would help (e.g., "what's the SOTA on online HRP recalibration?").
- **/validator** when a number needs a second look (e.g., "is this Sharpe survivability-bias-free?").
- **/pipeline** when the issue is upstream data quality, not strategy.

Trader is a lens, not a silo. Borrow other lenses freely when the question demands.

## CDAP wiring

No CDAP rules — mental models are advisory by nature. Behavioral guardrails are operationalized in other files (RISK_PLAYBOOK, SIZING_THEORY, LIFECYCLE).

## Cross-references

- PRE_DEPLOY_CHECKLIST.md — the canonical decision-quality checklist.
- RISK_PLAYBOOK.md — operational guardrails against revenge / FOMO / anchoring.
- LIFECYCLE.md — base-rate discipline via stage gates.
