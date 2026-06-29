# V5 DEPLOY RUNBOOK — paper-trade-only (post red-team, 2026-06-29)

**Engine:** V5 = breadth-hysteresis(0.5/0.4) gate **AND** walk-forward GBM P(good-week|market-state)>0.55 (quarterly
retrained) **AND** conv>0 → momentum14 top-8 inverse-vol. Long-only spot, lev=1, u100, 1d, next-open fills, prod cost 0.0028 RT.
**Frozen config:** hash `aa5f742288022b06`, seed=0 (GBM n_est=120 depth=3 lr=0.05 subsample=0.7 random_state=0). sklearn 1.7.0.

## VERDICT: `paper-trade-only` — NOT deployable with real capital yet

Adversarially verified (workflow `wbw77ivx2`, 4 independent red-team lenses + trader synthesis). The look-ahead audit is
**clean** (all surfaces causal). But the engine **fails every pre-ship statistical gate**:

| Gate | Result | Verdict |
|---|---|---|
| Look-ahead / leak | all surfaces causal | ✅ clean |
| Committed significance (Newey-West L3, HAC) | **p = 0.087** | ❌ not significant *even before* search penalty |
| Deflated-Sharpe (honest trial count ~hundreds) | DSR 0.13 @n=100, 0.07 @n=200 | ❌ ≪ 0.95 |
| Block-bootstrap p05 (OOS-ALL) | **−15%** | ❌ not robust |
| Concentration | **top-2 weeks = 77%** of compound (top-5 removal → −19%) | ❌ "one-or-two-weeks-from-zero" |
| Component decomposition | breadth-alone **+57%** / Pwf-timing-alone **−7%** / full +170% | ⚠️ the breadth gate is the real edge; timing overlay alone is negative |
| Bear (OOS-2025) | +3% (7/10 subsample fits positive, min −7%, p05 −37%) | ⚠️ fragile |

**Honest read:** the *breadth gate* (de-risking) is the robust, real piece (+57% alone — consistent with the whole
project's "breadth gate confirmed"). The *learned-timing overlay* only adds value by coincidentally landing 2-3 mega-weeks
inside its deploy windows; that magnitude is not statistically settled. The shuffle-null (beats at empirical p<0.001)
confirms the placement isn't random, but with 2 weeks carrying 77% it's a low-n coincidence test, not robust structure.

## What to do now

- **Paper-forward the frozen spec.** Each Monday 00:00 UTC, run the frozen config → emit the live book (or CASH) and log
  the realized next-open 7d return. **Zero real capital.** **Do not re-tune** any parameter — any change voids the
  pre-registration and restarts the count (new config_hash).
- **Current live state (as of 2026-05-28): CASH** — P(good-week)=0.344, unanimous across all subsample fits. Do not deploy.
- **Quarterly:** retrain the GBM on all data up to the quarter start (part of the frozen spec). Next: 2026-Q3 boundary.

## Pre-registered live-forward gate (ALL must hold before ANY real capital)

1. ≥ 20 **deployed** weeks accumulated live (at ~14% deploy rate ≈ 12-18 months — set a calendar reminder).
2. Deployed-week median return > 0 over that sample.
3. Compound over deployed weeks beats 100% cash.
4. Compound beats BTC buy-hold measured **only over the same deployed weeks**.
5. Observed deploy rate in [10%, 35%] (confirms the gate isn't degenerate).

## If a user insists on a live proof-of-concept before the gate passes

- Hard cap **2% of the crypto portfolio per deployed week** (inverse-vol across the frozen book). Rationale: magnitude is
  unplannable (±30-40% on subsample alone; p05 negative all windows; honest DSR < 0.13) — at 2% gross, a full deployed-week
  wipeout is a 2% portfolio loss (survivable). Do **not** scale up until the gate passes; then re-derive sizing from the
  *realized live* Sharpe/DD, not the backtest magnitude.

## Kill-switch (halt all deployment, return to cash, do NOT re-tune) on ANY of:

1. Deployed-week median ≤ 0 over ≥ 20 deployed weeks.
2. Compound < cash on deployed capital.
3. Live paper max drawdown > −35%.
4. Deploy rate < 5% over a 52-week period (gate degenerate / always-cash).
5. After a kill: treat it as a genuine refutation; any resumption needs a fresh pre-registration (new config_hash + new count).

## Honest risk profile (state this to anyone who asks)

V5 is **bear-fragile and lumpy**. Bear case (OOS-2025): +3% point estimate, p05 ≈ −37% on the weekly compound — the honest
downside is roughly a one-third drawdown, not a flat year. The entire +170% OOS-ALL compound is carried by ~2-3 mega-weeks;
miss them (outage, manual override, a kill triggered by a prior drawdown) and you underperform cash. In a sustained bear
(no breadth-gate passage) the engine is simply cash — which is correct, not a loss. **The +170% backtest is not a plan.**

## Artifacts
`strat_sota_audit_corrected.{py,json}` (the honest stats), `strat_sota_freeze.{py,json}` (frozen spec + live book),
`strat_sota_validate.{py,json}`, `strat_sota_v6.{py,json}` (levers), `strat_sota_pullback.{py,json}` (complement NULL),
`strat_sota_seed_robust.{py,json}` (subsample stability). Red-team transcript: workflow `wbw77ivx2`.
