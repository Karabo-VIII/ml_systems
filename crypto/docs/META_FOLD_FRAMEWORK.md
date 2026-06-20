# The META-FOLD framework (canonical, 2026-06-19)

A self-folding, adversarially-refereed research loop for crypto strategy discovery (or any quantitative question).
Validated on the 7d-slice campaign (3-cycle fold + the engine tournament) where it caught a real SMA-init look-ahead
bug, an iid-shuffle variance illusion (p~0.02 -> honest block-bootstrap p=0.169), and converged to an honest verdict
instead of shipping an overfit headline. **Use this for any multi-path search.**

## The loop
```
LEDGER (cumulative conclusions, survives context resets)
   |
   v
CYCLE = a Workflow:
   PHASE 1  N parallel EXPLORATION lanes  -- each a distinct hypothesis/engine, RWYB (writes+runs real code, real numbers)
   PHASE 2  1 adversarial REFEREE lane    -- INDEPENDENTLY re-derives the load-bearing numbers (trusts no lane),
                                             HUNTS look-ahead/leak/overfit, ranks vs the null + the passive baseline,
                                             and DESIGNS the next cycle OR declares CONVERGENCE
   |
   v
FOLD: append the cycle's conclusions to the ledger; the next cycle's design is a function of the CUMULATIVE ledger
      (deepen winners, KILL losers, test the lessons, pre-register the decisive test). Each cycle REFINES, never repeats.
   |
   v   (driven by workflow-completion notifications + a fallback ScheduleWakeup; loops until ...)
CONVERGENCE  -- 2 cycles with no new edge, OR a pre-registered decisive test fails, OR a fully-validated play remains.
               Then post the FINAL cumulative synthesis and STOP (no manufactured cycles).
```

## Canonical components
- **The LAB** -- a lightweight, causal, RWYB backtest module exposing aligned data + derived features + an `evaluate()`
  with ONE consistent metric set, so every lane/engine is measured identically. Reference impl: `src/strat/mover_lab.py`
  (load -> ind dict of causal DataFrames + indicators; evaluate(W) -> per-year compound, maxDD, checkpoint green-rate,
  exposure, turnover; lags positions, taker cost). Extend the lab per cycle (e.g. add chimera features, bar-types, TFs).
- **The CYCLE workflow** -- `parallel(exploration lanes)` then an `expert-quant`/`expert-auditor` referee synthesis. Give
  every lane the lab API + the causal rule + the cumulative ledger state + the win bar. The referee re-derives, not trusts.
- **The LEDGER** -- `runs/strat/_meta_fold_ledger.md`: dated cycle entries (focus | winners w/ numbers | losers | lessons |
  what the next cycle tests). The carried state.
- **The DRIVER** -- workflow-completion is the fast path; a ScheduleWakeup(~1200s) carrying the protocol is the fallback.

## The discipline (NON-NEGOTIABLE)
- **RWYB** -- every number from an actual run; the referee re-derives the load-bearing ones independently.
- **HUNT look-ahead** -- the #1 way a backtest/ML model fakes an edge. Strict walk-forward; no future feature; no
  train/test label-window overlap; no global scaler/threshold fit on full data.
- **PRE-REGISTER the decisive test** -- fix the single config BEFORE the make-or-break test (sweeping is a DoF that
  inflates significance). Correct for multiple comparisons (Holm) across any sweep.
- **The right NULL** -- shuffle for a quick check, but a serial-correlation-aware MOVING-BLOCK BOOTSTRAP for the verdict
  (iid shuffle understates variance ~6x on autocorrelated returns). Always compare to the passive baseline (buy-hold).
  **BROKEN-NULL WARNING (2026-06-20, sub-daily cycle w3bbolocj):** the SAME-EXPOSURE RANDOM-K SHUFFLE is a *broken* null
  for any strategy that RE-SELECTS positions per bar -- the random control eats a per-bar reshuffle-variance penalty the
  real (smoother) book doesn't, so `real - control` manufactures fake "alpha" (quantified: 4h-chop control -10.99 bp/bar
  vs real -2.75 bp/bar, both negative gross). It scales with bar-count, so it BLOOMS at sub-daily. The HONEST null is
  **HOLD-TO-MATURITY** (pick top-K, hold the slice, ~1 RT) + four churn cross-checks that any real selection edge must
  pass: (1) hold-to-maturity p05>0; (2) survives a REGIME-LABEL shuffle test = FAIL (real edge dies when labels destroyed);
  (3) REVERSE-SCORE (worst-K) goes negative (direction-sensitive, not concentration); (4) alpha/day is CALENDAR-INVARIANT,
  not growing with bar-count. Cheapest falsifier: re-run any cell with the hold-to-maturity null (`meta_tf_invariance_audit.py` S3).
  For move-CATCH questions use the CAPTURE-RATE null (realized/available move vs random-ENTRY at matched frequency) -- also churn-immune.
- **No sugarcoat, no manufactured cycles** -- converge when the question is honestly answered; report the wall if it's a wall.

## The search space (the "whole project" -- expand the lab/lanes across these)
- **Frame/horizon**: the objective window (e.g. 7d-slice / 2w-lookback), checkpoint cadence.
- **Universe**: u10 -> u50 -> wider (PIT, survivorship-clean).
- **Timeframes + chart/bar types**: 1d/4h/2h/1h/30m/15m + dollar / DIB / range bars (the chimera supports these).
- **Features/signals**: price-TIs (8 MA + 18 non-MA) + the CHIMERA families (funding, basis/premium, ETF flow, on-chain,
  order-flow, LOB, transfer-entropy, stablecoin, DVOL -- the largely-untapped exogenous-ish signal) + cross-asset lead-lag.
- **Engine types**: hand-rule -> ML meta-labeler / ensemble -> adaptive regime-ROUTER -> expert-designed.
- **Experts**: the `expert-*` agents (discover / quant / trader / oracle / pipeline / researcher) as exploration lanes.

## Apply to any question
1. Define the WIN bar + the passive baseline + the null. 2. Build/extend the lab. 3. Cycle 1 = a broad tournament of
distinct lanes. 4. Referee re-derives + designs cycle 2 from the ledger. 5. Fold until convergence. 6. Final synthesis.

Provenance: ad-hoc fold of the 7d-slice campaign (2026-06-19), formalized at the user's request to be canonical across
the project. Ledger: `runs/strat/_meta_fold_ledger.md`. Lab: `src/strat/mover_lab.py`.
