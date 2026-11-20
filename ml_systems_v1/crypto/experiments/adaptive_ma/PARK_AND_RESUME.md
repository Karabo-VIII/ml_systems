# Adaptive-MA — PARKED 2026-06-06 ~00:03 (resumable)

Parked by the orchestrator to (a) apply loop optimizations and (b) run the Market Wizards proof task. All state
preserved; resume any time.

## Where it stands (findings, in order)
1. **1d MA-cross adaptive system — REFUTED** (0/69 beat the random-entry firewall on held-out; UNSEEN exp negative).
2. **4h ER-gated fixed-MA — REFUTED by BOTH rigs, two-sided-sound** (0/77 beat the regime-matched null;
   positive_control PASSED so it's a real refutation; in all 6 configs random-within-regime out-earns the MA entry).
3. **Beta-residualized — REFUTED** (0/77; residual per-trade exp ~0.13% ≈ 0). Beta wasn't hiding MA-timing alpha.
4. **Conclusion (META + 2 researchers converge):** the **MA-cross TRIGGER family is dead** (1d+4h, raw+gated+resid).
   The MA/ER survives only as a **regime FILTER** (long-in-ER>0.4 is positive). The failure is the TRIGGER.

## NEXT STEP (pre-registered, ready to run) — change the TRIGGER, keep the apparatus
Per `RESEARCHER_REPORT_2.md`, EV-ranked non-MA triggers to test vs the regime-matched null (≤3 DOF each):
1. **Liquidation-cascade reversal** (HIGHEST) — `liq_short_z30.shift(1)`/`liq_short_panic.shift(1)` gate, LONG on
   reclaim. Kill-condition = beta → test on beta-residualized returns (reuse `expert/beta_residualize_4h.py`) + firewall.
2. **Momentum-acceleration** — `norm_momentum_accel` (pre-computed, lookahead_safe, no window to overfit).
3. **Isolated breakout** — `close>rolling_max(high,N)`+volume, WITHOUT the dead MA co-trigger; pre-register N.
Apparatus: `src/strat/setup_harness.py` + regime-matched firewall + positive_control (proven two-sided sound).

## How to RESUME
- Loops are durable: `python scripts/autonomy/run_metaop.py resume --thread {ama2-expert|ama2-plain|ama-meta} --budget N`
  (continues the saved frontier). OR relaunch fresh on the liquidation-cascade objective (cleaner, reuses experiment dir).
- Guidance is in the learnings lanes: `runs/autonomy/learnings/{expert,plain,meta}.jsonl` ([META]/[OVERSEER] entries).
- Full trail: `OVERSEER_LOG.md`, `META_LOOP_LOG.md`, `RESEARCHER_REPORT_1.md`, `RESEARCHER_REPORT_2.md`.

## Apparatus state (reusable, verified)
Built + sound at 4h: causal features (ER/RV/percentile), `setup_harness` ATR-trail/TP/SL exit, regime-matched
random-entry firewall, positive_control power-check (HAS_POWER=True at 4h), beta-residualize, block-bootstrap.
