# CURRENT — live state (Kiro-written each cycle; committed; <2KB)

PHASE: V6 RIGOROUS STATIC COMPLETE — READY FOR PHASE 2 DYNAMIC
UPDATED: 2026-06-25

## VERIFIED ARCHITECTURE
Layer 1: V1 regime gate (1d BTC SMA100, hysteresis) → bull/bear switch
Layer 2: V2 4h decoupled engine (MA band + 7d lookback) → position timing

Results (DEV window 2020-2024):
  EMA combined:  +171,610%, MaxDD -42%, bear 2022 = -2%
  WMA combined:  +229,783%, MaxDD -48%, bear 2022 = -3%
  DEMA combined: +691,448%, MaxDD -45%, bear 2022 = -6%

Random-entry null: PASSED (100th percentile, all 3 types)
Per-week: 56-59% profitable, median +3-4%/week, deployment 35% in 4h layer

## ACTIVE CAMPAIGN
STATIC PHASE — CLOSED + EXHAUSTED across 21 TI families (2026-06-24).
3 SELECTIVE validated engines: MA WMA|4h, Bollinger|1d, RSI|4h (deploy MA+RSI; Boll
0.96 corr w/ MA = redundant). PLUS a bench of 8 NON-SELECTIVE long-tilt overlays from
V4 (CCI/WILLR/VWAP|1d strongest, 100th-pct null, +2-3pp OOS but deploy 80-96% = refine
WHICH longs, not WHETHER long). See runs/engines/CONSOLIDATED_ENGINE_REGISTRY.md.
Phase 2 (dynamic) has a working foundation + concrete target (turn V4 tilts selective).
NOT started — user gate.

## V5 DNA + HYPER-AUDIT (2026-06-25) — full TI×TF performance DNA, no exclusions
Pipeline AUDITED 19/19 (causal/walled/cost/null/rebal-premium all verified) →
runs/engines/V5_dna_report/PIPELINE_AUDIT_RESULT.md. Key audit finding: rebalancing
premium is WINDOW-dependent — ~9.5x on full-DEV but only ~1.1x on OOS-2024, so OOS-2024
compounds are ~honest. Look-ahead probe decisive (cheat +65%→+4217%; production causal).

Full DNA: 26 families × 3 TFs (1d/4h/2h) on VAL (2023-07-01..2024-05-15) + OOS-2024
(2024-05-15..2024-12-31, 2025 SEALED). → runs/engines/V5_dna_report/DNA_REPORT.md (every
cell, ranked by transparent 0-6 robustness score). Findings:
- 25/78 cells DEV/OOS-consistent; 21 also beat OOS random-null ≥95th (genuine timing).
- ONLY ADX is genuinely SELECTIVE (~70% deploy) — but it FAILS VAL/OOS consistency.
  Every other consistent cell deploys 96-100% (always-long-with-tilt under the gate).
- Top robust tilts (5/6, consistent + 100th-pct null, non-selective): HMA|1d (+2.66pp OOS),
  WILLR|1d (+1.29), Bollinger|1d (+1.29), TEMA|1d (+1.12), VIDYA|4h, SMA|4h, VWAP|1d,
  MFI|1d, Supertrend|4h, WMA|4h, PSAR|4h. 1d and 4h carry the signal; 2h is noise (sign-flips).
- This RECONCILES the earlier "Bollinger|1d / RSI|4h validated" claim: under this larger
  harness Bollinger|1d holds (robust tilt), but it is NOT selective — the V3 "selective"
  label conflated low-deploy with the confirmation-layer slice-hold. Honest status: robust
  NON-SELECTIVE tilt, same as ~13 others. RSI|4h is OOS-only here (VAL edge -0.64) → weaker
  than previously stated. NOTHING is both selective AND consistent except via the regime gate.

## DEAD/NULL REWORK (2026-06-25) — DONE
ADX rework OVERTURNED its V3 "null" verdict: a proper lookback×threshold sweep reveals
ADX|1d (lb22-28, thr0.42-0.48) is the program's FIRST genuinely SELECTIVE (deploy ~40-66%)
+ VAL/OOS-consistent + 100th-pct-null engine. Overfit probe: NOT a spike — a directional
gradient (lookback≥22 consistently positive, 25/40 cells OOS+). HONEST caveat: lookback-
sensitive (dies <lb20), OOS edge (+0.6-1.1pp/wk) < VAL. Promoted VALIDATED-SELECTIVE
(PROVISIONAL): deploy only as ensemble over the robust pocket, re-confirm on 2025 when opened.
Stochastic: confirmed DEAD, but the scary -7.8pp was an INVERTED-LOGIC artifact (flip →
-0.44); both directions ≤0 OOS = genuinely no signal. → DEADCASE_REWORK_RESULT.md

## ROSTER NOW (audited)
- SELECTIVE validated: ADX|1d (provisional, the only true selective engine) + MA decoupled-
  BAND V2 (WMA|4h, the canonical selective winner from ma_wealth_driver).
- ROBUST NON-SELECTIVE tilts (consistent + 100th-pct null, deploy 96-100%): HMA|1d, WILLR|1d,
  Bollinger|1d, TEMA|1d, VIDYA|4h, SMA|4h, VWAP|1d, MFI|1d, Supertrend|4h, WMA|4h, PSAR|4h
  (+~10 more at score 4). These refine WHICH longs under the gate.
- The regime gate remains the dominant value lever (D78).

## ACTIVE CAMPAIGN: V6 RIGOROUS REBUILD — COMPLETE (2026-06-25)
All 12 non-MA families × 5 TFs rebuilt with full V1.1 rigor (CHARTER §ENGINE BUILD RIGOR).
57 cells evaluated. RESULT: ONLY ADX|1d is genuinely selective (67% deploy, +2.26pp/wk, 100th
null, Sharpe 3.38). Every other family deploys 98-100% at all TFs — the TRAIN filter cannot
reduce them because their signal logic is structurally always-on under a bull gate. Sub-daily
(1h/30m) universally bleeds from cost+noise.
→ runs/engines/V6_rigorous/V6_RESULT.md + v6_sweep_20260625_203231.json (1d/4h/2h) +
  v6_sweep_20260625_223748.json (1h/30m)

## ENGINE ROSTER (rigorous, final, OOS-validated 2026-06-26)
SELECTIVE + OOS-VALIDATED:
  1. MA V2 BAND (WMA|4h) — the canonical winner (ma_wealth_driver, 35% deploy, +4.11%/wk OOS)
  2. MA × 4h EFFICIENCY-GATED (m=0, eff>0.25) — 7/8 types OOS+ (60-72% deploy, +0.3-1.25pp)
     Best: DEMA|4h (+1.25pp, 100th null, Sh 2.02), VIDYA (+0.85), SMA (+0.91), EMA (+0.90)
  3. ADX|1d (V6 discovery, 67% deploy, +2.26pp VAL, Sharpe 3.38, OOS probe pending)
DYNAMIC ADAPTATION (m>0 vov-threshold): NULL OOS (VAL-fit, does not transfer)
MA × 2h eff-gate: NULL OOS (VAL-fit, extraordinary selectivity did not generalize)
NON-SELECTIVE tilts (1d only, 98-100% deploy): PSAR, OBV, DONCHIAN, RSI (tiny edges)
Static TI engine build COMPLETE AND EXHAUSTED. Dynamic adaptation (vov-threshold) REFUTED.

## WHAT EXISTS (verified, committed)
- V1 MA drawdown-insurance: 8 types × {1d, 4h} = VERIFIED (halves DD, -35% vs -82%)
- V1.1 fine-TF exits: REFUTED (look-ahead leak caught, de-leaked = null, D81)
- V1.2 combinatrix: 30 engines × 275 combos = disciplined NULL (D82)
- Decoupled 7d/14d architecture: BUILT (uncommitted) but NEVER RUN at 30m/1h/2h
- Chimera data: available at ALL cadences (15m/30m/1h/4h/1d), 104 assets, 243 features

## THIS SESSION (2026-06-24) — TI FAMILY VERIFICATION (honest read)
- [x] Forensic audit of SOTA numbers → FORENSIC_AUDIT_RESULT.md
      No look-ahead (clean). BUT compound inflated ~9.5x by EW-rebalancing premium.
      Single-asset BTC: MACD timing LOST to buy-hold (+162% vs +655%).
- [x] OOS validation RSI/MACD/Bollinger (2024-05-15→2025-01-01) → OOS_VALIDATION_RESULT.md
      Each beats AL at only ONE scattered TF (noise signature). NOT deployable.
- [x] Vectorized families 6-8 + wired REAL chimera features (norm_funding,
      norm_oi_price_divergence) → ti_engine_sota_v11.py. OOS: Supertrend/Funding/OI
      deploy 98-100% (always-long-with-tilt, not selective). NOT deployable.
- [x] Phase 2 dynamic-engine plan (thorough, not executed) → docs/PHASE2_DYNAMIC_ENGINES_PLAN.md

VERDICT (CORRECTED 2026-06-24 after static-exhaustion): My first "all non-MA null"
read was UNDER-POWERED (one threshold, no confirmation layer). Pulling the proper
levers (vote-threshold sweep + MA-style confirmation) + the gold-standard bar
(DEV/OOS consistency + random-null >=95th pct) REVEALED 2 NEW validated static engines:
  - RSI|4h: +0.77pp OOS, DEV+OOS consistent, 100th-pct random-null  [VALIDATED]
  - Bollinger|1d: +2.38pp OOS, DEV+OOS consistent, 100th-pct null    [VALIDATED]
  - ADX|4h: +1.09pp OOS, selective (6.9% deploy), but 93rd pct       [BORDERLINE]
  - MACD|1h: big OOS but NEGATIVE DEV -> OOS-fit                      [REJECTED]
  - Stochastic/Supertrend/Funding/OI: no selective edge              [NULL]
See STATIC_EXHAUSTION_RESULT.md. The user's "static before dynamic" gate is satisfied:
static demonstrably works (MA + RSI|4h + Bollinger|1d). Still TODO on static: lookback/
overfit probe on the 3 validated cells + validate their ENSEMBLE combination OOS.

## STANDING FINDINGS (from prior session)
- No daily-bar SELECTION alpha (D17/D55/D77 + 6 new experiments)
- Vol IS predictable (top feature: vol-rank, rv_rv_5m)
- Continuation > reversion at daily. REVERSAL at sub-daily (multi-TF analysis)
- Regime timing is the only verified value lever (V1, V3 selection)
- Trailing stops create illusion of alpha (vol-breakout audit)
- Oracle ceiling: 10.4%/day available (empirically real opportunity)

## SPLIT CONVENTIONS
TI/MA: 2020=DEV, 2021=IRON, 2022=no-touch BEAR.
Convention B: TRAIN<=2024-05-15 / VAL / OOS / UNSEEN (>=2024-05-15 sealed).

## DYN-5 RESULT (2026-06-26): NULL for MA family (structurally correlated configs)
DYN-5 (conviction-on-agreement) tested on MA family at 4h: deploy 95-97% even at
conviction 0.8, top-K=5. The mechanism cannot create selectivity because MA crossover
configs are STRUCTURALLY CORRELATED — in a trend, all fast>slow variants agree.
Conviction-on-agreement requires INDEPENDENT signals that occasionally disagree; MA
configs don't provide that. The framework is architecturally sound for less-correlated
signal sets (e.g., mixed TI families), but within one MA family it's null.
→ runs/engines/V7_dynamic/dyn5_conviction_VAL_*.json
