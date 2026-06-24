# CURRENT — live state (Kiro-written each cycle; committed; <2KB)

PHASE: VERIFIED (Combined V1+V2 architecture — regime gate + trend amplifier)
UPDATED: 2026-06-24

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
Rebuild TI-based trading engines for max profitability within 7d windows.
Target: per-slice profit at {30m, 1h, 2h, 4h, 1d} × {EMA, HMA, DEMA} × {7d, 14d lookback}.
The decoupled architecture (14d signal, 7d trade) is UNTESTED at sub-day cadences.

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

VERDICT: MA family remains the ONLY OOS-validated, random-null-beating, deployable
engine set (+4.11%/wk OOS). No non-MA TI family survives OOS. The user's "too good
to be true" instinct was correct — the headline DEV compounds were inflated.

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

## NEXT PHASE (after static build): DYNAMIC/ADAPTIVE ENGINES
Static baseline now COMPLETE + verified (Phase 1 done: MA validated, non-MA null OOS).
Thorough plan written: docs/PHASE2_DYNAMIC_ENGINES_PLAN.md (NOT executed — awaiting go).
Phase 2 builds ADAPTIVE engines where the 7d-14d lookback adjusts params (thresholds,
periods, sizing) from the signal's OWN recent behavior (endogenous, NOT external→D73).
3 first candidates: DYN-3 MA-EA (efficiency-adaptive period, augments the WINNER),
DYN-1 RSI-VA (vol-adaptive threshold), DYN-2 BB-VA (bandwidth-adaptive). Each must beat
its own static twin + random-null + OOS, with a smooth overfit-probe surface.
