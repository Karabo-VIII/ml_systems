# CHARTER — binding intent for crypto work (Kiro-owned; the user delegated this)

READ THIS FIRST, every cycle. The user does NOT ratify rules — Kiro owns this file and updates it as intent
sharpens. It is the <4KB distillation of the user's mandates so a fresh context starts WARM, not cold.

## OPERATING MODE — AUTONOMOUS BUILD (6h mandate, 2026-06-24)
Kiro runs autonomously: BUILD, TEST, ITERATE on TI-based trading engines. Report progress every ~15 min.
Own all three context files (CHARTER, CURRENT, KNOWLEDGE_MAP). Use Kiro specs for formal tracking.

## THE NORTH STAR (REFRAMED 2026-06-24)
Build TI-based trading engines that are **PROFITABLE within the smallest time window without decay**:
- Primary target: **profitable within any 7-day window** (per-slice ROI > 0)
- Stretch: profitable within 3d and 1d windows
- Lookback: 7d-14d (recent data informs current window decisions; a crossover in the last 7d
  is more relevant than one from 2 weeks ago — recency decay)
- NOT optimizing for 1-5%/day oracle capture (that was the wrong goal — confirmed D77)
- NOT per-candle prediction (dead — D55/D13)
- YES: per-SETUP, per-MOVE profitability within short windows

## THE BINDING SURFACE
- **TFs = {30m, 1h, 2h, 4h, 1d}** — all sub-day cadences included. 1d is macroscopic but included.
- **UNIVERSE = u100** — never collapse to single-asset.
- **TI families = the 8 MA types** (EMA/SMA/WMA/HMA/DEMA/TEMA/KAMA/VIDYA) as starting point,
  expandable to other indicators (RSI, MACD, Bollinger, etc.)
- **Architecture = TI × Timeframe** — engines are NOT tailored per-instrument initially
  (but buckets/clusters may emerge from the data).
- **Lookback = {7d, 14d}** — the decoupled signal window. Recent signals > stale signals.
- **Hold/Slice = {3d, 7d}** — the max trade duration per position.

## PERMANENT CONSTRAINTS
- LONG-ONLY + SPOT, lev=1. Short is OFF THE TABLE.
- Maker cost 0.0006 RT (wealth leg) / Taker 0.0024 RT (capture leg).
- SELECT on TRAIN+VAL, never OOS. DEV wall < 2024-05-15.
- WEALTH = compound net return (the objective). Sharpe is secondary.
- Run-What-You-Build (RWYB): code runs with real data before any claim.

## WAY OF WORKING
- Kiro MANAGES and BUILDS. Does not defer revertible decisions.
- Decompose the ideal; don't quit at a verdict. One null ≠ the problem is dead.
- Adversarial verification on ≥3 axes before any success claim.
- Use the committed infra: `ma_movecatch_engine`, `ma_wealth_driver`, `portfolio_replay`,
  `capture_lab`, `fleet_lab`, `ChimeraLoader`.
- Commit working code. Tag findings honestly.

## ENGINE BUILD RIGOR (BINDING — the V1.1 MA standard, 2026-06-25)
Every TI engine, regardless of family, MUST be built to the same rigor as the validated
MA family (V1.1/V2). This is NOT optional — it was a CORRECTIVE after the V3/V4 campaigns
produced under-powered builds. The requirements, per engine:

1. **SYSTEMATIC CONFIG GRID** — 50-100+ parameter combinations, not 12 hand-picked
   round numbers. Sweep the full meaningful range per indicator (period/threshold/multiplier).
2. **DECOUPLED LOOKBACK WINDOW** — the signal uses only the trailing 7d or 14d of bars
   to decide, not the full history. Signal relevance decays; recent > stale.
3. **MULTI-LAYER CONFIRMATION** — raw signal AND close > EMA(lookback_bars) AND EMA rising.
   This confirmation layer is what makes engines SELECTIVE (35% deploy, not 95%+).
4. **PORTFOLIO REPLAY via `ma_wealth_driver.run_wealth()`** (or equivalent audited harness) —
   proper lag-1 positions, MtM reconciliation, maker cost, WALL assertion.
5. **ALL 5 TIMEFRAMES** — {1d, 4h, 2h, 1h, 30m}. Do NOT optimise for one TF. Report all.
6. **TRAIN→VAL iteration** — tune on TRAIN (≤2023-07-01), validate on VAL (..2024-05-15),
   iterate, then freeze. Only after freezing: single OOS test on 2024 remainder.
7. **RANDOM-NULL (n≥100)** — engine must beat random ensemble at same deploy rate under
   same gate at ≥95th percentile. This isolates TI timing from being-long-under-the-gate.
8. **OVERFIT PROBE** — the hyperparameter surface around the best cell must be a SMOOTH
   plateau, not an isolated spike. Sensitivity = suspect; plateau = robust.

If ANY engine build does not meet these 8 requirements, it is INCOMPLETE and cannot be
claimed as validated. This applies to ALL current and future builds.

## THE HONEST FALLBACK
If all TI × TF × window cells are null: the verified V1 drawdown-insurance (capital preservation
at 1d/4h) stands as the deployable ceiling. But: sub-day (30m/1h/2h) with decoupled lookback/hold
is GENUINELY UNTESTED with the correct (de-leaked) architecture. Test it before accepting null.

## FUTURE: DYNAMIC/ADAPTIVE ENGINES (Phase 2 — ACTIVE, design approved 2026-06-25)
Design: docs/PHASE2_DYNAMIC_ENGINES_DESIGN.md (supersedes the earlier PLAN).
Build order: DYN-3 MA-EA (augment winner) → DYN-4 ADX-EA (augment V6 discovery) →
DYN-1 RSI-VA → DYN-2 BB-VA. Each: one free param, endogenous adaptation, static-twin A/B,
random-null + adaptation-null, OOS-2024, overfit-probe. Kill if deploy≥90% or ≤static-twin.
D73 (weekly external-factor config-mapping) is DEAD — Phase 2 uses ENDOGENOUS state vars
(vol, efficiency, autocorrelation from the signal's OWN price) which is mechanically different.
