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

## THE HONEST FALLBACK
If all TI × TF × window cells are null: the verified V1 drawdown-insurance (capital preservation
at 1d/4h) stands as the deployable ceiling. But: sub-day (30m/1h/2h) with decoupled lookback/hold
is GENUINELY UNTESTED with the correct (de-leaked) architecture. Test it before accepting null.

## FUTURE: DYNAMIC/ADAPTIVE ENGINES (noted, not yet built)
After the static engine build is complete (8 TI families × 5 TFs), the NEXT phase is:
**Adaptive engines that USE the lookback window to ADAPT their parameters** — not just confirm
a fixed signal, but actively adjust thresholds/periods based on recent regime/vol/momentum.
The 7d-14d lookback was designed for this: the past informs current-window decisions.
D73 killed weekly config-mapping with EXTERNAL factors, but per-bar adaptation using the
SIGNAL'S OWN recent behavior (e.g., RSI threshold widens in high-vol, Bollinger adapts bandwidth)
is a genuinely different mechanism — untested. This is the Phase 2 (ML-augmented) track from the
WEALTH_BOT_DEVELOPMENT_FRAMEWORK. Build the static baseline FIRST, then augment.
