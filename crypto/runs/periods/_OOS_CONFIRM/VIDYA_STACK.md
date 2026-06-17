# Carrying the VIDYA 2MA/3MA upgrade forward -- stack, 3MA, multi-bear

On-scope continuation of `ma_type_upgrade.py` (VIDYA upgraded the EMA cross). Three questions answered.
Tool: `src/strat/vidya_stack.py`. 4h, u10 book, UNSEEN sealed.

## (1) VIDYA STACKS with the overlays -- the best held-out 2MA result of the arc
| variant (2MA-slow) | bear | VAL | OOS | OOS Sharpe | OOS-heldout p05 |
|---|---|---|---|---|---|
| EMA_FIXED | -9.9 | 26.2 | -0.0 | 0.17 | -39.1 |
| EMA_FULL | -6.1 | 5.5 | +2.0 | 0.23 | -33.0 |
| VIDYA_FIXED | -0.8 | 20.8 | +4.7 | 0.34 | -38.0 |
| **VIDYA_FULL** | -1.1 | 2.8 | **+11.2** | **0.89** | **-14.7** |
VIDYA + (trail10 + minhold12 + maker) STACK -- they do not fight. VIDYA_FULL: OOS +11.2%, Sharpe 0.89,
maxDD -10.8%, p05 -14.7. This is the best held-out 2MA result of the entire arc and it EDGES OUT the
breakout detour (OOS Sharpe 0.72, p05 -21) -- the adaptive-MA UPGRADE beats the other-indicator detour,
exactly as the user's redirect implied.

## (2) VIDYA helps the 3MA too (generalizes within the MA family)
| variant (3MA-slow) | bear | VAL | OOS |
|---|---|---|---|
| EMA_FULL | -4.0 | 15.7 | +2.3 |
| VIDYA_FULL | -0.3 | 6.2 | **+8.6** |
The upgrade is not 2MA-specific -- VIDYA lifts the fragile 3MA OOS (2.3 -> 8.6) and bear (-4.0 -> -0.3).

## (3) The bear-sidestep is REGIME-SPECIFIC, not universal (honest)
| bear window | EMA | VIDYA | delta |
|---|---|---|---|
| 2021-05 crash | -7.5 | **-19.3** | -11.8 (WORSE) |
| 2022-06 bear | -9.9 | -0.8 | +9.1 |
| 2022-09 bear | -5.7 | -4.2 | +1.5 |
| 2022-11 FTX | -9.1 | **-11.1** | -2.0 (WORSE) |
| 2024-08 unwind | -8.6 | -7.2 | +1.4 |
VIDYA less-bad in 3/5. It sidesteps SLOW GRINDING bears (sits still without a clean trend) but HURTS in
SHARP V-CRASHES (2021-05, FTX): its adaptiveness turns fast, gets long on the dead-cat bounce, and gets
caught. **The -0.8% headline (Jun-2022) was a favorable slow-bear, NOT representative of all bears.**

## Verdict
**VIDYA is a real, on-scope upgrade to the 2MA/3MA** -- it stacks with the overlays (VIDYA_FULL OOS +11.2%,
Sharpe 0.89, maxDD -10.8%, p05 -14.7 = best of the arc, beats the breakout detour), generalizes to 3MA, and
sidesteps slow bears. It is the new MA baseline. HONEST limits: p05 still -14.7 (<0, not absolute-robust,
but the closest yet -- halved from EMA's -33); the bear benefit is slow-bear-specific (sharp crashes hurt);
one OOS window. NEXT on-scope: 2-span x seed robustness of VIDYA_FULL's p05 lift (is -14.7 real?), and
whether the trail rescues VIDYA's sharp-crash weakness. json: `vidya_stack.json`. RWYB: python -m strat.vidya_stack.

## ROBUSTNESS (vidya_robust.py, 2-span x 4-seed) -- the p05 upgrade is SPAN-ROBUST (unlike MR)
| span | EMA_FULL p05 | VIDYA_FULL p05 | VIDYA better (seeds) | compound | Sharpe |
|---|---|---|---|---|---|
| VAL | -32.5 | **-20.7** | **4/4** (+11.7) | EMA wins (5.5 vs 2.7) | EMA wins (.37 vs .28) |
| OOS | -33.0 | **-14.7** | **4/4** (+18.2) | VIDYA wins (11.2 vs 2.0) | VIDYA wins (.89 vs .23) |
**VIDYA robustly improves the TAIL (p05) on BOTH held-out spans, 4/4 seeds each** -- UNLIKE the MR lead
which reversed on VAL. This is the only lever in the whole arc to move p05 toward 0 on BOTH spans. The
RETURN advantage is regime-dependent (VIDYA wins the hard/choppy OOS, gives up upside on the bull-ish VAL
-- the adaptive-MA tradeoff). So VIDYA is a SPAN-ROBUST RISK/drawdown upgrade to the 2MA/3MA (better tail
+ maxDD on both spans), with a defensive regime-dependent return profile. Still p05<0 on both (-20.7/-14.7)
= not absolute-robust, but a real, replicated improvement. For the North Star (robust compound + maxDD<30%)
the tail upgrade is the part that matters. json: `vidya_robust.json`. RWYB: python -m strat.vidya_robust.
