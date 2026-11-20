# Archetype diversification (breakout + MA) -- does it clear the p05 ceiling?

Breakout beat MA OOS but both fail absolute robustness (OOS-heldout block-bootstrap p05 < 0). The textbook
lever for a fragile-but-positive book is DIVERSIFICATION across uncorrelated sources. MA-cross and Donchian
breakout capture trends differently, so an equal-weight ENSEMBLE *should* have a less-negative tail. Tested
pre-registered (MA-only / breakout-only / ensemble, all reported). Tool: `src/strat/ensemble_arc.py`. 4h,
FULL stack, u10 book, UNSEEN sealed.

## Result (FULL stack, 4h)
| book | bear | VAL | OOS | OOS breadth | OOS Sharpe | OOS-heldout p05 |
|---|---|---|---|---|---|---|
| MA | -6.1 | 5.5 | +2.0 | 53% | 0.23 | -32.97 |
| **BREAKOUT** | **-3.1** | 19.1 | **+11.0** | **72%** | **0.72** | **-21.21** |
| ENSEMBLE (MA+breakout) | -4.6 | 12.3 | +6.5 | 62% | 0.45 | -27.28 |

## Finding: diversification did NOT help -- it AVERAGED
The ensemble sits at the MIDPOINT of MA and breakout on EVERY metric (OOS +6.5 between +2.0/+11.0; Sharpe
0.45 between 0.23/0.72; p05 -27 between -33/-21; breadth 62% between 53/72). **There is no diversification
benefit** -- because MA-cross and Donchian breakout are NOT uncorrelated: both are long-only crypto-trend
beta, long in the same uptrends, flat/whipsawed in the same chop. Mixing a worse book (MA) into a better
one (breakout) just drags the better toward the average.

## Implications
1. **Don't ensemble across same-beta trend archetypes** -- it dilutes the better one. Pick the best
   archetype (breakout) and run it clean.
2. **The p05 ceiling is STRUCTURAL: it is the long-only crypto-trend BETA tail itself.** Diversifying
   across strategies that all ride that same beta cannot escape it (the ensemble p05 is still -27). The
   only escapes are genuinely ORTHOGONAL return sources: a different beta (not crypto-trend), SHORTING
   (out of scope), or EXTERNAL/leading information. This converges with the entire dead-list.
3. The MA-centric arc was doubly too-narrow: wrong base archetype (breakout is better) AND an MA+breakout
   ensemble would have made it worse, not more robust.

Caveat: one ensemble construction (equal-weight), one OOS window, TRAIN-era; UNSEEN never touched.
json: `ensemble_arc.json`. RWYB: `python -m strat.ensemble_arc`.
