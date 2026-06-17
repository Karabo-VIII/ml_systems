# Orthogonal-beta diversification -- does mean-reversion diversify the trend ceiling?

The MA+breakout ensemble failed to diversify because both are SAME beta (long-only crypto trend, corr high).
The textbook escape from a beta-tail ceiling is an ORTHOGONAL beta. Mean-reversion (buy oversold, sell at
the mean) is a DIFFERENT beta (contrarian/short-vol). This tests whether a TREND+MR ensemble lifts the
held-out p05 where trend+trend didn't. Tool: `src/strat/mr_diversify.py`. 4h, u10 book, UNSEEN sealed.
Honest prior: the dead-list (D37 crypto-trends-not-reverts; D48/D49 buy-the-extreme anti-edge) says MR is
individually weak -- but its DIVERSIFICATION value (anti-correlation) is a SEPARATE question.

## Result (FULL stack, 4h)
| book | bear | VAL | OOS | OOS breadth | OOS Sharpe | OOS-heldout p05 |
|---|---|---|---|---|---|---|
| MR_only | -14.8 | -12.9 | +3.1 | 54% | 0.28 | -28.86 |
| BREAKOUT | -3.1 | 19.1 | +11.0 | 72% | 0.72 | -21.21 |
| **BO+MR ensemble** | -6.5 | 10.1 | +9.5 | 66% | **0.70** | **-17.37** |

## The finding: orthogonal-beta diversification WORKS (directionally) -- the first lever that moved p05
- **corr(breakout daily, MR daily) = +0.15** -- genuinely orthogonal (vs MA+breakout which were highly
  correlated; that is exactly WHY the same-beta ensemble failed and this one helps).
- **The ensemble IMPROVED the OOS-heldout p05: -21.21 -> -17.37** (~18% less-negative tail) while KEEPING
  breakout's Sharpe (0.70 vs 0.72) and most of its compound (9.5 vs 11.0) AND improving maxDD (-13.8 vs
  -15.4). Combining a strong trend book with a weak-but-uncorrelated MR book shaved the tail at ~no return
  cost -- the diversification the same-beta MA+breakout ensemble (p05 got WORSE, -27) could not deliver.
- MR alone is weak (OOS Sharpe 0.28, +3.1%) -- consistent with the dead-list. It earns its place as a
  DIVERSIFIER (uncorrelated), not as alpha.

## But the ceiling still holds -- and the path is now concrete
**The ensemble p05 is still NEGATIVE (-17.37).** One weak orthogonal sleeve moves the tail toward 0 but
does not cross it. So:
1. **Orthogonal-beta diversification is the RIGHT lever** -- the first thing in the whole arc that pushed
   the held-out p05 toward robustness (every same-beta move -- MA, more configs, ensemble, regime gates --
   did not). The ceiling is NOT "no robust internal book is possible"; it is "no SINGLE archetype clears
   it, and you must STACK ORTHOGONAL betas to diversify the trend-beta tail."
2. **One MR sleeve is not enough.** The forward path is a RISK-PARITY stack of MULTIPLE uncorrelated
   sleeves (other MR variants, cross-sectional dispersion, carry/funding-harvest, calendar) -- each
   individually weak but mutually orthogonal -- continuing to push p05 toward 0. Each new sleeve must be
   ORTHOGONAL (low corr to the existing book), NOT another trend variant (that is the same-beta trap).
3. This REFRAMES the session's ceiling: the unblocked internal lever (stack orthogonal betas) is real and
   working -- it just needs more orthogonal sources than one weak MR to cross the bar. External data /
   shorting remain the bigger-orthogonality options, but internal orthogonal diversification is NOT
   exhausted -- it is the demonstrated, unblocked next direction.

Caveat: one MR construction, one OOS window, TRAIN-era; the -21->-17 move is real but modest (confirm it
is not one-window noise by adding sleeves + a second held-out span). UNSEEN never touched.
json: `mr_diversify.json`. RWYB: `python -m strat.mr_diversify`.

## CORRECTION (mr_robust.py, 2-span x 4-seed check) -- the diversification is SPAN-DEPENDENT, NOT robust
The honest caveat above was tested directly. Result:
| span | breakout p05 | BO+MR p05 | ens improves p05? |
|---|---|---|---|
| **OOS** (25-03..12) | ~-21 | ~-17 | **YES, 4/4 seeds** (+3.5..+4.6) |
| **VAL** (24-05..25-03) | ~-16 | ~-20 | **NO, 0/4 seeds** (-3..-5) |
The p05-lift is SEED-robust WITHIN each span but REVERSES across spans: MR diversified on OOS (where MR was
mildly +3.1%) but HURT on VAL (where MR lost -12.9% and dragged the book 19.1 -> 10.1). A real diversifier
helps -- or at least does not hurt -- in BOTH held-out windows. **So the orthogonal-beta diversification
with one WEAK MR sleeve is NOT a robust lever**: the sleeve's own regime-dependent weakness dominates its
diversification value. The -21->-17 OOS result was a one-window artifact (seed-robust but not span-robust)
-- exactly the trap the 2nd-span check exists to catch.

CORRECTED VERDICT: every internal lever tested in this arc (archetype MA/breakout, family-fix, overlays,
price/funding regime gates, same-beta ensemble, timeframe sweep, orthogonal-MR diversification) fails to
produce a held-out-robust positive book. Breakout @4h is the best BASE (best OOS), but p05<0 and the one
promising diversification lead did not survive a second span. The escape requires a genuinely RELIABLE
orthogonal source (not a weak regime-dependent MR) -- i.e. external/leading data, shorting, or a non-crypto
beta. The ceiling is real and now thoroughly mapped. UNSEEN never touched. RWYB: python -m strat.mr_robust.
