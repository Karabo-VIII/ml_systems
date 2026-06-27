# NORTH STAR RECKONING — per-slice ROI vs the mandate (2026-06-27)

## The mandate (CHARTER North Star, quoted)
"PROFITABLE within the smallest time window without decay":
- PRIMARY: profitable within ANY 7-day window (per-slice ROI > 0) -> target ~100% of windows
- STRETCH: 3d and 1d windows
- NOT optimizing for weekly magnitude / 1-5%/day (retired, D77)
The metric is CONSISTENCY (fraction of windows positive), not weekly % size.

## Where we ACTUALLY land (deployable cross-type conv book, 1d, prod, OOS-2024)
| slice | % profitable | median | mean | p10 | worst |
|-------|--------------|--------|------|-----|-------|
| 7d (PRIMARY) | **34%** | +0.00% | +2.34% | -5.64% | -12.4% |
| 3d (stretch) | 39% | +0.00% | +0.82% | -4.06% | -15.9% |
| 1d (stretch) | 32% | +0.24% | +0.24% | -2.74% | -14.4% |

## VERDICT: we FAIL the primary mandate. ~34% of 7d windows profitable (target ~100%).
The +1.65%/wk average is a RIGHT-SKEWED mean (median +0.00%): ~34% big winners carry it,
~66% are small losers. Classic trend-following asymmetry: lose small often, win big rarely.

## WHY this is structural, not a tuning gap
Trend-following is INHERENTLY low-hit-rate / high-payoff. You cannot make a pure trend
engine have ~90%+ positive 7d windows — the mechanism (ride trends, cut chop) GUARANTEES
it bleeds in the (majority) choppy windows. We've now confirmed BOTH ceilings:
- MAGNITUDE: ~+1.7%/wk at prudent cap (V11/V13: config selection adds nothing OOS)
- CONSISTENCY: ~34% of 7d windows (this) — structural to trend-following

The user's "1.65% is low" is directionally right but the real miss is CONSISTENCY, not size.
A +1.65%/wk engine with 90% windows-positive would SATISFY the mandate; ours at 34% does not.

## THE STRATEGIC IMPLICATION (the real path forward)
"Profitable within ANY 7-day window" REQUIRES engines whose winning windows are
ANTI-CORRELATED with the trend engine's losing windows. Pure trend cannot self-satisfy it.
Path:
  1. MEAN-REVERSION / market-neutral components that profit in chop (the 66% losing windows)
  2. ORTHOGONAL signal families (vol-clustering, deliberate asymmetric/Pareto capture,
     asset rotation) selected for ANTI-correlation with the trend book, not just standalone edge
  3. A DIVERSIFIED MULTI-STRATEGY book where different engines win in different regimes
     -> the only structural route to high per-window consistency.

The MA/trend engine is ONE COMPONENT (the bull-trend harvester), not the whole system.
NEXT CAMPAIGN: build the anti-correlated complements, measure per-7d-window consistency of
the COMBINED book (not standalone weekly %). That is the North Star metric.
