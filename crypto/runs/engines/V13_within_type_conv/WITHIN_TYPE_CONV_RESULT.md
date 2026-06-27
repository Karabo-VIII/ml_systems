# WITHIN-TYPE TOP-5 CONVERGENCE — RESULT 2026-06-27

USER MODEL (exact): each MA type is independent (SMA != EMA). Within EACH type, take its
OWN top-5 configs, deploy separately (no voting), grade signal by how many of THAT TYPE'S
top-5 converge, size by convergence.

ANTI-OVERFIT: top-5 ranked on TRAIN+VAL (2020..2024-05-15), FROZEN, tested OOS-2024.

| type | full band | top5 flat | convweight | gate>=3 | gate>=4 | gate>=5 |
|------|-----------|-----------|------------|---------|---------|---------|
| SMA  | **+1.67** | +1.39 | +1.31 | +1.02 | +0.75 | +0.60 |
| EMA  | **+1.68** | +1.60 | +1.61 | +1.42 | +1.44 | +1.46 |
| WMA  | **+1.60** | +1.36 | +1.34 | +1.09 | +1.06 | +0.79 |
| VIDYA| **+1.57** | +1.41 | +1.35 | +0.93 | +1.18 | +1.37 |
| KAMA | +1.58 | +1.36 | +1.48 | +0.99 | +1.72 | +1.51 |
| DEMA | **+1.17** | +0.96 | +0.93 | +0.88 | +0.82 | +1.06 |
(lin %/wk OOS-2024, 15% cap, prod cost)

## VERDICT: within-type top-5 convergence does NOT beat the full band (5/6 types).
Tightening the gate (>=3,>=4,>=5) generally HURTS (SMA +1.67 -> +0.60 at gate5). The one
exception (KAMA gate>=4 +1.72) is 1 of 6 types x several gates = multiple-comparison noise.

ROOT CAUSE: the 5 top configs WITHIN one type are NEAR-CLONES (all SMA crossovers / all EMA
crossovers). Requiring them to converge gives NO diversification — it's one opinion measured
5 ways, not 5 independent confirmations. So within-type convergence only shrinks deployment
without adding信息. Contrast: ACROSS-type convergence (V12: SMA vs EMA vs WMA — genuinely
different smoothers) DID add a small lift (+1.71 vs +1.61) because those ARE independent.

## CONSOLIDATED SELECTION VERDICT (4 independent tests, all consistent)
1. V9 rolling band reselect -> loses to frozen
2. V11 honest top-K band -> loses to full band
3. V13 within-type top-5 convergence (this) -> loses to full band
4. V12 ACROSS-type convergence -> small WIN (+0.1pp, Sharpe up)
=> The FULL BAND per type is the robust unit. Config concentration/selection does not add
OOS edge (config performance non-persistent — D62/D73 family, now 6 confirmations).
The ONLY convergence that helps is ACROSS different MA TYPES (orthogonal smoothers).

## DEPLOYABLE LANDING
Run each MA type's FULL frozen band independently; size by CROSS-TYPE convergence
(conv-weighted, V12) = +1.71%/wk, Sharpe 1.60, MaxDD -25% @ 15% cap (prod, OOS).
MA ceiling ~+1.7%/wk at prudent cap. 5%+ needs concentration (risk dial) or new signal family.
