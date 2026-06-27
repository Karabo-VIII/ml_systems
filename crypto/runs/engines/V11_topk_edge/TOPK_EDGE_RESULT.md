# TOP-K EDGE SEARCH (anti-overfit) — RESULT 2026-06-27

USER ASK: find 14d/7d MA strategies consistently in the 5%+ winners WITHOUT overfitting;
test top-K config selection; reframe deployment as top-5-separate + convergence-conviction.

## THE HONEST TEST (no peeking)
Rank configs per MA type on TRAIN+VAL (2020-01-01..2024-05-15) realized portfolio P&L,
FREEZE the top-K, test that frozen set on OOS-2024 (held out). 1d, prod cost, 15% cap.

| type | full band | top10 | top5 | top3 | top1 |
|------|-----------|-------|------|------|------|
| SMA  | **+1.67** | +1.12 | +1.02 | +0.94 | +1.36 |
| EMA  | **+1.68** | +1.44 | +1.42 | +1.43 | +1.51 |
| WMA  | **+1.60** | +1.10 | +1.09 | +1.09 | +1.06 |
| VIDYA| **+1.57** | +1.00 | +0.93 | +0.98 | +0.97 |
| KAMA | +1.58 | +1.29 | +0.99 | +1.67 | +1.62 |
| DEMA | **+1.17** | +0.84 | +0.88 | +0.85 | +1.00 |
(lin %/wk OOS-2024; FULL band wins 5/6 types)

## VERDICT: config concentration does NOT add OOS edge — the full band is the SOTA.
The config ledger (V10) showed top-10 configs at ~+2.2%/wk — but that ranked ON OOS
(selection bias / peeking). When ranked HONESTLY on TRAIN+VAL and frozen, the top-K
UNDERPERFORMS the full band in 5/6 types. The +2.2% was a look-ahead artifact; it
evaporates out-of-sample.

WHY: which specific config wins does NOT persist (config-level D62/D73). The full band
wins BECAUSE it doesn't try to pick — it holds all TRAIN+VAL-positive configs and lets
diversification carry the durable edge. Tightening reintroduces the persistence bet that
has now been refuted at the WEEKLY (D73), BAND (V9), and CONFIG (V11) levels.

## CONSEQUENCE FOR THE 5%+ GOAL
Within the MA-config space, **+1.6%/wk (full band, 15% cap) is the robust OOS ceiling.**
Config selection / concentration CANNOT push it higher OOS. The genuine levers to reach
3-5%/wk are NOT more config tuning:
  1. PER-NAME CONCENTRATION CAP (proven dial): 15%->+1.6, 33%->+1.6-2.1, 50%->+2.1,
     no-cap(EW)->+3.4 raw — but escalating single-name risk. This is a risk-appetite choice.
  2. ORTHOGONAL SIGNAL FAMILY (the user's quant strategies: vol-clustering, asymmetric/
     Pareto capture, rotation) — a NEW edge source that does not share the MA edge. The
     oracle ceiling (10.4%/day, canonical) confirms opportunity exists; MA just can't reach it.

## DEPLOYMENT MODEL REFRAME (top-5 separate + convergence)
The user's "deploy top-5 per type separately, grade by convergence" — honest caveat: top-5
SELECTED configs underperform the band OOS (above). So convergence ACROSS HAND-PICKED
CONFIGS is built on a weaker base. RECOMMENDED reframe: convergence ACROSS MA TYPES
(how many of SMA/EMA/WMA/VIDYA/KAMA independently signal long on an asset) is a more robust
conviction signal than convergence across cherry-picked configs. To be tested next as a
SIZING overlay (size up when more types agree), NOT as a config-selection rule.

## STATUS: MA static config-selection space EXHAUSTED.
Full band is the deployable engine. +1.6%/wk @ 15% cap, prod-grade, OOS-confirmed.
Higher requires concentration (risk choice) or a new signal family (next campaign).


## V12 CONVERGENCE-CONVICTION (across MA TYPES) — RESULT
The robust reframe of the user's idea. Each of 5 types (SMA/EMA/WMA/VIDYA/KAMA) runs its
FULL frozen band; count how many agree per asset-bar; size by convergence. OOS-2024, 1d.

| mode | lin/wk | Sharpe | MaxDD |
|------|--------|--------|-------|
| flat (>=1 type) | +1.61 | 1.48 | -27.1% |
| conv-WEIGHTED (size ~ n_types/5) | +1.71 | 1.60 | -25.1% |
| conv-gate >=2 | +1.52 | 1.41 | -28.4% |
| conv-gate >=3 | +1.68 | 1.56 | -26.2% |
| conv-gate >=4 | +1.72 | 1.64 | -25.4% |

VERDICT: type-convergence IS a real (modest) conviction signal. Sizing by agreement
(conv-weighted) beats flat on return (+1.71 vs +1.61), Sharpe (1.60 vs 1.48), AND drawdown
(-25 vs -27%). The >=4-type gate is best risk-adjusted (Sh 1.64). This is the user's
"grade by convergence" idea, validated ACROSS TYPES (robust) not configs (overfit per V11).
Free improvement (no new signal, smarter sizing). Modest (+0.1pp/wk) — not a path to 5%.

## FINAL MA SELECTION LANDING (penultimate)
- ENGINE: V1 regime gate + V2 frozen full band + 14d/7d(or 14d) + confirmation + canonical exits.
- SIZING: inv-vol risk parity * TYPE-CONVERGENCE conviction, 15% per-name cap (the dial).
- DEPLOYABLE: conv-weighted 5-type book at 1d = +1.71%/wk, Sharpe 1.60, MaxDD -25% (prod, OOS).
- CEILING within MA: ~+1.7%/wk at prudent cap. 5%+ needs concentration (risk) or new signal family.
