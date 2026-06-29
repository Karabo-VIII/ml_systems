# STRAT TIERS -- multi-strategy coverage + ranker (user /orc 2026-06-29)
OBJECTIVE: confirm single-TI numbers, then wire a TIERED, always-scanning, RANKED multi-strategy system
(static single-TI + dynamic TI + hyper-selective V5) so we are "in the market every week looking", with a
ranker picking which strategy is active. User hypothesis: single-TI 1-3%/wk, dynamic 2-5%, hyper-selective 2-7%.

## CONFIRM single-TI numbers (confirm_single_ti.py) -- HYPOTHESIS NOT CONFIRMED.
@1d EW long-only u100 breadth-gated PROD cost, OOS 2024-05..2026-05: deployed/wk Bollinger +2.29 (comp +25%),
MACD +1.35 (+1%), ADX +0.98 (-16%), Supertrend +0.71 (-6%), RSI +0.53 (-15%), Stoch +0.50 (-33%). BLENDED/wk ~0
(all). deploy-frac ~31% (NOT non-selective). => single-TI engines are NOT 1-3%/wk + NOT non-selective; most are
net-NEGATIVE over OOS; only BOLLINGER is solidly positive. Tiering is INVERTED from the hypothesis: hyper-selective
V5 (+5-8%/wk deployed) STRONGEST; Bollinger moderate; rest weak. => wire only the CONFIRMED engines + a recent-edge
ranker + the validated WF deploy-timing gate; do NOT wire weak engines as a fake 1-3% floor.

## RANKER (strat_tiers_ranker.py) -- multi-engine rotation does NOT beat the single best engine.
OOS weekly: momentum8 +57% (best single) > ma_conv +36% > RANKER (recent-edge rotation) +33% > equal_blend +27% >
bollinger +19% > macd -3%. The recent-edge ranker UNDERPERFORMS momentum8 (rotates into recently-good engines that
then fade -- same refutation as config-rotation D73/D83). equal-blend dilutes too. Active usage: momentum8 74%.
ROOT CAUSE: the engines are CORRELATED (all long-only trend, same breadth gate, ~19% weeks-positive each) -> they
deploy in the SAME weeks -> blending/rotating averages toward the weaker ones; no de-correlation = no coverage benefit.

## VERDICT -- the tiered/ranked multi-strategy vision does NOT add value; the single best engine + WF-timing wins.
1. Single-TI engines CONFIRMED weak (NOT 1-3%/wk; only Bollinger positive +25%; rest breakeven/negative OOS).
2. Recent-edge RANKER + equal-blend UNDERPERFORM the best single engine (momentum8) -- rotation dilutive, engines correlated.
3. The validated WF-timing gate + momentum8 (V5, prior campaign, +154-236% OOS) BEATS all of this -- the timing gate
   (WHEN to deploy) is the lever, not multi-engine selection (WHICH strategy).
DEPLOYABLE = V5 (WF-timing + momentum8), always-scan weekly (in market looking every week; CASH when unfavorable,
deploy the book when favorable). "Coverage" via multiple tiers is refuted (correlated engines). The only untested
de-correlation path = a long-only mean-reversion engine for CHOP weeks (deploys when momentum is cash) -- new build, uncertain.

## MEAN-REVERSION COVERAGE TEST (strat_tiers_mr.py) -- "in market every week" is COUNTERPRODUCTIVE.
MR (buy-oversold, z<-1) DE-correlates from momentum (corr +0.17) + fills 92% of momentum's CASH weeks (coverage
WORKS mechanically, deploys 87% of weeks). BUT MR = -85% compound (falling knives: long-only oversold -> more
oversold in down/chop). COMBINED momentum-else-MR = -61% (94% coverage) vs momentum-ALONE +57%. => filling the cash
weeks DESTROYS return. The cash weeks are cash for a REASON (gate flags unfavorable); deploying anything in them loses.
LESSON: "in the market every week" is the WRONG goal -- SELECTIVITY (cash ~70%) IS the edge. Distinction: "LOOKING
every week" (scanning -- good, built) vs "DEPLOYED every week" (refuted, harmful). Scan all weeks; deploy only the
favorable ~13-31%. FINAL deployable = V5 (WF-timing + momentum), selective, cash-when-unfavorable.
