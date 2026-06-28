# STRAT 2-7%/wk CAMPAIGN (autonomous 4h, start 2026-06-29 00:09:16 SAST -> 04:09:16)
OBJECTIVE: strat-layer changes so 2-7%/wk is realized as reliably as the long-only ceiling allows.
SUCCESS: a config whose DEPLOYED-week distribution beats momentum top-K (positive median, higher %in[2,7], cross-window).
## CYCLE 1 -- mover-SCORE comparison (strat_2to7_movers.py): score is NOT the bottleneck.
VAL (bull): all scores great (mom14 median +7.0, hit>=2 68%, +423%). OOS24: lumpy (high mean, choppy median).
FWD (net-down 2025Q2-2026): ALL scores NEGATIVE deployed-week median (mom14 -2.0, best mom_multi +0.1); hit>=2 ~30-40%.
=> A better SELECTION SCORE does not make 2-7%/wk reliable in down-markets. mom14/relrank/mom_multi best in bulls;
composite/breakout no better. The deployed-week problem in bears = the 7d HOLD captures full down-moves of reversers
(neg median, pos mean = few big weeks + many small losses). PIVOT to EXIT/HOLD ASYMMETRY (cut losers fast, let winners run).

## CYCLE 2 -- EXIT/HOLD ASYMMETRY (strat_2to7_exits.py): TAKE-PROFIT raises the in-band hit-rate (the objective).
TP ~5-7% RAISES in[2,7] across windows (VAL 16->25%, OOS24 0->17%, FWD 10->20-25%) + flips OOS24 median -2.9->+3.1.
STOP-LOSS alone does NOT help (confirms RED-TEAM). TP COSTS compound (caps the >7% tail; OOS24 +92->+30, FWD +8->-12)
= the D23/D61 compound-cost, but for the RELIABILITY objective it is a WIN (more weeks in band). Honest: FWD (net-down)
median stays ~0 even with TP -- the bear ceiling holds; 2-7%/wk is a CONDITIONAL (when-deployed/favorable) reality,
not unconditional. PIVOT cycle 3: deploy-SELECTIVITY (sharper gate) to cut the bear-deploy red weeks + stack with TP.

## CYCLE 3 -- DEPLOY-SELECTIVITY x TP7 (strat_2to7_deploy.py): regime-dependent, NOT robust.
Sharper breadth gate HELPS bull (OOS24 median -0.42->+4.73 hi.7/.5, in[2,7] 25%) but HURTS bear (FWD -0.86->-6.81;
breadth spikes in a bear = false signals). Base .5/.4 is the most robust. => the bull is largely solvable
(mom+TP7+selective deploy = median +4-7%/wk, 25-29% in band); the BEAR is the ceiling -- FWD deploys 30% into red chop.
The lever is a BEAR-DETECTOR (stay cash in the 2025-26 downturn), NOT a sharper breadth gate. Cycle 4: regime composite
(BTC-trend + market-drawdown) to cut FWD deploys to favorable weeks without hurting bulls.

## CYCLE 4 -- BEAR-DETECTOR regime composites (strat_2to7_regime.py): no robust bear/bull-dip separator.
breadth&notDD (don't deploy when EW mkt >12% below 60d high) is least-bad: FWD median -0.86->-0.01, comp -14->-3%,
deploy 30->24%. BUT it HURTS OOS24 (-0.42->-1.40) -- a drawdown looks identical whether it is a bear or a buyable dip.
btc_trend HURTS FWD (deploys more, BTC up early then fell). No detector robustly cuts the bear without hurting a
bull-recovery. Base breadth gate stays most robust. BEAR CEILING confirmed across 4 levers (score/exit/deploy/regime).

## CYCLE 5 -- LEARNED META-LABELER (strat_2to7_metalabeler.py): predictor ~= random ~= momentum.
GBM + logit on {mom14,accel,breakout,volexp,relrank}, train <=2024-05-15, test OOS. Deployed-week dist: FWD mom14
median -0.86/in[2,7] 20% BEATS GBM -2.93 + logit -2.11; OOS24 a wash with random. The learned predictor does NOT beat
simple momentum + barely beats random => the PREDICTABILITY CEILING IS THE MARKET, not the predictor. "Which names run
next 7d" is not extractable from price features beyond momentum. Selection-side surface CLOSED.

## CYCLE 6 -- TP-level robustness + trailing-TP (nail the one real win).

## CYCLE 6 -- TP-level + DEPLOYABLE (strat_2to7_deployable.py): TP5 robust on per-window in-band, but a WEALTH TRADE.
TP5 best per-window (VAL med +5.3/in27 25%, OOS24 +3.1/17%, FWD -0.36/20%). BUT full-OOS deployable (mom top5 + TP5,
2024-05..2026-05): $9,425 (-6% compound), maxDD -39%, deployed-week median +0.39, in[2,7] 12%, neg 50%. TP5 CAPS the
wealth-carrying tail (full-OOS mean +3.07 no-TP -> +1.60 with TP5) -> consistency-vs-wealth TRADE, not a free win;
over a full bull+bear cycle it nets NEGATIVE compound. Live book 2026-05-28 = CASH.

## CYCLE 7 -- CONCENTRATION (K) x TP frontier: wealth<->consistency TRADE-OFF (no free 2-7%/wk).
K = WEALTH dial: top3 noTP positive-compound ALL 3 windows (VAL +879, OOS24 +100, FWD +40) but FRAGILE (3-name
single-name tail, lumpy, negative medians -- the ZEN-mirage caveat). top8 less-lumpy-less-wealth. TP = CONSISTENCY
dial: top8+TP5 in-band(FWD) 25%, FWD median +2.1, but compound capped (-9%). You get lumpy-high-wealth OR
consistent-lower-wealth, NOT both; neither = 2-7% EVERY week (lumpiness is structural).

## FINAL VERDICT (4h autonomous campaign, 7 cycles + learned ML) -- 2-7%/wk is a CONDITIONAL reality, not unconditional.
EXHAUSTIVE strat-layer coverage: score (cy1), exit/TP (cy2,6), deploy-selectivity (cy3), bear-detector (cy4),
learned meta-labeler (cy5), concentration/frontier (cy7). FINDINGS:
1. SCORE is not the bottleneck; a LEARNED predictor (GBM/logit on 5 factors) ~= random ~= momentum -> the
   PREDICTABILITY CEILING IS THE MARKET (which-names-run-next-7d not extractable from price features).
2. BULL regimes DELIVER 2-7%/wk (VAL/OOS24 deployed median +3-7%/wk, 17-27% of weeks in [2,7]).
3. BEAR (net-down) CANNOT (long-only structural) -- no gate/detector/predictor breaks it; correct action = CASH.
4. TWO dials: K=wealth (top3 max but fragile), TP=consistency (raises in-band, caps wealth). A TRADE, not a free win.
5. 2-7%/wk is achievable as a DEPLOYED-FAVORABLE-REGIME reality + a wealth/consistency CHOICE; NOT a reliable
   every-week unconditional outcome. The +12%/wk oracle gap is un-capturable (prediction ceiling).
RECOMMENDED (retail, robust): momentum top-8 (wealth-leaning) OR +TP5 (consistency-leaning); breadth-gated, u100,
1d, 14d/7d. AVOID top3 (fragile single-name). HONEST: this is the ceiling of the strat layer within long-only.
