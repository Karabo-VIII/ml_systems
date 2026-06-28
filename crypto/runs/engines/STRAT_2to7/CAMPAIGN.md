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

## CYCLE 8-9 -- ENRICHED predictor + VERIFICATION (strat_2to7_enriched/verify.py): signal is TEMPORAL, not cross-sectional.
Cy8: enriched meta-labeler (OHLC-shape + MARKET-STATE + lead-lag) OOS AUC 0.667, 2.62x top-decile lift -- looked like
a breakthrough (cy5's ~random was a FEATURE failure). Cy9 VERIFICATION (RWYB): WITHIN-BAR (cross-sectional) AUC only
0.548 (null 0.51) -> the 0.667 was MOSTLY TEMPORAL (market-state distinguishes good/bad WEEKS, not which-NAME). Book
translation MIXED (enriched-P beats mom14 on FWD median +1.30 vs -0.36 but loses OOS24) -> does NOT robustly beat
momentum. WHICH-NAME ceiling largely HOLDS (weak 0.548); the real signal is WHEN-to-deploy (timing). Near-false-victory
caught by within-bar decomposition + book translation.

## CYCLE 10-11 -- LEARNED DEPLOY-TIMING gate + VERIFICATION: real but FRAGILE (promising-unconfirmed).
cy10: breadth&P(good-week)>0.55 -> FWD deployed median +6.8/comp+109, cross-window + cross-threshold (the temporal
signal cy9 pointed at). cy11 ADVERSARIAL: beats shuffle-null (+8.5 vs null-p90 +5.0) AND beats best rule gate
(+8.5 vs +2.8) -> signal REAL. BUT fragile: 6 FWD deployed weeks, top-2 = 61% of gain, ONE-SIDED (H1 +9.1 / H2 -4.4);
+ dead-list D85 (learned regime-timing refuted) + campaign-stale OOS. => PROMISING-UNCONFIRMED, needs PRISTINE FORWARD
data to settle (fragility is data-limited, not tuning-fixable).

## FINAL VERDICT (4h autonomous, 11 cycles + 2 learned ML + adversarial verify) -- HONEST end-state.
1. WHICH-NAME selection: weak ceiling (within-bar AUC 0.548 vs 0.51 null); momentum ~optimal; learned predictor does
   NOT robustly beat it -> the market caps cross-sectional name-prediction.
2. WHEN-TO-DEPLOY (timing): the real predictable signal is TEMPORAL (market-state: btc-mom/breadth/disp). A learned
   timing gate beats null+rule gates but is FRAGILE/one-sided/data-limited + dead-list-prior -> PROVISIONAL candidate.
3. TAKE-PROFIT ~5%: raises in-[2,7] consistency, caps the wealth tail -> a consistency<->wealth TRADE.
4. CONCENTRATION K: the wealth dial (top3 max but fragile single-name; top8 robust).
5. BULL delivers 2-7%/wk (deployed median +3-7%); BEAR is the structural ceiling (long-only) -- the learned-timing
   lead HINTS partial bear-avoidance but is unconfirmed.
ANSWER to "make 2-7%/wk a reality": it is a CONDITIONAL (favorable-regime) reality + a wealth/consistency CHOICE,
NOT a reliable every-week unconditional outcome. ONE promising lead (learned deploy-timing) could improve bear-
avoidance -> validate on fresh/forward data before trust. Deployable now = momentum top-8 (wealth) / +TP5
(consistency), breadth-gated, u100, 1d, 14d/7d. NO false victory: the campaign found a lead, not a solved 2-7%/wk.

## CYCLE 12-13 -- WALK-FORWARD learned deploy-timing HOLDS + shuffle-null-VALIDATED (the campaign's real win).
cy12 WALK-FORWARD (quarterly retrain, never peek): WF-timing gate (breadth & P(good-week)>0.55) vs breadth-rule, full
OOS 2024-05..2026-05, $10k: WF+mom8 (wealth) $28,085 (+181%) vs rule +57%, maxDD -27 vs -36, deploy 14%; WF+mom5+TP5
(consist) +101% (vs rule -6%), deployed-week MEDIAN +5.10 (IN the 2-7% band), deploy 13%. cy13 VERIFY: shuffle-null
(20x) -> WF +236% vs null mean +2%/p90 +36% => BEATS null DECISIVELY (signal REAL, not luck). Per-quarter: positive
5/9, deployed 6/9; AVOIDS 2025Q1 crash (+7.8 vs rule -15.8) + 2026Q2 (+3.0 vs -6.8). CAVEATS: 2024Q4 dominates
(+137 of +236); FEATURE-DESIGN overfit (market-state feats chosen on these windows = campaign-stale); contradicts
dead-list D85; small deployed sample (~13%). => STRONG validated LEAD; needs FRESH/LIVE forward data for full trust.

## FINAL CAMPAIGN VERDICT (4h, 13 cycles + 3 learned-ML + walk-forward + shuffle-null).
THE WIN: a LEARNED DEPLOY-TIMING gate (market-state: btc-mom/breadth/dispersion/trend/drawdown) is the real lever --
NOT which-name selection (within-bar AUC 0.548 = market-capped). It deploys ~13% of weeks (cash 87%) into the genuinely-
favorable weeks where deployed-week MEDIAN = +5%/wk (in the 2-7% band), ~3-4x's wealth (+57->+181-236% OOS) at lower
drawdown, walk-forward + shuffle-null validated. So "2-7%/wk a reality" = the DEPLOYED weeks reliably hit it (+5% median)
while the engine sits in cash the rest -- a SELECTIVE engine, not an every-week one. DEPLOYABLE: momentum top-8 (wealth)
or +TP5 (consistency), breadth & learned-timing gate, u100, 1d, 14d/7d. PROVISIONAL (feature-overfit + D85 + small
sample) -> validate live/forward. Live gate 2026-05-28 = CASH (P=0.34). NO false victory: validated as far as the data
allows; fresh data is the final gate.
