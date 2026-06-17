# IRONED COMBINED -- the COMBINATION / PARTICIPATION layer (2020 deep-dive)

CONSTRUCTION task (the user's CORE thesis, verbatim): *"combine and build these to get max opportunity
participation ... a crossover in one asset might miss, but across 50 it will hit somewhere ... combine
these to get max opportunity participation, and that should solve for ALL 4 aspects if we solve the
weaknesses right."*

Two per-TF IRONED MA trend systems were built+verified on the 2020 deep-dive. This layer COMBINES them
into ONE book and MEASURES whether combination delivers the participation + diversification the thesis
expects. All sleeve streams are RECONSTRUCTED deterministically from the verified builders (`ironed_coarse`,
`deep2020_ironed_fine`) -- they reproduce the source specs bit-exact (1d +49.8%, 4h +40.8%, 15m +67.0%).

- Tool: `src/strat/ironed_combined.py`. RWYB: `python -m strat.ironed_combined`. JSON:
  `runs/periods/TRAIN/2020/DEEP_DIVE/ironed_combined.json`. Cost maker, causal/lag-1, no look-ahead.
- Split (WITHIN-2020): OOS = 2020-10-01..2021-01-01 (the 3mo clean bull tail, ~0% bear at every TF).
- ALL numbers below are VERIFIED (RWYB this run). Capital weights are PRE-REGISTERED (deployment
  constants), NOT fit on OOS -- two rules, both reported: EQUAL-WEIGHT (1/N) and INVERSE-VOL (w~1/sd on
  the full OOS overlap, a level constant matching the core_satellite_book vol convention).

## TL;DR VERDICT (two-sided)

**Combination delivers ROBUSTNESS, not extra return -- because the per-TF trend sleeves are correlated
long-crypto beta (mean pairwise corr ~0.89, n_eff ~1.08 of 3 sleeves = effectively ONE bet). The genuine
multiplier the thesis hopes for comes from the ORTHOGONAL carry satellite, not from stacking timeframes.**

- The cross-TF trend book RAISES Sharpe (best deploy sleeve 3.03 -> book 3.79) and LOWERS maxDD
  (-14.1 -> -11.2) and roughly 10x's the bootstrap p05 (+0.58 -> +7.86) -- genuine variance-reduction even
  at high correlation. But it does NOT beat the best single sleeve on NET (book +52.6% vs 15m-alone +67.0%);
  it trades return for robustness, exactly as the within-TF family ensemble did.
- The funding-dispersion CARRY satellite (corr **-0.16** to the trend core) lifts Sharpe **+1.00**
  (3.79 -> 4.79), improves maxDD **+3.6pp** (-11.2 -> -7.6), and nearly DOUBLES p05 (7.86 -> 15.28) for a
  modest 10% risk-share at a sane 3x leverage cap. **This is the SOTA combined book.**
- "across TFs it hits somewhere" is PARTLY TRUE: the in-market UNION is 100% (>=1 trend sleeve is always
  engaged), but the *profitably*-engaged union is only ~65% -- because the sleeves are correlated, they
  tend to win and lose on the SAME days (the breadth does not buy independent shots).

## THE SLEEVES (recommended per-TF, OOS daily streams -- the inputs)

| sleeve | source spec | OOS net | maxDD(daily) | Sharpe | p05 | daily-pos% |
|--------|-------------|--------:|-------------:|-------:|----:|-----------:|
| 1d_family   | coarse: family-only (DEPLOY)        | +49.8% | -14.1 | 3.03 | +0.58 | 63.0% |
| 4h_ironed   | coarse: full ironed (DEPLOY)        | +40.8% | -13.1 | 3.18 | +0.16 | 62.0% |
| 15m_nogate  | fine: VIDYA nogate_voltgt (best net)| +67.0% |  -6.9 | 4.99 | +18.99 | 58.7% |
| (15m_half)  | fine: half-gate (best DD; alt only) | +62.1% |  -5.3 | 5.37 | +21.03 | -- |
| VOLTGT_BH(1d) | benchmark (the bull's 'best')     | +49.4% | -15.5 | 2.88 | -- | -- |
| BUYHOLD(1d)   | benchmark                         | +47.4% | -20.2 | -- | -- | -- |

NOTE on maxDD basis (reconciliation): combined-book metrics are computed on ALIGNED DAILY net streams (the
only common clock across cadences). Daily-compounding loses intra-day troughs, so a sleeve's daily maxDD is
*smaller* than its bar-level spec maxDD (4h: bar -16.0 = spec, daily -13.1; 15m: bar -8.6 = spec, daily
-6.9). The compound NET is identical to the spec at every TF; only maxDD shifts with the resample basis.
This is the correct basis for combination -- flagged for apples-to-apples.

## MEASURE 1 -- PARTICIPATION / COVERAGE UNION (the "hits somewhere" claim, quantified)

| book | OOS days | UNION in-market | per-sleeve in-market | UNION profitably-engaged | per-sleeve profit-days |
|------|---------:|----------------:|----------------------|-------------------------:|------------------------|
| {1d,4h}       | 92 | **100%** | 1d 100 / 4h 100 | **64.1%** | 1d 63.0 / 4h 62.0 |
| {1d,4h,15m}   | 92 | **100%** | all 100 | **65.2%** | 1d 63.0 / 4h 62.0 / 15m 58.7 |

READ: the IN-MARKET union is 100% -- at least one trend sleeve is always engaged (the family ensembles all
stay broadly long the bull). But the PROFITABLY-engaged union (>=1 sleeve net-positive that day) is only
~65%, barely above each sleeve's own ~62% solo profit-day rate. **Adding the 15m sleeve to {1d,4h} moves the
profit-union only +1.1pp (64.1 -> 65.2%).** The thesis ("a miss in one TF is a hit in another") would predict
the union to climb well above any single sleeve; it does NOT, because the sleeves are correlated -- they tend
to be right and wrong on the SAME days. Breadth across TIMEFRAMES does not buy independent shots the way the
user hopes; the breadth that *does* work is across the 10 ASSETS within each family (7/10 positive per TF,
documented in the source specs), already priced into each sleeve.

## MEASURE 2 -- CROSS-TF DIVERSIFICATION (the correlation matrix)

Cross-TF net-stream Pearson correlation on the OOS overlap (3-sleeve book):

|            | 1d_family | 4h_ironed | 15m_nogate |
|------------|----------:|----------:|-----------:|
| 1d_family  |  1.000    |  0.927    |  0.827     |
| 4h_ironed  |  0.927    |  1.000    |  0.907     |
| 15m_nogate |  0.827    |  0.907    |  1.000     |

- mean pairwise corr **0.887** -> **n_eff 1.08** of 3 sleeves (the {1d,4h} pair: corr 0.927, n_eff 1.04).
- This CONFIRMS multitf_family.json's prior read direction (cross-TF less correlated than within-TF
  configs) but the absolute level is BRUTAL: all three sleeves are ~0.85-0.93 correlated. They are ONE
  long-crypto-beta cluster sampled at three cadences. In a relentless single-factor bull, every long/flat
  trend system tracks the same market move; the cadence only changes the lag, not the factor.

**Does combining RAISE Sharpe + LOWER maxDD (genuine diversification)?** YES -- modestly, via
variance-reduction (averaging correlated-but-not-identical streams cancels idiosyncratic noise):

| book (equal-weight) | OOS net | maxDD | Sharpe | p05 | vs best single sleeve |
|---------------------|--------:|------:|-------:|----:|-----------------------|
| best single (1d, the best DEPLOY sleeve) | +49.8% | -14.1 | 3.03 | +0.58 | -- |
| {1d,4h} book        | +45.5% | -13.6 | 3.15 | +1.30 | Sh +0.12, maxDD +0.5pp, p05 x2.2 |
| {1d,4h,15m} book    | +52.6% | -11.2 | 3.79 | +7.86 | Sh +0.76, maxDD +2.9pp, p05 x13.6 |

(Inverse-vol weighting is near-identical: {1d,4h,15m} IV book +53.1% / maxDD -10.9 / Sh 3.88 / p05 +8.73,
weights 1d 0.27 / 4h 0.35 / 15m 0.38. The two pre-registered rules agree -- the result is not weight-fragile.)

The {1d,4h,15m} book beats the best DEPLOY sleeve (1d) on Sharpe, maxDD AND p05, and roughly matches
VOLTGT_BH net (+52.6 vs +49.4) at lower DD (-11.2 vs -15.5). But it does NOT beat the best sleeve OUTRIGHT
on net: 15m-alone is +67.0%. **The book de-risks; it does not out-return its best component** -- the same
robustness-for-return trade the within-TF family makes.

## MEASURE 3 -- BOOK-LEVEL NET (is the WHOLE > the parts?)

The {1d,4h,15m} equal-weight book OOS: **net +52.6% / ann-INDICATIVE ~+475%/yr (3mo, NOT a promise) /
maxDD -11.2 / Sharpe 3.79 / p05 +7.86 / daily-pos 62%.** vs VOLTGT_BH +49.4% (Sh 2.88) and vs the best
single sleeve 15m +67.0% (Sh 4.99).

- WHOLE > parts on RISK-ADJUSTED terms (Sharpe 3.79 > 1d/4h sleeves; p05 dramatically more robust).
- WHOLE < best-part on RAW NET (52.6% < 67.0% 15m-alone) -- combination is a de-risker, not a return
  amplifier, when the parts are correlated beta. Per the project objective (WEALTH, robust held-out
  compound), the book's value is the ROBUSTNESS (p05 +7.86, maxDD -11.2) bought without sacrificing much
  net, not a higher headline.

## MEASURE 4 -- THE ORTHOGONAL DIVERSIFIER (the genuine multiplier) -- the SOTA combined book

CORE = the {1d,4h,15m} equal-weight trend book (the participating beta core). SATELLITE = the
funding-dispersion dollar-neutral carry (1x-neutral OOS: +9.1% / maxDD -0.5 / Sharpe 8.09 / ann-vol 4.3%,
92 OOS days -- the funding data starts 2020-07-09, so it covers VAL+OOS). Sized per `core_satellite_book`:
gross-leverage cap 3x, capital cap 40%; the deployable split is 70/30 core/satellite.

| book | OOS net | maxDD | Sharpe | p05 | sat risk-share | corr to core |
|------|--------:|------:|-------:|----:|---------------:|-------------:|
| CORE-alone (eq trend book) | +52.6% | -11.2 | 3.79 | +7.86 | -- | -- |
| **CORE 70 / SAT 30 (3x cap)** | +46.3% | **-7.6** | **4.79** | **+15.28** | 10.4% | **-0.164** |

**CORE+SATELLITE BEATS CORE-alone decisively on risk: Sharpe +1.00 (3.79 -> 4.79), maxDD +3.6pp
(-11.2 -> -7.6), p05 nearly DOUBLES (+7.86 -> +15.28) -- all for a 10% risk-share.** The carry leg is
genuinely orthogonal (corr -0.16, vs the +0.83-0.93 among the trend sleeves). Net dips slightly (+52.6 ->
+46.3) because at 70/30 we hold less of the high-return beta core during a bull -- but the DD and tail
robustness improvement is exactly the whole-cycle protection the trend sleeves cannot provide alone.

This is the answer to the thesis: **the multiplier is not "more timeframes of the same trend" (correlated
beta), it is the orthogonal sleeve (market-neutral carry).** The per-TF trend sleeves are the participating
CORE; the carry is the diversifier that makes the combined book robust.

## MEASURE 5 -- "PROFIT DAILY" HONESTY (charter soft-bench, the REAL number)

Rolling-window positivity of the {1d,4h,15m} equal-weight book (OOS): **1d-window +60.9% positive /
3d-window +57.8% positive.** The BUYHOLD floor (the one-factor market itself) is 1d +56.5% / 3d +56.7%.

HONEST: the combined book is positive on ~61% of single days and ~58% of 3-day windows -- only marginally
above just holding the market (~56-57%), because in a bull almost everything is up most days. This is NOT
"daily profit"; it is a positively-skewed beta book in an up-market. Do NOT oversell daily positivity --
the 1-5%/day directional dream is closed (per the project dead-list); the book's edge is robust COMPOUND
return with a hard DD cap + the orthogonal carry tail, not a high daily hit-rate.

## METHODOLOGY / HONEST CAVEATS

- **Sleeve reconstruction is deterministic and bit-exact** to the source specs (compound NET identical at
  every TF). Combination is on ALIGNED DAILY net streams (the only common clock across {1d,4h,15m}); the
  daily-resample basis shifts maxDD vs the bar-level spec (smaller, intra-day troughs lost) -- flagged and
  reconciled above. Net is unaffected.
- **Capital weights are PRE-REGISTERED deployment constants**, NOT fit on OOS: equal-weight (1/N) and
  inverse-vol (w~1/sd on the full OOS overlap, a level not a forward timing signal). Both rules reported;
  they agree (result is not weight-fragile). The satellite leverage cap (3x) + capital cap (40%) + 70/30
  split are static (per core_satellite_book), not fit on the eval span -- no look-ahead introduced by the blend.
- **3mo-BULL-OOS limitation (load-bearing):** the OOS is ~0% bear at every TF. The de-risk/diversification
  VALUE measured here (lower maxDD, higher p05, the carry's DD protection) is partly a WHOLE-CYCLE product:
  the trend sleeves' bear-flat payoff and the carry's market-neutrality both pay MOST in a bear/chop, which
  this clean-bull tail cannot show. The numbers are an honest bull-window lower bound on the
  diversification benefit, asserted-but-not-fully-shown for the full cycle. A bear-inclusive + UNSEEN
  confirm is the next gate before live capital.
- **The carry satellite's decay-risk is UNCONFIRMED** (per the source finding) and its 2020-OOS Sharpe 8.09
  is a small-sample, in-its-best-discovery-era number -- the +1.00 Sharpe lift is real on THIS overlap but
  should be haircut for forward decay. The diversification DIRECTION (corr -0.16, maxDD down, p05 up) is the
  robust finding; the magnitude is optimistic.
- Cost maker (MAKER_RT=0.0006) throughout; the maker p_fill reality (0.21-0.40 per CLAUDE.md) is not
  re-modeled here (inherited from the source sleeves, where the maker->taker gap was already shown small).

## BOTTOM LINE (the honest SOTA combined book)

1. **The participation thesis is HALF-RIGHT.** "Across TFs it hits somewhere" is true for IN-MARKET coverage
   (100% union) but NOT for *profitable* coverage (~65% union, barely above each sleeve's solo ~62%) -- the
   trend sleeves are correlated beta (n_eff ~1.08 of 3), so they win/lose together. Cross-TIMEFRAME breadth
   does not multiply independent shots; cross-ASSET breadth (already inside each family) is the breadth that works.
2. **Combination delivers ROBUSTNESS, not extra return.** The {1d,4h,15m} book lifts Sharpe 3.03 -> 3.79,
   cuts maxDD -14.1 -> -11.2, and ~10x's p05 vs the best DEPLOY sleeve -- genuine variance-reduction even at
   high correlation -- but nets +52.6% vs 15m-alone's +67.0% (de-risk, not amplify).
3. **The genuine multiplier is the ORTHOGONAL carry satellite.** CORE(trend) + SATELLITE(funding-dispersion
   carry, corr -0.16) at 70/30 with a 3x cap lifts Sharpe +1.00 (-> 4.79), improves maxDD +3.6pp (-> -7.6),
   and nearly doubles p05 (-> +15.28) for a 10% risk-share. **This is the SOTA combined book = the per-TF
   ironed trend sleeves as the participating CORE + the orthogonal market-neutral carry as the diversifier.**

Repro: `python -m strat.ironed_combined`; git_sha in ironed_combined.json `repro` block. Does NOT git commit
(overseer commits).
