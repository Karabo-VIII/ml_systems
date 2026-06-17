# 2020 DEEP DIVE -- information gathering (6h autonomous, 2026-06-13)

User /orc: stay in 2020; "we are not optimally participating in the bull"; "lots to learn statistically/
mathematically"; "one config can't win them all -> a family per timeframe/instrument might win"; AND
(added) "include EXIT mechanisms + GRADING -- no point if we give up all our returns." Window 2020 H2
(VAL Jul-Sep + OOS Oct-Dec). FULL stack, equal-weight u10 book, maker. UNSEEN untouched. This doc
accumulates the blocks. Tools: `src/strat/deep2020_*.py`; data in this folder.

---
## BLOCK A -- PARTICIPATION: how much of the bull do we actually ride? (`deep2020_participation.py`)
Per-bar log-return decomposition by (position x sign): UPSIDE-CAPTURED (long in up-bars) / UPSIDE-MISSED
(flat in up-bars = the gap) / DOWNSIDE-AVOIDED (flat in down-bars = value-add) / DOWNSIDE-GIVEN-BACK (long
in down-bars). Two rates: upside-capture = captured/total_up; downside-avoidance = avoided/total_down.

**THE FINDING: we ride well under half the bull.** Family-avg upside-capture is ~0.44 at the best
cadences (4h/2h/1h) and as low as 0.09 (VIDYA@1d). We are FLAT 55-80% of the time -- the MA cross trades
*missed upside* for *downside-avoidance* (dodges 55-90% of drops). In a CLEAN BULL the drops are small, so
that trade usually LOSES (capt/BH < 1 for most configs): you give up more bull (missed) than drawdown you
save (avoided).
- **capt/BH by cadence** (best MA): 1d 0.10-1.16 / 4h 0.90-1.25 / 2h 0.70-1.20 / 1h 0.61-1.10 / 30m
  0.26-0.56 / 15m -0.08-0.56. The SWEET SPOT to actually *realize* capture is 4h-1h.
- **Two failure modes of participation:** (1) at COARSE, the conservative MAs (VIDYA@1d time-in 0.09)
  sit OUT the bull (under-participate); (2) at FINE, upside-capture is fine (~0.44) but COST destroys the
  net (15m HMA/TEMA capt/BH < 0) -- the cost drag, not the signal.
- The lever to raise participation: be IN MORE (higher time-in-market) AND cheaper (coarser / fewer
  trades). The best configs at 4h ride ~45% of the bull at time-in ~0.45 and beat BH (capt/BH 1.0-1.25).

---
## BLOCK B -- EXIT MECHANISMS: do we KEEP the returns or give them back? (`deep2020_exits.py`)
Fixed EMA entry; swept 11 exits; the downside-given-back term is what the exit controls.
**THE FINDING (counterintuitive, bull-specific): the exit is a RETURN<->DRAWDOWN dial, and TIGHTER is
WORSE for returns.** The LOOSEST exits (signal-flip / min-hold) keep the MOST; every tighter stop cuts net:
| 1d exit | up-capture | net% | maxDD% | OOS net% |
|---|---|---|---|---|
| flip / minhold12 (loosest) | 0.63 | **78-80** | -28 | **47-48** |
| trail20 | 0.40 | 58 | -22 | 31 |
| trail15 | 0.31 | 44 | -17 | 21 |
| take-profit 25% | 0.18 | 28 | -10 | 19 |
| trail5 (tightest) | 0.11 | **19** | -7 | 14 |
| chandelier ATR | 0.13 | 16 | -9 | 12 |
- In a bull, a tight stop/trail/TP **whipsaws you out** (time-in collapses to 0.09) and the bull continues
  WITHOUT you -- trail5 cuts net 78->19%. **The way you "give up your returns" in a bull is by EXITING
  TOO TIGHTLY, not by holding.** The given-back (eating pullbacks while long) is the *price* of riding the
  bull, and it is worth paying.
- The exit cleanly buys drawdown reduction: flip -28% DD for 78% net; trail5 -7% DD for 19% net. Choose
  by how much DD you'll trade for return.
- GRADING: OOS block-bootstrap p05 is NEGATIVE for every exit (none "robust" by that bar); tighter exits
  have less-negative p05 (lower tail) at large return cost.
- **CAVEAT (the key nuance):** this is BULL-specific. In a bear/choppy-top the loose exit gives back
  TERMINALLY (the given-back is not recovered). So the best exit is **continuation-dependent** -- ride
  (loose) when the move continues, cut (tight) when it reverses. That is the timing problem again; there
  is no fixed best exit.

---
## BLOCK C -- the FAMILY-OF-STRATS thesis: does a family beat the best single? (`deep2020_family.py`)
**THE FINDING: a family of MA configs = the AVERAGE config, NOT a winner -- because the configs barely
diversify.** Family OOS ~= avg-single OOS (1d 19.2~=19.3; 4h 27.9~=27.8; 1h 53.0~=53.5), and does NOT beat
the best-single (which, VAL/OOS being adjacent same-bull, transferred here).
| TF | mean config corr | diversification ratio | **effective N** (of 39 configs) |
|---|---|---|---|
| 1d | 0.81 | 1.08 | **1.2** |
| 4h | 0.92 | 1.04 | **1.1** |
| 1h | 0.94 | 1.03 | **1.1** |
- **39 configs collapse to ~1.2 EFFECTIVE independent bets** (corr ~0.8-0.94 -- all slow-MA crosses, long
  at the same times). Adding configs of the same archetype does NOT diversify.
- So the user's thesis is HALF-right: "one config can't win them all" is true (the best cell-winner varies),
  but the FAMILY does not WIN -- it gives the AVERAGE reliably. **Its real value is ROBUSTNESS: it avoids
  selection risk (the VAL-fluke-collapse, e.g. LINK) -- you get the mean without gambling on picking the
  best (or the worst).** It buys insurance, not alpha.
- **To actually diversify (raise effective N) you need UNCORRELATED strats -- a different archetype/beta,
  not more MA configs.** (Consistent with the earlier ensemble work: same-beta can't diversify.) In crypto
  2020 nearly everything is ~0.8-0.95 correlated, so a within-archetype family stays ~1-beta.

---
## RUNNING SYNTHESIS (what 2020 is teaching, so far)
1. We ride <half the bull (capture ~0.44) because we're flat most of the time -- the participation gap is
   real and is a TIME-IN-MARKET problem, not a signal problem.
2. In a bull, EXITS should be LOOSE (ride it); tight exits whipsaw you out and forfeit the bull -- "giving
   up returns" = over-exiting. The exit is a return<->DD dial; the right setting is continuation-dependent.
3. A family of MA configs is the ROBUST choice (= the average, avoids selection risk) but NOT a winner; it
   barely diversifies (eff N ~1.2). Real diversification needs an orthogonal beta, not more configs.
[Blocks D (statistical/math structure: move/run distributions, trade stats, trend persistence) + E
(participation ceiling + grading synthesis) to follow.]

---
## BLOCK D -- the STATISTICAL / MATHEMATICAL structure of 2020 (`deep2020_stats.py`)
**THE BIG ONE: 2020 H2 is a positive-DRIFT random walk, NOT a momentum trend.**
- **Hurst ~0.39 (1d) / 0.52 (4h) / 0.47 (1h); lag-1 autocorrelation SLIGHTLY NEGATIVE (-0.02 to -0.07).**
  There is NO per-bar momentum -- bar returns are ~random-walk-to-mildly-mean-reverting. The "trend" lives
  ENTIRELY in the positive DRIFT (mean return), not in autocorrelation.
- Implication (ties the whole project together): this is WHY per-bar IC ~= 0 (the dead-list lens) yet
  trend-following WORKS -- the edge is **staying long in a drifting market**, not timing. There is no
  per-bar timing edge to discover (ac1 ~ 0). **Participation (time-in-market) is the ONLY lever; the drift
  is the alpha.**
- **The MA cross CATCHES the big bars.** Big-bar capture (avg family position during the top-decile up-bars)
  = 0.22/0.40/0.47 at 1d/4h/1h vs overall avg position 0.21/0.34/0.44 -- i.e. the family is in-position
  MORE during the biggest up-bars (they occur mid-trend, when the cross is already long). So the
  participation gap is NOT missing the big moves -- it is **LATENESS** (flat during the small/early bars at
  move STARTS, before the cross confirms) + flatness in chop.
- **Concentration:** the top 5% / 10% of up-bars carry ~0.19-0.22 / ~0.32-0.35 of the total up-return
  (moderately concentrated; XRP most, top10% 0.44 = the Nov-2020 pump).
- **Trade math (EMA family):** win rate 0.62-0.72, payoff (avg-win/avg-loss) 4-7x, expectancy +3.8% (1h)
  to +20% (1d) per trade, median hold ~17-19 bars. High win rate AND payoff>1 -- the bull makes both good.

## SYNTHESIS UPDATE (the coherent picture)
The pieces fit: 2020's edge is DRIFT (not momentum), so capturing it is a TIME-IN-MARKET problem
(Block A: we ride <half because we're flat 55-80%); the cross catches the big bars but enters LATE
(Block D); to capture more drift you must stay IN, so EXITS should be LOOSE (Block B: tight exits forfeit
the continuing bull); and a family of same-archetype configs doesn't diversify the single drift-beta
(Block C: eff N ~1.2). NET: in a drift-bull the optimal play is MAXIMIZE participation in the drift while
causally dodging only the worst drawdowns -- which is closer to "stay long" than to "trade signals."

---
## BLOCK E -- OPTIMAL PARTICIPATION: can we beat buy-hold by timing the drift? (`deep2020_optimal.py`)
| TF | strategy | net% | maxDD% | capt/BH | time-in | OOSnet% |
|---|---|---|---|---|---|---|
| 1d | BUYHOLD | 101.9 | -42.7 | 1.00 | 1.00 | 55.8 |
| 1d | MA_FAMILY | 26.2 | -12.3 | 0.26 | 0.21 | 18.9 |
| 1d | LONGBIAS_dd20 | 71.1 | -44.6 | 0.70 | 0.81 | 45.0 |
| 1d | LONGBIAS_dd30 | 83.0 | -45.7 | 0.81 | 0.94 | 49.1 |
| 4h | MA_FAMILY | 62.4 | -22.7 | 0.62 | 0.34 | 32.0 |
| 4h | LONGBIAS_dd30 | 84.4 | -48.2 | 0.84 | 0.96 | 54.4 |

**THE ANSWER: in a drift-bull, BUY-HOLD is optimal participation -- you cannot beat it by timing.**
- The MA cross UNDER-participates badly (capt/BH 0.26 at 1d to 0.65 at 2h) because it is flat 56-79%.
- The long-biased disaster-stop participates much more (capt/BH 0.70-0.88, time-in 0.81-0.97) but STILL
  does not beat buy-hold on return AND does not reduce maxDD (-42 to -49% ~= buy-hold's -43 to -46%) --
  because crypto drawdowns are sharp-V: the -20%/-30% stop triggers AFTER the drop, then whipsaws on
  re-entry, eating the drawdown without dodging it.
- So every stepping-aside strategy is a DE-RISKED BUY-HOLD: less return, and in 2020 not even less DD. The
  MA cross's lower maxDD (-12% at 1d) is purely LOWER EXPOSURE (it's out 79% of the time), not skill/alpha.

## GRAND SYNTHESIS -- what 2020 taught us (the coherent, mathematically-grounded story)
1. **2020 is a positive-DRIFT random walk** (Hurst ~0.45, lag-1 autocorr < 0). No per-bar momentum -> no
   per-bar timing edge to find (explains IC~0). The ONLY alpha is the DRIFT.
2. **The drift is captured by TIME-IN-MARKET**, nothing else. The MA cross rides <half the bull because it
   is flat 55-80% of the time (a participation, not a signal, gap). It catches the BIG bars (mid-trend) but
   enters LATE, forfeiting the small/early bars of each move.
3. **You cannot beat buy-hold by timing the drift.** Buy-hold = optimal participation. Every long-only
   timing strategy (MA cross, disaster-stop) is a DE-RISKED buy-hold: it trades bull-upside for (intended)
   bear-protection. In 2020 that trade LOSES (you give up drift; the DD-dodge fails on sharp-V drops).
4. **EXITS confirm it:** looser = more return (ride the drift); tight stops whipsaw you out and forfeit the
   continuation. "Giving up returns" in a bull = OVER-EXITING.
5. **A family of MA configs does NOT win or diversify** (eff N ~1.2; corr 0.8-0.94) -- it is the AVERAGE
   config, valuable only as insurance against picking a VAL-fluke. Real diversification needs an orthogonal
   beta, not more configs.
6. **THE REGIME-FREE TRUTH (answers the user's framing):** a long-only MA strat is not a way to "beat" a
   bull -- it is a way to take LESS EXPOSURE to it. Its entire value proposition is conditional: it
   underperforms buy-hold in a bull (the cost) in exchange for dodging a bear (the benefit, untested here).
   So the right question is never "which MA/exit/config wins the bull" (buy-hold wins) but "how much
   bull-upside am I willing to give up for bear-protection, and can the strat actually deliver that
   protection when it matters" -- which is the 2022 test, and the real open question.

GRADING NOTE: across A-E, OOS block-bootstrap p05 is negative for the timing strategies -- none is a
"robust positive book" by the canonical bar; buy-hold has the highest return but the deepest drawdown.
The deep-dive's value is the MECHANISM (drift, participation, de-risked-buy-hold), not a shippable edge.

---
## BLOCK F -- IS THERE ANY TIMING SKILL? (permutation test) (`deep2020_timingalpha.py`)
Test: does the MA family go long during BETTER-than-average bars? ret_per_exposure vs buyhold_per_bar, with
a NULL that shuffles positions (same time-in, timing destroyed), 300x -> permutation p.
| TF | mean timing-alpha (bp) | instruments p<0.05 |
|---|---|---|
| 1d | +5.1 | **0 / 8** |
| 4h | +1.8 | **2 / 10** (XRP p=.01, DOGE p=.03) |
| 1h | +0.3 | **1 / 10** (XRP p=.027) |
**THE MA CROSS HAS ~NO TIMING SKILL (rigorously).** For 8-9 of 10 instruments its long-bars are NO better
than randomly-positioned bars of the same exposure (p > 0.3). It is long during AVERAGE bars, just less
often = de-risked buy-hold, CONFIRMED. The ONLY significant timing skill is on **XRP/DOGE** -- the most
SERIALLY-CONCENTRATED assets (the Nov-2020 pumps), where trend-alignment during the explosive run adds
real value. **Conclusion: trend-following timing pays ONLY where the move is concentrated/serially-
correlated (a sustained pump); in a diffuse positive-drift random walk (most assets), there is no edge
over buy-hold.** This is the rigorous statement of the whole deep-dive: 2020's bull is mostly drift (no
serial structure -> no timing edge), with rare concentrated pumps (the only place timing helps).

---
## BLOCK G -- the LAW: concentration predicts timing skill (`deep2020_conc_law.py`)
Joining Block F (timing-alpha, p) with Block D (concentration, hurst, bigcap) across 28 (instrument,TF) cells:
- timing-alpha ~ concentration(top10%): **r=+0.53**; p-value ~ concentration: **r=-0.56**; p-value ~
  big-bar-capture: **r=-0.70** (strongest). Higher concentration / big-bar-capture / hurst -> more timing skill.
- The 4 timing-skill cells (p<0.10) are ALL XRP(1d/4h/1h) + DOGE(4h) -- the concentrated, trending pumps;
  they have higher conc10 (0.43 vs 0.33), hurst (0.54 vs 0.47), bigcap (0.43 vs 0.38) than the no-skill cells.
**THE LAW: trend-following timing skill is proportional to an asset's serial CONCENTRATION / trend-
persistence.** Where moves are concentrated+persistent (pumps), trend-following adds real value; on diffuse
positive-drift assets (most), there is no timing edge -> buy-hold wins. ACTIONABLE + falsifiable: route
trend-following to pumpy/trending assets, hold the diffuse ones. (Caveat: n=28, 2020-only, 4 sig cells ->
suggestive not proven.) Chart: charts/concentration_law.png.

---
## BLOCK H -- CROSS-SECTIONAL structure: the edge is WHICH assets, not WHEN (`deep2020_xsection.py`)
Orthogonal axis: rank u10 by trailing-week return, long top-3 (XS_MOM) / bottom-3 (XS_REV) / all (EW).
| TF | strategy | net% | maxDD% | Sharpe | OOSnet% |
|---|---|---|---|---|---|
| 1d | EW | 121.9 | -30.3 | 2.48 | 47.4 |
| 1d | **XS_MOM** | **316.6** | -28.5 | **3.71** | **114.7** |
| 1d | XS_REV | 33.9 | -36.5 | 1.10 | 3.3 |
| 4h | XS_MOM | 316.8 | -29.5 | 3.74 | 100.0 |
**THE PIVOT: in 2020 the alpha is CROSS-SECTIONAL (which assets), not TIME-SERIES (when).** XS momentum
(chase recent winners) returns +317% vs +122% equal-weight (+195pp), higher Sharpe, lower maxDD, 2x OOS.
XS_REV (losers) is terrible -> it is MOMENTUM not reversal; winners keep winning. The bull had huge
DISPERSION (3.1%/bar at 1d) and asset-selection captures it where market-timing cannot. Ties the whole
deep-dive together: time-series timing has NO edge (drift random walk, Blocks A-G), but the
cross-sectional dispersion / concentrated pumps (Block G's XRP/DOGE) ARE capturable by holding the right
assets.
**HARD CAVEATS (do not over-claim):** (1) the DEAD-LIST refuted XS momentum OOS/UNSEEN -- this in-sample
2020-H2 edge likely does NOT transfer (the adjacent same-regime OOS here is not UNSEEN); (2) COST-NAIVE
(daily top-3 rebalance = high turnover -> real edge shrinks); (3) CONCENTRATED in 1-2 pump assets. So this
is a DESCRIPTIVE finding about 2020's STRUCTURE (the edge lives in cross-sectional dispersion), NOT a
deployable edge -- but it is the right place to look, and it reframes the strategy question from "time the
MA" to "select the asset."

---
## BLOCK H2 -- cross-sectional rigor: CORRECTING the caveats (`deep2020_xsection_rigor.py`)
Checked the 3 caveats I flagged on Block H. Two are LARGELY REFUTED:
- **COST: survives.** 1d XS_MOM net free 317% -> maker 307% -> TAKER 278% (vs EW 122%). Trailing-week
  ranking is stable -> low turnover -> NOT a cost artifact. (4h: free 317 -> taker 230, also survives.)
- **CONCENTRATION: broad, NOT 1-2 assets.** Holdings spread across the majors (BTC 18% / ETH 16% / LINK
  14% / LTC 12% / ADA 12% of asset-bars). Leave-one-out (drop the top-held BTC) -> 289% vs 307% -- barely
  moves. The edge is NOT carried by one pump asset.
- **ROBUSTNESS: holds across top-K{1,2,3,5} x lookback{3,7,14}** (mostly +; only 4h/top1/lb3 is fragile).
**CORRECTED VERDICT: the 2020 cross-sectional momentum edge is COST-ROBUST and BROAD in-sample -- not a
fragile artifact.** The ONE surviving caveat is the dead-list's UNSEEN/regime-transfer failure: it is a
BULL-momentum effect, and the within-2020 OOS (Oct-Dec) is adjacent same-regime, so the transfer question
is open (and prior work says it likely reverses/dies in a bear). So the honest status: a REAL, robust
structural feature of the 2020 bull (the alpha is cross-sectional dispersion); the deployability question
is purely regime-transfer, which is the bull-specificity, not a cost/concentration weakness. This is the
single most promising thread of the whole deep-dive.

---
## BLOCK I -- WHY XS momentum works (the mechanism, honestly) (`deep2020_xs_mechanism.py`)
- **Rank persistence is WEAK + NOT significant:** Spearman(trailing-rank -> forward-rank) = +0.057 (1d,
  t=0.98) / -0.005 (4h, t=-0.08). Winners only WEAKLY persist (winner-minus-loser next-window +3.4%/+2.4%,
  positive in 60%/54% of rebalances -- better than coin-flip but weak).
- **So the +278% XS_MOM is NOT reliable momentum skill** -- it is a WEAK (60% hit) persistence x LARGE
  dispersion (3.25%/bar) x compounding over 60 rebalances, which mechanistically is mostly **drift x a
  BULL-BETA tilt** (holding the trailing winners = holding the high-beta names, which outrun in a bull).
- **This is exactly WHY the dead-list refuted XS momentum OOS/UNSEEN:** it is a bull-beta bet that reverses
  in a bear, not robust cross-sectional alpha. The big in-sample number is the bull amplifying a beta tilt.

## GRAND CONCLUSION OF THE DEEP DIVE (unified + honest)
Everything reduces to ONE thing: **in 2020 the only edge is the positive DRIFT, and every "strategy" is
just a way to TILT EXPOSURE to it -- none is robust alpha.**
- TIME-SERIES timing (MA cross, exits) = tilts exposure DOWN (de-risked buy-hold) -> gives up drift; no
  timing skill (drift random walk). Buy-hold is the participation optimum.
- CROSS-SECTIONAL selection (XS momentum) = tilts exposure toward HIGH-BETA winners -> amplifies drift in
  the bull; weak true persistence (t~1) -> a bull-beta bet, not alpha (reverses in a bear; dead-list).
- A FAMILY of configs = the average tilt (eff N ~1.2); doesn't diversify one beta.
The bull makes everything more-exposed look good. There is no robust timing OR selection alpha in 2020 on
internal price data -- only the drift (beta), captured best by holding it. This is the honest, mechanistic
restatement of the project-wide internal-data ceiling, now PROVEN from six independent angles within the
single cleanest (bull) year. The real alpha question is unchanged: an ORTHOGONAL source (regime-transfer
test / external data / a non-beta signal), not another way to tilt the same drift-beta.

---
## BLOCK J -- the best way to HOLD the drift: vol-targeting beats flat buy-hold (`deep2020_voltarget.py`)
| TF | strategy | net% | maxDD% | Sharpe |
|---|---|---|---|---|
| 1d | BUYHOLD | 117.6 | -32.0 | 2.41 |
| 1d | **VOLTGT_hi** | **120.9** | **-24.8** | **2.97** |
| 4h | BUYHOLD | 115.4 | -34.4 | 2.50 |
| 4h | **VOLTGT_hi** | **123.4** | **-26.3** | **3.29** |
**THE ONE CLEAN IMPROVEMENT ON BUY-HOLD -- and it is RISK-SIZING, not alpha.** Vol-targeted holding (scale
long-only exposure inversely to recent realized vol, capped [0,1]) keeps the return (+3 to +8pp) while
lifting Sharpe (+0.56/+0.79) and cutting maxDD (+7-8pp). Fits the theory exactly: you cannot beat the drift
with TIMING or SELECTION alpha, but you CAN improve the RISK-ADJUSTED capture of it by SIZING -- and
vol-targeting is a risk IDENTITY (lower exposure in high-vol -> steadier Sharpe), not a return prediction,
so it tends to TRANSFER where momentum does not. ACTIONABLE: the best causal participation in a drift-bull
is **vol-targeted buy-hold**. (Caveat: maxDD cut is modest because crypto crashes are sharp -- vol spikes
WITH the crash, so the de-risk is slightly late; still a genuine, likely-transferable improvement.)

---
## BLOCK K -- does a family ACROSS TIMEFRAMES diversify? (completes the family thesis) (`deep2020_multitf.py`)
Cross-TF EMA-family book correlation (OOS daily): mean 0.82 (1d<->15m lowest at 0.68; adjacent TFs 0.88-0.91)
vs within-TF config corr ~0.90 (Block C). Multi-TF family Sharpe 3.20 vs avg single-TF 2.98 (lift +0.22 from
noise-averaging) but **effective N across TFs STILL = 1.2**. **COMPLETE family-thesis answer: NO form of
family -- across configs, MA types, timeframes, OR instruments -- meaningfully diversifies in crypto 2020.
Everything is ~0.82-0.90 correlated (ONE drift-beta) -> eff N ~1.2 for every grouping.** A family buys
robustness (avoid selection risk) + a tiny Sharpe lift, NEVER diversification. You cannot diversify a single
beta with more variants of the same long-only trend bet -- only an ORTHOGONAL beta would (which the
internal-data ceiling says we do not have). This definitively closes the user's "family per TF/instrument
might win" thesis: it does not win or diversify; it averages.

---
## BLOCK L -- the BEST causal 2020 book (constructive capstone) (`deep2020_bestbook.py`)
| 1d book | net% | maxDD% | Sharpe | OOSnet% |
|---|---|---|---|---|
| BUYHOLD | 121.9 | -30.3 | 2.48 | 47.4 |
| VOLTGT_BH | 118.1 | -26.5 | 2.89 | 50.1 |
| XS_MOM | 316.6 | -28.5 | 3.72 | 114.7 |
| **XS_MOM x VOLTGT** | 277.1 | **-23.6** | **4.40** | 106.9 |
Combining XS-momentum SELECTION + vol-target SIZING = the best in-sample causal book (Sharpe 4.40 vs 2.48
buy-hold, maxDD -24% vs -30%). HONEST DECOMPOSITION: the XS component drives the RETURN (the bull-beta tilt
that will NOT transfer -- weak persistence, dead-list), the vol-target component drives the SHARPE LIFT (the
part that likely DOES transfer, being a risk identity). So "best causal book, Sharpe 4.4" = mostly bull-luck
(XS bull-beta) + a real risk-sizing improvement (vol-target). The transferable kernel is vol-targeted holding;
the headline return is bull-beta.

---
## BLOCK M -- CALENDAR STRUCTURE: the first ORTHOGONAL-to-beta signal (`deep2020_seasonality.py`)
The drift is NOT calendar-uniform. Pooled u10, 2020 H2:
- **DAY-OF-WEEK:** Saturday +182bp/day (t=4.97), Sunday +75bp (t=2.69) -- strong WEEKEND drift (vs overall
  46bp/day). Tue/Wed mildly negative (not significant).
- **HOUR-OF-DAY (UTC):** 13h (US morning) +19.5bp (t=5.26), 0-1h positive (+9-10bp, t~2.7-2.9), 2-4h (Asia
  overnight) NEGATIVE (-8 to -14bp, t=-2.7 to -4.2), 15h +10.8 (t=3.4). 12/24 hours significant.
**This is the ONLY signal in the whole deep-dive that is NOT a drift-beta tilt** -- calendar/time-of-day is
ORTHOGONAL to price-trend beta, exactly the kind of source the internal-data ceiling says we need. CAVEATS:
in-sample 2020-H2 only; multiple comparisons (31 tests -> ~1.5 expected false-positives, but 14 significant
>> chance and Sat/13h t-stats are strong -> the structure is REAL, not all noise); calendar effects are
notoriously NON-PERSISTENT (decay/arbitrage). So: a genuinely NEW orthogonal lead, persistence/transfer
UNTESTED. THE most promising thread for "orthogonal alpha" -- worth a proper OOS/UNSEEN persistence test
(the one thing in the deep-dive that could be real non-beta structure).

## DEEP-DIVE COMPLETE (13 analytical blocks A-M + dashboard)
The 2020 bull is a positive-DRIFT random walk; the only price-edge is the beta, best held vol-targeted;
timing/selection/family are all exposure tilts (no robust alpha); the ONE orthogonal-to-beta structure is
CALENDAR (weekend/US-hours), real in-sample but persistence-untested. Everything UNSEEN-sealed.

---
## BLOCK N -- calendar PERSISTENCE + tradability (the lead holds up) (`deep2020_calendar_test.py`)
- **PERSISTENCE: 7/7 days agree in sign across 2020-H2 halves (Jul-Sep vs Oct-Dec); cross-half DOW-profile
  corr = +0.80.** Saturday +183bp then +182bp (nearly identical); Tue/Wed negative in both. A STABLE, REAL
  effect within 2020 -- NOT noise.
- **TRADABILITY: high-drift windows have ~3x the return-per-time-in of average** (weekend 305% vs BH 110%
  per time-in; good-hours 246% vs 182%). A naive long-only-weekend gives LOWER absolute return (flat 72% of
  the time) -> the real use is a calendar-WEIGHTED TILT, not a standalone filter.
**This is the genuine headline of the deep-dive: a REAL, within-2020-PERSISTENT, ORTHOGONAL-to-beta signal**
(the drift concentrates in weekends + US-hours). The honest caveat: calendar effects DECAY across years (the
crypto weekend effect weakened post-2020 as institutions entered) -> cross-year/UNSEEN persistence is the
real litmus (untested here per "stay in 2020"; prior evidence suggests decay). Within 2020 it is rock-solid
and is the most promising orthogonal lead -- the one thing found that is not just a drift-beta tilt.

---
## BLOCK O -- the CAUSAL calendar tilt BEATS buy-hold (the breakthrough) (`deep2020_calendar_tilt.py`)
Classified weekdays GOOD/BAD on the FIRST half (Jul-Sep): only TUESDAY was negative. Applied causally to the
SECOND half (Oct-Dec):
| OOS strategy | net% | maxDD% | Sharpe |
|---|---|---|---|
| BUYHOLD | 47.4 | -20.2 | 2.34 |
| **CAL (flat on Tuesdays)** | **60.1** | **-17.9** | **2.99** |
**Sitting out the one historically-negative weekday -- classified on VAL, applied causally to OOS -- BEATS
buy-hold on EVERY axis: +12.7pp net, lower maxDD, Sharpe 2.99 vs 2.34.** This is the FIRST thing in the whole
deep-dive that beats buy-hold via a SIGNAL (not just risk-sizing). It REFINES the grand conclusion: PRICE
timing has no edge (drift random walk), but ORTHOGONAL CALENDAR timing DOES -- the drift is genuinely not
calendar-uniform (Tuesday reliably lower). CAVEATS: within-2020 only (cross-year is the real litmus; crypto
calendar effects are known to decay post-2020); thin 1-day signal (multiple-comparisons exposed). But the
causal within-2020 result is clean + positive.

## UPDATED GRAND CONCLUSION (the refined headline)
On PRICE data, 2020 is a drift-beta with no robust timing/selection alpha -- everything is an exposure tilt,
best held vol-targeted. BUT there exists an ORTHOGONAL-to-price signal -- the CALENDAR -- that is (a) real,
(b) within-2020-PERSISTENT (7/7 days, corr 0.80), and (c) CAUSALLY beats buy-hold (sit out the bad weekday:
+12.7pp net, Sharpe 2.34->2.99). This is the deep-dive's most valuable discovery and it points the right
way: the alpha beyond beta is ORTHOGONAL (calendar / and by extension microstructure, flow, external) --
NOT another price-trend variant. The open litmus is cross-year/UNSEEN persistence (calendar effects decay),
which is the natural next test (outside 2020).

---
## BLOCK Q -- calendar ROBUSTNESS: real STRUCTURE, but the tradable edge is thin (honest correction)
- **BREADTH (strong):** Saturday positive in **10/10** assets (BTC +113, ETH +164, ADA +272, LINK +284,
  LTC +247 bp/day); Tuesday negative in 7/10. The calendar STRUCTURE is BROAD + real, NOT 1-asset.
- **PLACEBO (tempers tradability):** "sit out Tuesday" OOS net 63.3% beats BH 47.4% and the average
  random-1-day exclusion (39.7%), BUT permutation **p = 0.13** -- NOT significant. Excluding one specific
  day is thin + multiple-comparison-exposed; ~13% of random single-day exclusions do as well.
**CORRECTED VERDICT (tempering Block O's "breakthrough"):** the weekend/day-of-week STRUCTURE is genuine,
broad (10/10 Saturday), and persistent (7/7 across halves) -- a REAL orthogonal-to-price structural feature.
But a ROBUST TRADABLE edge from it is NOT cleanly established under long-only/lev=1: the single-day tilt
fails the placebo (p=0.13), and you cannot OVER-weight Saturdays at lev=1 (cap 1.0) so the tradable
expression is limited to thin under-weighting. Same lesson as the XS finding: the STRUCTURE is real, the
deployable EDGE needs more than 2020+lev=1 to express. The honest status: the calendar is the most
promising ORTHOGONAL STRUCTURE found (real+broad+persistent); turning it into robust alpha needs (a) a
cross-year persistence test and (b) a setting that can over-weight the good windows (leverage / tilt in a
larger book) -- NOT demonstrated here.
