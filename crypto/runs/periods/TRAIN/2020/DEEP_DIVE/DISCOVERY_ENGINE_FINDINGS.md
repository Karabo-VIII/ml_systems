# FINER-TF MA STRATEGY-DISCOVERY ENGINE -- consolidated findings (2026-06-14, 5h autonomous)

> User mandate: "trade finer timeframes <=1d, research thoroughly across ALL MA types, find a strat that
> works... build ENGINES for discovering strategies that are profitable, complementary, dynamic... if one
> is out one day the other is on and captures the missing gaps, ideally per timeframe... 2020 band only,
> synthetic data where needed, charts/figures." Built as 6 composable, honest, RWYB engine stages.

## The engine (6 composable stages -- the reusable "discovery engine" the user asked for)
| # | Stage | Tool | Answers |
|---|-------|------|---------|
| 1 | MA-type x TF research | `ma_type_tf_research.py` | which MA type/TF is the best trend sleeve |
| 2 | Complementarity matrix | `deep2020_complementarity.py` | is trend+MR orthogonal; does combining help |
| 3 | Dynamic/ML allocation | `dynamic_allocation_engine.py` | does regime-timing the blend beat a static blend |
| 4 | Synthetic regime-stress | `synthetic_regime_stress.py` | does it all survive bull/bear/chop (daily) |
| 4b| Intraday-resolution stress | `synthetic_intraday_stress.py` | does dynamic skill appear at TRUE sub-daily res |
| 5 | Complementary-sleeve search | `complementary_sleeve_search.py` | what sleeve TRULY fills a bear gap |

All: 2020 band only (TRAIN 6mo/VAL 3mo/OOS 3mo), u10, maker, causal/lag-1, held-out, two-sided, charted
(14 PNGs in `charts/`). Synthetic generator calibrated on 2020 data ONLY + VALIDATED before trusting.

## The convergent verdict (what the engine discovered)
**1. ADAPTIVE MA types win every finer TF.** VIDYA wins {4h,2h,1h,30m,15m}, KAMA wins 1d; the adaptive edge
over low-lag/simple WIDENS at finer cadence. The best trend sleeve is adaptive -- but it is participating
BETA (net < VOLTGT_BH in the 2020 bull), valuable for risk-adjusted return + as a complement component.

**2. COMPLEMENTARITY (static) is REAL but regime-conditional.** Trend (MA) and MR (oscillator) are
orthogonal at every TF (corr +0.21..+0.31 vs 0.85-0.94 within-trend). Combining DD-dampens (1d Sharpe
2.67->2.89, maxDD halved). BUT the honest catch (PHASE 1b -> confirmed PHASE 3): gap-filling is
DRAWDOWN-DAMPENING, not positive-return rescue -- both sleeves are long-only, so on a down day neither can
WIN, they only bleed less. And in a BEAR the long-only MR is a LIABILITY (it buys falling knives, +11.3pp
worse combined DD). Complementarity helps in CHOP, hurts in BEAR.

**3. DYNAMIC timing has NO skill -- robust to resolution.** A regime-conditional trend-vs-MR allocator beat
the static blend at only 1 of 6 TFs (30m) on real 2020 data -- at the multiple-comparisons chance rate.
On the VALIDATED synthetic generator (20 seeds, proper paired sign test) the 30m "edge" is significant ONLY
in a pure synthetic bull and REVERSES under a regime flip = an exposure-tilt level effect, not timing skill.
The named falsifier (a TRUE sub-daily generator, ~48 bars/day = far more timing opportunities) was built +
validated 3/3 and the verdict SURVIVED: more bars did not manufacture timeable structure. SHIP THE STATIC
BLEND; the dynamic ML layer is not worth its complexity.

**4. TRUE complementarity (filling a BEAR gap) REQUIRES a SHORT sleeve -- the long-only constraint is the
binding limit, now QUANTIFIED.** Only a SHORT/inverse-trend sleeve is RETURN-anticorrelated to trend (bear
corr -0.44; +13.2% bear net where trend bleeds -6.9%). Long-only "defensive" gates are corr ~+1.0 (scaled
trend -- dampen, never rescue). Swapping the long-only MR for a short-bearing complement buys +15.2pp bear
DD protection + +18.1pp bear net; LONGSHORT_MA is net-neutral full-cycle (+14.4pp bear protection).
Within long-only+spot, the best is a VOLTGT_DEF defensive overlay (deployable now, ~+1.4pp DD dampening,
no sign-off) -- a risk-reducer, not a gap-filler.

## What this means for the user's "profitable, complementary, dynamic" goal
- **Profitable (finer TF):** yes, as participating beta (adaptive-MA trend sleeve) -- but it is beta, not
  alpha (net < buy-hold in a bull; the value is risk-adjusted + complementarity). No internal-data alpha,
  consistent with the whole project arc.
- **Complementary:** YES (static trend+MR) for CHOP drawdown-dampening; but a long-only complement can
  only dampen, never fill a BEAR gap -- TRUE cross-regime complementarity needs a SHORT sleeve (the
  LO-exception, now quantified at +15-18pp bear value).
- **Dynamic:** NO -- regime-timing the blend has no skill, robust across resolution + regime + a proper
  sign test. The honest answer is a STATIC blend (+ defensive overlay), not a dynamic ML timer.

## The honest deployable recommendation (2020-band evidence, finer TF)
An ADAPTIVE-MA (VIDYA/KAMA) trend sleeve per TF + a trail-stop (most robust across regimes) + a static MR
complement for chop DD-dampening + a VOLTGT_DEF defensive overlay -- all long-only, participating beta with
risk control. The single highest-value UNLOCK is the long-only-exception: a LONGSHORT_MA sleeve is the one
thing that turns the bear from a liability into a near-flat (the +15-18pp), and it is the user's strategic call.

## PHASE 6-7 extensions -- the bear-rescue, built out + bounded (commits 2cd1329, 4770126)
The PHASE-4b "only a SHORT fills a bear gap" finding was built into a real engine + stress-tested two more ways:
- **PHASE 6 -- the LONGSHORT-MA engine** (`longshort_ma_engine.py`): symmetric long-short adaptive-MA, FIXED
  short trail-stop, modelled short-borrow. SOUND -- net-positive full-cycle (+5.9..+11.8% stitched), beats
  its cost-matched null by +4.8..+10pp; short-borrow is NEGLIGIBLE (~0.02pp). BUT it is regime-conditional
  BEAR-INSURANCE, best at a COARSE TF -- on REAL 2020 the always-on bull-drag GROWS with finer cadence
  (-8.2pp@1d -> -40.8pp@15m; the daily synthetic understates this). The static 4-sleeve book does NOT beat
  trend-alone on worst-regime net (a naive equal-risk mix dilutes the regime-winner + MR poisons the bear);
  adding longshort still improves worst-DD by +5..8.8pp -> it is a valuable COMPONENT, but "profit in every
  regime" needs ROUTING, not a static mix.
- **PHASE 7 -- the regime-GATED longshort** (`regime_gated_longshort.py`): deploy the short insurance ONLY in
  a detected sustained bear. Sharp two-sided result: **the bear DETECTOR genuinely WORKS** (frozen
  {close<SMA20, arm5, disarm3}: precision 0.36 vs base-rate 0.12 ~3x random, beats the equal-frequency
  SHUFFLE in 0.90 of seeds -- a real positive, unlike the continuous dynamic null) -- **but the gated BOOK
  still does NOT beat trend-alone** on 2020 (loses 0.3-2.0pp, no DD reduction). VERIFIED mechanism: the
  conditional bear gain is tiny (+0.11pp -- the 2020 bear is ~12% of the cycle + mild) while the gate's
  residual false-alarms in bull/chop (each shorting a rising market) + borrow cost ~-1.6pp. On a
  bull-dominated short cycle a precise gate's small false-alarm rate costs more than the short bear earns.

## THE REFINED DEPLOYABLE ANSWER (2020-band evidence, finer TF)
SHIP-TODAY (long-only, no sign-off needed): an ADAPTIVE-MA (VIDYA/KAMA) trend sleeve + a trail-stop (the
single most robust thing across regimes) + a static MR complement for CHOP drawdown-dampening + a VOLTGT_DEF
defensive overlay -- participating beta with risk control. This is a risk-managed beta book, NOT held-out alpha
and NOT a dynamic timer (proven to add nothing).
THE ONE UNLOCK (user's strategic call): the long-only-exception. A SHORT sleeve is the only thing that posts a
POSITIVE return when trend loses a bear (+15-18pp bear value); always-on it pays a bull-drag, and gating it on
a bear-detector does NOT pay off on the SHORT/MILD 2020 bear -- BUT the detector works and the book is built +
validated to run on a DEEPER/LONGER bear (e.g. 2022) on request, where the bear is a larger share of the cycle.
That 2022 test is the single highest-value next experiment (outside the 2020 band, so flagged for the user).

## The engine's real asset: it KILLS its own false positives
Every "edge" this run surfaced was adversarially falsified by the engine itself: the 30m dynamic candidate
(reversed under regime flip), a 3-seed "67% beats" (collapsed to n.s. at 20 seeds with a proper sign test),
the fine-TF MR magnitudes (flagged overfit), the bull-only DEPLOY illusions. The discipline -- validate the
generator first, proper sign test, RWYB, two-sided, multiple-comparisons aware -- is the durable deliverable,
the trading analogue of the chess engine's monotonic promotion gate. 2 look-ahead/mechanics bugs were caught
+ fixed mid-build (short trail-stop sign bug; regime same-bar lag).

Commits: 72e828f (stages 1-4) + 8f962ab (stage 5) + the intraday stage. Charts: `charts/`. All 2020-band.
