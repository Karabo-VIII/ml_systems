# PHASE 4b -- THE TRUE-COMPLEMENT SEARCH (what actually fills the trend book's gaps across regimes)

**Tool:** `src/strat/complementary_sleeve_search.py` -- **JSON:** `complementary_sleeve_search.json` --
**Charts:** `charts/{complement_return_corr_by_regime, complement_combined_by_regime, longonly_vs_short_value}.png`
**Repro:** `python -m strat.complementary_sleeve_search --seeds 20 --cadences 1d,30m` (git_sha at run: `72e828f`)
**Test surface:** PHASE 3's VALIDATED synthetic generator (`synthetic_regime_stress`, git_sha `1756867`),
calibrated on **REAL 2020-band data ONLY** (generator validation: **3/3 regimes match real-2020** stylized
facts). No 2026/other data was read. Maker cost, lag-1 causal, no MtM double-count. 20 seeds; distributions
(mean +- spread + worst seed) reported.

---

## THE QUESTION (PHASE 3 set it up)

PHASE 3 proved the long-only MR sleeve is a **BEAR LIABILITY**: on the validated synthetic bear (the 2020
COVID-crash exemplar) the trend+MR static blend's maxDD was **-11.3pp WORSE** than trend-alone, because BOTH
sleeves are long, so MR cannot rescue trend's down days -- it ADDS losses (buys falling knives). A long-only
complement therefore **cannot fill a bear gap**. The user's CORE vision -- "if one is out, the other captures
the gap" -- demands a sleeve with genuinely **anti-correlated RETURNS** (WINS when trend loses), not merely
anti-correlated *engagement* (engages when trend is flat). This module searches for that sleeve.

**Candidates (vs the deployable trend MA book), all run through the EXACT deployable mechanics** (MA-cross
signal -> trail-stop -> min_hold(12) -> lag-1 -> maker):
`SHORT_MA` (short on MA-cross-DOWN) | `LONGSHORT_MA` (symmetric) | `CASH_GATE` (past-only bear -> cash) |
`VOLTGT_DEF` (vol-target defensive overlay) | `MR_LONG` (the long-only bear-liability baseline).

`SHORT_MA` and `LONGSHORT_MA` **violate the standing long-only + spot constraint -> they are RESEARCH; a
DEPLOY needs the user's explicit long-only-exception sign-off.** They are explored here for the LEARNING
(scope was expanded for max learnings) and to QUANTIFY the value of relaxing the constraint.

---

## THE KEY METRIC -- corr(candidate, trend) on RETURNS, per regime  [CLAIM: synthetic, 20 seeds, mean +- sd]

A true complement is **RETURN-anticorrelated** (corr < 0): it wins when trend loses. (Chart 1.)

| candidate        | bull  | bear  | chop  | stitched | long-only? | verdict |
|------------------|-------|-------|-------|----------|-----------|---------|
| **SHORT_MA**     | -0.53 | **-0.44** | -0.43 | -0.36 | NO (research) | **RETURN-anticorrelated in EVERY regime** |
| LONGSHORT_MA     | +0.69 | +0.07 | +0.93 | +0.74 | NO (research) | net-correlated (it embeds the long book) |
| CASH_GATE        | +1.00 | +0.97 | +1.00 | +1.00 | yes | ~scaled trend -- canNOT win when trend loses |
| VOLTGT_DEF       | +0.95 | +0.90 | +0.98 | +0.95 | yes | ~scaled trend -- canNOT win when trend loses |
| MR_LONG          | +0.49 | +0.50 | +0.46 | +0.31 | yes | positively correlated (the bear liability) |

**[FINDING, two-sided]** Only `SHORT_MA` is RETURN-anticorrelated to trend across all four regimes
(bear corr **-0.44 +- 0.15**). The long-only "defensive" gates (`CASH_GATE`, `VOLTGT_DEF`) have corr ~ +1.0
-- they are mechanically just a **scaled-down trend book**, so by construction they cannot post a positive
return on a day trend is negative; they can only DAMPEN the bleed, never reverse it. This is the PHASE 3
lesson made precise: **engagement-anticorrelation is not enough; RETURN-anticorrelation requires SHORT
exposure** (or a genuinely independent positive-carry source, which none of these long-only sleeves is).

---

## THE BEAR GAP-FILL -- does the candidate post a POSITIVE return where trend bleeds?  [CLAIM: synthetic, bear]

Trend-alone BEAR net = **-6.9%** (mean; worst seed -12.1%). This is the gap to fill. (Chart 3, right panel.)

| candidate    | bear net (mean) | bear net worst-seed | positive-in-bear seed-frac | genuine fill? |
|--------------|-----------------|---------------------|----------------------------|---------------|
| **SHORT_MA** | **+13.2%**      | -17.6%              | **85%**                    | **YES (positive in bear)** |
| LONGSHORT_MA | +2.7%           | -11.5%              | 70%                        | YES (positive in bear) |
| CASH_GATE    | -7.2%           | -11.6%              | 0%                         | no (cash = 0 best case; here ~flat-to-trend) |
| VOLTGT_DEF   | -4.4%           | -9.0%               | 5%                         | no (dampens the bleed only) |
| MR_LONG      | -22.5%          | -56.8%              | 15%                        | no -- **ADDS to the bleed (the liability)** |

**[FINDING]** Only the SHORT-bearing sleeves post a POSITIVE bear return. `SHORT_MA` fills the gap in 85% of
seeds (+13.2% mean). **[HONEST CAVEAT]** it is NOT free: its bear worst-seed is **-17.6%** -- a short in a
bear that V-bounces (the 2020 crash exemplar bounced hard) can lose. The long-only gates do exactly what
their corr says: dampen, never reverse. `MR_LONG` is re-confirmed the liability (-22.5%, worst -56.8%).

---

## THE COMBINED trend+candidate BOOK across the FULL regime mix  [CLAIM: synthetic, 20 seeds]

Combined maxDD MINUS trend-alone maxDD (POSITIVE = the combined book draws down LESS than trend-alone). The
DECISIVE cross-regime test -- does it dampen DD everywhere, esp. the BEAR, not just chop? (Chart 2.)

| candidate    | bull  | bear  | chop  | stitched | mix mean | true full-mix complement? |
|--------------|-------|-------|-------|----------|----------|---------------------------|
| **SHORT_MA** | +2.6  | **+3.9** | +1.9  | +4.3  | **+3.19pp** | **YES** (bear-anticorr AND mix-DD-dampening) |
| LONGSHORT_MA | +1.6  | +3.2  | +1.1  | +2.8  | +2.17pp  | DD yes, but bear-corr +0.07 (not anticorr) |
| VOLTGT_DEF   | +1.0  | +1.5  | +0.4  | +2.4  | +1.35pp  | dampens, but bear-corr +0.90 (scaled trend) |
| CASH_GATE    | +0.0  | -0.1  | +0.0  | -0.1  | -0.06pp  | ~no effect (gate rarely fires at its threshold) |
| MR_LONG      | -0.3  | **-11.3** | +1.3  | -5.6  | -3.97pp  | NO -- worsens DD in bear + stitched (liability) |

**[DECISIVE FINDING]** `SHORT_MA` is the **only candidate that is RETURN-anticorrelated in the bear AND
lowers combined DD across the full mix** -- the single "true complement" by the two-sided test. Among
long-only candidates: `VOLTGT_DEF` is the best *within-constraint* option (dampens DD in every regime, mix
+1.35pp) but it CANNOT win when trend loses (it is scaled trend); `CASH_GATE` at its pre-registered
threshold barely fires (mix ~0); `MR_LONG` actively worsens the bear.

---

## QUANTIFIED -- the value of relaxing LONG-ONLY (trend+SHORT/long-short vs trend+MR long-only)

Mean over bull/bear/chop/stitched. (Chart 3, left panel.) **[CLAIM: synthetic; short economics EXCLUDE
borrow/funding -> these are an UPPER bound on the short advantage.]**

| comparison                          | mix NET adv | mix DD adv | **BEAR NET adv** | **BEAR DD adv** |
|-------------------------------------|-------------|------------|------------------|-----------------|
| trend+SHORT_MA   vs trend+MR(LO)    | **-3.3pp**  | +7.2pp     | **+18.1pp**      | **+15.2pp**     |
| trend+LONGSHORT_MA vs trend+MR(LO)  | **+0.5pp**  | +6.1pp     | +13.2pp          | +14.4pp         |

**[ANSWER to "is long-only what's blocking true complementarity?"]** YES, decisively in the regime that
matters. Replacing the long-only MR sleeve with a SHORT-bearing complement buys **+15.2pp of BEAR drawdown
and +18.1pp of BEAR net** -- i.e. it turns the bear from a combined-book liability into a near-flat outcome
(trend+SHORT bear comb DD -4.39% vs trend+MR -19.6%). **[HONEST, two-sided]** the cost: pure `SHORT_MA`
DRAGS the full-mix net (-3.3pp vs trend+MR) because it is directionally wrong in the bull/chop (it shorts a
rising market). `LONGSHORT_MA` is the **more balanced full-cycle relaxation** -- it keeps the long book in
the up-regimes yet adds the short defense, netting a slightly POSITIVE mix net (+0.5pp) AND +14.4pp bear DD.

---

## DECISIVE VERDICT (two-sided, honest)

**[HEADLINE]** **LONG-ONLY IS THE BINDING CONSTRAINT ON TRUE COMPLEMENTARITY.** The only sleeve that
genuinely complements the trend book across ALL regimes -- RETURN-anticorrelated in the bear (wins when
trend loses) AND mix-DD-dampening -- is a **SHORT/inverse-trend** sleeve (`SHORT_MA`; `LONGSHORT_MA` is the
balanced full-cycle variant). **No long-only candidate can truly fill the bear gap**: the defensive gates
(`CASH_GATE`, `VOLTGT_DEF`) only DAMPEN the bleed (they are scaled trend, corr ~ +1.0); long-only MR ADDS to
it. This directly confirms the PHASE 3 thesis: the long-only constraint, not the choice of indicator, is
what prevents "when one is out, the other captures the gap."

**WITHIN the current long-only + spot constraint:** the best you can do is a **defensive overlay**
(`VOLTGT_DEF`-style vol-targeting on the trend book) that dampens combined DD by ~+1.4pp across the mix and
~+1.5pp in the bear -- a risk-reducer, NOT a return-rescue. It is deployable today; it does not need a
sign-off. `CASH_GATE` as specified (10-day trailing mean < -2%/day) barely fires on the V-crash bear and
adds ~nothing; a more sensitive bear detector is an open thread, but it can never exceed the cash-floor (0
return) it provides -- a long-only gate's ceiling is "stop losing," never "start winning."

**TO TRULY FILL THE BEAR GAP requires SHORT exposure** -- which is RESEARCH, gated behind the user's
explicit long-only-exception sign-off. The learning value: we now have a quantified price tag on the
constraint (**~+15pp of bear drawdown protection forgone** by staying long-only), and a concrete deployable
candidate (`LONGSHORT_MA`, the balanced variant) ready for that decision if/when the user takes it.

### Recommendation framing (NOT a deploy -- a decision input)
- **Deploy-eligible now (long-only):** add a `VOLTGT_DEF` defensive overlay to the trend book. Modest,
  honest DD-dampening; preserves the long-only + spot constraint; no sign-off needed.
- **Decision for the user (needs LO-exception sign-off):** a `LONGSHORT_MA` complement is the genuine
  cross-regime gap-filler. It is the ONLY way the synthetic evidence shows to make the bear a non-liability.
  Quantified upside: ~+14pp bear DD / ~+13pp bear net vs the long-only book, mix-net-neutral.

---

## CAVEATS (binding)

1. **SYNTHETIC test surface** from PHASE 3's VALIDATED generator, calibrated to **2020 stylized facts ONLY**
   -- a stress surface, not real future data. A generator can only reproduce the facts it was calibrated on.
2. **SHORT_MA / LONGSHORT_MA VIOLATE the standing long-only + spot constraint** -> RESEARCH only. Deploying
   either needs the user's explicit long-only-exception sign-off. The defensive long-only gates do not.
3. **Short economics are OPTIMISTIC**: the inverse-trend short assumes the same maker fill economics as the
   long book. A real short carries **borrow / funding costs NOT modeled here** -> the Q5 short advantage is
   an **UPPER bound**. (Perp funding in a sustained bear is often favorable to shorts, but this is unmodeled
   and must be verified before any deploy.)
4. **The synthetic bear is the 2020-COVID V-crash exemplar** -- a fast crash that bounces hard. A slow
   grind-down bear would likely favor the SHORT sleeve MORE (less bounce risk) -- so the SHORT worst-seed
   (-17.6%) is partly a V-bounce artifact of this specific bear shape.
5. **Cadence note (SWEEP rule):** 1d and 30m were both run; their results are **identical** because the
   generator produces DAILY synthetic panels (the cadence selects deployable-sleeve windowing, but the
   underlying synthetic price path is the same daily series). So 30m does NOT add independent cross-cadence
   evidence here -- it is the same daily surface viewed twice. Reported honestly rather than as 2-cadence
   corroboration.
6. **20 seeds; distributions reported** (mean +- spread + WORST seed); no seed cherry-picked. The selftest
   (`--selftest`) is a two-sided sleeve-direction soundness gate (PASS): SHORT wins in bear / loses in bull /
   manufactures no edge in a no-trend null; MR_LONG is re-confirmed the bear liability.
