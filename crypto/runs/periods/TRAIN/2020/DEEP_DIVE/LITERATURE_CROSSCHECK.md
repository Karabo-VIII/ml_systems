# LITERATURE CROSS-CHECK -- the 2020 MA + MR-oscillator findings vs published research (2026-06-13)

User /deep-research: "what are your findings on the MA and the mean-reversion oscillator findings" -> elevated to
a LITERATURE VALIDATION of the four core 2020 findings (the cheapest out-of-sample test available: does the
published quant literature corroborate the MECHANISM behind our in-sample numbers?). Harness: deep-research
workflow (108 agents, 26 sources fetched, 116 claims extracted, 25 adversarially verified via 3-vote ->
18 CONFIRMED / 7 KILLED). All OUR numbers below are [VERIFIED-backtest, IN-SAMPLE 2020 H2]; all EXTERNAL
claims are [adversarially-verified external, mostly EQUITIES/cross-asset -- transfer is by MECHANISM, not direct
crypto proof]. Domain-transfer is the single biggest caveat across the board.

---
## THE CROSS-WALK (our finding -> literature verdict -> what changes)

### Finding 1 -- "MA crossovers are de-risked BETA, not timing alpha; lag buy-hold in a clean bull"
**Verdict: CORROBORATED as an asset-agnostic MECHANISM (HIGH confidence) -- with ONE real counter-example that
sharpens (not refutes) the claim.**
- Faber (2007), S&P since 1901 [REPORTED-Faber, external paper figures]: the 10-month-SMA timing model has a
  VIRTUALLY IDENTICAL average return to buy-hold (11.22% vs 11.26% -- i.e. ~zero return-alpha from timing); the
  compound edge comes ENTIRELY from variance-drain reduction, drawdown 83.7% -> 42.2%. Faber's own words: "a risk-reduction technique," and it
  "can underperform buy and hold during a roaring bull market ... value added is evident only over the course of
  entire business cycles." This is EXACTLY our 2020 read (participation/exit is a return<->DD dial; the family
  rides <half a clean bull and that's the price of avoiding drops that didn't come).
- Zakamulin (2015/2018): bias-free OOS MA timing is "statistically indistinguishable" from buy-hold; the
  outperformance concentrates in BEAR decades (1870s/1930s/2000s); the famous "too good" results are a
  look-ahead artifact. AQR (Hurst-Ooi-Pedersen): trend value is a CONVEX crash-protection payoff, "rather than
  through market timing."
- **SCOPED COUNTER-EXAMPLE (take seriously):** Han-Yang-Zhou (2013, JFQA) [REPORTED-HYZ, external paper figures]
  -- a 10-day MA rule on VOLATILITY-SORTED equity portfolios produces FF3 alphas of 7.5-21.4% that the verifiers UPHELD as genuine
  (the "it's just risk/CAPM-FF3 compensation" rebuttal was KILLED 0-3, and "it's just volatility-decile
  ordering" was KILLED 1-2). So MA timing CAN produce surviving alpha -- specifically in HIGH-VOLATILITY,
  drawdown-prone assets where there is something to time.
- **What changes:** sharpen the claim from "no timing skill, EVER" to **"no timing alpha in a CLEAN BULL (2020 H2)
  -- where there are no drops to avoid, MA timing collapses to de-risked beta."** The HYZ/Faber mechanism
  (drawdown-avoidance flips to absolute-return value over a FULL cycle, especially in high-vol assets) is
  exactly the value proposition of our SURVIVOR strategy (the regime-gated trend book that preserves the bear).
  The literature VALIDATES that survivor's pitch: it is not bull-alpha, it is full-cycle DD-avoidance -- and
  crypto, being higher-vol and more bear-prone than equities, is plausibly CLOSER to the HYZ alpha case than the
  Faber pure-risk-reduction case (this is the #1 open question to test cross-cycle).

### Finding 2 -- "An MA parameter-family does NOT diversify (eff N ~1.2)"
**Verdict: CORROBORATED (HIGH confidence) -- with the LITERAL number softened.**
- Zakamulin proves THEORETICALLY that every MA timing rule (momentum / price-minus-MA / change-of-direction /
  double-crossover, across SMA/EMA/WMA) is the SAME mathematical object -- a weighted moving average of past
  price changes "differing ONLY in the weighting scheme." A family is not a set of distinct bets, by
  construction. CONFIRMED.
- Etienne-Ohana et al. (arXiv 2510.23150, Oct-2025): cross-horizon trend sleeves correlate 0.84-0.94; "excessive
  layering across similar time scales may CONCEAL structural redundancy, creating the ILLUSION of
  diversification"; ablating a redundant horizon IMPROVED Sharpe. ThinkNewfound (2018): effective bets rise only
  ~1.0 -> ~1.2 combining short+long lookbacks. Our eff-N ~1.2 lands right on this.
- Sullivan-Timmermann-White (1999): best-of-~7,846-rules selection is DATA-SNOOPING that overstates performance;
  Zakamulin: "optimal intervals vary widely and frequently" -- no stable optimal lookback. CONFIRMED. (Direct
  warning for our cluster top-K: the #1 config is in-sample-snooped; target the CLUSTER, not the rank.)
- **What changes:** the strict "collapses to exactly 1-2 independent bets / barbell (1/2,0,1/2)" FORMALIZATION
  was KILLED (1-2); ensemble-robustness literature (ReSolve/Newfound) shows parameter ensembles DO add SOME real
  robustness. So the defensible phrasing is **"very few effective bets (~1-2), conceals redundancy"** -- which is
  IDENTICAL to what our Block C already concluded (the family's value is ROBUSTNESS / selection-risk-avoidance,
  NOT alpha or diversification). Fully consistent; just drop any "collapses to one beta" absolutism.

### Finding 3 -- "Trend + MR oscillators are orthogonal (~0.3); combining 50/50 improves Sharpe + halves DD"
**Verdict: MECHANISM CORROBORATED (MEDIUM confidence); the CRYPTO-SPECIFIC magnitudes are UNVALIDATED, and the
one direct crypto test was ADVERSARIALLY KILLED -- this is our weakest-supported finding.**
- Asness-Moskowitz-Pedersen "Value and Momentum Everywhere" (JoF 2013): value & momentum are NEGATIVELY
  correlated (~-0.60 in stocks, ~-0.49 cross-asset), and "a simple combination ... is much closer to the
  efficient frontier than either alone" (global Sharpe ~1.45 [REPORTED-AMP, external paper figure]). This is the canonical published proof of the
  GENERAL mechanism behind our combine-helps result: two positive-EV streams with low/negative mutual
  correlation raise risk-adjusted return. CONFIRMED.
- **THE RED FLAG (honest):** the construct above is CROSS-SECTIONAL value-vs-momentum in equities/futures -- NOT
  time-series TREND (MA) vs OSCILLATOR mean-reversion (RSI/Stoch/BB%b/CCI) in CRYPTO. The ONE direct crypto test
  the sweep found (BTC MIN/MAX paper, SSRN 4081000) claimed exactly our story -- trend+MR combination cuts
  drawdown -- and the adversarial verifiers KILLED both its pro-trend claim (0-3) AND its combination-cuts-DD
  claim (1-2) as unreliable/over-stated. **So the literature does NOT independently confirm our crypto numbers;
  the only crypto evidence on point is weak.** Our own OSCILLATORS.md already flagged this (standalone crypto MR
  is dead-list D37; corr ~0.3 is long-biased not zero; the benefit is TF-dependent + in-sample). The cross-check
  confirms the posture: BELIEVE the orthogonality MECHANISM, do NOT yet believe the magnitude -- cross-year /
  UNSEEN validation is mandatory before the two-family ML target leans on the combine-helps gain.

### Finding 4 -- "Crypto calendar (weekend) structure is real but THIN and may not persist"
**Verdict: CORROBORATED (HIGH confidence for BTC).**
- The BTC MIN/MAX study: "did not find any significant daily seasonality ... the highly performing days are not
  consecutive, so this occurrence is probably just a random coincidence" -- a direct null on robust BTC
  day-of-week. Post-COVID studies (Jul-2020+): "Bitcoin exhibits no discernible calendar anomalies ...
  interpreted as Bitcoin becoming efficient over time"; a 2020-2024 weekend study found "no detectable
  weekend-weekday gap." The decay MECHANISM is STW's own attribution for vanishing equity-rule profits
  (cheaper compute / lower costs / more liquidity) + Lo's Adaptive Markets Hypothesis.
- **What changes:** our call was right -- structure real in-sample 2020 but tradable edge THIN (placebo p=0.13)
  and likely decaying. NUANCE: ETH still shows a Thursday effect in some post-COVID studies, and AMH implies
  decay is CYCLICAL (old anomalies vanish, NEW ones emerge) rather than a one-way ratchet -- so don't treat
  "calendar is dead" as permanent either. Net: do NOT build the book on calendar; keep it as a thin optional
  conditioner at most.

---
## NET IMPACT ON OUR CONCLUSIONS
1. The DRIFT-BETA thesis (Findings 1+2) is now externally CORROBORATED at the mechanism level by the strongest
   names in the trend-following literature -- it is no longer just an in-sample 2020 read. Our survivor
   (regime-gated, bear-preserving trend book) is precisely the Faber/HYZ full-cycle DD-avoidance instrument; the
   literature validates itspitch and tells us where to look for any real alpha: HIGH-VOL, BEAR-PRONE regimes
   (crypto qualifies), tested over a FULL CYCLE -- not a clean bull.
2. The TWO-FAMILY ML target (trend clusters + MR oscillator clusters) keeps its MECHANISM endorsement (AMP 2013),
   but the combine-helps MAGNITUDE is the LEAST-supported piece and the only direct crypto test was killed. The
   ML handoff must therefore (a) target the orthogonality structurally, (b) NOT bank the Sharpe/DD gain until
   cross-year/UNSEEN confirms it, (c) treat the cluster (not the snooped #1 config) as the unit -- per STW.
3. No finding was REFUTED. Finding 1 was sharpened (bull-specific, plus the HYZ alpha caveat), Finding 2 softened
   (drop the literal "1-2 bets"), Finding 3 flagged as mechanism-only, Finding 4 confirmed.

## OPEN QUESTIONS THE CROSS-CHECK SURFACED (highest-EV next tests)
- Q1 [HIGH EV]: does "MA = de-risked beta, no timing alpha" hold in crypto over a FULL CYCLE (incl. a real bear),
  or does crypto's high vol put it in the HAN-YANG-ZHOU alpha case (higher CAGR AND Sharpe vs buy-hold via
  drawdown-avoidance)? This is the one quantitative test that could upgrade the trend book from "beta" to "edge."
- Q2 [HIGH EV]: measure the ACTUAL crypto trend-vs-MR-oscillator correlation + combined Sharpe/DD OUT-OF-SAMPLE
  (cross-year / UNSEEN), since the published ~-0.60 is equities value-vs-momentum and the only crypto test was
  killed. This is the gate before the two-family ML target trusts the combine-helps gain.
- Q3 [MED EV]: after realistic crypto costs + OOS, does best-of-family MA-param selection retain ANY net edge, or
  does the STW data-snooping result reproduce (best in-sample rule fails OOS) in the shorter, more over-mined
  crypto history? -- a direct check on our cluster top-K.

## SOURCES (primary, adversarially-verified)
- Faber 2007 (SSRN id962461); Zakamulin 2015/2018 (SSRN 2677212, 2585056); Sullivan-Timmermann-White 1999
  (SSRN 160330); Han-Yang-Zhou 2013 JFQA (SSRN 1656460); AQR Hurst-Ooi-Pedersen (SSRN 2993026);
  Asness-Moskowitz-Pedersen 2013 JoF (ValMomEverywhere); Etienne-Ohana et al. 2025 (arXiv 2510.23150);
  ThinkNewfound 2018 (process-diversification); BTC MIN/MAX seasonality (SSRN 4081000) [claims KILLED -- cited
  as the weak crypto evidence, NOT as support]. Full list + per-claim votes in the workflow output.

## HONEST CAVEATS
- DOMAIN TRANSFER: nearly all strong evidence is EQUITIES / cross-asset futures, not crypto. It supports our
  claims as MECHANISMS, not as direct crypto results. Our crypto magnitudes rest on our own in-sample 2020
  backtests and are NOT externally validated.
- The combine-helps (Finding 3) crypto magnitude is the single least-supported number in the whole deep-dive.
- Calendar-decay is time-sensitive (AMH: new anomalies can emerge); the arXiv redundancy paper is very recent
  (Oct-2025), not yet widely replicated.
- This cross-check is a LITERATURE test, not a new backtest -- it changes our CONFIDENCE in the mechanisms, not
  the numbers. The numbers still need cross-year/UNSEEN validation (Q1/Q2/Q3).
