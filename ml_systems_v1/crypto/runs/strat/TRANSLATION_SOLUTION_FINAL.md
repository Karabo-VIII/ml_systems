# Translation solution -- the definitive end-to-end answer (6h /quant build, 2026-06-16)

User /quant: "I need results to translate into 2021 -- find me a solution," then "work end-to-end
autonomously, report findings at end of solution. 6h." This is that report. Every number is
[VERIFIED-RWYB, held-out + deflated]; STRICT long-only + spot throughout; sealed UNSEEN touched once.
Commits 262f718 (diagnosis) -> 82c4e22 -> 6a4c1ff -> 97c7104 -> 4e0c34e -> 39435d4.

## The question, restated precisely
Can a strategy SELECTED on 2020 produce WEALTH that translates forward (2021 bull, 2022 bear, ... ,
sealed UNSEEN)? The forward-test had shown config NET-RANK does not translate (Spearman 0.11).

## The arc (each phase a pre-registered, held-out, deflated test)
1. **DIAGNOSIS (262f718):** config-selection translation = ARTIFACT (a family-class conflation; the
   planted null beat it too). What translates is the FAMILY granularity (trend/breakout/momentum/MA)
   + drawdown-preservation. Structural de-risk selection is REFUTED for net (it picks stalling MR).
2. **THE BOOK (82c4e22):** built the family-ensemble drawdown-preserving beta book. It PRESERVES the
   2022 bear (book -19.8% maxDD vs EW-BH -73.4% = 53.6pp) and OUT-compounds BH full-cycle 2020-2022
   (+24.7% vs +9.7%) by LOSING LESS in the bear. Consistent every year 2020-2025 (~10-30% bull capture,
   25-75% less drawdown). REAL_WITH_CAVEAT: it's insurance, captures only ~11% of the 2021 bull.
3. **SEALED UNSEEN + robustness (6a4c1ff):** the preserve-signature HOLDS out-of-sample (book -2.3%
   maxDD vs BH -40.2%) -- BUT ship=FALSE (held-out p05 -38.8%, OOS compound -19.4%). It earns NO
   positive held-out return; it only loses less by sitting in cash.
4. **THE MECHANISM (97c7104):** a family-FREE vol-gate on always-on beta does NOT reproduce the
   preservation (2022 -66.2% vs book -19.8%) -- vol-target is symmetric to VOLATILITY, not DRAWDOWN.
   Preservation = the trend SIGNAL going to CASH (~70% of the time) in downtrends. A directional gate
   is required; a vol-brake is not enough.
5. **PARTICIPATE-AND-PRESERVE (4e0c34e + 39435d4):** is there a long-only construction that
   participates in bulls AND preserves bears? Tested 4 directional gates + a regime-routed ENSEMBLE.
   One point (drawdown_aware) breaks NE on the in-window plane but FAILS held-out (p05 negative,
   permutation p~0.07-0.12, PBO 0.671 -- a 2021-bull + best-of-N mirage). The ensembles move ALONG the
   tradeoff, never beat it. => participation<->preservation is a FUNDAMENTAL long-only tradeoff (you
   preserve by being in cash = non-participation; a gate going to cash is bounded at +0% in a bear).
6. **THE LONG-ONLY FUNDING TILT (39435d4):** the one untested long-only wealth angle -- tilt a strict
   long-only book toward low-funding assets (no shorts). ARTIFACT: negative tilt-alpha in all 12
   variants x 3 splits, even gross of cost. Mechanism: corr(funding, next-day return) = +0.006 (wrong
   sign) -- high-funding names OUTPERFORM (funding is a spot momentum proxy). The dispersion edge lived
   in the SHORT leg; amputate it and a long-only spot tilt is anti-momentum + collects no funding.

## THE DEFINITIVE ANSWER
**Within STRICT long-only + spot, no internal-data strategy produces WEALTH that translates.** Every
avenue collapses to one of two outcomes, confirmed held-out + multiple-comparisons-deflated:
- **Drawdown-INSURANCE** (the directional/MA/TI/family book): preserves bears OOS, out-compounds BH
  full-cycle by losing less -- but a RISK tool, NOT alpha (fails ship-gate p05<0; loses to BH on raw
  wealth +57.7% vs +75.5%; wins only on Calmar 2.25 vs 0.79).
- **Value-destruction / noise** (config-rank, structural-selection, funding-tilt).

The participation<->preservation tradeoff is fundamental; the funding edge lives in the short leg.

## What this means for you (the honest deployable + the decision points)
- **The one thing that translates = drawdown-insurance.** If you want a DEFENSIVE risk-overlay (cut the
  bear, accept under-participating the bull), the family-ensemble book at LIGHT de-risk is real and
  OOS-replicated -- deploy it as a small RISK sleeve, NOT as a wealth engine. Expected by regime: ~10-30%
  bull capture, near-flat in crashes, ~30% time-in. Do not size it expecting alpha.
- **For WEALTH translation, the internal-data long-only-spot ceiling is rigorously confirmed.** The only
  doors that re-open it are STRATEGIC and YOURS to choose:
  (a) **EXTERNAL off-price data** (Coinglass liq-heatmap / on-chain netflow) to TIME de-risk and re-entry
      from information beta-reshaping cannot see -- a new SIGNAL lane, money-costing ingest, your call.
  (b) **Relax the long-only-spot constraint** (short / carry / market-neutral) -- which you have
      permanently ruled out as a shortcut. Honoring that, (a) is the only forward door.

## The single most important open question
Does an EXTERNAL leading signal (Coinglass/on-chain) clear held-out p05>0 where every internal price/
funding signal failed? That is the one test that could move the ceiling -- and it is a data-ingest
decision only you can make. Everything internal + long-only is, on this rigorous multi-angle evidence,
a drawdown-insurance ceiling.

Per-phase detail: FAMILY_ENSEMBLE_FINDINGS.md, PARTICIPATE_PRESERVE_FRONTIER.md, PP_ENSEMBLE.md,
FUNDING_TILT_LONGONLY.md, TRANSLATION_SOLUTION_2021.md.
