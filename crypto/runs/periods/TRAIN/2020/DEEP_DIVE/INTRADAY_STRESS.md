# INTRADAY-RESOLUTION FALSIFIER -- does dynamic timing skill appear at GENUINE sub-daily resolution?

PHASE 4a. The named falsifier of PHASE 3's "no dynamic-timing skill" verdict + the user's finer-TF (<=1d)
focus. PHASE 3's synthetic generator was DAILY, so its "30m" was a sleeve-config label, not real 30m bars.
This phase builds a TRUE sub-daily (intraday) synthetic regime generator at genuine 30m resolution
(~48 bars/day = far more timing opportunities) and re-tests the dynamic regime-allocation engine across
bull/bear/chop/stitched. Tool: `src/strat/synthetic_intraday_stress.py`. JSON: `synthetic_intraday_stress.json`.

## THE GENERATOR (built + VALIDATED 3/3 -- the load-bearing deliverable)
A genuine sub-daily generator now exists: intraday U-shape vol + GARCH-like vol-clustering + Student-t
fat tails (t_df ~4.2-4.5) + per-regime drift/vol + cross-asset BTC-beta (mean pairwise corr 0.563, beta
1.094), all calibrated on 2020 NATIVE 30m bars ONLY (no 2026/other data). VALIDATION (before any result
trusted): synthetic vs real-2020-native-intraday return distribution + |r| ACF + the daily-aggregate
consistency check -- ALL MATCH in all 3 regimes (bull/bear/chop). `generator_validation._summary` =
"VALIDATED (intraday dist + |r| ACF + daily-aggregate all match real 2020 native intraday)". Chart:
`charts/intraday_generator_validation.png`.

## THE VERDICT -- ROBUST TO RESOLUTION (dynamic timing is still null)
**Even at genuine intraday resolution across distinct bull/bear/chop/stitched regimes, the dynamic engine
did NOT significantly beat the static blend on the paired sign test at ANY (cadence,regime) cell.** More
bars did NOT manufacture timeable structure. The real-data 30m candidate's apparent skill did NOT
replicate on the validated intraday synthetic -> it was 2020-bull-specific exposure tilt / a
multiple-comparisons artifact, NOT genuine intraday timing skill. `dynamic_significant_hits = []`,
`dynamic_significant_stitched = []`. Chart: `charts/intraday_dynamic_skill_by_regime.png`.

=> The named falsifier was tested, with a VALIDATED intraday generator, and the verdict SURVIVED: SHIP
THE STATIC BLEND; the dynamic timing layer is not worth its complexity. This is the honest, two-sided
closure of the dynamic-engine question -- it was given the resolution it needed to shine and still showed
no skill.

## POWER + CAVEAT (firmed)
Run at **10 seeds, 30m** (`repro.command --seeds 10 --cadences 30m`; an initial 2-seed scout was
re-run firmer). At 10 seeds the paired sign test is adequately powered and the verdict held: 0 of 4
(cadence,regime) cells significant (`dynamic_significant_hits = []`), consistent with PHASE 3's
well-powered 20-seed DAILY result. So the "no dynamic-timing skill" verdict is robust BOTH to resolution
(daily -> genuine intraday) AND to power (2 -> 10 seeds). One residual limit: a single cadence (30m) +
30m=48-bars/day -- a 15m run would add a second intraday cadence (compute-heavy; the direction is already
robust across daily-20-seed + intraday-10-seed). Do not cite intraday net magnitudes as forward numbers
(synthetic, bull/bear/chop calibrated to 2020 only).

## CHARTS
- `charts/intraday_generator_validation.png` -- synthetic vs real-2020 intraday dist + ACF + daily-agg.
- `charts/intraday_dynamic_skill_by_regime.png` -- dynamic vs static vs trend-alone across regimes.

Repro + git_sha in `synthetic_intraday_stress.json`. Overseer commits.
