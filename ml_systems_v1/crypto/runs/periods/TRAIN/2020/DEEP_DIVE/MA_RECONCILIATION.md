# Reconciliation: why the two MA-leaderboard lanes show different numbers (2026-06-14)

User /orc: "check the other instance -- are their numbers correct? I asked them for the same conclusions but
their numbers are different." Two instances built 2020 MA per-config leaderboards in parallel:
- **THIS lane:** `deep2020_ma_remedy/top10/robust_split.py` -> MA_REMEDY/MA_TOP10/MA_ROBUST_SPLIT.md.
- **OTHER lane:** `ma_2020_config_leaderboard.py` -> CONFIG_LEADERBOARD.md + the WORKING BAND.

## VERDICT (empirically tested, not asserted)
**Both harnesses are CORRECTLY implemented and AGREE EXACTLY where the methodology is identical.** The
differences are all explained by deliberate methodology deltas, with ONE genuine artifact (skipna alignment at
fine TF). Proof: comparing the OOS (Oct-Dec) net of every SHARED config:

| TF | mean (mine - theirs) OOS net | verdict |
|---|---|---|
| 1d | **+0.0pp (|max| 0.0)** -- BIT-IDENTICAL | both correct; no methodology delta at 1d |
| 4h | -1.3pp (|max| 2.2) | tiny -- minor fine-bar alignment |
| 1h | **-17.0pp (|max| 25.2)** | the skipna artifact (theirs inflated) |

Examples: 1d EMA(2,3) OOS = 20.5 in BOTH; 1d HMA(2,3) OOS = 41.0 in BOTH. 1h HMA(2,3) OOS = theirs 75.5 vs
mine 40.1. [VERIFIED -- cross-harness diff, 2026-06-14]

## THE DELTAS (what makes the numbers look different)
1. **WINDOW (the headline gap).** OTHER lane's PRIMARY metric is **FULL-2020 net (whole year, Jan-Dec)** -- e.g.
   EMA(2,4,19) FULL = 193.1% (vs u10 buy-hold FULL = 199.2%). THIS lane reports **OOS = Q4 only** (~25-44%).
   193% vs 30% is the SAME strategy over a 12-month vs a 3-month window -- not a contradiction. Both lanes share
   the SAME VAL (Jul-Sep) and OOS (Oct-Dec) split boundaries; the OTHER lane just ALSO has a TRAIN (H1) leg and
   a FULL (year) aggregate that this lane doesn't compute.
2. **ALIGNMENT (the one genuine artifact).** OTHER lane averages assets with `mean(axis=1, skipna=True)` (its
   docstring flags this as a deliberate "EW-of-available" choice). At FINE TF, 2020 listing dates (SOL/AVAX list
   ~Sep 2020) mean many bars are missing one+ asset -> skipna reweights to EW-of-present -> INFLATES finer
   cadences (1h OOS reads ~75% where the cadence-invariant fixed-EW is ~40%). THIS lane fixed this to
   `fillna(0.0).mean(axis=1)` (unlisted = cash = 0) after an adversarial verification caught it -- making the
   book CADENCE-INVARIANT (1d/4h/1h buy-hold all ~47-52% here, vs theirs 89% at 1h). **At 1d/coarse the two
   conventions COINCIDE (no missing bars) -> identical numbers; the divergence grows with finer TF.**
3. **VAL convention.** Same root cause: in VAL (Jul-Sep) SOL/AVAX are partly unlisted. Their skipna VAL is
   ~6-14pp HIGHER than this lane's fixed-EW VAL (which charges the unlisted slots as cash). A convention choice;
   fixed-EW is cross-cadence-honest, skipna is "trade what's listed."
4. **GRID.** OTHER lane uses distinct_specs max_n=60 (120 configs/cell); THIS lane max_n=40 (80 configs). The
   shared configs match; the larger grid just adds longer-period configs (so a different #1 is possible).
5. **SHARPE annualization.** OTHER lane annualizes on NATIVE bars (sqrt(365*bars_per_day)); THIS lane on a
   DAILY-resampled series (sqrt(365)). At 1d identical; at finer TF the Sharpe magnitudes differ (net is the
   cleaner cross-lane comparison and is what was tested above).
6. **ROBUSTNESS definition.** OTHER lane's BAND = positive across TRAIN AND VAL AND OOS (3-way positivity).
   THIS lane's robust = |drift|<=10 (VAL~=OOS consistency). Different but compatible robustness lenses.
7. **OBJECTIVE.** OTHER lane sorts by FULL-2020 net (wealth, most data). THIS lane (after a user correction)
   sorts by OOS net (wealth) within a robust/non-robust split. Both are wealth-first.

## WHAT THIS MEANS
- **Trust either lane at 1d / coarse cadence** -- they are bit-identical (mutual cross-validation).
- **At fine TF (1h/2h/30m/15m), the OTHER lane's NET magnitudes are skipna-inflated** -- treat them as upper
  bounds, not cross-cadence-comparable. THIS lane's fixed-EW fine-TF nets are the honest ones. (The OTHER lane's
  primary deliverable -- the BAND / 3-way positivity -- is largely ROBUST to the skipna inflation, since it is a
  per-split SIGN test, not a magnitude; so their CONCLUSION "trust the band, ordering is noise" still stands.)
- **The 193%-vs-30% headline is window (full-year vs Q4), not a disagreement.**

## RECOMMENDED RECONCILIATION
The two lanes double-track the same 2020 MA terrain. Unify on ONE harness with the fixed-EW alignment (one-line
change in `ma_2020_config_leaderboard.py`: `.mean(axis=1, skipna=True)` -> `.fillna(0.0).mean(axis=1)` at the
book + buyhold aggregations) so the fine-TF nets become cadence-invariant; keep the OTHER lane's TRAIN/FULL
splits + BAND + rank-stability (those are additive and good). Then there is ONE set of numbers, not two.
(Not patched here: the OTHER lane is an actively-committing instance + skipna is a documented choice there;
this is a coordination action, flagged not forced.)
