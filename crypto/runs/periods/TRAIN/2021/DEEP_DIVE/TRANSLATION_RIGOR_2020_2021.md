# Translation rigor 2020->2021 -- what ACTUALLY translates (correcting an over-claim)

User caught an over-claim 2026-06-17: I had said "the robust BAND per (type,TF) translates." It does NOT, in the
predictive sense. This note records the rigorous test. [VERIFIED]

## The test: does the 2020 robust band carry to 2021 at the MEMBER level?
For each (TI, TF), take the configs in the 2020 robust band (iron TRAIN&VAL>0 in 2020) and measure their
2021-OOS-positive rate vs the BASE rate (all configs OOS-positive in 2021). LIFT = band_rate - base_rate.

RESULT across 98 (TI, TF) cells: **median LIFT = +0.000, mean -0.009.** Only 14/98 cells beat base by >2pp.
=> Being in the 2020 band gives ~ZERO predictive edge for 2021 OOS. The 2020 band does NOT translate.
(The "stayRob ~ 1.00" is a red herring: robust = TRAIN&VAL>0 is a low bar most configs clear in two bull-ish
years, so a band re-EXISTS both years -- but its existence is near-vacuous, not predictive.)

## The corrected translation hierarchy (2020->2021)
| level | translates? | evidence |
|---|---|---|
| single config #1 (frozen 2020) | NO | rank-transfer Spearman ~ 0; 2020-best -> 2021-worst (ADX) |
| 2020 robust BAND (frozen, member-level) | NO | OOS-positive lift over base = +0.000 (this test) |
| ROLLING band (recent/trailing, re-selected each window) | YES (modest) | rolling-pick > in-sample static-#1 in held-out tests; it tracks RECENT performance, not 2020 |
| de-risked-beta CLASS (participate bull / preserve via cash) | YES | every TI preserves; the class behaviour reproduces |

## What this means for deployment
- Do NOT freeze and deploy the 2020 band (or the 2020 #1) -- neither predicts 2021.
- The deployable object is a ROLLING band: re-capture the recent working region each window (the band is RE-FORMED
  from trailing data, not inherited from 2020). The "80 cross-year candidate cells" earlier = (type,TF) cells that
  INDEPENDENTLY had a robust+positive band in each year -- that is a statement about the CLASS/region recurring,
  NOT about the 2020 members carrying. Corrected.
- The honest translating edge = de-risked beta (class) + rolling re-selection. Magnitude, not direction, is what a
  frozen 2020 artifact loses going forward.

## Independent verification (multi-agent re-derivation, 2026-06-17) -- all 3 CONFIRMED + a NEW flag
A 3-agent adversarial workflow independently re-derived (not trusting the harness):
- **Band non-translation: CONFIRMED.** Median lift 0.0000 / mean -0.0086 across 98 cells; config-name match 100%.
- **PIT-2021 universe: CONFIRMED exactly.** 104 files -> 44 admitted / 34 pre-2021 / 10 new; window split 6/2/2;
  all 4 post-TRAIN listers (DEXE/DYDX/MOVR/QI) have 0 TRAIN bars.
- **Per-instrument dispersion: CONFIRMED.** AVAX modal carry (8/12 TIs, +ve in 12/12), ADA modal drag (10/12),
  median positive-fraction 0.60 (slim majority).

**NEW INTEGRITY FLAG (the verifier's catch): the "robust" band is too WIDE to be a filter.** 70% of 2020 configs
(1066/1529) and 81% of 2021 configs are flagged robust under the loose `TRAIN&VAL>0` rule -- in two bull-ish years
almost every long-only config is positive, so "robust" barely discriminates (48/98 candidate cells had band == the
entire config population, mechanically forcing lift=0). TIGHTENING to a discriminating rule (positive across
TRAIN&VAL&OOS 3-way AND |drift|<=10): robust drops 70%->49%, and cross-year candidate (TI,TF) cells drop **98 -> 48**.
The non-translation verdict is ROBUST to the definition (even the strict subset gives ~0 lift); but the candidate
COUNT was inflated by the loose filter -- use the strict rule going forward. Lesson: in bull years, "positive/robust"
is a low bar; the discriminating test is drift + held-out, not mere positivity.

## Caveat on the within-year universe (open, being addressed)
These tests used a FIXED u10 universe for both years. 2021 had NEW LISTINGS not in u10; the within-year 2021 should
use a POINT-IN-TIME universe (coins admitted at listing date, falling into TRAIN/VAL/OOS by when they listed). That
re-run is the next correction (per the user). The frozen-band non-translation result is universe-robust (it is a
within-u10 config-level statement), but the candidate magnitudes will shift under the PIT universe.
