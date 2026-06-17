# Cross-period: Jan 2020 vs Feb 2020 (TRAIN) — does the MA block hold?

Two consecutive months, same full analysis (`structural_fixes` + `ma_compare`, oldest data).
**Context:** Jan 2020 = a clean **rally**; Feb 7–Mar 7 2020 straddles the market **top (~Feb 19) + the
start of the COVID crash** = a rally→reversal **transition** (a natural stress test, not cherry-picked).

## 1. The killer REPLICATES (and Feb is harder)
Fast MA on a fine cadence dies in BOTH months; Feb worse (the choppy top whipsaws more):

| 15m, fast-MA | baseline | min-hold | whipsaw (base) |
|---|---|---|---|
| **Jan** | −23.0% | −9.9% | 12.5 |
| **Feb** | **−46.4%** | −32.5% | **26.9** |

## 2. The min-hold FIX replicates as the winner — but is PARTIAL + has a small regime cost
min_hold(12) is the best overlay at the failing cadences in BOTH months (15m: Jan +5.4pp, Feb +5.4pp;
30m: Jan +4.2, Feb +4.4) and eliminates whipsaw (→0.1) both times. BUT:
- **Feb residual is worse:** 15m-fast is still −32.5% after the fix (Jan was −9.9%). The fix rescues, it
  doesn't make the fastest-MA/finest-cadence profitable — and in a reversal month it's deeply negative.
- **The regime cost MATERIALIZED in Feb:** min-hold *hurt* the slow MAs at 4h (−0.6pp, 4h-slow & 4h-vslow)
  — exactly the "holds through a reversal" risk flagged in Jan. Small, but it shows up the moment the trend
  turns. (In Jan it hurt nothing.) So min-hold is a fine-cadence fix, NOT a free lunch at slow/coarse.

## 3. The CONSISTENT FAMILY is the same in both months: 2MA · slow(60-150)
| | Jan | Feb |
|---|---|---|
| most consistent family | **2MA slow(60-150)** | **2MA slow(60-150)** |
| mean / %pos / maxDD | +14.3% / 91% / −10% | +18.2% / 83% / −21% |
| 2MA vs 3MA | 2MA wins every speed | 2MA wins every speed |
| fast-MA | killer (49/37% pos) | killer (39/27% pos) |

**2MA-slow is the robust answer across both months; 3MA is consistently more fragile.** Feb is harder
(deeper DD −21% vs −10%, lower %pos), as expected for a transition month.

## 4. What Feb newly exposed (the value of going period-by-period)
- **Per-asset divergence under stress:** in Feb, **BTC (−1.5%) and XRP (−1.0%) went NEGATIVE** (Jan both
  positive); ETH +21%, LINK +34% led. The majors got chopped at the top; some alts kept trending. So even
  the consistent family has per-asset regime sensitivity.
- **maxDD roughly DOUBLED** Jan→Feb (−10%→−21% for the best family) — the transition month is where risk
  controls earn their keep.

## Verdict
The structural skeleton is ROBUST across two months: (killer = fast-MA/fine-cadence) + (fix = min-hold,
fine-cadence only) + (consistent family = 2MA-slow). The regime-dependence is REAL but BOUNDED — min-hold
costs ~0.6pp on slow/coarse in a reversal, and DD doubles in the transition. Next: a true bear month
(2022) to see if the family ranking + fix survive a sustained downtrend, not just a top.

Artifacts: `runs/periods/TRAIN/2020/{01_JAN,02_FEB}/`.
