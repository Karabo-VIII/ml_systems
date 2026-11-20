# Within-year TI deep-dive (6/3/3) -- 2021 solved + reconciled with 2020 (quant referee, 2026-06-16)

> User /quant 2026-06-16: *"run 2021 on its own (6mo Train / 3mo Val / 3mo OOS) across all TI families and
> individual TIs. Solve, iron out, upgrade, then reconcile with the 2020 findings."* Tool:
> [`src/strat/deep_ti_within_year.py`](../src/strat/deep_ti_within_year.py).

## The upgrade
The canonical 2020 TI pipeline used a 2-window (VAL Jul-Sep / OOS Oct-Dec) split. This **upgrades it to a
year-agnostic 6/3/3 within-year runner** (TRAIN 6mo / VAL 3mo / OOS 3mo, robust := TRAIN&VAL both positive, **OOS
held out**), reusing the EXACT signal functions + FULL stack (trail10+min_hold+lag+maker) + fixed-EW u10 book.
So 2020 and 2021 are compared under the **identical methodology** (apples-to-apples). All 18 TIs / 6 families,
BASE vs IRONED. Long-only spot, UNSEEN sealed.

## Pre-registration (referee)
- **H0:** the within-year TI structure does NOT reproduce 2020->2021 (family ordering, iron-effect sign, robust-band
  participation are regime-transient). **H1:** it reproduces (family OOS-xBH Spearman > 0.5; iron-DD sign holds;
  robust band participates both years).
- **Multiplicity is the headline risk** (thousands of TI x config x TF cells): report the **robust-band aggregate**,
  never the cherry-picked best; benchmark vs buy-hold; a single "best config" would need Deflated-Sharpe.

## Reconciliation (within-2020 vs within-2021, identical 6/3/3) [VERIFIED within-year]

| family | 2020 xBH | 2021 xBH | 2020 rob% | 2021 rob% | iron dDD 2020/2021 | beat-BH 20/21 |
|---|---|---|---|---|---|---|
| trend | 0.86 | 3.18* | 1.0 | 1.0 | +6.0 / +4.2 | 0 / 93 |
| momentum | 0.87 | 2.62* | 0.9 | 1.0 | +3.7 / +1.8 | 0 / 19 |
| breakout | 0.82 | 1.75* | 1.0 | 0.9 | +4.0 / +1.1 | 0 / 6 |
| mean-reversion | 0.53 | 1.91* | 0.57 | 0.69 | +6.3 / +10.7 | 0 / 47 |
| volume | 1.00 | 2.14* | 0.71 | 0.82 | +2.6 / +3.4 | 0 / 6 |

(iron dDD > 0 = iron REDUCES drawdown; *2021 xBH is inflated by a near-zero BH denominator -- see below.)

### What reproduces (the de-risked-beta STRUCTURE is stable)
- **Iron buys drawdown, both years:** iron reduces maxDD in **all 5 families in both years** (dDD > 0 everywhere).
  The iron mechanism reproduces. [VERIFIED]
- **The robust band participates in every family, both years** (robust_frac > 0 for all). The TRAIN&VAL-positive
  band carries OOS in both 2020 and 2021.
- **Mean-reversion is the weakest family in both years** (lowest robust_frac, lowest 2020 xBH).

### What does NOT reproduce (rank-fragility, one level up)
- **Family OOS-xBH rank-transfer: Spearman(2020, 2021) = 0.50** -- weak. The *leading* family reshuffles
  (2020: volume/momentum/trend-led; 2021: trend/momentum-led); only "MR is worst" is stable. **You cannot reliably
  pick "the best family" forward** -- the same rank-fragility as the config level (cross-year config rank ~0), now
  shown one level up at the family level.

### The decisive referee catch -- 2021-OOS "beats buy-hold" is EXPOSURE, not alpha
- The within-2021 **OOS = Oct-Dec 2021 was the post-ATH DECLINE/chop**, not a bull: **BH OOS net was only +6.3%**
  (vs 2020-OOS BH +49%, a clean bull). On that low base, **171 configs "beat" BH in 2021** (vs 0 in 2020).
- **But the beat-BH configs have median OOS time-in = 0.32** -- they are in CASH ~68% of the quarter. They beat the
  +6% BH by **sitting out the decline** (avoiding BH's -33% drawdown and netting modestly positive). This is
  **exposure-timing / preservation -- the "trend-signal going to cash" mechanism** (the cross-instance falsifier
  97c7104) -- **NOT genuine signal**. The 2021 xBH ratios > 1 are a near-zero-denominator artifact, not outperformance.
- The SAME de-risked configs **lose** to BH in the 2020 bull-OOS (0/thousands beat). So **2020-OOS and 2021-OOS are
  the two SIDES of de-risked beta**: underparticipate-and-lose in a bull quarter, preserve-and-win in a down quarter.
  There is no regime-independent alpha -- the relative outcome is entirely set by the OOS quarter's direction.

## Verdict
**The within-year TI structure PARTIALLY reproduces 2020->2021.** What is STABLE: the de-risked-beta mechanism --
iron reduces drawdown in every family both years; the robust band participates both years; MR is weakest both years.
What is FRAGILE: the family *ranking* (Spearman 0.50 -- can't pick the best family forward). The deployable read is
unchanged and now confirmed from both regime sides: **the families are a drawdown-preserving de-risked beta; the
iron buys DD + robustness, not bull-net.** "Beating buy-hold" is regime-conditional exposure (win the down-quarter
by holding cash, lose the bull-quarter by underparticipating) -- the same going-to-cash mechanism the cross-instance
family-ensemble falsifier found (no wealth edge over a free control). No new alpha; the long-only ceiling holds.

**Multiplicity / exposure discipline:** per-family numbers are the robust-band (TRAIN&VAL-positive) OOS aggregate,
not cherry-picks; the 171 "beat-BH" configs are one exposure artifact (cash in a decline), not 171 edges; a single
"best config" claim would require Deflated-Sharpe vs the N tried (not made -- the band, not the #1, is the unit).
**Caveat:** within-year, single OOS quarter per year (n small); the exposure control is the time-in proxy, not a
full exposure-matched-BH bootstrap (the cross-instance free-control falsifier already established the mechanism).

## RWYB
```
python -m strat.deep_ti_within_year --selftest
python -m strat.deep_ti_within_year --year 2021 --tfs 1d,4h,2h,1h
python -m strat.deep_ti_within_year --year 2020 --tfs 1d,4h,2h,1h
python -m strat.deep_ti_within_year --reconcile
```
Outputs under `runs/strat/within_year/` (within_2020.json, within_2021.json, reconciliation_2020_2021.json).
Reconciles with [FAMILY_ENSEMBLE_FINDINGS](../runs/strat/FAMILY_ENSEMBLE_FINDINGS.md) (the full-cycle book: same
de-risked-beta, no wealth edge) + [translation findings](FORWARD_TEST_2021_2026_06_15.md) (cross-year config rank ~0).
