# PHASE 7 -- CONDITIONAL BEAR-INSURANCE: the regime-gated longshort book (TWO-SIDED, claim-tagged)

**Question (the synthesis PHASE 6 teed up):** PHASE 6 found the ALWAYS-ON longshort sleeve gives bear DD-protection
but pays a heavy bull-drag at fine TF. The obvious fix: deploy the longshort insurance sleeve **ONLY when a sustained
bear is DETECTED** (long-only trend-alone otherwise) -- capture the bear protection, dodge the bull-drag. This tests a
**BINARY insurance toggle** driven by a **SLOW HYSTERETIC CAUSAL** market-regime detector. Genuinely different from the
continuous dynamic allocator that showed no skill in PHASE 2/3. The **SHUFFLE control is mandatory** (the same discipline
that killed the dynamic engine).

**Code:** `src/strat/regime_gated_longshort.py` (selftest PASS; `--selftest`).
**Run (RWYB):** `python -m strat.regime_gated_longshort --seeds 20 --cadences 1d,4h,2h,1h,30m,15m`
**Artifacts:** `runs/periods/TRAIN/2020/DEEP_DIVE/regime_gated_book.json` + the 2 PNGs below.
**Discipline:** 2020 band + the VALIDATED PHASE-3/6 synthetic generator ONLY (generator VALIDATED 3/3 regimes); detector
PRE-REGISTERED on the 2020 TRAIN band then FROZEN; SHUFFLE control mandatory; FIXED short trail-stop reused; maker +
modelled short-borrow (20 bps/yr); no MtM double-count; lag-1 causal; >=20 seeds (distributions). **SHORT side = RESEARCH
(LO-exception sign-off to deploy).** Not committed.

---

## HEADLINE [VERIFIED]

**The bear DETECTOR genuinely WORKS, but the gated longshort BOOK does NOT deploy on 2020-calibrated data.**

This is NOT the dynamic-engine null -- the detector has real, shuffle-beating TIMING skill. But the conditional book still
loses ~0.3-2.0pp net vs trend-alone over the full cycle and does not reduce drawdown. The honest deployable answer remains
**TREND-ALONE + trail (long-only)**; binary longshort insurance is **-EV** on a short/mild bear after borrow.

---

## THE DETECTOR (pre-registered, frozen) [VERIFIED]

- **Substrate:** equal-weight u10 basket CLOSE, daily; raw bear = `close < SMA(N)`; HYSTERESIS = arm after `K_ON`
  consecutive raw-bear bars, disarm after `K_OFF` consecutive calm bars; 1-bar-shifted (**causal**, no look-ahead --
  unit-tested).
- **Pre-registration:** grid `{sma in [20,30,50], k_on in [3,5], k_off in [3,5]}` scored on the **canonical 2020 TRAIN
  band `M2.SPLIT["TRAIN"]` = 2020-01-01..2020-07-01** (the real in-sample window that CONTAINS the Feb-Mar COVID bear --
  the narrow sleeve scoring window Jul-Dec 2020 is a clean bull and cannot train a bear gate). Objective = TRAIN-only
  forward-return separation (off-day fwd mean minus on-day fwd mean), guarded to on-frac in [0.05, 0.6]. **FROZEN** to
  `{sma:20, k_on:5, k_off:3}` (fwd-ret separation +0.00956), then applied unchanged to the synthetic stress + the OOS.
  **Never fit on OOS or the synthetic test paths.**

## Q0 / Q3 -- THE DETECTOR SKILL TEST (decisive, stitched path) [VERIFIED]

A regime-TRANSITION detector can only be skill-tested where transitions exist: the **STITCHED full-cycle path**
(bull->bear->chop->bull; the embedded bear is bars 92-131 of 328). (Standalone single-regime panels are too short for a
slow gate to warm up -> the gate is ~always-off there, which makes the standalone shuffle DEGENERATE -- so the stitched
path is the proper surface.)

| Metric (stitched, vs true bear window) | Value | Read |
|---|---|---|
| gate precision (on-day is true-bear) | **0.36** | vs **base-rate 0.12** -> ~3x better than random |
| gate recall (bear covered) | 0.74 | covers most of the embedded bear |
| precision minus equal-freq shuffle | +0.24pp | |
| frac of seeds beats shuffle-95th pct | **0.90** | the gate beats a random toggle of equal frequency |
| GATED book net vs SHUFFLE book (per TF) | **+0.76 to +1.43pp** | TIMING value, not exposure (same exposure, different timing) |

**Verdict [VERIFIED]: the detector HAS real timing skill** -- it is NOT the dynamic-engine null.

## Q1 / Q2 -- THE BOOK CONTROLS (does gating deploy?) [VERIFIED -- NEGATIVE]

| TF | Q1 bear-protection CAPTURED (stitched DD) | Q1 full-cycle drag PAID | Q2 GATED vs TREND net | Q2 GATED vs TREND maxDD |
|---|---|---|---|---|
| 1d  | **-64.9%** | 14.4% | -2.03pp (FALSE) | -0.71pp (FALSE) |
| 4h  | -15.5% | 10.6% | -0.53pp (FALSE) | -0.13pp (FALSE) |
| 2h  | -20.3% | 10.3% | -0.52pp (FALSE) | -0.16pp (FALSE) |
| 1h  | -11.2% | 5.6%  | -0.33pp (FALSE) | -0.11pp (FALSE) |
| 30m | -6.5%  | 7.2%  | -0.44pp (FALSE) | -0.06pp (FALSE) |
| 15m | -6.5%  | 7.2%  | -0.44pp (FALSE) | -0.06pp (FALSE) |

The two ratios are computed on the **STITCHED full-cycle path** (where the gate actually operates across transitions; on
a standalone single-regime panel the gate barely fires, so a standalone capture/drag is structurally ~0 and meaningless).
**Bear-protection CAPTURED is NEGATIVE at every TF** -- the gated book does NOT keep the always-on's bear DD-protection.
**The gated book does NOT clear trend-alone** on net or maxDD at any cadence. PHASE 6's static book failed this bar; the
conditional book fails it too.

## Q3b -- THE MECHANISM (per-segment decomposition, REPRODUCIBLE, not asserted) [VERIFIED]

Per-stitched-segment cumulative `gated - trend` (1d, 20 seeds):

| Segment | Regime | gated - trend (cum pp) | gate on-frac |
|---|---|---|---|
| bull0 | bull | -0.53pp | 0.18 |
| bear1 | **bear** | **+0.11pp** | 0.74 |
| chop2 | chop | -0.41pp | 0.18 |
| bull3 | bull | -0.71pp | 0.22 |

**The conditional BEAR gain is TINY (+0.11pp)** -- the synthetic bear is only ~12% of the cycle and mild in NET terms.
Meanwhile the gate's **residual FALSE-ALARM firings in the bull/chop segments** (it still fires ~18-22% of the time even
there, each firing a SHORT into a RISING market) plus borrow sum to **~-1.6pp**. On a bull-dominated cycle, even a precise
gate's small false-alarm rate costs more than the short bear earns. **(Note: I initially hypothesized the gate "arms after
the trough" -- a direct check REFUTED that: the gate arms ~8 bars after bear onset and ~24 bars BEFORE the trough; only 5%
of seeds arm late. The false-alarm-in-the-88%-non-bear mechanism above is the verified one.)**

## Q4 -- 2020-REAL OOS anchor (Oct-Dec, a real BULL) [VERIFIED]

On the real bull the gate correctly stays mostly off (on-frac 0.08-0.25) and the gated book ~= trend with a small give-up
(e.g. 1d: TREND 13.6% / GATED 12.3%; 1h: TREND 50.4% / GATED 47.6%). Consistent with the synthetic finding: in a bull the
gate's small residual false-alarm rate is a pure cost, and there is no bear to offset it.

---

## DEPLOYABLE SYNTHESIS [VERIFIED]

- **It does NOT work as a deployable book.** Conditional bear-insurance via a slow hysteretic binary gate is **-EV** on
  2020-calibrated data. **TREND-ALONE + trail is the honest deployable long-only answer.** The bear rescue is
  **unconditional-only** (the always-on short, accepting the bull-drag -- RESEARCH) **or none**.
- **What is NEW and valuable here:** the detector is the FIRST regime signal in this campaign with **verified timing skill
  over its shuffle** (precision 0.36 vs 0.12; beats shuffle-95 in 0.9 of seeds). The reason it does not convert is purely
  a **payoff-arithmetic** problem on a short/mild bear, not a detection failure. This is a sharper, more useful closure
  than a flat null: detection works; binary longshort insurance is the wrong vehicle for it on this regime mix.
- **The explicit open door (untestable on 2020):** a **DEEPER or LONGER bear** (a larger NET share of the cycle, e.g. a
  2022-style grind-down) is the regime where the bear gain could exceed the non-bear false-alarm cost and flip the book
  +EV. The detector + book are BUILT and VALIDATED to run on such a band the moment it is in scope.

## TWO-SIDED CAVEATS (binding)

1. SYNTHETIC stress surface (PHASE 3/6 generator, 2020-calibrated stylized facts ONLY) + the 2020 real OOS -- not future data.
2. The longshort INSURANCE leg VIOLATES long-only+spot -> the gated book is **RESEARCH**; deploy needs the user's explicit
   **LO-exception sign-off**.
3. Detector PRE-REGISTERED on the 2020 TRAIN band ONLY then FROZEN -- never fit on OOS or the synthetic test paths.
4. The SHUFFLE control is a random-timed toggle of EQUAL on-frequency -- the detector beats it (TIMING skill, not exposure).
5. Short-borrow MODELLED at 20 bps/yr (prorated per-bar); a squeeze can spike it.
6. maker cost, no MtM double-count, lag-1 causal; >=20 seeds (distributions: mean +- spread + WORST seed).
7. The 2020-calibrated synthetic bear is SHORT (~12% of the cycle) and mild in NET terms; the verified mechanism is that
   the gate's false-alarm cost in the 88% non-bear time outweighs the small bear gain -- so the NULL is **specific to a
   short/mild bear**. A deeper/longer bear is the open door, NOT testable on the 2020 band.

## CHARTS

- `charts/regime_gated_bear_capture.png` -- bear-protection captured vs full-cycle drag paid, gated vs always-on, per TF
  (all TFs sit bottom-left: low drag but NEGATIVE capture -> gating fails the synthesis).
- `charts/gated_vs_trend_vs_shuffle.png` -- gated vs trend-alone vs the shuffled-toggle control across regimes (net +
  maxDD) + the per-TF book skill test (gated beats shuffle on net, ~0 on DD).

## REPRO

```
git_sha (at run): 2cd1329
python -m strat.regime_gated_longshort --selftest
python -m strat.regime_gated_longshort --seeds 20 --cadences 1d,4h,2h,1h,30m,15m
```
Detector FROZEN params: `{sma:20, k_on:5, k_off:3}` (pre-registered on M2.SPLIT["TRAIN"] = 2020-01-01..2020-07-01).
Generator validation: VALIDATED (3/3 regimes match real 2020 stylized facts).
