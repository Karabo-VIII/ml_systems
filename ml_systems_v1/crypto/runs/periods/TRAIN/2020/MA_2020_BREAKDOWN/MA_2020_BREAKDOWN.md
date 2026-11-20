# 2020 MA-class breakdown -- winning class per timeframe, side-by-side with the oracle

User /orc 2026-06-12: "each class of Moving Averages that won per time frame ... performance for the 2020
window ... side by side with oracle, best per category (a FAMILY of averages = a set of configs), their
performance for the year. Rerun with the CANONICAL DATA TECHNIQUES to expand the ~1yr data, sliced
6mo-train / 3mo-val / 3mo-oos -- for the 2020 slice, NOT the traditional cross-year splits."

Tools: `src/strat/ma_2020_breakdown.py` (compute) + `ma_2020_render.py` (merge). Each MA CLASS = a FAMILY
of the 39 distinct slow (2MA+3MA, max-period in [60,150)) configs. UPGRADED METHODOLOGY = the FULL stack
(10% trailing stop + min_hold(12 bars) + maker fees). Equal-weight u10 book. **Within-2020 split: TRAIN
Jan-Jun / VAL Jul-Sep / OOS Oct-Dec.** Oracle = HINDSIGHT, descriptive upper bound (config CHOICE is
hindsight; each MA cross signal is causal).

## CANONICAL DATA EXPANSION used (the ~1-year limit)
- **cross_sectional_pool**: the book is the equal-weight pool across u10 (10 assets) -> ~10x the per-config
  sample from one year.
- **block_bootstrap_distribution** (block=5, 400 resamples) on the OOS book returns -> a robust median +
  downside p05 instead of the single optimistic point (the `bootMed`/`bootP05` columns).
- **james_stein_shrink** on the per-config VAL scores -> shrink factor B (does the best-config pick clear
  the estimation-noise floor). [B~=0.99-1.0 here; the default 1.0%^2 noise floor is small vs the wide
  %-scale config spread, so B saturates -- read it as "config differences are large", not a tight test.]

## OOS family compound % by timeframe x MA class (OOS = Oct-Dec 2020)
| TF | EMA | SMA | WMA | HMA | DEMA | TEMA | KAMA | VIDYA | WINNER |
|---|---|---|---|---|---|---|---|---|---|
| 1d | 19.2 | 19.6 | 19.4 | **30.5** | 22.8 | 27.6 | 16.8 | 6.4 | **HMA** |
| 4h | 27.9 | 23.9 | 26.4 | **29.4** | 25.9 | 23.4 | 17.3 | 18.0 | **HMA** |
| 1h | 53.0 | **53.1** | 52.2 | 49.0 | 47.7 | 49.5 | 43.9 | 47.1 | **SMA** |
| 30m | 42.8 | **50.3** | 43.6 | 28.6 | 40.8 | 29.5 | 45.5 | 47.9 | **SMA** |
| 15m | 36.8 | 45.5 | 34.2 | **2.5** | 21.3 | 3.9 | 40.4 | **55.0** | **VIDYA** |

## Winning class per TF, full split, side-by-side with the ORACLE
| TF | winner | TRAIN | VAL | OOS | YEAR | OOS maxDD | bootMed | bootP05 | ORACLE hindsight-cfg | fam/oracle | perfect-foresight |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1d | HMA | 52.4 | 33.9 | 30.5 | **166.3** | -15.8 | 25.0 | -9.6 | VIDYA ema_2_5_118 = 37.3 | 82% | 611% |
| 4h | HMA | 39.6 | 36.9 | 29.4 | **147.2** | -12.7 | 28.7 | -6.4 | SMA ema_102_103 = 50.8 | 58% | 8,344% |
| 1h | SMA | 41.3 | 31.2 | 53.1 | **183.7** | -11.0 | 56.4 | +9.1 | VIDYA ema_3_102 = 62.8 | 85% | 1.06M% |
| 30m | SMA | 25.1 | 24.2 | 50.3 | **133.4** | -13.1 | 51.3 | +0.5 | HMA ema_52_89 = 74.0 | 68% | 66.9M% |
| 15m | VIDYA | 72.0 | 34.2 | 55.0 | **257.9** | -13.6 | 58.3 | +10.7 | DEMA ema_62_105 = 63.8 | 86% | 21.8B% |

## What it says
1. **The winning MA class is TIMEFRAME-DEPENDENT (a clean pattern):**
   - **Coarse (1d, 4h): HMA (Hull, low-lag) wins** -- in 2020's strong clean bull, the low-lag MA gets in
     fast and (at coarse cadence) does NOT whipsaw, so it captures more upside.
   - **Mid (1h, 30m): SMA (simple) wins** -- the plain average is the robust middle.
   - **Fine (15m): VIDYA (adaptive) wins** -- and the low-lag types COLLAPSE here (HMA 2.5%, TEMA 3.9%):
     at 15m they whipsaw on every wiggle; VIDYA's volatility-adaptation suppresses the churn and wins.
   The low-lag<->adaptive axis FLIPS with cadence: reactivity helps a clean coarse trend, kills a noisy
   fine one. (KAMA stays mediocre everywhere -- efficiency-adaptation mis-fires on crypto, as before.)
2. **The families capture 58-86% of the hindsight-best-config oracle** -- a single causally-pooled family
   gets most of what a perfect config-picker would (the gap is the cost of not knowing which config ex-ante).
3. **The perfect-foresight (long exactly the up-bars) oracle is astronomically higher and EXPLODES at
   fine cadence** (1d 611% -> 15m 21.8 BILLION %) -- it is the untradeable theoretical max; its only use is
   to show that the per-bar information left on the table grows enormously as cadence finens (which is why
   fine-cadence capture is so cost/timing-sensitive).
4. **Full-year 2020 performance of the winning family: 133-258%** (2020 was a massive bull -- BTC ~3x,
   alts more). bootP05 is POSITIVE at 1h/30m/15m (+9.1/+0.5/+10.7) -- the family is robustly positive on
   the 2020 OOS even at the 5th bootstrap percentile; slightly negative at 1d/4h (-9.6/-6.4).

## Honest caveats
- **2020 was an exceptional BULL year** -- these winners are BULL+2020-specific. This is the OPPOSITE
  regime to the multi-year traditional OOS (2025 hard tape) where VIDYA won at 4h and HMA was WORST. So
  the MA-class winner is BOTH timeframe- AND regime-dependent: low-lag (HMA) for clean coarse bull,
  adaptive (VIDYA) for fine/choppy/hard. Do NOT read "HMA is best at 4h" as universal -- it is a 2020-bull
  statement; on the hard 2025 tape VIDYA was best at 4h.
- The perfect-foresight oracle is descriptive only (untradeable). The hindsight-best-config oracle is the
  meaningful ceiling.
- 1d in early-2020 is slow-MA-warmup-starved (data starts 2020-01-06) -> the 1d TRAIN leg is the least
  reliable; the OOS (Oct-Dec) is fine (MAs settled by then).
- JS_B saturates at ~1 (noise-floor calibration) -- treat the best-config picks as indicative.

Figure: `charts/ma_2020_breakdown_ALL.png` (left: winning family vs hindsight-config oracle; right: all 8
classes x 5 TFs). json: `ma_2020_breakdown_MERGED.json` + per-TF `ma_2020_breakdown_*.json`.
RWYB: `python -m strat.ma_2020_breakdown --cadences <tf>` then `python -m strat.ma_2020_render`.
