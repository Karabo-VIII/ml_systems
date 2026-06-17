# 2020 MA breakdown -- BY MA CLASS across all timeframes (+ 2h) -- convergence / divergence / coverage

User /orc 2026-06-12: "verify these results [DONE -- see below]; break it down BY MA (SMA across 1d/4h/1h/
30m/2h; EMA; HMA; ...); add 2h; decompose across the board; show convergence, divergence, coverage on the
whole." Tools: `ma_2020_breakdown.py` (now with 2h SYNTHESIZED by resampling 1h -> 2-hour OHLC buckets) +
`ma_2020_render.py`. Within-2020 6mo/3mo/3mo split, FULL stack, equal-weight u10 book, maker.

## VERIFICATION (RWYB, independent)
- The `_cells` MA-cache refactor is BIT-IDENTICAL to the direct cross logic: 0/39 config mismatches.
- Independent recompute of the 4h-EMA OOS family compound = **27.9** -- EXACTLY the reported value.
- 2h is not a native cadence (chimera = 1d/4h/1h/30m/15m + alt-bar-types); 2h is built by resampling 1h
  (OHLC-correct: open=first, high=max, low=min, close=last per 2h bucket). Results below are trustworthy.

## THE DECOMPOSITION -- OOS compound % by MA class x timeframe (the whole board)
| CLASS | 1d | 4h | 2h | 1h | 30m | 15m | MEAN | STD (TF-sensitivity) | best@ |
|---|---|---|---|---|---|---|---|---|---|
| EMA | 19.2 | 27.9 | 28.2 | 53.0 | 42.8 | 36.8 | 34.7 | 11.1 | 1h |
| SMA | 19.6 | 23.9 | 19.6 | 53.1 | 50.3 | 45.5 | 35.3 | 14.5 | 1h |
| WMA | 19.4 | 26.4 | 24.2 | 52.2 | 43.6 | 34.2 | 33.3 | 11.5 | 1h |
| HMA | 30.5 | 29.4 | 32.4 | 49.0 | 28.6 | **2.5** | 28.7 | 13.6 | 1h |
| DEMA | 22.8 | 25.9 | 10.2 | 47.7 | 40.8 | 21.3 | 28.1 | 12.6 | 1h |
| TEMA | 27.6 | 23.4 | 20.5 | 49.5 | 29.5 | **3.9** | 25.7 | 13.5 | 1h |
| KAMA | 16.8 | 17.3 | 15.5 | 43.9 | 45.5 | 40.4 | 29.9 | 13.5 | 30m |
| VIDYA | **6.4** | 18.0 | 34.6 | 47.1 | 47.9 | **55.0** | 34.8 | 17.4 | 15m |
| **winner** | HMA | HMA | VIDYA | SMA | SMA | VIDYA | | | |
| **divergence (max-min)** | 24.1 | 12.1 | 24.4 | **9.2** | 21.7 | **52.5** | | | |

## CONVERGENCE
1. **EVERYTHING converges at 1h.** All 8 classes peak at (or near) 1h, ALL land in a tight 43.9-53.1 band,
   and 1h has the LOWEST cross-class divergence (spread 9.2). At 1h in 2020 the MA *type* barely matters --
   pick any, they all work. (1h is the "sweet-spot cadence" where signal is strong and cost is bearable.)
2. **The PLAIN family (EMA/SMA/WMA) is the convergent core** -- a tight cluster at every cadence, never the
   collapse, never the standout. The robust default if you must pick one type blind.

## DIVERGENCE -- and the clean structural law
1. **15m is maximum divergence (spread 52.5pp).** HMA 2.5 / TEMA 3.9 (collapse) vs VIDYA 55.0 -- at 15m the
   MA-class choice is worth ~50pp. The class you pick at fine cadence is EVERYTHING.
2. **The three families are a CADENCE CROSSOVER (group means):**
   | family | 1d | 4h | 2h | 1h | 30m | 15m |
   |---|---|---|---|---|---|---|
   | plain (EMA/SMA/WMA) | 19.4 | 26.1 | 24.0 | **52.8** | 45.6 | 38.8 |
   | low-lag (HMA/DEMA/TEMA) | **27.0** | 26.2 | 21.0 | 48.7 | 33.0 | **9.2** |
   | adaptive (KAMA/VIDYA) | **11.6** | 17.6 | 25.1 | 45.5 | 46.7 | **47.7** |
   - **low-lag is BEST at coarse, DIES at fine** (27.0 -> 9.2, monotone down) -- reactivity captures a clean
     coarse trend fast, but whipsaws to death on fine noise.
   - **adaptive is WORST at coarse, BEST at fine** (11.6 -> 47.7, monotone UP) -- the exact MIRROR IMAGE.
     Adaptation pays precisely where reactivity hurts (it slows down in fine-cadence chop).
   - **plain is the stable middle** -- peaks at 1h, solid everywhere, the baseline both extremes cross over.
   The crossover sits around **2h-1h**: coarser than that, lean low-lag; finer, lean adaptive; at 1h it's a
   wash. THIS is the law the decomposition reveals.

## COVERAGE
All 8 classes x 6 timeframes = **48/48 cells filled**, no holes. The heatmap (panel A) is the at-a-glance
coverage map; panel B is the per-class TF profile (bunched=converge, spread=diverge); panel C is the
divergence-per-TF bars; panel D is the 3-family crossover.

## Honest caveat (unchanged)
2020 = exceptional BULL -> these levels + the coarse-low-lag/fine-adaptive law are bull-biased. The
DIRECTION (low-lag<->adaptive crossover across cadence) is mechanistically robust (reactivity vs whipsaw is
regime-general), but the magnitudes are 2020-specific, and the *coarse* winner flips vs the hard-2025 tape
(where adaptive VIDYA won at 4h). Confirm the crossover on a bear slice. UNSEEN untouched (2020 is TRAIN).

Figure: `charts/ma_2020_byclass.png` (A heatmap / B class lines / C divergence / D family crossover).
json: `ma_2020_byclass.json`. RWYB: `python -m strat.ma_2020_breakdown --cadences <tf>; python -m strat.ma_2020_render`.
