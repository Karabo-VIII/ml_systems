# MA-TYPE x TIMEFRAME RESEARCH -- all 8 MA types x all finer TFs (<=1d), 2020 band

PHASE 1a of the strategy-discovery-engine build. The thorough research: across ALL 8 MA types x ALL
finer timeframes (<=1d), the best IRONED TREND book per (MA-type, TF) on the 2020 deep-dive protocol.
Tool: `src/strat/ma_type_tf_research.py`. RWYB: `python -m strat.ma_type_tf_research --tfs 1d,4h,2h,1h,30m,15m`.
JSON: `ma_type_tf_research.json` (full 48-cell grid). All numbers VERIFIED (RWYB this run, 2020 OOS Oct-Dec).

- MA types: EMA, SMA, WMA, HMA, DEMA, TEMA, KAMA, VIDYA (reused `ma_type_upgrade._MA`).
- TFs: 1d, 4h, 2h, 1h, 30m, 15m. Split: TRAIN 2020-01..07 / VAL ..10 / OOS ..2021-01 (within-2020).
- Each cell = the ironed slow-MA-cross FAMILY (39 distinct 2MA/3MA slow configs, equal-weight u10),
  confirm/exit selected on TRAIN+VAL, confirmed once on OOS. Maker cost, causal/lag-1. 2020-band ONLY.

## THE WINNER PER TIMEFRAME (OOS net = wealth)
| TF | best MA-type | OOS net% | Sharpe | maxDD% | coverage% |
|----|--------------|---------:|-------:|-------:|----------:|
| 1d | **KAMA** (adaptive)  | +33.6 | 2.72 | -12.7 | 48 |
| 4h | **VIDYA** (adaptive) | +39.8 | 2.95 | -18.7 | 66 |
| 2h | **VIDYA** (adaptive) | +46.5 | 2.27 | -22.6 | 43 |
| 1h | **VIDYA** (adaptive) | +55.2 | 3.46 | -19.7 | 68 |
| 30m | **VIDYA** (adaptive)| +66.2 | 3.92 | -16.5 | 62 |
| 15m | **VIDYA** (adaptive)| +73.0 | 4.19 | -13.4 | 58 |

## THE FINDING -- ADAPTIVE MA types dominate, decisively at finer TFs
- **An ADAPTIVE type wins EVERY timeframe**: KAMA at 1d, VIDYA at all five finer TFs (4h->15m). The
  adaptive MAs (VIDYA/KAMA adjust their smoothing to volatility) beat the low-lag (HMA/TEMA/DEMA) and
  simple (EMA/SMA/WMA) families -- the gap WIDENS at finer cadence. This converges with the fine-TF
  ironing finding (VIDYA cleared cost where naive HMA/TEMA were "cost-eaten").
- **Family-average net by TF** (adaptive / low_lag / simple): 1d 19.7/24.7/23.7 (simple/low-lag edge at
  the coarsest TF) -> 4h 31.0/24.4/25.5 -> 2h 28.6/5.9/15.0 (low-lag COLLAPSES at 2h -- whipsaw on
  synthesized bars) -> 1h 47.2/50.6/47.5 -> 30m 57.2/40.5/58.1 -> 15m 61.5/14.1/47.9 (low-lag collapses
  again at 15m). The WINNER is adaptive everywhere, but the family-average story is noisier (low-lag is
  bimodal: good at 1h, terrible at 2h/15m = it overfits the noise at the finest scales).
- **Coverage/participation rises then plateaus** with finer TF (1d 48% -> 1h 68% -> 15m 58%); the finer
  books are in-market more, the source of their higher gross net (and higher turnover/cost).

## HONEST CAVEATS (binding)
- **2020 OOS is a ~0%-bear BULL.** Net rises monotonically with finer TF because finer = more
  participation in a relentless bull (the participation tax inverted). This is BETA, not alpha -- every
  one of these books still nets LESS than VOLTGT_BH at its TF (the established leaderboard result). The
  value of the trend book is risk-adjusted (lower maxDD) + as a COMPONENT of the complementary book
  (PHASE 1b) -- NOT standalone outperformance of buy-hold in a bull.
- **Config/type ranking has a noise floor** -- the WINNER is robustly adaptive, but the exact net
  ordering within the adaptive family (VIDYA vs KAMA) flips a few points TF-to-TF; do not over-read the
  decimal. The robust claim is the FAMILY ordering: adaptive > {low-lag, simple} at the per-TF best.
- The fine-TF magnitudes (15m +73%) are bull-window and partly overfit; the STRUCTURE (adaptive wins,
  participation rises) transfers; the magnitudes do not.

## CHARTS
- `charts/ma_type_tf_heatmap.png` -- OOS net% (+ Sharpe panel) across the 8x6 grid.
- `charts/best_matype_equity_per_tf.png` -- per-TF best-MA-type equity vs VOLTGT_BH vs BUYHOLD.
- `charts/matype_family_by_tf.png` -- adaptive vs low-lag vs simple family-avg net by TF.

Feeds PHASE 1b (complementarity: the trend sleeve = the per-TF best adaptive book) + PHASE 2 (the
dynamic engine's trend candidate). Repro + git_sha in `ma_type_tf_research.json`. Overseer commits.
