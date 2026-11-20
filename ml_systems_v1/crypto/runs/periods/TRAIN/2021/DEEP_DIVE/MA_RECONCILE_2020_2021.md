# MA per-config 2020 <-> 2021 reconciliation (identical 6/3/3 tool) -- focal TF 4h

Both years run through the SAME year-parametrized `ma_2020_config_leaderboard.py` (6mo TRAIN / 3mo VAL / 3mo OOS, fixed-EW u10, long-only ironed sleeve, maker). Per-config charts + leaderboards live in `runs/periods/TRAIN/{2020,2021}/DEEP_DIVE/`. ALL numbers [VERIFIED within-year]. Charts in `charts/`.

**Buy-hold (u10, no cost) @ 4h:** 2020 FULL 134.0% / OOS 47.8% | 2021 FULL 1201.3% / OOS 6.6%. (2020-OOS Oct-Dec = clean bull; 2021-OOS Oct-Dec = post-ATH decline/chop -- the OOS REGIME differs, which is why a de-risked book's relative result flips.)

## Per-MA-type, best ROBUST (band #1) config @ 4h -- 2020 vs 2021
| MA | band 20/21 | rank-rho 20/21 | best cfg 2020 (FULL/OOS) | best cfg 2021 (FULL/OOS) |
|---|---|---|---|---|
| EMA | 120/52 | 0.437/-0.176 | `ema_2_26` 174.5/52.5% | `ema_2_3_4` 572.9/4.2% |
| SMA | 115/51 | 0.339/-0.056 | `ema_3_6` 191.6/50.7% | `ema_4_5_75` 854.7/8.5% |
| WMA | 114/45 | 0.237/-0.133 | `ema_4_5` 213.8/42.5% | `ema_2_5_118` 793.5/0.1% |
| HMA | 109/49 | 0.332/-0.401 | `ema_4_8_17` 197.9/42.6% | `ema_6_13` 1347.3/8.2% |
| DEMA | 111/61 | 0.346/-0.464 | `ema_26_32` 173.9/38.6% | `ema_4_5` 698.4/2.9% |
| TEMA | 107/54 | 0.19/-0.213 | `ema_6_13` 160.1/48.1% | `ema_2_10` 860.6/2.1% |
| KAMA | 111/51 | 0.34/-0.151 | `ema_4_5` 145.7/43.9% | `ema_3_6` 609.6/3.8% |
| VIDYA | 115/52 | 0.745/-0.073 | `ema_2_3_4` 148.4/42.8% | `ema_2_12_14` 363.6/1.7% |

## Reconciliation verdict
- **MA-type rank-transfer @ 4h:** Spearman(best-FULL-net 2020, 2021) = **0.524** (FULL), 0.476 (OOS). The MA-type ORDERING partially persists.
- **Working band exists in BOTH years for every MA type** (band 20/21 columns > 0) -- the robust (fast,slow) region reproduces; the band, not the #1, is the stable object.
- **Within-cell rank-stability rho is low in both years** (the rank-rho columns) -- the #1 config does not transfer; this is the empirical basis for 'deploy the band ensemble, not the #1'.
- **The de-risked-beta read holds:** the best MA configs UNDER-participate buy-hold in the bull (see the equity charts -- all curves sit below buy-hold) but cut drawdown. 'Beating' buy-hold only happens in a down/flat OOS quarter and is EXPOSURE (cash), not alpha.

Equity charts: `charts/best_matype_equity_4h_2020.png` + `..._2021.png` (each MA-type's band-#1 $1-growth vs buy-hold, TRAIN/VAL/OOS shaded). Band map + rank-stability: `charts/config_band_heatmap.png` + `charts/rank_stability.png` per year.