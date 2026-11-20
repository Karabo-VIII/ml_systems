# Y4 Critical-Phenomena Early-Warning Engines (TRAIN-only)
- Generated: 2026-05-23T09:49:20
- TRAIN cutoff: 2024-05-15
- BTC source: `btcusdt_v51_chimera_1d_20260519.parquet`
- Assets analyzed: 49
- Regime flips identified: 13  (debounced min_len=7)
- BTC drawdown events (>15% over 20d): 0
- Combined critical-event dates: 13
- Random baseline hit rates: K=5: 0.049, K=10: 0.088, K=20: 0.163

## Regime flips found
| flip_date | prev | next | prev_len | next_len |
|---|---|---|---|---|
| 2020-07-24 | CHOP | BULL | 199 | 319 |
| 2021-06-08 | BULL | CHOP | 319 | 22 |
| 2021-06-30 | CHOP | BEAR | 22 | 43 |
| 2021-08-12 | BEAR | CHOP | 43 | 61 |
| 2021-10-12 | CHOP | BULL | 61 | 77 |
| 2021-12-28 | BULL | CHOP | 77 | 27 |
| 2022-01-24 | CHOP | BEAR | 27 | 69 |
| 2022-04-03 | BEAR | CHOP | 69 | 8 |
| 2022-04-11 | CHOP | BEAR | 8 | 280 |
| 2023-01-16 | BEAR | CHOP | 280 | 22 |
| 2023-02-07 | CHOP | BULL | 22 | 191 |
| 2023-08-17 | BULL | CHOP | 191 | 77 |
| 2023-11-02 | CHOP | BULL | 77 | 196 |

## Universal leaders (median precision across assets, n_assets >= 5)
| observable | threshold | K | median_p | mean_p | median_lift | n_assets |
|---|---|---|---|---|---|---|
| vol_of_vol>p80 | 0.0005059 | 5 | 0.046 | 0.041 | 0.94 | 49 |
| var_susc>1.3 | 1.3 | 5 | 0.044 | 0.044 | 0.89 | 47 |
| ac1_14>0.3 | 0.3 | 5 | 0.042 | 0.049 | 0.86 | 49 |
| ac1_14>0.4 | 0.4 | 5 | 0.032 | 0.053 | 0.66 | 49 |
| var_susc>1.5 | 1.5 | 5 | 0.000 | 0.042 | 0.00 | 42 |
| var_susc>1.3 | 1.3 | 10 | 0.080 | 0.079 | 0.91 | 47 |
| ac1_14>0.3 | 0.3 | 10 | 0.073 | 0.086 | 0.83 | 49 |
| vol_of_vol>p80 | 0.0005059 | 10 | 0.069 | 0.070 | 0.79 | 49 |
| ac1_14>0.4 | 0.4 | 10 | 0.066 | 0.089 | 0.75 | 49 |
| var_susc>1.5 | 1.5 | 10 | 0.000 | 0.067 | 0.00 | 42 |
| ac1_14>0.3 | 0.3 | 20 | 0.158 | 0.160 | 0.97 | 49 |
| var_susc>1.3 | 1.3 | 20 | 0.154 | 0.156 | 0.94 | 47 |
| vol_of_vol>p80 | 0.0005059 | 20 | 0.148 | 0.133 | 0.90 | 49 |
| ac1_14>0.4 | 0.4 | 20 | 0.125 | 0.162 | 0.76 | 49 |
| var_susc>1.5 | 1.5 | 20 | 0.000 | 0.119 | 0.00 | 42 |

## Top per-asset specialists (highest precision, n_triggers>=6)
| asset | observable | threshold | K | n_triggers | precision | lift |
|---|---|---|---|---|---|---|
| arkmusdt | ac1_14 | 0.4 | 20 | 19 | 0.737 | 4.51 |
| jstusdt | var_susc | 1.5 | 20 | 67 | 0.627 | 3.84 |
| wldusdt | ac1_14 | 0.4 | 20 | 12 | 0.583 | 3.57 |
| arbusdt | var_susc | 1.3 | 20 | 29 | 0.517 | 3.17 |
| shibusdt | ac1_14 | 0.4 | 20 | 31 | 0.419 | 2.57 |
| avaxusdt | var_susc | 1.5 | 20 | 79 | 0.405 | 2.48 |
| zecusdt | var_susc | 1.3 | 20 | 123 | 0.390 | 2.39 |
| enjusdt | var_susc | 1.5 | 20 | 60 | 0.367 | 2.24 |
| xrpusdt | var_susc | 1.5 | 20 | 34 | 0.353 | 2.16 |
| xlmusdt | var_susc | 1.5 | 20 | 101 | 0.347 | 2.12 |
| promusdt | vol_of_vol | 0.001661 | 20 | 61 | 0.344 | 2.11 |
| zenusdt | ac1_14 | 0.4 | 20 | 36 | 0.333 | 2.04 |
| ethusdt | vol_of_vol | 0.0002228 | 20 | 299 | 0.311 | 1.90 |
| injusdt | var_susc | 1.3 | 20 | 165 | 0.309 | 1.89 |
| dexeusdt | ac1_14 | 0.3 | 20 | 73 | 0.301 | 1.84 |
| fetusdt | var_susc | 1.3 | 20 | 183 | 0.301 | 1.84 |
| trxusdt | var_susc | 1.3 | 20 | 77 | 0.299 | 1.83 |
| linkusdt | var_susc | 1.5 | 20 | 61 | 0.295 | 1.81 |
| chzusdt | var_susc | 1.3 | 20 | 228 | 0.285 | 1.74 |
| nearusdt | var_susc | 1.5 | 20 | 76 | 0.276 | 1.69 |

## 3-fold validation of best universal recipe
| fold | start | end | n_triggers | n_events | precision |
|---|---|---|---|---|---|
| fold1 | 2020-01-07 | 2021-06-20 | 1082 | 2 | 0.079 |
| fold2 | 2021-06-27 | 2022-12-02 | 1507 | 7 | 0.196 |
| fold3 | 2022-12-09 | 2024-05-15 | 1874 | 4 | 0.149 |

## Honest caveats
- TRAIN-only. No OOS / no post-2024-05-15 data.
- Regime label is BTC-conditional, SMA-200 + 30d EMA slope (log_close). Thresholds = +/-0.10 log; debounce min_len = 7 days.
- Number of flips is small (<= ~15 after debounce). Precision estimates have wide CIs.
- Per-asset specialists with low n_triggers (<6) are excluded from the specialist table.
- LPPL fitting (Y4.4) intentionally SKIPPED to honor 5-min budget.
- Lift > 1.5 + 3-fold consistency are the gates for a recipe to be considered actionable.
- Cross-asset agreement metric uses 7-day cumulative-return sign; can be confounded by BTC dominance.
- AC(1) on log-returns: theoretical rise pre-tipping (slowing of fluctuations); on financial returns the signal is much weaker than in physical systems (returns are near-white).
