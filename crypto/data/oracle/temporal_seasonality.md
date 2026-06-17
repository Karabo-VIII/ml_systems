# Temporal Seasonality of Catch-Tier Fires (2026-05-23T09:47)

- TRAIN cutoff: 2024-05-15
- Catch-tier engines: 234
- Total fire rows: 71533

## By day of week

| DoW | n_fires | mean_pnl% | median_pnl% | hit_rate% |
|---|---:|---:|---:|---:|
| Monday | 11180 | +0.567 | +0.906 | 53.2 |
| Tuesday | 10671 | -0.273 | -0.120 | 49.5 |
| Wednesday | 10000 | +0.543 | +0.976 | 54.7 |
| Thursday | 10170 | -0.057 | +0.455 | 52.2 |
| Friday | 10331 | +0.139 | +0.590 | 51.6 |
| Saturday | 9493 | +0.478 | +0.863 | 55.9 |
| Sunday | 9688 | +0.204 | +0.516 | 52.6 |

## By month of year

| Month | n_fires | mean_pnl% | median_pnl% | hit_rate% |
|---|---:|---:|---:|---:|
| January | 8150 | -0.340 | -0.466 | 48.1 |
| February | 7225 | +0.853 | +1.059 | 57.0 |
| March | 8067 | +0.731 | +0.873 | 52.4 |
| April | 5904 | -0.024 | +1.004 | 55.6 |
| May | 1551 | +0.627 | +0.982 | 57.1 |
| June | 0 | +nan | +nan | nan |
| July | 6702 | -0.129 | +0.041 | 50.2 |
| August | 5038 | -0.688 | -0.144 | 48.5 |
| September | 5887 | -0.192 | +0.193 | 50.8 |
| October | 8484 | +0.437 | +0.784 | 55.2 |
| November | 7318 | +0.522 | +0.990 | 55.1 |
| December | 7207 | +0.562 | +0.905 | 52.7 |

## By quarter

| Q | n_fires | mean_pnl% | median_pnl% | hit_rate% |
|---|---:|---:|---:|---:|
| Q1 | 23442 | +0.396 | +0.665 | 52.3 |
| Q2 | 7455 | +0.111 | +0.984 | 55.9 |
| Q3 | 17627 | -0.310 | -0.040 | 49.9 |
| Q4 | 23009 | +0.503 | +0.905 | 54.4 |

## By year (sanity)

| year | n_fires | mean_pnl% | hit_rate% |
|---|---:|---:|---:|
| 2023 | 40636 | +0.150 | 52.4 |
| 2024 | 30897 | +0.327 | 53.2 |

## Top indicator classes with strongest DoW spread

| class | Mon | Tue | Wed | Thu | Fri | Sat | Sun | max-min spread |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| measure_engines/liq_long_xsec_z | +0.62 | +0.02 | -4.52 | -3.81 | -29.33 | -1.18 | -1.34 | +29.95 |
| measure_engines/rv_bpv_5m | +14.26 | +5.14 | +4.99 | -0.66 | +9.53 | +8.73 | +1.36 | +14.92 |
| measure_engines/bd_imbalance_l1 | +3.62 | -1.07 | +4.08 | +2.03 | -0.16 | +11.76 | +3.28 | +12.83 |
| measure_engines/stbl_total_zscore_30d | +2.68 | -4.50 | -8.93 | +1.52 | +2.74 | +0.42 | -1.36 | +11.67 |
| measure_engines/norm_deviation | +0.79 | -1.00 | -1.02 | -9.22 | +1.29 | -5.08 | -1.47 | +10.51 |
| YZ_vol_regime | +2.76 | -0.95 | -4.28 | -1.14 | +6.22 | +0.34 | +0.37 | +10.50 |
| confluence_engines/UNI_pair_4 | -1.37 | -3.50 | -6.35 | -1.46 | +1.97 | +2.94 | -0.24 | +9.30 |
| measure_engines/rv_rv_5m | +8.12 | +7.66 | +5.84 | -1.14 | +3.56 | +3.91 | +0.73 | +9.26 |
| measure_engines/wh_whale_trade_count_500k | +1.25 | -4.13 | +5.13 | +3.12 | -0.08 | +2.42 | +3.77 | +9.26 |
| measure_engines/liq_short_z30 | +1.36 | -2.14 | +3.86 | +6.23 | +5.17 | +1.62 | +0.93 | +8.37 |
| measure_engines/wh_whale_net_usd | +4.54 | +3.51 | -2.46 | +0.90 | +2.39 | -3.68 | +2.61 | +8.22 |
| measure_engines/hbr_eta_buy | +0.48 | -2.20 | -1.55 | -0.67 | +2.72 | +5.52 | +0.70 | +7.72 |
| measure_engines/liq_long_usd | +2.55 | -3.21 | +4.05 | +2.70 | +1.28 | +3.44 | -2.79 | +7.26 |
| measure_engines/te_imb | +3.16 | +1.11 | +1.48 | -0.80 | +2.60 | +5.82 | -0.13 | +6.63 |
| measure_engines/rv_jump_count | -0.68 | -1.06 | +5.30 | +0.56 | +2.39 | +0.87 | -0.34 | +6.36 |

## Headline

- Best day-of-week: **Monday** (mean +0.567%)
- Worst day-of-week: Tuesday (mean -0.273%)
- DoW spread (best - worst): +0.840pp
- Best month: **February** (mean +0.853%)
- Worst month: August (mean -0.688%)
- Month spread (best - worst): +1.541pp

- DoW-EFFECT IS PRESENT: fires on Monday earn +0.84pp more than Tuesday. Consider DoW-aware sizing.
- MONTHLY SEASONALITY PRESENT: February vs August gap of +1.54pp. Consider seasonality-aware sizing.