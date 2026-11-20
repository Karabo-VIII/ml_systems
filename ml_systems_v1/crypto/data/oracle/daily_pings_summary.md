# Daily Pings -- 2026-05-23T01:05

Total ping rows: 887
Distinct dates: 471
Distinct assets: 4
Date range: 2024-05-16 -> 2026-05-19

## By engine_family
- ta_state: 558 ping rows
- measure_event: 329 ping rows

## By OOS status
- ok: 887 ping rows

## Top-10 dates by n_pings (highest-conviction days)
| date | n_pings |
|---|---:|
| 2024-10-14 | 4 |
| 2024-06-12 | 4 |
| 2024-05-18 | 4 |
| 2026-01-06 | 4 |
| 2026-01-07 | 4 |
| 2024-12-17 | 4 |
| 2024-10-02 | 4 |
| 2024-06-01 | 4 |
| 2024-10-03 | 4 |
| 2024-06-15 | 4 |

## Top-10 assets by n_pings
| asset | n_pings |
|---|---:|
| PEPE | 558 |
| HBAR | 120 |
| SUI | 110 |
| XRP | 99 |

## Sample 5 ping rows (most recent dates)

- **2026-05-19**: LONG HBAR | measure_event:measure_engines/norm_deviation(op_gt_thr_1.0) hold=1d | TRAIN-only (conviction=17.580)
- **2026-05-18**: LONG HBAR | measure_event:measure_engines/norm_deviation(op_gt_thr_1.0) hold=1d | TRAIN-only (conviction=17.580)
- **2026-05-18**: LONG XRP | measure_event:measure_engines/hbr_eta_buy(op_gt_thr_1.0) hold=1d | TRAIN-only (conviction=22.207)
- **2026-05-17**: LONG XRP | measure_event:measure_engines/hbr_eta_buy(op_gt_thr_1.0) hold=1d | TRAIN-only (conviction=22.207)
- **2026-05-16**: LONG PEPE | ta_state:MA_state_SMA_above(period_50) hold=1d | TRAIN-only (conviction=15.199)