# UNION Basket (TRAIN-best + VAL-best) Sim (2026-05-23T13:31)

## Basket members (10 engines)
### TRAIN-best (from F31)
- FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull
- PEPE | Donchian_state_above_midline | period_55 | chop
- FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull
- SUPER | measure_engines/norm_deviation | op_abs_gt_thr_1.0 | chop
- WLD | MA_state_SMA_above | period_50 | chop

### VAL-best (from F36)
- LINK | RSI_threshold | p_8_lo_40_hi_60 | chop
- LINK | RSI_threshold | p_5_lo_40_hi_60 | chop
- LINK | RSI_threshold | p_7_lo_40_hi_60 | chop
- DASH | OBV_zscore | p_100_t_1.5 | chop
- ADA | VPIN_threshold | t_1.0 | bull

## Methodology
- top-3 picks/day by n_engines firing, 25% sizing, 1d hold, per-bucket maker cost
- Signal re-derivation: simplified (60-day z-score for measure, midline for Donchian, MA crossover for MA_SMA, RSI lo-cross-up for RSI, std-based VPIN proxy)
- Signals may not match catalog's exact implementation

## Results per window

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN_<=2024-05-15 | 524 | -0.018 | -0.13 | 49.2 | -19.7 | -50.3 |
| VAL_2024-05-16_to_2025-03-15 | 121 | +0.004 | 0.02 | 45.5 | -4.0 | -32.6 |
| OOS_pre_2025-03-16_to_2025-12-31 | 100 | +0.537 | 2.58 | 50.0 | +62.5 | -13.5 |
| UNSEEN_2026-01-01_to_2026-05-19 | 47 | -0.232 | -2.83 | 48.9 | -10.7 | -11.0 |
| FULL_POST_TRAIN | 268 | +0.162 | 0.91 | 47.8 | +39.3 | -32.6 |

## Comparison vs F31 (5-engine TRAIN-best only)

F31 5-engine basket results were: TRAIN +0.206/1.35/+48.8 / VAL -0.029/-0.16/-5.3 / OOS_pre +0.541/2.38/+33.4 / UNSEEN -0.412/-6.54/-10.3