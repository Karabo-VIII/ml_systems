# F41 18-Engine Multi-Regime Basket Sim (2026-05-23T13:50)

## Basket members: 18 OOS-validated engines covering bull/chop/BEAR
- LINK | RSI_threshold | p_5_lo_40_hi_60 | chop
- LINK | RSI_threshold | p_8_lo_40_hi_60 | chop
- LINK | RSI_threshold | p_5_lo_35_hi_60 | chop
- ZEC | measure_engines/te_btc_imb | op_abs_gt_thr_1.0 | bear
- FET | measure_engines/te_imb | op_abs_gt_thr_1.0 | bear
- APT | RSI_threshold | p_5_lo_40_hi_65 | bull
- APT | RSI_threshold | p_13_lo_35_hi_65 | bull
- APT | RSI_threshold | p_27_lo_40_hi_60 | bull
- JST | RSI_threshold | p_7_lo_40_hi_75 | chop
- JST | RSI_threshold | p_10_lo_40_hi_70 | chop
- JST | RSI_threshold | p_6_lo_40_hi_80 | chop
- JST | RSI_threshold | p_9_lo_40_hi_70 | chop
- DASH | OBV_zscore | p_100_t_1.5 | chop
- PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop
- PEPE | Donchian_state_above_midline | period_55 | chop
- ARB | measure_engines/bs_basis_z30 | op_abs_gt_thr_1.0 | bull
- FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull
- HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull

## Methodology
- top-3 picks/day by n_engines firing per (asset, date), 25% sizing each, 1d hold
- Simplified signals (60-day z-score for measure_engines, RSI cross-up, OBV z-score abs, Donchian midline)
- Per-bucket maker cost (DEGEN 44bp, BLUE 28bp, STEADY/VOLATILE 32-36bp)

## Results per window

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 703 | +0.022 | 0.16 | 49.5 | +0.1 | -56.8 |
| VAL | 208 | -0.086 | -0.65 | 44.7 | -20.0 | -33.9 |
| OOS_pre | 169 | -0.032 | -0.21 | 48.5 | -9.8 | -28.4 |
| UNSEEN | 89 | +0.211 | 1.71 | 53.9 | +18.7 | -15.8 |
| FULL_POST_TRAIN | 466 | -0.009 | -0.07 | 47.9 | -14.3 | -45.6 |