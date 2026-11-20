# F41 18-engine basket -- CSCV PBO test

**Probability of Backtest Overfitting (PBO)**: across all (M choose M/2)=924 time-block splits, the IS-best engine ranks BELOW median on OOS in **74.0%** of splits.

Interpretation:
- PBO < 0.25: selection is robust (catalog mining picks generalizable winners)
- PBO 0.25-0.50: selection has some signal but is fragile
- PBO > 0.50: selection is no better than chance (full overfitting)

**F41 result: PBO = 0.7403**

VERDICT: OVERFIT — F41 selection methodology is no better than chance.

Full-window Sharpe per engine:

|                                                                          |   sharpe |
|:-------------------------------------------------------------------------|---------:|
| LINK__RSI_threshold__p_5_lo_40_hi_60__chop__h3                           | 1.41247  |
| LINK__RSI_threshold__p_5_lo_35_hi_60__chop__h3                           | 1.35228  |
| LINK__RSI_threshold__p_8_lo_40_hi_60__chop__h3                           | 1.12196  |
| JST__RSI_threshold__p_7_lo_40_hi_75__chop__h1                            | 0.946487 |
| APT__RSI_threshold__p_5_lo_40_hi_65__bull__h3                            | 0.806628 |
| JST__RSI_threshold__p_6_lo_40_hi_80__chop__h1                            | 0.800995 |
| ZEC__measure_engines/te_btc_imb__op_abs_gt_thr_1.0__bear__h1             | 0.788342 |
| JST__RSI_threshold__p_10_lo_40_hi_70__chop__h1                           | 0.77078  |
| APT__RSI_threshold__p_13_lo_35_hi_65__bull__h1                           | 0.761543 |
| FET__measure_engines/wh_whale_net_usd__op_abs_gt_thr_2.0__bull__h1       | 0.708855 |
| JST__RSI_threshold__p_9_lo_40_hi_70__chop__h1                            | 0.619512 |
| DASH__OBV_zscore__p_100_t_1.5__chop__h1                                  | 0.604604 |
| ARB__measure_engines/bs_basis_z30__op_abs_gt_thr_1.0__bull__h1           | 0.601014 |
| PEPE__RSI_threshold__p_10_lo_40_hi_80__chop__h3                          | 0.557322 |
| FET__measure_engines/te_imb__op_abs_gt_thr_1.0__bear__h1                 | 0.497858 |
| APT__RSI_threshold__p_27_lo_40_hi_60__bull__h1                           | 0.376579 |
| HBAR__measure_engines/wh_whale_trade_count_500k__op_gt_thr_1.0__bull__h1 | 0.2884   |