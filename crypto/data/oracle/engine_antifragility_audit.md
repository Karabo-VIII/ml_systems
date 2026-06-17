# Engine Anti-Fragility Audit (Z3, POV-3)
Generated: 2026-05-23 09:32:24
Source: data/oracle/engine_catalog_discovery.parquet (catch-tier subset)
Fire data: runs/oracle_layer3/<class>/event_eval_rows.parquet (fired=True only)
Scope: TRAIN-window only, NO OOS / paper-replay involvement.

## Methodology
Per (asset, indicator_class, indicator_config, btc_regime_30d, cadence) engine:
- Pull fire-bar pnl_post_cost_pct from layer3 event_eval_rows.
- Distributional moments: mean, std, skew, excess kurtosis, p10/p50/p90.
- max_dd_in_fires: drawdown of cumulative fire-pnl sequence.
- convexity_score = mean(pnl | |vol_z| > 1.5) - mean(pnl | |vol_z| < 0.5)
  vol_z is the cross-sectional z of abs_magnitude_pct on the fire date,
  computed within the indicator_class pool (same config + regime + cadence).
- Classification:
  - CONVEX   : skew > 0.5 AND (p90 - p50) > 1.5 * (p50 - p10)
  - CONCAVE  : skew < -0.5 OR  (p50 - p10) > 1.5 * (p90 - p50)
  - SYMMETRIC: otherwise
  - INSUFFICIENT: n_fires < 5 OR no event_eval_rows for this class

## Headline Counts
- n_convex      = 28
- n_concave     = 82
- n_symmetric   = 124
- n_insufficient = 0
- n_total catch-tier audited = 234
- n_classes_missing_layer3   = 0

## Stability Cross-Reference (engine_lifecycle_decay.parquet)
- n_stable (lifecycle-TRAIN thirds)               = 17
- n_BOTH stable AND CONVEX (gold-standard)        = 2
- Gold-standard share of stable engines           = 11.8%

## Gold-Standard Subset (stable AND CONVEX)
| asset | indicator_class | indicator_config | btc_regime_30d | n_fires | mean_pnl_pct | skew_pnl | convexity_score | max_dd_in_fires |
|---|---|---|---|---|---|---|---|---|
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 106 | 1.689 | 0.671 | 4.584 | -41.218 |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 106 | 1.689 | 0.671 | 4.549 | -41.218 |

## Top 30 CONVEX (anti-fragile candidates, sorted by convexity_score desc)
| asset | indicator_class | indicator_config | btc_regime_30d | n_fires | mean_pnl_pct | skew_pnl | p10_pnl | p50_pnl | p90_pnl | convexity_score | stable_flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.5 | bull | 76 | 7.849 | 2.018 | -3.946 | 2.690 | 34.284 | 39.677 | False |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_1.5 | bull | 76 | 7.849 | 2.018 | -3.946 | 2.690 | 34.284 | 39.677 | False |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_2.0 | bull | 66 | 9.403 | 1.825 | -3.946 | 3.247 | 34.455 | 38.627 | False |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_2.0 | bull | 66 | 9.403 | 1.825 | -3.946 | 3.247 | 34.455 | 38.627 | False |
| SUPER | measure_engines/te_imb | op_gt_thr_1.0 | bull | 144 | 4.935 | 3.434 | -4.107 | 2.313 | 13.650 | 20.023 | False |
| SUPER | OBV_zscore | p_50_t_1.0 | bull | 176 | 5.044 | 2.259 | -6.017 | 2.851 | 21.249 | 17.793 | False |
| CHZ | measure_engines/xd_funding_spread | op_abs_gt_thr_1.0 | bear | 53 | 2.441 | 2.103 | -1.782 | 0.974 | 12.723 | 14.614 | False |
| BCH | measure_engines/rv_jump_count | op_abs_gt_thr_1.0 | bull | 68 | 1.720 | 0.870 | -3.929 | 0.981 | 12.388 | 14.547 | False |
| BCH | measure_engines/rv_jump_count | op_gt_thr_1.0 | bull | 68 | 1.720 | 0.870 | -3.929 | 0.981 | 12.388 | 14.547 | False |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 201 | 3.394 | 1.570 | -5.302 | 2.431 | 14.042 | 13.906 | False |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 186 | 3.469 | 1.514 | -5.302 | 2.431 | 14.042 | 12.029 | False |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.5 | bull | 131 | 1.974 | 0.592 | -6.503 | 0.986 | 12.243 | 8.807 | False |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.5 | bull | 131 | 1.974 | 0.592 | -6.503 | 0.986 | 12.243 | 8.807 | False |
| DASH | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 156 | -0.499 | 0.690 | -4.233 | -1.587 | 3.686 | 8.010 | False |
| PEPE | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.0 | bull | 136 | 2.303 | 1.393 | -10.440 | -1.569 | 19.043 | 7.107 | False |
| AR | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 98 | 2.524 | 0.887 | -7.121 | 0.701 | 17.243 | 6.811 | False |
| ADA | measure_engines/liq_short_z30 | op_gt_thr_1.0 | bull | 106 | 3.191 | 1.554 | -2.242 | 2.276 | 9.771 | 5.221 | False |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 106 | 1.689 | 0.671 | -5.972 | 1.183 | 13.151 | 4.584 | True |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 106 | 1.689 | 0.671 | -5.972 | 1.183 | 13.151 | 4.549 | True |
| ARKM | OBV_zscore | p_30_t_1.0 | chop | 153 | -0.443 | 0.781 | -6.617 | -1.511 | 6.944 | 3.797 | False |
| PEPE | RSI_threshold | p_9_lo_40_hi_75 | chop | 154 | 0.830 | 0.905 | -4.827 | -0.130 | 7.077 | 3.414 | False |
| LINK | MACD_threshold | f_12_s_35_g_9 | chop | 379 | 0.382 | 1.254 | -3.734 | -0.882 | 5.043 | 3.320 | False |
| LINK | MACD_threshold | f_12_s_21_g_9 | chop | 379 | 0.469 | 1.222 | -3.606 | -0.863 | 5.043 | 2.968 | False |
| FET | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bull | 162 | 0.648 | 1.398 | -5.793 | -1.724 | 6.883 | 1.575 | False |
| PEPE | RSI_threshold | p_7_lo_35_hi_80 | chop | 119 | 0.947 | 1.153 | -4.616 | -0.130 | 9.767 | 1.448 | False |
| SHIB | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | chop | 166 | 0.031 | 1.015 | -4.302 | -0.852 | 4.526 | -0.100 | False |
| ZEC | measure_engines/bd_imbalance_l5 | op_abs_gt_thr_1.0 | bull | 192 | 0.819 | 0.808 | -3.805 | 0.867 | 8.461 | -1.733 | False |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 112 | 0.441 | 0.788 | -4.639 | 0.376 | 8.483 | -1.852 | False |

## Top 20 CONCAVE (fragile -- exclude from robust basket)
| asset | indicator_class | indicator_config | btc_regime_30d | n_fires | mean_pnl_pct | skew_pnl | p10_pnl | p50_pnl | p90_pnl | convexity_score | max_dd_in_fires |
|---|---|---|---|---|---|---|---|---|---|---|---|
| UNI | measure_engines/liq_long_xsec_z | op_abs_gt_thr_1.0 | bull | 122 | -3.135 | -3.828 | -8.800 | -1.163 | 3.977 | -37.243 | -394.606 |
| JST | RSI_threshold | p_21_lo_35_hi_60 | bull | 141 | -1.286 | -1.698 | -6.989 | -0.965 | 4.535 | -15.485 | -224.077 |
| JST | RSI_threshold | p_22_lo_35_hi_60 | bull | 141 | -1.286 | -1.698 | -6.989 | -0.965 | 4.535 | -15.485 | -224.077 |
| ICP | RSI_threshold | p_6_lo_40_hi_65 | bull | 257 | -2.010 | -2.489 | -8.421 | -0.603 | 4.706 | -14.869 | -629.266 |
| AR | OBV_zscore | p_100_t_1.5 | bull | 89 | -2.486 | -1.770 | -5.997 | -1.475 | 3.113 | -13.128 | -270.746 |
| DASH | MA_state_EMA_above | period_50 | bull | 354 | 0.800 | 0.035 | -5.035 | 1.616 | 4.102 | -12.469 | -98.808 |
| BCH | measure_engines/stbl_total_zscore_30d | op_abs_gt_thr_1.0 | bull | 82 | -0.051 | -1.172 | -6.639 | 2.016 | 5.686 | -11.930 | -130.181 |
| DASH | MA_state_SMA_above | period_50 | bull | 368 | 0.797 | 0.031 | -4.724 | 1.569 | 4.823 | -11.731 | -98.808 |
| ICP | RSI_threshold | p_5_lo_40_hi_70 | bull | 210 | -1.547 | -2.673 | -8.288 | 0.559 | 4.715 | -11.563 | -410.058 |
| LTC | RSI_threshold | p_5_lo_35_hi_70 | bear | 79 | 0.147 | -2.274 | -3.165 | 1.106 | 3.011 | -11.300 | -57.027 |
| ICP | RSI_threshold | p_5_lo_40_hi_75 | bull | 158 | -1.018 | -2.854 | -7.935 | 1.100 | 4.822 | -11.283 | -280.559 |
| LTC | RSI_threshold | p_6_lo_40_hi_70 | bear | 97 | 0.015 | -2.214 | -3.243 | 0.982 | 2.943 | -10.914 | -57.027 |
| LTC | RSI_threshold | p_7_lo_40_hi_70 | bear | 90 | -0.102 | -2.080 | -3.359 | 0.646 | 2.943 | -10.914 | -57.027 |
| LTC | RSI_threshold | p_8_lo_40_hi_65 | bear | 90 | -0.096 | -2.084 | -3.359 | 0.721 | 2.943 | -10.910 | -57.027 |
| LTC | RSI_threshold | p_5_lo_40_hi_75 | bear | 97 | -0.262 | -2.119 | -3.243 | 0.982 | 2.943 | -10.856 | -57.027 |
| LTC | RSI_threshold | p_6_lo_40_hi_75 | bear | 97 | 0.015 | -2.214 | -3.243 | 0.982 | 2.943 | -10.791 | -57.027 |
| LTC | RSI_threshold | p_5_lo_40_hi_70 | bear | 97 | -0.262 | -2.119 | -3.243 | 0.982 | 2.943 | -10.408 | -57.027 |
| LTC | RSI_threshold | p_5_lo_30_hi_70 | bear | 53 | -0.226 | -2.252 | -3.359 | 1.099 | 2.943 | -10.375 | -43.844 |
| LTC | RSI_threshold | p_5_lo_30_hi_75 | bear | 53 | -0.226 | -2.252 | -3.359 | 1.099 | 2.943 | -10.375 | -43.844 |
| LTC | RSI_threshold | p_5_lo_35_hi_65 | bear | 87 | 0.265 | -2.099 | -3.165 | 1.099 | 3.282 | -10.232 | -57.027 |

## Per Indicator-Class Tally
| indicator_class | SYMMETRIC | CONCAVE | CONVEX |
|---|---|---|---|
| ATR_bands | 3 | 0 | 0 |
| Bollinger_band_breach | 1 | 0 | 0 |
| Distance_z_state | 10 | 1 | 0 |
| Donchian_state_above_midline | 4 | 2 | 0 |
| ETF_flow_z | 4 | 0 | 0 |
| Hawkes_branching_imbalance | 1 | 0 | 0 |
| Kyle_lambda_threshold | 0 | 4 | 0 |
| Liquidation_cascade | 0 | 1 | 0 |
| MACD_threshold | 5 | 0 | 2 |
| MA_state_EMA_above | 7 | 3 | 0 |
| MA_state_SMA_above | 10 | 3 | 0 |
| OBV_zscore | 3 | 6 | 2 |
| RSI_threshold | 25 | 30 | 2 |
| VPIN_threshold | 3 | 1 | 0 |
| VWAP_state_above | 2 | 0 | 0 |
| YZ_vol_regime | 1 | 0 | 0 |
| confluence_engines/UNI_pair_4 | 0 | 1 | 0 |
| measure_engines/bd_imbalance_l1 | 1 | 0 | 0 |
| measure_engines/bd_imbalance_l5 | 2 | 0 | 1 |
| measure_engines/bs_basis_z30 | 1 | 4 | 0 |
| measure_engines/hbr_eta_buy | 3 | 0 | 0 |
| measure_engines/hbr_eta_total | 1 | 1 | 0 |
| measure_engines/liq_long_usd | 1 | 0 | 2 |
| measure_engines/liq_long_xsec_z | 0 | 1 | 0 |
| measure_engines/liq_short_z30 | 0 | 0 | 1 |
| measure_engines/norm_deviation | 2 | 1 | 0 |
| measure_engines/norm_efficiency | 1 | 5 | 0 |
| measure_engines/norm_flow_imbalance | 4 | 1 | 1 |
| measure_engines/rv_bpv_5m | 2 | 0 | 4 |
| measure_engines/rv_jump_count | 0 | 2 | 2 |
| measure_engines/rv_jump_frac | 0 | 3 | 0 |
| measure_engines/rv_rv_5m | 2 | 0 | 1 |
| measure_engines/stbl_total_zscore_30d | 0 | 2 | 0 |
| measure_engines/te_btc_imb | 4 | 3 | 0 |
| measure_engines/te_imb | 2 | 1 | 1 |
| measure_engines/te_in_btc | 4 | 0 | 0 |
| measure_engines/wh_whale_net_usd | 2 | 1 | 1 |
| measure_engines/wh_whale_trade_count_500k | 1 | 0 | 5 |
| measure_engines/xd_btc_return | 2 | 0 | 1 |
| measure_engines/xd_btc_volatility | 2 | 1 | 0 |
| measure_engines/xd_funding_spread | 1 | 3 | 1 |
| measure_engines/xd_ma_distance | 2 | 1 | 0 |
| measure_engines/xd_momentum_rank | 5 | 0 | 1 |

## Per Regime Tally
| btc_regime_30d | CONCAVE | CONVEX | SYMMETRIC |
|---|---|---|---|
| bear | 17 | 1 | 3 |
| bull | 30 | 21 | 62 |
| chop | 35 | 6 | 59 |

## Caveats (read before trusting)
- TRAIN-only. No OOS / paper-replay validation. A CONVEX classification on
  TRAIN does not guarantee CONVEX on OOS; the LIFECYCLE_DECAY sibling shows
  only 17/234 catch-tier engines are even temporally stable across TRAIN thirds.
- Skewness on n < 50 fires is noisy (sample skew has standard error ~ sqrt(6/n)).
  Treat CONVEX tags with n_fires < 50 as suggestive, not confirmatory.
- vol_z uses cross-sectional abs_magnitude_pct of co-firing assets in the same
  class+config+regime+cadence pool. When that pool has < 2 fires on a date,
  vol_z is undefined and the fire is excluded from the convexity computation.
- A handful of catch-tier engines belong to indicator_classes with no
  runs/oracle_layer3/<class>/event_eval_rows.parquet on disk; those are
  tagged INSUFFICIENT and skipped from the tallies above.
- The 'CONVEX skew > 0.5 AND tail-asymmetry > 1.5' threshold is judgment,
  not calibrated to a target Type-I/II rate; cross-validate before deploy.
- max_dd_in_fires is computed on the fire-sequence ordered by event_eval_rows
  default order; if fires are not chronologically sorted within (asset,cfg,regime)
  this is a regime-conditional drawdown proxy, not a strict time-series DD.
