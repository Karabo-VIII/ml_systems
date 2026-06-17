# Oracle Engine Catalog v0 -- Summary (2026-05-23T14:23)

Total candidate engines evaluated: **1,689**
Engines passing ship-gate: **1689**

## Top 20 engines (passed gate, sorted by compound_return_pct)

| asset | indicator_class | indicator_config | btc_regime_30d | hold_days | n_fires | hit_rate | expectancy_pct | compound_return_pct | max_dd_pct | shic_ratio | 3fold_sign_consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SUPER | Distance_z_state | period_20_threshold_1.0 | bull | 1 | 29 | 0.586 | 5.599 | 316.5 | -12.681 | 0.207 | True |
| ICP | measure_engines/xrel_wh_whale_net_usd_xrank | op_abs_gt_thr_1.0 | bull | 1 | 21 | 0.714 | 6.415 | 237.8 | -5.702 | 0.208 | True |
| WLD | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 3 | 10 | 0.700 | 14.873 | 236.4 | -4.286 | 0.278 | True |
| SUPER | measure_engines/te_out | op_abs_gt_thr_1.0 | bull | 3 | 11 | 0.636 | 12.842 | 225.5 | -10.477 | 0.287 | True |
| NEAR | measure_engines/s3_oi_usd | op_abs_gt_thr_1.0 | bull | 1 | 26 | 0.731 | 4.809 | 215.6 | -13.830 | 0.162 | True |
| NEAR | measure_engines/s3_oi_usd | op_gt_thr_1.0 | bull | 1 | 26 | 0.731 | 4.809 | 215.6 | -13.830 | 0.162 | True |
| SUPER | measure_engines/te_imb | op_gt_thr_1.0 | bull | 1 | 18 | 0.667 | 6.967 | 207.5 | -4.978 | 0.252 | True |
| SUPER | measure_engines/te_in | op_gt_thr_1.0 | bull | 1 | 16 | 0.688 | 7.630 | 197.8 | -4.978 | 0.216 | True |
| WLD | measure_engines/te_imb | op_abs_gt_thr_1.5 | bull | 1 | 13 | 0.692 | 10.046 | 197.4 | -13.455 | 0.208 | True |
| SHIB | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 1 | 11 | 0.636 | 10.759 | 170.6 | -1.969 | 0.274 | True |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 1 | 22 | 0.591 | 5.025 | 167.0 | -5.702 | 0.242 | True |
| PEPE | measure_engines/xrel_rv_rv_5m_xratio | op_abs_gt_thr_1.0 | bull | 1 | 15 | 0.667 | 7.495 | 165.4 | -10.704 | 0.162 | True |
| ICP | measure_engines/bd_depth_l1pct_mean | op_abs_gt_thr_1.0 | bull | 1 | 20 | 0.650 | 5.379 | 161.0 | -6.058 | 0.249 | True |
| WLD | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | bull | 1 | 11 | 0.818 | 9.914 | 158.4 | -1.437 | 0.159 | True |
| FET | measure_engines/te_out_btc | op_abs_gt_thr_1.0 | bull | 3 | 10 | 0.700 | 10.462 | 155.8 | -4.789 | 0.242 | True |
| AR | measure_engines/xrel_hbr_eta_total_xrank | op_abs_gt_thr_1.0 | bull | 1 | 18 | 0.667 | 5.637 | 149.3 | -12.534 | 0.157 | True |
| SEI | OBV_zscore | p_30_t_1.0 | chop | 1 | 21 | 0.762 | 4.704 | 147.7 | -12.095 | 0.230 | True |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_1.5 | bull | 1 | 10 | 0.600 | 10.711 | 142.5 | -3.461 | 0.299 | True |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.5 | bull | 1 | 10 | 0.600 | 10.711 | 142.5 | -3.461 | 0.299 | True |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_2.0 | bull | 1 | 10 | 0.600 | 10.711 | 142.5 | -3.461 | 0.299 | True |

## Engines per indicator class (passed gate)

| indicator_class | n_engines |
|---|---|
| RSI_threshold | 444 |
| OBV_zscore | 48 |
| measure_engines/stbl_usdt_delta_7d_pct | 39 |
| Distance_z_state | 27 |
| MA_state_SMA_above | 26 |
| measure_engines/bd_notional_l1pct_mean | 26 |
| measure_engines/bd_depth_l1pct_mean | 23 |
| measure_engines/s3_top_pos_lsr | 23 |
| measure_engines/te_out | 23 |
| measure_engines/te_btc_imb | 21 |
| measure_engines/bd_depth_l1pct_p90 | 21 |
| measure_engines/s3_oi_usd | 21 |
| measure_engines/s3_smart_vs_retail | 20 |
| measure_engines/xd_cross_return_mean | 20 |
| measure_engines/stbl_total_delta_30d_pct | 20 |
| measure_engines/norm_ma_distance | 20 |
| Kyle_lambda_threshold | 20 |
| MA_state_EMA_above | 19 |
| measure_engines/s3_top_pos_lsr_xsec_z | 19 |
| measure_engines/norm_fd_close | 19 |
| measure_engines/te_imb | 19 |
| measure_engines/xd_momentum_rank | 18 |
| Bollinger_band_breach | 17 |
| measure_engines/xd_ma_distance | 17 |
| measure_engines/te_out_btc | 17 |
| measure_engines/norm_efficiency | 16 |
| measure_engines/s3_global_lsr_z | 16 |
| measure_engines/stbl_total_delta_7d_pct | 16 |
| measure_engines/norm_flow_persistence | 15 |
| VPIN_threshold | 15 |
| measure_engines/xrel_hbr_eta_total_xrank | 14 |
| measure_engines/te_in_btc | 14 |
| measure_engines/norm_return_16 | 14 |
| measure_engines/s3_global_lsr | 14 |
| measure_engines/hbr_n_trades | 14 |
| measure_engines/norm_vol_cluster | 13 |
| measure_engines/norm_perm_entropy | 13 |
| measure_engines/bd_total_depth_l5_p10 | 13 |
| measure_engines/hbr_eta_total | 13 |
| measure_engines/hbr_eta_buy | 13 |
| measure_engines/stbl_usdt_zscore_30d | 13 |
| measure_engines/te_in | 13 |
| measure_engines/s3_smart_vs_retail_z | 12 |
| measure_engines/norm_return_4 | 12 |
| measure_engines/norm_deviation | 12 |
| measure_engines/xd_btc_return | 12 |
| measure_engines/norm_flow_imbalance | 12 |
| measure_engines/xrel_wh_whale_net_usd_xrank | 12 |
| measure_engines/norm_oi_price_divergence | 11 |
| measure_engines/norm_momentum_accel | 11 |
| ATR_bands | 11 |
| measure_engines/bs_basis_delta_3d | 11 |
| ETF_flow_z | 11 |
| measure_engines/hbr_eta_sell | 10 |
| measure_engines/bd_total_depth_l5_mean | 10 |
| measure_engines/xrel_hbr_n_trades_xratio | 10 |
| Donchian_state_above_midline | 10 |
| measure_engines/xrel_hbr_eta_total_xratio | 10 |
| measure_engines/liq_long_xsec_z | 9 |
| measure_engines/xd_funding_spread | 9 |
| measure_engines/s3_top_acct_lsr | 9 |
| MACD_threshold | 9 |
| measure_engines/s3_top_pos_lsr_z | 9 |
| measure_engines/liq_short_xsec_z | 9 |
| measure_engines/norm_return_1 | 9 |
| measure_engines/bs_basis_delta_1d | 9 |
| measure_engines/wh_whale_sell_usd | 8 |
| measure_engines/bs_basis_z30 | 8 |
| measure_engines/norm_cs_spread | 8 |
| Liquidation_cascade | 8 |
| measure_engines/norm_tick_count | 8 |
| measure_engines/rv_bpv_5m | 7 |
| measure_engines/rv_jump_frac | 7 |
| measure_engines/xrel_liq_long_usd_xrank | 7 |
| measure_engines/xrel_rv_bpv_5m_xratio | 7 |
| measure_engines/bs_basis_xsec_z | 6 |
| measure_engines/wh_whale_trade_count_500k | 6 |
| measure_engines/bd_notional_skew | 6 |
| measure_engines/xd_btc_volatility | 6 |
| measure_engines/norm_vol_ratio | 6 |
| measure_engines/norm_vol_price_corr | 6 |
| measure_engines/xrel_rv_rv_5m_xratio | 6 |
| measure_engines/bd_imbalance_l1 | 5 |
| measure_engines/norm_hl_spread | 5 |
| measure_engines/norm_spread_bps | 5 |
| measure_engines/norm_return_kurtosis | 5 |
| measure_engines/norm_bar_duration | 5 |
| measure_engines/rv_rv_5m | 5 |
| measure_engines/xd_cross_vol_mean | 4 |
| measure_engines/wh_whale_net_usd | 4 |
| VWAP_state_above | 4 |
| measure_engines/xrel_wh_whale_net_usd_xratio | 4 |
| measure_engines/bd_imbalance_l5 | 4 |
| measure_engines/rv_jump_count | 4 |
| measure_engines/norm_log_volume | 4 |
| measure_engines/norm_oi_change | 4 |
| measure_engines/s3_smart_bullish | 4 |
| measure_engines/bs_basis_pct | 3 |
| measure_engines/s3_taker_lsr | 3 |
| measure_engines/liq_delta_usd | 3 |
| measure_engines/liq_short_z30 | 3 |
| measure_engines/liq_long_usd | 3 |
| measure_engines/liq_short_usd | 3 |
| Donchian_breakout | 2 |
| measure_engines/xrel_liq_long_usd_xratio | 2 |
| measure_engines/norm_hawkes_sell_intensity | 2 |
| measure_engines/s3_smart_bearish | 2 |
| measure_engines/stbl_total_zscore_30d | 2 |
| measure_engines/wh_whale_buy_usd | 2 |
| measure_engines/stbl_usdc_zscore_30d | 2 |
| Hawkes_branching_imbalance | 1 |
| confluence_engines/UNI_pair_4 | 1 |
| measure_engines/norm_funding_momentum | 1 |
| measure_engines/norm_hawkes_intensity | 1 |
| YZ_vol_regime | 1 |
| measure_engines/liq_long_z30 | 1 |

## Engines per asset (passed gate)

| asset | n_engines |
|---|---|
| XRP | 108 |
| ENJ | 102 |
| HBAR | 69 |
| JST | 61 |
| ALGO | 57 |
| LINK | 56 |
| SUI | 51 |
| APT | 48 |
| SHIB | 47 |
| CHZ | 46 |
| OP | 46 |
| NEAR | 46 |
| ICP | 46 |
| DASH | 45 |
| DOT | 44 |
| FET | 43 |
| SUPER | 42 |
| DYDX | 41 |
| UNI | 40 |
| LTC | 39 |
| ETC | 39 |
| PEPE | 37 |
| ARB | 35 |
| FIL | 34 |
| SOL | 33 |
| AR | 33 |
| SEI | 32 |
| ADA | 32 |
| LDO | 31 |
| AAVE | 30 |

## Engines per hold (passed gate)

| hold_days | n_engines |
|---|---|
| 1d | 1505 |
| 3d | 184 |
