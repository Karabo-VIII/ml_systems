# Red-Team Exploitation Surface (Y3) — TRAIN

Per-engine 6-component adversarial vulnerability score.
- Catch-tier engines scored: **234**
- TRAIN window: <= 2024-05-15

## Components
- Y3.1 calendar Gini: fires concentrated in specific DOW/month (0=uniform, 1=clustered)
- Y3.2 wf-cov instability: 1 - wf_cov_stability (low fold stability = predictable surface)
- Y3.3 size-discoverable: our $10k notional vs 5min ADV proxy (vol_usd/288)
- Y3.4 intraday signature: **N/A on 1d cadence** (all 234 engines are 1d)
- Y3.5 fee fragility: maker_expectancy minus taker_expectancy gap
- Y3.6 cancel-spam survival: **DATA GAP — no cancel-ratio in v51 dataset**, skipped

Composite = mean of Y3.1, Y3.2, Y3.3, Y3.5 (Y3.4 and Y3.6 excluded).

## Component score distributions

| Component | p10 | p50 | p90 | mean |
|---|---:|---:|---:|---:|
| y31_calendar_gini | 0.090 | 0.157 | 0.237 | 0.163 |
| y32_instability | 0.612 | 0.879 | 1.000 | 0.840 |
| y33_size_discoverable | 1.000 | 1.000 | 1.000 | 0.992 |
| y35_fee_fragility | 0.100 | 0.160 | 0.160 | 0.142 |
| composite_vulnerability | 0.472 | 0.549 | 0.582 | 0.534 |

## Top 30 MOST VULNERABLE (deploy EXCLUSION candidates)

| Rank | engine_id | composite | Y3.1 | Y3.2 | Y3.3 | Y3.5 | fee_fragile |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | SHIB__measure_engines/rv_bpv_5m__op_abs_gt_thr_2.0__rbull__c2__mall__h1d | 0.621 | 0.322 | 1.000 | 1.000 | 0.160 | N |
| 2 | SHIB__measure_engines/rv_bpv_5m__op_gt_thr_2.0__rbull__c2__mall__h1d | 0.621 | 0.322 | 1.000 | 1.000 | 0.160 | N |
| 3 | NEAR__measure_engines/wh_whale_trade_count_500k__op_gt_thr_1.0__rbull__c5__mall__h1d | 0.613 | 0.330 | 0.961 | 1.000 | 0.160 | N |
| 4 | SHIB__measure_engines/rv_bpv_5m__op_abs_gt_thr_1.5__rbull__c2__mall__h1d | 0.610 | 0.280 | 1.000 | 1.000 | 0.160 | N |
| 5 | SHIB__measure_engines/rv_bpv_5m__op_gt_thr_1.5__rbull__c2__mall__h1d | 0.610 | 0.280 | 1.000 | 1.000 | 0.160 | N |
| 6 | NEAR__measure_engines/wh_whale_trade_count_500k__op_abs_gt_thr_1.0__rbull__c5__mall__h1d | 0.603 | 0.280 | 0.971 | 1.000 | 0.160 | N |
| 7 | DOT__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h1d | 0.600 | 0.240 | 1.000 | 1.000 | 0.160 | N |
| 8 | ADA__measure_engines/liq_short_z30__op_gt_thr_1.0__rbull__c1__mall__h1d | 0.599 | 0.254 | 1.000 | 1.000 | 0.140 | N |
| 9 | FIL__measure_engines/liq_long_usd__op_abs_gt_thr_1.0__rbull__c2__mall__h1d | 0.596 | 0.283 | 1.000 | 1.000 | 0.100 | N |
| 10 | FIL__measure_engines/liq_long_usd__op_gt_thr_1.0__rbull__c2__mall__h1d | 0.596 | 0.283 | 1.000 | 1.000 | 0.100 | N |
| 11 | ICP__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h1d | 0.595 | 0.221 | 1.000 | 1.000 | 0.160 | N |
| 12 | ICP__measure_engines/te_btc_imb__op_abs_gt_thr_1.0__rbull__c5__mall__h3d | 0.594 | 0.215 | 1.000 | 1.000 | 0.160 | N |
| 13 | ARB__measure_engines/bs_basis_z30__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.593 | 0.213 | 1.000 | 1.000 | 0.160 | N |
| 14 | HBAR__ETF_flow_z__t_0.5__rbull__c4__mall__h1d | 0.593 | 0.231 | 0.982 | 1.000 | 0.160 | N |
| 15 | APT__Liquidation_cascade__t_1.0__rbull__c1__mall__h1d | 0.593 | 0.211 | 1.000 | 1.000 | 0.160 | N |
| 16 | ADA__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h1d | 0.590 | 0.221 | 1.000 | 1.000 | 0.140 | N |
| 17 | HBAR__measure_engines/norm_efficiency__op_gt_thr_1.0__rbull__c1__mall__h1d | 0.587 | 0.188 | 1.000 | 1.000 | 0.160 | N |
| 18 | BNB__measure_engines/te_btc_imb__op_abs_gt_thr_1.5__rbear__c5__mall__h1d | 0.587 | 0.211 | 1.000 | 0.995 | 0.140 | N |
| 19 | TRX__Distance_z_state__period_50_threshold_1.5__rchop__c5__mall__h3d | 0.586 | 0.262 | 0.941 | 1.000 | 0.140 | N |
| 20 | NEAR__measure_engines/xd_btc_volatility__op_abs_gt_thr_1.0__rbull__c5__mall__h3d | 0.584 | 0.177 | 1.000 | 1.000 | 0.160 | N |
| 21 | FLOKI__ETF_flow_z__t_0.5__rbull__c4__mall__h1d | 0.583 | 0.232 | 1.000 | 1.000 | 0.100 | N |
| 22 | SOL__Distance_z_state__period_20_threshold_1.5__rbull__c5__mall__h3d | 0.583 | 0.190 | 1.000 | 1.000 | 0.140 | N |
| 23 | APT__RSI_threshold__p_13_lo_35_hi_65__rbull__c1__mall__h1d | 0.582 | 0.169 | 1.000 | 1.000 | 0.160 | N |
| 24 | PEPE__Hawkes_branching_imbalance__t_0.1__rchop__c4__mall__h1d | 0.582 | 0.220 | 0.947 | 1.000 | 0.160 | N |
| 25 | DYDX__MA_state_EMA_above__period_20__rchop__c1__mall__h1d | 0.582 | 0.167 | 1.000 | 1.000 | 0.160 | N |
| 26 | PEPE__Donchian_state_above_midline__period_100__rchop__c1__mall__h1d | 0.581 | 0.165 | 1.000 | 1.000 | 0.160 | N |
| 27 | ETC__measure_engines/norm_efficiency__op_abs_gt_thr_1.0__rchop__c5__mall__h1d | 0.581 | 0.185 | 1.000 | 1.000 | 0.140 | N |
| 28 | ICP__measure_engines/rv_jump_frac__op_abs_gt_thr_1.0__rchop__c5__mall__h1d | 0.581 | 0.164 | 1.000 | 1.000 | 0.160 | N |
| 29 | DOT__measure_engines/xd_btc_return__op_abs_gt_thr_1.0__rbull__c5__mall__h1d | 0.581 | 0.163 | 1.000 | 1.000 | 0.160 | N |
| 30 | HBAR__measure_engines/xd_funding_spread__op_abs_gt_thr_1.0__rbear__c5__mall__h1d | 0.580 | 0.158 | 1.000 | 1.000 | 0.160 | N |

## Top 30 LEAST VULNERABLE (safer deploy candidates)

| Rank | engine_id | composite | Y3.1 | Y3.2 | Y3.3 | Y3.5 | mean_pnl% | stable |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ICP__measure_engines/te_in_btc__op_abs_gt_thr_1.5__rbull__c1__mall__h1d | 0.363 | 0.260 | 0.034 | 1.000 | 0.160 | 2.35 | N |
| 2 | SOL__MACD_threshold__f_12_s_21_g_9__rchop__c1__mall__h3d | 0.376 | 0.050 | 0.314 | 1.000 | 0.140 | 3.67 | N |
| 3 | SOL__MACD_threshold__f_12_s_26_g_9__rchop__c1__mall__h3d | 0.376 | 0.050 | 0.314 | 1.000 | 0.140 | 3.67 | N |
| 4 | SOL__MACD_threshold__f_8_s_35_g_9__rchop__c1__mall__h3d | 0.376 | 0.050 | 0.314 | 1.000 | 0.140 | 3.67 | N |
| 5 | SOL__MACD_threshold__f_12_s_35_g_9__rchop__c1__mall__h3d | 0.380 | 0.050 | 0.331 | 1.000 | 0.140 | 3.67 | N |
| 6 | LINK__MACD_threshold__f_12_s_21_g_9__rchop__c1__mall__h3d | 0.396 | 0.060 | 0.364 | 1.000 | 0.160 | 2.19 | N |
| 7 | LINK__MACD_threshold__f_12_s_35_g_9__rchop__c1__mall__h3d | 0.396 | 0.060 | 0.364 | 1.000 | 0.160 | 3.53 | N |
| 8 | DASH__OBV_zscore__p_100_t_1.5__rchop__c5__mall__h1d | 0.397 | 0.136 | 0.354 | 1.000 | 0.100 | 0.56 | N |
| 9 | FET__measure_engines/rv_rv_5m__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.412 | 0.220 | 0.328 | 1.000 | 0.100 | 3.38 | Y |
| 10 | FIL__measure_engines/xd_ma_distance__op_abs_gt_thr_1.0__rbull__c5__mall__h1d | 0.412 | 0.120 | 0.430 | 1.000 | 0.100 | 3.29 | N |
| 11 | DASH__measure_engines/xd_momentum_rank__op_abs_gt_thr_1.0__rchop__c5__mall__h1d | 0.415 | 0.121 | 0.439 | 1.000 | 0.100 | 0.63 | N |
| 12 | FET__measure_engines/rv_bpv_5m__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.415 | 0.215 | 0.346 | 1.000 | 0.100 | 3.43 | Y |
| 13 | DASH__OBV_zscore__p_50_t_1.0__rchop__c5__mall__h1d | 0.419 | 0.096 | 0.481 | 1.000 | 0.100 | 0.76 | N |
| 14 | FET__Distance_z_state__period_20_threshold_1.0__rbull__c2__mall__h1d | 0.424 | 0.142 | 0.453 | 1.000 | 0.100 | 3.14 | N |
| 15 | FIL__measure_engines/norm_efficiency__op_abs_gt_thr_1.0__rchop__c5__mall__h1d | 0.430 | 0.152 | 0.468 | 1.000 | 0.100 | 2.10 | N |
| 16 | ICP__RSI_threshold__p_5_lo_40_hi_70__rbull__c1__mall__h1d | 0.439 | 0.086 | 0.511 | 1.000 | 0.160 | 0.68 | N |
| 17 | ICP__measure_engines/bd_imbalance_l1__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.439 | 0.189 | 0.407 | 1.000 | 0.160 | 2.38 | N |
| 18 | ICP__RSI_threshold__p_5_lo_40_hi_75__rbull__c1__mall__h1d | 0.440 | 0.089 | 0.511 | 1.000 | 0.160 | 0.73 | N |
| 19 | SUPER__OBV_zscore__p_50_t_1.0__rbull__c1__mall__h1d | 0.447 | 0.079 | 0.609 | 1.000 | 0.100 | 3.46 | N |
| 20 | JST__RSI_threshold__p_21_lo_35_hi_60__rbull__c1__mall__h1d | 0.449 | 0.186 | 0.510 | 1.000 | 0.100 | 0.98 | N |
| 21 | JST__RSI_threshold__p_22_lo_35_hi_60__rbull__c1__mall__h1d | 0.451 | 0.194 | 0.510 | 1.000 | 0.100 | 0.98 | N |
| 22 | ZEC__measure_engines/bd_imbalance_l5__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.453 | 0.165 | 0.547 | 1.000 | 0.100 | 1.37 | N |
| 23 | ARKM__Kyle_lambda_threshold__t_0.5__rbull__c1__mall__h1d | 0.469 | 0.071 | 0.707 | 1.000 | 0.100 | 3.98 | N |
| 24 | SHIB__measure_engines/xd_momentum_rank__op_abs_gt_thr_1.0__rchop__c1__mall__h1d | 0.471 | 0.104 | 0.621 | 1.000 | 0.160 | 1.09 | N |
| 25 | FET__VPIN_threshold__t_0.5__rbull__c1__mall__h3d | 0.475 | 0.069 | 0.730 | 1.000 | 0.100 | 7.12 | N |
| 26 | ARKM__VPIN_threshold__t_0.5__rchop__c1__mall__h3d | 0.476 | 0.079 | 0.723 | 1.000 | 0.100 | 3.69 | N |
| 27 | LINK__MACD_threshold__f_8_s_21_g_9__rchop__c1__mall__h3d | 0.476 | 0.060 | 0.684 | 1.000 | 0.160 | 1.98 | N |
| 28 | JST__MA_state_SMA_above__period_20__rbull__c5__mall__h1d | 0.477 | 0.154 | 0.652 | 1.000 | 0.100 | 1.36 | N |
| 29 | AAVE__VPIN_threshold__t_0.5__rchop__c4__mall__h3d | 0.477 | 0.101 | 0.646 | 1.000 | 0.160 | 1.10 | N |
| 30 | DASH__measure_engines/xd_momentum_rank__op_abs_gt_thr_1.0__rchop__c1__mall__h1d | 0.482 | 0.121 | 0.708 | 1.000 | 0.100 | 1.13 | N |

## V7 triple-filter (12 engines) x vulnerability

These are the 12 engines in the V7 triple-filter basket (stable + not-concave + |mean_pnl|>=1.5%).
The user's headline: V7 basket = +0.822%/d / Sharpe 3.65 on FIXED data.
Vulnerability flag = composite vulnerability ABOVE the median of all 234 catch-tier engines.

Median composite vulnerability (234 engines): **0.549**

| engine_id | composite | Y3.1 | Y3.2 | Y3.3 | Y3.5 | fee_fragile | size_discov | HIGH_VULN |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| DOT__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h1d | 0.600 | 0.240 | 1.000 | 1.000 | 0.160 | N | 3600.710% | **YES** |
| FIL__measure_engines/liq_long_usd__op_abs_gt_thr_1.0__rbull__c2__mall__h1d | 0.596 | 0.283 | 1.000 | 1.000 | 0.100 | N | 3605.061% | **YES** |
| FIL__measure_engines/liq_long_usd__op_gt_thr_1.0__rbull__c2__mall__h1d | 0.596 | 0.283 | 1.000 | 1.000 | 0.100 | N | 3605.061% | **YES** |
| ICP__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h1d | 0.595 | 0.221 | 1.000 | 1.000 | 0.160 | N | 5768.945% | **YES** |
| AR__measure_engines/te_in_btc__op_abs_gt_thr_1.0__rbull__c5__mall__h3d | 0.574 | 0.138 | 1.000 | 1.000 | 0.160 | N | 19075.792% | **YES** |
| AR__MA_state_SMA_above__period_20__rbull__c4__mall__h1d | 0.569 | 0.184 | 0.931 | 1.000 | 0.160 | N | 19075.792% | **YES** |
| FET__measure_engines/rv_rv_5m__op_gt_thr_1.0__rbull__c1__mall__h1d | 0.547 | 0.271 | 0.815 | 1.000 | 0.100 | N | 4795.323% | no |
| FET__measure_engines/rv_bpv_5m__op_gt_thr_1.0__rbull__c1__mall__h1d | 0.544 | 0.250 | 0.826 | 1.000 | 0.100 | N | 4793.916% | no |
| APT__Distance_z_state__period_50_threshold_1.5__rbull__c5__mall__h3d | 0.542 | 0.194 | 0.813 | 1.000 | 0.160 | N | 3598.297% | no |
| APT__MA_state_EMA_above__period_20__rbull__c5__mall__h3d | 0.524 | 0.227 | 0.707 | 1.000 | 0.160 | N | 3599.114% | no |
| FET__measure_engines/rv_bpv_5m__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.415 | 0.215 | 0.346 | 1.000 | 0.100 | N | 4785.849% | no |
| FET__measure_engines/rv_rv_5m__op_abs_gt_thr_1.0__rbull__c1__mall__h1d | 0.412 | 0.220 | 0.328 | 1.000 | 0.100 | N | 4795.323% | no |

**V7 high-vuln count**: 6/12 — DEPLOY-READINESS FLAG.

## Per-indicator-class vulnerability summary

| indicator_class | n | mean_composite | mean_Y31 | mean_Y32 | mean_Y33 | mean_Y35 | n_fee_fragile |
|---|---:|---:|---:|---:|---:|---:|---:|
| measure_engines/liq_short_z30 | 1 | 0.599 | 0.254 | 1.000 | 1.000 | 0.140 | 0 |
| Liquidation_cascade | 1 | 0.593 | 0.211 | 1.000 | 1.000 | 0.160 | 0 |
| Hawkes_branching_imbalance | 1 | 0.582 | 0.220 | 0.947 | 1.000 | 0.160 | 0 |
| VWAP_state_above | 2 | 0.574 | 0.157 | 1.000 | 1.000 | 0.140 | 0 |
| measure_engines/rv_bpv_5m | 6 | 0.570 | 0.278 | 0.862 | 1.000 | 0.140 | 0 |
| ETF_flow_z | 4 | 0.568 | 0.228 | 0.930 | 1.000 | 0.115 | 0 |
| measure_engines/wh_whale_trade_count_500k | 6 | 0.567 | 0.265 | 0.843 | 1.000 | 0.160 | 0 |
| measure_engines/norm_flow_imbalance | 6 | 0.566 | 0.168 | 0.974 | 0.994 | 0.127 | 0 |
| measure_engines/liq_long_usd | 3 | 0.565 | 0.265 | 0.873 | 1.000 | 0.120 | 0 |
| measure_engines/xd_btc_volatility | 3 | 0.564 | 0.147 | 0.949 | 1.000 | 0.160 | 0 |
| Distance_z_state | 11 | 0.561 | 0.176 | 0.928 | 1.000 | 0.138 | 0 |
| measure_engines/bs_basis_z30 | 5 | 0.558 | 0.142 | 0.930 | 1.000 | 0.160 | 1 |
| measure_engines/rv_jump_frac | 3 | 0.556 | 0.157 | 0.907 | 1.000 | 0.160 | 0 |
| measure_engines/xd_funding_spread | 5 | 0.554 | 0.160 | 0.913 | 1.000 | 0.144 | 0 |
| measure_engines/liq_long_xsec_z | 1 | 0.554 | 0.187 | 0.868 | 1.000 | 0.160 | 0 |
| MA_state_EMA_above | 10 | 0.550 | 0.182 | 0.888 | 0.979 | 0.152 | 0 |
| measure_engines/te_btc_imb | 7 | 0.547 | 0.183 | 0.869 | 0.996 | 0.143 | 0 |
| Donchian_state_above_midline | 6 | 0.547 | 0.178 | 0.874 | 0.965 | 0.173 | 0 |
| measure_engines/te_imb | 4 | 0.546 | 0.162 | 0.904 | 1.000 | 0.120 | 0 |
| measure_engines/wh_whale_net_usd | 4 | 0.543 | 0.267 | 0.775 | 1.000 | 0.130 | 0 |
| RSI_threshold | 57 | 0.541 | 0.131 | 0.900 | 0.999 | 0.135 | 2 |
| measure_engines/norm_efficiency | 6 | 0.536 | 0.172 | 0.827 | 1.000 | 0.143 | 0 |
| measure_engines/hbr_eta_total | 2 | 0.531 | 0.165 | 0.831 | 1.000 | 0.130 | 0 |
| MA_state_SMA_above | 13 | 0.531 | 0.178 | 0.814 | 0.984 | 0.146 | 0 |
| measure_engines/xd_btc_return | 3 | 0.530 | 0.139 | 0.862 | 1.000 | 0.120 | 0 |
| measure_engines/stbl_total_zscore_30d | 2 | 0.525 | 0.239 | 0.768 | 0.895 | 0.200 | 0 |
| measure_engines/hbr_eta_buy | 3 | 0.525 | 0.128 | 0.873 | 1.000 | 0.100 | 0 |
| measure_engines/rv_jump_count | 4 | 0.523 | 0.204 | 0.739 | 1.000 | 0.150 | 0 |
| measure_engines/norm_deviation | 3 | 0.522 | 0.184 | 0.765 | 1.000 | 0.140 | 0 |
| ATR_bands | 3 | 0.519 | 0.181 | 0.845 | 0.790 | 0.260 | 2 |
| confluence_engines/UNI_pair_4 | 1 | 0.517 | 0.198 | 0.711 | 1.000 | 0.160 | 0 |
| measure_engines/rv_rv_5m | 3 | 0.511 | 0.256 | 0.667 | 1.000 | 0.120 | 0 |
| OBV_zscore | 11 | 0.509 | 0.101 | 0.797 | 1.000 | 0.136 | 0 |
| measure_engines/xd_ma_distance | 3 | 0.508 | 0.143 | 0.748 | 1.000 | 0.140 | 0 |
| Kyle_lambda_threshold | 4 | 0.507 | 0.102 | 0.797 | 1.000 | 0.130 | 1 |
| YZ_vol_regime | 1 | 0.501 | 0.130 | 0.773 | 1.000 | 0.100 | 0 |
| Bollinger_band_breach | 1 | 0.500 | 0.174 | 0.708 | 0.979 | 0.140 | 0 |
| measure_engines/xd_momentum_rank | 6 | 0.500 | 0.133 | 0.764 | 0.965 | 0.137 | 0 |
| measure_engines/te_in_btc | 4 | 0.494 | 0.200 | 0.617 | 1.000 | 0.160 | 0 |
| measure_engines/bd_imbalance_l5 | 3 | 0.488 | 0.179 | 0.654 | 1.000 | 0.120 | 0 |
| VPIN_threshold | 4 | 0.482 | 0.098 | 0.704 | 1.000 | 0.125 | 0 |
| measure_engines/bd_imbalance_l1 | 1 | 0.439 | 0.189 | 0.407 | 1.000 | 0.160 | 0 |
| MACD_threshold | 7 | 0.397 | 0.054 | 0.384 | 1.000 | 0.149 | 2 |

## Fee-fragile engines (maker_exp > 0 BUT taker_exp < 0): 8

| engine_id | maker_exp%/trade | taker_exp%/trade | gap |
|---|---:|---:|---:|
| BTC__ATR_bands__p_20_k_1.5__rbull__c1__mall__h3d | 0.0856 | -0.0444 | 0.1300 |
| BTC__ATR_bands__p_20_k_1.5__rbull__c2__mall__h1d | 0.0856 | -0.0444 | 0.1300 |
| PEPE__RSI_threshold__p_9_lo_40_hi_75__rchop__c5__mall__h3d | 0.0257 | -0.0543 | 0.0800 |
| SHIB__measure_engines/bs_basis_z30__op_abs_gt_thr_1.0__rchop__c1__mall__h1d | 0.0639 | -0.0161 | 0.0800 |
| LINK__RSI_threshold__p_8_lo_40_hi_60__rchop__c1__mall__h3d | 0.0652 | -0.0148 | 0.0800 |
| ENJ__Kyle_lambda_threshold__t_1.0__rbull__c4__mall__h3d | 0.0685 | -0.0115 | 0.0800 |
| SOL__MACD_threshold__f_12_s_21_g_9__rchop__c1__mall__h3d | 0.0487 | -0.0213 | 0.0700 |
| SOL__MACD_threshold__f_12_s_26_g_9__rchop__c1__mall__h3d | 0.0552 | -0.0148 | 0.0700 |

## Size-discoverable engines (median notional > 1% of 5min ADV): 234

| engine_id | median_notional %/5min ADV | % fires above 1% |
|---|---:|---:|
| SUPER__Distance_z_state__period_20_threshold_1.0__rchop__c1__mall__h1d | 28851.460% | 100.0% |
| SUPER__measure_engines/norm_deviation__op_abs_gt_thr_1.0__rchop__c5__mall__h1d | 28838.633% | 100.0% |
| SUPER__measure_engines/te_imb__op_gt_thr_1.0__rbull__c1__mall__h1d | 28739.742% | 100.0% |
| SUPER__OBV_zscore__p_50_t_1.0__rbull__c1__mall__h1d | 28734.493% | 100.0% |
| JST__RSI_threshold__p_6_lo_40_hi_80__rchop__c5__mall__h1d | 24026.852% | 100.0% |
| JST__RSI_threshold__p_13_lo_40_hi_65__rchop__c5__mall__h1d | 24012.205% | 100.0% |
| JST__RSI_threshold__p_5_lo_40_hi_75__rchop__c5__mall__h1d | 24006.732% | 100.0% |
| JST__RSI_threshold__p_6_lo_40_hi_75__rchop__c5__mall__h1d | 23988.647% | 100.0% |
| ALGO__measure_engines/xd_funding_spread__op_abs_gt_thr_1.0__rbull__c5__mall__h1d | 23970.903% | 100.0% |
| JST__RSI_threshold__p_12_lo_40_hi_65__rchop__c5__mall__h1d | 23965.199% | 100.0% |
| JST__RSI_threshold__p_8_lo_40_hi_70__rchop__c5__mall__h1d | 23934.302% | 100.0% |
| JST__RSI_threshold__p_10_lo_40_hi_70__rchop__c5__mall__h1d | 23932.684% | 100.0% |
| JST__RSI_threshold__p_22_lo_35_hi_60__rbull__c1__mall__h1d | 23932.684% | 100.0% |
| JST__RSI_threshold__p_5_lo_35_hi_75__rchop__c5__mall__h1d | 23932.684% | 100.0% |
| JST__RSI_threshold__p_6_lo_35_hi_75__rchop__c5__mall__h1d | 23932.684% | 100.0% |
| JST__RSI_threshold__p_7_lo_40_hi_70__rchop__c5__mall__h1d | 23932.684% | 100.0% |
| JST__RSI_threshold__p_7_lo_40_hi_75__rchop__c5__mall__h1d | 23932.684% | 100.0% |
| JST__MA_state_EMA_above__period_20__rbull__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_10_lo_40_hi_65__rchop__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_11_lo_40_hi_65__rchop__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_7_lo_35_hi_70__rchop__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_9_lo_40_hi_65__rchop__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_9_lo_40_hi_70__rchop__c5__mall__h1d | 23932.521% | 100.0% |
| JST__RSI_threshold__p_15_lo_40_hi_60__rchop__c5__mall__h1d | 23932.084% | 100.0% |
| JST__RSI_threshold__p_21_lo_35_hi_60__rbull__c1__mall__h1d | 23932.084% | 100.0% |
| JST__MA_state_SMA_above__period_20__rbull__c5__mall__h1d | 23887.939% | 100.0% |
| JST__MA_state_SMA_above__period_100__rchop__c5__mall__h1d | 23850.338% | 100.0% |
| JST__MA_state_EMA_above__period_100__rchop__c5__mall__h1d | 23846.989% | 100.0% |
| AR__measure_engines/te_btc_imb__op_lt_thr_1.0__rbull__c1__mall__h1d | 19228.509% | 100.0% |
| AR__measure_engines/rv_rv_5m__op_gt_thr_1.0__rbull__c2__mall__h1d | 19159.824% | 100.0% |

## Honest caveats

- **Y3.4 (slow-rebalancer signature) is N/A** — all 234 catch-tier engines are 1d cadence; no intraday timing to exploit.
- **Y3.6 (cancel-spam survival) skipped** — DATA GAP: chimera v51 lacks cancel-ratio columns.
- **ADV proxy is daily-volume-based**, not actual 5min order-book depth. Real 5min ADV depends on time-of-day and exchange routing; this proxy assumes uniform intraday distribution (volume_usd / 288 bars). Likely understates true depth at peak hours and overstates it overnight.
- **Y3.2 uses wf_cov_stability as a proxy** for true epsilon-perturbation of indicator thresholds — a direct probe (shift RSI threshold by 0.5sigma) was out-of-scope for the 5-minute budget and would require re-running the engine simulator.
- **Composite is unweighted mean** of 4 components. A learned weighting (e.g. via OOS validation) is the natural next step but requires held-out data this mining task cannot touch.
- **Fee fragility uses event-level pre-cost pnl** — does NOT model fill probability degradation at maker (p_fill=0.21-0.40 per MakerCostModel calibration). True maker-vs-taker tradeoff is asymmetric in a way this score does not capture.
