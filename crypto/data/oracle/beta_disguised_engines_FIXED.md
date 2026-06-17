# Y2 (FIXED) -- Cluster-Residual Engine Audit (Beta-Disguised Detection)

Generated: 2026-05-23 10:55:14  
TRAIN end: 2024-05-15  
Source returns: **close.pct_change()** (CORRECTED). Previous run used corrupted `target_return_1_raw`.

## Headline counts -- FIXED vs OLD

| class | OLD | FIXED | delta |
|---|---|---|---|
| TRUE_ALPHA | 130 | 93 | -37 |
| MIXED | 34 | 51 | +17 |
| BETA_DISGUISED | 70 | 90 | +20 |

Engines audited: **234** (FIXED) vs **234** (OLD)
Ultimate deploy-eligible (TRUE_ALPHA AND stable AND NOT CONCAVE): **3**

## Reclassification flips (OLD -> FIXED)

- TRUE_ALPHA -> BETA_DISGUISED: **51**
- BETA_DISGUISED -> TRUE_ALPHA: **27**
- Other flips (involving MIXED or in/out of audit set): **69**

## Cluster summary

- Refreshes total: **220**
- Typical n_clusters: **6**

### FIRST refresh (2020-03-03) -- n_assets_clustered=15
- cluster 0 (n=51): AAVE, ADA, APT, AR, ARB, ARKM, AVAX, BCH, BLUR, BNB, BONK, BTC, CHZ, CRV, DEXE (+36 more)
- cluster 1 (n=2): ETC, ZEC
- cluster 2 (n=1): DASH
- cluster 3 (n=1): ALGO
- cluster 4 (n=1): LINK
- cluster 5 (n=1): ATOM

### MID refresh (2022-04-12) -- n_assets_clustered=38
- cluster 0 (n=51): AAVE, ADA, ALGO, APT, AR, ARB, ARKM, ATOM, AVAX, BCH, BLUR, BNB, BONK, BTC, CHZ (+36 more)
- cluster 1 (n=1): JST
- cluster 2 (n=2): FET, NEAR
- cluster 3 (n=1): DEXE
- cluster 4 (n=1): INJ
- cluster 5 (n=1): ZEC

### LAST refresh (2024-05-14) -- n_assets_clustered=57
- cluster 0 (n=52): AAVE, ADA, ALGO, APT, ARB, ARKM, ATOM, AVAX, BCH, BLUR, BNB, BONK, BTC, CHZ, CRV (+37 more)
- cluster 1 (n=1): SUI
- cluster 2 (n=1): TIA
- cluster 3 (n=1): ENA
- cluster 4 (n=1): LTC
- cluster 5 (n=1): AR

## Top 30 TRUE_ALPHA engines (by residual expectancy %)

| asset | indicator_class | indicator_config | regime | n_fires | exp_raw_% | exp_residual_% | ratio |
|---|---|---|---|---|---|---|---|
| FET | measure_engines/norm_flow_imbalance | op_gt_thr_1.0 | bull | 24 | 6.818 | 4.600 | 0.67 |
| PEPE | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.0 | bull | 38 | 5.718 | 4.559 | 0.80 |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_2.0 | bull | 17 | 4.605 | 3.624 | 0.79 |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_2.0 | bull | 17 | 4.605 | 3.624 | 0.79 |
| ICP | measure_engines/te_in_btc | op_gt_thr_1.0 | bull | 37 | 3.163 | 2.704 | 0.86 |
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 48 | 3.768 | 2.501 | 0.66 |
| SUPER | measure_engines/te_imb | op_gt_thr_1.0 | bull | 43 | 3.038 | 2.289 | 0.75 |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 49 | 3.449 | 2.195 | 0.64 |
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.5 | bull | 34 | 3.374 | 2.180 | 0.65 |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.5 | bull | 21 | 3.521 | 2.165 | 0.61 |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_1.5 | bull | 21 | 3.521 | 2.165 | 0.61 |
| ICP | measure_engines/xd_btc_volatility | op_abs_gt_thr_1.0 | bull | 44 | 2.685 | 1.971 | 0.73 |
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull | 26 | 2.939 | 1.966 | 0.67 |
| ICP | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 55 | 2.432 | 1.915 | 0.79 |
| FET | measure_engines/xd_funding_spread | op_abs_gt_thr_1.0 | bear | 25 | 1.992 | 1.814 | 0.91 |
| AVAX | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bull | 43 | 2.112 | 1.639 | 0.78 |
| PEPE | MA_state_EMA_above | period_100 | chop | 33 | 1.521 | 1.636 | 1.08 |
| PEPE | MA_state_SMA_above | period_100 | chop | 40 | 1.282 | 1.472 | 1.15 |
| PEPE | MA_state_SMA_above | period_100 | chop | 40 | 1.282 | 1.472 | 1.15 |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 67 | 1.892 | 1.430 | 0.76 |
| SUPER | OBV_zscore | p_50_t_1.0 | bull | 126 | 1.828 | 1.404 | 0.77 |
| CHZ | measure_engines/xd_ma_distance | op_gt_thr_1.0 | chop | 15 | 2.146 | 1.369 | 0.64 |
| FET | measure_engines/te_imb | op_abs_gt_thr_1.0 | bear | 37 | 1.604 | 1.262 | 0.79 |
| ICP | RSI_threshold | p_5_lo_40_hi_75 | bull | 109 | 1.526 | 1.211 | 0.79 |
| SEI | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 35 | 1.528 | 1.198 | 0.78 |
| FLOKI | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 46 | 1.374 | 1.161 | 0.84 |
| FLOKI | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 46 | 1.374 | 1.161 | 0.84 |
| SOL | measure_engines/te_imb | op_abs_gt_thr_1.0 | chop | 46 | 1.043 | 1.149 | 1.10 |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.5 | bull | 39 | 1.520 | 1.071 | 0.70 |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.5 | bull | 39 | 1.520 | 1.071 | 0.70 |

## Top 20 BETA_DISGUISED engines (largest raw - residual gap)

| asset | indicator_class | indicator_config | regime | n_fires | exp_raw_% | exp_residual_% | gap_% |
|---|---|---|---|---|---|---|---|
| HBAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 41 | 1.538 | -0.494 | 2.033 |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 35 | 1.304 | -0.580 | 1.884 |
| HBAR | measure_engines/norm_efficiency | op_abs_gt_thr_1.0 | bull | 38 | 1.786 | 0.086 | 1.701 |
| BTC | measure_engines/stbl_total_zscore_30d | op_abs_gt_thr_1.0 | bull | 26 | 1.137 | -0.332 | 1.469 |
| BCH | measure_engines/rv_jump_count | op_abs_gt_thr_1.0 | bull | 24 | 0.688 | -0.738 | 1.427 |
| BCH | measure_engines/rv_jump_count | op_gt_thr_1.0 | bull | 24 | 0.688 | -0.738 | 1.427 |
| ADA | measure_engines/te_imb | op_abs_gt_thr_1.0 | bull | 67 | 1.496 | 0.172 | 1.324 |
| BCH | measure_engines/stbl_total_zscore_30d | op_abs_gt_thr_1.0 | bull | 26 | 1.044 | -0.269 | 1.313 |
| PEPE | measure_engines/bd_imbalance_l5 | op_gt_thr_1.0 | chop | 28 | 1.649 | 0.360 | 1.289 |
| ENJ | Kyle_lambda_threshold | t_1.5 | bull | 53 | 1.410 | 0.134 | 1.276 |
| DASH | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 51 | 0.804 | -0.455 | 1.259 |
| ARKM | Kyle_lambda_threshold | t_0.5 | bull | 106 | 1.518 | 0.340 | 1.177 |
| ENJ | Kyle_lambda_threshold | t_1.0 | bull | 91 | 1.338 | 0.173 | 1.165 |
| SUI | measure_engines/xd_ma_distance | op_abs_gt_thr_1.0 | bull | 47 | 0.700 | -0.457 | 1.158 |
| DOT | Distance_z_state | period_50_threshold_1.5 | bull | 65 | 0.984 | -0.052 | 1.037 |
| FIL | measure_engines/bd_imbalance_l5 | op_abs_gt_thr_1.0 | bull | 54 | 1.169 | 0.154 | 1.015 |
| ZEC | measure_engines/bd_imbalance_l5 | op_abs_gt_thr_1.0 | bull | 60 | 0.495 | -0.517 | 1.012 |
| ADA | Distance_z_state | period_50_threshold_1.5 | bull | 72 | 1.043 | 0.044 | 0.999 |
| ETC | measure_engines/norm_efficiency | op_gt_thr_1.0 | chop | 26 | 0.871 | -0.106 | 0.977 |
| HBAR | ETF_flow_z | t_0.5 | bull | 38 | 0.953 | -0.004 | 0.957 |

## Per-indicator-class tally

| indicator_class | classification | n |
|---|---|---|
| ATR_bands | BETA_DISGUISED | 3 |
| Bollinger_band_breach | BETA_DISGUISED | 1 |
| Distance_z_state | BETA_DISGUISED | 6 |
| Distance_z_state | MIXED | 3 |
| Distance_z_state | TRUE_ALPHA | 2 |
| Donchian_state_above_midline | BETA_DISGUISED | 2 |
| Donchian_state_above_midline | TRUE_ALPHA | 4 |
| ETF_flow_z | BETA_DISGUISED | 3 |
| ETF_flow_z | MIXED | 1 |
| Hawkes_branching_imbalance | BETA_DISGUISED | 1 |
| Kyle_lambda_threshold | BETA_DISGUISED | 3 |
| Kyle_lambda_threshold | MIXED | 1 |
| Liquidation_cascade | MIXED | 1 |
| MACD_threshold | TRUE_ALPHA | 7 |
| MA_state_EMA_above | BETA_DISGUISED | 4 |
| MA_state_EMA_above | TRUE_ALPHA | 6 |
| MA_state_SMA_above | BETA_DISGUISED | 3 |
| MA_state_SMA_above | MIXED | 2 |
| MA_state_SMA_above | TRUE_ALPHA | 8 |
| OBV_zscore | BETA_DISGUISED | 5 |
| OBV_zscore | MIXED | 4 |
| OBV_zscore | TRUE_ALPHA | 2 |
| RSI_threshold | BETA_DISGUISED | 26 |
| RSI_threshold | MIXED | 13 |
| RSI_threshold | TRUE_ALPHA | 18 |
| VPIN_threshold | MIXED | 2 |
| VPIN_threshold | TRUE_ALPHA | 2 |
| VWAP_state_above | BETA_DISGUISED | 2 |
| YZ_vol_regime | MIXED | 1 |
| confluence_engines/UNI_pair_4 | MIXED | 1 |
| measure_engines/bd_imbalance_l1 | TRUE_ALPHA | 1 |
| measure_engines/bd_imbalance_l5 | BETA_DISGUISED | 3 |
| measure_engines/bs_basis_z30 | BETA_DISGUISED | 2 |
| measure_engines/bs_basis_z30 | TRUE_ALPHA | 3 |
| measure_engines/hbr_eta_buy | TRUE_ALPHA | 3 |
| measure_engines/hbr_eta_total | TRUE_ALPHA | 2 |
| measure_engines/liq_long_usd | MIXED | 2 |
| measure_engines/liq_long_usd | TRUE_ALPHA | 1 |
| measure_engines/liq_long_xsec_z | BETA_DISGUISED | 1 |
| measure_engines/liq_short_z30 | MIXED | 1 |
| measure_engines/norm_deviation | BETA_DISGUISED | 1 |
| measure_engines/norm_deviation | MIXED | 2 |
| measure_engines/norm_efficiency | BETA_DISGUISED | 5 |
| measure_engines/norm_efficiency | MIXED | 1 |
| measure_engines/norm_flow_imbalance | BETA_DISGUISED | 1 |
| measure_engines/norm_flow_imbalance | MIXED | 3 |
| measure_engines/norm_flow_imbalance | TRUE_ALPHA | 2 |
| measure_engines/rv_bpv_5m | MIXED | 1 |
| measure_engines/rv_bpv_5m | TRUE_ALPHA | 5 |
| measure_engines/rv_jump_count | BETA_DISGUISED | 2 |
| measure_engines/rv_jump_count | TRUE_ALPHA | 2 |
| measure_engines/rv_jump_frac | TRUE_ALPHA | 3 |
| measure_engines/rv_rv_5m | MIXED | 2 |
| measure_engines/rv_rv_5m | TRUE_ALPHA | 1 |
| measure_engines/stbl_total_zscore_30d | BETA_DISGUISED | 2 |
| measure_engines/te_btc_imb | BETA_DISGUISED | 2 |
| measure_engines/te_btc_imb | MIXED | 1 |
| measure_engines/te_btc_imb | TRUE_ALPHA | 4 |
| measure_engines/te_imb | BETA_DISGUISED | 1 |
| measure_engines/te_imb | TRUE_ALPHA | 3 |
| measure_engines/te_in_btc | MIXED | 2 |
| measure_engines/te_in_btc | TRUE_ALPHA | 2 |
| measure_engines/wh_whale_net_usd | MIXED | 1 |
| measure_engines/wh_whale_net_usd | TRUE_ALPHA | 3 |
| measure_engines/wh_whale_trade_count_500k | BETA_DISGUISED | 2 |
| measure_engines/wh_whale_trade_count_500k | MIXED | 1 |
| measure_engines/wh_whale_trade_count_500k | TRUE_ALPHA | 3 |
| measure_engines/xd_btc_return | BETA_DISGUISED | 2 |
| measure_engines/xd_btc_return | MIXED | 1 |
| measure_engines/xd_btc_volatility | MIXED | 1 |
| measure_engines/xd_btc_volatility | TRUE_ALPHA | 2 |
| measure_engines/xd_funding_spread | BETA_DISGUISED | 2 |
| measure_engines/xd_funding_spread | MIXED | 1 |
| measure_engines/xd_funding_spread | TRUE_ALPHA | 2 |
| measure_engines/xd_ma_distance | BETA_DISGUISED | 1 |
| measure_engines/xd_ma_distance | MIXED | 1 |
| measure_engines/xd_ma_distance | TRUE_ALPHA | 1 |
| measure_engines/xd_momentum_rank | BETA_DISGUISED | 4 |
| measure_engines/xd_momentum_rank | MIXED | 1 |
| measure_engines/xd_momentum_rank | TRUE_ALPHA | 1 |

## Cross-reference summary

- TRUE_ALPHA engines: 90
- Stable engines (lifecycle stable_flag=True): 17
- CONCAVE engines (antifragility audit): 80
- Intersection TRUE_ALPHA & stable: 3
- Intersection TRUE_ALPHA & stable & NOT concave (ultimate deploy-eligible): **3**

### Ultimate deploy-eligible engines (full list, FIXED)

| asset | indicator_class | indicator_config | regime | n_fires | exp_residual_% | ratio |
|---|---|---|---|---|---|---|
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 48 | 2.501 | 0.66 |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 49 | 2.195 | 0.64 |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 67 | 1.430 | 0.76 |