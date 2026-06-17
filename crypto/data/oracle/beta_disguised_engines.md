# Y2 -- Cluster-Residual Engine Audit (Beta-Disguised Detection)

Generated: 2026-05-23 09:42:49  
TRAIN end: 2024-05-15  
Rolling correlation window: 60 days  
Cluster refresh: every 7 days (no look-ahead)  
Target n_clusters: 6 (Agglomerative, average linkage, precomputed 1-corr distance)

## Cluster summary

- Refreshes total: **220**
- Typical n_clusters: **6**

### FIRST refresh (2020-03-03) -- n_assets_clustered=15
- cluster 0 (n=46): AAVE, APT, AR, ARB, ARKM, AVAX, BCH, BLUR, BONK, CHZ, CRV, DASH, DEXE, DOGE, DOT (+31 more)
- cluster 1 (n=5): ADA, ATOM, BTC, XLM, XRP
- cluster 2 (n=3): ALGO, ETH, TRX
- cluster 3 (n=1): ZEC
- cluster 4 (n=1): BNB
- cluster 5 (n=1): LINK

### MID refresh (2022-04-12) -- n_assets_clustered=38
- cluster 0 (n=51): AAVE, ADA, ALGO, APT, AR, ARB, ARKM, ATOM, AVAX, BCH, BLUR, BNB, BONK, BTC, CHZ (+36 more)
- cluster 1 (n=2): MOVR, NEAR
- cluster 2 (n=1): JST
- cluster 3 (n=1): INJ
- cluster 4 (n=1): FET
- cluster 5 (n=1): DEXE

### LAST refresh (2024-05-14) -- n_assets_clustered=57
- cluster 0 (n=34): ADA, ALGO, AR, ARB, ATOM, AVAX, BCH, BNB, BONK, BTC, CHZ, CRV, DOGE, DOT, DYDX (+19 more)
- cluster 1 (n=17): AAVE, APT, BLUR, DASH, ENJ, HBAR, INJ, JST, MOVR, PROM, TRX, UNI, WIF, XLM, XRP (+2 more)
- cluster 2 (n=1): NEAR
- cluster 3 (n=2): ARKM, ENA
- cluster 4 (n=2): DEXE, ORDI
- cluster 5 (n=1): FLOKI

## Headline counts

- TRUE_ALPHA: **130**
- MIXED: **34**
- BETA_DISGUISED: **70**
- Engines audited: **234** of 234 catch-tier (skipped if <10 fires on TRAIN)
- Ultimate deploy-eligible (TRUE_ALPHA AND stable AND not CONCAVE): **9**

## Top 30 TRUE_ALPHA engines (by residual expectancy %)

| asset | indicator_class | indicator_config | regime | n_fires | exp_raw_% | exp_residual_% | ratio |
|---|---|---|---|---|---|---|---|
| DASH | ETF_flow_z | t_0.5 | bull | 38 | 0.094 | 0.089 | 0.95 |
| ETC | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bear | 28 | 0.050 | 0.082 | 1.64 |
| JST | RSI_threshold | p_10_lo_40_hi_65 | chop | 86 | 0.032 | 0.074 | 2.32 |
| JST | RSI_threshold | p_7_lo_35_hi_70 | chop | 79 | 0.028 | 0.072 | 2.54 |
| JST | RSI_threshold | p_6_lo_35_hi_75 | chop | 77 | 0.029 | 0.072 | 2.45 |
| JST | RSI_threshold | p_7_lo_40_hi_75 | chop | 80 | 0.028 | 0.072 | 2.62 |
| JST | RSI_threshold | p_9_lo_40_hi_65 | chop | 88 | 0.036 | 0.072 | 1.99 |
| JST | RSI_threshold | p_6_lo_40_hi_80 | chop | 80 | 0.034 | 0.071 | 2.07 |
| CHZ | measure_engines/xd_funding_spread | op_abs_gt_thr_1.0 | bear | 18 | 0.074 | 0.071 | 0.95 |
| JST | RSI_threshold | p_8_lo_40_hi_70 | chop | 82 | 0.028 | 0.070 | 2.56 |
| SEI | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 35 | 0.093 | 0.070 | 0.75 |
| JST | RSI_threshold | p_9_lo_40_hi_70 | chop | 80 | 0.030 | 0.070 | 2.33 |
| PEPE | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.0 | bull | 38 | 0.092 | 0.069 | 0.75 |
| JST | RSI_threshold | p_10_lo_40_hi_70 | chop | 77 | 0.022 | 0.068 | 3.08 |
| JST | RSI_threshold | p_5_lo_35_hi_75 | chop | 81 | 0.032 | 0.064 | 2.01 |
| NEAR | measure_engines/xd_btc_volatility | op_abs_gt_thr_1.0 | bull | 41 | 0.055 | 0.063 | 1.15 |
| JST | RSI_threshold | p_7_lo_40_hi_70 | chop | 86 | 0.031 | 0.063 | 2.06 |
| JST | RSI_threshold | p_6_lo_40_hi_75 | chop | 85 | 0.029 | 0.058 | 2.00 |
| JST | RSI_threshold | p_11_lo_40_hi_65 | chop | 84 | 0.037 | 0.058 | 1.58 |
| WLD | MA_state_SMA_above | period_50 | chop | 43 | 0.066 | 0.057 | 0.86 |
| DOGE | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | chop | 31 | 0.018 | 0.056 | 3.19 |
| DASH | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | chop | 43 | 0.084 | 0.056 | 0.67 |
| DASH | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | chop | 43 | 0.084 | 0.056 | 0.67 |
| DASH | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | chop | 43 | 0.084 | 0.056 | 0.67 |
| JST | RSI_threshold | p_5_lo_40_hi_75 | chop | 89 | 0.032 | 0.054 | 1.68 |
| JST | RSI_threshold | p_12_lo_40_hi_65 | chop | 82 | 0.032 | 0.053 | 1.64 |
| FLOKI | ETF_flow_z | t_0.5 | bull | 38 | 0.067 | 0.051 | 0.76 |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 67 | 0.060 | 0.046 | 0.77 |
| AAVE | measure_engines/xd_btc_volatility | op_abs_gt_thr_1.0 | chop | 37 | 0.020 | 0.046 | 2.29 |
| FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | 36 | 0.059 | 0.044 | 0.75 |

## Top 20 BETA_DISGUISED engines (largest raw - residual gap)

| asset | indicator_class | indicator_config | regime | n_fires | exp_raw_% | exp_residual_% | gap_% |
|---|---|---|---|---|---|---|---|
| HBAR | measure_engines/te_btc_imb | op_abs_gt_thr_1.5 | bear | 18 | 0.016 | -0.075 | 0.091 |
| ADA | measure_engines/liq_short_z30 | op_gt_thr_1.0 | bull | 32 | 0.104 | 0.017 | 0.087 |
| AR | measure_engines/te_btc_imb | op_lt_thr_1.0 | bull | 36 | 0.061 | 0.010 | 0.052 |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_2.0 | bull | 18 | 0.000 | -0.049 | 0.050 |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_2.0 | bull | 18 | 0.000 | -0.049 | 0.050 |
| FIL | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | bull | 52 | 0.012 | -0.028 | 0.040 |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 35 | 0.007 | -0.032 | 0.039 |
| ICP | measure_engines/te_in_btc | op_abs_gt_thr_1.5 | bull | 52 | 0.033 | -0.004 | 0.037 |
| DASH | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 51 | 0.029 | -0.004 | 0.033 |
| ARKM | Kyle_lambda_threshold | t_1.0 | bull | 76 | 0.034 | 0.002 | 0.033 |
| HBAR | measure_engines/norm_efficiency | op_gt_thr_1.0 | bull | 23 | 0.010 | -0.021 | 0.031 |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 41 | 0.014 | -0.016 | 0.030 |
| AR | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 32 | 0.017 | -0.014 | 0.030 |
| FET | measure_engines/xd_funding_spread | op_abs_gt_thr_1.0 | bear | 25 | 0.006 | -0.023 | 0.030 |
| AR | Distance_z_state | period_20_threshold_1.0 | bull | 86 | 0.004 | -0.024 | 0.028 |
| AR | Distance_z_state | period_20_threshold_1.0 | bull | 86 | 0.004 | -0.024 | 0.028 |
| FIL | measure_engines/xd_ma_distance | op_abs_gt_thr_1.0 | bull | 44 | 0.003 | -0.022 | 0.026 |
| BNB | measure_engines/te_btc_imb | op_abs_gt_thr_1.0 | bull | 77 | 0.018 | -0.007 | 0.025 |
| ARB | measure_engines/bs_basis_z30 | op_abs_gt_thr_1.0 | bull | 36 | 0.033 | 0.008 | 0.025 |
| ENJ | Kyle_lambda_threshold | t_1.0 | bull | 91 | 0.022 | -0.002 | 0.024 |

## Per-indicator-class tally

| indicator_class | classification | n |
|---|---|---|
| ATR_bands | TRUE_ALPHA | 3 |
| Bollinger_band_breach | MIXED | 1 |
| Distance_z_state | BETA_DISGUISED | 6 |
| Distance_z_state | MIXED | 3 |
| Distance_z_state | TRUE_ALPHA | 2 |
| Donchian_state_above_midline | TRUE_ALPHA | 6 |
| ETF_flow_z | MIXED | 1 |
| ETF_flow_z | TRUE_ALPHA | 3 |
| Hawkes_branching_imbalance | MIXED | 1 |
| Kyle_lambda_threshold | BETA_DISGUISED | 3 |
| Kyle_lambda_threshold | TRUE_ALPHA | 1 |
| Liquidation_cascade | TRUE_ALPHA | 1 |
| MACD_threshold | BETA_DISGUISED | 4 |
| MACD_threshold | TRUE_ALPHA | 3 |
| MA_state_EMA_above | BETA_DISGUISED | 2 |
| MA_state_EMA_above | TRUE_ALPHA | 8 |
| MA_state_SMA_above | BETA_DISGUISED | 3 |
| MA_state_SMA_above | MIXED | 2 |
| MA_state_SMA_above | TRUE_ALPHA | 8 |
| OBV_zscore | BETA_DISGUISED | 7 |
| OBV_zscore | MIXED | 2 |
| OBV_zscore | TRUE_ALPHA | 2 |
| RSI_threshold | BETA_DISGUISED | 11 |
| RSI_threshold | MIXED | 4 |
| RSI_threshold | TRUE_ALPHA | 42 |
| VPIN_threshold | BETA_DISGUISED | 2 |
| VPIN_threshold | TRUE_ALPHA | 2 |
| VWAP_state_above | MIXED | 1 |
| VWAP_state_above | TRUE_ALPHA | 1 |
| YZ_vol_regime | TRUE_ALPHA | 1 |
| confluence_engines/UNI_pair_4 | TRUE_ALPHA | 1 |
| measure_engines/bd_imbalance_l1 | BETA_DISGUISED | 1 |
| measure_engines/bd_imbalance_l5 | TRUE_ALPHA | 3 |
| measure_engines/bs_basis_z30 | BETA_DISGUISED | 3 |
| measure_engines/bs_basis_z30 | MIXED | 1 |
| measure_engines/bs_basis_z30 | TRUE_ALPHA | 1 |
| measure_engines/hbr_eta_buy | MIXED | 2 |
| measure_engines/hbr_eta_buy | TRUE_ALPHA | 1 |
| measure_engines/hbr_eta_total | MIXED | 2 |
| measure_engines/liq_long_usd | MIXED | 2 |
| measure_engines/liq_long_usd | TRUE_ALPHA | 1 |
| measure_engines/liq_long_xsec_z | TRUE_ALPHA | 1 |
| measure_engines/liq_short_z30 | BETA_DISGUISED | 1 |
| measure_engines/norm_deviation | MIXED | 2 |
| measure_engines/norm_deviation | TRUE_ALPHA | 1 |
| measure_engines/norm_efficiency | BETA_DISGUISED | 2 |
| measure_engines/norm_efficiency | TRUE_ALPHA | 4 |
| measure_engines/norm_flow_imbalance | MIXED | 1 |
| measure_engines/norm_flow_imbalance | TRUE_ALPHA | 5 |
| measure_engines/rv_bpv_5m | BETA_DISGUISED | 2 |
| measure_engines/rv_bpv_5m | TRUE_ALPHA | 4 |
| measure_engines/rv_jump_count | BETA_DISGUISED | 2 |
| measure_engines/rv_jump_count | TRUE_ALPHA | 2 |
| measure_engines/rv_jump_frac | MIXED | 3 |
| measure_engines/rv_rv_5m | BETA_DISGUISED | 1 |
| measure_engines/rv_rv_5m | MIXED | 1 |
| measure_engines/rv_rv_5m | TRUE_ALPHA | 1 |
| measure_engines/stbl_total_zscore_30d | BETA_DISGUISED | 1 |
| measure_engines/stbl_total_zscore_30d | TRUE_ALPHA | 1 |
| measure_engines/te_btc_imb | BETA_DISGUISED | 4 |
| measure_engines/te_btc_imb | MIXED | 1 |
| measure_engines/te_btc_imb | TRUE_ALPHA | 2 |
| measure_engines/te_imb | BETA_DISGUISED | 1 |
| measure_engines/te_imb | MIXED | 1 |
| measure_engines/te_imb | TRUE_ALPHA | 2 |
| measure_engines/te_in_btc | BETA_DISGUISED | 2 |
| measure_engines/te_in_btc | MIXED | 1 |
| measure_engines/te_in_btc | TRUE_ALPHA | 1 |
| measure_engines/wh_whale_net_usd | BETA_DISGUISED | 1 |
| measure_engines/wh_whale_net_usd | MIXED | 1 |
| measure_engines/wh_whale_net_usd | TRUE_ALPHA | 2 |
| measure_engines/wh_whale_trade_count_500k | BETA_DISGUISED | 4 |
| measure_engines/wh_whale_trade_count_500k | MIXED | 1 |
| measure_engines/wh_whale_trade_count_500k | TRUE_ALPHA | 1 |
| measure_engines/xd_btc_return | BETA_DISGUISED | 2 |
| measure_engines/xd_btc_return | TRUE_ALPHA | 1 |
| measure_engines/xd_btc_volatility | TRUE_ALPHA | 3 |
| measure_engines/xd_funding_spread | BETA_DISGUISED | 2 |
| measure_engines/xd_funding_spread | TRUE_ALPHA | 3 |
| measure_engines/xd_ma_distance | BETA_DISGUISED | 2 |
| measure_engines/xd_ma_distance | TRUE_ALPHA | 1 |
| measure_engines/xd_momentum_rank | BETA_DISGUISED | 1 |
| measure_engines/xd_momentum_rank | TRUE_ALPHA | 5 |

## Per-regime tally

| btc_regime_30d | classification | n |
|---|---|---|
| bear | BETA_DISGUISED | 11 |
| bear | TRUE_ALPHA | 10 |
| bull | BETA_DISGUISED | 43 |
| bull | MIXED | 14 |
| bull | TRUE_ALPHA | 56 |
| chop | BETA_DISGUISED | 16 |
| chop | MIXED | 20 |
| chop | TRUE_ALPHA | 64 |

## Cross-reference summary

- TRUE_ALPHA engines: 127
- Stable engines (lifecycle stable_flag=True): 17
- CONCAVE engines (antifragility audit): 80
- Intersection TRUE_ALPHA & stable: 12
- Intersection TRUE_ALPHA & stable & NOT concave (ultimate deploy-eligible): **9**

### Ultimate deploy-eligible engines (full list)

| asset | indicator_class | indicator_config | regime | n_fires | exp_residual_% | ratio |
|---|---|---|---|---|---|---|
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 67 | 0.046 | 0.77 |
| FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | 36 | 0.044 | 0.75 |
| FET | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 37 | 0.037 | 0.64 |
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 48 | 0.025 | 0.70 |
| AR | MA_state_SMA_above | period_20 | bull | 120 | -0.014 | 3.46 |
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 72 | -0.021 | 1.67 |
| APT | MA_state_EMA_above | period_20 | bull | 117 | -0.029 | 2.23 |
| BTC | MA_state_EMA_above | period_20 | bull | 120 | -0.057 | 6.49 |
| BTC | Donchian_state_above_midline | period_20 | bull | 112 | -0.060 | 7.55 |

## Honest caveats

- **60-day rolling correlation is unstable for short-lived assets**: assets listed <60 days before a refresh are clustered using whatever history exists (>=30 obs required) or fall through to cluster 0; their cluster assignment is noisy.
- **Cluster refreshes are weekly (every 7 days) and use ONLY prior data** -- no look-ahead. But cluster boundaries can shift sharply across a refresh; expectancy in a fire-week that straddles a refresh may be computed against two different cluster centroids.
- **Residual return subtracts cluster mean (excluding self)**. This is a LONG-SHORT alpha: residual expectancy of +0.5% is realized by going LONG the asset AND SHORT the equal-weight basket of the other cluster members. A raw long-only deploy WILL NOT realize residual expectancy.
- **Beta disguised ratio uses signed expectancy**. If raw and residual have opposite sign, classification is BETA_DISGUISED regardless of magnitude (cluster beta was holding the engine up; the asset-specific signal is anti-correlated).
- **MIN_FIRES filter**: engines with <10 fires on TRAIN are skipped (insufficient statistics).
- **Cluster count fixed at 6**: this is a heuristic; varying n_clusters changes residuals. We chose 6 to align with typical crypto-asset macro buckets (BTC-beta, ETH-beta, mid-cap-alts, memes, DeFi-leaders, stables-adjacent). Sensitivity to n_clusters NOT tested.
- **Catch-tier eligibility is the input filter**: any engine that did not earn catch_tier_eligibility on the discovery pass is invisible to this audit, even if it is a stable, convex, residual-alpha generator on its own.
