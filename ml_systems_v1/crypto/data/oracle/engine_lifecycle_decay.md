# Engine Lifecycle Decay (TRAIN 3-third stability) (2026-05-23T09:20)

- Catch-tier engines analyzed: 234
- STABLE (sign-consistent across 3 thirds, stab_ratio<0.40, positive overall): **17** (7.3%)
- Positive overall: 127 (54.3%)
- Decaying (early > late by >0.5pp): 86
- Strengthening (late > early by >0.5pp): 148

## Top 30 STABLE engines (sign-consistent, stab_ratio<0.40, positive)

| asset | class | config | regime | n_fires | early% | mid% | late% | stab_ratio |
|---|---|---|---|---:|---:|---:|---:|---:|
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 175 | +447.21 | +439.01 | +414.31 | 0.03 |
| FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | 136 | +347.16 | +519.96 | +345.73 | 0.20 |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 174 | +398.94 | +363.49 | +417.84 | 0.06 |
| FET | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 137 | +360.75 | +401.48 | +355.97 | 0.05 |
| AR | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 240 | +309.95 | +189.64 | +461.30 | 0.35 |
| AR | MA_state_SMA_above | period_20 | bull | 409 | +239.22 | +184.59 | +432.27 | 0.37 |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 222 | +204.41 | +342.45 | +205.80 | 0.26 |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 106 | +111.53 | +212.64 | +182.05 | 0.25 |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 106 | +111.53 | +212.64 | +182.05 | 0.25 |
| APT | MA_state_EMA_above | period_20 | bull | 382 | +102.95 | +185.06 | +199.44 | 0.26 |
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 229 | +92.39 | +223.56 | +153.22 | 0.34 |
| DOT | Distance_z_state | period_50_threshold_1.5 | bull | 222 | +100.66 | +93.08 | +148.50 | 0.22 |
| BTC | Donchian_state_above_midline | period_20 | bull | 295 | +134.70 | +69.97 | +135.36 | 0.27 |
| BTC | MA_state_EMA_above | period_20 | bull | 308 | +96.28 | +74.81 | +125.08 | 0.21 |
| DASH | MA_state_EMA_above | period_50 | bull | 354 | +67.63 | +56.89 | +115.58 | 0.32 |
| DASH | MA_state_SMA_above | period_50 | bull | 368 | +68.45 | +64.50 | +105.77 | 0.23 |
| BTC | MA_state_SMA_above | period_50 | chop | 151 | +20.57 | +61.42 | +50.33 | 0.39 |

## Top 20 DECAYING (early > late, may be regime-shift fragile)

| asset | class | config | regime | n_fires | early% | mid% | late% | decay_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| UNI | measure_engines/liq_long_xsec_z | op_abs_gt_thr_1.0 | bull | 122 | -87.64 | -29.73 | -798.88 | +711.24 |
| WLD | YZ_vol_regime | t_0.5 | bull | 207 | +305.07 | +169.04 | -292.52 | +597.58 |
| UNI | confluence_engines/UNI_pair_4 | A_MA_state_EMA_above::period_20__AND_3b__B_measure_engines/hbr_eta_buy::op_abs_gt_thr_1.0 | chop | 76 | +191.68 | -198.38 | -361.02 | +552.70 |
| ARKM | Kyle_lambda_threshold | t_0.5 | bull | 559 | +174.64 | -304.56 | -368.98 | +543.62 |
| ARKM | Kyle_lambda_threshold | t_1.0 | bull | 351 | +55.88 | -257.23 | -455.62 | +511.49 |
| HBAR | ETF_flow_z | t_0.5 | bull | 125 | +407.47 | +257.00 | -103.88 | +511.35 |
| BCH | measure_engines/stbl_total_zscore_30d | op_abs_gt_thr_1.0 | bull | 82 | +177.85 | +108.26 | -290.75 | +468.60 |
| ADA | VPIN_threshold | t_1.0 | bull | 256 | +279.24 | +257.52 | -148.20 | +427.44 |
| FET | measure_engines/te_imb | op_abs_gt_thr_1.0 | bear | 116 | +171.01 | +192.74 | -248.55 | +419.56 |
| FLOKI | ETF_flow_z | t_0.5 | bull | 120 | +201.85 | +1029.98 | -181.88 | +383.73 |
| WLD | MA_state_SMA_above | period_50 | chop | 139 | +108.87 | +19.99 | -267.29 | +376.16 |
| LINK | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 124 | +364.84 | -116.36 | +1.09 | +363.75 |
| ADA | measure_engines/liq_short_z30 | op_gt_thr_1.0 | bull | 106 | +485.62 | +352.39 | +124.83 | +360.79 |
| SHIB | measure_engines/bs_basis_z30 | op_abs_gt_thr_1.0 | chop | 96 | -134.32 | -192.59 | -490.25 | +355.93 |
| FET | measure_engines/xd_funding_spread | op_abs_gt_thr_1.0 | bear | 80 | +140.75 | +427.54 | -167.91 | +308.66 |
| CHZ | measure_engines/xd_ma_distance | op_gt_thr_1.0 | chop | 46 | +148.87 | +44.21 | -158.47 | +307.33 |
| SUPER | measure_engines/norm_deviation | op_abs_gt_thr_1.0 | chop | 96 | +194.82 | -1047.69 | -107.09 | +301.91 |
| BNB | measure_engines/te_btc_imb | op_abs_gt_thr_1.0 | bull | 199 | +80.22 | +235.85 | -221.66 | +301.88 |
| ZEC | ETF_flow_z | t_0.5 | bull | 129 | +191.82 | +194.60 | -97.83 | +289.65 |
| PEPE | measure_engines/bd_imbalance_l5 | op_gt_thr_1.0 | chop | 93 | +111.58 | +122.30 | -155.24 | +266.82 |

## Top 20 STRENGTHENING (late > early, recent regime favors)

| asset | class | config | regime | n_fires | early% | mid% | late% | gain_pp |
|---|---|---|---|---:|---:|---:|---:|---:|
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_2.0 | bull | 66 | +211.05 | +349.36 | +2260.63 | +2049.58 |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_2.0 | bull | 66 | +211.05 | +349.36 | +2260.63 | +2049.58 |
| SHIB | measure_engines/rv_bpv_5m | op_gt_thr_1.5 | bull | 76 | +144.17 | +249.86 | +1915.53 | +1771.36 |
| SHIB | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.5 | bull | 76 | +144.17 | +249.86 | +1915.53 | +1771.36 |
| PEPE | MA_state_EMA_above | period_100 | chop | 114 | -120.63 | -126.19 | +712.07 | +832.71 |
| ICP | measure_engines/te_in_btc | op_gt_thr_1.0 | bull | 129 | +68.70 | +401.21 | +887.03 | +818.33 |
| PEPE | Donchian_state_above_midline | period_100 | chop | 124 | -125.56 | -165.08 | +536.49 | +662.05 |
| PEPE | MA_state_SMA_above | period_100 | chop | 136 | -77.09 | -123.17 | +559.80 | +636.89 |
| PEPE | MA_state_SMA_above | period_100 | chop | 136 | -77.09 | -123.17 | +559.80 | +636.89 |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 201 | +66.64 | +249.09 | +702.62 | +635.98 |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 186 | +109.92 | +224.01 | +706.89 | +596.97 |
| PEPE | Donchian_state_above_midline | period_55 | chop | 98 | -286.57 | -196.05 | +288.99 | +575.56 |
| BCH | measure_engines/rv_jump_count | op_abs_gt_thr_1.0 | bull | 68 | -74.54 | +91.22 | +471.90 | +546.44 |
| BCH | measure_engines/rv_jump_count | op_gt_thr_1.0 | bull | 68 | -74.54 | +91.22 | +471.90 | +546.44 |
| FET | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.5 | bull | 73 | -175.57 | -333.24 | +366.40 | +541.97 |
| PEPE | MA_state_SMA_above | period_50 | chop | 101 | -275.48 | -261.16 | +263.00 | +538.48 |
| AR | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 98 | +22.02 | +215.49 | +503.86 | +481.84 |
| FET | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bull | 162 | -132.80 | -19.33 | +346.49 | +479.28 |
| HBAR | measure_engines/te_btc_imb | op_abs_gt_thr_1.5 | bear | 49 | +47.90 | +334.14 | +522.51 | +474.61 |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.5 | bull | 131 | -75.37 | +277.78 | +381.20 | +456.57 |

## Stability by indicator class

| class | n_engines | n_stable | stable% | mean_decay_pp | median_stab_ratio |
|---|---:|---:|---:|---:|---:|
| Distance_z_state | 11 | 3 | 27.3% | -85.78 | 0.65 |
| MA_state_EMA_above | 10 | 3 | 30.0% | -49.09 | 1.14 |
| MA_state_SMA_above | 13 | 3 | 23.1% | -124.37 | 1.16 |
| measure_engines/liq_long_usd | 3 | 2 | 66.7% | +74.24 | 0.25 |
| measure_engines/rv_bpv_5m | 6 | 2 | 33.3% | -1267.93 | 0.99 |
| measure_engines/rv_rv_5m | 3 | 2 | 66.7% | -165.32 | 0.06 |
| measure_engines/te_in_btc | 4 | 1 | 25.0% | -158.65 | 0.54 |
| Donchian_state_above_midline | 6 | 1 | 16.7% | -206.41 | 1.42 |
| measure_engines/te_btc_imb | 7 | 0 | 0.0% | -65.89 | 1.84 |
| measure_engines/norm_efficiency | 6 | 0 | 0.0% | -117.65 | 2.30 |
| measure_engines/norm_flow_imbalance | 6 | 0 | 0.0% | -305.96 | 2.34 |
| measure_engines/rv_jump_count | 4 | 0 | 0.0% | -281.90 | 61.12 |
| measure_engines/rv_jump_frac | 3 | 0 | 0.0% | +245.76 | 2.39 |
| measure_engines/stbl_total_zscore_30d | 2 | 0 | 0.0% | +259.24 | 68.04 |
| ATR_bands | 3 | 0 | 0.0% | +107.79 | 4.62 |
| measure_engines/te_imb | 4 | 0 | 0.0% | +27.25 | 3.16 |
| measure_engines/liq_short_z30 | 1 | 0 | 0.0% | +360.79 | 0.46 |
| measure_engines/wh_whale_net_usd | 4 | 0 | 0.0% | -188.82 | 1.18 |
| measure_engines/wh_whale_trade_count_500k | 6 | 0 | 0.0% | -379.10 | 0.90 |
| measure_engines/xd_btc_return | 3 | 0 | 0.0% | -8.18 | 1.71 |
| measure_engines/xd_btc_volatility | 3 | 0 | 0.0% | -168.53 | 30.86 |
| measure_engines/xd_funding_spread | 5 | 0 | 0.0% | +0.69 | 3.75 |
| measure_engines/xd_ma_distance | 3 | 0 | 0.0% | +113.15 | 10.00 |
| measure_engines/norm_deviation | 3 | 0 | 0.0% | -29.70 | 1.65 |
| measure_engines/hbr_eta_total | 2 | 0 | 0.0% | -201.74 | 0.74 |
| measure_engines/liq_long_xsec_z | 1 | 0 | 0.0% | +711.24 | 1.15 |
| Bollinger_band_breach | 1 | 0 | 0.0% | -30.83 | 0.43 |
| ETF_flow_z | 4 | 0 | 0.0% | +342.57 | 1.44 |
| Hawkes_branching_imbalance | 1 | 0 | 0.0% | +168.33 | 1.08 |
| Kyle_lambda_threshold | 4 | 0 | 0.0% | +320.46 | 1.98 |
| Liquidation_cascade | 1 | 0 | 0.0% | +34.19 | 1.18 |
| MACD_threshold | 7 | 0 | 0.0% | +19.06 | 0.63 |
| OBV_zscore | 11 | 0 | 0.0% | +37.68 | 1.08 |
| RSI_threshold | 57 | 0 | 0.0% | -76.95 | 2.06 |
| VPIN_threshold | 4 | 0 | 0.0% | +94.70 | 3.40 |
| VWAP_state_above | 2 | 0 | 0.0% | +82.10 | 0.55 |
| YZ_vol_regime | 1 | 0 | 0.0% | +597.58 | 4.22 |
| confluence_engines/UNI_pair_4 | 1 | 0 | 0.0% | +552.70 | 1.89 |
| measure_engines/bd_imbalance_l1 | 1 | 0 | 0.0% | -220.57 | 0.69 |
| measure_engines/bd_imbalance_l5 | 3 | 0 | 0.0% | -42.24 | 1.23 |
| measure_engines/bs_basis_z30 | 5 | 0 | 0.0% | -45.16 | 4.09 |
| measure_engines/hbr_eta_buy | 3 | 0 | 0.0% | -325.41 | 0.90 |
| measure_engines/xd_momentum_rank | 6 | 0 | 0.0% | -77.97 | 3.39 |

## Stability by regime

| regime | n_engines | n_stable | stable% | mean_decay_pp |
|---|---:|---:|---:|---:|
| bull | 113 | 16 | 14.2% | -108.63 |
| chop | 100 | 1 | 1.0% | -41.75 |
| bear | 21 | 0 | 0.0% | -132.10 |

## Headline finding

- **17 of 234 catch-tier engines** are STABLE across TRAIN time thirds.
- Top stable assets: {'FET': 4, 'BTC': 3, 'AR': 2, 'FIL': 2, 'APT': 2, 'DASH': 2, 'ICP': 1, 'DOT': 1}
- Top stable classes: {'MA_state_SMA_above': 3, 'Distance_z_state': 3, 'MA_state_EMA_above': 3, 'measure_engines/rv_bpv_5m': 2, 'measure_engines/rv_rv_5m': 2}
- Interpretation: STABLE = sign-consistent across early/mid/late thirds + low slice-variance + positive overall mean.
- Use case: STABLE engines are the LOW-DECAY backbone for any ROI-floor basket. Decaying engines (n=86) need to be EXCLUDED or marked for re-validation.