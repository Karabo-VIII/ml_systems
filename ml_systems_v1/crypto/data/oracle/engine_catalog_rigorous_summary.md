# Oracle Engine Catalog v0 -- Summary (2026-05-23T00:27)

Total candidate engines evaluated: **213**
Engines passing ship-gate: **213**

## Top 20 engines (passed gate, sorted by compound_return_pct)

| asset | indicator_class | indicator_config | btc_regime_30d | hold_days | n_fires | hit_rate | expectancy_pct | compound_return_pct | max_dd_pct | shic_ratio | 3fold_sign_consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SHIB | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 1 | 11 | 0.636 | 10.759 | 170.6 | -1.969 | 0.274 | True |
| PEPE | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.0 | bull | 1 | 12 | 0.667 | 7.313 | 115.1 | -6.787 | 0.176 | True |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 1 | 17 | 0.647 | 4.663 | 104.8 | -14.675 | 0.241 | True |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 1 | 15 | 0.667 | 5.203 | 102.5 | -12.354 | 0.223 | True |
| SUPER | VPIN_threshold | t_0.5 | bull | 3 | 13 | 0.769 | 5.416 | 93.971 | -6.048 | 0.248 | True |
| FLOKI | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 3 | 11 | 0.727 | 6.805 | 92.555 | -1.331 | 0.195 | True |
| FET | VPIN_threshold | t_0.5 | bull | 3 | 11 | 0.545 | 7.120 | 88.896 | -10.416 | 0.298 | True |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.5 | bull | 1 | 11 | 0.636 | 5.880 | 78.937 | -12.354 | 0.237 | True |
| NEAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.5 | bull | 1 | 11 | 0.636 | 5.880 | 78.937 | -12.354 | 0.237 | True |
| SUI | MA_state_EMA_above | period_200 | bull | 1 | 21 | 0.667 | 2.935 | 73.030 | -6.783 | 0.183 | True |
| LINK | VPIN_threshold | t_0.5 | bull | 1 | 23 | 0.739 | 2.348 | 66.437 | -4.279 | 0.241 | True |
| ADA | VWAP_state_above | period_20 | bull | 3 | 12 | 0.750 | 4.591 | 62.481 | -11.066 | 0.274 | True |
| ALGO | MA_state_SMA_above | period_50 | bull | 1 | 37 | 0.568 | 1.423 | 59.807 | -12.967 | 0.223 | True |
| ICP | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 1 | 21 | 0.571 | 2.381 | 58.616 | -13.540 | 0.253 | True |
| ALGO | MA_state_EMA_above | period_50 | bull | 1 | 39 | 0.564 | 1.324 | 58.098 | -12.967 | 0.206 | True |
| ALGO | MA_state_SMA_above | period_20 | bull | 1 | 32 | 0.562 | 1.545 | 55.423 | -12.967 | 0.279 | True |
| ARKM | VPIN_threshold | t_0.5 | chop | 3 | 13 | 0.692 | 3.686 | 55.247 | -10.506 | 0.261 | True |
| LINK | VPIN_threshold | t_1.0 | bull | 1 | 14 | 0.786 | 3.306 | 54.656 | -1.896 | 0.195 | True |
| AR | MA_state_SMA_above | period_20 | bull | 1 | 10 | 0.700 | 4.448 | 52.268 | -0.438 | 0.251 | True |
| SUI | measure_engines/bd_imbalance_l5 | op_lt_thr_1.0 | chop | 1 | 11 | 0.818 | 3.803 | 49.756 | -2.105 | 0.170 | True |

## Engines per indicator class (passed gate)

| indicator_class | n_engines |
|---|---|
| MA_state_SMA_above | 26 |
| MA_state_EMA_above | 19 |
| measure_engines/norm_efficiency | 16 |
| VPIN_threshold | 15 |
| measure_engines/hbr_eta_total | 13 |
| measure_engines/hbr_eta_buy | 13 |
| measure_engines/xd_btc_return | 12 |
| measure_engines/norm_deviation | 12 |
| ATR_bands | 11 |
| Donchian_state_above_midline | 10 |
| MACD_threshold | 9 |
| measure_engines/xd_funding_spread | 9 |
| measure_engines/bs_basis_z30 | 8 |
| measure_engines/rv_jump_frac | 7 |
| measure_engines/wh_whale_trade_count_500k | 6 |
| measure_engines/bd_imbalance_l1 | 5 |
| measure_engines/bd_imbalance_l5 | 4 |
| VWAP_state_above | 4 |
| measure_engines/wh_whale_net_usd | 4 |
| measure_engines/liq_long_usd | 3 |
| measure_engines/liq_short_usd | 3 |
| measure_engines/stbl_total_zscore_30d | 2 |
| YZ_vol_regime | 1 |
| measure_engines/norm_funding_momentum | 1 |

## Engines per asset (passed gate)

| asset | n_engines |
|---|---|
| LINK | 14 |
| HBAR | 12 |
| XRP | 10 |
| ALGO | 9 |
| SOL | 8 |
| PEPE | 8 |
| ADA | 8 |
| BTC | 8 |
| AAVE | 7 |
| DOT | 7 |
| JST | 7 |
| FET | 7 |
| ETC | 7 |
| OP | 6 |
| DYDX | 6 |
| FIL | 6 |
| ARB | 6 |
| ZEC | 5 |
| CHZ | 5 |
| DASH | 5 |
| AVAX | 4 |
| APT | 4 |
| XLM | 4 |
| UNI | 4 |
| NEAR | 4 |
| ICP | 4 |
| BLUR | 4 |
| BNB | 4 |
| BCH | 3 |
| SUI | 3 |

## Engines per hold (passed gate)

| hold_days | n_engines |
|---|---|
| 1d | 177 |
| 3d | 36 |
