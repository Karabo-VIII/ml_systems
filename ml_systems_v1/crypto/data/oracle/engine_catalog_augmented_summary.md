# Engine Catalog Augmented -- summary (2026-05-23T00:28)

Catalog rows: 213
By engine_family: {'measure_event': 118, 'ta_state': 59, 'ta_event': 36}
By direction: {'long_fire': 213}
Pass ship gate: 213

## Top 20 by composite_score (passed gate)

| asset | engine_family | indicator_class | indicator_config | btc_regime_30d | hold_days | n_fires | hit_rate | expectancy_pct | compound_return_pct | max_dd_pct | shic_ratio | composite_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SHIB | measure_event | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 1 | 11 | 0.636 | 10.759 | 170.6 | -1.969 | 0.274 | 45.044 |
| FLOKI | measure_event | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 3 | 11 | 0.727 | 6.805 | 92.555 | -1.331 | 0.195 | 32.563 |
| PEPE | measure_event | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.0 | bull | 1 | 12 | 0.667 | 7.313 | 115.1 | -6.787 | 0.176 | 32.075 |
| SUPER | ta_event | VPIN_threshold | t_0.5 | bull | 3 | 13 | 0.769 | 5.416 | 93.971 | -6.048 | 0.248 | 27.412 |
| FET | ta_event | VPIN_threshold | t_0.5 | bull | 3 | 11 | 0.545 | 7.120 | 88.896 | -10.416 | 0.298 | 25.550 |
| NEAR | measure_event | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.5 | bull | 1 | 11 | 0.636 | 5.880 | 78.937 | -12.354 | 0.237 | 24.618 |
| NEAR | measure_event | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.5 | bull | 1 | 11 | 0.636 | 5.880 | 78.937 | -12.354 | 0.237 | 24.618 |
| NEAR | measure_event | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 1 | 15 | 0.667 | 5.203 | 102.5 | -12.354 | 0.223 | 22.822 |
| ADA | ta_state | VWAP_state_above | period_20 | bull | 3 | 12 | 0.750 | 4.591 | 62.481 | -11.066 | 0.274 | 22.653 |
| XRP | measure_event | measure_engines/hbr_eta_buy | op_gt_thr_1.0 | bull | 1 | 10 | 0.900 | 3.750 | 42.711 | -0.286 | 0.207 | 22.207 |
| AR | ta_state | MA_state_SMA_above | period_20 | bull | 1 | 10 | 0.700 | 4.448 | 52.268 | -0.438 | 0.251 | 20.487 |
| SUI | measure_event | measure_engines/bd_imbalance_l5 | op_lt_thr_1.0 | chop | 1 | 11 | 0.818 | 3.803 | 49.756 | -2.105 | 0.170 | 20.471 |
| NEAR | measure_event | measure_engines/wh_whale_trade_count_500k | op_abs_gt_thr_1.0 | bull | 1 | 17 | 0.647 | 4.663 | 104.8 | -14.675 | 0.241 | 19.853 |
| HBAR | measure_event | measure_engines/norm_efficiency | op_gt_thr_1.0 | bull | 1 | 11 | 0.727 | 3.965 | 49.659 | -1.755 | 0.163 | 18.972 |
| HBAR | measure_event | measure_engines/norm_deviation | op_gt_thr_1.0 | chop | 1 | 10 | 0.700 | 3.817 | 43.574 | -3.809 | 0.216 | 17.580 |
| ADA | ta_event | VPIN_threshold | t_1.0 | bull | 3 | 10 | 0.600 | 4.419 | 49.315 | -5.212 | 0.164 | 17.443 |
| LINK | ta_event | VPIN_threshold | t_1.0 | bull | 1 | 14 | 0.786 | 3.306 | 54.656 | -1.896 | 0.195 | 17.092 |
| LINK | ta_event | MACD_threshold | f_12_s_35_g_9 | chop | 3 | 11 | 0.727 | 3.528 | 44.472 | -4.701 | 0.249 | 16.881 |
| ARKM | ta_event | VPIN_threshold | t_0.5 | chop | 3 | 13 | 0.692 | 3.686 | 55.247 | -10.506 | 0.261 | 16.788 |
| BLUR | measure_event | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 1 | 12 | 0.833 | 2.831 | 39.032 | -2.299 | 0.228 | 15.520 |