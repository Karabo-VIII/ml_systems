# Move-Lifecycle FULL Mining Summary (2026-05-23T09:09)

- 173 measures × all TRAIN-eligible assets × 4 lags + t-0 + EXIT classification
- pre_rows: 27,196, in_rows: 6,803, exit_rows: 6,803

## TOP-15 PRE-MOVE LEADERS (largest z_lift at t-3, sign-consistent)

| asset | measure | lag | z_lift | n_movers |
|---|---|---:|---:|---:|
| ETC | stbl_total_delta_30d_pct | t-3 | +0.579 | 1010 |
| MOVR | liq_delta_z30 | t-3 | +0.537 | 39 |
| MOVR | liq_short_z30 | t-3 | +0.527 | 39 |
| CHZ | stbl_total_delta_30d_pct | t-3 | +0.518 | 903 |
| SEI | stbl_usde_zscore_30d | t-3 | +0.515 | 58 |
| ETC | etf_btc_etf_total_7d_z | t-3 | -0.510 | 39 |
| LTC | bd_imbalance_l5 | t-3 | +0.502 | 61 |
| WLD | etf_btc_etf_total_7d_z | t-3 | +0.497 | 46 |
| ENJ | etf_btc_etf_total_7d_z | t-3 | +0.493 | 35 |
| ETH | xrel_liq_long_usd_xrank | t-3 | -0.492 | 1135 |
| BCH | stbl_total_delta_30d_pct | t-3 | +0.485 | 1052 |
| SOL | xex_ok_bn_spread_bps | t-3 | -0.478 | 62 |
| LTC | bd_notional_skew | t-3 | +0.478 | 61 |
| BNB | stbl_total_delta_30d_pct | t-3 | +0.463 | 1089 |
| ETH | xrel_liq_long_usd_xpct10 | t-3 | -0.461 | 1135 |

## TOP-15 IN-MOVE (t-0) CONCURRENT (sign-consistent)

| asset | measure | z_lift_t0 | n_movers |
|---|---|---:|---:|
| XRP | xrel_liq_long_usd_xrank | -0.795 | 1023 |
| XRP | xrel_liq_long_usd_xratio | -0.721 | 1023 |
| XRP | liq_long_xsec_z | -0.696 | 1023 |
| XRP | liq_short_xsec_z | -0.673 | 1023 |
| ATOM | xrel_hbr_n_trades_xrank | -0.572 | 940 |
| ETC | stbl_total_delta_30d_pct | +0.571 | 1013 |
| ETH | xrel_hbr_n_trades_xratio | -0.530 | 1138 |
| CHZ | stbl_total_delta_30d_pct | +0.521 | 906 |
| FET | etf_btc_etf_total_z30 | +0.491 | 52 |
| BCH | stbl_total_delta_30d_pct | +0.485 | 1055 |
| BTC | hbr_eta_sell | -0.484 | 1131 |
| ETH | xrel_liq_long_usd_xpct10 | -0.477 | 1138 |
| BCH | xrel_hbr_n_trades_xrank | +0.477 | 1055 |
| DASH | stbl_total_delta_30d_pct | +0.472 | 1057 |
| ADA | stbl_total_delta_30d_pct | +0.465 | 1026 |

## TOP-15 EXIT-PREDICTORS (lift CONTINUES vs REVERTS)

| asset | measure | lift_cont_vs_rev | n_continues | n_reverts |
|---|---|---:|---:|---:|
| CRV | etf_any_inflow_shock | -1.062 | 17 | 6 |
| CRV | etf_btc_etf_inflow_shock | -1.062 | 17 | 6 |
| MOVR | liq_short_panic | -0.955 | 21 | 7 |
| MOVR | liq_short_spike | -0.955 | 21 | 7 |
| LINK | bd_total_depth_l5_p10 | +0.950 | 21 | 12 |
| CRV | etf_btc_etf_total_usdm | -0.926 | 17 | 6 |
| SOL | etf_btc_etf_total_z30 | +0.900 | 27 | 12 |
| ZEN | etf_btc_etf_outflow_shock | +0.885 | 18 | 8 |
| ETH | bd_depth_l1pct_mean | +0.882 | 21 | 17 |
| BNB | etf_btc_etf_total_usdm | +0.875 | 24 | 12 |
| ETH | etf_btc_etf_mega_inflow | +0.873 | 18 | 15 |
| TRX | bd_n_snapshots | +0.862 | 48 | 32 |
| ETH | etf_any_inflow_shock | +0.859 | 18 | 15 |
| ETH | etf_btc_etf_inflow_shock | +0.859 | 18 | 15 |
| MOVR | s3_oi_usd | -0.856 | 23 | 7 |

## TOP-10 UNIVERSAL PRE-MOVE LEADERS (median |z_lift| at t-3 across all assets)

| measure | n_assets | median_abs_lift | mean_lift | n_sign_consistent |
|---|---:|---:|---:|---:|
| stbl_total_delta_30d_pct | 12 | +0.298 | +0.283 | 12 |
| liq_long_xsec_z | 10 | +0.221 | -0.191 | 10 |
| s3_top_pos_lsr | 7 | +0.219 | -0.215 | 7 |
| etf_btc_etf_total_usdm | 18 | +0.199 | +0.055 | 18 |
| stbl_total_delta_7d_pct | 7 | +0.196 | +0.198 | 7 |
| etf_btc_etf_inflow_shock | 23 | +0.186 | +0.039 | 23 |
| etf_any_inflow_shock | 23 | +0.186 | +0.039 | 23 |
| stbl_usde_zscore_30d | 27 | +0.181 | +0.007 | 27 |
| bd_n_snapshots | 15 | +0.180 | -0.016 | 15 |
| liq_short_xsec_z | 10 | +0.175 | -0.137 | 10 |