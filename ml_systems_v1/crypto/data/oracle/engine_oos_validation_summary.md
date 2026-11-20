> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# Engine OOS validation -- 141 engines tested (2026-05-23T13:45)

OOS window: 2024-05-16 -> 2026-05-19

Status counts: {'ok': 133, 'error': 7, 'skip': 1}

## OOS-passing engines sorted by OOS compound

| asset | indicator_class | indicator_config | btc_regime_30d | hold_days | n_fires | compound_return_pct | oos_n_fires_oos | oos_compound_return_pct_oos | oos_train_oos_compound_ratio | oos_hit_rate_oos | oos_max_dd_pct_oos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| APT | RSI_threshold | p_5_lo_40_hi_65 | bull | 3 | 11 | 34.522 | 181.0 | 150.1 | 4.348 | 0.519 | -60.070 |
| ZEC | measure_engines/te_btc_imb | op_abs_gt_thr_1.0 | bear | 1 | 12 | 22.819 | 116.0 | 102.5 | 4.490 | 0.560 | -38.169 |
| PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop | 3 | 11 | 11.998 | 72.000 | 62.909 | 5.243 | 0.500 | -72.725 |
| APT | RSI_threshold | p_13_lo_35_hi_65 | bull | 1 | 10 | 12.872 | 67.000 | 58.270 | 4.527 | 0.507 | -14.028 |
| JST | RSI_threshold | p_7_lo_40_hi_75 | chop | 1 | 18 | 16.713 | 70.000 | 52.490 | 3.141 | 0.557 | -16.645 |
| JST | RSI_threshold | p_7_lo_35_hi_70 | chop | 1 | 17 | 19.246 | 69.000 | 48.438 | 2.517 | 0.536 | -15.655 |
| ARB | measure_engines/bs_basis_z30 | op_abs_gt_thr_1.0 | bull | 1 | 17 | 35.092 | 73.000 | 47.708 | 1.360 | 0.534 | -34.222 |
| FET | measure_engines/te_imb | op_abs_gt_thr_1.0 | bear | 1 | 10 | 36.436 | 100.0 | 37.405 | 1.027 | 0.560 | -30.107 |
| JST | RSI_threshold | p_11_lo_40_hi_65 | chop | 1 | 18 | 23.208 | 79.000 | 35.372 | 1.524 | 0.506 | -18.690 |
| DASH | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 1 | 16 | 17.194 | 90.000 | 32.791 | 1.907 | 0.544 | -34.361 |
| JST | RSI_threshold | p_8_lo_40_hi_70 | chop | 1 | 18 | 13.534 | 77.000 | 29.687 | 2.194 | 0.506 | -15.386 |
| JST | RSI_threshold | p_10_lo_40_hi_65 | chop | 1 | 18 | 18.222 | 83.000 | 28.244 | 1.550 | 0.506 | -21.245 |
| JST | RSI_threshold | p_7_lo_40_hi_70 | chop | 1 | 22 | 19.210 | 87.000 | 25.446 | 1.325 | 0.506 | -17.150 |
| JST | RSI_threshold | p_9_lo_40_hi_65 | chop | 1 | 20 | 17.576 | 87.000 | 24.906 | 1.417 | 0.506 | -21.123 |
| APT | RSI_threshold | p_12_lo_35_hi_65 | bull | 1 | 10 | 12.872 | 74.000 | 23.641 | 1.837 | 0.473 | -20.785 |
| JST | RSI_threshold | p_6_lo_40_hi_75 | chop | 1 | 22 | 17.818 | 85.000 | 21.443 | 1.203 | 0.529 | -17.591 |
| JST | RSI_threshold | p_12_lo_40_hi_65 | chop | 1 | 16 | 19.316 | 71.000 | 20.972 | 1.086 | 0.493 | -20.133 |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 1 | 10 | 22.619 | 36.000 | 16.003 | 0.707 | 0.556 | -46.713 |
| JST | RSI_threshold | p_13_lo_40_hi_65 | chop | 1 | 15 | 19.488 | 69.000 | 15.596 | 0.800 | 0.478 | -23.534 |
| APT | RSI_threshold | p_16_lo_40_hi_65 | bull | 1 | 11 | 9.432 | 89.000 | 8.399 | 0.890 | 0.449 | -28.860 |