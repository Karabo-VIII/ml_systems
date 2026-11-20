> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# V7 + V1 OOS Validation Report (2026-05-23T12:56)

OOS window: 2024-05-16 -> 2026-05-19 (TRAIN cutoff: <= 2024-05-15)

Cost model: bucket-specific spot taker (BLUE 28bp / STEADY 32bp / VOLATILE 36bp / DEGEN 44bp), 24bp-class for FET/FIL/ICP class assets via per-asset bucketing.

Classifier: SURVIVED = OOS sign matches TRAIN AND |OOS|/|TRAIN| >= 0.30; DECAYED = OOS sign matches TRAIN but ratio < 0.30; INVERTED = OOS sign flipped; DEAD = thin/error/no fires.


## Classification headline

**V7 (12 engines)**: SURVIVED=2 DECAYED=0 INVERTED=10 DEAD=0

**V1 (32 engines)**: SURVIVED=2 DECAYED=1 INVERTED=25 DEAD=4


**Prior 30-engine basket** (top composite_score): INVERTED=25/30 = 83%
**V7 INVERTED rate**: 10/12 = 83%
**V1 INVERTED rate**: 25/32 = 78%


## V7 per-engine TRAIN -> OOS

| asset | indicator | config | regime | hold | TRAIN cmp% | OOS n | OOS cmp% | ratio | class |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 3 | +62.6 | 86 | -83.0 | -1.33 | INVERTED |
| APT | MA_state_EMA_above | period_20 | bull | 3 | +34.4 | 109 | -88.2 | -2.56 | INVERTED |
| AR | MA_state_SMA_above | period_20 | bull | 1 | +52.3 | 107 | -80.2 | -1.53 | INVERTED |
| AR | measure_engines/te_in_btc | op_abs_gt_thr_1.0 | bull | 3 | +44.7 | 71 | -51.6 | -1.15 | INVERTED |
| DOT | Distance_z_state | period_50_threshold_1.5 | bull | 1 | +42.6 | 72 | -55.2 | -1.30 | INVERTED |
| FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | 1 | +77.4 | 65 | -8.1 | -0.11 | INVERTED |
| FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | 1 | +69.3 | 31 | -11.6 | -0.17 | INVERTED |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 1 | +75.8 | 61 | -20.6 | -0.27 | INVERTED |
| FET | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | 1 | +67.4 | 31 | -25.0 | -0.37 | INVERTED |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | 1 | +38.6 | 34 | +13.3 | +0.34 | SURVIVED |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 1 | +38.6 | 32 | +14.2 | +0.37 | SURVIVED |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 1 | +167.0 | 65 | -51.9 | -0.31 | INVERTED |

## V1 per-engine TRAIN -> OOS (sorted by OOS compound)

| asset | indicator | config | regime | hold | TRAIN cmp% | OOS n | OOS cmp% | ratio | class |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.5 | bull | 1 | +39.5 | 25 | +31.1 | +0.79 | SURVIVED |
| AVAX | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bull | 1 | +38.1 | 102 | +10.9 | +0.29 | DECAYED |
| WLD | YZ_vol_regime | t_0.5 | bull | 1 | +11.2 | 80 | +3.9 | +0.34 | SURVIVED |
| SUPER | measure_engines/te_imb | op_gt_thr_1.0 | bull | 1 | +207.5 | 70 | -4.4 | -0.02 | INVERTED |
| FET | measure_engines/norm_flow_imbalance | op_gt_thr_1.0 | bull | 1 | +75.9 | 41 | -12.1 | -0.16 | INVERTED |
| SOL | measure_engines/te_imb | op_abs_gt_thr_1.0 | chop | 1 | +21.9 | 103 | -16.8 | -0.77 | INVERTED |
| AAVE | measure_engines/rv_jump_frac | op_gt_thr_1.0 | chop | 1 | +23.4 | 40 | -19.4 | -0.83 | INVERTED |
| PEPE | MA_state_SMA_above | period_100 | chop | 1 | +23.1 | 95 | -19.4 | -0.84 | INVERTED |
| FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | 1 | +75.8 | 61 | -20.6 | -0.27 | INVERTED |
| LTC | RSI_threshold | p_5_lo_40_hi_65 | bear | 1 | +14.1 | 128 | -20.9 | -1.48 | INVERTED |
| SOL | Distance_z_state | period_20_threshold_1.5 | bull | 3 | +58.6 | 86 | -27.1 | -0.46 | INVERTED |
| ADA | VWAP_state_above | period_20 | bull | 3 | +62.5 | 116 | -32.3 | -0.52 | INVERTED |
| HBAR | measure_engines/norm_efficiency | op_abs_gt_thr_1.0 | bull | 1 | +44.3 | 94 | -33.7 | -0.76 | INVERTED |
| FET | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 1 | +33.2 | 74 | -38.5 | -1.16 | INVERTED |
| ADA | VPIN_threshold | t_1.0 | bull | 3 | +49.3 | 82 | -42.9 | -0.87 | INVERTED |
| ICP | measure_engines/bd_imbalance_l1 | op_abs_gt_thr_1.0 | bull | 1 | +58.6 | 92 | -43.9 | -0.75 | INVERTED |
| NEAR | measure_engines/wh_whale_trade_count_5 | op_abs_gt_thr_1.0 | bull | 1 | +104.8 | 71 | -51.1 | -0.49 | INVERTED |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 1 | +167.0 | 65 | -51.9 | -0.31 | INVERTED |
| FET | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.0 | bull | 1 | +95.0 | 84 | -53.8 | -0.57 | INVERTED |
| ICP | measure_engines/te_in_btc | op_abs_gt_thr_1.5 | bull | 1 | +58.2 | 78 | -54.8 | -0.94 | INVERTED |
| BTC | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | bull | 1 | +28.6 | 126 | -57.5 | -2.01 | INVERTED |
| FET | measure_engines/norm_flow_imbalance | op_abs_gt_thr_1.5 | bull | 1 | +40.9 | 40 | -59.7 | -1.46 | INVERTED |
| ENJ | Kyle_lambda_threshold | t_1.5 | bull | 1 | +19.8 | 51 | -59.8 | -3.03 | INVERTED |
| ZEC | measure_engines/bd_imbalance_l5 | op_abs_gt_thr_1.0 | bull | 1 | +21.7 | 108 | -71.6 | -3.30 | INVERTED |
| ICP | measure_engines/te_in_btc | op_gt_thr_1.0 | bull | 1 | +53.0 | 74 | -73.9 | -1.39 | INVERTED |
| FLOKI | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 3 | +92.6 | 60 | -77.3 | -0.84 | INVERTED |
| ZEC | ETF_flow_z | t_0.5 | bull | 1 | +19.4 | 187 | -89.5 | -4.62 | INVERTED |
| LINK | MACD_threshold | f_12_s_35_g_9 | chop | 3 | +44.5 | 224 | -97.8 | -2.20 | INVERTED |
| FET | measure_engines/xd_btc_return | op_abs_gt_thr_1.0 | bull | 1 | +9.8 | 0 | +0.0 | +0.00 | DEAD |
| ICP | measure_engines/xd_btc_volatility | op_abs_gt_thr_1.0 | bull | 1 | +80.0 | 0 | +0.0 | +0.00 | DEAD |
| FIL | measure_engines/xd_momentum_rank | op_abs_gt_thr_1.0 | bull | 1 | +40.9 | 0 | +0.0 | +0.00 | DEAD |
| FIL | measure_engines/xd_ma_distance | op_abs_gt_thr_1.0 | bull | 1 | +59.8 | 0 | +0.0 | +0.00 | DEAD |

## Per-asset OOS classification (V7)

| asset   |   INVERTED |   SURVIVED |   total |
|:--------|-----------:|-----------:|--------:|
| APT     |          2 |          0 |       2 |
| AR      |          2 |          0 |       2 |
| DOT     |          1 |          0 |       1 |
| FET     |          4 |          0 |       4 |
| FIL     |          0 |          2 |       2 |
| ICP     |          1 |          0 |       1 |


## Per-asset OOS classification (V1)

| asset   |   DEAD |   DECAYED |   INVERTED |   SURVIVED |   total |
|:--------|-------:|----------:|-----------:|-----------:|--------:|
| AAVE    |      0 |         0 |          1 |          0 |       1 |
| ADA     |      0 |         0 |          2 |          0 |       2 |
| AVAX    |      0 |         1 |          0 |          0 |       1 |
| BTC     |      0 |         0 |          1 |          0 |       1 |
| ENJ     |      0 |         0 |          1 |          0 |       1 |
| FET     |      1 |         0 |          5 |          1 |       7 |
| FIL     |      2 |         0 |          0 |          0 |       2 |
| FLOKI   |      0 |         0 |          1 |          0 |       1 |
| HBAR    |      0 |         0 |          1 |          0 |       1 |
| ICP     |      1 |         0 |          4 |          0 |       5 |
| LINK    |      0 |         0 |          1 |          0 |       1 |
| LTC     |      0 |         0 |          1 |          0 |       1 |
| NEAR    |      0 |         0 |          1 |          0 |       1 |
| PEPE    |      0 |         0 |          1 |          0 |       1 |
| SOL     |      0 |         0 |          2 |          0 |       2 |
| SUPER   |      0 |         0 |          1 |          0 |       1 |
| WLD     |      0 |         0 |          0 |          1 |       1 |
| ZEC     |      0 |         0 |          2 |          0 |       2 |


## Per-regime OOS classification (V7)

| btc_regime_30d   |   INVERTED |   SURVIVED |   total |
|:-----------------|-----------:|-----------:|--------:|
| bull             |         10 |          2 |      12 |


## Per-regime OOS classification (V1)

| btc_regime_30d   |   DEAD |   DECAYED |   INVERTED |   SURVIVED |   total |
|:-----------------|-------:|----------:|-----------:|-----------:|--------:|
| bear             |      0 |         0 |          1 |          0 |       1 |
| bull             |      4 |         1 |         18 |          2 |      25 |
| chop             |      0 |         0 |          6 |          0 |       6 |

## Honest verdict

- Prior 30-engine top-composite basket OOS-inversion rate: 83%
- V7 OOS-inversion rate: 83% (10/12)
- V1 OOS-inversion rate: 78% (25/32)
- V7 SURVIVED rate: 17%
- V1 SURVIVED rate: 6%