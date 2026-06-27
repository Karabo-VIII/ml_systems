# MA CONFIG LEDGER — every config, working and not (OOS-2024, prod-grade)

Each config evaluated INDIVIDUALLY (standalone, n=1) through the prod allocator
(inv-vol + 15% cap + regime gate + prod cost 0.0028), confirmed + slice-held.
OOS-2024 (2024-05-15..2024-12-31, 2025 sealed). lin/wk = linear %/week.

NOTE: ranking here is ON OOS (descriptive ledger, not a selection rule). For the
HONEST deployable top-K (selected on TRAIN+VAL), see V11_topk_edge.


## Timeframe: 1d

| MA type | configs | positive | best | top5-avg | top10-avg | band-avg | median | worst |
|---|---|---|---|---|---|---|---|---|
| SMA | 84 | 84/84 | +2.37 | +2.29 | +2.17 | +1.46 | +1.49 | +0.55 |
| EMA | 105 | 105/105 | +2.15 | +2.10 | +2.07 | +1.73 | +1.76 | +1.15 |
| WMA | 106 | 106/106 | +2.42 | +2.28 | +2.20 | +1.63 | +1.64 | +0.71 |
| VIDYA | 74 | 74/74 | +2.15 | +2.10 | +2.04 | +1.63 | +1.68 | +0.92 |
| KAMA | 74 | 74/74 | +2.32 | +2.20 | +2.09 | +1.48 | +1.46 | +0.61 |
| DEMA | 114 | 114/114 | +2.38 | +2.31 | +2.23 | +1.49 | +1.49 | +0.65 |
| HMA | 108 | 104/108 | +2.26 | +2.12 | +1.94 | +1.15 | +1.18 | -0.41 |
| TEMA | 112 | 112/112 | +2.47 | +2.30 | +2.22 | +1.44 | +1.40 | +0.65 |

## Timeframe: 4h

| MA type | configs | positive | best | top5-avg | top10-avg | band-avg | median | worst |
|---|---|---|---|---|---|---|---|---|
| SMA | 115 | 106/115 | +1.91 | +1.85 | +1.75 | +0.94 | +1.03 | -0.83 |
| EMA | 120 | 118/120 | +1.83 | +1.77 | +1.71 | +1.09 | +1.09 | -0.17 |
| WMA | 114 | 108/114 | +1.91 | +1.84 | +1.78 | +1.05 | +1.14 | -0.53 |
| VIDYA | 115 | 115/115 | +2.09 | +1.94 | +1.89 | +1.34 | +1.29 | +0.92 |
| KAMA | 111 | 109/111 | +1.94 | +1.86 | +1.81 | +1.24 | +1.27 | -0.52 |
| DEMA | 111 | 92/111 | +1.86 | +1.71 | +1.65 | +0.82 | +1.16 | -2.37 |
| HMA | 109 | 77/109 | +2.07 | +1.86 | +1.77 | +0.42 | +0.74 | -2.98 |
| TEMA | 107 | 81/107 | +2.09 | +1.95 | +1.86 | +0.72 | +1.14 | -2.61 |

## Top-5 configs per MA type — 1d

- **SMA**: ema_2_3 (+2.37, Sh2.3) | ema_62_105 (+2.37, Sh2.3) | ema_2_3_4 (+2.28, Sh2.2) | ema_22_151 (+2.25, Sh2.2) | ema_12_178 (+2.19, Sh2.1)
- **EMA**: ema_37_89 (+2.15, Sh2.1) | ema_24_43_60 (+2.12, Sh2.0) | ema_62_105 (+2.11, Sh2.1) | ema_26_75 (+2.07, Sh2.0) | ema_31_53 (+2.06, Sh1.9)
- **WMA**: ema_48_67_118 (+2.42, Sh2.4) | ema_73_145 (+2.35, Sh2.3) | ema_24_106_118 (+2.26, Sh2.2) | ema_3_75_132 (+2.20, Sh2.1) | ema_31_124 (+2.17, Sh2.1)
- **VIDYA**: ema_15_28 (+2.15, Sh2.1) | ema_18_23 (+2.14, Sh2.1) | ema_22_28 (+2.14, Sh2.1) | ema_22_65 (+2.04, Sh2.0) | ema_12_33 (+2.02, Sh1.9)
- **KAMA**: ema_2_5_118 (+2.32, Sh2.1) | ema_15_151 (+2.26, Sh2.2) | ema_26_172 (+2.18, Sh2.1) | ema_12_178 (+2.13, Sh2.0) | ema_26_75 (+2.10, Sh2.0)
- **DEMA**: ema_86_122 (+2.38, Sh2.3) | ema_102_103 (+2.35, Sh2.2) | ema_24_106_118 (+2.32, Sh2.2) | ema_44_203 (+2.27, Sh2.1) | ema_75_132_148 (+2.22, Sh2.2)
- **HMA**: ema_5_12 (+2.26, Sh2.2) | ema_6_9_24 (+2.23, Sh2.2) | ema_5_10_14 (+2.17, Sh2.1) | ema_6_13 (+1.99, Sh1.9) | ema_4_15 (+1.92, Sh1.9)
- **TEMA**: ema_102_237 (+2.47, Sh2.3) | ema_75_132_148 (+2.30, Sh2.2) | ema_5_118_208 (+2.29, Sh2.2) | ema_5_10_14 (+2.27, Sh2.2) | ema_30_132_186 (+2.19, Sh2.1)

## Top-5 configs per MA type — 4h

- **SMA**: ema_8_91 (+1.91, Sh1.8) | ema_4_86 (+1.90, Sh1.8) | ema_12_77 (+1.84, Sh1.8) | ema_5_86 (+1.83, Sh1.7) | ema_15_65 (+1.78, Sh1.7)
- **EMA**: ema_19_27_38 (+1.83, Sh1.8) | ema_26_32 (+1.82, Sh1.7) | ema_10_34_60 (+1.76, Sh1.7) | ema_18_55 (+1.74, Sh1.7) | ema_15_22_48 (+1.69, Sh1.6)
- **WMA**: ema_12_22_118 (+1.91, Sh2.0) | ema_10_128 (+1.88, Sh1.8) | ema_37_89 (+1.82, Sh1.8) | ema_4_30_67 (+1.80, Sh1.8) | ema_6_34_94 (+1.79, Sh1.8)
- **VIDYA**: ema_5_12_208 (+2.09, Sh2.0) | ema_4_15_208 (+1.98, Sh1.9) | ema_10_11_233 (+1.92, Sh1.9) | ema_5_12 (+1.87, Sh1.8) | ema_8_16 (+1.87, Sh1.8)
- **KAMA**: ema_4_86 (+1.94, Sh1.9) | ema_8_91 (+1.90, Sh1.9) | ema_8_14_38 (+1.83, Sh1.8) | ema_3_75_132 (+1.82, Sh1.8) | ema_8_39 (+1.80, Sh1.8)
- **DEMA**: ema_10_34_60 (+1.86, Sh1.9) | ema_12_43_75 (+1.71, Sh1.8) | ema_26_75 (+1.68, Sh1.7) | ema_15_34_186 (+1.66, Sh1.7) | ema_10_128 (+1.64, Sh1.6)
- **HMA**: ema_12_106_148 (+2.07, Sh2.2) | ema_15_84_148 (+1.82, Sh1.9) | ema_26_172 (+1.82, Sh1.9) | ema_37_203 (+1.80, Sh1.8) | ema_24_106_118 (+1.79, Sh1.9)
- **TEMA**: ema_52_89 (+2.09, Sh2.1) | ema_31_124 (+2.04, Sh2.1) | ema_30_43_186 (+1.92, Sh2.0) | ema_30_132_186 (+1.85, Sh2.0) | ema_22_151 (+1.84, Sh1.9)

## Reading
- The edge is BROAD: ~100% of configs individually positive in most bands (HMA the
  exception with a few negatives). The band edge is NOT carried by a few lucky configs.
- Individual top configs (~+2.2-2.5%/wk at 1d) OUTPERFORM the full-band ensemble
  (~+1.6%/wk) because the band averages across 100+ configs of varying quality.
- 1d configs > 4h configs on raw lin/wk (top 1d ~+2.4, top 4h ~+1.8).
- CAVEAT: this ledger ranks on OOS (descriptive). Deployable selection must rank on
  TRAIN+VAL and hold OOS (V11_topk_edge) to avoid look-ahead bias.