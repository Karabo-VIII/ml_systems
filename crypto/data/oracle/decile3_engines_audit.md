# Decile-3 Engines Audit (2026-05-23T13:37)

## Why decile 3?
- Per F36.c: TRAIN decile 3 (25-40th percentile by TRAIN compound) has best VAL mean compound (+61.20%)
- The catalog's 'best by TRAIN compound' filter SELECTS DECILE 9 — the WORST VAL decile
- Mining decile 3 is the corrective: find engines moderate on TRAIN, strong on VAL

## Decile 3 best (top 15 by VAL compound)

| Asset | Class | Config | Regime | TRAIN comp | VAL comp | VAL fires | VAL hit |
|---|---|---|---|---:|---:|---:|---:|
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | 18.1% | **+258.4%** | 38 | 0.61
| LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | 18.1% | **+226.4%** | 37 | 0.62
| LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | 18.1% | **+213.5%** | 38 | 0.61
| LINK | RSI_threshold | p_7_lo_40_hi_60 | chop | 18.1% | **+212.0%** | 38 | 0.61
| LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | 18.1% | **+204.2%** | 34 | 0.65
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | 18.1% | **+189.0%** | 35 | 0.63
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 18.1% | **+177.0%** | 33 | 0.61
| JST | RSI_threshold | p_7_lo_35_hi_70 | chop | 19.2% | **+39.4%** | 28 | 0.50
| JST | RSI_threshold | p_10_lo_40_hi_65 | chop | 18.2% | **+28.6%** | 33 | 0.48
| JST | RSI_threshold | p_6_lo_40_hi_75 | chop | 17.8% | **+26.3%** | 42 | 0.45
| JST | RSI_threshold | p_9_lo_40_hi_65 | chop | 17.6% | **+25.1%** | 36 | 0.47
| JST | RSI_threshold | p_7_lo_40_hi_70 | chop | 19.2% | **+20.6%** | 41 | 0.44
| JST | RSI_threshold | p_12_lo_40_hi_65 | chop | 19.3% | **+18.8%** | 27 | 0.44
| JST | RSI_threshold | p_13_lo_40_hi_65 | chop | 19.5% | **+18.8%** | 27 | 0.44
| SOL | Donchian_state_above_midline | period_20 | chop | 17.6% | **+1.8%** | 23 | 0.57

## Decile 0 best (top 15 by VAL compound)

Decile 0 = WORST TRAIN compound. Even here, some engines positive on VAL.

| Asset | Class | Config | Regime | TRAIN comp | VAL comp | VAL fires | VAL hit |
|---|---|---|---|---:|---:|---:|---:|
| DASH | OBV_zscore | p_100_t_1.5 | chop | 10.0% | **+45.0%** | 31 | 0.58
| DYDX | OBV_zscore | p_30_t_1.0 | bull | 8.5% | **+37.1%** | 63 | 0.43
| JST | RSI_threshold | p_6_lo_40_hi_80 | chop | 8.2% | **+30.2%** | 40 | 0.47
| JST | RSI_threshold | p_10_lo_40_hi_70 | chop | 10.9% | **+26.8%** | 32 | 0.47
| ICP | RSI_threshold | p_6_lo_40_hi_65 | bull | 10.9% | **+26.6%** | 48 | 0.52
| JST | RSI_threshold | p_9_lo_40_hi_70 | chop | 10.9% | **+25.8%** | 35 | 0.49
| APT | RSI_threshold | p_29_lo_40_hi_60 | bull | 11.7% | **+23.3%** | 34 | 0.44
| JST | RSI_threshold | p_6_lo_35_hi_75 | chop | 9.1% | **+20.1%** | 33 | 0.42
| ETC | measure_engines/norm_efficiency | op_abs_gt_thr_1.0 | chop | 8.1% | **+16.4%** | 27 | 0.48
| APT | RSI_threshold | p_27_lo_40_hi_60 | bull | 10.3% | **+13.4%** | 37 | 0.43
| APT | RSI_threshold | p_28_lo_40_hi_60 | bull | 10.3% | **+11.3%** | 36 | 0.42
| SHIB | measure_engines/bs_basis_z30 | op_abs_gt_thr_1.0 | chop | 9.7% | **+2.9%** | 33 | 0.52
| APT | RSI_threshold | p_16_lo_40_hi_65 | bull | 9.4% | **+2.8%** | 41 | 0.44
| WLD | YZ_vol_regime | t_0.5 | bull | 11.2% | **+2.1%** | 27 | 0.56
| APT | RSI_threshold | p_25_lo_40_hi_60 | bull | 11.4% | **-1.6%** | 40 | 0.40

## Decile 3 class distribution: {'RSI_threshold': 15, 'Donchian_state_above_midline': 2, 'MA_state_SMA_above': 2, 'OBV_zscore': 1, 'confluence_engines/UNI_pair_4': 1, 'Bollinger_band_breach': 1, 'MA_state_EMA_above': 1, 'ETF_flow_z': 1}

## Decile 3 asset distribution: {'JST': 7, 'LINK': 7, 'SOL': 2, 'PEPE': 2, 'DASH': 2, 'DOT': 1, 'UNI': 1, 'XRP': 1, 'ZEC': 1}

## Decile 3 robust filter (val_fires≥10, val_fold_consist=True, val_compound>5%): 7

| Asset | Class | Config | Regime | TRAIN | VAL | VAL fires |
|---|---|---|---|---:|---:|---:|
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 18.1% | +177.0% | 33 |
| LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | 18.1% | +213.5% | 38 |
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | 18.1% | +189.0% | 35 |
| LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | 18.1% | +226.4% | 37 |
| LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | 18.1% | +204.2% | 34 |
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | 18.1% | +258.4% | 38 |
| LINK | RSI_threshold | p_7_lo_40_hi_60 | chop | 18.1% | +212.0% | 38 |