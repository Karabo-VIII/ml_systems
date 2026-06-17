# Bear/Chop Regime Mining Report (Oracle POV #10)

Generated: 2026-05-23  TRAIN end: 2024-05-15

## Catalog regime coverage
| Regime | All | Catch-tier |
|---|---|---|
| bull | 341 | 113 |
| chop | 523 | 100 |
| bear | 73 | 21 |

## Basket TRAIN-realized (close.pct_change return, 24bp RT, 25% sizing, top-3/day)
| Basket | n_engines | days | mean %/d | Sharpe | NAV |
|---|---|---|---|---|---|
| BULL (V7-proxy) | 2 | 123 | 0.6346 | 4.148 | 2.0709 |
| CHOP-only       | 10 | 119 | 0.3728 | 2.724 | 1.4955 |
| BEAR-only       | 10 | 62 | 0.4842 | 5.183 | 1.3364 |
| COMPOSITE       | — | 304 | 0.5014 | 3.686 | 4.1388 |

## Top 5 BEAR engines
- LTC RSI_threshold cfg=p_5_lo_40_hi_65  mean=0.755% n=724  L/S=382/342
- LTC RSI_threshold cfg=p_5_lo_40_hi_70  mean=0.714% n=650  L/S=348/302
- LTC RSI_threshold cfg=p_5_lo_40_hi_75  mean=0.692% n=596  L/S=322/274
- LTC RSI_threshold cfg=p_5_lo_35_hi_65  mean=0.631% n=626  L/S=334/292
- LTC RSI_threshold cfg=p_5_lo_25_hi_65  mean=0.576% n=418  L/S=210/208

## Top 5 CHOP engines
- SOL MA_state_SMA_above cfg=period_20  mean=0.539% n=406  L/S=238/168
- SOL Donchian_state_above_midline cfg=period_20  mean=0.526% n=426  L/S=260/166
- LINK RSI_threshold cfg=p_6_lo_40_hi_60  mean=0.471% n=1686  L/S=1050/636
- LINK MACD_threshold cfg=f_8_s_21_g_9  mean=0.465% n=3032  L/S=1896/1136
- LINK MACD_threshold cfg=f_12_s_35_g_9  mean=0.465% n=3032  L/S=1896/1136

## Honest gap audit
- bear catch-tier engines (catalog): 21
- chop catch-tier engines (catalog): 100
- bear basket fold-sign-consistent: 0/10
- chop basket fold-sign-consistent: 9/10
- median n_fires bear: 580; chop: 1462 (statistical-power concern)
- bear long-share: 53.25%; chop long-share: 61.95% (1.0=long-rebounds; 0.0=short)

## Caveats
- BEAR sample tiny: bear catch-tier = 21 engines vs 113 bull (5.4x imbalance).
- BTC bear regime is ~15% of TRAIN days -> per-engine n_fires often <10.
- 'mean_pnl_fixed' uses close.pct_change.shift(-1) directly (target_return_1_raw is corrupted).
- Composite uses event's own btc_regime_30d label as routing key (in-sample regime ID).