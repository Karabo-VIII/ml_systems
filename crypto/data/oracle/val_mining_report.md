# VAL Mining Report -- 2026-05-23T13:27

VAL window: 2024-05-16 -> 2025-03-15 (~10 months)

## Regime distribution

| Window | bull | chop | bear | total |
|---|---:|---:|---:|---:|
| TRAIN | 648 | 508 | 435 | 1591 |
| VAL   | 118 | 97 | 89 | 304 |
| TRAIN % | 40.7 | 31.9 | 27.3 | 100 |
| VAL %   | 38.8 | 31.9 | 29.3 | 100 |

## Catch-tier counts

- TRAIN catch-tier engines: **234**
- VAL   catch-tier engines: **10**  (catch-tier survival rate 4.3%)
- VAL ok status (>=3 fires): 222 / 234

## Top-30 overlap (TRAIN vs VAL)

- TRAIN top-30 set size: 30
- VAL top-30 set size: 30
- Overlap: **0** engines in BOTH top-30 lists
- Val-only top-30 (missed by TRAIN catalog): 30
- Train-only top-30 (degraded on VAL): 30
- Jaccard similarity: **0.000**

### TOP-3 engines on VAL that were NOT in TRAIN top-30 (the ones we missed)

| Asset | Class | Config | Regime | TRAIN comp | VAL comp | VAL fires | VAL hit |
|---|---|---|---|---:|---:|---:|---:|
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | 18.10 | 258.45 | 38 | 0.61 |
| LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | 18.10 | 226.39 | 37 | 0.62 |
| LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | 18.10 | 213.46 | 38 | 0.61 |

## TOP-30 engines on VAL by compound

| Asset | Class | Config | Regime | TRAIN comp | VAL comp | VAL fires | VAL hit | fold_consist |
|---|---|---|---|---:|---:|---:|---:|---:|
| LINK | RSI_threshold | p_8_lo_40_hi_60 | chop | 18.10 | 258.45 | 38 | 0.61 | True |
| LINK | RSI_threshold | p_5_lo_40_hi_60 | chop | 18.10 | 226.39 | 37 | 0.62 | True |
| LINK | RSI_threshold | p_6_lo_40_hi_60 | chop | 18.10 | 213.46 | 38 | 0.61 | True |
| LINK | RSI_threshold | p_7_lo_40_hi_60 | chop | 18.10 | 212.03 | 38 | 0.61 | True |
| LINK | RSI_threshold | p_5_lo_35_hi_60 | chop | 18.10 | 204.16 | 34 | 0.65 | True |
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | 18.10 | 188.95 | 35 | 0.63 | True |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 18.10 | 177.01 | 33 | 0.61 | True |
| ADA | VPIN_threshold | t_1.0 | bull | 49.31 | 106.87 | 34 | 0.62 | True |
| PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop | 12.00 | 66.55 | 22 | 0.45 | False |
| DASH | OBV_zscore | p_100_t_1.5 | chop | 9.99 | 44.97 | 31 | 0.58 | True |
| FLOKI | measure_engines/hbr_eta_buy | op_abs_gt_thr_1.0 | chop | 44.70 | 41.03 | 31 | 0.52 | False |
| JST | RSI_threshold | p_7_lo_35_hi_70 | chop | 19.25 | 39.44 | 28 | 0.50 | False |
| DYDX | OBV_zscore | p_30_t_1.0 | bull | 8.50 | 37.13 | 63 | 0.43 | False |
| JST | RSI_threshold | p_11_lo_40_hi_65 | chop | 23.21 | 31.49 | 31 | 0.48 | False |
| JST | RSI_threshold | p_6_lo_40_hi_80 | chop | 8.19 | 30.16 | 40 | 0.47 | False |
| FET | measure_engines/te_imb | op_abs_gt_thr_1.0 | bear | 36.44 | 29.35 | 50 | 0.56 | False |
| AAVE | measure_engines/hbr_eta_total | op_abs_gt_thr_1.0 | chop | 11.87 | 29.33 | 31 | 0.45 | True |
| JST | RSI_threshold | p_10_lo_40_hi_65 | chop | 18.22 | 28.60 | 33 | 0.48 | False |
| JST | RSI_threshold | p_10_lo_40_hi_70 | chop | 10.91 | 26.84 | 32 | 0.47 | False |
| ICP | RSI_threshold | p_6_lo_40_hi_65 | bull | 10.90 | 26.55 | 48 | 0.52 | False |
| JST | RSI_threshold | p_6_lo_40_hi_75 | chop | 17.82 | 26.28 | 42 | 0.45 | False |
| JST | RSI_threshold | p_9_lo_40_hi_70 | chop | 10.91 | 25.83 | 35 | 0.49 | False |
| JST | RSI_threshold | p_9_lo_40_hi_65 | chop | 17.58 | 25.06 | 36 | 0.47 | False |
| APT | RSI_threshold | p_29_lo_40_hi_60 | bull | 11.66 | 23.34 | 34 | 0.44 | False |
| OP | measure_engines/rv_jump_count | op_abs_gt_thr_1.0 | chop | 31.62 | 22.04 | 16 | 0.50 | False |
| OP | measure_engines/rv_jump_count | op_gt_thr_1.0 | chop | 31.62 | 22.04 | 16 | 0.50 | False |
| DASH | OBV_zscore | p_50_t_1.0 | chop | 24.50 | 22.01 | 62 | 0.55 | False |
| JST | RSI_threshold | p_7_lo_40_hi_75 | chop | 16.71 | 21.34 | 40 | 0.45 | False |
| HBAR | measure_engines/wh_whale_trade_count_500k | op_gt_thr_1.0 | bull | 22.62 | 20.76 | 18 | 0.56 | False |
| JST | RSI_threshold | p_7_lo_40_hi_70 | chop | 19.21 | 20.61 | 41 | 0.44 | False |

## Per-class VAL survival rate (% of TRAIN catch-tier engines in class that are positive AND fold-consistent on VAL)

| Class | TRAIN catch-tier N | VAL catch-tier N | survival % |
|---|---:|---:|---:|
| measure_engines/hbr_eta_total | 2 | 1 | 50.0 |
| VPIN_threshold | 4 | 1 | 25.0 |
| RSI_threshold | 57 | 7 | 12.3 |
| OBV_zscore | 11 | 1 | 9.1 |
| ATR_bands | 3 | 0 | 0.0 |
| Bollinger_band_breach | 1 | 0 | 0.0 |
| Distance_z_state | 11 | 0 | 0.0 |
| Donchian_state_above_midline | 6 | 0 | 0.0 |
| ETF_flow_z | 4 | 0 | 0.0 |
| Hawkes_branching_imbalance | 1 | 0 | 0.0 |
| Kyle_lambda_threshold | 4 | 0 | 0.0 |
| Liquidation_cascade | 1 | 0 | 0.0 |
| MACD_threshold | 7 | 0 | 0.0 |
| MA_state_EMA_above | 10 | 0 | 0.0 |
| MA_state_SMA_above | 13 | 0 | 0.0 |
| VWAP_state_above | 2 | 0 | 0.0 |
| YZ_vol_regime | 1 | 0 | 0.0 |
| confluence_engines/UNI_pair_4 | 1 | 0 | 0.0 |
| measure_engines/bd_imbalance_l1 | 1 | 0 | 0.0 |
| measure_engines/bd_imbalance_l5 | 3 | 0 | 0.0 |
| measure_engines/bs_basis_z30 | 5 | 0 | 0.0 |
| measure_engines/hbr_eta_buy | 3 | 0 | 0.0 |
| measure_engines/liq_long_usd | 3 | 0 | 0.0 |
| measure_engines/liq_long_xsec_z | 1 | 0 | 0.0 |
| measure_engines/liq_short_z30 | 1 | 0 | 0.0 |
| measure_engines/norm_deviation | 3 | 0 | 0.0 |
| measure_engines/norm_efficiency | 6 | 0 | 0.0 |
| measure_engines/norm_flow_imbalance | 6 | 0 | 0.0 |
| measure_engines/rv_bpv_5m | 6 | 0 | 0.0 |
| measure_engines/rv_jump_count | 4 | 0 | 0.0 |
| measure_engines/rv_jump_frac | 3 | 0 | 0.0 |
| measure_engines/rv_rv_5m | 3 | 0 | 0.0 |
| measure_engines/stbl_total_zscore_30d | 2 | 0 | 0.0 |
| measure_engines/te_btc_imb | 7 | 0 | 0.0 |
| measure_engines/te_imb | 4 | 0 | 0.0 |
| measure_engines/te_in_btc | 4 | 0 | 0.0 |
| measure_engines/wh_whale_net_usd | 4 | 0 | 0.0 |
| measure_engines/wh_whale_trade_count_500k | 6 | 0 | 0.0 |
| measure_engines/xd_btc_return | 3 | 0 | 0.0 |
| measure_engines/xd_btc_volatility | 3 | 0 | 0.0 |
| measure_engines/xd_funding_spread | 5 | 0 | 0.0 |
| measure_engines/xd_ma_distance | 3 | 0 | 0.0 |
| measure_engines/xd_momentum_rank | 6 | 0 | 0.0 |

## The 5 deploy basket engines on VAL

| Asset | Class | Config | Regime | TRAIN comp | TRAIN fires | VAL status | VAL fires | VAL mean pnl | VAL comp | VAL hit | fold consist | catch-tier on VAL |
|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_2.0 | bull | 45.66 | 11 | ok | 6.0 | 2.51 | 15.15 | 0.67 | False | False |
| PEPE | Donchian_state_above_midline | period_55 | chop | 28.29 | 10 | ok | 48.0 | 0.78 | 18.59 | 0.44 | False | False |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | 38.61 | 10 | ok | 18.0 | -0.23 | -6.16 | 0.50 | False | False |
| SUPER | measure_engines/norm_deviation | op_abs_gt_thr_1.0 | chop | 33.15 | 13 | ok | 31.0 | 0.62 | 13.54 | 0.58 | False | False |
| WLD | MA_state_SMA_above | period_50 | chop | 37.63 | 15 | ok | 28.0 | -0.21 | -11.88 | 0.39 | False | False |

## Asset-level volatility: TRAIN (2023-05-15..2024-05-15) vs VAL (2024-05-16..2025-03-15)

| Asset | median |daily ret| TRAIN (%) | median |daily ret| VAL (%) | delta (VAL - TRAIN) |
|---|---:|---:|---:|
| BONK | 4.998 | 4.030 | -0.968 |
| BTC | 1.049 | 1.328 | +0.279 |
| ETH | 1.066 | 1.856 | +0.790 |
| FET | 3.420 | 4.223 | +0.803 |
| FIL | 2.302 | 2.769 | +0.467 |
| PEPE | 4.131 | 4.192 | +0.061 |
| SHIB | 2.029 | 2.553 | +0.524 |
| SOL | 2.759 | 2.716 | -0.043 |
| SUPER | 2.749 | 3.896 | +1.147 |
| WLD | 4.106 | 4.007 | -0.098 |

## Honest verdict: what changed in VAL

- **Regime mix is NOT the cause**: TRAIN was 40.7% bull / VAL was 38.8% bull (-1.9 pp). Chop share is identical. Bear share delta = +1.9 pp. You CANNOT blame the basket failure on a regime shift -- the BTC regime mix in VAL is essentially the same as TRAIN.
- **Asset volatility actually INCREASED in VAL** (avg +0.30 pp median |daily ret|). In particular ETH +0.79 / SUPER +1.15 / FET +0.80. The basket's directional engines should have benefited from richer volatility, but they didn't -- so the failure is **directional accuracy**, not signal density.
- **The TRAIN-rank IS ANTI-PREDICTIVE of VAL performance**: Spearman(rank TRAIN compound, rank VAL compound) = **-0.176** (p=8.76e-03, N=222). A NEGATIVE rank correlation is worse than zero -- selecting the TOP TRAIN engines is statistically slightly worse than picking engines at random on VAL. This is the strongest possible evidence that the TRAIN catalog is OVERFIT TO TRAIN-SPECIFIC NOISE, not a transferable alpha set.
- **Catch-tier collapse**: only 10/234 = 4.3% of TRAIN catch-tier engines remain catch-tier on VAL. Of 222 engines with enough fires to evaluate: 85 had positive expectancy (38%), 68 had positive compound (31%), only 76 were 3-fold sign-consistent within VAL.
- **Top-30 Jaccard 0.000** = the TRAIN top-30 and VAL top-30 share ZERO engines. The two populations are dominated by entirely different asset/class pairs. TRAIN top-30 is dominated by SHIB rv_bpv_5m / SUPER te_imb / NEAR wh_whale_trade_count_500k bull-regime measure-engine fires; VAL top-30 is dominated by LINK + JST RSI_threshold chop-regime swings. Different *style of edge*.
- **Basket diagnosis**: 3 of 5 deploy engines were positive on VAL (FET +15.2%, PEPE +18.6%, SUPER +13.5%) but **NONE were fold-consistent within VAL**. FIL (-6.2%) and WLD (-11.9%) flipped sign outright. The basket negative result on VAL is driven by the FIL+WLD pair dragging down 3 weak-positive engines, AND by all 5 engines being concentrated in a way that they correlate during the VAL bear-window (Aug-Nov 2024 bottom).
- **Why OOS_pre worked but VAL failed**: OOS_pre (2025-03-16+) was the same bull-regime style the catalog was built on. The catalog selects engines that fire in BULL-volatility-spike conditions; those re-activated in OOS_pre. VAL has the same regime LABEL mix but a fundamentally different micro-structure regime (the Aug-Nov 2024 chop-bear transition followed by a Jan-Feb 2025 chop). BTC regime_label of 'bull' in VAL is short-lived bull pops that the catalog's bull-gated engines interpret as confirmation but that don't actually carry through.
- **Fundamental indictment**: the engine_catalog_discovery TRAIN catalog has the wrong selection criterion. compound_return_pct over TRAIN with 10-25 fires is a small-sample, multiple-comparisons problem masquerading as discovery. The Bonferroni-equivalent for 234 catch-tier survivors out of ~937 candidates is well below the per-engine evidence. The 4.3% VAL catch-tier survival rate is approximately what you'd expect if the TRAIN catalog were noise.

## Caveats

- VAL is ~10 months (~304 BTC bars) vs TRAIN's ~52 months. Per-engine fire counts are 3-6x smaller; confidence intervals proportionally wider.
- VAL catch-tier eligibility uses the same rules as TRAIN: positive expectancy + 3-fold sign consistency + n>=5. With shorter VAL window the fold size is smaller, so fold-consistency is a more demanding test in VAL.
- Confluence engines are skipped (no recipe persistence yet).
- Cost model is bucket-aware taker, identical to TRAIN. No maker rebate modeling here.
- We do NOT select engines on VAL -- this is diagnostic mining only.
