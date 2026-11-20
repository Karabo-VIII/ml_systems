# Synthetic Regime Stress Test (Oracle Z1 / POV-9)

Generated: 2026-05-23 11:21:48

## Methodology

- **Block bootstrap** of basket daily pnl streams from `data/oracle/sim_all_baskets_fixed.parquet` (close-derived fwd_ret_1d, NOT corrupted target_return_1_raw).
- Block size = 30 trading days, 1000 basket-level samples (200 samples per engine in step 2 to fit time budget).
- Per-engine bootstrap uses event_eval_rows.pnl_post_cost_pct filtered to TRAIN <= 2024-05-15, fires as samples, block ~ min(30, n_fires/3).
- Regime completeness: rolling 30-day return moments per asset projected to 6-D (mean, std, kurt, skew, max-dd, ar1) over TRAIN window.
- Synthetic stress: BLACK_SWAN single -50pct asset day, PROLONGED_BEAR -2pct/d for 30d, HIGH_VOL 3x sigma for 60d. Basket day_ret derived by applying the live sizing (3 picks x 25pct x cost 24bp) to the shocked asset return.

## 1. Basket Block-Bootstrap (point vs 5th/95th percentile)

| Variant | n_days | point mean %/d | p05 mean %/d | p50 | p95 | point Sharpe | p05 Sharpe | p95 Sharpe | point maxDD | p05 maxDD | DEFLATION FACTOR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1_decoupling_J0_50_32eng | 317 | +0.5832 | +0.1657 | +0.5639 | +1.0134 | +3.18 | +0.96 | +5.25 | -26.8760 | -42.7647 | 0.284 |
| V7_TRIPLE_stable_AND_NOT_concave_AND_mag_ge_1.5 | 136 | +0.8216 | +0.3143 | +0.8091 | +1.4449 | +3.67 | +1.64 | +5.59 | -14.3883 | -26.5293 | 0.383 |

**Deflation factor** = (5th percentile bootstrap mean) / (point estimate mean).  If <1.0 the point estimate is OPTIMISTIC vs the bootstrap CI.

## 2. Per-Engine Bootstrap (top 30 catch-tier engines)

| Engine | Asset | Class | n_fires | point mean %/fire | p05 | p50 | p95 | deflation |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SOL__MACD_threshold__f_12_s_21_g_9__rchop__c1__m | SOL | MACD_threshold | 1348 | -0.238 | -0.534 | -0.224 | +0.095 | 2.241 |
| LINK__MACD_threshold__f_12_s_21_g_9__rchop__c1__ | LINK | MACD_threshold | 1516 | -0.262 | -0.485 | -0.264 | -0.033 | 1.849 |
| ADA__VWAP_state_above__period_50__rbull__c2__mal | ADA | VWAP_state_above | 1257 | +1.101 | +0.420 | +1.091 | +1.845 | 0.381 |
| ADA__OBV_zscore__p_100_t_1.0__rbull__c2__mall__h | ADA | OBV_zscore | 1144 | +0.334 | -0.362 | +0.308 | +1.135 | -1.085 |
| ARKM__OBV_zscore__p_30_t_1.0__rchop__c1__mall__h | ARKM | OBV_zscore | 770 | -0.623 | -1.337 | -0.629 | +0.052 | 2.146 |
| DOT__OBV_zscore__p_20_t_1.0__rbull__c2__mall__h1 | DOT | OBV_zscore | 1015 | +0.383 | -0.171 | +0.350 | +1.132 | -0.447 |
| APT__RSI_threshold__p_5_lo_40_hi_65__rbull__c1__ | APT | RSI_threshold | 863 | +0.151 | -0.527 | +0.168 | +0.576 | -3.481 |
| BTC__ATR_bands__p_20_k_1.5__rbull__c1__mall__h3d | BTC | ATR_bands | 784 | +0.120 | -0.529 | +0.177 | +1.009 | -4.396 |
| DYDX__OBV_zscore__p_30_t_1.0__rbull__c2__mall__h | DYDX | OBV_zscore | 949 | -0.178 | -0.663 | -0.185 | +0.367 | 3.727 |
| LINK__RSI_threshold__p_5_lo_40_hi_60__rchop__c1_ | LINK | RSI_threshold | 924 | +0.094 | -0.291 | +0.068 | +0.555 | -3.098 |
| SUPER__OBV_zscore__p_50_t_1.0__rbull__c1__mall__ | SUPER | OBV_zscore | 912 | +0.400 | -0.786 | +0.407 | +1.606 | -1.962 |
| ICP__RSI_threshold__p_6_lo_40_hi_65__rbull__c1__ | ICP | RSI_threshold | 860 | -0.984 | -1.886 | -0.985 | -0.187 | 1.917 |
| DASH__OBV_zscore__p_50_t_1.0__rchop__c5__mall__h | DASH | OBV_zscore | 789 | -0.476 | -0.906 | -0.451 | -0.091 | 1.904 |
| FET__VPIN_threshold__t_0.5__rbull__c1__mall__h3d | FET | VPIN_threshold | 594 | +1.131 | +0.277 | +1.076 | +1.899 | 0.245 |
| AR__OBV_zscore__p_20_t_1.5__rbull__c5__mall__h1d | AR | OBV_zscore | 583 | +1.367 | +0.165 | +1.186 | +2.690 | 0.121 |
| ARKM__VPIN_threshold__t_0.5__rchop__c1__mall__h3 | ARKM | VPIN_threshold | 548 | +0.068 | -0.724 | +0.062 | +0.738 | -10.613 |
| PEPE__RSI_threshold__p_8_lo_40_hi_80__rchop__c5_ | PEPE | RSI_threshold | 557 | +0.059 | -0.665 | +0.071 | +0.924 | -11.235 |
| ARKM__Kyle_lambda_threshold__t_0.5__rbull__c1__m | ARKM | Kyle_lambda_threshold | 559 | -1.667 | -2.894 | -1.564 | -0.149 | 1.736 |
| AAVE__VPIN_threshold__t_0.5__rchop__c4__mall__h3 | AAVE | VPIN_threshold | 412 | +0.120 | -0.444 | +0.115 | +0.610 | -3.705 |
| JST__RSI_threshold__p_9_lo_40_hi_65__rchop__c5__ | JST | RSI_threshold | 474 | -0.334 | -1.075 | -0.404 | +0.302 | 3.216 |
| AR__MA_state_SMA_above__period_20__rbull__c4__ma | AR | MA_state_SMA_above | 409 | +2.857 | +1.505 | +2.894 | +4.377 | 0.527 |
| FET__Distance_z_state__period_20_threshold_1.0__ | FET | Distance_z_state | 331 | +2.577 | +0.994 | +2.365 | +4.264 | 0.386 |
| ENJ__Kyle_lambda_threshold__t_1.0__rbull__c4__ma | ENJ | Kyle_lambda_threshold | 410 | +0.210 | -0.461 | +0.212 | +0.994 | -2.195 |
| APT__MA_state_EMA_above__period_20__rbull__c5__m | APT | MA_state_EMA_above | 382 | +1.626 | +1.099 | +1.624 | +2.128 | 0.676 |
| LTC__RSI_threshold__p_5_lo_40_hi_65__rbear__c5__ | LTC | RSI_threshold | 362 | -0.263 | -1.329 | -0.279 | +0.530 | 5.059 |
| FET__measure_engines/norm_flow_imbalance__op_abs | FET | measure_engines/norm_flo | 162 | +0.648 | -0.917 | +0.536 | +2.440 | -1.416 |
| LINK__Donchian_state_above_midline__period_100__ | LINK | Donchian_state_above_mid | 322 | +0.710 | +0.141 | +0.678 | +1.414 | 0.199 |
| LINK__MA_state_EMA_above__period_200__rchop__c5_ | LINK | MA_state_EMA_above | 327 | +0.725 | +0.079 | +0.680 | +1.240 | 0.108 |
| DASH__MA_state_SMA_above__period_50__rbull__c5__ | DASH | MA_state_SMA_above | 368 | +0.797 | +0.314 | +0.823 | +1.316 | 0.394 |
| AR__Distance_z_state__period_20_threshold_1.0__r | AR | Distance_z_state | 296 | +2.586 | +0.735 | +2.447 | +4.754 | 0.284 |

**Per-engine median deflation = 0.264**, 5th-percentile deflation = -7.815.
**20/30 engines have a 5th-percentile mean below ZERO** (no edge once block-bootstrap CI is applied).

## 3. Synthetic Stress Regimes (V1 vs V7)

| Variant | Baseline NAV % | Baseline maxDD | BLACK_SWAN NAV delta | BLACK_SWAN maxDD | PROLONGED_BEAR NAV delta | PROLONGED_BEAR maxDD | HIGH_VOL median NAV delta | HIGH_VOL p05 NAV delta | HIGH_VOL zero-mean median | HIGH_VOL p05 maxDD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1_decoupling_J0_50_32eng | +454.18% | -0.2688 | -208.81% | -0.4580 | -220.82% | -0.4769 | +102.14% | -383.09% | -89.65% | -0.7850 |
| V7_TRIPLE_stable_AND_NOT_concave_AND_mag_ge_1.5 | +180.18% | -0.1439 | -105.57% | -0.4492 | -111.65% | -0.4683 | +63.41% | -213.02% | -69.72% | -0.8463 |

Stress-regime notes:
- BLACK_SWAN = single day where all 3 picked assets drop -50%. Basket day_ret = 3 x 0.25 x (-0.50 - 0.0024) ~ -37.8%. The fact that V1 and V7 receive the SAME basket day return for the swan day reflects the construction (3 picks x 25%); the DIFFERENCE in NAV delta is the post-stress compounding of the surviving baseline.
- PROLONGED_BEAR = 30 consecutive days each at 3 x 0.25 x (-0.02 - 0.0024) ~ -1.7% basket day. Cumulative NAV impact = (1 - 0.0168)^30 - 1 ~ -39.8% on the incremental window, applied to the basket NAV at TRAIN end.
- HIGH_VOL = 60 days drawn from N(mu, 3*std(x)) with 200 RNG draws averaged. Median NAV delta reflects realised drift; zero-mean variant isolates pure variance drag (geometric drag = exp(-0.5*sigma^2)*T per 60 days at 3x vol). **The maxDD p05 is the load-bearing number** -- with 3x vol, the basket experiences -50pct+ rolling drawdowns in 5pct of HIGH_VOL realisations.

## 4. TRAIN Regime Coverage (6-D return space)

| dim | p01 | p05 | p50 | p95 | p99 | min | max | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roll_mean | -0.0202 | -0.0133 | 0.0006 | 0.0252 | 0.0499 | -0.0473 | 0.1645 | 0.0133 |
| roll_std | 0.0157 | 0.0216 | 0.0490 | 0.1128 | 0.1519 | 0.0088 | 0.7327 | 0.0334 |
| roll_kurt | -0.9971 | -0.6648 | 0.9142 | 7.0683 | 13.4106 | -1.5099 | 28.4883 | 2.8647 |
| roll_skew | -1.9788 | -1.2176 | 0.0882 | 1.8642 | 2.9410 | -4.2292 | 5.2685 | 0.9597 |
| roll_maxdd | -0.6224 | -0.4800 | -0.2088 | -0.0764 | -0.0478 | -0.8294 | -0.0163 | 0.1273 |
| roll_ar1 | -0.4922 | -0.3790 | -0.0893 | 0.1971 | 0.3220 | -0.8057 | 0.7786 | 0.1762 |

### Detected gaps in TRAIN coverage

- GAP-BASKET V1_decoupling_J0_50_32eng: realised worst basket-day in TRAIN = -11.18%; BLACK_SWAN injects -37.80% (~3.4x worse, untested)
- GAP-BASKET V7_TRIPLE_stable_AND_NOT_concave_AND_mag_ge_1.5: realised worst basket-day in TRAIN = -9.27%; BLACK_SWAN injects -37.80% (~4.1x worse, untested)

## 5. Caveats

- Block-bootstrap assumes **no autocorrelation across blocks**. Real basket day returns have multi-day clusters of negative pnl during regime breaks; 30-day blocks captures most clustering but not all.
- The 1000-sample bootstrap inflates compute; per-engine step uses 200 samples (still gives 5-95 CI to ~0.4 std-err on the mean).
- Synthetic stress regimes are **constructed, not historical**. BLACK_SWAN at -50% is a covid/FTX-style worst-case proxy.
- The basket day_ret stream in sim_all_baskets_fixed.parquet is already net of 24bp RT cost; the stress regimes apply the SAME cost to the synthetic asset return, so the resulting basket day_ret is comparable.
- HIGH_VOL shocks variance only; if the regime breaks correlate (e.g. all assets down 3 sigma simultaneously), drag is much larger than what this captures.
- The corrupted target_return_1_raw is NOT used. Per-engine pnl uses the verified-correct event_eval_rows.pnl_post_cost_pct.
