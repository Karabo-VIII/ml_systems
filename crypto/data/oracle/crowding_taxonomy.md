# Crowding & Reflexivity Taxonomy (Y5)

_Generated: TRAIN end 2024-05-15; catch-tier classes only_

## Headline counts

- CROWDED engines (>70% same-direction as 20d momentum): **588**
- NEUTRAL engines (40-70%): **2187**
- CONTRARIAN engines (<40%): **368**
- Reflexivity-decay flagged (late half pnl < early half by >30%): **1359** (43.2%)

## Per-bucket per-engine mean pnl (avg of per-engine mean_pnl_pct)

| bucket | n | avg per-engine mean_pnl_pct |
|---|---|---|
| CONTRARIAN | 368 | -0.6303% |
| CROWDED | 588 | +0.6910% |
| NEUTRAL | 2187 | +0.0551% |

## Herding-cascade days (per-asset daily fire density)

- Spec requested >=5 sigma per-asset day-density; heavy right skew in fire counts yields n=0 at that cut, so we use the **95th percentile** as the 'super-crowded' cut and 99th as 'extreme'.
- Super-crowded day fires (>=p95): n=23686, mean pnl = +0.6708%
- Extreme-crowded day fires (>=p99): n=7753, mean pnl = +1.4918%
- Normal day fires (<p95): n=133192, mean pnl = -0.0476%
- Delta (super - normal): +0.7184%

## Funding-extreme interaction (|xd_funding_spread z|>2)

- Fund-extreme fires: n=11643, mean pnl = -0.0686%
- Fund-normal fires:  mean pnl = +0.0712%

## Whale-extreme interaction (|wh_whale_net_usd z|>2)

- Whale-extreme fires: n=5250, mean pnl = +0.7770%
- Whale-normal fires:  mean pnl = +0.1118%

## Top 30 CONTRARIAN engines (>=20 fires)

| asset | indicator_class | indicator_config | regime | crowding | n_fires | mean_pnl_pct | reflexivity_decay |
|---|---|---|---|---|---|---|---|
| JST | RSI_threshold | p_22_lo_35_hi_60 | bull | 0.05 | 21 | +3.2466% | False |
| JST | RSI_threshold | p_21_lo_35_hi_60 | bull | 0.05 | 21 | +3.2466% | False |
| JST | RSI_threshold | p_21_lo_35_hi_60 | bull | 0.05 | 22 | +3.1397% | False |
| JST | RSI_threshold | p_22_lo_35_hi_60 | bull | 0.05 | 22 | +3.1397% | False |
| XRP | RSI_threshold | p_7_lo_20_hi_60 | chop | 0.12 | 24 | +2.7059% | False |
| XRP | RSI_threshold | p_7_lo_20_hi_60 | chop | 0.12 | 24 | +2.7059% | False |
| JST | RSI_threshold | p_22_lo_35_hi_60 | bull | 0.28 | 25 | +2.6953% | False |
| JST | RSI_threshold | p_22_lo_35_hi_60 | bull | 0.28 | 25 | +2.6953% | False |
| JST | RSI_threshold | p_15_lo_40_hi_60 | chop | 0.26 | 34 | +2.4008% | False |
| JST | RSI_threshold | p_15_lo_40_hi_60 | chop | 0.28 | 36 | +2.2840% | False |
| JST | RSI_threshold | p_21_lo_35_hi_60 | bull | 0.30 | 27 | +2.2714% | False |
| JST | RSI_threshold | p_21_lo_35_hi_60 | bull | 0.30 | 27 | +2.2714% | False |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 0.31 | 54 | +1.8325% | False |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 0.32 | 56 | +1.7604% | False |
| AR | MA_state_SMA_above | period_20 | bull | 0.18 | 103 | +1.5475% | False |
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | 0.38 | 61 | +1.5293% | True |
| AR | OBV_zscore | p_30_t_1.5 | chop | 0.38 | 47 | +1.5158% | False |
| LINK | RSI_threshold | p_6_lo_35_hi_60 | chop | 0.38 | 63 | +1.4748% | True |
| WLD | MA_state_SMA_above | period_50 | chop | 0.25 | 69 | +1.4440% | False |
| AR | OBV_zscore | p_20_t_1.5 | bull | 0.36 | 47 | +1.3966% | False |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 0.33 | 57 | +1.3764% | False |
| AR | OBV_zscore | p_100_t_1.5 | bull | 0.36 | 56 | +1.3671% | False |
| LINK | RSI_threshold | p_7_lo_35_hi_60 | chop | 0.33 | 58 | +1.3655% | False |
| AR | OBV_zscore | p_20_t_1.5 | bull | 0.37 | 49 | +1.3640% | False |
| PEPE | RSI_threshold | p_7_lo_35_hi_80 | chop | 0.36 | 33 | +1.3479% | True |
| JST | RSI_threshold | p_13_lo_40_hi_65 | chop | 0.36 | 25 | +1.2530% | False |
| PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop | 0.30 | 40 | +1.2511% | True |
| PEPE | RSI_threshold | p_10_lo_40_hi_80 | chop | 0.30 | 40 | +1.2511% | True |
| PEPE | RSI_threshold | p_8_lo_40_hi_80 | chop | 0.37 | 46 | +1.2282% | True |
| PEPE | RSI_threshold | p_8_lo_40_hi_80 | chop | 0.37 | 46 | +1.2282% | True |

## Top 20 CROWDED engines (likely deflated at deploy due to consensus overlap)

| asset | indicator_class | indicator_config | regime | crowding | n_fires | mean_pnl_pct |
|---|---|---|---|---|---|---|
| PEPE | Donchian_state_above_midline | period_55 | chop | 0.73 | 71 | +9.2291% |
| PEPE | Donchian_state_above_midline | period_55 | chop | 0.73 | 71 | +9.2291% |
| PEPE | MA_state_SMA_above | period_50 | chop | 0.75 | 76 | +9.0704% |
| PEPE | MA_state_SMA_above | period_50 | chop | 0.75 | 76 | +9.0704% |
| WLD | MA_state_SMA_above | period_50 | chop | 0.88 | 65 | +8.7849% |
| WLD | MA_state_SMA_above | period_50 | chop | 0.88 | 65 | +8.7849% |
| PEPE | MA_state_SMA_above | period_50 | chop | 0.71 | 83 | +8.3515% |
| WLD | MA_state_SMA_above | period_50 | chop | 0.87 | 71 | +8.1041% |
| AR | MA_state_SMA_above | period_20 | bull | 0.95 | 78 | +7.2416% |
| AR | MA_state_SMA_above | period_20 | bull | 0.95 | 78 | +7.2416% |
| AR | MA_state_SMA_above | period_20 | bull | 0.95 | 79 | +7.1597% |
| SOL | MA_state_SMA_above | period_20 | chop | 0.98 | 94 | +5.5968% |
| SOL | MA_state_SMA_above | period_20 | chop | 0.98 | 94 | +5.5968% |
| SOL | MA_state_SMA_above | period_20 | chop | 0.98 | 94 | +5.5968% |
| SOL | Donchian_state_above_midline | period_20 | chop | 0.98 | 96 | +5.5679% |
| SOL | Donchian_state_above_midline | period_20 | chop | 0.98 | 96 | +5.5679% |
| SOL | Donchian_state_above_midline | period_20 | chop | 0.98 | 96 | +5.5679% |
| OP | Donchian_state_above_midline | period_20 | chop | 0.99 | 67 | +5.4397% |
| OP | Donchian_state_above_midline | period_20 | chop | 0.99 | 67 | +5.4397% |
| OP | MA_state_SMA_above | period_20 | chop | 0.96 | 68 | +5.2784% |

## Gold-standard intersection: CONTRARIAN + TRUE_ALPHA + stable + not-concave

- Count: **27**

| asset | indicator_class | indicator_config | regime | crowding | n_fires | mean_pnl_pct |
|---|---|---|---|---|---|---|
| AR | MA_state_SMA_above | period_20 | bull | 0.18 | 103 | +1.5475% |
| BTC | MA_state_SMA_above | period_50 | chop | 0.18 | 89 | +0.7160% |
| BTC | Donchian_state_above_midline | period_20 | bull | 0.12 | 80 | +0.6195% |
| BTC | MA_state_EMA_above | period_20 | bull | 0.18 | 87 | +0.4945% |
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 0.40 | 73 | +0.3430% |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 0.32 | 56 | +0.3137% |
| APT | MA_state_EMA_above | period_20 | bull | 0.19 | 97 | +0.1095% |
| DASH | MA_state_EMA_above | period_50 | bull | 0.23 | 74 | +0.0595% |
| DASH | MA_state_SMA_above | period_50 | bull | 0.23 | 80 | +0.0080% |
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 0.33 | 43 | -1.9701% |
| APT | Distance_z_state | period_50_threshold_1.5 | bull | 0.33 | 42 | -1.9905% |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 0.21 | 43 | -2.7234% |
| ICP | Distance_z_state | period_50_threshold_1.5 | bull | 0.23 | 40 | -2.8490% |
| BTC | Donchian_state_above_midline | period_20 | bull | 0.27 | 45 | -2.8605% |
| BTC | Donchian_state_above_midline | period_20 | bull | 0.28 | 43 | -2.9400% |
| BTC | MA_state_EMA_above | period_20 | bull | 0.31 | 48 | -2.9662% |
| BTC | MA_state_SMA_above | period_50 | chop | 0.29 | 48 | -2.9870% |
| BTC | MA_state_EMA_above | period_20 | bull | 0.33 | 46 | -3.0451% |
| BTC | MA_state_SMA_above | period_50 | chop | 0.31 | 45 | -3.1097% |
| APT | MA_state_EMA_above | period_20 | bull | 0.26 | 68 | -3.8726% |
| APT | MA_state_EMA_above | period_20 | bull | 0.26 | 65 | -4.0023% |
| DASH | MA_state_SMA_above | period_50 | bull | 0.33 | 49 | -4.2979% |
| DASH | MA_state_EMA_above | period_50 | bull | 0.33 | 46 | -4.4034% |
| DASH | MA_state_SMA_above | period_50 | bull | 0.32 | 47 | -4.4443% |
| DASH | MA_state_EMA_above | period_50 | bull | 0.32 | 44 | -4.5646% |
| AR | MA_state_SMA_above | period_20 | bull | 0.27 | 78 | -4.7971% |
| AR | MA_state_SMA_above | period_20 | bull | 0.27 | 75 | -4.9452% |

## Caveats

- 20-day momentum-consensus is *one* possible crowding signal; alternatives (funding-arb consensus, ETF-flow consensus, vol-targeting consensus) would yield different crowding distributions.
- Reflexivity decay (>30% drop in late half) needs longer TRAIN history to be definitive; with median ~30-100 fires per engine, late-half estimates have high variance.
- Super-crowded days threshold (5 sigma) is per-asset; ETH/BTC with denser fire histories produce different absolute counts than long-tail alts.
- We join on (asset,class,config) — engines in catalog distinguished by btc_regime_30d are collapsed for the crowding score; per-regime crowding may differ.
- pnl_post_cost_pct already nets cost; we don't apply additional crowding-decay haircut.
- **Sign inversion in headline bucket pnl**: CROWDED engines (mom-aligned) average +0.69% per fire while CONTRARIAN average -0.63%. This contradicts the naive 'crowding => deflated edge' hypothesis on this corpus: at the catch-tier level over TRAIN, going with the trend (long when 20d-up, short when 20d-down) pays. Mining the residual CONTRARIAN winners (the gold list) is therefore isolating *true mean-reversion edge* rather than 'capacity in uncrowded space'.
- TRUE_ALPHA + stable + not-concave + CONTRARIAN intersection contains engines with NEGATIVE mean_pnl (TRUE_ALPHA measures excess-over-cluster, not raw expectancy). The 9 entries with positive mean_pnl are the real gold-standard; the negative-pnl rows are alpha *on residual* but bleeding *on raw*.