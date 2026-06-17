# Chimera v51 — Per-Measure Encyclopedia

Generated: 2026-05-23T06:40:11.298549Z  |  TRAIN end: 2024-05-15  |  Top-K mover = 22 of ~80 assets/day

Convention: `t-N` = indicator value N days BEFORE the top-25%-mover event day. `t-0` = same day. `t+1` = day after event. z-values are per-asset z-scores of the column (TRAIN window).

---

## Top columns by |discriminator_score|

| rank | column | discriminator | lift_top_t-1 | lift_bot_t-1 | n_engines | fold_stable |
|---:|:---|---:|---:|---:|---:|:---:|
| 1 | `xex_by_bn_spread_bps` | -0.207 | -0.043 | +0.165 | 0 | no |
| 2 | `xrel_rv_bpv_5m_xrank` | -0.148 | +0.060 | +0.208 | 0 | YES |
| 3 | `xrel_rv_rv_5m_xrank` | -0.143 | +0.063 | +0.205 | 0 | YES |
| 4 | `xrel_hbr_n_trades_xrank` | -0.133 | +0.040 | +0.173 | 0 | YES |
| 5 | `xrel_rv_bpv_5m_xratio` | -0.132 | +0.056 | +0.188 | 0 | YES |
| 6 | `s3_oi_usd` | -0.130 | +0.002 | +0.132 | 0 | no |
| 7 | `xrel_rv_rv_5m_xratio` | -0.129 | +0.056 | +0.185 | 0 | YES |
| 8 | `xrel_hbr_n_trades_xratio` | -0.122 | +0.038 | +0.159 | 0 | YES |
| 9 | `xex_ok_bn_spread_bps` | -0.115 | +0.020 | +0.136 | 0 | no |
| 10 | `xrel_hbr_eta_total_xrank` | -0.106 | +0.028 | +0.133 | 0 | YES |
| 11 | `bd_notional_l1pct_mean` | -0.103 | -0.012 | +0.091 | 0 | no |
| 12 | `hbr_n_trades` | -0.094 | +0.026 | +0.120 | 0 | no |
| 13 | `xrel_hbr_eta_total_xratio` | -0.090 | +0.023 | +0.113 | 0 | no |
| 14 | `xrel_rv_rv_5m_xpct10` | -0.089 | +0.056 | +0.145 | 0 | YES |
| 15 | `xrel_rv_bpv_5m_xpct10` | -0.088 | +0.056 | +0.144 | 0 | YES |
| 16 | `xrel_hbr_n_trades_xpct10` | -0.085 | +0.017 | +0.101 | 0 | no |
| 17 | `hbr_eta_buy` | -0.084 | +0.016 | +0.100 | 13 | no |
| 18 | `xrel_liq_long_usd_xratio` | -0.083 | +0.026 | +0.109 | 0 | no |
| 19 | `xrel_hbr_eta_total_xpct10` | -0.082 | +0.016 | +0.098 | 0 | no |
| 20 | `wh_whale_trade_count_500k` | -0.081 | +0.021 | +0.102 | 6 | no |
| 21 | `xrel_liq_long_usd_xrank` | -0.080 | +0.026 | +0.106 | 0 | YES |
| 22 | `hbr_eta_total` | -0.079 | +0.016 | +0.096 | 13 | no |
| 23 | `hbr_eta_sell` | -0.078 | +0.020 | +0.097 | 0 | no |
| 24 | `wh_whale_buy_usd` | -0.075 | +0.022 | +0.098 | 0 | no |
| 25 | `wh_whale_sell_usd` | -0.074 | +0.021 | +0.095 | 0 | no |

---

## `xex_by_bn_spread_bps`

**Family**: `xex_*` — cross-exchange spread (binance vs okx / bybit, bps)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 680
- mean = +0.040758
- std  = 1.819119
- p10 = -1.8687
- p50 = +0.0000
- p90 = +1.8434
- skew = +2.104
- excess kurtosis = +29.865

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.060 |
| t-3 | indicator 3 days BEFORE event | -0.033 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.043 |
| t-0 | indicator on event day (concurrent) | +0.161 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.027 |

**Lead/Lag verdict**: CONCURRENT: t-0 dominates (co-incident, not predictive).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.165**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.207**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.331
- chop (label=1) = -0.022
- bear (label=0) = +0.074

Raw regime keys (column-as-found):
  - `label_0`: +0.074
  - `label_2`: +0.331
  - `label_1`: -0.022

Regime story: strongest absolute deviation in **bull** (z=+0.331); weakest in chop (z=-0.022); spread=0.353 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.082
- STEADY = +0.124
- VOLATILE = +0.418
- DEGEN = +nan

DNA story: strongest in **VOLATILE** (z=+0.418); weakest in BLUE (z=+0.082); cross-bucket spread = 0.336 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.074
- F3 (2024-03-01 .. 2024-05-15) = +0.219
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.50. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xex_by_bn_spread_bps`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xex_by_bn_spread_bps` is a STRONG concurrent discriminator: top-25%-mover z-lift goes t-3=-0.03 -> t-1=-0.04 -> t-0=+0.16 -> t+1=+0.03, while bottom-25% movers sit at t-1=+0.16 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.207. Cross-regime: strongest in bull (z=+0.33). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.42). Fold consistency: fold-unstable (F1=+nan, F2=+0.07, F3=+0.22). Distributionally mean=+0.0408 std=1.8191 skew=+2.10 kurt=+29.86. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `xrel_rv_bpv_5m_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.513484
- std  = 0.288612
- p10 = +0.1111
- p50 = +0.5135
- p90 = +0.9143
- skew = -0.000
- excess kurtosis = -1.201

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.052 |
| t-3 | indicator 3 days BEFORE event | +0.057 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.060 |
| t-0 | indicator on event day (concurrent) | +0.080 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.492 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.208**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.148**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.058
- chop (label=1) = +0.080
- bear (label=0) = +0.113

Raw regime keys (column-as-found):
  - `label_2`: +0.058
  - `label_1`: +0.080
  - `label_0`: +0.113

Regime story: strongest absolute deviation in **bear** (z=+0.113); weakest in bull (z=+0.058); spread=0.055 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.018
- STEADY = +0.078
- VOLATILE = +0.093
- DEGEN = +0.071

DNA story: strongest in **VOLATILE** (z=+0.093); weakest in BLUE (z=+0.018); cross-bucket spread = 0.075 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.108
- F2 (2023-11-01 .. 2024-02-29) = -0.160
- F3 (2024-03-01 .. 2024-05-15) = -0.276
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.39. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_bpv_5m_xrank`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_bpv_5m_xrank` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.06 -> t-1=+0.06 -> t-0=+0.08 -> t+1=+0.49, while bottom-25% movers sit at t-1=+0.21 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.148. Cross-regime: strongest in bear (z=+0.11). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.09). Fold consistency: fold-stable (F1=-0.11, F2=-0.16, F3=-0.28). Distributionally mean=+0.5135 std=0.2886 skew=-0.00 kurt=-1.20. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_rv_rv_5m_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.513484
- std  = 0.288612
- p10 = +0.1111
- p50 = +0.5135
- p90 = +0.9143
- skew = -0.000
- excess kurtosis = -1.201

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.051 |
| t-3 | indicator 3 days BEFORE event | +0.060 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.063 |
| t-0 | indicator on event day (concurrent) | +0.081 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.493 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.205**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.143**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.060
- chop (label=1) = +0.083
- bear (label=0) = +0.107

Raw regime keys (column-as-found):
  - `label_0`: +0.107
  - `label_2`: +0.060
  - `label_1`: +0.083

Regime story: strongest absolute deviation in **bear** (z=+0.107); weakest in bull (z=+0.060); spread=0.047 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.025
- STEADY = +0.080
- VOLATILE = +0.092
- DEGEN = +0.072

DNA story: strongest in **VOLATILE** (z=+0.092); weakest in BLUE (z=+0.025); cross-bucket spread = 0.067 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.114
- F2 (2023-11-01 .. 2024-02-29) = -0.170
- F3 (2024-03-01 .. 2024-05-15) = -0.282
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.37. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_rv_5m_xrank`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_rv_5m_xrank` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.06 -> t-1=+0.06 -> t-0=+0.08 -> t+1=+0.49, while bottom-25% movers sit at t-1=+0.21 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.143. Cross-regime: strongest in bear (z=+0.11). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.09). Fold consistency: fold-stable (F1=-0.11, F2=-0.17, F3=-0.28). Distributionally mean=+0.5135 std=0.2886 skew=-0.00 kurt=-1.20. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_n_trades_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.513484
- std  = 0.288612
- p10 = +0.1111
- p50 = +0.5135
- p90 = +0.9143
- skew = -0.000
- excess kurtosis = -1.201

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.017 |
| t-3 | indicator 3 days BEFORE event | +0.030 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.040 |
| t-0 | indicator on event day (concurrent) | +0.049 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.357 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.173**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.133**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.025
- chop (label=1) = +0.060
- bear (label=0) = +0.071

Raw regime keys (column-as-found):
  - `label_1`: +0.060
  - `label_2`: +0.025
  - `label_0`: +0.071

Regime story: strongest absolute deviation in **bear** (z=+0.071); weakest in bull (z=+0.025); spread=0.046 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.014
- STEADY = +0.045
- VOLATILE = +0.044
- DEGEN = +0.080

DNA story: strongest in **DEGEN** (z=+0.080); weakest in BLUE (z=+0.014); cross-bucket spread = 0.067 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.084
- F2 (2023-11-01 .. 2024-02-29) = +0.040
- F3 (2024-03-01 .. 2024-05-15) = +0.002
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.80. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_n_trades_xrank`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_n_trades_xrank` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.04 -> t-0=+0.05 -> t+1=+0.36, while bottom-25% movers sit at t-1=+0.17 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.133. Cross-regime: strongest in bear (z=+0.07). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-stable (F1=+0.08, F2=+0.04, F3=+0.00). Distributionally mean=+0.5135 std=0.2886 skew=-0.00 kurt=-1.20. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_rv_bpv_5m_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +1.000000
- std  = 1.379401
- p10 = +0.2547
- p50 = +0.6795
- p90 = +1.8417
- skew = +7.572
- excess kurtosis = +89.497

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.055 |
| t-3 | indicator 3 days BEFORE event | +0.054 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.056 |
| t-0 | indicator on event day (concurrent) | +0.061 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.449 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.188**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.132**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.041
- chop (label=1) = +0.062
- bear (label=0) = +0.088

Raw regime keys (column-as-found):
  - `label_2`: +0.041
  - `label_0`: +0.088
  - `label_1`: +0.062

Regime story: strongest absolute deviation in **bear** (z=+0.088); weakest in bull (z=+0.041); spread=0.047 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.020
- STEADY = +0.079
- VOLATILE = +0.049
- DEGEN = +0.072

DNA story: strongest in **STEADY** (z=+0.079); weakest in BLUE (z=+0.020); cross-bucket spread = 0.059 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.285
- F2 (2023-11-01 .. 2024-02-29) = -0.179
- F3 (2024-03-01 .. 2024-05-15) = -0.222
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.19. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_bpv_5m_xratio`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_bpv_5m_xratio` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.05 -> t-1=+0.06 -> t-0=+0.06 -> t+1=+0.45, while bottom-25% movers sit at t-1=+0.19 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.132. Cross-regime: strongest in bear (z=+0.09). Cross-DNA: most pronounced in STEADY bucket (z=+0.08). Fold consistency: fold-stable (F1=-0.28, F2=-0.18, F3=-0.22). Distributionally mean=+1.0000 std=1.3794 skew=+7.57 kurt=+89.50. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_oi_usd`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31685
- mean = +179437615.581853
- std  = 562032581.238212
- p10 = +7332567.6523
- p50 = +37866965.7896
- p90 = +172129976.5086
- skew = +5.021
- excess kurtosis = +27.883

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.007 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.080 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.132**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.130**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.002
- chop (label=1) = -0.025
- bear (label=0) = +0.041

Raw regime keys (column-as-found):
  - `label_1`: -0.025
  - `label_0`: +0.041
  - `label_2`: -0.002

Regime story: strongest absolute deviation in **bear** (z=+0.041); weakest in bull (z=-0.002); spread=0.043 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.088
- STEADY = +0.023
- VOLATILE = -0.028
- DEGEN = +0.029

DNA story: strongest in **BLUE** (z=+0.088); weakest in STEADY (z=+0.023); cross-bucket spread = 0.064 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.344
- F2 (2023-11-01 .. 2024-02-29) = +0.208
- F3 (2024-03-01 .. 2024-05-15) = +0.826
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.08. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_oi_usd`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`s3_oi_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.08, while bottom-25% movers sit at t-1=+0.13 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.130. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=+0.09). Fold consistency: fold-unstable (F1=-0.34, F2=+0.21, F3=+0.83). Distributionally mean=+179437615.5819 std=562032581.2382 skew=+5.02 kurt=+27.88. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_rv_rv_5m_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +1.000000
- std  = 1.403370
- p10 = +0.2529
- p50 = +0.6777
- p90 = +1.8316
- skew = +7.667
- excess kurtosis = +90.846

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.052 |
| t-3 | indicator 3 days BEFORE event | +0.054 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.056 |
| t-0 | indicator on event day (concurrent) | +0.061 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.445 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.185**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.129**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.043
- chop (label=1) = +0.062
- bear (label=0) = +0.085

Raw regime keys (column-as-found):
  - `label_2`: +0.043
  - `label_0`: +0.085
  - `label_1`: +0.062

Regime story: strongest absolute deviation in **bear** (z=+0.085); weakest in bull (z=+0.043); spread=0.042 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.026
- STEADY = +0.077
- VOLATILE = +0.049
- DEGEN = +0.073

DNA story: strongest in **STEADY** (z=+0.077); weakest in BLUE (z=+0.026); cross-bucket spread = 0.051 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.294
- F2 (2023-11-01 .. 2024-02-29) = -0.188
- F3 (2024-03-01 .. 2024-05-15) = -0.219
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.19. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_rv_5m_xratio`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_rv_5m_xratio` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.05 -> t-1=+0.06 -> t-0=+0.06 -> t+1=+0.45, while bottom-25% movers sit at t-1=+0.18 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.129. Cross-regime: strongest in bear (z=+0.08). Cross-DNA: most pronounced in STEADY bucket (z=+0.08). Fold consistency: fold-stable (F1=-0.29, F2=-0.19, F3=-0.22). Distributionally mean=+1.0000 std=1.4034 skew=+7.67 kurt=+90.85. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_n_trades_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +1.000000
- std  = 2.380531
- p10 = +0.0686
- p50 = +0.3783
- p90 = +2.0844
- skew = +7.507
- excess kurtosis = +73.394

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.018 |
| t-3 | indicator 3 days BEFORE event | +0.029 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.038 |
| t-0 | indicator on event day (concurrent) | +0.037 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.341 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.159**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.122**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.031
- chop (label=1) = +0.028
- bear (label=0) = +0.057

Raw regime keys (column-as-found):
  - `label_1`: +0.028
  - `label_0`: +0.057
  - `label_2`: +0.031

Regime story: strongest absolute deviation in **bear** (z=+0.057); weakest in chop (z=+0.028); spread=0.029 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.051
- STEADY = +0.061
- VOLATILE = +0.021
- DEGEN = +0.064

DNA story: strongest in **DEGEN** (z=+0.064); weakest in VOLATILE (z=+0.021); cross-bucket spread = 0.043 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.048
- F2 (2023-11-01 .. 2024-02-29) = +0.216
- F3 (2024-03-01 .. 2024-05-15) = +0.191
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.49. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_n_trades_xratio`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_n_trades_xratio` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.04 -> t-0=+0.04 -> t+1=+0.34, while bottom-25% movers sit at t-1=+0.16 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.122. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-stable (F1=+0.05, F2=+0.22, F3=+0.19). Distributionally mean=+1.0000 std=2.3805 skew=+7.51 kurt=+73.39. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xex_ok_bn_spread_bps`

**Family**: `xex_*` — cross-exchange spread (binance vs okx / bybit, bps)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 680
- mean = +0.096845
- std  = 1.715714
- p10 = -1.9314
- p50 = +0.0000
- p90 = +1.9642
- skew = -0.298
- excess kurtosis = +1.806

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.014 |
| t-3 | indicator 3 days BEFORE event | -0.051 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.020 |
| t-0 | indicator on event day (concurrent) | +0.032 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.093 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.136**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.115**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.065
- chop (label=1) = -0.117
- bear (label=0) = +0.117

Raw regime keys (column-as-found):
  - `label_0`: +0.117
  - `label_2`: +0.065
  - `label_1`: -0.117

Regime story: strongest absolute deviation in **bear** (z=+0.117); weakest in bull (z=+0.065); spread=0.051 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.042
- STEADY = +0.004
- VOLATILE = +0.070
- DEGEN = +nan

DNA story: strongest in **VOLATILE** (z=+0.070); weakest in STEADY (z=+0.004); cross-bucket spread = 0.065 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.102
- F3 (2024-03-01 .. 2024-05-15) = -0.015
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.33. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xex_ok_bn_spread_bps`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xex_ok_bn_spread_bps` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=-0.05 -> t-1=+0.02 -> t-0=+0.03 -> t+1=-0.09, while bottom-25% movers sit at t-1=+0.14 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.115. Cross-regime: strongest in bear (z=+0.12). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.07). Fold consistency: fold-unstable (F1=+nan, F2=+0.10, F3=-0.01). Distributionally mean=+0.0968 std=1.7157 skew=-0.30 kurt=+1.81. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_eta_total_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.513484
- std  = 0.288612
- p10 = +0.1111
- p50 = +0.5135
- p90 = +0.9143
- skew = -0.000
- excess kurtosis = -1.201

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.016 |
| t-3 | indicator 3 days BEFORE event | +0.012 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.028 |
| t-0 | indicator on event day (concurrent) | +0.018 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.405 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.133**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.106**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.004
- chop (label=1) = +0.027
- bear (label=0) = +0.028

Raw regime keys (column-as-found):
  - `label_0`: +0.028
  - `label_1`: +0.027
  - `label_2`: +0.004

Regime story: strongest absolute deviation in **bear** (z=+0.028); weakest in bull (z=+0.004); spread=0.024 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.031
- STEADY = +0.008
- VOLATILE = +0.019
- DEGEN = +0.050

DNA story: strongest in **DEGEN** (z=+0.050); weakest in STEADY (z=+0.008); cross-bucket spread = 0.041 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.009
- F2 (2023-11-01 .. 2024-02-29) = +0.025
- F3 (2024-03-01 .. 2024-05-15) = +0.022
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.37. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_eta_total_xrank`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_eta_total_xrank` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.03 -> t-0=+0.02 -> t+1=+0.40, while bottom-25% movers sit at t-1=+0.13 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.106. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.05). Fold consistency: fold-stable (F1=+0.01, F2=+0.03, F3=+0.02). Distributionally mean=+0.5135 std=0.2886 skew=-0.00 kurt=-1.20. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_notional_l1pct_mean`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +4443979.268758
- std  = 19820252.360894
- p10 = +484363.3179
- p50 = +1759942.5785
- p90 = +4311602.4057
- skew = +9.700
- excess kurtosis = +101.073

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.024 |
| t-3 | indicator 3 days BEFORE event | -0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.012 |
| t-0 | indicator on event day (concurrent) | -0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.016 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.091**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.103**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.015
- chop (label=1) = -0.027
- bear (label=0) = +0.022

Raw regime keys (column-as-found):
  - `label_2`: -0.015
  - `label_1`: -0.027
  - `label_0`: +0.022

Regime story: strongest absolute deviation in **chop** (z=-0.027); weakest in bull (z=-0.015); spread=0.012 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.159
- STEADY = -0.032
- VOLATILE = -0.026
- DEGEN = +0.042

DNA story: strongest in **BLUE** (z=+0.159); weakest in VOLATILE (z=-0.026); cross-bucket spread = 0.185 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.230
- F2 (2023-11-01 .. 2024-02-29) = -0.203
- F3 (2024-03-01 .. 2024-05-15) = +0.594
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 7.09. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_notional_l1pct_mean`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`bd_notional_l1pct_mean` is a weak concurrent discriminator: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=-0.01 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.09 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.103. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.16). Fold consistency: fold-unstable (F1=-0.23, F2=-0.20, F3=+0.59). Distributionally mean=+4443979.2688 std=19820252.3609 skew=+9.70 kurt=+101.07. not yet exploited in catalog engines.

### Playbook hint

- Strong standalone discriminator; candidate for a primary event-engine trigger.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `hbr_n_trades`

**Family**: `hbr_*` — Hawkes branching-ratio / self-exciting trade-process intensity

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +165010.442419
- std  = 475233.858510
- p10 = +7423.2000
- p50 = +48685.0000
- p90 = +355707.4000
- skew = +10.414
- excess kurtosis = +159.390

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.014 |
| t-3 | indicator 3 days BEFORE event | +0.019 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.026 |
| t-0 | indicator on event day (concurrent) | +0.023 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.257 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.120**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.094**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.036
- chop (label=1) = -0.029
- bear (label=0) = +0.065

Raw regime keys (column-as-found):
  - `label_1`: -0.029
  - `label_2`: +0.036
  - `label_0`: +0.065

Regime story: strongest absolute deviation in **bear** (z=+0.065); weakest in chop (z=-0.029); spread=0.095 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.031
- STEADY = +0.020
- VOLATILE = +0.020
- DEGEN = +0.054

DNA story: strongest in **DEGEN** (z=+0.054); weakest in STEADY (z=+0.020); cross-bucket spread = 0.033 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.492
- F2 (2023-11-01 .. 2024-02-29) = +0.086
- F3 (2024-03-01 .. 2024-05-15) = +0.400
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 157.25. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/hbr_n_trades`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`hbr_n_trades` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.02 -> t+1=+0.26, while bottom-25% movers sit at t-1=+0.12 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.094. Cross-regime: strongest in bear (z=+0.07). Cross-DNA: most pronounced in DEGEN bucket (z=+0.05). Fold consistency: fold-unstable (F1=-0.49, F2=+0.09, F3=+0.40). Distributionally mean=+165010.4424 std=475233.8585 skew=+10.41 kurt=+159.39. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_eta_total_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +1.000000
- std  = 0.098027
- p10 = +0.8714
- p50 = +1.0167
- p90 = +1.1028
- skew = -1.009
- excess kurtosis = +2.199

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.016 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.023 |
| t-0 | indicator on event day (concurrent) | +0.018 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.329 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.113**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.090**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.002
- chop (label=1) = +0.025
- bear (label=0) = +0.033

Raw regime keys (column-as-found):
  - `label_1`: +0.025
  - `label_2`: +0.002
  - `label_0`: +0.033

Regime story: strongest absolute deviation in **bear** (z=+0.033); weakest in bull (z=+0.002); spread=0.031 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.004
- STEADY = +0.005
- VOLATILE = +0.020
- DEGEN = +0.041

DNA story: strongest in **DEGEN** (z=+0.041); weakest in BLUE (z=-0.004); cross-bucket spread = 0.045 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.104
- F2 (2023-11-01 .. 2024-02-29) = -0.011
- F3 (2024-03-01 .. 2024-05-15) = -0.070
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 9.03. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_eta_total_xratio`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_eta_total_xratio` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.02 -> t+1=+0.33, while bottom-25% movers sit at t-1=+0.11 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.090. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-unstable (F1=+0.10, F2=-0.01, F3=-0.07). Distributionally mean=+1.0000 std=0.0980 skew=-1.01 kurt=+2.20. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_rv_rv_5m_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.111560
- std  = 0.314824
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.468
- excess kurtosis = +4.089

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.037 |
| t-3 | indicator 3 days BEFORE event | +0.053 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.056 |
| t-0 | indicator on event day (concurrent) | +0.064 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.377 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.145**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.089**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.058
- chop (label=1) = +0.057
- bear (label=0) = +0.078

Raw regime keys (column-as-found):
  - `label_1`: +0.057
  - `label_2`: +0.058
  - `label_0`: +0.078

Regime story: strongest absolute deviation in **bear** (z=+0.078); weakest in chop (z=+0.057); spread=0.021 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.084
- STEADY = +0.074
- VOLATILE = +0.058
- DEGEN = +0.057

DNA story: strongest in **BLUE** (z=+0.084); weakest in DEGEN (z=+0.057); cross-bucket spread = 0.027 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.070
- F2 (2023-11-01 .. 2024-02-29) = -0.096
- F3 (2024-03-01 .. 2024-05-15) = -0.181
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.41. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_rv_5m_xpct10`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_rv_5m_xpct10` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.05 -> t-1=+0.06 -> t-0=+0.06 -> t+1=+0.38, while bottom-25% movers sit at t-1=+0.15 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.089. Cross-regime: strongest in bear (z=+0.08). Cross-DNA: most pronounced in BLUE bucket (z=+0.08). Fold consistency: fold-stable (F1=-0.07, F2=-0.10, F3=-0.18). Distributionally mean=+0.1116 std=0.3148 skew=+2.47 kurt=+4.09. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_rv_bpv_5m_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.111560
- std  = 0.314824
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.468
- excess kurtosis = +4.089

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.031 |
| t-3 | indicator 3 days BEFORE event | +0.057 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.056 |
| t-0 | indicator on event day (concurrent) | +0.059 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.373 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.144**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.088**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.047
- chop (label=1) = +0.044
- bear (label=0) = +0.088

Raw regime keys (column-as-found):
  - `label_0`: +0.088
  - `label_2`: +0.047
  - `label_1`: +0.044

Regime story: strongest absolute deviation in **bear** (z=+0.088); weakest in chop (z=+0.044); spread=0.043 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.011
- STEADY = +0.067
- VOLATILE = +0.058
- DEGEN = +0.068

DNA story: strongest in **DEGEN** (z=+0.068); weakest in BLUE (z=-0.011); cross-bucket spread = 0.079 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.088
- F2 (2023-11-01 .. 2024-02-29) = -0.092
- F3 (2024-03-01 .. 2024-05-15) = -0.182
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.36. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_rv_bpv_5m_xpct10`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_rv_bpv_5m_xpct10` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.06 -> t-1=+0.06 -> t-0=+0.06 -> t+1=+0.37, while bottom-25% movers sit at t-1=+0.14 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.088. Cross-regime: strongest in bear (z=+0.09). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-stable (F1=-0.09, F2=-0.09, F3=-0.18). Distributionally mean=+0.1116 std=0.3148 skew=+2.47 kurt=+4.09. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_n_trades_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.111560
- std  = 0.314824
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.468
- excess kurtosis = +4.089

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | +0.029 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.017 |
| t-0 | indicator on event day (concurrent) | +0.022 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.203 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.101**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.085**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.034
- chop (label=1) = +0.010
- bear (label=0) = +0.013

Raw regime keys (column-as-found):
  - `label_0`: +0.013
  - `label_1`: +0.010
  - `label_2`: +0.034

Regime story: strongest absolute deviation in **bull** (z=+0.034); weakest in chop (z=+0.010); spread=0.024 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.025
- STEADY = +0.045
- VOLATILE = +0.014
- DEGEN = +0.017

DNA story: strongest in **STEADY** (z=+0.045); weakest in VOLATILE (z=+0.014); cross-bucket spread = 0.032 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.006
- F2 (2023-11-01 .. 2024-02-29) = +0.068
- F3 (2024-03-01 .. 2024-05-15) = -0.010
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.58. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_n_trades_xpct10`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_n_trades_xpct10` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.02 -> t-0=+0.02 -> t+1=+0.20, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.085. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in STEADY bucket (z=+0.05). Fold consistency: fold-unstable (F1=+0.01, F2=+0.07, F3=-0.01). Distributionally mean=+0.1116 std=0.3148 skew=+2.47 kurt=+4.09. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `hbr_eta_buy`

**Family**: `hbr_*` — Hawkes branching-ratio / self-exciting trade-process intensity

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.809044
- std  = 0.109224
- p10 = +0.6564
- p50 = +0.8307
- p90 = +0.9297
- skew = -1.033
- excess kurtosis = +1.075

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.018 |
| t-3 | indicator 3 days BEFORE event | +0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.016 |
| t-0 | indicator on event day (concurrent) | +0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.269 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.100**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.084**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.002
- chop (label=1) = -0.017
- bear (label=0) = +0.058

Raw regime keys (column-as-found):
  - `label_2`: -0.002
  - `label_1`: -0.017
  - `label_0`: +0.058

Regime story: strongest absolute deviation in **bear** (z=+0.058); weakest in bull (z=-0.002); spread=0.060 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.053
- STEADY = -0.004
- VOLATILE = +0.014
- DEGEN = +0.042

DNA story: strongest in **BLUE** (z=-0.053); weakest in STEADY (z=-0.004); cross-bucket spread = 0.049 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.576
- F2 (2023-11-01 .. 2024-02-29) = +0.033
- F3 (2024-03-01 .. 2024-05-15) = +0.322
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 5.09. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**13 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/hbr_eta_buy`).

### Emergent story

`hbr_eta_buy` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.27, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.084. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-unstable (F1=-0.58, F2=+0.03, F3=+0.32). Distributionally mean=+0.8090 std=0.1092 skew=-1.03 kurt=+1.07. used by 13 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_liq_long_usd_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51794
- mean = +1.000000
- std  = 2.643045
- p10 = +0.0000
- p50 = +0.0000
- p90 = +3.0779
- skew = +3.894
- excess kurtosis = +17.541

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | +0.021 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.026 |
| t-0 | indicator on event day (concurrent) | +0.029 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.138 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.109**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.083**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.015
- chop (label=1) = +0.022
- bear (label=0) = +0.053

Raw regime keys (column-as-found):
  - `label_2`: +0.015
  - `label_1`: +0.022
  - `label_0`: +0.053

Regime story: strongest absolute deviation in **bear** (z=+0.053); weakest in bull (z=+0.015); spread=0.038 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.017
- STEADY = +0.032
- VOLATILE = +0.021
- DEGEN = +0.061

DNA story: strongest in **DEGEN** (z=+0.061); weakest in BLUE (z=-0.017); cross-bucket spread = 0.077 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.028
- F2 (2023-11-01 .. 2024-02-29) = +0.120
- F3 (2024-03-01 .. 2024-05-15) = +0.143
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.97. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_liq_long_usd_xratio`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_liq_long_usd_xratio` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.03 -> t+1=+0.14, while bottom-25% movers sit at t-1=+0.11 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.083. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.03, F2=+0.12, F3=+0.14). Distributionally mean=+1.0000 std=2.6430 skew=+3.89 kurt=+17.54. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_hbr_eta_total_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.111560
- std  = 0.314824
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.468
- excess kurtosis = +4.089

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.016 |
| t-3 | indicator 3 days BEFORE event | +0.012 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.016 |
| t-0 | indicator on event day (concurrent) | +0.013 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.286 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.098**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.082**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.001
- chop (label=1) = +0.030
- bear (label=0) = +0.010

Raw regime keys (column-as-found):
  - `label_2`: +0.001
  - `label_1`: +0.030
  - `label_0`: +0.010

Regime story: strongest absolute deviation in **chop** (z=+0.030); weakest in bull (z=+0.001); spread=0.030 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.017
- STEADY = +0.048
- VOLATILE = -0.018
- DEGEN = +0.042

DNA story: strongest in **STEADY** (z=+0.048); weakest in BLUE (z=-0.017); cross-bucket spread = 0.065 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.005
- F2 (2023-11-01 .. 2024-02-29) = +0.028
- F3 (2024-03-01 .. 2024-05-15) = +0.023
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.94. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_hbr_eta_total_xpct10`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_hbr_eta_total_xpct10` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.29, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.082. Cross-regime: strongest in chop (z=+0.03). Cross-DNA: most pronounced in STEADY bucket (z=+0.05). Fold consistency: fold-unstable (F1=-0.00, F2=+0.03, F3=+0.02). Distributionally mean=+0.1116 std=0.3148 skew=+2.47 kurt=+4.09. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `wh_whale_trade_count_500k`

**Family**: `wh_*` — whale on-chain / large-trade activity (>500k USD trades, net flow)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +41.758649
- std  = 239.814863
- p10 = +0.0000
- p50 = +4.0000
- p90 = +78.0000
- skew = +34.542
- excess kurtosis = +1993.898

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | +0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.021 |
| t-0 | indicator on event day (concurrent) | +0.013 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.197 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.102**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.081**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.004
- chop (label=1) = -0.001
- bear (label=0) = +0.054

Raw regime keys (column-as-found):
  - `label_0`: +0.054
  - `label_1`: -0.001
  - `label_2`: -0.004

Regime story: strongest absolute deviation in **bear** (z=+0.054); weakest in chop (z=-0.001); spread=0.054 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.043
- STEADY = +0.011
- VOLATILE = +0.003
- DEGEN = +0.073

DNA story: strongest in **DEGEN** (z=+0.073); weakest in VOLATILE (z=+0.003); cross-bucket spread = 0.070 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.295
- F2 (2023-11-01 .. 2024-02-29) = -0.071
- F3 (2024-03-01 .. 2024-05-15) = +0.180
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.12. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**6 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/wh_whale_trade_count_500k`).

### Emergent story

`wh_whale_trade_count_500k` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.20, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.081. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-unstable (F1=-0.29, F2=-0.07, F3=+0.18). Distributionally mean=+41.7586 std=239.8149 skew=+34.54 kurt=+1993.90. used by 6 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_liq_long_usd_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +0.515397
- std  = 0.263029
- p10 = +0.2449
- p50 = +0.3659
- p90 = +0.9167
- skew = +0.465
- excess kurtosis = -1.303

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | +0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.026 |
| t-0 | indicator on event day (concurrent) | +0.029 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.135 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.106**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.080**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.001
- chop (label=1) = +0.054
- bear (label=0) = +0.040

Raw regime keys (column-as-found):
  - `label_1`: +0.054
  - `label_2`: -0.001
  - `label_0`: +0.040

Regime story: strongest absolute deviation in **chop** (z=+0.054); weakest in bull (z=-0.001); spread=0.056 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.011
- STEADY = +0.032
- VOLATILE = +0.023
- DEGEN = +0.056

DNA story: strongest in **DEGEN** (z=+0.056); weakest in BLUE (z=-0.011); cross-bucket spread = 0.067 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.231
- F2 (2023-11-01 .. 2024-02-29) = +0.183
- F3 (2024-03-01 .. 2024-05-15) = +0.169
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.14. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_liq_long_usd_xrank`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_liq_long_usd_xrank` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.03 -> t+1=+0.14, while bottom-25% movers sit at t-1=+0.11 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.080. Cross-regime: strongest in chop (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-stable (F1=+0.23, F2=+0.18, F3=+0.17). Distributionally mean=+0.5154 std=0.2630 skew=+0.46 kurt=-1.30. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `hbr_eta_total`

**Family**: `hbr_*` — Hawkes branching-ratio / self-exciting trade-process intensity

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.850253
- std  = 0.093570
- p10 = +0.7192
- p50 = +0.8723
- p90 = +0.9483
- skew = -1.244
- excess kurtosis = +1.803

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.016 |
| t-3 | indicator 3 days BEFORE event | +0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.016 |
| t-0 | indicator on event day (concurrent) | +0.011 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.281 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.096**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.079**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.004
- chop (label=1) = -0.011
- bear (label=0) = +0.058

Raw regime keys (column-as-found):
  - `label_1`: -0.011
  - `label_2`: -0.004
  - `label_0`: +0.058

Regime story: strongest absolute deviation in **bear** (z=+0.058); weakest in bull (z=-0.004); spread=0.062 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.055
- STEADY = -0.009
- VOLATILE = +0.020
- DEGEN = +0.042

DNA story: strongest in **BLUE** (z=-0.055); weakest in STEADY (z=-0.009); cross-bucket spread = 0.046 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.532
- F2 (2023-11-01 .. 2024-02-29) = +0.036
- F3 (2024-03-01 .. 2024-05-15) = +0.295
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 5.14. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**13 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/hbr_eta_total`).

### Emergent story

`hbr_eta_total` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.28, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.079. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-unstable (F1=-0.53, F2=+0.04, F3=+0.29). Distributionally mean=+0.8503 std=0.0936 skew=-1.24 kurt=+1.80. used by 13 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `hbr_eta_sell`

**Family**: `hbr_*` — Hawkes branching-ratio / self-exciting trade-process intensity

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.812178
- std  = 0.111331
- p10 = +0.6583
- p50 = +0.8357
- p90 = +0.9321
- skew = -1.166
- excess kurtosis = +1.722

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.017 |
| t-3 | indicator 3 days BEFORE event | +0.009 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.020 |
| t-0 | indicator on event day (concurrent) | +0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.286 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.097**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.078**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.000
- chop (label=1) = -0.002
- bear (label=0) = +0.059

Raw regime keys (column-as-found):
  - `label_0`: +0.059
  - `label_1`: -0.002
  - `label_2`: -0.000

Regime story: strongest absolute deviation in **bear** (z=+0.059); weakest in bull (z=-0.000); spread=0.059 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.052
- STEADY = -0.005
- VOLATILE = +0.026
- DEGEN = +0.046

DNA story: strongest in **BLUE** (z=-0.052); weakest in STEADY (z=-0.005); cross-bucket spread = 0.048 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.512
- F2 (2023-11-01 .. 2024-02-29) = +0.025
- F3 (2024-03-01 .. 2024-05-15) = +0.292
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 5.13. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/hbr_eta_sell`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`hbr_eta_sell` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.02 -> t+1=+0.29, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.078. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-unstable (F1=-0.51, F2=+0.02, F3=+0.29). Distributionally mean=+0.8122 std=0.1113 skew=-1.17 kurt=+1.72. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `wh_whale_buy_usd`

**Family**: `wh_*` — whale on-chain / large-trade activity (>500k USD trades, net flow)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +2615486.961451
- std  = 13744777.407480
- p10 = +0.0000
- p50 = +146032.7372
- p90 = +4983033.2012
- skew = +30.379
- excess kurtosis = +1678.665

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | +0.018 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.022 |
| t-0 | indicator on event day (concurrent) | +0.012 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.227 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.098**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.075**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.005
- chop (label=1) = -0.003
- bear (label=0) = +0.038

Raw regime keys (column-as-found):
  - `label_2`: +0.005
  - `label_1`: -0.003
  - `label_0`: +0.038

Regime story: strongest absolute deviation in **bear** (z=+0.038); weakest in chop (z=-0.003); spread=0.041 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.045
- STEADY = +0.012
- VOLATILE = -0.004
- DEGEN = +0.080

DNA story: strongest in **DEGEN** (z=+0.080); weakest in VOLATILE (z=-0.004); cross-bucket spread = 0.083 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.263
- F2 (2023-11-01 .. 2024-02-29) = -0.073
- F3 (2024-03-01 .. 2024-05-15) = +0.145
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.61. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/wh_whale_buy_usd`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`wh_whale_buy_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.23, while bottom-25% movers sit at t-1=+0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.075. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=-0.26, F2=-0.07, F3=+0.14). Distributionally mean=+2615486.9615 std=13744777.4075 skew=+30.38 kurt=+1678.66. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `wh_whale_sell_usd`

**Family**: `wh_*` — whale on-chain / large-trade activity (>500k USD trades, net flow)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +2601392.563764
- std  = 13161968.938237
- p10 = +0.0000
- p50 = +154774.1749
- p90 = +5088897.9583
- skew = +27.803
- excess kurtosis = +1367.960

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.014 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.021 |
| t-0 | indicator on event day (concurrent) | +0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.143 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.095**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.074**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.006
- chop (label=1) = +0.001
- bear (label=0) = +0.062

Raw regime keys (column-as-found):
  - `label_1`: +0.001
  - `label_0`: +0.062
  - `label_2`: -0.006

Regime story: strongest absolute deviation in **bear** (z=+0.062); weakest in chop (z=+0.001); spread=0.061 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.047
- STEADY = +0.009
- VOLATILE = +0.012
- DEGEN = +0.065

DNA story: strongest in **DEGEN** (z=+0.065); weakest in STEADY (z=+0.009); cross-bucket spread = 0.056 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.283
- F2 (2023-11-01 .. 2024-02-29) = -0.080
- F3 (2024-03-01 .. 2024-05-15) = +0.154
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.56. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/wh_whale_sell_usd`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`wh_whale_sell_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.14, while bottom-25% movers sit at t-1=+0.09 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.074. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.28, F2=-0.08, F3=+0.15). Distributionally mean=+2601392.5638 std=13161968.9382 skew=+27.80 kurt=+1367.96. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `wh_whale_trade_count`

**Family**: `wh_*` — whale on-chain / large-trade activity (>500k USD trades, net flow)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +8.721156
- std  = 67.265532
- p10 = +0.0000
- p50 = +0.0000
- p90 = +13.0000
- skew = +37.689
- excess kurtosis = +2199.107

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.005 |
| t-3 | indicator 3 days BEFORE event | +0.016 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.021 |
| t-0 | indicator on event day (concurrent) | +0.012 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.179 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.093**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.072**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.001
- chop (label=1) = +0.004
- bear (label=0) = +0.040

Raw regime keys (column-as-found):
  - `label_0`: +0.040
  - `label_2`: -0.001
  - `label_1`: +0.004

Regime story: strongest absolute deviation in **bear** (z=+0.040); weakest in bull (z=-0.001); spread=0.042 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.063
- STEADY = +0.010
- VOLATILE = +0.006
- DEGEN = +0.064

DNA story: strongest in **DEGEN** (z=+0.064); weakest in VOLATILE (z=+0.006); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.251
- F2 (2023-11-01 .. 2024-02-29) = -0.087
- F3 (2024-03-01 .. 2024-05-15) = +0.117
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.04. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/wh_whale_trade_count`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`wh_whale_trade_count` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.09 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.072. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.25, F2=-0.09, F3=+0.12). Distributionally mean=+8.7212 std=67.2655 skew=+37.69 kurt=+2199.11. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_total_usd`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +20615855.618942
- std  = 90792985.019218
- p10 = +0.0000
- p50 = +120887.5838
- p90 = +48222082.4584
- skew = +24.932
- excess kurtosis = +1278.095

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | +0.018 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.024 |
| t-0 | indicator on event day (concurrent) | +0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.184 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.087**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.063**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.007
- chop (label=1) = -0.003
- bear (label=0) = +0.050

Raw regime keys (column-as-found):
  - `label_0`: +0.050
  - `label_1`: -0.003
  - `label_2`: +0.007

Regime story: strongest absolute deviation in **bear** (z=+0.050); weakest in chop (z=-0.003); spread=0.053 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.047
- STEADY = +0.014
- VOLATILE = +0.005
- DEGEN = +0.076

DNA story: strongest in **DEGEN** (z=+0.076); weakest in VOLATILE (z=+0.005); cross-bucket spread = 0.071 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.249
- F2 (2023-11-01 .. 2024-02-29) = -0.036
- F3 (2024-03-01 .. 2024-05-15) = +0.218
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 8.56. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_total_usd`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`liq_total_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.02 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.09 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.063. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=-0.25, F2=-0.04, F3=+0.22). Distributionally mean=+20615855.6189 std=90792985.0192 skew=+24.93 kurt=+1278.10. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `rv_bpv_5m`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.004658
- std  = 0.011479
- p10 = +0.0005
- p50 = +0.0021
- p90 = +0.0096
- skew = +20.086
- excess kurtosis = +909.342

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.019 |
| t-3 | indicator 3 days BEFORE event | +0.024 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.020 |
| t-0 | indicator on event day (concurrent) | +0.028 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.177 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.082**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.062**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.039
- chop (label=1) = -0.016
- bear (label=0) = +0.063

Raw regime keys (column-as-found):
  - `label_2`: +0.039
  - `label_0`: +0.063
  - `label_1`: -0.016

Regime story: strongest absolute deviation in **bear** (z=+0.063); weakest in chop (z=-0.016); spread=0.079 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.047
- STEADY = +0.017
- VOLATILE = +0.041
- DEGEN = +0.036

DNA story: strongest in **BLUE** (z=-0.047); weakest in STEADY (z=+0.017); cross-bucket spread = 0.064 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.314
- F2 (2023-11-01 .. 2024-02-29) = -0.056
- F3 (2024-03-01 .. 2024-05-15) = -0.022
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 1.00. (Magnitudes are tight - stable signal.)

### Catalog usage

**7 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_bpv_5m`).

### Emergent story

`rv_bpv_5m` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.03 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.062. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-stable (F1=-0.31, F2=-0.06, F3=-0.02). Distributionally mean=+0.0047 std=0.0115 skew=+20.09 kurt=+909.34. used by 7 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `rv_rv_5m`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.004999
- std  = 0.011886
- p10 = +0.0006
- p50 = +0.0023
- p90 = +0.0104
- skew = +18.238
- excess kurtosis = +753.508

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.021 |
| t-3 | indicator 3 days BEFORE event | +0.024 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.021 |
| t-0 | indicator on event day (concurrent) | +0.027 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.179 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.083**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.062**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.037
- chop (label=1) = -0.015
- bear (label=0) = +0.060

Raw regime keys (column-as-found):
  - `label_0`: +0.060
  - `label_1`: -0.015
  - `label_2`: +0.037

Regime story: strongest absolute deviation in **bear** (z=+0.060); weakest in chop (z=-0.015); spread=0.075 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.045
- STEADY = +0.017
- VOLATILE = +0.040
- DEGEN = +0.034

DNA story: strongest in **BLUE** (z=-0.045); weakest in STEADY (z=+0.017); cross-bucket spread = 0.061 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.311
- F2 (2023-11-01 .. 2024-02-29) = -0.061
- F3 (2024-03-01 .. 2024-05-15) = -0.041
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.89. (Magnitudes are tight - stable signal.)

### Catalog usage

**5 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_rv_5m`).

### Emergent story

`rv_rv_5m` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.03 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.062. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=-0.04). Fold consistency: fold-stable (F1=-0.31, F2=-0.06, F3=-0.04). Distributionally mean=+0.0050 std=0.0119 skew=+18.24 kurt=+753.51. used by 5 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_global_lsr`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31145
- mean = +2.515135
- std  = 1.106994
- p10 = +1.2140
- p50 = +2.3684
- p90 = +3.9878
- skew = +0.877
- excess kurtosis = +1.217

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.046 |
| t-3 | indicator 3 days BEFORE event | -0.062 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.066 |
| t-0 | indicator on event day (concurrent) | -0.074 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.238 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.128**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.062**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.006
- chop (label=1) = -0.118
- bear (label=0) = -0.115

Raw regime keys (column-as-found):
  - `label_2`: -0.006
  - `label_0`: -0.115
  - `label_1`: -0.118

Regime story: strongest absolute deviation in **chop** (z=-0.118); weakest in bull (z=-0.006); spread=0.112 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.025
- STEADY = -0.039
- VOLATILE = -0.098
- DEGEN = -0.115

DNA story: strongest in **DEGEN** (z=-0.115); weakest in BLUE (z=+0.025); cross-bucket spread = 0.141 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.325
- F2 (2023-11-01 .. 2024-02-29) = +0.040
- F3 (2024-03-01 .. 2024-05-15) = +0.653
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.28. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_global_lsr`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`s3_global_lsr` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=-0.06 -> t-1=-0.07 -> t-0=-0.07 -> t+1=-0.24, while bottom-25% movers sit at t-1=-0.13 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.062. Cross-regime: strongest in chop (z=-0.12). Cross-DNA: most pronounced in DEGEN bucket (z=-0.12). Fold consistency: fold-unstable (F1=-0.32, F2=+0.04, F3=+0.65). Distributionally mean=+2.5151 std=1.1070 skew=+0.88 kurt=+1.22. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_short_usd`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +10178145.714441
- std  = 45750562.263820
- p10 = +0.0000
- p50 = +0.0000
- p90 = +23389104.9120
- skew = +26.459
- excess kurtosis = +1436.487

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.020 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.023 |
| t-0 | indicator on event day (concurrent) | +0.013 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.210 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.084**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.061**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.013
- chop (label=1) = -0.006
- bear (label=0) = +0.036

Raw regime keys (column-as-found):
  - `label_1`: -0.006
  - `label_0`: +0.036
  - `label_2`: +0.013

Regime story: strongest absolute deviation in **bear** (z=+0.036); weakest in chop (z=-0.006); spread=0.042 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.045
- STEADY = +0.017
- VOLATILE = -0.003
- DEGEN = +0.077

DNA story: strongest in **DEGEN** (z=+0.077); weakest in VOLATILE (z=-0.003); cross-bucket spread = 0.080 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.230
- F2 (2023-11-01 .. 2024-02-29) = -0.028
- F3 (2024-03-01 .. 2024-05-15) = +0.227
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 17.77. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**3 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_short_usd`).

### Emergent story

`liq_short_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.21, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.061. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=-0.23, F2=-0.03, F3=+0.23). Distributionally mean=+10178145.7144 std=45750562.2638 skew=+26.46 kurt=+1436.49. used by 3 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_wh_whale_net_usd_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +0.113445
- std  = 0.317136
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.438
- excess kurtosis = +3.943

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.015 |
| t-3 | indicator 3 days BEFORE event | +0.017 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.009 |
| t-0 | indicator on event day (concurrent) | +0.030 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.215 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.069**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.060**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.028
- chop (label=1) = +0.029
- bear (label=0) = +0.030

Raw regime keys (column-as-found):
  - `label_0`: +0.030
  - `label_1`: +0.029
  - `label_2`: +0.028

Regime story: strongest absolute deviation in **bear** (z=+0.030); weakest in bull (z=+0.028); spread=0.002 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.015
- STEADY = +0.029
- VOLATILE = +0.027
- DEGEN = +0.044

DNA story: strongest in **DEGEN** (z=+0.044); weakest in BLUE (z=+0.015); cross-bucket spread = 0.029 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.015
- F2 (2023-11-01 .. 2024-02-29) = +0.011
- F3 (2024-03-01 .. 2024-05-15) = +0.082
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.91. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_wh_whale_net_usd_xpct10`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`xrel_wh_whale_net_usd_xpct10` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.01 -> t-0=+0.03 -> t+1=+0.21, while bottom-25% movers sit at t-1=+0.07 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.060. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-stable (F1=+0.02, F2=+0.01, F3=+0.08). Distributionally mean=+0.1134 std=0.3171 skew=+2.44 kurt=+3.94. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `soc_wiki_views`

**Family**: `soc_*` — social / sentiment (Twitter/Reddit volume, sentiment z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 1360
- mean = +1702.004412
- std  = 2915.460392
- p10 = +131.0000
- p50 = +507.0000
- p90 = +5254.8000
- skew = +3.555
- excess kurtosis = +18.734

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.072 |
| t-3 | indicator 3 days BEFORE event | +0.053 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.013 |
| t-0 | indicator on event day (concurrent) | -0.073 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.086 |

**Lead/Lag verdict**: mixed (no single lag dominates).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.045**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.058**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.027
- chop (label=1) = -0.168
- bear (label=0) = -0.052

Raw regime keys (column-as-found):
  - `label_0`: -0.052
  - `label_1`: -0.168
  - `label_2`: -0.027

Regime story: strongest absolute deviation in **chop** (z=-0.168); weakest in bull (z=-0.027); spread=0.141 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.126
- STEADY = -0.091
- VOLATILE = +0.171
- DEGEN = +nan

DNA story: strongest in **VOLATILE** (z=+0.171); weakest in STEADY (z=-0.091); cross-bucket spread = 0.262 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = -0.419
- F3 (2024-03-01 .. 2024-05-15) = +0.178
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.48. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/soc_wiki_views`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`soc_wiki_views` is a weak concurrent discriminator: top-25%-mover z-lift goes t-3=+0.05 -> t-1=-0.01 -> t-0=-0.07 -> t+1=+0.09, while bottom-25% movers sit at t-1=+0.04 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.058. Cross-regime: strongest in chop (z=-0.17). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.17). Fold consistency: fold-unstable (F1=+nan, F2=-0.42, F3=+0.18). Distributionally mean=+1702.0044 std=2915.4604 skew=+3.55 kurt=+18.73. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `liq_long_usd`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +10437709.904501
- std  = 45264776.237576
- p10 = +0.0000
- p50 = +0.0000
- p90 = +24693995.6473
- skew = +23.363
- excess kurtosis = +1125.174

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | +0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.024 |
| t-0 | indicator on event day (concurrent) | +0.017 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.132 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.082**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.058**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.004
- chop (label=1) = +0.002
- bear (label=0) = +0.064

Raw regime keys (column-as-found):
  - `label_1`: +0.002
  - `label_0`: +0.064
  - `label_2`: -0.004

Regime story: strongest absolute deviation in **bear** (z=+0.064); weakest in chop (z=+0.002); spread=0.063 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.048
- STEADY = +0.011
- VOLATILE = +0.016
- DEGEN = +0.064

DNA story: strongest in **DEGEN** (z=+0.064); weakest in STEADY (z=+0.011); cross-bucket spread = 0.053 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.247
- F2 (2023-11-01 .. 2024-02-29) = -0.039
- F3 (2024-03-01 .. 2024-05-15) = +0.200
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 6.30. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**3 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_long_usd`).

### Emergent story

`liq_long_usd` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.02 -> t+1=+0.13, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.058. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.25, F2=-0.04, F3=+0.20). Distributionally mean=+10437709.9045 std=45264776.2376 skew=+23.36 kurt=+1125.17. used by 3 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_taker_lsr`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30815
- mean = +1.166290
- std  = 0.986562
- p10 = +1.0178
- p50 = +1.0893
- p90 = +1.3052
- skew = +115.519
- excess kurtosis = +15978.558

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.008 |
| t-0 | indicator on event day (concurrent) | -0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.060 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.066**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.058**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.022
- chop (label=1) = -0.003
- bear (label=0) = -0.063

Raw regime keys (column-as-found):
  - `label_2`: +0.022
  - `label_0`: -0.063
  - `label_1`: -0.003

Regime story: strongest absolute deviation in **bear** (z=-0.063); weakest in chop (z=-0.003); spread=0.060 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.086
- STEADY = +0.023
- VOLATILE = -0.042
- DEGEN = -0.016

DNA story: strongest in **BLUE** (z=+0.086); weakest in DEGEN (z=-0.016); cross-bucket spread = 0.102 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.624
- F2 (2023-11-01 .. 2024-02-29) = -0.193
- F3 (2024-03-01 .. 2024-05-15) = -0.192
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.82. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_taker_lsr`).

**Untapped opportunity flag**: |discriminator| >= 0.05 with zero current engines — candidate for new event-engine construction.

### Emergent story

`s3_taker_lsr` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.01 -> t-0=-0.01 -> t+1=-0.06, while bottom-25% movers sit at t-1=-0.07 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.058. Cross-regime: strongest in bear (z=-0.06). Cross-DNA: most pronounced in BLUE bucket (z=+0.09). Fold consistency: fold-unstable (F1=+0.62, F2=-0.19, F3=-0.19). Distributionally mean=+1.1663 std=0.9866 skew=+115.52 kurt=+15978.56. not yet exploited in catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_short_z30`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.050248
- std  = 1.126416
- p10 = -0.7600
- p50 = -0.2450
- p90 = +1.2076
- skew = +2.653
- excess kurtosis = +7.897

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | +0.022 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.024 |
| t-0 | indicator on event day (concurrent) | +0.027 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.328 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.076**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.051**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.016
- chop (label=1) = +0.035
- bear (label=0) = +0.035

Raw regime keys (column-as-found):
  - `label_1`: +0.035
  - `label_0`: +0.035
  - `label_2`: +0.016

Regime story: strongest absolute deviation in **bear** (z=+0.035); weakest in bull (z=+0.016); spread=0.019 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.042
- STEADY = +0.041
- VOLATILE = +0.002
- DEGEN = +0.097

DNA story: strongest in **DEGEN** (z=+0.097); weakest in VOLATILE (z=+0.002); cross-bucket spread = 0.095 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.023
- F2 (2023-11-01 .. 2024-02-29) = +0.122
- F3 (2024-03-01 .. 2024-05-15) = -0.055
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.42. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**3 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_short_z30`).

### Emergent story

`liq_short_z30` is a LAGGING confirm (post-event drift): top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.03 -> t+1=+0.33, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.051. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.10). Fold consistency: fold-unstable (F1=+0.02, F2=+0.12, F3=-0.06). Distributionally mean=+0.0502 std=1.1264 skew=+2.65 kurt=+7.90. used by 3 catalog engines.

### Playbook hint

- Modest standalone discriminator; viable as a sizing-multiplier or filter input.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_long_z30`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.048900
- std  = 1.105463
- p10 = -0.7860
- p50 = -0.1826
- p90 = +1.1789
- skew = +2.626
- excess kurtosis = +7.982

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.008 |
| t-3 | indicator 3 days BEFORE event | +0.019 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.025 |
| t-0 | indicator on event day (concurrent) | +0.029 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.216 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.071**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.046**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.002
- chop (label=1) = +0.037
- bear (label=0) = +0.064

Raw regime keys (column-as-found):
  - `label_1`: +0.037
  - `label_0`: +0.064
  - `label_2`: -0.002

Regime story: strongest absolute deviation in **bear** (z=+0.064); weakest in bull (z=-0.002); spread=0.066 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.053
- STEADY = +0.027
- VOLATILE = +0.027
- DEGEN = +0.074

DNA story: strongest in **DEGEN** (z=+0.074); weakest in VOLATILE (z=+0.027); cross-bucket spread = 0.047 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.040
- F2 (2023-11-01 .. 2024-02-29) = +0.135
- F3 (2024-03-01 .. 2024-05-15) = -0.056
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.97. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**1 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_long_z30`).

### Emergent story

`liq_long_z30` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.03 -> t+1=+0.22, while bottom-25% movers sit at t-1=+0.07 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.046. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-unstable (F1=+0.04, F2=+0.14, F3=-0.06). Distributionally mean=+0.0489 std=1.1055 skew=+2.63 kurt=+7.98. used by 1 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_top_acct_lsr`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30659
- mean = +2.360956
- std  = 1.001538
- p10 = +1.2055
- p50 = +2.2381
- p90 = +3.6427
- skew = +1.035
- excess kurtosis = +2.226

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.043 |
| t-3 | indicator 3 days BEFORE event | -0.058 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.064 |
| t-0 | indicator on event day (concurrent) | -0.065 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.182 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.105**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.041**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.006
- chop (label=1) = -0.106
- bear (label=0) = -0.116

Raw regime keys (column-as-found):
  - `label_1`: -0.106
  - `label_0`: -0.116
  - `label_2`: +0.006

Regime story: strongest absolute deviation in **bear** (z=-0.116); weakest in bull (z=+0.006); spread=0.122 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.034
- STEADY = -0.025
- VOLATILE = -0.088
- DEGEN = -0.099

DNA story: strongest in **DEGEN** (z=-0.099); weakest in STEADY (z=-0.025); cross-bucket spread = 0.074 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.315
- F2 (2023-11-01 .. 2024-02-29) = +0.036
- F3 (2024-03-01 .. 2024-05-15) = +0.805
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.67. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_top_acct_lsr`).

### Emergent story

`s3_top_acct_lsr` is a low-signal background: top-25%-mover z-lift goes t-3=-0.06 -> t-1=-0.06 -> t-0=-0.07 -> t+1=-0.18, while bottom-25% movers sit at t-1=-0.10 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.041. Cross-regime: strongest in bear (z=-0.12). Cross-DNA: most pronounced in DEGEN bucket (z=-0.10). Fold consistency: fold-unstable (F1=-0.31, F2=+0.04, F3=+0.81). Distributionally mean=+2.3610 std=1.0015 skew=+1.03 kurt=+2.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_global_lsr_z`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30685
- mean = +0.063921
- std  = 1.298846
- p10 = -1.5448
- p50 = +0.0383
- p90 = +1.7423
- skew = +0.002
- excess kurtosis = +0.286

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.018 |
| t-3 | indicator 3 days BEFORE event | -0.043 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.044 |
| t-0 | indicator on event day (concurrent) | -0.051 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.296 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.083**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.040**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.009
- chop (label=1) = -0.113
- bear (label=0) = -0.038

Raw regime keys (column-as-found):
  - `label_1`: -0.113
  - `label_2`: -0.009
  - `label_0`: -0.038

Regime story: strongest absolute deviation in **chop** (z=-0.113); weakest in bull (z=-0.009); spread=0.104 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.006
- STEADY = -0.039
- VOLATILE = -0.055
- DEGEN = -0.095

DNA story: strongest in **DEGEN** (z=-0.095); weakest in BLUE (z=+0.006); cross-bucket spread = 0.101 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.069
- F2 (2023-11-01 .. 2024-02-29) = +0.025
- F3 (2024-03-01 .. 2024-05-15) = -0.064
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.20. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_global_lsr_z`).

### Emergent story

`s3_global_lsr_z` is a low-signal background: top-25%-mover z-lift goes t-3=-0.04 -> t-1=-0.04 -> t-0=-0.05 -> t+1=-0.30, while bottom-25% movers sit at t-1=-0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.040. Cross-regime: strongest in chop (z=-0.11). Cross-DNA: most pronounced in DEGEN bucket (z=-0.10). Fold consistency: fold-unstable (F1=-0.07, F2=+0.02, F3=-0.06). Distributionally mean=+0.0639 std=1.2988 skew=+0.00 kurt=+0.29. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_smart_vs_retail`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30659
- mean = -1.262748
- std  = 1.019478
- p10 = -2.5503
- p50 = -1.1599
- p90 = -0.0338
- skew = -0.563
- excess kurtosis = +0.722

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.033 |
| t-3 | indicator 3 days BEFORE event | +0.048 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.050 |
| t-0 | indicator on event day (concurrent) | +0.052 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.179 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.089**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.039**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.012
- chop (label=1) = +0.065
- bear (label=0) = +0.092

Raw regime keys (column-as-found):
  - `label_0`: +0.092
  - `label_1`: +0.065
  - `label_2`: +0.012

Regime story: strongest absolute deviation in **bear** (z=+0.092); weakest in bull (z=+0.012); spread=0.080 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.030
- STEADY = +0.017
- VOLATILE = +0.075
- DEGEN = +0.065

DNA story: strongest in **VOLATILE** (z=+0.075); weakest in STEADY (z=+0.017); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.233
- F2 (2023-11-01 .. 2024-02-29) = +0.213
- F3 (2024-03-01 .. 2024-05-15) = +0.027
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.59. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_vs_retail`).

### Emergent story

`s3_smart_vs_retail` is a low-signal background: top-25%-mover z-lift goes t-3=+0.05 -> t-1=+0.05 -> t-0=+0.05 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.09 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.039. Cross-regime: strongest in bear (z=+0.09). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.08). Fold consistency: fold-stable (F1=+0.23, F2=+0.21, F3=+0.03). Distributionally mean=-1.2627 std=1.0195 skew=-0.56 kurt=+0.72. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_hawkes_imbalance`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.004596
- std  = 0.642408
- p10 = -0.3327
- p50 = +0.0033
- p90 = +0.3517
- skew = -0.259
- excess kurtosis = +25.287

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.006 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.015 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.020**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.035**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.009
- chop (label=1) = +0.003
- bear (label=0) = +0.016

Raw regime keys (column-as-found):
  - `label_1`: +0.003
  - `label_2`: -0.009
  - `label_0`: +0.016

Regime story: strongest absolute deviation in **bear** (z=+0.016); weakest in chop (z=+0.003); spread=0.013 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.045
- STEADY = +0.001
- VOLATILE = +0.002
- DEGEN = -0.012

DNA story: strongest in **BLUE** (z=+0.045); weakest in STEADY (z=+0.001); cross-bucket spread = 0.044 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.017
- F2 (2023-11-01 .. 2024-02-29) = -0.017
- F3 (2024-03-01 .. 2024-05-15) = -0.042
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.47. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_hawkes_imbalance`).

### Emergent story

`norm_hawkes_imbalance` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.02 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.02 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.035. Cross-regime: strongest in bear (z=+0.02). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-stable (F1=-0.02, F2=-0.02, F3=-0.04). Distributionally mean=+0.0046 std=0.6424 skew=-0.26 kurt=+25.29. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `liq_short_xsec_z`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 50868
- mean = +0.010287
- std  = 0.985406
- p10 = -0.4714
- p50 = -0.3288
- p90 = +0.8944
- skew = +3.200
- excess kurtosis = +10.270

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.008 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.008 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.136 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.040**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.035**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.004
- chop (label=1) = +0.046
- bear (label=0) = -0.020

Raw regime keys (column-as-found):
  - `label_1`: +0.046
  - `label_0`: -0.020
  - `label_2`: -0.004

Regime story: strongest absolute deviation in **chop** (z=+0.046); weakest in bull (z=-0.004); spread=0.049 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.014
- STEADY = +0.018
- VOLATILE = -0.004
- DEGEN = +0.030

DNA story: strongest in **DEGEN** (z=+0.030); weakest in VOLATILE (z=-0.004); cross-bucket spread = 0.034 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.311
- F2 (2023-11-01 .. 2024-02-29) = +0.131
- F3 (2024-03-01 .. 2024-05-15) = +0.267
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.32. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_short_xsec_z`).

### Emergent story

`liq_short_xsec_z` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.14, while bottom-25% movers sit at t-1=+0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.035. Cross-regime: strongest in chop (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.03). Fold consistency: fold-stable (F1=+0.31, F2=+0.13, F3=+0.27). Distributionally mean=+0.0103 std=0.9854 skew=+3.20 kurt=+10.27. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_notional_skew`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +0.118579
- std  = 0.079894
- p10 = +0.0231
- p50 = +0.1148
- p90 = +0.2207
- skew = +0.062
- excess kurtosis = +0.894

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.040 |
| t-3 | indicator 3 days BEFORE event | -0.032 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.024 |
| t-0 | indicator on event day (concurrent) | -0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.227 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.058**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.034**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.014
- chop (label=1) = -0.049
- bear (label=0) = +0.044

Raw regime keys (column-as-found):
  - `label_0`: +0.044
  - `label_1`: -0.049
  - `label_2`: -0.014

Regime story: strongest absolute deviation in **chop** (z=-0.049); weakest in bull (z=-0.014); spread=0.035 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.074
- STEADY = -0.008
- VOLATILE = -0.012
- DEGEN = -0.013

DNA story: strongest in **BLUE** (z=+0.074); weakest in STEADY (z=-0.008); cross-bucket spread = 0.083 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.165
- F2 (2023-11-01 .. 2024-02-29) = +0.207
- F3 (2024-03-01 .. 2024-05-15) = +0.412
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.58. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_notional_skew`).

### Emergent story

`bd_notional_skew` is a low-signal background: top-25%-mover z-lift goes t-3=-0.03 -> t-1=-0.02 -> t-0=-0.01 -> t+1=-0.23, while bottom-25% movers sit at t-1=-0.06 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.034. Cross-regime: strongest in chop (z=-0.05). Cross-DNA: most pronounced in BLUE bucket (z=+0.07). Fold consistency: fold-unstable (F1=-0.17, F2=+0.21, F3=+0.41). Distributionally mean=+0.1186 std=0.0799 skew=+0.06 kurt=+0.89. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_long_xsec_z`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 50868
- mean = +0.010876
- std  = 0.987092
- p10 = -0.4807
- p50 = -0.3358
- p90 = +0.9403
- skew = +3.136
- excess kurtosis = +9.866

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.009 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.004 |
| t-0 | indicator on event day (concurrent) | +0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.083 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.038**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.034**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.003
- chop (label=1) = +0.050
- bear (label=0) = -0.020

Raw regime keys (column-as-found):
  - `label_1`: +0.050
  - `label_0`: -0.020
  - `label_2`: -0.003

Regime story: strongest absolute deviation in **chop** (z=+0.050); weakest in bull (z=-0.003); spread=0.053 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.018
- STEADY = +0.016
- VOLATILE = +0.003
- DEGEN = +0.025

DNA story: strongest in **DEGEN** (z=+0.025); weakest in VOLATILE (z=+0.003); cross-bucket spread = 0.022 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.346
- F2 (2023-11-01 .. 2024-02-29) = +0.118
- F3 (2024-03-01 .. 2024-05-15) = +0.251
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.39. (Magnitudes are tight - stable signal.)

### Catalog usage

**9 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_long_xsec_z`).

### Emergent story

`liq_long_xsec_z` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=+0.01 -> t+1=+0.08, while bottom-25% movers sit at t-1=+0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.034. Cross-regime: strongest in chop (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.03). Fold consistency: fold-stable (F1=+0.35, F2=+0.12, F3=+0.25). Distributionally mean=+0.0109 std=0.9871 skew=+3.14 kurt=+9.87. used by 9 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_smart_vs_retail_z`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 26587
- mean = +0.064477
- std  = 1.363763
- p10 = -1.6427
- p50 = +0.1251
- p90 = +1.7116
- skew = -0.319
- excess kurtosis = +0.770

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.029 |
| t-3 | indicator 3 days BEFORE event | +0.047 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.049 |
| t-0 | indicator on event day (concurrent) | +0.046 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.239 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.081**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.031**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.004
- chop (label=1) = +0.089
- bear (label=0) = +0.067

Raw regime keys (column-as-found):
  - `label_2`: -0.004
  - `label_0`: +0.067
  - `label_1`: +0.089

Regime story: strongest absolute deviation in **chop** (z=+0.089); weakest in bull (z=-0.004); spread=0.092 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.020
- STEADY = +0.029
- VOLATILE = +0.056
- DEGEN = +0.077

DNA story: strongest in **DEGEN** (z=+0.077); weakest in BLUE (z=-0.020); cross-bucket spread = 0.096 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.027
- F2 (2023-11-01 .. 2024-02-29) = +0.020
- F3 (2024-03-01 .. 2024-05-15) = -0.076
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.92. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_vs_retail_z`).

### Emergent story

`s3_smart_vs_retail_z` is a low-signal background: top-25%-mover z-lift goes t-3=+0.05 -> t-1=+0.05 -> t-0=+0.05 -> t+1=+0.24, while bottom-25% movers sit at t-1=+0.08 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.031. Cross-regime: strongest in chop (z=+0.09). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=+0.03, F2=+0.02, F3=-0.08). Distributionally mean=+0.0645 std=1.3638 skew=-0.32 kurt=+0.77. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_short_spike`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.062771
- std  = 0.242551
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +3.605
- excess kurtosis = +10.998

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.007 |
| t-3 | indicator 3 days BEFORE event | +0.020 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.028 |
| t-0 | indicator on event day (concurrent) | +0.024 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.265 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.058**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.030**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.021
- chop (label=1) = +0.024
- bear (label=0) = +0.028

Raw regime keys (column-as-found):
  - `label_1`: +0.024
  - `label_2`: +0.021
  - `label_0`: +0.028

Regime story: strongest absolute deviation in **bear** (z=+0.028); weakest in bull (z=+0.021); spread=0.007 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.001
- STEADY = +0.036
- VOLATILE = +0.000
- DEGEN = +0.073

DNA story: strongest in **DEGEN** (z=+0.073); weakest in VOLATILE (z=+0.000); cross-bucket spread = 0.072 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.021
- F2 (2023-11-01 .. 2024-02-29) = +0.113
- F3 (2024-03-01 .. 2024-05-15) = -0.034
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.42. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_short_spike`).

### Emergent story

`liq_short_spike` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.02 -> t+1=+0.26, while bottom-25% movers sit at t-1=+0.06 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.030. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-unstable (F1=-0.02, F2=+0.11, F3=-0.03). Distributionally mean=+0.0628 std=0.2426 skew=+3.61 kurt=+11.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_imbalance_l5`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +1.350575
- std  = 0.230225
- p10 = +1.0975
- p50 = +1.3189
- p90 = +1.6439
- skew = +1.183
- excess kurtosis = +7.016

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.038 |
| t-3 | indicator 3 days BEFORE event | -0.030 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.018 |
| t-0 | indicator on event day (concurrent) | -0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.203 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.048**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.030**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.002
- chop (label=1) = -0.052
- bear (label=0) = +0.053

Raw regime keys (column-as-found):
  - `label_2`: +0.002
  - `label_0`: +0.053
  - `label_1`: -0.052

Regime story: strongest absolute deviation in **bear** (z=+0.053); weakest in bull (z=+0.002); spread=0.051 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.058
- STEADY = -0.005
- VOLATILE = -0.005
- DEGEN = +0.003

DNA story: strongest in **BLUE** (z=+0.058); weakest in DEGEN (z=+0.003); cross-bucket spread = 0.055 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.186
- F2 (2023-11-01 .. 2024-02-29) = +0.231
- F3 (2024-03-01 .. 2024-05-15) = +0.410
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.65. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**4 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_imbalance_l5`).

### Emergent story

`bd_imbalance_l5` is a low-signal background: top-25%-mover z-lift goes t-3=-0.03 -> t-1=-0.02 -> t-0=-0.00 -> t+1=-0.20, while bottom-25% movers sit at t-1=-0.05 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.030. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in BLUE bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.19, F2=+0.23, F3=+0.41). Distributionally mean=+1.3506 std=0.2302 skew=+1.18 kurt=+7.02. used by 4 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_momentum_accel`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.008611
- std  = 0.987148
- p10 = -1.2344
- p50 = -0.0016
- p90 = +1.2670
- skew = +0.046
- excess kurtosis = +0.327

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.004 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.015 |
| t-0 | indicator on event day (concurrent) | -0.007 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.012 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.015**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.030**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.005
- chop (label=1) = -0.026
- bear (label=0) = -0.003

Raw regime keys (column-as-found):
  - `label_1`: -0.026
  - `label_2`: +0.005
  - `label_0`: -0.003

Regime story: strongest absolute deviation in **chop** (z=-0.026); weakest in bear (z=-0.003); spread=0.024 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.007
- STEADY = -0.003
- VOLATILE = -0.010
- DEGEN = -0.012

DNA story: strongest in **DEGEN** (z=-0.012); weakest in STEADY (z=-0.003); cross-bucket spread = 0.009 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.052
- F2 (2023-11-01 .. 2024-02-29) = -0.050
- F3 (2024-03-01 .. 2024-05-15) = -0.011
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 14.56. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_momentum_accel`).

### Emergent story

`norm_momentum_accel` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.01 -> t-0=-0.01 -> t+1=-0.01, while bottom-25% movers sit at t-1=+0.02 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.030. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in DEGEN bucket (z=-0.01). Fold consistency: fold-unstable (F1=+0.05, F2=-0.05, F3=-0.01). Distributionally mean=+0.0086 std=0.9871 skew=+0.05 kurt=+0.33. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `stbl_total_delta_30d_pct`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.058061
- std  = 0.109885
- p10 = -0.0245
- p50 = +0.0269
- p90 = +0.2258
- skew = +1.763
- excess kurtosis = +3.699

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | +0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.009 |
| t-0 | indicator on event day (concurrent) | +0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.009 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.038**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.029**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.067
- chop (label=1) = -0.089
- bear (label=0) = +0.035

Raw regime keys (column-as-found):
  - `label_0`: +0.035
  - `label_1`: -0.089
  - `label_2`: +0.067

Regime story: strongest absolute deviation in **chop** (z=-0.089); weakest in bear (z=+0.035); spread=0.124 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.017
- STEADY = +0.008
- VOLATILE = +0.013
- DEGEN = +0.008

DNA story: strongest in **BLUE** (z=-0.017); weakest in DEGEN (z=+0.008); cross-bucket spread = 0.026 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.632
- F2 (2023-11-01 .. 2024-02-29) = -0.143
- F3 (2024-03-01 .. 2024-05-15) = +0.472
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.47. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_total_delta_30d_pct`).

### Emergent story

`stbl_total_delta_30d_pct` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.029. Cross-regime: strongest in chop (z=-0.09). Cross-DNA: most pronounced in BLUE bucket (z=-0.02). Fold consistency: fold-unstable (F1=-0.63, F2=-0.14, F3=+0.47). Distributionally mean=+0.0581 std=0.1099 skew=+1.76 kurt=+3.70. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `rv_jump_signed_var`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.000038
- std  = 0.002960
- p10 = -0.0004
- p50 = +0.0000
- p90 = +0.0004
- skew = +16.608
- excess kurtosis = +1401.228

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.006 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.001 |
| t-0 | indicator on event day (concurrent) | +0.006 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.158 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.027**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.029**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.032
- chop (label=1) = -0.019
- bear (label=0) = -0.005

Raw regime keys (column-as-found):
  - `label_1`: -0.019
  - `label_0`: -0.005
  - `label_2`: +0.032

Regime story: strongest absolute deviation in **bull** (z=+0.032); weakest in bear (z=-0.005); spread=0.038 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.016
- STEADY = +0.023
- VOLATILE = +0.008
- DEGEN = -0.026

DNA story: strongest in **DEGEN** (z=-0.026); weakest in VOLATILE (z=+0.008); cross-bucket spread = 0.034 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.009
- F2 (2023-11-01 .. 2024-02-29) = -0.050
- F3 (2024-03-01 .. 2024-05-15) = -0.038
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.53. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_jump_signed_var`).

### Emergent story

`rv_jump_signed_var` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=+0.01 -> t+1=+0.16, while bottom-25% movers sit at t-1=+0.03 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.029. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-stable (F1=-0.01, F2=-0.05, F3=-0.04). Distributionally mean=-0.0000 std=0.0030 skew=+16.61 kurt=+1401.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bd_n_snapshots`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +2835.560322
- std  = 283.859526
- p10 = +2853.0000
- p50 = +2880.0000
- p90 = +2880.0000
- skew = -8.620
- excess kurtosis = +75.828

Shape: left-skewed (downside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.005 |
| t-3 | indicator 3 days BEFORE event | -0.018 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.009 |
| t-0 | indicator on event day (concurrent) | +0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.010 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.037**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.029**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.054
- chop (label=1) = -0.064
- bear (label=0) = +0.009

Raw regime keys (column-as-found):
  - `label_0`: +0.009
  - `label_1`: -0.064
  - `label_2`: +0.054

Regime story: strongest absolute deviation in **chop** (z=-0.064); weakest in bear (z=+0.009); spread=0.073 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.035
- STEADY = +0.005
- VOLATILE = +0.002
- DEGEN = +0.004

DNA story: strongest in **BLUE** (z=+0.035); weakest in VOLATILE (z=+0.002); cross-bucket spread = 0.033 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.128
- F2 (2023-11-01 .. 2024-02-29) = +0.112
- F3 (2024-03-01 .. 2024-05-15) = +0.107
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.08. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_n_snapshots`).

### Emergent story

`bd_n_snapshots` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=+0.00 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.029. Cross-regime: strongest in chop (z=-0.06). Cross-DNA: most pronounced in BLUE bucket (z=+0.04). Fold consistency: fold-stable (F1=+0.13, F2=+0.11, F3=+0.11). Distributionally mean=+2835.5603 std=283.8595 skew=-8.62 kurt=+75.83. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_hawkes_sell_intensity`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.168430
- std  = 0.642860
- p10 = -0.5115
- p50 = -0.3141
- p90 = +0.1851
- skew = +5.196
- excess kurtosis = +33.288

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.008 |
| t-3 | indicator 3 days BEFORE event | -0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.012 |
| t-0 | indicator on event day (concurrent) | +0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.007 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.016**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.028**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.027
- chop (label=1) = -0.010
- bear (label=0) = +0.006

Raw regime keys (column-as-found):
  - `label_0`: +0.006
  - `label_1`: -0.010
  - `label_2`: +0.027

Regime story: strongest absolute deviation in **bull** (z=+0.027); weakest in bear (z=+0.006); spread=0.021 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.030
- STEADY = +0.013
- VOLATILE = +0.006
- DEGEN = +0.004

DNA story: strongest in **BLUE** (z=+0.030); weakest in DEGEN (z=+0.004); cross-bucket spread = 0.026 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.067
- F2 (2023-11-01 .. 2024-02-29) = +0.010
- F3 (2024-03-01 .. 2024-05-15) = +0.040
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 7.71. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_hawkes_sell_intensity`).

### Emergent story

`norm_hawkes_sell_intensity` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=-0.02 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.028. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.07, F2=+0.01, F3=+0.04). Distributionally mean=-0.1684 std=0.6429 skew=+5.20 kurt=+33.29. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `rv_jv_5m`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.000404
- std  = 0.001788
- p10 = +0.0000
- p50 = +0.0001
- p90 = +0.0008
- skew = +48.581
- excess kurtosis = +4357.898

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.013 |
| t-3 | indicator 3 days BEFORE event | +0.011 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.015 |
| t-0 | indicator on event day (concurrent) | +0.012 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.106 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.043**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.028**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.004
- chop (label=1) = +0.010
- bear (label=0) = +0.025

Raw regime keys (column-as-found):
  - `label_1`: +0.010
  - `label_0`: +0.025
  - `label_2`: +0.004

Regime story: strongest absolute deviation in **bear** (z=+0.025); weakest in bull (z=+0.004); spread=0.021 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.006
- STEADY = +0.009
- VOLATILE = +0.015
- DEGEN = +0.010

DNA story: strongest in **VOLATILE** (z=+0.015); weakest in BLUE (z=+0.006); cross-bucket spread = 0.008 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.139
- F2 (2023-11-01 .. 2024-02-29) = -0.069
- F3 (2024-03-01 .. 2024-05-15) = -0.119
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.27. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_jv_5m`).

### Emergent story

`rv_jv_5m` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.02 -> t-0=+0.01 -> t+1=+0.11, while bottom-25% movers sit at t-1=+0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.028. Cross-regime: strongest in bear (z=+0.02). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.01). Fold consistency: fold-stable (F1=-0.14, F2=-0.07, F3=-0.12). Distributionally mean=+0.0004 std=0.0018 skew=+48.58 kurt=+4357.90. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xex_cb_bn_spread_bps`

**Family**: `xex_*` — cross-exchange spread (binance vs okx / bybit, bps)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 680
- mean = +2.147351
- std  = 16.276010
- p10 = -6.3529
- p50 = +1.0980
- p90 = +9.3341
- skew = +19.389
- excess kurtosis = +453.286

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.067 |
| t-3 | indicator 3 days BEFORE event | -0.023 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.046 |
| t-0 | indicator on event day (concurrent) | -0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.033 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.019**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.027**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.143
- chop (label=1) = -0.093
- bear (label=0) = -0.142

Raw regime keys (column-as-found):
  - `label_2`: +0.143
  - `label_1`: -0.093
  - `label_0`: -0.142

Regime story: strongest absolute deviation in **bull** (z=+0.143); weakest in chop (z=-0.093); spread=0.236 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.107
- STEADY = -0.085
- VOLATILE = -0.083
- DEGEN = +nan

DNA story: strongest in **BLUE** (z=+0.107); weakest in VOLATILE (z=-0.083); cross-bucket spread = 0.191 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = -0.061
- F3 (2024-03-01 .. 2024-05-15) = +0.032
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.26. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xex_cb_bn_spread_bps`).

### Emergent story

`xex_cb_bn_spread_bps` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=+0.05 -> t-0=-0.00 -> t+1=+0.03, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.027. Cross-regime: strongest in bull (z=+0.14). Cross-DNA: most pronounced in BLUE bucket (z=+0.11). Fold consistency: fold-unstable (F1=+nan, F2=-0.06, F3=+0.03). Distributionally mean=+2.1474 std=16.2760 skew=+19.39 kurt=+453.29. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_top_pos_lsr_z`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 26587
- mean = +0.330669
- std  = 1.502849
- p10 = -1.5363
- p50 = +0.2441
- p90 = +2.3055
- skew = +0.533
- excess kurtosis = +5.867

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.005 |
| t-3 | indicator 3 days BEFORE event | +0.009 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.000 |
| t-0 | indicator on event day (concurrent) | -0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.023 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.024**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.025**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.021
- chop (label=1) = -0.027
- bear (label=0) = +0.042

Raw regime keys (column-as-found):
  - `label_1`: -0.027
  - `label_0`: +0.042
  - `label_2`: -0.021

Regime story: strongest absolute deviation in **bear** (z=+0.042); weakest in bull (z=-0.021); spread=0.063 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.088
- STEADY = -0.012
- VOLATILE = -0.001
- DEGEN = +0.031

DNA story: strongest in **BLUE** (z=-0.088); weakest in VOLATILE (z=-0.001); cross-bucket spread = 0.087 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.043
- F2 (2023-11-01 .. 2024-02-29) = +0.159
- F3 (2024-03-01 .. 2024-05-15) = -0.261
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.54. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_top_pos_lsr_z`).

### Emergent story

`s3_top_pos_lsr_z` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.02, while bottom-25% movers sit at t-1=+0.02 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.025. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=-0.09). Fold consistency: fold-unstable (F1=-0.04, F2=+0.16, F3=-0.26). Distributionally mean=+0.3307 std=1.5028 skew=+0.53 kurt=+5.87. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_flow_imbalance`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.014509
- std  = 0.867285
- p10 = -1.1056
- p50 = +0.0096
- p90 = +1.1490
- skew = +0.017
- excess kurtosis = +0.021

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.021 |
| t-0 | indicator on event day (concurrent) | -0.013 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.007 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.025**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.010
- chop (label=1) = -0.014
- bear (label=0) = -0.046

Raw regime keys (column-as-found):
  - `label_2`: +0.010
  - `label_1`: -0.014
  - `label_0`: -0.046

Regime story: strongest absolute deviation in **bear** (z=-0.046); weakest in bull (z=+0.010); spread=0.056 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.036
- STEADY = -0.024
- VOLATILE = -0.004
- DEGEN = -0.034

DNA story: strongest in **BLUE** (z=+0.036); weakest in VOLATILE (z=-0.004); cross-bucket spread = 0.039 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.019
- F2 (2023-11-01 .. 2024-02-29) = +0.004
- F3 (2024-03-01 .. 2024-05-15) = -0.075
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.37. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**12 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_flow_imbalance`).

### Emergent story

`norm_flow_imbalance` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.02 -> t-0=-0.01 -> t+1=-0.01, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.025. Cross-regime: strongest in bear (z=-0.05). Cross-DNA: most pronounced in BLUE bucket (z=+0.04). Fold consistency: fold-unstable (F1=+0.02, F2=+0.00, F3=-0.07). Distributionally mean=+0.0145 std=0.8673 skew=+0.02 kurt=+0.02. used by 12 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `liq_long_spike`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.061138
- std  = 0.239583
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +3.664
- excess kurtosis = +11.422

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | +0.022 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.030 |
| t-0 | indicator on event day (concurrent) | +0.029 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.179 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.054**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.025**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.017
- chop (label=1) = +0.013
- bear (label=0) = +0.066

Raw regime keys (column-as-found):
  - `label_1`: +0.013
  - `label_2`: +0.017
  - `label_0`: +0.066

Regime story: strongest absolute deviation in **bear** (z=+0.066); weakest in chop (z=+0.013); spread=0.053 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.023
- STEADY = +0.023
- VOLATILE = +0.027
- DEGEN = +0.071

DNA story: strongest in **DEGEN** (z=+0.071); weakest in BLUE (z=-0.023); cross-bucket spread = 0.094 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.024
- F2 (2023-11-01 .. 2024-02-29) = +0.132
- F3 (2024-03-01 .. 2024-05-15) = -0.001
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.93. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_long_spike`).

### Emergent story

`liq_long_spike` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.03 -> t+1=+0.18, while bottom-25% movers sit at t-1=+0.05 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.025. Cross-regime: strongest in bear (z=+0.07). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-unstable (F1=-0.02, F2=+0.13, F3=-0.00). Distributionally mean=+0.0611 std=0.2396 skew=+3.66 kurt=+11.42. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_liq_long_usd_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +0.114797
- std  = 0.318776
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.417
- excess kurtosis = +3.841

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | +0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.011 |
| t-0 | indicator on event day (concurrent) | +0.020 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.052 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.035**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.024**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.018
- chop (label=1) = +0.007
- bear (label=0) = +0.020

Raw regime keys (column-as-found):
  - `label_1`: +0.007
  - `label_0`: +0.020
  - `label_2`: +0.018

Regime story: strongest absolute deviation in **bear** (z=+0.020); weakest in chop (z=+0.007); spread=0.013 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.008
- STEADY = +0.038
- VOLATILE = +0.006
- DEGEN = +0.020

DNA story: strongest in **STEADY** (z=+0.038); weakest in VOLATILE (z=+0.006); cross-bucket spread = 0.032 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.058
- F2 (2023-11-01 .. 2024-02-29) = +0.065
- F3 (2024-03-01 .. 2024-05-15) = +0.114
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.32. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_liq_long_usd_xpct10`).

### Emergent story

`xrel_liq_long_usd_xpct10` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=+0.02 -> t+1=+0.05, while bottom-25% movers sit at t-1=+0.04 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.024. Cross-regime: strongest in bear (z=+0.02). Cross-DNA: most pronounced in STEADY bucket (z=+0.04). Fold consistency: fold-stable (F1=+0.06, F2=+0.07, F3=+0.11). Distributionally mean=+0.1148 std=0.3188 skew=+2.42 kurt=+3.84. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `rv_jump_frac`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.084016
- std  = 0.085283
- p10 = +0.0000
- p50 = +0.0646
- p90 = +0.1961
- skew = +1.567
- excess kurtosis = +4.100

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.013 |
| t-3 | indicator 3 days BEFORE event | +0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.020 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.019**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.024**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.027
- chop (label=1) = +0.042
- bear (label=0) = -0.010

Raw regime keys (column-as-found):
  - `label_1`: +0.042
  - `label_2`: -0.027
  - `label_0`: -0.010

Regime story: strongest absolute deviation in **chop** (z=+0.042); weakest in bear (z=-0.010); spread=0.052 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.047
- STEADY = -0.002
- VOLATILE = -0.005
- DEGEN = +0.002

DNA story: strongest in **BLUE** (z=+0.047); weakest in STEADY (z=-0.002); cross-bucket spread = 0.050 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.264
- F2 (2023-11-01 .. 2024-02-29) = -0.193
- F3 (2024-03-01 .. 2024-05-15) = -0.358
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.74. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**7 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_jump_frac`).

### Emergent story

`rv_jump_frac` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=-0.02, while bottom-25% movers sit at t-1=-0.02 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.024. Cross-regime: strongest in chop (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-unstable (F1=+0.26, F2=-0.19, F3=-0.36). Distributionally mean=+0.0840 std=0.0853 skew=+1.57 kurt=+4.10. used by 7 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_return_4`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.030332
- std  = 0.931908
- p10 = -1.1485
- p50 = +0.0304
- p90 = +1.2098
- skew = -0.007
- excess kurtosis = +0.306

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.010 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.013 |
| t-0 | indicator on event day (concurrent) | -0.011 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.015 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.008**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.021**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.100
- chop (label=1) = -0.021
- bear (label=0) = -0.161

Raw regime keys (column-as-found):
  - `label_1`: -0.021
  - `label_2`: +0.100
  - `label_0`: -0.161

Regime story: strongest absolute deviation in **bear** (z=-0.161); weakest in chop (z=-0.021); spread=0.140 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.009
- STEADY = -0.009
- VOLATILE = -0.010
- DEGEN = -0.022

DNA story: strongest in **DEGEN** (z=-0.022); weakest in BLUE (z=+0.009); cross-bucket spread = 0.031 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.110
- F2 (2023-11-01 .. 2024-02-29) = -0.034
- F3 (2024-03-01 .. 2024-05-15) = -0.085
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 26.49. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_return_4`).

### Emergent story

`norm_return_4` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.01 -> t+1=-0.02, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.021. Cross-regime: strongest in bear (z=-0.16). Cross-DNA: most pronounced in DEGEN bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.11, F2=-0.03, F3=-0.09). Distributionally mean=+0.0303 std=0.9319 skew=-0.01 kurt=+0.31. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `stbl_total_delta_7d_pct`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.012221
- std  = 0.031431
- p10 = -0.0064
- p50 = +0.0045
- p90 = +0.0399
- skew = +4.259
- excess kurtosis = +44.114

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.010 |
| t-3 | indicator 3 days BEFORE event | +0.011 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.010 |
| t-0 | indicator on event day (concurrent) | +0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.009 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.029**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.020**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.024
- chop (label=1) = -0.061
- bear (label=0) = +0.062

Raw regime keys (column-as-found):
  - `label_1`: -0.061
  - `label_0`: +0.062
  - `label_2`: +0.024

Regime story: strongest absolute deviation in **bear** (z=+0.062); weakest in bull (z=+0.024); spread=0.038 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.014
- STEADY = +0.021
- VOLATILE = -0.002
- DEGEN = +0.013

DNA story: strongest in **STEADY** (z=+0.021); weakest in VOLATILE (z=-0.002); cross-bucket spread = 0.023 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.469
- F2 (2023-11-01 .. 2024-02-29) = -0.033
- F3 (2024-03-01 .. 2024-05-15) = +0.294
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.52. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_total_delta_7d_pct`).

### Emergent story

`stbl_total_delta_7d_pct` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.03 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.020. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in STEADY bucket (z=+0.02). Fold consistency: fold-unstable (F1=-0.47, F2=-0.03, F3=+0.29). Distributionally mean=+0.0122 std=0.0314 skew=+4.26 kurt=+44.11. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `rv_jump_count`

**Family**: `rv_*` — realized-variance / bipower / jump components on intraday returns

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.646281
- std  = 0.824897
- p10 = +0.0000
- p50 = +0.0000
- p90 = +2.0000
- skew = +1.263
- excess kurtosis = +1.481

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | -0.005 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.006 |
| t-0 | indicator on event day (concurrent) | -0.011 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.083 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.013**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.019**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.037
- chop (label=1) = +0.041
- bear (label=0) = -0.032

Raw regime keys (column-as-found):
  - `label_0`: -0.032
  - `label_1`: +0.041
  - `label_2`: -0.037

Regime story: strongest absolute deviation in **chop** (z=+0.041); weakest in bear (z=-0.032); spread=0.072 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.027
- STEADY = -0.014
- VOLATILE = -0.024
- DEGEN = +0.020

DNA story: strongest in **BLUE** (z=+0.027); weakest in STEADY (z=-0.014); cross-bucket spread = 0.041 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.037
- F2 (2023-11-01 .. 2024-02-29) = -0.160
- F3 (2024-03-01 .. 2024-05-15) = -0.199
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.96. (Magnitudes are tight - stable signal.)

### Catalog usage

**4 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/rv_jump_count`).

### Emergent story

`rv_jump_count` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.01 -> t+1=+0.08, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.019. Cross-regime: strongest in chop (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-unstable (F1=+0.04, F2=-0.16, F3=-0.20). Distributionally mean=+0.6463 std=0.8249 skew=+1.26 kurt=+1.48. used by 4 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `fp_fund_panel`

**Family**: `fp_*` — funding-payment-implied (fp_*) measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 42656
- mean = +0.000123
- std  = 0.000412
- p10 = -0.0001
- p50 = +0.0001
- p90 = +0.0004
- skew = -3.188
- excess kurtosis = +124.737

Shape: left-skewed (downside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.010 |
| t-3 | indicator 3 days BEFORE event | -0.021 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.024 |
| t-0 | indicator on event day (concurrent) | -0.024 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.039 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.005**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.019**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.013
- chop (label=1) = -0.090
- bear (label=0) = +0.036

Raw regime keys (column-as-found):
  - `label_2`: -0.013
  - `label_1`: -0.090
  - `label_0`: +0.036

Regime story: strongest absolute deviation in **chop** (z=-0.090); weakest in bull (z=-0.013); spread=0.077 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.024
- STEADY = -0.013
- VOLATILE = -0.044
- DEGEN = -0.011

DNA story: strongest in **VOLATILE** (z=-0.044); weakest in DEGEN (z=-0.011); cross-bucket spread = 0.034 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.325
- F2 (2023-11-01 .. 2024-02-29) = +0.178
- F3 (2024-03-01 .. 2024-05-15) = +0.185
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 18.69. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/fp_fund_panel`).

### Emergent story

`fp_fund_panel` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.02 -> t-0=-0.02 -> t+1=-0.04, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.019. Cross-regime: strongest in chop (z=-0.09). Cross-DNA: most pronounced in VOLATILE bucket (z=-0.04). Fold consistency: fold-unstable (F1=-0.32, F2=+0.18, F3=+0.18). Distributionally mean=+0.0001 std=0.0004 skew=-3.19 kurt=+124.74. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_short_panic`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.045897
- std  = 0.209262
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +4.340
- excess kurtosis = +16.836

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | +0.021 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.033 |
| t-0 | indicator on event day (concurrent) | +0.020 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.227 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.052**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.018**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.021
- chop (label=1) = +0.005
- bear (label=0) = +0.037

Raw regime keys (column-as-found):
  - `label_2`: +0.021
  - `label_1`: +0.005
  - `label_0`: +0.037

Regime story: strongest absolute deviation in **bear** (z=+0.037); weakest in chop (z=+0.005); spread=0.031 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.017
- STEADY = +0.030
- VOLATILE = +0.001
- DEGEN = +0.070

DNA story: strongest in **DEGEN** (z=+0.070); weakest in VOLATILE (z=+0.001); cross-bucket spread = 0.070 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.030
- F2 (2023-11-01 .. 2024-02-29) = +0.073
- F3 (2024-03-01 .. 2024-05-15) = -0.036
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 23.76. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_short_panic`).

### Emergent story

`liq_short_panic` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.02 -> t+1=+0.23, while bottom-25% movers sit at t-1=+0.05 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.018. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.07). Fold consistency: fold-unstable (F1=-0.03, F2=+0.07, F3=-0.04). Distributionally mean=+0.0459 std=0.2093 skew=+4.34 kurt=+16.84. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_oi_price_divergence`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.088996
- std  = 0.724063
- p10 = -0.6758
- p50 = +0.0000
- p90 = +0.9806
- skew = +1.526
- excess kurtosis = +5.380

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.010 |
| t-3 | indicator 3 days BEFORE event | -0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.001 |
| t-0 | indicator on event day (concurrent) | -0.008 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.031 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.019**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.018**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.003
- chop (label=1) = +0.047
- bear (label=0) = -0.088

Raw regime keys (column-as-found):
  - `label_0`: -0.088
  - `label_1`: +0.047
  - `label_2`: +0.003

Regime story: strongest absolute deviation in **bear** (z=-0.088); weakest in bull (z=+0.003); spread=0.091 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.044
- STEADY = +0.001
- VOLATILE = -0.008
- DEGEN = -0.014

DNA story: strongest in **BLUE** (z=-0.044); weakest in STEADY (z=+0.001); cross-bucket spread = 0.045 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.133
- F2 (2023-11-01 .. 2024-02-29) = +0.108
- F3 (2024-03-01 .. 2024-05-15) = +0.097
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.14. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_oi_price_divergence`).

### Emergent story

`norm_oi_price_divergence` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.03, while bottom-25% movers sit at t-1=-0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.018. Cross-regime: strongest in bear (z=-0.09). Cross-DNA: most pronounced in BLUE bucket (z=-0.04). Fold consistency: fold-stable (F1=+0.13, F2=+0.11, F3=+0.10). Distributionally mean=+0.0890 std=0.7241 skew=+1.53 kurt=+5.38. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bs_basis_bear_shock`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = +0.036017
- std  = 0.186333
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +4.980
- excess kurtosis = +22.802

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.005 |
| t-3 | indicator 3 days BEFORE event | -0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.009 |
| t-0 | indicator on event day (concurrent) | +0.029 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.006 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.009**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.018**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.001
- chop (label=1) = +0.025
- bear (label=0) = +0.074

Raw regime keys (column-as-found):
  - `label_1`: +0.025
  - `label_0`: +0.074
  - `label_2`: +0.001

Regime story: strongest absolute deviation in **bear** (z=+0.074); weakest in bull (z=+0.001); spread=0.073 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.027
- STEADY = +0.027
- VOLATILE = +0.037
- DEGEN = +0.036

DNA story: strongest in **VOLATILE** (z=+0.037); weakest in STEADY (z=+0.027); cross-bucket spread = 0.010 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.051
- F2 (2023-11-01 .. 2024-02-29) = +0.019
- F3 (2024-03-01 .. 2024-05-15) = -0.011
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.30. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_bear_shock`).

### Emergent story

`bs_basis_bear_shock` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.01 -> t-0=+0.03 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.018. Cross-regime: strongest in bear (z=+0.07). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.04). Fold consistency: fold-unstable (F1=+0.05, F2=+0.02, F3=-0.01). Distributionally mean=+0.0360 std=0.1863 skew=+4.98 kurt=+22.80. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `stbl_usdt_delta_7d_pct`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.012084
- std  = 0.033117
- p10 = -0.0015
- p50 = +0.0045
- p90 = +0.0391
- skew = +7.263
- excess kurtosis = +91.528

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.011 |
| t-3 | indicator 3 days BEFORE event | +0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.007 |
| t-0 | indicator on event day (concurrent) | +0.007 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.007 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.024**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.017**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.021
- chop (label=1) = -0.037
- bear (label=0) = +0.030

Raw regime keys (column-as-found):
  - `label_1`: -0.037
  - `label_2`: +0.021
  - `label_0`: +0.030

Regime story: strongest absolute deviation in **chop** (z=-0.037); weakest in bull (z=+0.021); spread=0.058 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.017
- STEADY = +0.028
- VOLATILE = -0.007
- DEGEN = +0.001

DNA story: strongest in **STEADY** (z=+0.028); weakest in DEGEN (z=+0.001); cross-bucket spread = 0.027 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.410
- F2 (2023-11-01 .. 2024-02-29) = +0.033
- F3 (2024-03-01 .. 2024-05-15) = +0.187
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.98. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_usdt_delta_7d_pct`).

### Emergent story

`stbl_usdt_delta_7d_pct` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.017. Cross-regime: strongest in chop (z=-0.04). Cross-DNA: most pronounced in STEADY bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.41, F2=+0.03, F3=+0.19). Distributionally mean=+0.0121 std=0.0331 skew=+7.26 kurt=+91.53. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_capitulation`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.042767
- std  = 0.202332
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +4.520
- excess kurtosis = +18.427

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | +0.027 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.033 |
| t-0 | indicator on event day (concurrent) | +0.026 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.150 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.050**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.017**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.018
- chop (label=1) = +0.009
- bear (label=0) = +0.058

Raw regime keys (column-as-found):
  - `label_1`: +0.009
  - `label_2`: +0.018
  - `label_0`: +0.058

Regime story: strongest absolute deviation in **bear** (z=+0.058); weakest in chop (z=+0.009); spread=0.048 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.023
- STEADY = +0.010
- VOLATILE = +0.030
- DEGEN = +0.077

DNA story: strongest in **DEGEN** (z=+0.077); weakest in STEADY (z=+0.010); cross-bucket spread = 0.067 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.054
- F2 (2023-11-01 .. 2024-02-29) = +0.097
- F3 (2024-03-01 .. 2024-05-15) = +0.003
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.04. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_capitulation`).

### Emergent story

`liq_capitulation` is a low-signal background: top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.03 -> t-0=+0.03 -> t+1=+0.15, while bottom-25% movers sit at t-1=+0.05 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.017. Cross-regime: strongest in bear (z=+0.06). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=-0.05, F2=+0.10, F3=+0.00). Distributionally mean=+0.0428 std=0.2023 skew=+4.52 kurt=+18.43. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_hawkes_intensity`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.203898
- std  = 0.631544
- p10 = -0.5747
- p50 = -0.3487
- p90 = +0.1735
- skew = +4.947
- excess kurtosis = +31.564

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.006 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.009 |
| t-0 | indicator on event day (concurrent) | +0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.005 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.009**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.017**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.023
- chop (label=1) = -0.014
- bear (label=0) = +0.038

Raw regime keys (column-as-found):
  - `label_2`: +0.023
  - `label_0`: +0.038
  - `label_1`: -0.014

Regime story: strongest absolute deviation in **bear** (z=+0.038); weakest in chop (z=-0.014); spread=0.052 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.055
- STEADY = +0.023
- VOLATILE = +0.008
- DEGEN = +0.009

DNA story: strongest in **BLUE** (z=+0.055); weakest in VOLATILE (z=+0.008); cross-bucket spread = 0.047 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.087
- F2 (2023-11-01 .. 2024-02-29) = +0.001
- F3 (2024-03-01 .. 2024-05-15) = -0.002
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.39. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_hawkes_intensity`).

### Emergent story

`norm_hawkes_intensity` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=+0.02 -> t+1=+0.01, while bottom-25% movers sit at t-1=-0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.017. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-unstable (F1=-0.09, F2=+0.00, F3=-0.00). Distributionally mean=-0.2039 std=0.6315 skew=+4.95 kurt=+31.56. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `s3_smart_bullish`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31685
- mean = +0.164463
- std  = 0.370695
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +1.810
- excess kurtosis = +1.277

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.011 |
| t-3 | indicator 3 days BEFORE event | -0.009 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.020 |
| t-0 | indicator on event day (concurrent) | -0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.035 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.005**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.015**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.010
- chop (label=1) = -0.034
- bear (label=0) = -0.028

Raw regime keys (column-as-found):
  - `label_1`: -0.034
  - `label_0`: -0.028
  - `label_2`: +0.010

Regime story: strongest absolute deviation in **chop** (z=-0.034); weakest in bull (z=+0.010); spread=0.044 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.084
- STEADY = -0.024
- VOLATILE = +0.002
- DEGEN = -0.029

DNA story: strongest in **BLUE** (z=-0.084); weakest in VOLATILE (z=+0.002); cross-bucket spread = 0.086 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.143
- F2 (2023-11-01 .. 2024-02-29) = +0.289
- F3 (2024-03-01 .. 2024-05-15) = +0.112
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.43. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_bullish`).

### Emergent story

`s3_smart_bullish` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.02 -> t-0=-0.02 -> t+1=-0.03, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.015. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=-0.08). Fold consistency: fold-stable (F1=+0.14, F2=+0.29, F3=+0.11). Distributionally mean=+0.1645 std=0.3707 skew=+1.81 kurt=+1.28. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_wh_whale_net_usd_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51779
- mean = -0.019571
- std  = 2.538868
- p10 = -1.2180
- p50 = +0.0000
- p90 = +1.0273
- skew = +0.379
- excess kurtosis = +24.936

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.017 |
| t-3 | indicator 3 days BEFORE event | +0.011 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.229 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.019**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.014**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.032
- chop (label=1) = -0.011
- bear (label=0) = -0.031

Raw regime keys (column-as-found):
  - `label_0`: -0.031
  - `label_2`: +0.032
  - `label_1`: -0.011

Regime story: strongest absolute deviation in **bull** (z=+0.032); weakest in chop (z=-0.011); spread=0.043 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.029
- STEADY = -0.003
- VOLATILE = -0.020
- DEGEN = +0.058

DNA story: strongest in **DEGEN** (z=+0.058); weakest in STEADY (z=-0.003); cross-bucket spread = 0.061 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.017
- F2 (2023-11-01 .. 2024-02-29) = -0.002
- F3 (2024-03-01 .. 2024-05-15) = +0.059
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.40. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_wh_whale_net_usd_xratio`).

### Emergent story

`xrel_wh_whale_net_usd_xratio` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.23, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.014. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.02, F2=-0.00, F3=+0.06). Distributionally mean=-0.0196 std=2.5389 skew=+0.38 kurt=+24.94. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_depth_l1pct_mean`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +35446236.597326
- std  = 181015464.386336
- p10 = +21455.6004
- p50 = +695803.7903
- p90 = +17797546.3969
- skew = +7.450
- excess kurtosis = +61.594

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.026 |
| t-3 | indicator 3 days BEFORE event | +0.022 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.024 |
| t-0 | indicator on event day (concurrent) | +0.030 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.013 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.011**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.013**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.020
- chop (label=1) = +0.115
- bear (label=0) = +0.007

Raw regime keys (column-as-found):
  - `label_2`: -0.020
  - `label_0`: +0.007
  - `label_1`: +0.115

Regime story: strongest absolute deviation in **chop** (z=+0.115); weakest in bear (z=+0.007); spread=0.108 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.114
- STEADY = +0.006
- VOLATILE = +0.031
- DEGEN = +0.042

DNA story: strongest in **BLUE** (z=+0.114); weakest in STEADY (z=+0.006); cross-bucket spread = 0.107 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.824
- F2 (2023-11-01 .. 2024-02-29) = -0.229
- F3 (2024-03-01 .. 2024-05-15) = -0.539
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 31.52. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_depth_l1pct_mean`).

### Emergent story

`bd_depth_l1pct_mean` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.02 -> t-0=+0.03 -> t+1=-0.01, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.013. Cross-regime: strongest in chop (z=+0.11). Cross-DNA: most pronounced in BLUE bucket (z=+0.11). Fold consistency: fold-unstable (F1=+0.82, F2=-0.23, F3=-0.54). Distributionally mean=+35446236.5973 std=181015464.3863 skew=+7.45 kurt=+61.59. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `liq_delta_usd`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = -259564.190060
- std  = 6376652.874899
- p10 = -1364599.2583
- p50 = +0.0000
- p90 = +582828.6073
- skew = +5.183
- excess kurtosis = +588.519

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.011 |
| t-3 | indicator 3 days BEFORE event | +0.012 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.012 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.203 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.025**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.013**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.034
- chop (label=1) = +0.002
- bear (label=0) = -0.032

Raw regime keys (column-as-found):
  - `label_1`: +0.002
  - `label_0`: -0.032
  - `label_2`: +0.034

Regime story: strongest absolute deviation in **bull** (z=+0.034); weakest in chop (z=+0.002); spread=0.032 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.023
- STEADY = +0.021
- VOLATILE = -0.022
- DEGEN = +0.037

DNA story: strongest in **DEGEN** (z=+0.037); weakest in STEADY (z=+0.021); cross-bucket spread = 0.016 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.010
- F2 (2023-11-01 .. 2024-02-29) = -0.010
- F3 (2024-03-01 .. 2024-05-15) = +0.073
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.21. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_delta_usd`).

### Emergent story

`liq_delta_usd` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.20, while bottom-25% movers sit at t-1=+0.03 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.013. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-unstable (F1=-0.01, F2=-0.01, F3=+0.07). Distributionally mean=-259564.1901 std=6376652.8749 skew=+5.18 kurt=+588.52. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_return_1`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.011880
- std  = 0.969440
- p10 = -1.2028
- p50 = +0.0089
- p90 = +1.2277
- skew = +0.011
- excess kurtosis = +0.555

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.007 |
| t-0 | indicator on event day (concurrent) | -0.014 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.003 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.005**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.013**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.052
- chop (label=1) = -0.032
- bear (label=0) = -0.089

Raw regime keys (column-as-found):
  - `label_2`: +0.052
  - `label_0`: -0.089
  - `label_1`: -0.032

Regime story: strongest absolute deviation in **bear** (z=-0.089); weakest in chop (z=-0.032); spread=0.057 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.017
- STEADY = -0.011
- VOLATILE = -0.016
- DEGEN = -0.023

DNA story: strongest in **DEGEN** (z=-0.023); weakest in STEADY (z=-0.011); cross-bucket spread = 0.012 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.051
- F2 (2023-11-01 .. 2024-02-29) = -0.006
- F3 (2024-03-01 .. 2024-05-15) = -0.104
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.24. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_return_1`).

### Emergent story

`norm_return_1` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.01 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.013. Cross-regime: strongest in bear (z=-0.09). Cross-DNA: most pronounced in DEGEN bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.05, F2=-0.01, F3=-0.10). Distributionally mean=+0.0119 std=0.9694 skew=+0.01 kurt=+0.55. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_log_volume`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.067116
- std  = 0.780085
- p10 = -0.5654
- p50 = +0.0602
- p90 = +0.7679
- skew = -0.609
- excess kurtosis = +11.046

Shape: left-skewed (downside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | -0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.006 |
| t-0 | indicator on event day (concurrent) | +0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.069 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.006**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.012**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.080
- chop (label=1) = -0.000
- bear (label=0) = +0.130

Raw regime keys (column-as-found):
  - `label_1`: -0.000
  - `label_2`: -0.080
  - `label_0`: +0.130

Regime story: strongest absolute deviation in **bear** (z=+0.130); weakest in chop (z=-0.000); spread=0.130 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.038
- STEADY = +0.011
- VOLATILE = +0.011
- DEGEN = -0.016

DNA story: strongest in **BLUE** (z=-0.038); weakest in STEADY (z=+0.011); cross-bucket spread = 0.049 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.045
- F2 (2023-11-01 .. 2024-02-29) = -0.027
- F3 (2024-03-01 .. 2024-05-15) = -0.042
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.75. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_log_volume`).

### Emergent story

`norm_log_volume` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.01 -> t-0=+0.00 -> t+1=-0.07, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.012. Cross-regime: strongest in bear (z=+0.13). Cross-DNA: most pronounced in BLUE bucket (z=-0.04). Fold consistency: fold-unstable (F1=+0.05, F2=-0.03, F3=-0.04). Distributionally mean=+0.0671 std=0.7801 skew=-0.61 kurt=+11.05. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `hbr_eta_imbalance`

**Family**: `hbr_*` — Hawkes branching-ratio / self-exciting trade-process intensity

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.003134
- std  = 0.043569
- p10 = -0.0438
- p50 = -0.0033
- p90 = +0.0366
- skew = +0.728
- excess kurtosis = +13.145

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.011 |
| t-0 | indicator on event day (concurrent) | -0.017 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.042 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.012**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.010
- chop (label=1) = -0.035
- bear (label=0) = -0.005

Raw regime keys (column-as-found):
  - `label_0`: -0.005
  - `label_1`: -0.035
  - `label_2`: -0.010

Regime story: strongest absolute deviation in **chop** (z=-0.035); weakest in bear (z=-0.005); spread=0.030 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.013
- STEADY = +0.001
- VOLATILE = -0.029
- DEGEN = -0.015

DNA story: strongest in **VOLATILE** (z=-0.029); weakest in STEADY (z=+0.001); cross-bucket spread = 0.030 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.140
- F2 (2023-11-01 .. 2024-02-29) = +0.027
- F3 (2024-03-01 .. 2024-05-15) = +0.075
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 7.24. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/hbr_eta_imbalance`).

### Emergent story

`hbr_eta_imbalance` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.01 -> t-0=-0.02 -> t+1=-0.04, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.012. Cross-regime: strongest in chop (z=-0.04). Cross-DNA: most pronounced in VOLATILE bucket (z=-0.03). Fold consistency: fold-unstable (F1=-0.14, F2=+0.03, F3=+0.07). Distributionally mean=-0.0031 std=0.0436 skew=+0.73 kurt=+13.14. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_yz_volatility`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.051574
- std  = 1.081346
- p10 = -1.2273
- p50 = -0.0675
- p90 = +1.4915
- skew = +0.472
- excess kurtosis = +0.528

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.009 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.003**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.012**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.109
- chop (label=1) = -0.010
- bear (label=0) = +0.173

Raw regime keys (column-as-found):
  - `label_0`: +0.173
  - `label_1`: -0.010
  - `label_2`: -0.109

Regime story: strongest absolute deviation in **bear** (z=+0.173); weakest in chop (z=-0.010); spread=0.183 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.016
- STEADY = -0.004
- VOLATILE = +0.017
- DEGEN = -0.026

DNA story: strongest in **DEGEN** (z=-0.026); weakest in STEADY (z=-0.004); cross-bucket spread = 0.022 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.002
- F2 (2023-11-01 .. 2024-02-29) = -0.043
- F3 (2024-03-01 .. 2024-05-15) = -0.237
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 1.09. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_yz_volatility`).

### Emergent story

`norm_yz_volatility` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.01 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.012. Cross-regime: strongest in bear (z=+0.17). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-stable (F1=-0.00, F2=-0.04, F3=-0.24). Distributionally mean=+0.0516 std=1.0813 skew=+0.47 kurt=+0.53. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_delta_1d`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51367
- mean = +0.000108
- std  = 0.295631
- p10 = -0.2690
- p50 = +0.0000
- p90 = +0.2751
- skew = -0.257
- excess kurtosis = +24.238

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.004 |
| t-3 | indicator 3 days BEFORE event | -0.009 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | -0.034 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.072 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.006**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.011**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.022
- chop (label=1) = -0.084
- bear (label=0) = +0.004

Raw regime keys (column-as-found):
  - `label_1`: -0.084
  - `label_2`: -0.022
  - `label_0`: +0.004

Regime story: strongest absolute deviation in **chop** (z=-0.084); weakest in bear (z=+0.004); spread=0.087 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.048
- STEADY = -0.041
- VOLATILE = -0.029
- DEGEN = -0.029

DNA story: strongest in **BLUE** (z=-0.048); weakest in DEGEN (z=-0.029); cross-bucket spread = 0.019 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.056
- F2 (2023-11-01 .. 2024-02-29) = -0.032
- F3 (2024-03-01 .. 2024-05-15) = -0.002
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.74. (Magnitudes are tight - stable signal.)

### Catalog usage

**9 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_delta_1d`).

### Emergent story

`bs_basis_delta_1d` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.03 -> t+1=+0.07, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.011. Cross-regime: strongest in chop (z=-0.08). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-stable (F1=-0.06, F2=-0.03, F3=-0.00). Distributionally mean=+0.0001 std=0.2956 skew=-0.26 kurt=+24.24. used by 9 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `liq_delta_z30`

**Family**: `liq_*` — liquidation flow (long/short USD liquidated, magnitude / direction)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51441
- mean = +0.023767
- std  = 1.105969
- p10 = -0.8301
- p50 = +0.0000
- p90 = +0.9347
- skew = +0.580
- excess kurtosis = +7.884

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | +0.009 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.006 |
| t-0 | indicator on event day (concurrent) | +0.011 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.215 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.017**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.011**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.044
- chop (label=1) = +0.006
- bear (label=0) = -0.032

Raw regime keys (column-as-found):
  - `label_1`: +0.006
  - `label_0`: -0.032
  - `label_2`: +0.044

Regime story: strongest absolute deviation in **bull** (z=+0.044); weakest in chop (z=+0.006); spread=0.038 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.038
- STEADY = +0.022
- VOLATILE = -0.015
- DEGEN = +0.042

DNA story: strongest in **DEGEN** (z=+0.042); weakest in VOLATILE (z=-0.015); cross-bucket spread = 0.057 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.007
- F2 (2023-11-01 .. 2024-02-29) = +0.005
- F3 (2024-03-01 .. 2024-05-15) = +0.018
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.86. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/liq_delta_z30`).

### Emergent story

`liq_delta_z30` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=+0.01 -> t+1=+0.21, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.011. Cross-regime: strongest in bull (z=+0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-unstable (F1=-0.01, F2=+0.00, F3=+0.02). Distributionally mean=+0.0238 std=1.1060 skew=+0.58 kurt=+7.88. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_btc_etf_total_7d_z`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 5918
- mean = -0.047939
- std  = 1.409813
- p10 = -1.3278
- p50 = -0.2848
- p90 = +1.5058
- skew = -0.140
- excess kurtosis = +0.048

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.034 |
| t-3 | indicator 3 days BEFORE event | -0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.002 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.005 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.009**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.093
- chop (label=1) = -0.156
- bear (label=0) = +0.014

Raw regime keys (column-as-found):
  - `label_0`: +0.014
  - `label_2`: +0.093
  - `label_1`: -0.156

Regime story: strongest absolute deviation in **chop** (z=-0.156); weakest in bear (z=+0.014); spread=0.170 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.057
- STEADY = -0.057
- VOLATILE = -0.012
- DEGEN = +0.083

DNA story: strongest in **DEGEN** (z=+0.083); weakest in VOLATILE (z=-0.012); cross-bucket spread = 0.095 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.785
- F3 (2024-03-01 .. 2024-05-15) = -0.314
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.33. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_total_7d_z`).

### Emergent story

`etf_btc_etf_total_7d_z` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.010. Cross-regime: strongest in chop (z=-0.16). Cross-DNA: most pronounced in DEGEN bucket (z=+0.08). Fold consistency: fold-unstable (F1=+nan, F2=+0.79, F3=-0.31). Distributionally mean=-0.0479 std=1.4098 skew=-0.14 kurt=+0.05. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_panic`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = +0.016336
- std  = 0.126764
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +7.631
- excess kurtosis = +56.231

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | -0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.004 |
| t-0 | indicator on event day (concurrent) | +0.020 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.011 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.007**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.010
- chop (label=1) = +0.016
- bear (label=0) = +0.041

Raw regime keys (column-as-found):
  - `label_2`: +0.010
  - `label_1`: +0.016
  - `label_0`: +0.041

Regime story: strongest absolute deviation in **bear** (z=+0.041); weakest in bull (z=+0.010); spread=0.031 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.007
- STEADY = -0.008
- VOLATILE = +0.058
- DEGEN = -0.009

DNA story: strongest in **VOLATILE** (z=+0.058); weakest in BLUE (z=+0.007); cross-bucket spread = 0.050 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.010
- F2 (2023-11-01 .. 2024-02-29) = -0.045
- F3 (2024-03-01 .. 2024-05-15) = -0.083
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.65. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_panic`).

### Emergent story

`bs_basis_panic` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=+0.00 -> t-0=+0.02 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.010. Cross-regime: strongest in bear (z=+0.04). Cross-DNA: most pronounced in VOLATILE bucket (z=+0.06). Fold consistency: fold-stable (F1=-0.01, F2=-0.04, F3=-0.08). Distributionally mean=+0.0163 std=0.1268 skew=+7.63 kurt=+56.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `xd_momentum_rank`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.009926
- std  = 1.018645
- p10 = -1.1607
- p50 = +0.4931
- p90 = +1.1606
- skew = -0.025
- excess kurtosis = -1.864

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.011 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | -0.007 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.007 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.006**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.029
- chop (label=1) = -0.016
- bear (label=0) = -0.048

Raw regime keys (column-as-found):
  - `label_2`: +0.029
  - `label_0`: -0.048
  - `label_1`: -0.016

Regime story: strongest absolute deviation in **bear** (z=-0.048); weakest in chop (z=-0.016); spread=0.032 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.016
- STEADY = -0.001
- VOLATILE = -0.010
- DEGEN = -0.017

DNA story: strongest in **DEGEN** (z=-0.017); weakest in STEADY (z=-0.001); cross-bucket spread = 0.016 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.034
- F2 (2023-11-01 .. 2024-02-29) = -0.019
- F3 (2024-03-01 .. 2024-05-15) = -0.079
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.16. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**18 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_momentum_rank`).

### Emergent story

`xd_momentum_rank` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.01, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.010. Cross-regime: strongest in bear (z=-0.05). Cross-DNA: most pronounced in DEGEN bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.03, F2=-0.02, F3=-0.08). Distributionally mean=+0.0099 std=1.0186 skew=-0.02 kurt=-1.86. used by 18 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `xd_btc_return`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.023563
- std  = 0.904096
- p10 = -1.1529
- p50 = +0.0000
- p90 = +1.1010
- skew = +0.021
- excess kurtosis = +0.681

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.006 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | -0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.006 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.005**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.013
- chop (label=1) = -0.004
- bear (label=0) = -0.013

Raw regime keys (column-as-found):
  - `label_0`: -0.013
  - `label_1`: -0.004
  - `label_2`: -0.013

Regime story: strongest absolute deviation in **bear** (z=-0.013); weakest in chop (z=-0.004); spread=0.009 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.018
- STEADY = -0.000
- VOLATILE = -0.008
- DEGEN = -0.044

DNA story: strongest in **DEGEN** (z=-0.044); weakest in STEADY (z=-0.000); cross-bucket spread = 0.044 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.061
- F2 (2023-11-01 .. 2024-02-29) = -0.036
- F3 (2024-03-01 .. 2024-05-15) = -0.050
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 6.14. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**12 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_btc_return`).

### Emergent story

`xd_btc_return` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=-0.01 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.010. Cross-regime: strongest in bear (z=-0.01). Cross-DNA: most pronounced in DEGEN bucket (z=-0.04). Fold consistency: fold-unstable (F1=+0.06, F2=-0.04, F3=-0.05). Distributionally mean=-0.0236 std=0.9041 skew=+0.02 kurt=+0.68. used by 12 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `xd_funding_spread`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.005634
- std  = 0.707897
- p10 = -0.9051
- p50 = +0.0000
- p90 = +0.9037
- skew = -0.013
- excess kurtosis = +1.842

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.010 |
| t-3 | indicator 3 days BEFORE event | -0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.002 |
| t-0 | indicator on event day (concurrent) | -0.007 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.023 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.007**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.022
- chop (label=1) = +0.003
- bear (label=0) = +0.002

Raw regime keys (column-as-found):
  - `label_2`: -0.022
  - `label_1`: +0.003
  - `label_0`: +0.002

Regime story: strongest absolute deviation in **bull** (z=-0.022); weakest in bear (z=+0.002); spread=0.024 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.024
- STEADY = -0.000
- VOLATILE = -0.016
- DEGEN = -0.008

DNA story: strongest in **BLUE** (z=+0.024); weakest in STEADY (z=-0.000); cross-bucket spread = 0.025 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.053
- F2 (2023-11-01 .. 2024-02-29) = +0.015
- F3 (2024-03-01 .. 2024-05-15) = +0.021
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 6.12. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**9 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_funding_spread`).

### Emergent story

`xd_funding_spread` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=-0.01 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.010. Cross-regime: strongest in bull (z=-0.02). Cross-DNA: most pronounced in BLUE bucket (z=+0.02). Fold consistency: fold-unstable (F1=-0.05, F2=+0.02, F3=+0.02). Distributionally mean=+0.0056 std=0.7079 skew=-0.01 kurt=+1.84. used by 9 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_kyle_lambda`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.011277
- std  = 1.185102
- p10 = -1.5835
- p50 = +0.0690
- p90 = +1.4971
- skew = -0.204
- excess kurtosis = -0.228

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.004 |
| t-3 | indicator 3 days BEFORE event | -0.011 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.010 |
| t-0 | indicator on event day (concurrent) | -0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.018 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.022
- chop (label=1) = -0.008
- bear (label=0) = -0.029

Raw regime keys (column-as-found):
  - `label_1`: -0.008
  - `label_2`: +0.022
  - `label_0`: -0.029

Regime story: strongest absolute deviation in **bear** (z=-0.029); weakest in chop (z=-0.008); spread=0.021 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.031
- STEADY = -0.019
- VOLATILE = +0.007
- DEGEN = -0.006

DNA story: strongest in **BLUE** (z=+0.031); weakest in DEGEN (z=-0.006); cross-bucket spread = 0.037 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.017
- F2 (2023-11-01 .. 2024-02-29) = +0.039
- F3 (2024-03-01 .. 2024-05-15) = +0.082
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.16. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_kyle_lambda`).

### Emergent story

`norm_kyle_lambda` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=+0.01 -> t-0=-0.00 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.010. Cross-regime: strongest in bear (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.02, F2=+0.04, F3=+0.08). Distributionally mean=+0.0113 std=1.1851 skew=-0.20 kurt=-0.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `wh_whale_net_usd`

**Family**: `wh_*` — whale on-chain / large-trade activity (>500k USD trades, net flow)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +14094.397688
- std  = 3133079.075633
- p10 = -728247.9302
- p50 = +0.0000
- p90 = +610980.3309
- skew = +4.617
- excess kurtosis = +761.799

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.011 |
| t-3 | indicator 3 days BEFORE event | +0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.014 |
| t-0 | indicator on event day (concurrent) | +0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.220 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.023**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.010**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.027
- chop (label=1) = +0.007
- bear (label=0) = -0.037

Raw regime keys (column-as-found):
  - `label_2`: +0.027
  - `label_0`: -0.037
  - `label_1`: +0.007

Regime story: strongest absolute deviation in **bear** (z=-0.037); weakest in chop (z=+0.007); spread=0.044 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.011
- STEADY = +0.016
- VOLATILE = -0.022
- DEGEN = +0.039

DNA story: strongest in **DEGEN** (z=+0.039); weakest in BLUE (z=+0.011); cross-bucket spread = 0.029 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.010
- F2 (2023-11-01 .. 2024-02-29) = +0.003
- F3 (2024-03-01 .. 2024-05-15) = +0.049
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.79. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**4 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/wh_whale_net_usd`).

### Emergent story

`wh_whale_net_usd` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.01 -> t-0=+0.00 -> t+1=+0.22, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.010. Cross-regime: strongest in bear (z=-0.04). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-unstable (F1=-0.01, F2=+0.00, F3=+0.05). Distributionally mean=+14094.3977 std=3133079.0756 skew=+4.62 kurt=+761.80. used by 4 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_return_kurtosis`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.011827
- std  = 1.180244
- p10 = -1.2279
- p50 = -0.2598
- p90 = +1.6892
- skew = +0.961
- excess kurtosis = +0.982

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.006 |
| t-3 | indicator 3 days BEFORE event | +0.012 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.018 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.007**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.009**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.011
- chop (label=1) = +0.007
- bear (label=0) = -0.001

Raw regime keys (column-as-found):
  - `label_1`: +0.007
  - `label_2`: -0.011
  - `label_0`: -0.001

Regime story: strongest absolute deviation in **bull** (z=-0.011); weakest in bear (z=-0.001); spread=0.009 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.046
- STEADY = -0.006
- VOLATILE = -0.009
- DEGEN = +0.006

DNA story: strongest in **BLUE** (z=+0.046); weakest in STEADY (z=-0.006); cross-bucket spread = 0.052 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.006
- F2 (2023-11-01 .. 2024-02-29) = -0.039
- F3 (2024-03-01 .. 2024-05-15) = -0.016
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.68. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_return_kurtosis`).

### Emergent story

`norm_return_kurtosis` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=-0.00 -> t+1=-0.02, while bottom-25% movers sit at t-1=-0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.009. Cross-regime: strongest in bull (z=-0.01). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-stable (F1=-0.01, F2=-0.04, F3=-0.02). Distributionally mean=+0.0118 std=1.1802 skew=+0.96 kurt=+0.98. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_funding_momentum`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.003177
- std  = 0.201126
- p10 = -0.1606
- p50 = +0.0000
- p90 = +0.1366
- skew = -0.008
- excess kurtosis = +63.653

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | -0.007 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.008 |
| t-0 | indicator on event day (concurrent) | -0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.011 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.001**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.009**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.011
- chop (label=1) = -0.009
- bear (label=0) = +0.010

Raw regime keys (column-as-found):
  - `label_1`: -0.009
  - `label_0`: +0.010
  - `label_2`: -0.011

Regime story: strongest absolute deviation in **bull** (z=-0.011); weakest in chop (z=-0.009); spread=0.003 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.006
- STEADY = -0.009
- VOLATILE = -0.001
- DEGEN = -0.009

DNA story: strongest in **DEGEN** (z=-0.009); weakest in VOLATILE (z=-0.001); cross-bucket spread = 0.008 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.082
- F2 (2023-11-01 .. 2024-02-29) = +0.006
- F3 (2024-03-01 .. 2024-05-15) = +0.028
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.99. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**1 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_funding_momentum`).

### Emergent story

`norm_funding_momentum` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=+0.01 -> t-0=-0.00 -> t+1=+0.01, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.009. Cross-regime: strongest in bull (z=-0.01). Cross-DNA: most pronounced in DEGEN bucket (z=-0.01). Fold consistency: fold-unstable (F1=-0.08, F2=+0.01, F3=+0.03). Distributionally mean=-0.0032 std=0.2011 skew=-0.01 kurt=+63.65. used by 1 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_z30`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 50890
- mean = -0.000184
- std  = 1.076569
- p10 = -1.2621
- p50 = +0.0168
- p90 = +1.2473
- skew = -0.099
- excess kurtosis = +2.093

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | -0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.047 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.048 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.006**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.009**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.036
- chop (label=1) = -0.060
- bear (label=0) = -0.046

Raw regime keys (column-as-found):
  - `label_2`: -0.036
  - `label_1`: -0.060
  - `label_0`: -0.046

Regime story: strongest absolute deviation in **chop** (z=-0.060); weakest in bull (z=-0.036); spread=0.023 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.002
- STEADY = -0.061
- VOLATILE = -0.039
- DEGEN = -0.052

DNA story: strongest in **STEADY** (z=-0.061); weakest in BLUE (z=-0.002); cross-bucket spread = 0.059 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.081
- F2 (2023-11-01 .. 2024-02-29) = -0.062
- F3 (2024-03-01 .. 2024-05-15) = +0.006
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.82. (Magnitudes are tight - stable signal.)

### Catalog usage

**8 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_z30`).

### Emergent story

`bs_basis_z30` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.05 -> t+1=+0.05, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.009. Cross-regime: strongest in chop (z=-0.06). Cross-DNA: most pronounced in STEADY bucket (z=-0.06). Fold consistency: fold-unstable (F1=-0.08, F2=-0.06, F3=+0.01). Distributionally mean=-0.0002 std=1.0766 skew=-0.10 kurt=+2.09. used by 8 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `s3_top_pos_lsr_xsec_z`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30659
- mean = +0.002090
- std  = 0.981340
- p10 = -1.1106
- p50 = -0.0866
- p90 = +1.2050
- skew = +0.666
- excess kurtosis = +1.181

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.004 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.016 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.005**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.009**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.025
- chop (label=1) = -0.003
- bear (label=0) = +0.026

Raw regime keys (column-as-found):
  - `label_1`: -0.003
  - `label_0`: +0.026
  - `label_2`: -0.025

Regime story: strongest absolute deviation in **bear** (z=+0.026); weakest in chop (z=-0.003); spread=0.029 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.074
- STEADY = +0.027
- VOLATILE = -0.018
- DEGEN = +0.003

DNA story: strongest in **BLUE** (z=-0.074); weakest in DEGEN (z=+0.003); cross-bucket spread = 0.077 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.009
- F2 (2023-11-01 .. 2024-02-29) = +0.037
- F3 (2024-03-01 .. 2024-05-15) = +0.022
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.51. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_top_pos_lsr_xsec_z`).

### Emergent story

`s3_top_pos_lsr_xsec_z` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.00 -> t-0=-0.00 -> t+1=-0.02, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.009. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in BLUE bucket (z=-0.07). Fold consistency: fold-stable (F1=+0.01, F2=+0.04, F3=+0.02). Distributionally mean=+0.0021 std=0.9813 skew=+0.67 kurt=+1.18. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bd_total_depth_l5_mean`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +115175839.364394
- std  = 622676786.862374
- p10 = +65781.3068
- p50 = +2054526.1025
- p90 = +47092968.0839
- skew = +7.723
- excess kurtosis = +64.757

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.031 |
| t-3 | indicator 3 days BEFORE event | +0.032 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.037 |
| t-0 | indicator on event day (concurrent) | +0.046 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.070 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.046**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.008**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.017
- chop (label=1) = +0.109
- bear (label=0) = +0.068

Raw regime keys (column-as-found):
  - `label_1`: +0.109
  - `label_0`: +0.068
  - `label_2`: -0.017

Regime story: strongest absolute deviation in **chop** (z=+0.109); weakest in bull (z=-0.017); spread=0.127 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.080
- STEADY = +0.016
- VOLATILE = +0.049
- DEGEN = +0.061

DNA story: strongest in **BLUE** (z=+0.080); weakest in STEADY (z=+0.016); cross-bucket spread = 0.064 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.837
- F2 (2023-11-01 .. 2024-02-29) = -0.051
- F3 (2024-03-01 .. 2024-05-15) = -0.694
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 20.56. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_total_depth_l5_mean`).

### Emergent story

`bd_total_depth_l5_mean` is a low-signal background: top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.04 -> t-0=+0.05 -> t+1=+0.07, while bottom-25% movers sit at t-1=+0.05 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.008. Cross-regime: strongest in chop (z=+0.11). Cross-DNA: most pronounced in BLUE bucket (z=+0.08). Fold consistency: fold-unstable (F1=+0.84, F2=-0.05, F3=-0.69). Distributionally mean=+115175839.3644 std=622676786.8624 skew=+7.72 kurt=+64.76. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_btc_etf_total_usdm`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +127.160477
- std  = 246.102754
- p10 = -126.6000
- p50 = +66.0000
- p90 = +505.4000
- skew = +0.715
- excess kurtosis = +0.936

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.019 |
| t-3 | indicator 3 days BEFORE event | -0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.008**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.073
- chop (label=1) = +0.056
- bear (label=0) = -0.163

Raw regime keys (column-as-found):
  - `label_1`: +0.056
  - `label_0`: -0.163
  - `label_2`: +0.073

Regime story: strongest absolute deviation in **bear** (z=-0.163); weakest in chop (z=+0.056); spread=0.219 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.075
- STEADY = -0.037
- VOLATILE = +0.013
- DEGEN = +0.024

DNA story: strongest in **BLUE** (z=-0.075); weakest in VOLATILE (z=+0.013); cross-bucket spread = 0.088 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.315
- F3 (2024-03-01 .. 2024-05-15) = -0.198
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.39. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_total_usdm`).

### Emergent story

`etf_btc_etf_total_usdm` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.008. Cross-regime: strongest in bear (z=-0.16). Cross-DNA: most pronounced in BLUE bucket (z=-0.08). Fold consistency: fold-unstable (F1=+nan, F2=+0.32, F3=-0.20). Distributionally mean=+127.1605 std=246.1028 skew=+0.71 kurt=+0.94. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_ma_distance`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.047177
- std  = 1.198616
- p10 = -1.5423
- p50 = +0.0633
- p90 = +1.6094
- skew = -0.034
- excess kurtosis = -0.698

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.004 |
| t-3 | indicator 3 days BEFORE event | -0.016 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.004 |
| t-0 | indicator on event day (concurrent) | -0.011 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.019 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.012**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.008**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.244
- chop (label=1) = +0.017
- bear (label=0) = -0.419

Raw regime keys (column-as-found):
  - `label_2`: +0.244
  - `label_0`: -0.419
  - `label_1`: +0.017

Regime story: strongest absolute deviation in **bear** (z=-0.419); weakest in chop (z=+0.017); spread=0.436 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.016
- STEADY = -0.031
- VOLATILE = -0.003
- DEGEN = +0.005

DNA story: strongest in **STEADY** (z=-0.031); weakest in VOLATILE (z=-0.003); cross-bucket spread = 0.027 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.065
- F2 (2023-11-01 .. 2024-02-29) = +0.029
- F3 (2024-03-01 .. 2024-05-15) = -0.036
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.18. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_ma_distance`).

### Emergent story

`norm_ma_distance` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.00 -> t-0=-0.01 -> t+1=+0.02, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.008. Cross-regime: strongest in bear (z=-0.42). Cross-DNA: most pronounced in STEADY bucket (z=-0.03). Fold consistency: fold-unstable (F1=+0.06, F2=+0.03, F3=-0.04). Distributionally mean=+0.0472 std=1.1986 skew=-0.03 kurt=-0.70. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bs_basis_xsec_z`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = -0.000003
- std  = 0.982170
- p10 = -1.1005
- p50 = +0.0184
- p90 = +1.0566
- skew = -0.104
- excess kurtosis = +3.126

Shape: approximately symmetric; heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | -0.013 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.050 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.058 |

**Lead/Lag verdict**: mixed (no single lag dominates).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.005**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.008**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.033
- chop (label=1) = -0.064
- bear (label=0) = -0.059

Raw regime keys (column-as-found):
  - `label_0`: -0.059
  - `label_1`: -0.064
  - `label_2`: -0.033

Regime story: strongest absolute deviation in **chop** (z=-0.064); weakest in bull (z=-0.033); spread=0.031 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.018
- STEADY = -0.058
- VOLATILE = -0.041
- DEGEN = -0.070

DNA story: strongest in **DEGEN** (z=-0.070); weakest in BLUE (z=-0.018); cross-bucket spread = 0.052 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.054
- F2 (2023-11-01 .. 2024-02-29) = -0.036
- F3 (2024-03-01 .. 2024-05-15) = -0.012
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.51. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_xsec_z`).

### Emergent story

`bs_basis_xsec_z` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.05 -> t+1=+0.06, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.008. Cross-regime: strongest in chop (z=-0.06). Cross-DNA: most pronounced in DEGEN bucket (z=-0.07). Fold consistency: fold-stable (F1=-0.05, F2=-0.04, F3=-0.01). Distributionally mean=-0.0000 std=0.9822 skew=-0.10 kurt=+3.13. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `s3_top_pos_lsr`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 30659
- mean = +1.261504
- std  = 0.523035
- p10 = +0.8970
- p50 = +1.0782
- p90 = +1.8560
- skew = +3.015
- excess kurtosis = +13.103

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.043 |
| t-3 | indicator 3 days BEFORE event | -0.037 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.039 |
| t-0 | indicator on event day (concurrent) | -0.040 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.049 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.032**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.008**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.025
- chop (label=1) = -0.100
- bear (label=0) = -0.059

Raw regime keys (column-as-found):
  - `label_1`: -0.100
  - `label_2`: +0.025
  - `label_0`: -0.059

Regime story: strongest absolute deviation in **chop** (z=-0.100); weakest in bull (z=+0.025); spread=0.125 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.036
- STEADY = -0.007
- VOLATILE = -0.051
- DEGEN = -0.085

DNA story: strongest in **DEGEN** (z=-0.085); weakest in STEADY (z=-0.007); cross-bucket spread = 0.078 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.306
- F2 (2023-11-01 .. 2024-02-29) = +0.515
- F3 (2024-03-01 .. 2024-05-15) = +1.555
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.29. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_top_pos_lsr`).

### Emergent story

`s3_top_pos_lsr` is a low-signal background: top-25%-mover z-lift goes t-3=-0.04 -> t-1=-0.04 -> t-0=-0.04 -> t+1=-0.05, while bottom-25% movers sit at t-1=-0.03 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.008. Cross-regime: strongest in chop (z=-0.10). Cross-DNA: most pronounced in DEGEN bucket (z=-0.09). Fold consistency: fold-unstable (F1=-0.31, F2=+0.51, F3=+1.56). Distributionally mean=+1.2615 std=0.5230 skew=+3.02 kurt=+13.10. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `te_in`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = +0.131797
- std  = 0.027599
- p10 = +0.0982
- p50 = +0.1295
- p90 = +0.1677
- skew = +0.535
- excess kurtosis = +0.504

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.002**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.044
- chop (label=1) = -0.053
- bear (label=0) = +0.013

Raw regime keys (column-as-found):
  - `label_1`: -0.053
  - `label_0`: +0.013
  - `label_2`: +0.044

Regime story: strongest absolute deviation in **chop** (z=-0.053); weakest in bear (z=+0.013); spread=0.066 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.021
- STEADY = +0.005
- VOLATILE = +0.002
- DEGEN = +0.020

DNA story: strongest in **BLUE** (z=-0.021); weakest in VOLATILE (z=+0.002); cross-bucket spread = 0.022 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.070
- F2 (2023-11-01 .. 2024-02-29) = +0.357
- F3 (2024-03-01 .. 2024-05-15) = +0.154
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.19. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_in`).

### Emergent story

`te_in` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.007. Cross-regime: strongest in chop (z=-0.05). Cross-DNA: most pronounced in BLUE bucket (z=-0.02). Fold consistency: fold-unstable (F1=-0.07, F2=+0.36, F3=+0.15). Distributionally mean=+0.1318 std=0.0276 skew=+0.53 kurt=+0.50. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_delta_3d`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51261
- mean = +0.000121
- std  = 0.294665
- p10 = -0.2726
- p50 = +0.0000
- p90 = +0.2692
- skew = +0.089
- excess kurtosis = +24.487

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | -0.032 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.032 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.002**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.029
- chop (label=1) = -0.029
- bear (label=0) = -0.039

Raw regime keys (column-as-found):
  - `label_0`: -0.039
  - `label_1`: -0.029
  - `label_2`: -0.029

Regime story: strongest absolute deviation in **bear** (z=-0.039); weakest in chop (z=-0.029); spread=0.010 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.028
- STEADY = -0.028
- VOLATILE = -0.031
- DEGEN = -0.043

DNA story: strongest in **DEGEN** (z=-0.043); weakest in BLUE (z=-0.028); cross-bucket spread = 0.015 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.053
- F2 (2023-11-01 .. 2024-02-29) = -0.040
- F3 (2024-03-01 .. 2024-05-15) = -0.006
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.60. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_delta_3d`).

### Emergent story

`bs_basis_delta_3d` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.03 -> t+1=+0.03, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.007. Cross-regime: strongest in bear (z=-0.04). Cross-DNA: most pronounced in DEGEN bucket (z=-0.04). Fold consistency: fold-stable (F1=-0.05, F2=-0.04, F3=-0.01). Distributionally mean=+0.0001 std=0.2947 skew=+0.09 kurt=+24.49. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `etf_btc_etf_mega_inflow`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.100844
- std  = 0.301122
- p10 = +0.0000
- p50 = +0.0000
- p90 = +1.0000
- skew = +2.651
- excess kurtosis = +5.028

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.017 |
| t-3 | indicator 3 days BEFORE event | +0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.003 |
| t-0 | indicator on event day (concurrent) | +0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.003 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.011**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.067
- chop (label=1) = +0.061
- bear (label=0) = -0.150

Raw regime keys (column-as-found):
  - `label_2`: +0.067
  - `label_1`: +0.061
  - `label_0`: -0.150

Regime story: strongest absolute deviation in **bear** (z=-0.150); weakest in chop (z=+0.061); spread=0.212 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.031
- STEADY = -0.070
- VOLATILE = +0.019
- DEGEN = +0.034

DNA story: strongest in **STEADY** (z=-0.070); weakest in VOLATILE (z=+0.019); cross-bucket spread = 0.088 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.186
- F3 (2024-03-01 .. 2024-05-15) = -0.113
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.06. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_mega_inflow`).

### Emergent story

`etf_btc_etf_mega_inflow` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.007. Cross-regime: strongest in bear (z=-0.15). Cross-DNA: most pronounced in STEADY bucket (z=-0.07). Fold consistency: fold-unstable (F1=+nan, F2=+0.19, F3=-0.11). Distributionally mean=+0.1008 std=0.3011 skew=+2.65 kurt=+5.03. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bs_basis_bull_shock`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = +0.033567
- std  = 0.180111
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +5.179
- excess kurtosis = +24.826

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.005 |
| t-3 | indicator 3 days BEFORE event | -0.017 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.006 |
| t-0 | indicator on event day (concurrent) | -0.019 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.027 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.030
- chop (label=1) = -0.031
- bear (label=0) = +0.013

Raw regime keys (column-as-found):
  - `label_0`: +0.013
  - `label_1`: -0.031
  - `label_2`: -0.030

Regime story: strongest absolute deviation in **chop** (z=-0.031); weakest in bear (z=+0.013); spread=0.044 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.003
- STEADY = -0.039
- VOLATILE = -0.009
- DEGEN = -0.001

DNA story: strongest in **STEADY** (z=-0.039); weakest in DEGEN (z=-0.001); cross-bucket spread = 0.038 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.004
- F2 (2023-11-01 .. 2024-02-29) = -0.053
- F3 (2024-03-01 .. 2024-05-15) = -0.048
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.63. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_bull_shock`).

### Emergent story

`bs_basis_bull_shock` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=-0.02 -> t+1=+0.03, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.007. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in STEADY bucket (z=-0.04). Fold consistency: fold-stable (F1=-0.00, F2=-0.05, F3=-0.05). Distributionally mean=+0.0336 std=0.1801 skew=+5.18 kurt=+24.83. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `te_in_btc`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = +0.074947
- std  = 0.031801
- p10 = +0.0379
- p50 = +0.0729
- p90 = +0.1165
- skew = +0.267
- excess kurtosis = +0.474

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.006 |
| t-0 | indicator on event day (concurrent) | -0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.001
- chop (label=1) = -0.017
- bear (label=0) = +0.008

Raw regime keys (column-as-found):
  - `label_1`: -0.017
  - `label_0`: +0.008
  - `label_2`: -0.001

Regime story: strongest absolute deviation in **chop** (z=-0.017); weakest in bull (z=-0.001); spread=0.015 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.021
- STEADY = -0.005
- VOLATILE = +0.004
- DEGEN = -0.017

DNA story: strongest in **BLUE** (z=-0.021); weakest in VOLATILE (z=+0.004); cross-bucket spread = 0.025 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.011
- F2 (2023-11-01 .. 2024-02-29) = +0.017
- F3 (2024-03-01 .. 2024-05-15) = -0.133
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.97. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**14 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_in_btc`).

### Emergent story

`te_in_btc` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.01 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.007. Cross-regime: strongest in chop (z=-0.02). Cross-DNA: most pronounced in BLUE bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.01, F2=+0.02, F3=-0.13). Distributionally mean=+0.0749 std=0.0318 skew=+0.27 kurt=+0.47. used by 14 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_funding`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.008286
- std  = 0.452046
- p10 = -0.3922
- p50 = +0.0000
- p90 = +0.4885
- skew = -0.523
- excess kurtosis = +8.092

Shape: left-skewed (downside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.009 |
| t-3 | indicator 3 days BEFORE event | +0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | -0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.019 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.007**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.007**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.013
- chop (label=1) = -0.008
- bear (label=0) = -0.003

Raw regime keys (column-as-found):
  - `label_1`: -0.008
  - `label_0`: -0.003
  - `label_2`: -0.013

Regime story: strongest absolute deviation in **bull** (z=-0.013); weakest in bear (z=-0.003); spread=0.011 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.009
- STEADY = -0.007
- VOLATILE = -0.009
- DEGEN = -0.017

DNA story: strongest in **DEGEN** (z=-0.017); weakest in STEADY (z=-0.007); cross-bucket spread = 0.009 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.056
- F2 (2023-11-01 .. 2024-02-29) = -0.011
- F3 (2024-03-01 .. 2024-05-15) = -0.040
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.52. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_funding`).

### Emergent story

`norm_funding` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=-0.01 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.007. Cross-regime: strongest in bull (z=-0.01). Cross-DNA: most pronounced in DEGEN bucket (z=-0.02). Fold consistency: fold-stable (F1=-0.06, F2=-0.01, F3=-0.04). Distributionally mean=+0.0083 std=0.4520 skew=-0.52 kurt=+8.09. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_hawkes_buy_intensity`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.164041
- std  = 0.643241
- p10 = -0.5162
- p50 = -0.3128
- p90 = +0.2093
- skew = +5.068
- excess kurtosis = +32.079

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.000 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.001 |
| t-0 | indicator on event day (concurrent) | +0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.010 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.005**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.015
- chop (label=1) = -0.009
- bear (label=0) = +0.026

Raw regime keys (column-as-found):
  - `label_2`: +0.015
  - `label_1`: -0.009
  - `label_0`: +0.026

Regime story: strongest absolute deviation in **bear** (z=+0.026); weakest in chop (z=-0.009); spread=0.035 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.053
- STEADY = +0.017
- VOLATILE = +0.006
- DEGEN = -0.005

DNA story: strongest in **BLUE** (z=+0.053); weakest in DEGEN (z=-0.005); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.080
- F2 (2023-11-01 .. 2024-02-29) = -0.013
- F3 (2024-03-01 .. 2024-05-15) = -0.013
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.89. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_hawkes_buy_intensity`).

### Emergent story

`norm_hawkes_buy_intensity` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.01 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.006. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-stable (F1=-0.08, F2=-0.01, F3=-0.01). Distributionally mean=-0.1640 std=0.6432 skew=+5.07 kurt=+32.08. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `norm_deviation`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.040985
- std  = 0.987069
- p10 = -1.2375
- p50 = +0.0548
- p90 = +1.3002
- skew = -0.046
- excess kurtosis = -0.012

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.017 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.017 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.009**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.206
- chop (label=1) = +0.010
- bear (label=0) = -0.374

Raw regime keys (column-as-found):
  - `label_2`: +0.206
  - `label_1`: +0.010
  - `label_0`: -0.374

Regime story: strongest absolute deviation in **bear** (z=-0.374); weakest in chop (z=+0.010); spread=0.384 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.018
- STEADY = -0.025
- VOLATILE = -0.010
- DEGEN = -0.019

DNA story: strongest in **STEADY** (z=-0.025); weakest in VOLATILE (z=-0.010); cross-bucket spread = 0.015 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.109
- F2 (2023-11-01 .. 2024-02-29) = -0.021
- F3 (2024-03-01 .. 2024-05-15) = -0.112
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 11.33. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**12 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_deviation`).

### Emergent story

`norm_deviation` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.00 -> t-0=-0.02 -> t+1=+0.00, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.006. Cross-regime: strongest in bear (z=-0.37). Cross-DNA: most pronounced in STEADY bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.11, F2=-0.02, F3=-0.11). Distributionally mean=+0.0410 std=0.9871 skew=-0.05 kurt=-0.01. used by 12 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_spread_bps`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.038265
- std  = 0.871389
- p10 = -0.9869
- p50 = -0.1761
- p90 = +1.0901
- skew = +1.064
- excess kurtosis = +2.237

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.003 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.009 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.009**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.058
- chop (label=1) = -0.003
- bear (label=0) = +0.108

Raw regime keys (column-as-found):
  - `label_1`: -0.003
  - `label_0`: +0.108
  - `label_2`: -0.058

Regime story: strongest absolute deviation in **bear** (z=+0.108); weakest in chop (z=-0.003); spread=0.110 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.028
- STEADY = +0.017
- VOLATILE = +0.007
- DEGEN = -0.027

DNA story: strongest in **BLUE** (z=+0.028); weakest in VOLATILE (z=+0.007); cross-bucket spread = 0.021 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.045
- F2 (2023-11-01 .. 2024-02-29) = -0.072
- F3 (2024-03-01 .. 2024-05-15) = -0.141
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.47. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_spread_bps`).

### Emergent story

`norm_spread_bps` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.006. Cross-regime: strongest in bear (z=+0.11). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-stable (F1=-0.05, F2=-0.07, F3=-0.14). Distributionally mean=-0.0383 std=0.8714 skew=+1.06 kurt=+2.24. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_whale`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.145420
- std  = 0.603561
- p10 = -0.5512
- p50 = -0.2464
- p90 = +0.2640
- skew = +4.283
- excess kurtosis = +27.763

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.010 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.004 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.011 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.010**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.005
- chop (label=1) = -0.016
- bear (label=0) = +0.016

Raw regime keys (column-as-found):
  - `label_1`: -0.016
  - `label_2`: +0.005
  - `label_0`: +0.016

Regime story: strongest absolute deviation in **bear** (z=+0.016); weakest in bull (z=+0.005); spread=0.011 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.005
- STEADY = +0.010
- VOLATILE = -0.011
- DEGEN = +0.016

DNA story: strongest in **DEGEN** (z=+0.016); weakest in BLUE (z=+0.005); cross-bucket spread = 0.011 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.029
- F2 (2023-11-01 .. 2024-02-29) = -0.008
- F3 (2024-03-01 .. 2024-05-15) = -0.013
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.55. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_whale`).

### Emergent story

`norm_whale` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=+0.00 -> t+1=+0.01, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.006. Cross-regime: strongest in bear (z=+0.02). Cross-DNA: most pronounced in DEGEN bucket (z=+0.02). Fold consistency: fold-stable (F1=-0.03, F2=-0.01, F3=-0.01). Distributionally mean=-0.1454 std=0.6036 skew=+4.28 kurt=+27.76. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xd_ma_distance`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.011660
- std  = 1.296518
- p10 = -1.6943
- p50 = -0.0231
- p90 = +1.6940
- skew = +0.033
- excess kurtosis = -0.405

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.008 |
| t-3 | indicator 3 days BEFORE event | -0.014 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.000 |
| t-0 | indicator on event day (concurrent) | -0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.003 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.006**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.026
- chop (label=1) = +0.018
- bear (label=0) = +0.003

Raw regime keys (column-as-found):
  - `label_2`: -0.026
  - `label_1`: +0.018
  - `label_0`: +0.003

Regime story: strongest absolute deviation in **bull** (z=-0.026); weakest in bear (z=+0.003); spread=0.028 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.006
- STEADY = -0.013
- VOLATILE = -0.000
- DEGEN = +0.003

DNA story: strongest in **STEADY** (z=-0.013); weakest in VOLATILE (z=-0.000); cross-bucket spread = 0.012 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.007
- F2 (2023-11-01 .. 2024-02-29) = +0.036
- F3 (2024-03-01 .. 2024-05-15) = +0.023
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.55. (Magnitudes are tight - stable signal.)

### Catalog usage

**17 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_ma_distance`).

### Emergent story

`xd_ma_distance` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.006. Cross-regime: strongest in bull (z=-0.03). Cross-DNA: most pronounced in STEADY bucket (z=-0.01). Fold consistency: fold-stable (F1=+0.01, F2=+0.04, F3=+0.02). Distributionally mean=-0.0117 std=1.2965 skew=+0.03 kurt=-0.40. used by 17 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xd_btc_volatility`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.031143
- std  = 0.881984
- p10 = -1.0922
- p50 = -0.0715
- p90 = +1.0868
- skew = +0.590
- excess kurtosis = +1.582

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.003 |
| t-0 | indicator on event day (concurrent) | +0.018 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.009**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.053
- chop (label=1) = -0.021
- bear (label=0) = +0.167

Raw regime keys (column-as-found):
  - `label_2`: -0.053
  - `label_0`: +0.167
  - `label_1`: -0.021

Regime story: strongest absolute deviation in **bear** (z=+0.167); weakest in chop (z=-0.021); spread=0.187 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.036
- STEADY = +0.016
- VOLATILE = +0.025
- DEGEN = -0.001

DNA story: strongest in **BLUE** (z=+0.036); weakest in DEGEN (z=-0.001); cross-bucket spread = 0.037 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.113
- F2 (2023-11-01 .. 2024-02-29) = -0.100
- F3 (2024-03-01 .. 2024-05-15) = +0.008
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.80. (Magnitudes are tight - stable signal.)

### Catalog usage

**6 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_btc_volatility`).

### Emergent story

`xd_btc_volatility` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=+0.02 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.006. Cross-regime: strongest in bear (z=+0.17). Cross-DNA: most pronounced in BLUE bucket (z=+0.04). Fold consistency: fold-unstable (F1=-0.11, F2=-0.10, F3=+0.01). Distributionally mean=-0.0311 std=0.8820 skew=+0.59 kurt=+1.58. used by 6 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_bar_duration`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.513124
- std  = 0.742384
- p10 = -0.2825
- p50 = +0.6529
- p90 = +1.1989
- skew = -2.153
- excess kurtosis = +8.448

Shape: left-skewed (downside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.048 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.009**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.006**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.015
- chop (label=1) = +0.075
- bear (label=0) = -0.121

Raw regime keys (column-as-found):
  - `label_1`: +0.075
  - `label_2`: -0.015
  - `label_0`: -0.121

Regime story: strongest absolute deviation in **bear** (z=-0.121); weakest in bull (z=-0.015); spread=0.106 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.061
- STEADY = -0.015
- VOLATILE = -0.008
- DEGEN = -0.019

DNA story: strongest in **BLUE** (z=-0.061); weakest in VOLATILE (z=-0.008); cross-bucket spread = 0.052 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.194
- F2 (2023-11-01 .. 2024-02-29) = +0.037
- F3 (2024-03-01 .. 2024-05-15) = -0.014
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.22. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_bar_duration`).

### Emergent story

`norm_bar_duration` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.05, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.006. Cross-regime: strongest in bear (z=-0.12). Cross-DNA: most pronounced in BLUE bucket (z=-0.06). Fold consistency: fold-unstable (F1=+0.19, F2=+0.04, F3=-0.01). Distributionally mean=+0.5131 std=0.7424 skew=-2.15 kurt=+8.45. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xd_cross_vol_mean`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.052833
- std  = 1.207950
- p10 = -1.5310
- p50 = -0.1422
- p90 = +1.5624
- skew = +0.340
- excess kurtosis = +0.153

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.003 |
| t-3 | indicator 3 days BEFORE event | +0.008 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.001 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.011 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.007**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.005**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.089
- chop (label=1) = +0.028
- bear (label=0) = +0.085

Raw regime keys (column-as-found):
  - `label_1`: +0.028
  - `label_2`: -0.089
  - `label_0`: +0.085

Regime story: strongest absolute deviation in **bull** (z=-0.089); weakest in chop (z=+0.028); spread=0.117 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.018
- STEADY = -0.005
- VOLATILE = -0.002
- DEGEN = -0.012

DNA story: strongest in **BLUE** (z=+0.018); weakest in VOLATILE (z=-0.002); cross-bucket spread = 0.020 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.098
- F2 (2023-11-01 .. 2024-02-29) = -0.018
- F3 (2024-03-01 .. 2024-05-15) = -0.098
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 13.30. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_cross_vol_mean`).

### Emergent story

`xd_cross_vol_mean` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=-0.00 -> t-0=-0.00 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.005. Cross-regime: strongest in bull (z=-0.09). Cross-DNA: most pronounced in BLUE bucket (z=+0.02). Fold consistency: fold-unstable (F1=+0.10, F2=-0.02, F3=-0.10). Distributionally mean=-0.0528 std=1.2079 skew=+0.34 kurt=+0.15. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_compound_shock`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.021287
- std  = 0.144341
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +6.633
- excess kurtosis = +41.998

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.006**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.005**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.057
- chop (label=1) = -0.071
- bear (label=0) = +0.000

Raw regime keys (column-as-found):
  - `label_0`: +0.000
  - `label_1`: -0.071
  - `label_2`: +0.057

Regime story: strongest absolute deviation in **chop** (z=-0.071); weakest in bear (z=+0.000); spread=0.071 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.033
- STEADY = +0.009
- VOLATILE = -0.005
- DEGEN = -0.010

DNA story: strongest in **BLUE** (z=+0.033); weakest in VOLATILE (z=-0.005); cross-bucket spread = 0.038 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.071
- F2 (2023-11-01 .. 2024-02-29) = -0.078
- F3 (2024-03-01 .. 2024-05-15) = +0.046
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.65. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_compound_shock`).

### Emergent story

`stbl_compound_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.005. Cross-regime: strongest in chop (z=-0.07). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.07, F2=-0.08, F3=+0.05). Distributionally mean=+0.0213 std=0.1443 skew=+6.63 kurt=+42.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xrel_wh_whale_net_usd_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51796
- mean = +0.515397
- std  = 0.281289
- p10 = +0.1176
- p50 = +0.5185
- p90 = +0.9143
- skew = -0.006
- excess kurtosis = -1.063

Shape: approximately symmetric; platykurtic (compressed tails).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.004 |
| t-0 | indicator on event day (concurrent) | -0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.205 |

**Lead/Lag verdict**: LAGGING: largest deviation appears AFTER event.

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.009**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.005**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.020
- chop (label=1) = -0.017
- bear (label=0) = -0.016

Raw regime keys (column-as-found):
  - `label_1`: -0.017
  - `label_0`: -0.016
  - `label_2`: +0.020

Regime story: strongest absolute deviation in **bull** (z=+0.020); weakest in bear (z=-0.016); spread=0.036 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.001
- STEADY = -0.008
- VOLATILE = -0.009
- DEGEN = +0.034

DNA story: strongest in **DEGEN** (z=+0.034); weakest in BLUE (z=+0.001); cross-bucket spread = 0.033 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.012
- F2 (2023-11-01 .. 2024-02-29) = -0.023
- F3 (2024-03-01 .. 2024-05-15) = +0.023
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 5.31. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_wh_whale_net_usd_xrank`).

### Emergent story

`xrel_wh_whale_net_usd_xrank` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=-0.00 -> t+1=+0.21, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.005. Cross-regime: strongest in bull (z=+0.02). Cross-DNA: most pronounced in DEGEN bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.01, F2=-0.02, F3=+0.02). Distributionally mean=+0.5154 std=0.2813 skew=-0.01 kurt=-1.06. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `te_out`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = +0.132990
- std  = 0.027146
- p10 = +0.0999
- p50 = +0.1311
- p90 = +0.1685
- skew = +0.523
- excess kurtosis = +0.491

Shape: right-skewed (upside-spike biased); near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.007 |
| t-3 | indicator 3 days BEFORE event | -0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.004**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.005**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.011
- chop (label=1) = -0.034
- bear (label=0) = +0.023

Raw regime keys (column-as-found):
  - `label_0`: +0.023
  - `label_1`: -0.034
  - `label_2`: +0.011

Regime story: strongest absolute deviation in **chop** (z=-0.034); weakest in bull (z=+0.011); spread=0.045 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.006
- STEADY = -0.006
- VOLATILE = +0.008
- DEGEN = -0.010

DNA story: strongest in **DEGEN** (z=-0.010); weakest in STEADY (z=-0.006); cross-bucket spread = 0.004 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.093
- F2 (2023-11-01 .. 2024-02-29) = +0.349
- F3 (2024-03-01 .. 2024-05-15) = +0.265
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.11. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_out`).

### Emergent story

`te_out` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.005. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in DEGEN bucket (z=-0.01). Fold consistency: fold-unstable (F1=-0.09, F2=+0.35, F3=+0.26). Distributionally mean=+0.1330 std=0.0271 skew=+0.52 kurt=+0.49. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_pct`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = -0.009138
- std  = 0.209398
- p10 = -0.1981
- p50 = +0.0000
- p90 = +0.1730
- skew = +0.067
- excess kurtosis = +28.564

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.003 |
| t-3 | indicator 3 days BEFORE event | -0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | -0.048 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.054 |

**Lead/Lag verdict**: mixed (no single lag dominates).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.005**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.005**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.053
- chop (label=1) = -0.053
- bear (label=0) = -0.033

Raw regime keys (column-as-found):
  - `label_2`: -0.053
  - `label_0`: -0.033
  - `label_1`: -0.053

Regime story: strongest absolute deviation in **bull** (z=-0.053); weakest in bear (z=-0.033); spread=0.020 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.002
- STEADY = -0.055
- VOLATILE = -0.044
- DEGEN = -0.059

DNA story: strongest in **DEGEN** (z=-0.059); weakest in BLUE (z=-0.002); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.074
- F2 (2023-11-01 .. 2024-02-29) = -0.112
- F3 (2024-03-01 .. 2024-05-15) = +0.047
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.47. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_pct`).

### Emergent story

`bs_basis_pct` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=-0.05 -> t+1=+0.05, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.005. Cross-regime: strongest in bull (z=-0.05). Cross-DNA: most pronounced in DEGEN bucket (z=-0.06). Fold consistency: fold-unstable (F1=-0.07, F2=-0.11, F3=+0.05). Distributionally mean=-0.0091 std=0.2094 skew=+0.07 kurt=+28.56. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_vol_price_corr`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.006984
- std  = 1.070825
- p10 = -1.3968
- p50 = +0.0206
- p90 = +1.3295
- skew = -0.083
- excess kurtosis = +0.224

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.004 |
| t-3 | indicator 3 days BEFORE event | -0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.007 |
| t-0 | indicator on event day (concurrent) | -0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.006 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.004**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.000
- chop (label=1) = -0.003
- bear (label=0) = -0.029

Raw regime keys (column-as-found):
  - `label_0`: -0.029
  - `label_1`: -0.003
  - `label_2`: +0.000

Regime story: strongest absolute deviation in **bear** (z=-0.029); weakest in bull (z=+0.000); spread=0.029 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.054
- STEADY = +0.002
- VOLATILE = -0.008
- DEGEN = -0.019

DNA story: strongest in **BLUE** (z=-0.054); weakest in STEADY (z=+0.002); cross-bucket spread = 0.056 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.001
- F2 (2023-11-01 .. 2024-02-29) = +0.030
- F3 (2024-03-01 .. 2024-05-15) = -0.008
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.17. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_vol_price_corr`).

### Emergent story

`norm_vol_price_corr` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.01 -> t+1=-0.01, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.004. Cross-regime: strongest in bear (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-unstable (F1=+0.00, F2=+0.03, F3=-0.01). Distributionally mean=-0.0070 std=1.0708 skew=-0.08 kurt=+0.22. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_vol_ratio`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.071630
- std  = 0.947840
- p10 = -1.1227
- p50 = +0.0688
- p90 = +1.2638
- skew = +0.021
- excess kurtosis = +0.335

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.004 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.007 |
| t-0 | indicator on event day (concurrent) | -0.009 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.004**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.076
- chop (label=1) = -0.021
- bear (label=0) = +0.102

Raw regime keys (column-as-found):
  - `label_0`: +0.102
  - `label_1`: -0.021
  - `label_2`: -0.076

Regime story: strongest absolute deviation in **bear** (z=+0.102); weakest in chop (z=-0.021); spread=0.124 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.020
- STEADY = -0.014
- VOLATILE = -0.012
- DEGEN = -0.004

DNA story: strongest in **BLUE** (z=+0.020); weakest in DEGEN (z=-0.004); cross-bucket spread = 0.023 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.061
- F2 (2023-11-01 .. 2024-02-29) = -0.025
- F3 (2024-03-01 .. 2024-05-15) = -0.090
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.45. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_vol_ratio`).

### Emergent story

`norm_vol_ratio` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=-0.01 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.004. Cross-regime: strongest in bear (z=+0.10). Cross-DNA: most pronounced in BLUE bucket (z=+0.02). Fold consistency: fold-stable (F1=-0.06, F2=-0.03, F3=-0.09). Distributionally mean=+0.0716 std=0.9478 skew=+0.02 kurt=+0.34. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_fd_close`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.039818
- std  = 0.954258
- p10 = -1.2007
- p50 = +0.0630
- p90 = +1.2467
- skew = -0.123
- excess kurtosis = +0.030

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | -0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.010 |
| t-0 | indicator on event day (concurrent) | -0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.047 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.005**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.004**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.294
- chop (label=1) = -0.004
- bear (label=0) = -0.481

Raw regime keys (column-as-found):
  - `label_0`: -0.481
  - `label_2`: +0.294
  - `label_1`: -0.004

Regime story: strongest absolute deviation in **bear** (z=-0.481); weakest in chop (z=-0.004); spread=0.477 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.005
- STEADY = -0.025
- VOLATILE = -0.011
- DEGEN = -0.016

DNA story: strongest in **STEADY** (z=-0.025); weakest in BLUE (z=+0.005); cross-bucket spread = 0.030 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.087
- F2 (2023-11-01 .. 2024-02-29) = +0.024
- F3 (2024-03-01 .. 2024-05-15) = -0.069
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.60. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_fd_close`).

### Emergent story

`norm_fd_close` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=-0.02 -> t+1=+0.05, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.004. Cross-regime: strongest in bear (z=-0.48). Cross-DNA: most pronounced in STEADY bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.09, F2=+0.02, F3=-0.07). Distributionally mean=+0.0398 std=0.9543 skew=-0.12 kurt=+0.03. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_thin_book_frac`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18306
- mean = +0.000328
- std  = 0.018101
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +55.209
- excess kurtosis = +3046.000

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.002 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.002 |
| t-0 | indicator on event day (concurrent) | -0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.004 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.004**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.006
- chop (label=1) = +0.005
- bear (label=0) = +0.001

Raw regime keys (column-as-found):
  - `label_1`: +0.005
  - `label_0`: +0.001
  - `label_2`: -0.006

Regime story: strongest absolute deviation in **bull** (z=-0.006); weakest in bear (z=+0.001); spread=0.007 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = +0.000
- VOLATILE = -0.000
- DEGEN = +0.000

DNA story: strongest in **VOLATILE** (z=-0.000); weakest in BLUE (z=+0.000); cross-bucket spread = 0.000 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.007
- F2 (2023-11-01 .. 2024-02-29) = -0.006
- F3 (2024-03-01 .. 2024-05-15) = +0.005
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.29. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_thin_book_frac`).

### Emergent story

`bd_thin_book_frac` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.004. Cross-regime: strongest in bull (z=-0.01). Cross-DNA: most pronounced in VOLATILE bucket (z=-0.00). Fold consistency: fold-unstable (F1=-0.01, F2=-0.01, F3=+0.01). Distributionally mean=+0.0003 std=0.0181 skew=+55.21 kurt=+3046.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `etf_btc_etf_total_z30`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6130
- mean = -0.025914
- std  = 1.093388
- p10 = -1.0922
- p50 = -0.1986
- p90 = +1.6958
- skew = +0.301
- excess kurtosis = +0.951

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.012 |
| t-3 | indicator 3 days BEFORE event | -0.022 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.012 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.005 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.008**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.004**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.047
- chop (label=1) = +0.072
- bear (label=0) = -0.136

Raw regime keys (column-as-found):
  - `label_2`: +0.047
  - `label_1`: +0.072
  - `label_0`: -0.136

Regime story: strongest absolute deviation in **bear** (z=-0.136); weakest in bull (z=+0.047); spread=0.182 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.077
- STEADY = -0.010
- VOLATILE = +0.019
- DEGEN = +0.001

DNA story: strongest in **BLUE** (z=-0.077); weakest in DEGEN (z=+0.001); cross-bucket spread = 0.078 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.521
- F3 (2024-03-01 .. 2024-05-15) = -0.232
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.61. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_total_z30`).

### Emergent story

`etf_btc_etf_total_z30` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.004. Cross-regime: strongest in bear (z=-0.14). Cross-DNA: most pronounced in BLUE bucket (z=-0.08). Fold consistency: fold-unstable (F1=+nan, F2=+0.52, F3=-0.23). Distributionally mean=-0.0259 std=1.0934 skew=+0.30 kurt=+0.95. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_efficiency`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.172821
- std  = 0.928414
- p10 = -1.2149
- p50 = -0.3595
- p90 = +1.1580
- skew = +0.846
- excess kurtosis = +0.263

Shape: right-skewed (upside-spike biased); near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | +0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.004 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.002**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.010
- chop (label=1) = -0.033
- bear (label=0) = +0.013

Raw regime keys (column-as-found):
  - `label_2`: +0.010
  - `label_0`: +0.013
  - `label_1`: -0.033

Regime story: strongest absolute deviation in **chop** (z=-0.033); weakest in bull (z=+0.010); spread=0.043 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.007
- STEADY = +0.000
- VOLATILE = -0.005
- DEGEN = -0.009

DNA story: strongest in **DEGEN** (z=-0.009); weakest in STEADY (z=+0.000); cross-bucket spread = 0.009 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.012
- F2 (2023-11-01 .. 2024-02-29) = -0.018
- F3 (2024-03-01 .. 2024-05-15) = +0.008
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 26.08. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**16 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_efficiency`).

### Emergent story

`norm_efficiency` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.01 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in DEGEN bucket (z=-0.01). Fold consistency: fold-unstable (F1=+0.01, F2=-0.02, F3=+0.01). Distributionally mean=-0.1728 std=0.9284 skew=+0.85 kurt=+0.26. used by 16 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `xd_cross_return_mean`

**Family**: `xd_*` — cross-asset dispersion / rank / momentum panel measures

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.021000
- std  = 0.944907
- p10 = -1.1935
- p50 = +0.0408
- p90 = +1.2159
- skew = -0.096
- excess kurtosis = +0.145

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | -0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.015 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.017 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.006**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.095
- chop (label=1) = -0.035
- bear (label=0) = -0.152

Raw regime keys (column-as-found):
  - `label_2`: +0.095
  - `label_0`: -0.152
  - `label_1`: -0.035

Regime story: strongest absolute deviation in **bear** (z=-0.152); weakest in chop (z=-0.035); spread=0.116 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.013
- STEADY = -0.020
- VOLATILE = -0.012
- DEGEN = -0.014

DNA story: strongest in **STEADY** (z=-0.020); weakest in VOLATILE (z=-0.012); cross-bucket spread = 0.008 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.116
- F2 (2023-11-01 .. 2024-02-29) = -0.003
- F3 (2024-03-01 .. 2024-05-15) = -0.179
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 5.57. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xd_cross_return_mean`).

### Emergent story

`xd_cross_return_mean` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.02, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.003. Cross-regime: strongest in bear (z=-0.15). Cross-DNA: most pronounced in STEADY bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.12, F2=-0.00, F3=-0.18). Distributionally mean=+0.0210 std=0.9449 skew=-0.10 kurt=+0.15. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_tick_count`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.169114
- std  = 0.987028
- p10 = -0.9661
- p50 = +0.0706
- p90 = +1.4120
- skew = +0.752
- excess kurtosis = +1.688

Shape: right-skewed (upside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.006 |
| t-3 | indicator 3 days BEFORE event | +0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | -0.010 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.024 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.002**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.027
- chop (label=1) = +0.008
- bear (label=0) = -0.006

Raw regime keys (column-as-found):
  - `label_0`: -0.006
  - `label_1`: +0.008
  - `label_2`: -0.027

Regime story: strongest absolute deviation in **bull** (z=-0.027); weakest in bear (z=-0.006); spread=0.021 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.018
- STEADY = -0.001
- VOLATILE = -0.005
- DEGEN = -0.037

DNA story: strongest in **DEGEN** (z=-0.037); weakest in STEADY (z=-0.001); cross-bucket spread = 0.036 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.065
- F2 (2023-11-01 .. 2024-02-29) = +0.046
- F3 (2024-03-01 .. 2024-05-15) = +0.071
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.17. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_tick_count`).

### Emergent story

`norm_tick_count` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=-0.01 -> t+1=-0.02, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.003. Cross-regime: strongest in bull (z=-0.03). Cross-DNA: most pronounced in DEGEN bucket (z=-0.04). Fold consistency: fold-stable (F1=+0.06, F2=+0.05, F3=+0.07). Distributionally mean=+0.1691 std=0.9870 skew=+0.75 kurt=+1.69. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `s3_smart_bearish`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31685
- mean = +0.084078
- std  = 0.277504
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +2.998
- excess kurtosis = +6.986

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.014 |
| t-3 | indicator 3 days BEFORE event | -0.018 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.016 |
| t-0 | indicator on event day (concurrent) | -0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.004 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.013**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.019
- chop (label=1) = +0.026
- bear (label=0) = -0.069

Raw regime keys (column-as-found):
  - `label_1`: +0.026
  - `label_0`: -0.069
  - `label_2`: +0.019

Regime story: strongest absolute deviation in **bear** (z=-0.069); weakest in bull (z=+0.019); spread=0.088 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.054
- STEADY = -0.016
- VOLATILE = +0.001
- DEGEN = -0.013

DNA story: strongest in **BLUE** (z=+0.054); weakest in VOLATILE (z=+0.001); cross-bucket spread = 0.053 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.166
- F2 (2023-11-01 .. 2024-02-29) = -0.044
- F3 (2024-03-01 .. 2024-05-15) = +0.232
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.00. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_bearish`).

### Emergent story

`s3_smart_bearish` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.02 -> t-0=-0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in bear (z=-0.07). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-unstable (F1=+0.17, F2=-0.04, F3=+0.23). Distributionally mean=+0.0841 std=0.2775 skew=+3.00 kurt=+6.99. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `te_btc_imb`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = +0.000527
- std  = 0.037366
- p10 = -0.0455
- p50 = +0.0000
- p90 = +0.0471
- skew = -0.013
- excess kurtosis = +0.547

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | -0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.007 |
| t-0 | indicator on event day (concurrent) | -0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.004 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.008
- chop (label=1) = -0.016
- bear (label=0) = +0.017

Raw regime keys (column-as-found):
  - `label_1`: -0.016
  - `label_2`: -0.008
  - `label_0`: +0.017

Regime story: strongest absolute deviation in **bear** (z=+0.017); weakest in bull (z=-0.008); spread=0.025 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.043
- STEADY = -0.008
- VOLATILE = -0.005
- DEGEN = +0.023

DNA story: strongest in **BLUE** (z=-0.043); weakest in VOLATILE (z=-0.005); cross-bucket spread = 0.038 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.047
- F2 (2023-11-01 .. 2024-02-29) = -0.046
- F3 (2024-03-01 .. 2024-05-15) = -0.198
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.54. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**21 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_btc_imb`).

### Emergent story

`te_btc_imb` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.01 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in bear (z=+0.02). Cross-DNA: most pronounced in BLUE bucket (z=-0.04). Fold consistency: fold-unstable (F1=+0.05, F2=-0.05, F3=-0.20). Distributionally mean=+0.0005 std=0.0374 skew=-0.01 kurt=+0.55. used by 21 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_cs_spread`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.133261
- std  = 0.950822
- p10 = -1.1969
- p50 = +0.3477
- p90 = +1.1498
- skew = -1.047
- excess kurtosis = +1.129

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.006 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.027
- chop (label=1) = +0.038
- bear (label=0) = -0.097

Raw regime keys (column-as-found):
  - `label_0`: -0.097
  - `label_2`: +0.027
  - `label_1`: +0.038

Regime story: strongest absolute deviation in **bear** (z=-0.097); weakest in bull (z=+0.027); spread=0.125 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.042
- STEADY = +0.002
- VOLATILE = -0.011
- DEGEN = +0.020

DNA story: strongest in **BLUE** (z=-0.042); weakest in STEADY (z=+0.002); cross-bucket spread = 0.043 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.003
- F2 (2023-11-01 .. 2024-02-29) = +0.020
- F3 (2024-03-01 .. 2024-05-15) = +0.026
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.58. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_cs_spread`).

### Emergent story

`norm_cs_spread` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.01 -> t-0=-0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.003. Cross-regime: strongest in bear (z=-0.10). Cross-DNA: most pronounced in BLUE bucket (z=-0.04). Fold consistency: fold-stable (F1=+0.00, F2=+0.02, F3=+0.03). Distributionally mean=+0.1333 std=0.9508 skew=-1.05 kurt=+1.13. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_stable_shock_strong`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.015251
- std  = 0.122550
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +7.911
- excess kurtosis = +60.584

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.020
- chop (label=1) = -0.023
- bear (label=0) = +0.005

Raw regime keys (column-as-found):
  - `label_2`: +0.020
  - `label_1`: -0.023
  - `label_0`: +0.005

Regime story: strongest absolute deviation in **chop** (z=-0.023); weakest in bear (z=+0.005); spread=0.028 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.003
- STEADY = -0.008
- VOLATILE = -0.001
- DEGEN = +0.033

DNA story: strongest in **DEGEN** (z=+0.033); weakest in VOLATILE (z=-0.001); cross-bucket spread = 0.034 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.024
- F2 (2023-11-01 .. 2024-02-29) = -0.119
- F3 (2024-03-01 .. 2024-05-15) = +0.104
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 30.74. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_stable_shock_strong`).

### Emergent story

`stbl_stable_shock_strong` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in chop (z=-0.02). Cross-DNA: most pronounced in DEGEN bucket (z=+0.03). Fold consistency: fold-unstable (F1=+0.02, F2=-0.12, F3=+0.10). Distributionally mean=+0.0153 std=0.1226 skew=+7.91 kurt=+60.58. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_perm_entropy`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.068831
- std  = 1.245158
- p10 = -1.8760
- p50 = +0.2313
- p90 = +1.2423
- skew = -0.949
- excess kurtosis = +0.612

Shape: left-skewed (downside-spike biased); modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.009 |
| t-3 | indicator 3 days BEFORE event | -0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.007 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.009 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.008
- chop (label=1) = +0.019
- bear (label=0) = -0.006

Raw regime keys (column-as-found):
  - `label_1`: +0.019
  - `label_0`: -0.006
  - `label_2`: -0.008

Regime story: strongest absolute deviation in **chop** (z=+0.019); weakest in bear (z=-0.006); spread=0.025 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.012
- STEADY = +0.009
- VOLATILE = +0.008
- DEGEN = -0.028

DNA story: strongest in **DEGEN** (z=-0.028); weakest in VOLATILE (z=+0.008); cross-bucket spread = 0.036 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.012
- F2 (2023-11-01 .. 2024-02-29) = +0.020
- F3 (2024-03-01 .. 2024-05-15) = -0.010
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 35.29. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_perm_entropy`).

### Emergent story

`norm_perm_entropy` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=+0.01 -> t-0=+0.00 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.003. Cross-regime: strongest in chop (z=+0.02). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-unstable (F1=-0.01, F2=+0.02, F3=-0.01). Distributionally mean=-0.0688 std=1.2452 skew=-0.95 kurt=+0.61. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_smart_extreme_long`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31685
- mean = +0.016064
- std  = 0.125723
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +7.698
- excess kurtosis = +57.266

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.013 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | -0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.015 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.005
- chop (label=1) = -0.008
- bear (label=0) = -0.009

Raw regime keys (column-as-found):
  - `label_0`: -0.009
  - `label_2`: +0.005
  - `label_1`: -0.008

Regime story: strongest absolute deviation in **bear** (z=-0.009); weakest in bull (z=+0.005); spread=0.014 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = +0.010
- VOLATILE = -0.006
- DEGEN = -0.030

DNA story: strongest in **DEGEN** (z=-0.030); weakest in BLUE (z=+0.000); cross-bucket spread = 0.030 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.078
- F2 (2023-11-01 .. 2024-02-29) = -0.001
- F3 (2024-03-01 .. 2024-05-15) = +0.489
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.84. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_extreme_long`).

### Emergent story

`s3_smart_extreme_long` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.01 -> t-0=-0.00 -> t+1=-0.02, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in bear (z=-0.01). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-unstable (F1=-0.08, F2=-0.00, F3=+0.49). Distributionally mean=+0.0161 std=0.1257 skew=+7.70 kurt=+57.27. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_total_depth_l5_p10`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +104700632.756618
- std  = 560776656.006877
- p10 = +60829.5096
- p50 = +1885153.9000
- p90 = +43834129.8000
- skew = +7.670
- excess kurtosis = +63.825

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.023 |
| t-3 | indicator 3 days BEFORE event | +0.028 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.033 |
| t-0 | indicator on event day (concurrent) | +0.040 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.026 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.030**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.037
- chop (label=1) = +0.113
- bear (label=0) = +0.073

Raw regime keys (column-as-found):
  - `label_2`: -0.037
  - `label_1`: +0.113
  - `label_0`: +0.073

Regime story: strongest absolute deviation in **chop** (z=+0.113); weakest in bull (z=-0.037); spread=0.150 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.095
- STEADY = +0.020
- VOLATILE = +0.042
- DEGEN = +0.049

DNA story: strongest in **BLUE** (z=+0.095); weakest in STEADY (z=+0.020); cross-bucket spread = 0.075 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.867
- F2 (2023-11-01 .. 2024-02-29) = -0.066
- F3 (2024-03-01 .. 2024-05-15) = -0.708
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 20.89. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_total_depth_l5_p10`).

### Emergent story

`bd_total_depth_l5_p10` is a low-signal background: top-25%-mover z-lift goes t-3=+0.03 -> t-1=+0.03 -> t-0=+0.04 -> t+1=+0.03, while bottom-25% movers sit at t-1=+0.03 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.003. Cross-regime: strongest in chop (z=+0.11). Cross-DNA: most pronounced in BLUE bucket (z=+0.10). Fold consistency: fold-unstable (F1=+0.87, F2=-0.07, F3=-0.71). Distributionally mean=+104700632.7566 std=560776656.0069 skew=+7.67 kurt=+63.83. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_stable_shock`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.037587
- std  = 0.190195
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +4.863
- excess kurtosis = +21.644

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.003**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.057
- chop (label=1) = -0.052
- bear (label=0) = -0.023

Raw regime keys (column-as-found):
  - `label_1`: -0.052
  - `label_0`: -0.023
  - `label_2`: +0.057

Regime story: strongest absolute deviation in **bull** (z=+0.057); weakest in bear (z=-0.023); spread=0.081 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.030
- STEADY = +0.007
- VOLATILE = -0.008
- DEGEN = +0.000

DNA story: strongest in **BLUE** (z=+0.030); weakest in DEGEN (z=+0.000); cross-bucket spread = 0.029 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.017
- F2 (2023-11-01 .. 2024-02-29) = -0.065
- F3 (2024-03-01 .. 2024-05-15) = +0.139
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.63. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_stable_shock`).

### Emergent story

`stbl_stable_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.003. Cross-regime: strongest in bull (z=+0.06). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-unstable (F1=-0.02, F2=-0.07, F3=+0.14). Distributionally mean=+0.0376 std=0.1902 skew=+4.86 kurt=+21.64. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_vol_cluster`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.006151
- std  = 1.130458
- p10 = -1.2980
- p50 = -0.1806
- p90 = +1.5842
- skew = +0.633
- excess kurtosis = +0.192

Shape: right-skewed (upside-spike biased); near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.005 |
| t-3 | indicator 3 days BEFORE event | +0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.002 |
| t-0 | indicator on event day (concurrent) | +0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.015 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.025
- chop (label=1) = +0.018
- bear (label=0) = +0.031

Raw regime keys (column-as-found):
  - `label_2`: -0.025
  - `label_0`: +0.031
  - `label_1`: +0.018

Regime story: strongest absolute deviation in **bear** (z=+0.031); weakest in chop (z=+0.018); spread=0.013 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.013
- STEADY = +0.010
- VOLATILE = +0.001
- DEGEN = -0.001

DNA story: strongest in **BLUE** (z=+0.013); weakest in DEGEN (z=-0.001); cross-bucket spread = 0.014 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.020
- F2 (2023-11-01 .. 2024-02-29) = -0.012
- F3 (2024-03-01 .. 2024-05-15) = +0.017
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.77. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_vol_cluster`).

### Emergent story

`norm_vol_cluster` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.00 -> t-0=+0.00 -> t+1=+0.02, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.002. Cross-regime: strongest in bear (z=+0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.01). Fold consistency: fold-unstable (F1=+0.02, F2=-0.01, F3=+0.02). Distributionally mean=-0.0062 std=1.1305 skew=+0.63 kurt=+0.19. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_btc_etf_mega_outflow`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.008295
- std  = 0.090696
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +10.843
- excess kurtosis = +115.570

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.008 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | -0.004 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.000**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.101
- chop (label=1) = -0.091
- bear (label=0) = -0.090

Raw regime keys (column-as-found):
  - `label_1`: -0.091
  - `label_0`: -0.090
  - `label_2`: +0.101

Regime story: strongest absolute deviation in **bull** (z=+0.101); weakest in bear (z=-0.090); spread=0.191 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.089
- STEADY = -0.060
- VOLATILE = +0.005
- DEGEN = +0.041

DNA story: strongest in **BLUE** (z=-0.089); weakest in VOLATILE (z=+0.005); cross-bucket spread = 0.094 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = -0.089
- F3 (2024-03-01 .. 2024-05-15) = +0.050
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 3.53. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_mega_outflow`).

### Emergent story

`etf_btc_etf_mega_outflow` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.002. Cross-regime: strongest in bull (z=+0.10). Cross-DNA: most pronounced in BLUE bucket (z=-0.09). Fold consistency: fold-unstable (F1=+nan, F2=-0.09, F3=+0.05). Distributionally mean=+0.0083 std=0.0907 skew=+10.84 kurt=+115.57. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `bs_basis_frenzy`

**Family**: `bs_*` — futures basis / term-structure (basis_pct, basis_z)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 51420
- mean = +0.002606
- std  = 0.050982
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +19.512
- excess kurtosis = +378.734

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.004 |
| t-3 | indicator 3 days BEFORE event | -0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | -0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.017 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.007
- chop (label=1) = -0.016
- bear (label=0) = +0.011

Raw regime keys (column-as-found):
  - `label_0`: +0.011
  - `label_1`: -0.016
  - `label_2`: -0.007

Regime story: strongest absolute deviation in **chop** (z=-0.016); weakest in bull (z=-0.007); spread=0.009 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = -0.014
- VOLATILE = +0.003
- DEGEN = -0.005

DNA story: strongest in **STEADY** (z=-0.014); weakest in BLUE (z=+0.000); cross-bucket spread = 0.014 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.017
- F2 (2023-11-01 .. 2024-02-29) = -0.023
- F3 (2024-03-01 .. 2024-05-15) = -0.027
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.18. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bs_basis_frenzy`).

### Emergent story

`bs_basis_frenzy` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=-0.00 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in chop (z=-0.02). Cross-DNA: most pronounced in STEADY bucket (z=-0.01). Fold consistency: fold-stable (F1=-0.02, F2=-0.02, F3=-0.03). Distributionally mean=+0.0026 std=0.0510 skew=+19.51 kurt=+378.73. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_oi_change`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.002902
- std  = 0.611544
- p10 = -0.5123
- p50 = +0.0000
- p90 = +0.5263
- skew = -0.163
- excess kurtosis = +10.366

Shape: approximately symmetric; extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.002 |
| t-3 | indicator 3 days BEFORE event | -0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.003 |
| t-0 | indicator on event day (concurrent) | -0.008 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.000**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.023
- chop (label=1) = -0.021
- bear (label=0) = -0.038

Raw regime keys (column-as-found):
  - `label_1`: -0.021
  - `label_0`: -0.038
  - `label_2`: +0.023

Regime story: strongest absolute deviation in **bear** (z=-0.038); weakest in chop (z=-0.021); spread=0.017 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.001
- STEADY = +0.000
- VOLATILE = -0.014
- DEGEN = -0.009

DNA story: strongest in **VOLATILE** (z=-0.014); weakest in STEADY (z=+0.000); cross-bucket spread = 0.014 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.060
- F2 (2023-11-01 .. 2024-02-29) = +0.015
- F3 (2024-03-01 .. 2024-05-15) = -0.082
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 27.71. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_oi_change`).

### Emergent story

`norm_oi_change` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=-0.00 -> t-0=-0.01 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in bear (z=-0.04). Cross-DNA: most pronounced in VOLATILE bucket (z=-0.01). Fold consistency: fold-unstable (F1=+0.06, F2=+0.02, F3=-0.08). Distributionally mean=+0.0029 std=0.6115 skew=-0.16 kurt=+10.37. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `te_out_btc`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = +0.074419
- std  = 0.031739
- p10 = +0.0380
- p50 = +0.0721
- p90 = +0.1159
- skew = +0.298
- excess kurtosis = +0.522

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.002 |
| t-3 | indicator 3 days BEFORE event | +0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.009
- chop (label=1) = +0.003
- bear (label=0) = -0.014

Raw regime keys (column-as-found):
  - `label_2`: +0.009
  - `label_0`: -0.014
  - `label_1`: +0.003

Regime story: strongest absolute deviation in **bear** (z=-0.014); weakest in chop (z=+0.003); spread=0.017 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.032
- STEADY = +0.008
- VOLATILE = +0.008
- DEGEN = -0.048

DNA story: strongest in **DEGEN** (z=-0.048); weakest in STEADY (z=+0.008); cross-bucket spread = 0.056 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.031
- F2 (2023-11-01 .. 2024-02-29) = +0.092
- F3 (2024-03-01 .. 2024-05-15) = +0.076
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.19. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_out_btc`).

### Emergent story

`te_out_btc` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in bear (z=-0.01). Cross-DNA: most pronounced in DEGEN bucket (z=-0.05). Fold consistency: fold-unstable (F1=-0.03, F2=+0.09, F3=+0.08). Distributionally mean=+0.0744 std=0.0317 skew=+0.30 kurt=+0.52. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_depth_l1pct_p90`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +40504127.612154
- std  = 209420391.381913
- p10 = +24132.1646
- p50 = +780832.0000
- p90 = +19553504.1200
- skew = +7.526
- excess kurtosis = +63.226

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.029 |
| t-3 | indicator 3 days BEFORE event | +0.023 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.026 |
| t-0 | indicator on event day (concurrent) | +0.032 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.013 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.025**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.010
- chop (label=1) = +0.111
- bear (label=0) = +0.001

Raw regime keys (column-as-found):
  - `label_0`: +0.001
  - `label_2`: -0.010
  - `label_1`: +0.111

Regime story: strongest absolute deviation in **chop** (z=+0.111); weakest in bear (z=+0.001); spread=0.110 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.091
- STEADY = -0.002
- VOLATILE = +0.035
- DEGEN = +0.047

DNA story: strongest in **BLUE** (z=+0.091); weakest in STEADY (z=-0.002); cross-bucket spread = 0.093 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.782
- F2 (2023-11-01 .. 2024-02-29) = -0.212
- F3 (2024-03-01 .. 2024-05-15) = -0.539
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 52.91. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_depth_l1pct_p90`).

### Emergent story

`bd_depth_l1pct_p90` is a low-signal background: top-25%-mover z-lift goes t-3=+0.02 -> t-1=+0.03 -> t-0=+0.03 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.02 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.002. Cross-regime: strongest in chop (z=+0.11). Cross-DNA: most pronounced in BLUE bucket (z=+0.09). Fold consistency: fold-unstable (F1=+0.78, F2=-0.21, F3=-0.54). Distributionally mean=+40504127.6122 std=209420391.3819 skew=+7.53 kurt=+63.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_vpin`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.174285
- std  = 0.878596
- p10 = -1.1558
- p50 = -0.3292
- p90 = +1.0857
- skew = +0.842
- excess kurtosis = +0.466

Shape: right-skewed (upside-spike biased); near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.014 |
| t-3 | indicator 3 days BEFORE event | +0.003 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.005 |
| t-0 | indicator on event day (concurrent) | +0.006 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.014 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.037
- chop (label=1) = -0.008
- bear (label=0) = -0.022

Raw regime keys (column-as-found):
  - `label_1`: -0.008
  - `label_2`: +0.037
  - `label_0`: -0.022

Regime story: strongest absolute deviation in **bull** (z=+0.037); weakest in chop (z=-0.008); spread=0.044 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.052
- STEADY = +0.013
- VOLATILE = +0.009
- DEGEN = +0.005

DNA story: strongest in **BLUE** (z=-0.052); weakest in DEGEN (z=+0.005); cross-bucket spread = 0.057 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.027
- F2 (2023-11-01 .. 2024-02-29) = -0.025
- F3 (2024-03-01 .. 2024-05-15) = +0.061
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 13.40. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_vpin`).

### Emergent story

`norm_vpin` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.00 -> t-0=+0.01 -> t+1=+0.01, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in bull (z=+0.04). Cross-DNA: most pronounced in BLUE bucket (z=-0.05). Fold consistency: fold-unstable (F1=-0.03, F2=-0.02, F3=+0.06). Distributionally mean=-0.1743 std=0.8786 skew=+0.84 kurt=+0.47. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_usdt_shock`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.063828
- std  = 0.244447
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +3.569
- excess kurtosis = +10.735

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | -0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.002**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.033
- chop (label=1) = -0.031
- bear (label=0) = -0.016

Raw regime keys (column-as-found):
  - `label_0`: -0.016
  - `label_2`: +0.033
  - `label_1`: -0.031

Regime story: strongest absolute deviation in **bull** (z=+0.033); weakest in bear (z=-0.016); spread=0.049 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.012
- STEADY = -0.003
- VOLATILE = +0.003
- DEGEN = -0.008

DNA story: strongest in **BLUE** (z=+0.012); weakest in VOLATILE (z=+0.003); cross-bucket spread = 0.009 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.002
- F2 (2023-11-01 .. 2024-02-29) = +0.128
- F3 (2024-03-01 .. 2024-05-15) = +0.041
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.93. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_usdt_shock`).

### Emergent story

`stbl_usdt_shock` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.002. Cross-regime: strongest in bull (z=+0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.01). Fold consistency: fold-stable (F1=+0.00, F2=+0.13, F3=+0.04). Distributionally mean=+0.0638 std=0.2444 skew=+3.57 kurt=+10.74. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `stbl_dai_zscore_30d`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.002456
- std  = 2.071761
- p10 = -1.1531
- p50 = +0.0199
- p90 = +1.0609
- skew = +11.907
- excess kurtosis = +302.398

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | -0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.048
- chop (label=1) = -0.056
- bear (label=0) = -0.002

Raw regime keys (column-as-found):
  - `label_1`: -0.056
  - `label_2`: +0.048
  - `label_0`: -0.002

Regime story: strongest absolute deviation in **chop** (z=-0.056); weakest in bear (z=-0.002); spread=0.054 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.013
- STEADY = -0.016
- VOLATILE = +0.007
- DEGEN = +0.012

DNA story: strongest in **STEADY** (z=-0.016); weakest in VOLATILE (z=+0.007); cross-bucket spread = 0.023 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.089
- F2 (2023-11-01 .. 2024-02-29) = -0.018
- F3 (2024-03-01 .. 2024-05-15) = -0.070
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 247.18. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_dai_zscore_30d`).

### Emergent story

`stbl_dai_zscore_30d` is a low-signal background: top-25%-mover z-lift goes t-3=-0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in chop (z=-0.06). Cross-DNA: most pronounced in STEADY bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.09, F2=-0.02, F3=-0.07). Distributionally mean=-0.0025 std=2.0718 skew=+11.91 kurt=+302.40. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_usdc_zscore_30d`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.051915
- std  = 1.500940
- p10 = -1.1785
- p50 = -0.0087
- p90 = +1.2782
- skew = +6.355
- excess kurtosis = +130.231

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.001 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.003**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.002**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.022
- chop (label=1) = -0.009
- bear (label=0) = -0.020

Raw regime keys (column-as-found):
  - `label_2`: +0.022
  - `label_0`: -0.020
  - `label_1`: -0.009

Regime story: strongest absolute deviation in **bull** (z=+0.022); weakest in chop (z=-0.009); spread=0.031 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.019
- STEADY = +0.006
- VOLATILE = +0.005
- DEGEN = -0.025

DNA story: strongest in **DEGEN** (z=-0.025); weakest in VOLATILE (z=+0.005); cross-bucket spread = 0.030 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.045
- F2 (2023-11-01 .. 2024-02-29) = +0.015
- F3 (2024-03-01 .. 2024-05-15) = -0.019
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.52. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_usdc_zscore_30d`).

### Emergent story

`stbl_usdc_zscore_30d` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.002. Cross-regime: strongest in bull (z=+0.02). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-unstable (F1=-0.04, F2=+0.02, F3=-0.02). Distributionally mean=+0.0519 std=1.5009 skew=+6.36 kurt=+130.23. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_btc_etf_outflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.016153
- std  = 0.126062
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +7.676
- excess kurtosis = +56.926

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | +0.004 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | -0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.001**
- Top vs Bot symmetry: ANTI-SYMMETRIC: top and bot push in opposite directions (directional discriminator).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.013
- chop (label=1) = -0.128
- bear (label=0) = +0.090

Raw regime keys (column-as-found):
  - `label_0`: +0.090
  - `label_1`: -0.128
  - `label_2`: +0.013

Regime story: strongest absolute deviation in **chop** (z=-0.128); weakest in bull (z=+0.013); spread=0.141 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.126
- STEADY = -0.065
- VOLATILE = +0.006
- DEGEN = +0.058

DNA story: strongest in **BLUE** (z=-0.126); weakest in VOLATILE (z=+0.006); cross-bucket spread = 0.132 z. DNA-conditional signal (different asset classes respond differently).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = -0.126
- F3 (2024-03-01 .. 2024-05-15) = +0.077
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.10. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_outflow_shock`).

### Emergent story

`etf_btc_etf_outflow_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (anti-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in chop (z=-0.13). Cross-DNA: most pronounced in BLUE bucket (z=-0.13). Fold consistency: fold-unstable (F1=+nan, F2=-0.13, F3=+0.08). Distributionally mean=+0.0162 std=0.1261 skew=+7.68 kurt=+56.93. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot anti-symmetric - long/short directional rule is well-posed on this signal.

---

## `te_imb`

**Family**: `te_*` — transfer-entropy cross-asset information flow

### Distribution (TRAIN window, all assets pooled)

- n_observations = 53666
- mean = -0.001193
- std  = 0.033270
- p10 = -0.0433
- p50 = -0.0011
- p90 = +0.0405
- skew = +0.045
- excess kurtosis = +0.362

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.007 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.003 |
| t-0 | indicator on event day (concurrent) | +0.003 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.002**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.024
- chop (label=1) = -0.016
- bear (label=0) = -0.005

Raw regime keys (column-as-found):
  - `label_0`: -0.005
  - `label_1`: -0.016
  - `label_2`: +0.024

Regime story: strongest absolute deviation in **bull** (z=+0.024); weakest in bear (z=-0.005); spread=0.029 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.011
- STEADY = +0.009
- VOLATILE = -0.005
- DEGEN = +0.019

DNA story: strongest in **DEGEN** (z=+0.019); weakest in VOLATILE (z=-0.005); cross-bucket spread = 0.024 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.012
- F2 (2023-11-01 .. 2024-02-29) = +0.018
- F3 (2024-03-01 .. 2024-05-15) = -0.091
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.44. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**19 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/te_imb`).

### Emergent story

`te_imb` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in bull (z=+0.02). Cross-DNA: most pronounced in DEGEN bucket (z=+0.02). Fold consistency: fold-unstable (F1=+0.01, F2=+0.02, F3=-0.09). Distributionally mean=-0.0012 std=0.0333 skew=+0.04 kurt=+0.36. used by 19 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_hl_spread`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.084500
- std  = 0.923266
- p10 = -1.0693
- p50 = +0.0425
- p90 = +1.2556
- skew = +0.364
- excess kurtosis = +0.807

Shape: approximately symmetric; modestly leptokurtic.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.006 |
| t-3 | indicator 3 days BEFORE event | +0.005 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.007 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.009 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.008**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.068
- chop (label=1) = +0.022
- bear (label=0) = +0.092

Raw regime keys (column-as-found):
  - `label_1`: +0.022
  - `label_0`: +0.092
  - `label_2`: -0.068

Regime story: strongest absolute deviation in **bear** (z=+0.092); weakest in chop (z=+0.022); spread=0.070 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.060
- STEADY = -0.008
- VOLATILE = +0.009
- DEGEN = -0.000

DNA story: strongest in **BLUE** (z=+0.060); weakest in DEGEN (z=-0.000); cross-bucket spread = 0.061 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.050
- F2 (2023-11-01 .. 2024-02-29) = +0.002
- F3 (2024-03-01 .. 2024-05-15) = -0.151
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.96. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_hl_spread`).

### Emergent story

`norm_hl_spread` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.01 -> t-0=+0.00 -> t+1=+0.01, while bottom-25% movers sit at t-1=+0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.001. Cross-regime: strongest in bear (z=+0.09). Cross-DNA: most pronounced in BLUE bucket (z=+0.06). Fold consistency: fold-unstable (F1=-0.05, F2=+0.00, F3=-0.15). Distributionally mean=+0.0845 std=0.9233 skew=+0.36 kurt=+0.81. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_usde_zscore_30d`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 8355
- mean = +0.632501
- std  = 2.689770
- p10 = -0.8340
- p50 = -0.2408
- p90 = +2.4059
- skew = +4.417
- excess kurtosis = +24.675

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | -0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.006 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.041
- chop (label=1) = -0.015
- bear (label=0) = +0.071

Raw regime keys (column-as-found):
  - `label_2`: -0.041
  - `label_0`: +0.071
  - `label_1`: -0.015

Regime story: strongest absolute deviation in **bear** (z=+0.071); weakest in chop (z=-0.015); spread=0.086 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.033
- STEADY = -0.030
- VOLATILE = -0.011
- DEGEN = +0.041

DNA story: strongest in **DEGEN** (z=+0.041); weakest in VOLATILE (z=-0.011); cross-bucket spread = 0.053 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.174
- F3 (2024-03-01 .. 2024-05-15) = -0.176
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 191.22. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_usde_zscore_30d`).

### Emergent story

`stbl_usde_zscore_30d` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=-0.00 -> t+1=-0.01, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in bear (z=+0.07). Cross-DNA: most pronounced in DEGEN bucket (z=+0.04). Fold consistency: fold-unstable (F1=+nan, F2=+0.17, F3=-0.18). Distributionally mean=+0.6325 std=2.6898 skew=+4.42 kurt=+24.68. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_flow_persistence`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -0.080565
- std  = 1.081208
- p10 = -1.4312
- p50 = -0.1283
- p90 = +1.3390
- skew = +0.215
- excess kurtosis = +0.048

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.006 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | -0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.017 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.002**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.031
- chop (label=1) = +0.022
- bear (label=0) = +0.017

Raw regime keys (column-as-found):
  - `label_1`: +0.022
  - `label_0`: +0.017
  - `label_2`: -0.031

Regime story: strongest absolute deviation in **bull** (z=-0.031); weakest in bear (z=+0.017); spread=0.048 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.020
- STEADY = -0.016
- VOLATILE = +0.005
- DEGEN = +0.006

DNA story: strongest in **BLUE** (z=+0.020); weakest in VOLATILE (z=+0.005); cross-bucket spread = 0.016 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.011
- F2 (2023-11-01 .. 2024-02-29) = -0.005
- F3 (2024-03-01 .. 2024-05-15) = +0.056
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.26. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_flow_persistence`).

### Emergent story

`norm_flow_persistence` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=-0.00 -> t+1=+0.02, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.001. Cross-regime: strongest in bull (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.02). Fold consistency: fold-unstable (F1=+0.01, F2=-0.01, F3=+0.06). Distributionally mean=-0.0806 std=1.0812 skew=+0.22 kurt=+0.05. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_total_zscore_30d`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.022307
- std  = 1.717206
- p10 = -1.1448
- p50 = -0.0374
- p90 = +1.2377
- skew = +9.796
- excess kurtosis = +386.756

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.002 |
| t-0 | indicator on event day (concurrent) | +0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.025
- chop (label=1) = -0.032
- bear (label=0) = +0.002

Raw regime keys (column-as-found):
  - `label_0`: +0.002
  - `label_1`: -0.032
  - `label_2`: +0.025

Regime story: strongest absolute deviation in **chop** (z=-0.032); weakest in bear (z=+0.002); spread=0.034 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.012
- STEADY = -0.003
- VOLATILE = +0.000
- DEGEN = +0.008

DNA story: strongest in **BLUE** (z=+0.012); weakest in VOLATILE (z=+0.000); cross-bucket spread = 0.012 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.009
- F2 (2023-11-01 .. 2024-02-29) = +0.045
- F3 (2024-03-01 .. 2024-05-15) = -0.012
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 1.67. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**2 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_total_zscore_30d`).

### Emergent story

`stbl_total_zscore_30d` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in chop (z=-0.03). Cross-DNA: most pronounced in BLUE bucket (z=+0.01). Fold consistency: fold-unstable (F1=+0.01, F2=+0.05, F3=-0.01). Distributionally mean=+0.0223 std=1.7172 skew=+9.80 kurt=+386.76. used by 2 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_btc_etf_inflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.071304
- std  = 0.257332
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +3.332
- excess kurtosis = +9.101

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.015 |
| t-3 | indicator 3 days BEFORE event | +0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.062
- chop (label=1) = +0.085
- bear (label=0) = -0.170

Raw regime keys (column-as-found):
  - `label_1`: +0.085
  - `label_2`: +0.062
  - `label_0`: -0.170

Regime story: strongest absolute deviation in **bear** (z=-0.170); weakest in bull (z=+0.062); spread=0.232 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.051
- STEADY = +0.030
- VOLATILE = -0.009
- DEGEN = -0.007

DNA story: strongest in **BLUE** (z=+0.051); weakest in DEGEN (z=-0.007); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.110
- F3 (2024-03-01 .. 2024-05-15) = -0.067
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.04. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_btc_etf_inflow_shock`).

### Emergent story

`etf_btc_etf_inflow_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in bear (z=-0.17). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-unstable (F1=+nan, F2=+0.11, F3=-0.07). Distributionally mean=+0.0713 std=0.2573 skew=+3.33 kurt=+9.10. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_any_inflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.071304
- std  = 0.257332
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +3.332
- excess kurtosis = +9.101

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.015 |
| t-3 | indicator 3 days BEFORE event | +0.010 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.005 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.002 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.004**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.001**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.062
- chop (label=1) = +0.085
- bear (label=0) = -0.170

Raw regime keys (column-as-found):
  - `label_1`: +0.085
  - `label_0`: -0.170
  - `label_2`: +0.062

Regime story: strongest absolute deviation in **bear** (z=-0.170); weakest in bull (z=+0.062); spread=0.232 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.051
- STEADY = +0.030
- VOLATILE = -0.009
- DEGEN = -0.007

DNA story: strongest in **BLUE** (z=+0.051); weakest in DEGEN (z=-0.007); cross-bucket spread = 0.058 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.110
- F3 (2024-03-01 .. 2024-05-15) = -0.067
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.04. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_any_inflow_shock`).

### Emergent story

`etf_any_inflow_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.01 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.001. Cross-regime: strongest in bear (z=-0.17). Cross-DNA: most pronounced in BLUE bucket (z=+0.05). Fold consistency: fold-unstable (F1=+nan, F2=+0.11, F3=-0.07). Distributionally mean=+0.0713 std=0.2573 skew=+3.33 kurt=+9.10. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_imbalance_l1`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +1.168853
- std  = 0.148358
- p10 = +1.0153
- p50 = +1.1454
- p90 = +1.3538
- skew = +1.293
- excess kurtosis = +4.232

Shape: right-skewed (upside-spike biased); heavy-tailed (fat-tail regime).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.025 |
| t-3 | indicator 3 days BEFORE event | -0.012 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.002 |
| t-0 | indicator on event day (concurrent) | +0.005 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.047 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.026
- chop (label=1) = +0.010
- bear (label=0) = +0.049

Raw regime keys (column-as-found):
  - `label_1`: +0.010
  - `label_2`: -0.026
  - `label_0`: +0.049

Regime story: strongest absolute deviation in **bear** (z=+0.049); weakest in chop (z=+0.010); spread=0.039 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.038
- STEADY = +0.015
- VOLATILE = -0.018
- DEGEN = +0.054

DNA story: strongest in **DEGEN** (z=+0.054); weakest in STEADY (z=+0.015); cross-bucket spread = 0.039 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = -0.170
- F2 (2023-11-01 .. 2024-02-29) = +0.053
- F3 (2024-03-01 .. 2024-05-15) = +0.439
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.34. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**5 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_imbalance_l1`).

### Emergent story

`bd_imbalance_l1` is a low-signal background: top-25%-mover z-lift goes t-3=-0.01 -> t-1=-0.00 -> t-0=+0.01 -> t+1=-0.05, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.000. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in DEGEN bucket (z=+0.05). Fold consistency: fold-unstable (F1=-0.17, F2=+0.05, F3=+0.44). Distributionally mean=+1.1689 std=0.1484 skew=+1.29 kurt=+4.23. used by 5 catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_stable_crash`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.031635
- std  = 0.175027
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +5.352
- excess kurtosis = +26.643

Shape: right-skewed (upside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.001 |
| t-0 | indicator on event day (concurrent) | -0.001 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = -0.033
- chop (label=1) = -0.003
- bear (label=0) = +0.049

Raw regime keys (column-as-found):
  - `label_1`: -0.003
  - `label_2`: -0.033
  - `label_0`: +0.049

Regime story: strongest absolute deviation in **bear** (z=+0.049); weakest in chop (z=-0.003); spread=0.052 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.021
- STEADY = -0.005
- VOLATILE = +0.005
- DEGEN = -0.001

DNA story: strongest in **BLUE** (z=-0.021); weakest in DEGEN (z=-0.001); cross-bucket spread = 0.019 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.052
- F2 (2023-11-01 .. 2024-02-29) = -0.042
- F3 (2024-03-01 .. 2024-05-15) = +0.037
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 2.63. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_stable_crash`).

### Emergent story

`stbl_stable_crash` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=-0.00 -> t-0=-0.00 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.000. Cross-regime: strongest in bear (z=+0.05). Cross-DNA: most pronounced in BLUE bucket (z=-0.02). Fold consistency: fold-unstable (F1=+0.05, F2=-0.04, F3=+0.04). Distributionally mean=+0.0316 std=0.1750 skew=+5.35 kurt=+26.64. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `stbl_usdt_zscore_30d`

**Family**: `stbl_*` — stablecoin flow / depeg / crash signals

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = -116.858451
- std  = 4534.432055
- p10 = -0.8467
- p50 = -0.1765
- p90 = +1.4085
- skew = -40.369
- excess kurtosis = +1630.854

Shape: left-skewed (downside-spike biased); extreme heavy-tailed (event-driven spikes dominate).

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.001 |
| t-3 | indicator 3 days BEFORE event | +0.002 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.001 |
| t-0 | indicator on event day (concurrent) | +0.002 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.001 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.001**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.047
- chop (label=1) = -0.055
- bear (label=0) = +0.002

Raw regime keys (column-as-found):
  - `label_0`: +0.002
  - `label_1`: -0.055
  - `label_2`: +0.047

Regime story: strongest absolute deviation in **chop** (z=-0.055); weakest in bear (z=+0.002); spread=0.057 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.026
- STEADY = +0.002
- VOLATILE = +0.002
- DEGEN = -0.005

DNA story: strongest in **BLUE** (z=+0.026); weakest in VOLATILE (z=+0.002); cross-bucket spread = 0.025 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.034
- F2 (2023-11-01 .. 2024-02-29) = +0.036
- F3 (2024-03-01 .. 2024-05-15) = +0.004
- **sign_consistent_3_fold = True**

Cross-fold magnitude variability: std/|mean| = 0.60. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/stbl_usdt_zscore_30d`).

### Emergent story

`stbl_usdt_zscore_30d` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.000. Cross-regime: strongest in chop (z=-0.05). Cross-DNA: most pronounced in BLUE bucket (z=+0.03). Fold consistency: fold-stable (F1=+0.03, F2=+0.04, F3=+0.00). Distributionally mean=-116.8585 std=4534.4321 skew=-40.37 kurt=+1630.85. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-stable across F1/F2/F3 - safe to deploy without regime-specific gating.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `norm_return_16`

**Family**: `norm_*` — normalized core feature (legacy v50 normalized indicator)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 59143
- mean = +0.035803
- std  = 0.949029
- p10 = -1.1749
- p50 = +0.0388
- p90 = +1.2418
- skew = +0.015
- excess kurtosis = +0.160

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | -0.001 |
| t-3 | indicator 3 days BEFORE event | -0.015 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | -0.006 |
| t-0 | indicator on event day (concurrent) | -0.022 |
| t+1 | indicator 1 day AFTER event (lag/drift) | -0.004 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **-0.005**
- Top vs Bot symmetry: CO-SYMMETRIC: top and bot both deviate the SAME direction (magnitude indicator, not direction).

**Discriminator score (top_t-1 - bot_t-1)** = **-0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.164
- chop (label=1) = -0.012
- bear (label=0) = -0.308

Raw regime keys (column-as-found):
  - `label_0`: -0.308
  - `label_2`: +0.164
  - `label_1`: -0.012

Regime story: strongest absolute deviation in **bear** (z=-0.308); weakest in chop (z=-0.012); spread=0.295 z. Regime-conditional signal.

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = -0.025
- STEADY = -0.024
- VOLATILE = -0.019
- DEGEN = -0.028

DNA story: strongest in **DEGEN** (z=-0.028); weakest in VOLATILE (z=-0.019); cross-bucket spread = 0.009 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.103
- F2 (2023-11-01 .. 2024-02-29) = -0.023
- F3 (2024-03-01 .. 2024-05-15) = -0.157
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 4.11. (Magnitudes vary widely across folds - signal strength regime-dependent.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/norm_return_16`).

### Emergent story

`norm_return_16` is a low-signal background: top-25%-mover z-lift goes t-3=-0.02 -> t-1=-0.01 -> t-0=-0.02 -> t+1=-0.00, while bottom-25% movers sit at t-1=-0.01 (co-symmetric top vs bot). Discriminator (top_t-1 - bot_t-1) = -0.000. Cross-regime: strongest in bear (z=-0.31). Cross-DNA: most pronounced in DEGEN bucket (z=-0.03). Fold consistency: fold-unstable (F1=+0.10, F2=-0.02, F3=-0.16). Distributionally mean=+0.0358 std=0.9490 skew=+0.02 kurt=+0.16. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `s3_smart_extreme_short`

**Family**: `s3_*` — smart-money vs retail signal (taker LSR, smart wallet OI)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 31685
- mean = +0.000000
- std  = 0.000000
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +0.000
- excess kurtosis = +0.000

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.000**
- Top vs Bot symmetry: asymmetric (one side ~0).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.000
- chop (label=1) = +0.000
- bear (label=0) = +0.000

Raw regime keys (column-as-found):
  - `label_0`: +0.000
  - `label_2`: +0.000
  - `label_1`: +0.000

Regime story: strongest absolute deviation in **bull** (z=+0.000); weakest in bull (z=+0.000); spread=0.000 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = +0.000
- VOLATILE = +0.000
- DEGEN = +0.000

DNA story: strongest in **BLUE** (z=+0.000); weakest in BLUE (z=+0.000); cross-bucket spread = 0.000 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.000
- F2 (2023-11-01 .. 2024-02-29) = +0.000
- F3 (2024-03-01 .. 2024-05-15) = +0.000
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.00. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/s3_smart_extreme_short`).

### Emergent story

`s3_smart_extreme_short` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (one-sided top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.000. Cross-regime: strongest in bull (z=+0.00). Cross-DNA: most pronounced in BLUE bucket (z=+0.00). Fold consistency: fold-unstable (F1=+0.00, F2=+0.00, F3=+0.00). Distributionally mean=+0.0000 std=0.0000 skew=+0.00 kurt=+0.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_both_inflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 6872
- mean = +0.000000
- std  = 0.000000
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +0.000
- excess kurtosis = +0.000

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.000**
- Top vs Bot symmetry: asymmetric (one side ~0).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.000
- chop (label=1) = +0.000
- bear (label=0) = +0.000

Raw regime keys (column-as-found):
  - `label_2`: +0.000
  - `label_1`: +0.000
  - `label_0`: +0.000

Regime story: strongest absolute deviation in **bull** (z=+0.000); weakest in bull (z=+0.000); spread=0.000 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = +0.000
- VOLATILE = +0.000
- DEGEN = +0.000

DNA story: strongest in **BLUE** (z=+0.000); weakest in BLUE (z=+0.000); cross-bucket spread = 0.000 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +0.000
- F3 (2024-03-01 .. 2024-05-15) = +0.000
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.00. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_both_inflow_shock`).

### Emergent story

`etf_both_inflow_shock` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (one-sided top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.000. Cross-regime: strongest in bull (z=+0.00). Cross-DNA: most pronounced in BLUE bucket (z=+0.00). Fold consistency: fold-unstable (F1=+nan, F2=+0.00, F3=+0.00). Distributionally mean=+0.0000 std=0.0000 skew=+0.00 kurt=+0.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `bd_depth_at_02pct`

**Family**: `bd_*` — book-depth (L1/L5 notional, thin-book frac, depth slopes)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 18509
- mean = +0.000000
- std  = 0.000000
- p10 = +0.0000
- p50 = +0.0000
- p90 = +0.0000
- skew = +0.000
- excess kurtosis = +0.000

Shape: approximately symmetric; near-Gaussian shape.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +0.000 |
| t-3 | indicator 3 days BEFORE event | +0.000 |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +0.000 |
| t-0 | indicator on event day (concurrent) | +0.000 |
| t+1 | indicator 1 day AFTER event (lag/drift) | +0.000 |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+0.000**
- Top vs Bot symmetry: asymmetric (one side ~0).

**Discriminator score (top_t-1 - bot_t-1)** = **+0.000**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +0.000
- chop (label=1) = +0.000
- bear (label=0) = +0.000

Raw regime keys (column-as-found):
  - `label_2`: +0.000
  - `label_0`: +0.000
  - `label_1`: +0.000

Regime story: strongest absolute deviation in **bull** (z=+0.000); weakest in bull (z=+0.000); spread=0.000 z. Regime-invariant (signal lives in dispersion, not in BTC trend).

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +0.000
- STEADY = +0.000
- VOLATILE = +0.000
- DEGEN = +0.000

DNA story: strongest in **BLUE** (z=+0.000); weakest in BLUE (z=+0.000); cross-bucket spread = 0.000 z. DNA-invariant (signal generalizes across asset classes).

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +0.000
- F2 (2023-11-01 .. 2024-02-29) = +0.000
- F3 (2024-03-01 .. 2024-05-15) = +0.000
- **sign_consistent_3_fold = False**

Cross-fold magnitude variability: std/|mean| = 0.00. (Magnitudes are tight - stable signal.)

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/bd_depth_at_02pct`).

### Emergent story

`bd_depth_at_02pct` is a low-signal background: top-25%-mover z-lift goes t-3=+0.00 -> t-1=+0.00 -> t-0=+0.00 -> t+1=+0.00, while bottom-25% movers sit at t-1=+0.00 (one-sided top vs bot). Discriminator (top_t-1 - bot_t-1) = +0.000. Cross-regime: strongest in bull (z=+0.00). Cross-DNA: most pronounced in BLUE bucket (z=+0.00). Fold consistency: fold-unstable (F1=+0.00, F2=+0.00, F3=+0.00). Distributionally mean=+0.0000 std=0.0000 skew=+0.00 kurt=+0.00. not yet exploited in catalog engines.

### Playbook hint

- Low standalone discriminator; only consider as a confluence-stack input or interaction term.
- Sign-flips across sub-folds - gate by regime (BTC regime_label) or DNA bucket before deployment.
- Top/Bot co-symmetric - use as a magnitude / volatility regime detector, not a directional signal.

---

## `etf_eth_etf_total_usdm`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_eth_etf_total_usdm`).

### Emergent story

`etf_eth_etf_total_usdm` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `etf_eth_etf_total_z30`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_eth_etf_total_z30`).

### Emergent story

`etf_eth_etf_total_z30` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `etf_eth_etf_inflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_eth_etf_inflow_shock`).

### Emergent story

`etf_eth_etf_inflow_shock` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `etf_eth_etf_outflow_shock`

**Family**: `etf_*` — spot ETF flow (BTC / ETH ETF net USD, shock detectors)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/etf_eth_etf_outflow_shock`).

### Emergent story

`etf_eth_etf_outflow_shock` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_l1_imb_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_l1_imb_mean`).

### Emergent story

`lob_l1_imb_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_l1_imb_std`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_l1_imb_std`).

### Emergent story

`lob_l1_imb_std` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_l5_imb_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_l5_imb_mean`).

### Emergent story

`lob_l5_imb_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_l5_imb_std`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_l5_imb_std`).

### Emergent story

`lob_l5_imb_std` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_spread_bps_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_spread_bps_mean`).

### Emergent story

`lob_spread_bps_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_spread_bps_p90`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_spread_bps_p90`).

### Emergent story

`lob_spread_bps_p90` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_top_pressure_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_top_pressure_mean`).

### Emergent story

`lob_top_pressure_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_count_imb_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_count_imb_mean`).

### Emergent story

`lob_count_imb_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_run_length_p50`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_run_length_p50`).

### Emergent story

`lob_run_length_p50` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_kyle_lambda_mean`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_kyle_lambda_mean`).

### Emergent story

`lob_kyle_lambda_mean` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_kyle_lambda_abs_max`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_kyle_lambda_abs_max`).

### Emergent story

`lob_kyle_lambda_abs_max` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `lob_n_bars`

**Family**: `lob_*` — limit-order-book microstructure (imbalance, Kyle lambda, spread)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/lob_n_bars`).

### Emergent story

`lob_n_bars` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `xrel_lob_kyle_lambda_mean_xrank`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_lob_kyle_lambda_mean_xrank`).

### Emergent story

`xrel_lob_kyle_lambda_mean_xrank` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `xrel_lob_kyle_lambda_mean_xpct10`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_lob_kyle_lambda_mean_xpct10`).

### Emergent story

`xrel_lob_kyle_lambda_mean_xpct10` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---

## `xrel_lob_kyle_lambda_mean_xratio`

**Family**: `xrel_*` — cross-sectional RELATIVE measure (rank / pct / ratio of feature vs panel)

### Distribution (TRAIN window, all assets pooled)

- n_observations = 0
- mean = +nan
- std  = nan
- p10 = +nan
- p50 = +nan
- p90 = +nan
- skew = +nan
- excess kurtosis = +nan

Shape: approximately symmetric; tail shape unavailable.

### Top-25%-mover signature (lead/concurrent/lag, per-asset z)

z_lift = mean of per-asset z-scored column value across all (asset, date) pairs where the asset is a top-25%-mover at the specified relative day.

| relative lag | meaning | z_lift |
|:---|:---|---:|
| t-6 | indicator 6 days BEFORE event | +nan |
| t-3 | indicator 3 days BEFORE event | +nan |
| t-1 | indicator 1 day BEFORE event (primary lead signal) | +nan |
| t-0 | indicator on event day (concurrent) | +nan |
| t+1 | indicator 1 day AFTER event (lag/drift) | +nan |

**Lead/Lag verdict**: low-signal (all lags < 0.05 z).

### Bottom-25%-mover signature

- Bottom-25%-mover z_lift at t-1 = **+nan**

**Discriminator score (top_t-1 - bot_t-1)** = **+nan**

### Per-regime top-25% lift (BTC regime_label: 0=bear, 1=chop, 2=bull)

- bull (label=2) = +nan
- chop (label=1) = +nan
- bear (label=0) = +nan

### Per-DNA-bucket top-25% lift

DNA buckets: BLUE = BTC/ETH; STEADY = top-15 large caps (SOL, BNB, etc); DEGEN = explicitly volatile alts (PEPE, BONK, SHIB, FLOKI...); VOLATILE = everything else.

- BLUE = +nan
- STEADY = +nan
- VOLATILE = +nan
- DEGEN = +nan

### Fold stability (top-25% lift per TRAIN sub-fold)

Three non-overlapping TRAIN sub-folds. Sign-consistent across all three = the indicator is not a regime artifact of one period.

- F1 (2023-07-01 .. 2023-10-31) = +nan
- F2 (2023-11-01 .. 2024-02-29) = +nan
- F3 (2024-03-01 .. 2024-05-15) = +nan
- **sign_consistent_3_fold = False**

### Catalog usage

**0 engines** in `data/oracle/engine_catalog.parquet` currently mine this column (indicator_class = `measure_engines/xrel_lob_kyle_lambda_mean_xratio`).

### Emergent story

`xrel_lob_kyle_lambda_mean_xratio` is a uncharacterized (insufficient data): top-25%-mover z-lift goes t-3=+nan -> t-1=+nan -> t-0=+nan -> t+1=+nan, while bottom-25% movers sit at t-1=+nan (asymmetric (incomplete) top vs bot). Discriminator (top_t-1 - bot_t-1) = +nan. Cross-regime: regime breakdown unavailable. Cross-DNA: DNA breakdown unavailable. Fold consistency: fold-unstable (F1=+nan, F2=+nan, F3=+nan). Distributionally mean=+nan std=nan skew=+nan kurt=+nan. not yet exploited in catalog engines.

### Playbook hint

- Insufficient data; do not consume as an engine input until backfill lands.

---
