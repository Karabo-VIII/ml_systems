# Per-Asset Discovery Narratives (TRAIN+WF only, no OOS, no deployment)

Generated: 2026-05-23T03:39

Source: engine_catalog_discovery.parquet (213 engines), basket_catalog (WF-tier), basket_catalog_catch_tier (catch-tier), confluence_catalog_v2.

Mandate: comprehensive stories + patterns + best configs per asset. NOT deployment-ready.


## Reading guide

- **WF-tier**: engines passing strict walk-forward stability (all 3 sub-folds positive, cov > 0.3, DD > -20%). Most reliable.

- **catch-tier**: engines with high TRUE top-25% mover catch-rate (>45% vs 25.3% random). Statistical alpha, less reliable across folds.

- **confluence**: 2-of-K AND-within-3-bars pairs on the same asset. Most show lift < 1.0x (combination FILTERS rather than amplifies — single engine usually wins).


## AAVE

- Catalog engines: 13 total
- WF-tier basket: 1 members
- Catch-tier basket: 5 members
- Confluence pairs: 13

**Family distribution**: measure_event(9), ta_event(4)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/rv_jump_frac(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=10
    Hit 90.0% | expectancy +2.15%/fire | compound +23.4% | max DD -0.7%
    WF folds: +15.7% / +6.7% / +0.0% (stability=0.14, ShIC=0.17)
    76.1% catch on 71 fires (vs 25.3% random)

  - **VPIN_threshold(t_0.5)** | regime=chop | hold=3d | n_fires=13
    Hit 61.5% | expectancy +1.10%/fire | compound +13.3% | max DD -11.9%
    WF folds: +3.5% / +8.9% / +20.5% (stability=0.35, ShIC=0.15)
    50.2% catch on 412 fires (vs 25.3% random)

  - **measure_engines/hbr_eta_total(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=10
    Hit 60.0% | expectancy +1.16%/fire | compound +11.9% | max DD -2.1%
    WF folds: +6.3% / +3.8% / +0.0% (stability=0.23, ShIC=0.16)
    48.6% catch on 142 fires (vs 25.3% random)

  - **measure_engines/bs_basis_z30(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=10
    Hit 70.0% | expectancy +1.84%/fire | compound +19.7% | max DD -0.7%
    WF folds: +14.1% / +0.0% / +5.0% (stability=0.08, ShIC=0.16)
    47.8% catch on 90 fires (vs 25.3% random)

  - **measure_engines/xd_btc_volatility(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=14
    Hit 64.3% | expectancy +0.87%/fire | compound +11.7% | max DD -10.1%
    WF folds: +3.0% / +6.8% / +0.0% (stability=0.15, ShIC=0.19)
    47.4% catch on 114 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 1 engine(s) survive STRICT WF stability — top scout candidates. 5 catch-tier engine(s), best catch_rate 76.1%. confluence does NOT add value (best lift < 1.2x).

---

## ADA

- Catalog engines: 24 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 5

**Family distribution**: ta_event(13), measure_event(6), ta_state(5)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/te_imb(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=21
    Hit 66.7% | expectancy +2.18%/fire | compound +52.6% | max DD -7.6%
    WF folds: +13.2% / +34.3% / +0.3% (stability=0.12, ShIC=0.17)
    51.4% catch on 222 fires (vs 25.3% random)

  - **VPIN_threshold(t_1.0)** | regime=bull | hold=3d | n_fires=10
    Hit 60.0% | expectancy +4.42%/fire | compound +49.3% | max DD -5.2%
    WF folds: +0.0% / +19.2% / +15.9% (stability=0.28, ShIC=0.16)
    48.8% catch on 256 fires (vs 25.3% random)

  - **OBV_zscore(p_100_t_1.0)** | regime=bull | hold=1d | n_fires=15
    Hit 60.0% | expectancy +0.97%/fire | compound +14.4% | max DD -7.8%
    WF folds: +0.0% / +17.8% / +12.5% (stability=0.26, ShIC=0.14)
    46.9% catch on 1144 fires (vs 25.3% random)

  - **VWAP_state_above(period_50)** | regime=bull | hold=1d | n_fires=19
    Hit 73.7% | expectancy +1.43%/fire | compound +30.0% | max DD -7.8%
    WF folds: +0.0% / +22.2% / +3.6% (stability=-0.13, ShIC=0.12)
    46.5% catch on 1257 fires (vs 25.3% random)

  - **Distance_z_state(period_50_threshold_1.5)** | regime=bull | hold=1d | n_fires=21
    Hit 66.7% | expectancy +1.76%/fire | compound +39.9% | max DD -9.3%
    WF folds: +0.0% / +37.0% / +1.0% (stability=-0.36, ShIC=0.29)
    45.8% catch on 238 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 51.4%. confluence does NOT add value (best lift < 1.2x).

---

## ALGO

- Catalog engines: 43 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(27), measure_event(11), ta_state(5)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_funding_spread(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 64.3% | expectancy +2.31%/fire | compound +35.7% | max DD -6.4%
    WF folds: +7.3% / +26.5% / +0.0% (stability=0.01, ShIC=0.19)
    46.2% catch on 143 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 46.2%.

---

## APT

- Catalog engines: 37 total
- WF-tier basket: 2 members
- Catch-tier basket: 5 members
- Confluence pairs: 3

**Family distribution**: ta_event(27), measure_event(8), ta_state(2)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **Distance_z_state(period_50_threshold_1.5)** | regime=bull | hold=3d | n_fires=10
    Hit 70.0% | expectancy +5.16%/fire | compound +62.6% | max DD -3.3%
    WF folds: +0.0% / +31.0% / +15.7% (stability=0.19, ShIC=0.26)
    52.0% catch on 229 fires (vs 25.3% random)

  - **RSI_threshold(p_5_lo_40_hi_65)** | regime=bull | hold=3d | n_fires=11
    Hit 72.7% | expectancy +2.86%/fire | compound +34.5% | max DD -9.0%
    WF folds: +0.0% / -0.9% / -14.8% (stability=-0.30, ShIC=0.27)
    49.1% catch on 863 fires (vs 25.3% random)

  - **Liquidation_cascade(t_1.0)** | regime=bull | hold=1d | n_fires=11
    Hit 63.6% | expectancy +1.55%/fire | compound +17.4% | max DD -7.1%
    WF folds: +0.0% / +12.2% / +3.2% (stability=-0.01, ShIC=0.29)
    47.6% catch on 166 fires (vs 25.3% random)

  - **MA_state_EMA_above(period_20)** | regime=bull | hold=3d | n_fires=13
    Hit 61.5% | expectancy +2.53%/fire | compound +34.4% | max DD -13.0%
    WF folds: +0.0% / +15.8% / +15.7% (stability=0.29, ShIC=0.21)
    46.3% catch on 382 fires (vs 25.3% random)

  - **measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=13
    Hit 46.2% | expectancy +2.99%/fire | compound +41.3% | max DD -8.5%
    WF folds: +17.0% / +18.5% / +1.9% (stability=0.40, ShIC=0.22)
    45.8% catch on 155 fires (vs 25.3% random)

### WF-tier additions (1 engines NOT already in catch-tier)

  - **VPIN_threshold(t_0.5)** | regime=chop | hold=1d | n_fires=24
    Hit 58.3% | expectancy +1.58%/fire | compound +42.8% | max DD -14.0%
    WF folds: +15.3% / +21.7% / +14.0% (stability=0.80, ShIC=0.26)
    catch n/a

### Confluence pairs (2 notable)

  - **ta_state_MA + measure_wh**
    A: MA_state_EMA_above(period_20) regime=bull, train_compound=+34.4%
    B: measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0) regime=bull, train_compound=+41.3%
    Confluence: n=107 hit=46.7% mean=+0.69% compound=+82.3% lift_vs_max=0.23x

  - **ta_VPIN + measure_wh**
    A: VPIN_threshold(t_0.5) regime=chop, train_compound=+42.8%
    B: measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0) regime=bull, train_compound=+41.3%
    Confluence: n=12 hit=58.3% mean=+2.14% compound=+27.2% lift_vs_max=0.72x

**Verdict (TRAIN-only)**: 2 engine(s) survive STRICT WF stability — top scout candidates. 5 catch-tier engine(s), best catch_rate 52.0%. confluence does NOT add value (best lift < 1.2x).

---

## AR

- Catalog engines: 13 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 0

**Family distribution**: measure_event(5), ta_event(5), ta_state(3)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/rv_rv_5m(op_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=11
    Hit 72.7% | expectancy +7.68%/fire | compound +106.4% | max DD -6.0%
    WF folds: +0.0% / +64.6% / +27.8% (stability=0.14, ShIC=0.26)
    53.1% catch on 98 fires (vs 25.3% random)

  - **OBV_zscore(p_20_t_1.5)** | regime=bull | hold=1d | n_fires=18
    Hit 72.2% | expectancy +0.86%/fire | compound +16.0% | max DD -6.3%
    WF folds: +3.9% / +20.1% / +0.0% (stability=-0.09, ShIC=0.28)
    52.8% catch on 583 fires (vs 25.3% random)

  - **measure_engines/te_in_btc(op_abs_gt_thr_1.0)** | regime=bull | hold=3d | n_fires=11
    Hit 63.6% | expectancy +3.59%/fire | compound +44.7% | max DD -6.7%
    WF folds: +6.3% / +46.0% / +0.0% (stability=-0.17, ShIC=0.25)
    51.7% catch on 240 fires (vs 25.3% random)

  - **Distance_z_state(period_20_threshold_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 50.0% | expectancy +1.56%/fire | compound +15.1% | max DD -11.1%
    WF folds: +11.6% / +0.8% / +0.0% (stability=-0.27, ShIC=0.21)
    49.7% catch on 296 fires (vs 25.3% random)

  - **MA_state_SMA_above(period_20)** | regime=bull | hold=1d | n_fires=10
    Hit 70.0% | expectancy +4.45%/fire | compound +52.3% | max DD -0.4%
    WF folds: +0.6% / +11.2% / +36.1% (stability=0.07, ShIC=0.25)
    49.4% catch on 409 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 53.1%.

---

## ARB

- Catalog engines: 16 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: measure_event(10), ta_event(5), ta_state(1)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/bs_basis_z30(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=17
    Hit 58.8% | expectancy +1.94%/fire | compound +35.1% | max DD -9.2%
    WF folds: +0.0% / +28.1% / +5.4% (stability=-0.09, ShIC=0.24)
    47.2% catch on 125 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 47.2%.

---

## ARKM

- Catalog engines: 4 total
- WF-tier basket: 0 members
- Catch-tier basket: 3 members
- Confluence pairs: 0

**Family distribution**: ta_event(4)

### Catch-tier basket (3 engines, ranked by catch_rate × sqrt(n))

  - **OBV_zscore(p_30_t_1.0)** | regime=chop | hold=1d | n_fires=21
    Hit 52.4% | expectancy +1.38%/fire | compound +28.7% | max DD -12.9%
    WF folds: +7.0% / +0.0% / +12.6% (stability=0.21, ShIC=0.29)
    56.0% catch on 770 fires (vs 25.3% random)

  - **VPIN_threshold(t_0.5)** | regime=chop | hold=3d | n_fires=13
    Hit 69.2% | expectancy +3.69%/fire | compound +55.2% | max DD -10.5%
    WF folds: +12.4% / +15.9% / +0.0% (stability=0.28, ShIC=0.26)
    49.8% catch on 548 fires (vs 25.3% random)

  - **Kyle_lambda_threshold(t_0.5)** | regime=bull | hold=1d | n_fires=20
    Hit 75.0% | expectancy +3.98%/fire | compound +104.1% | max DD -14.5%
    WF folds: +0.0% / +49.8% / +48.8% (stability=0.29, ShIC=0.21)
    45.8% catch on 559 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 3 catch-tier engine(s), best catch_rate 56.0%.

---

## AVAX

- Catalog engines: 11 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: measure_event(7), ta_event(3), ta_state(1)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_flow_imbalance(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 58.3% | expectancy +2.91%/fire | compound +38.1% | max DD -6.7%
    WF folds: +0.0% / +29.0% / +13.0% (stability=0.15, ShIC=0.30)
    52.6% catch on 137 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 52.6%.

---

## BCH

- Catalog engines: 7 total
- WF-tier basket: 0 members
- Catch-tier basket: 2 members
- Confluence pairs: 0

**Family distribution**: measure_event(5), ta_event(2)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/rv_jump_count(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 50.0% | expectancy +2.39%/fire | compound +25.8% | max DD -3.3%
    WF folds: +7.0% / +7.7% / +0.0% (stability=0.29, ShIC=0.22)
    51.5% catch on 68 fires (vs 25.3% random)

  - **measure_engines/stbl_total_zscore_30d(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=11
    Hit 54.5% | expectancy +2.33%/fire | compound +27.6% | max DD -4.2%
    WF folds: +15.0% / +6.1% / +4.6% (stability=0.46, ShIC=0.23)
    50.0% catch on 82 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 2 catch-tier engine(s), best catch_rate 51.5%.

---

## BNB

- Catalog engines: 9 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: measure_event(5), ta_event(3), ta_state(1)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/te_btc_imb(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=22
    Hit 54.5% | expectancy +1.17%/fire | compound +27.9% | max DD -3.2%
    WF folds: +1.3% / +28.7% / +0.0% (stability=-0.32, ShIC=0.23)
    47.2% catch on 199 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 47.2%.

---

## BTC

- Catalog engines: 10 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 17

**Family distribution**: measure_event(4), ta_event(3), ta_state(3)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_momentum_rank(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 75.0% | expectancy +2.17%/fire | compound +28.6% | max DD -2.3%
    WF folds: +0.0% / +18.7% / +9.4% (stability=0.18, ShIC=0.19)
    55.8% catch on 154 fires (vs 25.3% random)

  - **measure_engines/stbl_total_zscore_30d(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 60.0% | expectancy +1.55%/fire | compound +15.9% | max DD -2.3%
    WF folds: +12.5% / +2.2% / +0.8% (stability=-0.01, ShIC=0.24)
    55.7% catch on 61 fires (vs 25.3% random)

  - **ATR_bands(p_20_k_1.5)** | regime=bull | hold=3d | n_fires=12
    Hit 66.7% | expectancy +2.19%/fire | compound +26.5% | max DD -10.2%
    WF folds: +0.0% / +5.2% / +11.9% (stability=0.14, ShIC=0.29)
    51.1% catch on 784 fires (vs 25.3% random)

  - **Donchian_state_above_midline(period_20)** | regime=bull | hold=1d | n_fires=33
    Hit 63.6% | expectancy +0.89%/fire | compound +32.3% | max DD -6.7%
    WF folds: +0.0% / +21.4% / +5.8% (stability=0.00, ShIC=0.29)
    47.8% catch on 295 fires (vs 25.3% random)

  - **MA_state_EMA_above(period_20)** | regime=bull | hold=1d | n_fires=11
    Hit 72.7% | expectancy +1.22%/fire | compound +13.8% | max DD -1.4%
    WF folds: +0.0% / +1.4% / +13.0% (stability=-0.22, ShIC=0.21)
    46.8% catch on 308 fires (vs 25.3% random)

### Confluence pairs (1 notable)

  - **ta_state_MA + measure_stbl**
    A: MA_state_SMA_above(period_50) regime=chop, train_compound=+36.6%
    B: measure_engines/stbl_total_zscore_30d(op_abs_gt_thr_1.0) regime=bull, train_compound=+15.9%
    Confluence: n=8 hit=62.5% mean=+2.08% compound=+17.4% lift_vs_max=1.30x

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 55.8%. confluence adds value (best lift 1.30x).

---

## CHZ

- Catalog engines: 28 total
- WF-tier basket: 1 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(17), measure_event(10), ta_state(1)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_ma_distance(op_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=11
    Hit 72.7% | expectancy +1.85%/fire | compound +21.8% | max DD -2.1%
    WF folds: +6.9% / +3.4% / +0.0% (stability=0.18, ShIC=0.26)
    58.7% catch on 46 fires (vs 25.3% random)

### WF-tier additions (1 engines NOT already in catch-tier)

  - **measure_engines/xd_btc_return(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 66.7% | expectancy +2.81%/fire | compound +38.0% | max DD -4.0%
    WF folds: +7.2% / +17.6% / +9.5% (stability=0.61, ShIC=0.22)
    catch n/a

**Verdict (TRAIN-only)**: 1 engine(s) survive STRICT WF stability — top scout candidates. 1 catch-tier engine(s), best catch_rate 58.7%.

---

## DASH

- Catalog engines: 27 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 2

**Family distribution**: ta_event(16), measure_event(7), ta_state(4)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **ETF_flow_z(t_0.5)** | regime=bull | hold=1d | n_fires=11
    Hit 54.5% | expectancy +2.58%/fire | compound +29.9% | max DD -6.3%
    WF folds: +0.0% / +8.5% / +19.7% (stability=0.14, ShIC=0.19)
    50.0% catch on 128 fires (vs 25.3% random)

  - **measure_engines/xd_btc_return(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=16
    Hit 62.5% | expectancy +1.02%/fire | compound +17.2% | max DD -3.1%
    WF folds: +9.5% / +4.3% / +0.0% (stability=0.16, ShIC=0.27)
    48.7% catch on 156 fires (vs 25.3% random)

  - **OBV_zscore(p_50_t_1.0)** | regime=chop | hold=1d | n_fires=30
    Hit 66.7% | expectancy +0.76%/fire | compound +24.5% | max DD -7.4%
    WF folds: +14.8% / +19.7% / +4.6% (stability=0.52, ShIC=0.15)
    48.3% catch on 789 fires (vs 25.3% random)

  - **MA_state_SMA_above(period_50)** | regime=bull | hold=1d | n_fires=30
    Hit 63.3% | expectancy +0.63%/fire | compound +18.4% | max DD -12.3%
    WF folds: +0.0% / +11.7% / +4.9% (stability=0.14, ShIC=0.24)
    45.1% catch on 368 fires (vs 25.3% random)

  - **Distance_z_state(period_50_threshold_1.0)** | regime=bull | hold=1d | n_fires=23
    Hit 69.6% | expectancy +1.22%/fire | compound +30.4% | max DD -7.4%
    WF folds: +0.0% / +25.7% / +2.7% (stability=-0.22, ShIC=0.23)
    45.1% catch on 264 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 50.0%. confluence does NOT add value (best lift < 1.2x).

---

## DOGE

- Catalog engines: 7 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: measure_event(5), ta_event(2)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_flow_imbalance(op_abs_gt_thr_1.0)** | regime=chop | hold=3d | n_fires=10
    Hit 60.0% | expectancy +3.54%/fire | compound +38.2% | max DD -3.7%
    WF folds: +2.3% / +9.0% / +0.0% (stability=-0.02, ShIC=0.23)
    57.4% catch on 94 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 57.4%.

---

## DOT

- Catalog engines: 21 total
- WF-tier basket: 0 members
- Catch-tier basket: 4 members
- Confluence pairs: 1

**Family distribution**: ta_event(13), measure_event(7), ta_state(1)

### Catch-tier basket (4 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_efficiency(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 66.7% | expectancy +2.32%/fire | compound +30.7% | max DD -2.3%
    WF folds: +0.0% / +12.1% / +8.3% (stability=0.26, ShIC=0.27)
    51.8% catch on 139 fires (vs 25.3% random)

  - **Distance_z_state(period_50_threshold_1.5)** | regime=bull | hold=1d | n_fires=21
    Hit 71.4% | expectancy +1.81%/fire | compound +42.6% | max DD -10.4%
    WF folds: +0.0% / +30.4% / +5.7% (stability=-0.09, ShIC=0.28)
    46.4% catch on 222 fires (vs 25.3% random)

  - **measure_engines/xd_btc_return(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 70.0% | expectancy +2.59%/fire | compound +28.2% | max DD -2.3%
    WF folds: +0.0% / +1.7% / +20.8% (stability=-0.26, ShIC=0.20)
    45.5% catch on 123 fires (vs 25.3% random)

  - **OBV_zscore(p_20_t_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 50.0% | expectancy +0.73%/fire | compound +9.7% | max DD -8.4%
    WF folds: +0.0% / +14.7% / +1.4% (stability=-0.23, ShIC=0.28)
    45.4% catch on 1015 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 4 catch-tier engine(s), best catch_rate 51.8%. confluence does NOT add value (best lift < 1.2x).

---

## DYDX

- Catalog engines: 11 total
- WF-tier basket: 0 members
- Catch-tier basket: 2 members
- Confluence pairs: 0

**Family distribution**: measure_event(8), ta_state(2), ta_event(1)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **MA_state_EMA_above(period_20)** | regime=chop | hold=1d | n_fires=10
    Hit 70.0% | expectancy +1.26%/fire | compound +12.5% | max DD -2.7%
    WF folds: +1.7% / +10.7% / +0.0% (stability=-0.14, ShIC=0.26)
    47.2% catch on 163 fires (vs 25.3% random)

  - **OBV_zscore(p_30_t_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 66.7% | expectancy +0.74%/fire | compound +8.5% | max DD -5.8%
    WF folds: +6.8% / +0.4% / +0.0% (stability=-0.31, ShIC=0.27)
    46.3% catch on 949 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 2 catch-tier engine(s), best catch_rate 47.2%.

---

## ENJ

- Catalog engines: 86 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(84), measure_event(2)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **Kyle_lambda_threshold(t_1.0)** | regime=bull | hold=3d | n_fires=10
    Hit 60.0% | expectancy +1.76%/fire | compound +15.8% | max DD -13.7%
    WF folds: +0.0% / +5.6% / +27.5% (stability=-0.07, ShIC=0.17)
    45.1% catch on 410 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 45.1%.

---

## ETC

- Catalog engines: 20 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(11), measure_event(9)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_efficiency(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=16
    Hit 75.0% | expectancy +0.53%/fire | compound +8.1% | max DD -9.6%
    WF folds: +3.0% / +16.1% / +0.0% (stability=-0.10, ShIC=0.29)
    48.4% catch on 128 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 48.4%.

---

## FET

- Catalog engines: 17 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 8

**Family distribution**: measure_event(14), ta_event(2), ta_state(1)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_flow_imbalance(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=23
    Hit 65.2% | expectancy +3.13%/fire | compound +95.0% | max DD -7.7%
    WF folds: +1.9% / +59.8% / +0.0% (stability=-0.35, ShIC=0.18)
    70.4% catch on 162 fires (vs 25.3% random)

  - **measure_engines/rv_bpv_5m(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=18
    Hit 66.7% | expectancy +3.43%/fire | compound +77.4% | max DD -10.7%
    WF folds: +19.9% / +30.9% / +13.0% (stability=0.65, ShIC=0.15)
    64.0% catch on 175 fires (vs 25.3% random)

  - **measure_engines/xd_btc_return(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=11
    Hit 54.5% | expectancy +1.11%/fire | compound +9.8% | max DD -14.2%
    WF folds: +4.6% / +3.1% / +0.0% (stability=0.26, ShIC=0.26)
    59.2% catch on 142 fires (vs 25.3% random)

  - **VPIN_threshold(t_0.5)** | regime=bull | hold=3d | n_fires=11
    Hit 54.5% | expectancy +7.12%/fire | compound +88.9% | max DD -10.4%
    WF folds: +0.0% / -42.9% / -58.0% (stability=0.27, ShIC=0.30)
    54.4% catch on 594 fires (vs 25.3% random)

  - **Distance_z_state(period_20_threshold_1.0)** | regime=bull | hold=1d | n_fires=21
    Hit 57.1% | expectancy +3.14%/fire | compound +81.3% | max DD -13.5%
    WF folds: +15.4% / +36.7% / +15.0% (stability=0.55, ShIC=0.23)
    54.1% catch on 331 fires (vs 25.3% random)

### Confluence pairs (6 notable)

  - **ta_VPIN + measure_wh**
    A: VPIN_threshold(t_0.5) regime=bull, train_compound=+88.9%
    B: measure_engines/wh_whale_net_usd(op_abs_gt_thr_2.0) regime=bull, train_compound=+45.7%
    Confluence: n=78 hit=62.8% mean=+3.26% compound=+840.7% lift_vs_max=0.46x

  - **ta_VPIN + measure_wh**
    A: VPIN_threshold(t_0.5) regime=bull, train_compound=+88.9%
    B: measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.5) regime=bull, train_compound=+39.5%
    Confluence: n=86 hit=62.8% mean=+2.94% compound=+807.0% lift_vs_max=0.41x

  - **ta_VPIN + measure_xd**
    A: VPIN_threshold(t_0.5) regime=bull, train_compound=+88.9%
    B: measure_engines/xd_btc_return(op_abs_gt_thr_1.0) regime=bull, train_compound=+9.8%
    Confluence: n=117 hit=55.6% mean=+1.82% compound=+493.3% lift_vs_max=0.26x

  - **measure_wh + measure_xd**
    A: measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.5) regime=bull, train_compound=+39.5%
    B: measure_engines/xd_btc_return(op_abs_gt_thr_1.0) regime=bull, train_compound=+9.8%
    Confluence: n=54 hit=61.1% mean=+2.83% compound=+288.7% lift_vs_max=0.99x

  - **measure_wh + measure_xd**
    A: measure_engines/wh_whale_net_usd(op_abs_gt_thr_2.0) regime=bull, train_compound=+45.7%
    B: measure_engines/xd_btc_return(op_abs_gt_thr_1.0) regime=bull, train_compound=+9.8%
    Confluence: n=49 hit=61.2% mean=+2.73% compound=+225.5% lift_vs_max=0.73x

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 70.4%. confluence adds value (best lift 1.53x).

---

## FIL

- Catalog engines: 14 total
- WF-tier basket: 1 members
- Catch-tier basket: 4 members
- Confluence pairs: 5

**Family distribution**: measure_event(10), ta_event(3), ta_state(1)

### Catch-tier basket (4 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_momentum_rank(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=17
    Hit 58.8% | expectancy +2.18%/fire | compound +40.9% | max DD -6.6%
    WF folds: +4.4% / +31.0% / +3.1% (stability=-0.00, ShIC=0.24)
    51.8% catch on 164 fires (vs 25.3% random)

  - **measure_engines/norm_efficiency(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=14
    Hit 71.4% | expectancy +2.10%/fire | compound +33.0% | max DD -2.1%
    WF folds: +3.5% / +14.3% / +12.4% (stability=0.53, ShIC=0.16)
    49.5% catch on 99 fires (vs 25.3% random)

  - **measure_engines/liq_long_usd(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 60.0% | expectancy +3.56%/fire | compound +38.6% | max DD -4.9%
    WF folds: +0.0% / +1.2% / +42.5% (stability=-0.36, ShIC=0.21)
    48.1% catch on 106 fires (vs 25.3% random)

  - **measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 64.3% | expectancy +2.18%/fire | compound +31.1% | max DD -11.5%
    WF folds: +16.1% / +17.0% / +0.0% (stability=0.29, ShIC=0.23)
    45.7% catch on 175 fires (vs 25.3% random)

### Confluence pairs (2 notable)

  - **measure_bd + measure_liq**
    A: measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0) regime=bull, train_compound=+31.1%
    B: measure_engines/liq_long_usd(op_abs_gt_thr_1.0) regime=bull, train_compound=+38.6%
    Confluence: n=53 hit=50.9% mean=+1.06% compound=+59.6% lift_vs_max=0.30x

  - **measure_bd + measure_liq**
    A: measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0) regime=bull, train_compound=+31.1%
    B: measure_engines/liq_long_usd(op_gt_thr_1.0) regime=bull, train_compound=+38.6%
    Confluence: n=53 hit=50.9% mean=+1.06% compound=+59.6% lift_vs_max=0.30x

**Verdict (TRAIN-only)**: 1 engine(s) survive STRICT WF stability — top scout candidates. 4 catch-tier engine(s), best catch_rate 51.8%. confluence does NOT add value (best lift < 1.2x).

---

## FLOKI

- Catalog engines: 8 total
- WF-tier basket: 0 members
- Catch-tier basket: 2 members
- Confluence pairs: 0

**Family distribution**: measure_event(5), ta_event(3)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/hbr_eta_buy(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=16
    Hit 68.8% | expectancy +2.46%/fire | compound +44.7% | max DD -3.3%
    WF folds: +15.9% / +20.7% / +0.0% (stability=0.28, ShIC=0.22)
    52.4% catch on 143 fires (vs 25.3% random)

  - **ETF_flow_z(t_0.5)** | regime=bull | hold=1d | n_fires=11
    Hit 72.7% | expectancy +3.61%/fire | compound +45.9% | max DD -5.8%
    WF folds: +0.0% / +4.9% / +39.0% (stability=-0.19, ShIC=0.24)
    47.5% catch on 120 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 2 catch-tier engine(s), best catch_rate 52.4%.

---

## HBAR

- Catalog engines: 36 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 4

**Family distribution**: ta_event(22), measure_event(11), ta_state(3)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/norm_efficiency(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=13
    Hit 69.2% | expectancy +3.09%/fire | compound +44.3% | max DD -6.2%
    WF folds: +0.0% / +37.3% / +7.9% (stability=-0.07, ShIC=0.19)
    58.9% catch on 129 fires (vs 25.3% random)

  - **measure_engines/xd_funding_spread(op_abs_gt_thr_1.0)** | regime=bear | hold=1d | n_fires=10
    Hit 80.0% | expectancy +2.46%/fire | compound +26.6% | max DD -4.9%
    WF folds: +2.5% / +0.0% / +23.4% (stability=-0.21, ShIC=0.26)
    54.4% catch on 57 fires (vs 25.3% random)

  - **measure_engines/te_btc_imb(op_abs_gt_thr_1.5)** | regime=bear | hold=1d | n_fires=11
    Hit 63.6% | expectancy +2.42%/fire | compound +28.6% | max DD -6.7%
    WF folds: +7.3% / +0.0% / +19.8% (stability=0.09, ShIC=0.23)
    51.0% catch on 49 fires (vs 25.3% random)

  - **measure_engines/wh_whale_trade_count_500k(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 70.0% | expectancy +2.09%/fire | compound +22.6% | max DD -2.4%
    WF folds: +0.0% / +9.1% / +11.0% (stability=0.28, ShIC=0.20)
    50.0% catch on 134 fires (vs 25.3% random)

  - **ETF_flow_z(t_0.5)** | regime=bull | hold=1d | n_fires=11
    Hit 72.7% | expectancy +3.22%/fire | compound +37.6% | max DD -12.3%
    WF folds: +0.0% / +27.6% / +7.9% (stability=0.02, ShIC=0.15)
    46.4% catch on 125 fires (vs 25.3% random)

### Confluence pairs (4 notable)

  - **measure_norm + measure_wh**
    A: measure_engines/norm_efficiency(op_gt_thr_1.0) regime=bull, train_compound=+49.7%
    B: measure_engines/wh_whale_trade_count_500k(op_abs_gt_thr_1.0) regime=bull, train_compound=+22.6%
    Confluence: n=42 hit=59.5% mean=+1.96% compound=+115.3% lift_vs_max=0.49x

  - **measure_norm + measure_wh**
    A: measure_engines/norm_efficiency(op_abs_gt_thr_1.0) regime=bull, train_compound=+44.3%
    B: measure_engines/wh_whale_trade_count_500k(op_abs_gt_thr_1.0) regime=bull, train_compound=+22.6%
    Confluence: n=57 hit=57.9% mean=+1.35% compound=+101.7% lift_vs_max=0.44x

  - **measure_norm + measure_wh**
    A: measure_engines/norm_efficiency(op_gt_thr_1.0) regime=bull, train_compound=+49.7%
    B: measure_engines/wh_whale_trade_count_500k(op_gt_thr_1.0) regime=bull, train_compound=+22.6%
    Confluence: n=41 hit=58.5% mean=+1.77% compound=+96.0% lift_vs_max=0.45x

  - **measure_norm + measure_wh**
    A: measure_engines/norm_efficiency(op_abs_gt_thr_1.0) regime=bull, train_compound=+44.3%
    B: measure_engines/wh_whale_trade_count_500k(op_gt_thr_1.0) regime=bull, train_compound=+22.6%
    Confluence: n=54 hit=57.4% mean=+1.31% compound=+91.1% lift_vs_max=0.42x

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 58.9%. confluence does NOT add value (best lift < 1.2x).

---

## ICP

- Catalog engines: 16 total
- WF-tier basket: 2 members
- Catch-tier basket: 5 members
- Confluence pairs: 2

**Family distribution**: measure_event(8), ta_event(7), ta_state(1)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_btc_volatility(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 71.4% | expectancy +5.00%/fire | compound +80.0% | max DD -13.2%
    WF folds: +4.1% / +65.1% / +0.0% (stability=-0.29, ShIC=0.16)
    57.7% catch on 149 fires (vs 25.3% random)

  - **measure_engines/bd_imbalance_l1(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=21
    Hit 57.1% | expectancy +2.38%/fire | compound +58.6% | max DD -13.5%
    WF folds: +12.4% / +11.6% / +26.5% (stability=0.59, ShIC=0.25)
    54.0% catch on 176 fires (vs 25.3% random)

  - **Distance_z_state(period_50_threshold_1.5)** | regime=bull | hold=1d | n_fires=22
    Hit 59.1% | expectancy +5.03%/fire | compound +167.0% | max DD -5.7%
    WF folds: +0.0% / +148.0% / +7.8% (stability=-0.31, ShIC=0.24)
    50.0% catch on 222 fires (vs 25.3% random)

  - **RSI_threshold(p_6_lo_40_hi_65)** | regime=bull | hold=3d | n_fires=12
    Hit 50.0% | expectancy +1.02%/fire | compound +10.9% | max DD -9.2%
    WF folds: +0.0% / -64.9% / -14.1% (stability=-0.06, ShIC=0.29)
    46.3% catch on 860 fires (vs 25.3% random)

  - **measure_engines/te_btc_imb(op_abs_gt_thr_1.0)** | regime=bull | hold=3d | n_fires=12
    Hit 75.0% | expectancy +4.62%/fire | compound +60.3% | max DD -14.1%
    WF folds: +0.0% / +42.3% / +2.3% (stability=-0.31, ShIC=0.19)
    45.5% catch on 286 fires (vs 25.3% random)

### WF-tier additions (1 engines NOT already in catch-tier)

  - **measure_engines/xd_funding_spread(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=13
    Hit 69.2% | expectancy +2.05%/fire | compound +29.1% | max DD -5.8%
    WF folds: +3.0% / +10.2% / +13.7% (stability=0.51, ShIC=0.21)
    catch n/a z_lift_t-3=-0.10

### Confluence pairs (1 notable)

  - **measure_bd + measure_xd**
    A: measure_engines/bd_imbalance_l1(op_abs_gt_thr_1.0) regime=bull, train_compound=+58.6%
    B: measure_engines/xd_funding_spread(op_abs_gt_thr_1.0) regime=chop, train_compound=+29.1%
    Confluence: n=8 hit=75.0% mean=+2.30% compound=+19.2% lift_vs_max=0.97x

**Verdict (TRAIN-only)**: 2 engine(s) survive STRICT WF stability — top scout candidates. 5 catch-tier engine(s), best catch_rate 57.7%. confluence does NOT add value (best lift < 1.2x).

---

## JST

- Catalog engines: 55 total
- WF-tier basket: 1 members
- Catch-tier basket: 2 members
- Confluence pairs: 0

**Family distribution**: ta_event(45), ta_state(7), measure_event(3)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **MA_state_SMA_above(period_20)** | regime=bull | hold=1d | n_fires=32
    Hit 62.5% | expectancy +1.36%/fire | compound +48.9% | max DD -7.9%
    WF folds: +27.3% / +12.1% / +4.4% (stability=0.35, ShIC=0.28)
    47.2% catch on 286 fires (vs 25.3% random)

  - **RSI_threshold(p_9_lo_40_hi_65)** | regime=chop | hold=1d | n_fires=20
    Hit 50.0% | expectancy +0.87%/fire | compound +17.6% | max DD -6.4%
    WF folds: +7.3% / +2.2% / +0.0% (stability=0.03, ShIC=0.14)
    46.8% catch on 474 fires (vs 25.3% random)

### WF-tier additions (1 engines NOT already in catch-tier)

  - **MA_state_EMA_above(period_200)** | regime=chop | hold=1d | n_fires=21
    Hit 76.2% | expectancy +1.05%/fire | compound +23.8% | max DD -4.4%
    WF folds: +7.3% / +6.7% / +8.2% (stability=0.92, ShIC=0.26)
    catch n/a

**Verdict (TRAIN-only)**: 1 engine(s) survive STRICT WF stability — top scout candidates. 2 catch-tier engine(s), best catch_rate 47.2%.

---

## LINK

- Catalog engines: 28 total
- WF-tier basket: 3 members
- Catch-tier basket: 5 members
- Confluence pairs: 38

**Family distribution**: ta_event(18), ta_state(7), measure_event(3)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **Donchian_state_above_midline(period_100)** | regime=chop | hold=1d | n_fires=44
    Hit 63.6% | expectancy +0.57%/fire | compound +25.8% | max DD -11.9%
    WF folds: +7.2% / +16.4% / +0.0% (stability=0.14, ShIC=0.22)
    48.8% catch on 322 fires (vs 25.3% random)

  - **MA_state_EMA_above(period_200)** | regime=chop | hold=1d | n_fires=44
    Hit 63.6% | expectancy +0.57%/fire | compound +25.8% | max DD -11.9%
    WF folds: +7.2% / +16.4% / +0.0% (stability=0.14, ShIC=0.22)
    48.0% catch on 327 fires (vs 25.3% random)

  - **RSI_threshold(p_5_lo_40_hi_60)** | regime=chop | hold=3d | n_fires=10
    Hit 80.0% | expectancy +1.75%/fire | compound +18.1% | max DD -8.0%
    WF folds: +36.7% / +6.1% / +12.2% (stability=0.28, ShIC=0.23)
    46.9% catch on 924 fires (vs 25.3% random)

  - **MACD_threshold(f_12_s_21_g_9)** | regime=chop | hold=3d | n_fires=11
    Hit 54.5% | expectancy +2.19%/fire | compound +24.6% | max DD -8.6%
    WF folds: -4.9% / -1.8% / -3.7% (stability=0.64, ShIC=0.29)
    46.4% catch on 1516 fires (vs 25.3% random)

  - **measure_engines/liq_long_usd(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=10
    Hit 70.0% | expectancy +3.14%/fire | compound +34.7% | max DD -2.9%
    WF folds: +7.4% / +19.8% / +4.7% (stability=0.38, ShIC=0.22)
    46.0% catch on 124 fires (vs 25.3% random)

### WF-tier additions (3 engines NOT already in catch-tier)

  - **MA_state_SMA_above(period_20)** | regime=bull | hold=1d | n_fires=17
    Hit 58.8% | expectancy +1.72%/fire | compound +30.9% | max DD -9.1%
    WF folds: +7.4% / +12.2% / +8.7% (stability=0.78, ShIC=0.28)
    catch n/a

  - **VPIN_threshold(t_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 78.6% | expectancy +3.31%/fire | compound +54.7% | max DD -1.9%
    WF folds: +21.7% / +18.1% / +4.6% (stability=0.50, ShIC=0.19)
    catch n/a

  - **Donchian_state_above_midline(period_20)** | regime=bull | hold=1d | n_fires=17
    Hit 58.8% | expectancy +1.63%/fire | compound +28.8% | max DD -11.8%
    WF folds: +7.4% / +15.5% / +3.9% (stability=0.45, ShIC=0.26)
    catch n/a

### Confluence pairs (2 notable)

  - **ta_state_MA + ta_DC**
    A: MA_state_SMA_above(period_200) regime=chop, train_compound=+25.8%
    B: Donchian_state_above_midline(period_20) regime=bull, train_compound=+28.8%
    Confluence: n=31 hit=54.8% mean=+1.23% compound=+39.7% lift_vs_max=0.76x

  - **ta_state_MA + ta_DC**
    A: MA_state_EMA_above(period_200) regime=chop, train_compound=+25.8%
    B: Donchian_state_above_midline(period_20) regime=bull, train_compound=+28.8%
    Confluence: n=31 hit=54.8% mean=+1.23% compound=+39.7% lift_vs_max=0.76x

**Verdict (TRAIN-only)**: 3 engine(s) survive STRICT WF stability — top scout candidates. 5 catch-tier engine(s), best catch_rate 48.8%. confluence does NOT add value (best lift < 1.2x).

---

## LTC

- Catalog engines: 30 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(28), measure_event(2)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **RSI_threshold(p_5_lo_40_hi_65)** | regime=bear | hold=1d | n_fires=16
    Hit 68.8% | expectancy +0.84%/fire | compound +14.1% | max DD -1.6%
    WF folds: -3.7% / +0.0% / -0.9% (stability=-0.02, ShIC=0.25)
    47.2% catch on 362 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 47.2%.

---

## NEAR

- Catalog engines: 21 total
- WF-tier basket: 0 members
- Catch-tier basket: 2 members
- Confluence pairs: 0

**Family distribution**: ta_event(13), measure_event(8)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/wh_whale_trade_count_500k(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=17
    Hit 64.7% | expectancy +4.66%/fire | compound +104.8% | max DD -14.7%
    WF folds: +13.9% / +47.0% / +0.0% (stability=0.03, ShIC=0.24)
    53.7% catch on 201 fires (vs 25.3% random)

  - **measure_engines/xd_btc_volatility(op_abs_gt_thr_1.0)** | regime=bull | hold=3d | n_fires=10
    Hit 70.0% | expectancy +6.96%/fire | compound +87.3% | max DD -3.5%
    WF folds: +7.3% / +72.0% / +0.0% (stability=-0.23, ShIC=0.22)
    46.5% catch on 127 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 2 catch-tier engine(s), best catch_rate 53.7%.

---

## OP

- Catalog engines: 22 total
- WF-tier basket: 0 members
- Catch-tier basket: 4 members
- Confluence pairs: 5

**Family distribution**: ta_event(13), measure_event(5), ta_state(4)

### Catch-tier basket (4 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/rv_jump_count(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=10
    Hit 90.0% | expectancy +2.81%/fire | compound +31.6% | max DD -1.5%
    WF folds: +10.6% / +17.5% / +0.0% (stability=0.23, ShIC=0.25)
    61.5% catch on 52 fires (vs 25.3% random)

  - **measure_engines/bs_basis_z30(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=14
    Hit 64.3% | expectancy +1.37%/fire | compound +20.3% | max DD -6.2%
    WF folds: +0.7% / +10.3% / +0.0% (stability=-0.29, ShIC=0.29)
    55.1% catch on 78 fires (vs 25.3% random)

  - **MA_state_SMA_above(period_20)** | regime=chop | hold=1d | n_fires=20
    Hit 70.0% | expectancy +1.45%/fire | compound +31.2% | max DD -12.9%
    WF folds: +8.1% / +21.4% / +0.0% (stability=0.10, ShIC=0.29)
    49.6% catch on 139 fires (vs 25.3% random)

  - **Donchian_state_above_midline(period_20)** | regime=chop | hold=1d | n_fires=19
    Hit 73.7% | expectancy +1.87%/fire | compound +40.4% | max DD -6.9%
    WF folds: +15.6% / +21.4% / +0.0% (stability=0.27, ShIC=0.25)
    47.4% catch on 152 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 4 catch-tier engine(s), best catch_rate 61.5%. confluence does NOT add value (best lift < 1.2x).

---

## PEPE

- Catalog engines: 18 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 16

**Family distribution**: ta_state(7), ta_event(6), measure_event(5)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **MA_state_SMA_above(period_100)** | regime=chop | hold=1d | n_fires=15
    Hit 60.0% | expectancy +1.56%/fire | compound +23.1% | max DD -11.0%
    WF folds: +0.0% / +2.0% / +20.7% (stability=-0.23, ShIC=0.23)
    55.9% catch on 136 fires (vs 25.3% random)

  - **Donchian_state_above_midline(period_100)** | regime=chop | hold=1d | n_fires=13
    Hit 53.8% | expectancy +1.57%/fire | compound +19.4% | max DD -11.0%
    WF folds: +0.0% / +2.0% / +17.0% (stability=-0.19, ShIC=0.28)
    51.6% catch on 124 fires (vs 25.3% random)

  - **Hawkes_branching_imbalance(t_0.1)** | regime=chop | hold=1d | n_fires=11
    Hit 63.6% | expectancy +1.57%/fire | compound +16.7% | max DD -8.3%
    WF folds: +12.7% / +4.1% / +0.0% (stability=0.05, ShIC=0.18)
    47.7% catch on 128 fires (vs 25.3% random)

  - **RSI_threshold(p_8_lo_40_hi_80)** | regime=chop | hold=3d | n_fires=12
    Hit 66.7% | expectancy +0.82%/fire | compound +8.4% | max DD -10.3%
    WF folds: +27.8% / +31.9% / +0.0% (stability=0.29, ShIC=0.24)
    46.9% catch on 557 fires (vs 25.3% random)

  - **measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=12
    Hit 66.7% | expectancy +7.31%/fire | compound +115.1% | max DD -6.8%
    WF folds: +1.5% / +64.5% / +28.8% (stability=0.18, ShIC=0.18)
    45.6% catch on 136 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 55.9%. confluence does NOT add value (best lift < 1.2x).

---

## SEI

- Catalog engines: 19 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(15), measure_event(4)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/te_in_btc(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=13
    Hit 61.5% | expectancy +2.33%/fire | compound +27.7% | max DD -8.9%
    WF folds: +0.0% / +13.3% / +12.7% (stability=0.29, ShIC=0.27)
    45.5% catch on 112 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 45.5%.

---

## SHIB

- Catalog engines: 27 total
- WF-tier basket: 0 members
- Catch-tier basket: 3 members
- Confluence pairs: 1

**Family distribution**: ta_event(16), measure_event(11)

### Catch-tier basket (3 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/bs_basis_z30(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=10
    Hit 50.0% | expectancy +0.94%/fire | compound +9.7% | max DD -1.4%
    WF folds: +0.0% / +0.9% / +8.7% (stability=-0.22, ShIC=0.28)
    58.3% catch on 96 fires (vs 25.3% random)

  - **measure_engines/rv_bpv_5m(op_abs_gt_thr_2.0)** | regime=bull | hold=1d | n_fires=10
    Hit 60.0% | expectancy +10.71%/fire | compound +142.5% | max DD -3.5%
    WF folds: +0.0% / +5.9% / +132.7% (stability=-0.33, ShIC=0.30)
    53.0% catch on 66 fires (vs 25.3% random)

  - **measure_engines/xd_momentum_rank(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=18
    Hit 66.7% | expectancy +1.09%/fire | compound +21.2% | max DD -1.3%
    WF folds: +5.6% / +2.3% / +12.2% (stability=0.38, ShIC=0.13)
    47.6% catch on 166 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 3 catch-tier engine(s), best catch_rate 58.3%. confluence does NOT add value (best lift < 1.2x).

---

## SOL

- Catalog engines: 11 total
- WF-tier basket: 0 members
- Catch-tier basket: 5 members
- Confluence pairs: 9

**Family distribution**: ta_event(5), ta_state(4), measure_event(2)

### Catch-tier basket (5 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/te_imb(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=12
    Hit 66.7% | expectancy +1.70%/fire | compound +21.9% | max DD -1.7%
    WF folds: +11.5% / +8.2% / +0.0% (stability=0.26, ShIC=0.18)
    61.8% catch on 131 fires (vs 25.3% random)

  - **MACD_threshold(f_12_s_21_g_9)** | regime=chop | hold=3d | n_fires=11
    Hit 63.6% | expectancy +3.67%/fire | compound +44.3% | max DD -6.6%
    WF folds: +10.1% / +22.2% / +15.0% (stability=0.69, ShIC=0.22)
    54.9% catch on 1348 fires (vs 25.3% random)

  - **MA_state_SMA_above(period_20)** | regime=chop | hold=3d | n_fires=14
    Hit 57.1% | expectancy +1.36%/fire | compound +17.6% | max DD -14.1%
    WF folds: +6.7% / +11.8% / +0.0% (stability=0.22, ShIC=0.27)
    54.2% catch on 203 fires (vs 25.3% random)

  - **Donchian_state_above_midline(period_20)** | regime=chop | hold=3d | n_fires=14
    Hit 57.1% | expectancy +1.36%/fire | compound +17.6% | max DD -14.1%
    WF folds: +6.7% / +2.1% / +0.0% (stability=0.05, ShIC=0.26)
    53.5% catch on 213 fires (vs 25.3% random)

  - **Distance_z_state(period_20_threshold_1.5)** | regime=bull | hold=3d | n_fires=10
    Hit 50.0% | expectancy +5.31%/fire | compound +58.6% | max DD -7.0%
    WF folds: +9.6% / +36.7% / +0.0% (stability=-0.01, ShIC=0.21)
    53.5% catch on 217 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 5 catch-tier engine(s), best catch_rate 61.8%. confluence does NOT add value (best lift < 1.2x).

---

## SUI

- Catalog engines: 24 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: ta_event(17), measure_event(5), ta_state(2)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/xd_ma_distance(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=17
    Hit 58.8% | expectancy +1.66%/fire | compound +31.5% | max DD -2.9%
    WF folds: +6.1% / +22.5% / +0.0% (stability=0.00, ShIC=0.26)
    46.3% catch on 162 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 46.3%.

---

## SUPER

- Catalog engines: 29 total
- WF-tier basket: 0 members
- Catch-tier basket: 4 members
- Confluence pairs: 0

**Family distribution**: ta_event(23), measure_event(4), ta_state(2)

### Catch-tier basket (4 engines, ranked by catch_rate × sqrt(n))

  - **measure_engines/te_imb(op_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=18
    Hit 66.7% | expectancy +6.97%/fire | compound +207.5% | max DD -5.0%
    WF folds: +6.7% / +172.3% / +0.0% (stability=-0.34, ShIC=0.25)
    61.1% catch on 144 fires (vs 25.3% random)

  - **OBV_zscore(p_50_t_1.0)** | regime=bull | hold=1d | n_fires=25
    Hit 64.0% | expectancy +3.46%/fire | compound +118.4% | max DD -9.3%
    WF folds: +4.7% / +23.6% / +10.5% (stability=0.39, ShIC=0.26)
    46.2% catch on 912 fires (vs 25.3% random)

  - **measure_engines/norm_deviation(op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=13
    Hit 69.2% | expectancy +2.28%/fire | compound +33.2% | max DD -5.2%
    WF folds: +10.5% / +14.7% / +0.0% (stability=0.26, ShIC=0.25)
    45.8% catch on 96 fires (vs 25.3% random)

  - **Distance_z_state(period_20_threshold_1.0)** | regime=chop | hold=1d | n_fires=18
    Hit 50.0% | expectancy +4.63%/fire | compound +88.1% | max DD -13.1%
    WF folds: +11.0% / +68.6% / +0.6% (stability=-0.12, ShIC=0.29)
    45.5% catch on 176 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 4 catch-tier engine(s), best catch_rate 61.1%.

---

## TRX

- Catalog engines: 2 total
- WF-tier basket: 0 members
- Catch-tier basket: 1 members
- Confluence pairs: 0

**Family distribution**: measure_event(1), ta_state(1)

### Catch-tier basket (1 engines, ranked by catch_rate × sqrt(n))

  - **Distance_z_state(period_50_threshold_1.5)** | regime=chop | hold=3d | n_fires=10
    Hit 60.0% | expectancy +1.52%/fire | compound +16.0% | max DD -2.2%
    WF folds: +4.0% / +12.3% / +0.0% (stability=0.06, ShIC=0.23)
    49.0% catch on 98 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 1 catch-tier engine(s), best catch_rate 49.0%.

---

## UNI

- Catalog engines: 21 total
- WF-tier basket: 0 members
- Catch-tier basket: 3 members
- Confluence pairs: 0

**Family distribution**: ta_event(9), ta_state(6), measure_event(5), confluence(1)

### Catch-tier basket (3 engines, ranked by catch_rate × sqrt(n))

  - **confluence_engines/UNI_pair_4(A_MA_state_EMA_above::period_20__AND_3b__B_measure_engines/hbr_eta_buy::op_abs_gt_thr_1.0)** | regime=chop | hold=1d | n_fires=12
    Hit 75.0% | expectancy +1.41%/fire | compound +17.8% | max DD -4.3%
    WF folds: +8.0% / +9.0% / +0.0% (stability=0.29, ShIC=0.30)
    52.6% catch on 76 fires (vs 25.3% random)

  - **measure_engines/liq_long_xsec_z(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=14
    Hit 78.6% | expectancy +1.39%/fire | compound +20.7% | max DD -4.3%
    WF folds: +3.9% / +9.2% / +0.0% (stability=0.13, ShIC=0.17)
    45.9% catch on 122 fires (vs 25.3% random)

  - **MA_state_EMA_above(period_20)** | regime=chop | hold=1d | n_fires=23
    Hit 69.6% | expectancy +1.02%/fire | compound +24.7% | max DD -7.0%
    WF folds: +16.0% / +7.4% / +0.0% (stability=0.16, ShIC=0.24)
    45.1% catch on 142 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 3 catch-tier engine(s), best catch_rate 52.6%.

---

## WLD

- Catalog engines: 8 total
- WF-tier basket: 0 members
- Catch-tier basket: 2 members
- Confluence pairs: 1

**Family distribution**: measure_event(5), ta_event(2), ta_state(1)

### Catch-tier basket (2 engines, ranked by catch_rate × sqrt(n))

  - **MA_state_SMA_above(period_50)** | regime=chop | hold=1d | n_fires=15
    Hit 60.0% | expectancy +2.24%/fire | compound +37.6% | max DD -4.2%
    WF folds: +26.4% / +8.9% / +0.0% (stability=0.07, ShIC=0.21)
    51.8% catch on 139 fires (vs 25.3% random)

  - **YZ_vol_regime(t_0.5)** | regime=bull | hold=1d | n_fires=10
    Hit 60.0% | expectancy +1.18%/fire | compound +11.2% | max DD -9.3%
    WF folds: +0.0% / +5.9% / +9.9% (stability=0.23, ShIC=0.29)
    45.4% catch on 207 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 0 WF-stable engines on this asset. 2 catch-tier engine(s), best catch_rate 51.8%. confluence does NOT add value (best lift < 1.2x).

---

## XRP

- Catalog engines: 90 total
- WF-tier basket: 4 members
- Catch-tier basket: 3 members
- Confluence pairs: 6

**Family distribution**: ta_event(80), measure_event(6), ta_state(4)

### Catch-tier basket (3 engines, ranked by catch_rate × sqrt(n))

  - **Bollinger_band_breach(p_50_k_1.5)** | regime=chop | hold=1d | n_fires=14
    Hit 64.3% | expectancy +1.20%/fire | compound +17.9% | max DD -1.7%
    WF folds: +8.6% / +8.0% / +0.0% (stability=0.29, ShIC=0.15)
    47.9% catch on 309 fires (vs 25.3% random)

  - **measure_engines/te_btc_imb(op_gt_thr_1.5)** | regime=bull | hold=1d | n_fires=10
    Hit 60.0% | expectancy +1.68%/fire | compound +17.2% | max DD -4.5%
    WF folds: +0.0% / +10.0% / +6.6% (stability=0.25, ShIC=0.30)
    46.8% catch on 79 fires (vs 25.3% random)

  - **RSI_threshold(p_7_lo_20_hi_60)** | regime=chop | hold=1d | n_fires=15
    Hit 73.3% | expectancy +1.24%/fire | compound +19.6% | max DD -8.4%
    WF folds: +7.8% / +3.6% / +0.0% (stability=0.16, ShIC=0.18)
    45.3% catch on 276 fires (vs 25.3% random)

### WF-tier additions (4 engines NOT already in catch-tier)

  - **MACD_threshold(f_12_s_35_g_9)** | regime=bull | hold=3d | n_fires=14
    Hit 57.1% | expectancy +1.68%/fire | compound +23.3% | max DD -9.1%
    WF folds: +15.1% / +10.6% / +12.9% (stability=0.85, ShIC=0.24)
    catch n/a

  - **MA_state_EMA_above(period_100)** | regime=bull | hold=3d | n_fires=13
    Hit 69.2% | expectancy +2.02%/fire | compound +26.7% | max DD -12.2%
    WF folds: +12.3% / +14.6% / +6.0% (stability=0.67, ShIC=0.16)
    catch n/a

  - **measure_engines/xd_btc_return(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=17
    Hit 52.9% | expectancy +1.32%/fire | compound +23.4% | max DD -4.7%
    WF folds: +10.7% / +8.1% / +3.1% (stability=0.57, ShIC=0.21)
    catch n/a

  - **measure_engines/norm_efficiency(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=15
    Hit 86.7% | expectancy +2.00%/fire | compound +33.9% | max DD -3.3%
    WF folds: +2.7% / +14.9% / +13.6% (stability=0.47, ShIC=0.16)
    catch n/a z_lift_t-3=+0.14

**Verdict (TRAIN-only)**: 4 engine(s) survive STRICT WF stability — top scout candidates. 3 catch-tier engine(s), best catch_rate 47.9%. confluence does NOT add value (best lift < 1.2x).

---

## ZEC

- Catalog engines: 8 total
- WF-tier basket: 1 members
- Catch-tier basket: 3 members
- Confluence pairs: 0

**Family distribution**: measure_event(6), ta_event(1), ta_state(1)

### Catch-tier basket (3 engines, ranked by catch_rate × sqrt(n))

  - **ETF_flow_z(t_0.5)** | regime=bull | hold=1d | n_fires=11
    Hit 54.5% | expectancy +1.79%/fire | compound +19.4% | max DD -10.0%
    WF folds: +0.0% / +5.4% / +13.3% (stability=0.12, ShIC=0.24)
    58.9% catch on 129 fires (vs 25.3% random)

  - **measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0)** | regime=bull | hold=1d | n_fires=15
    Hit 66.7% | expectancy +1.37%/fire | compound +21.7% | max DD -4.1%
    WF folds: +4.1% / +12.1% / +4.3% (stability=0.45, ShIC=0.19)
    51.0% catch on 192 fires (vs 25.3% random)

  - **measure_engines/te_btc_imb(op_abs_gt_thr_1.0)** | regime=bear | hold=1d | n_fires=12
    Hit 83.3% | expectancy +1.74%/fire | compound +22.8% | max DD -0.6%
    WF folds: +10.8% / +0.0% / +9.7% (stability=0.29, ShIC=0.10)
    50.0% catch on 92 fires (vs 25.3% random)

**Verdict (TRAIN-only)**: 1 engine(s) survive STRICT WF stability — top scout candidates. 3 catch-tier engine(s), best catch_rate 58.9%.

---
