# Per-Indicator Emergent Stories (2026-05-23T09:01)

Generated from `data/oracle/engine_catalog_discovery.parquet` (937 engines).

Per indicator class: story at 4 nested scopes (per-asset / per-group / per-asset×regime / per-group×regime).

---

## ATR_bands

- Engines: 11 total, 11 pass rigorous gate, 0 WF-eligible, 3 catch-tier
- Assets covered: 6

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| LINK | 1 | +37.4% | +33.5% | 0.13 |
| BNB | 1 | +32.3% | +40.4% | -0.03 |
| ADA | 3 | +31.3% | +40.8% | 0.31 |
| AVAX | 2 | +29.4% | +39.6% | 0.21 |
| BTC | 3 | +26.5% | +50.5% | 0.16 |
| ETH | 1 | +14.0% | +40.5% | -1.00 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 7 | 4 | +27.0% | +37.4% | +39.4% |
| BLUE | 4 | 2 | +22.5% | +26.5% | +48.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 9 | 5 | +24.8% | 0 |
| chop | 2 | 1 | +27.9% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 4 | 0 |
| STEADY | 5 | 2 |

### Best single engine of class

- **LINK** | config=p_14_k_1.5 | regime=bull | hold=3d | compound=+37.4% | DD=-12.8% | hit=53.8%

### Emergent story

**ATR_bands** has 11 engines across 6 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=STEADY. Weakest regime=bull. WF-eligible 0; catch-tier 3.

---

## Bollinger_band_breach

- Engines: 17 total, 17 pass rigorous gate, 0 WF-eligible, 1 catch-tier
- Assets covered: 10

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| OP | 2 | +54.7% | +41.2% | 0.29 |
| SUI | 3 | +38.2% | +39.1% | 0.11 |
| SUPER | 2 | +37.3% | +40.8% | 0.29 |
| HBAR | 1 | +36.9% | +39.9% | 0.26 |
| JST | 3 | +23.3% | +39.3% | 0.00 |
| XRP | 1 | +17.9% | +47.9% | 0.29 |
| ARB | 1 | +15.2% | +36.1% | -0.39 |
| ENJ | 2 | +13.6% | +36.5% | 0.22 |
| NEAR | 1 | +12.8% | +42.2% | -0.11 |
| UNI | 1 | +9.6% | +39.5% | -0.22 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 13 | 8 | +25.6% | +54.7% | +39.4% |
| DEGEN | 3 | 1 | +18.6% | +23.3% | +39.3% |
| STEADY | 1 | 1 | +17.9% | +17.9% | +47.9% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 13 | 9 | +19.0% | 0 |
| bull | 4 | 2 | +39.7% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| DEGEN | 0 | 3 |
| STEADY | 0 | 1 |
| VOLATILE | 4 | 9 |

### Best single engine of class

- **OP** | config=p_14_k_1.5 | regime=bull | hold=1d | compound=+54.7% | DD=-3.0% | hit=75.0%

### Emergent story

**Bollinger_band_breach** has 17 engines across 10 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 0; catch-tier 1.

---

## Distance_z_state

- Engines: 27 total, 27 pass rigorous gate, 3 WF-eligible, 11 catch-tier
- Assets covered: 22

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUPER | 2 | +316.5% | +43.5% | -0.12 |
| ICP | 1 | +167.0% | +50.0% | -0.31 |
| FET | 1 | +81.3% | +54.1% | 0.55 |
| PEPE | 1 | +66.2% | +37.8% | -0.33 |
| AVAX | 1 | +62.6% | +42.3% | 0.14 |
| APT | 1 | +62.6% | +52.0% | 0.19 |
| ALGO | 1 | +60.2% | +43.1% | -0.10 |
| SOL | 1 | +58.6% | +53.5% | -0.01 |
| DOT | 1 | +42.6% | +46.4% | -0.09 |
| ADA | 1 | +39.9% | +45.8% | -0.36 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 14 | 9 | +62.6% | +316.5% | +42.1% |
| STEADY | 8 | 8 | +40.6% | +62.6% | +44.3% |
| DEGEN | 5 | 5 | +44.0% | +81.3% | +42.6% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 19 | 16 | +61.3% | 3 |
| chop | 6 | 6 | +33.0% | 0 |
| bear | 2 | 2 | +29.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 3 | 2 |
| STEADY | 0 | 7 | 1 |
| VOLATILE | 2 | 9 | 3 |

### Best single engine of class

- **SUPER** | config=period_20_threshold_1.0 | regime=bull | hold=1d | compound=+316.5% | DD=-12.7% | hit=58.6%

### Emergent story

**Distance_z_state** has 27 engines across 22 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=bear. WF-eligible 3; catch-tier 11.

---

## Donchian_breakout

- Engines: 2 total, 2 pass rigorous gate, 0 WF-eligible, 0 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUPER | 1 | +41.0% | +35.7% | 0.21 |
| SHIB | 1 | +27.3% | +36.1% | 0.14 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 1 | 1 | +27.3% | +27.3% | +36.1% |
| VOLATILE | 1 | 1 | +41.0% | +41.0% | +35.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 2 | 2 | +34.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 1 |
| VOLATILE | 1 |

### Best single engine of class

- **SUPER** | config=p_10 | regime=bull | hold=1d | compound=+41.0% | DD=-9.3% | hit=75.0%

### Emergent story

**Donchian_breakout** has 2 engines across 2 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 0; catch-tier 0.

---

## Donchian_state_above_midline

- Engines: 10 total, 10 pass rigorous gate, 1 WF-eligible, 6 catch-tier
- Assets covered: 8

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| HBAR | 1 | +44.2% | +36.4% | 0.29 |
| JST | 1 | +41.7% | +40.8% | -0.12 |
| OP | 1 | +40.4% | +47.4% | 0.27 |
| UNI | 1 | +34.2% | +40.4% | 0.22 |
| BTC | 1 | +32.3% | +47.8% | 0.00 |
| LINK | 2 | +28.8% | +41.3% | 0.45 |
| PEPE | 2 | +28.3% | +54.4% | 0.29 |
| SOL | 1 | +17.6% | +53.5% | 0.05 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 3 | 2 | +29.8% | +41.7% | +49.9% |
| STEADY | 3 | 2 | +24.1% | +28.8% | +45.4% |
| VOLATILE | 3 | 3 | +39.6% | +44.2% | +41.4% |
| BLUE | 1 | 1 | +32.3% | +32.3% | +47.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 7 | 6 | +30.0% | 0 |
| bull | 3 | 3 | +34.3% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 1 | 0 |
| DEGEN | 1 | 2 |
| STEADY | 1 | 2 |
| VOLATILE | 0 | 3 |

### Best single engine of class

- **HBAR** | config=period_55 | regime=chop | hold=1d | compound=+44.2% | DD=-5.8% | hit=56.5%

### Emergent story

**Donchian_state_above_midline** has 10 engines across 8 assets. Dominant by count: regime=chop, bucket=DEGEN. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 1; catch-tier 6.

---

## ETF_flow_z

- Engines: 11 total, 11 pass rigorous gate, 0 WF-eligible, 4 catch-tier
- Assets covered: 11

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| WLD | 1 | +76.8% | +45.0% | 0.06 |
| FIL | 1 | +71.9% | +38.9% | 0.13 |
| SUPER | 1 | +49.1% | +40.3% | 0.29 |
| FLOKI | 1 | +45.9% | +47.5% | -0.19 |
| HBAR | 1 | +37.6% | +46.4% | 0.02 |
| DASH | 1 | +29.9% | +50.0% | 0.14 |
| ENJ | 1 | +26.0% | +37.6% | 0.12 |
| LTC | 1 | +21.3% | +35.9% | 0.16 |
| BNB | 1 | +20.6% | +33.6% | 0.10 |
| ZEC | 1 | +19.4% | +58.9% | 0.12 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 5 | 5 | +48.8% | +76.8% | +48.1% |
| VOLATILE | 4 | 4 | +32.8% | +49.1% | +39.9% |
| STEADY | 2 | 2 | +20.9% | +21.3% | +34.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 11 | 11 | +37.9% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 5 |
| STEADY | 2 |
| VOLATILE | 4 |

### Best single engine of class

- **WLD** | config=t_0.5 | regime=bull | hold=1d | compound=+76.8% | DD=-3.7% | hit=63.6%

### Emergent story

**ETF_flow_z** has 11 engines across 11 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bull. WF-eligible 0; catch-tier 4.

---

## Hawkes_branching_imbalance

- Engines: 1 total, 1 pass rigorous gate, 0 WF-eligible, 1 catch-tier
- Assets covered: 1

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| PEPE | 1 | +16.7% | +47.7% | 0.05 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 1 | 1 | +16.7% | +16.7% | +47.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 1 | 1 | +16.7% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | chop |
|---|:---:|
| DEGEN | 1 |

### Best single engine of class

- **PEPE** | config=t_0.1 | regime=chop | hold=1d | compound=+16.7% | DD=-8.3% | hit=63.6%

### Emergent story

**Hawkes_branching_imbalance** has 1 engines across 1 assets. Dominant by count: regime=chop, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=?. WF-eligible 0; catch-tier 1.

---

## Kyle_lambda_threshold

- Engines: 20 total, 20 pass rigorous gate, 1 WF-eligible, 4 catch-tier
- Assets covered: 14

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SHIB | 1 | +113.5% | +34.1% | -0.06 |
| ARKM | 2 | +104.1% | +45.8% | 0.29 |
| CHZ | 2 | +77.4% | +41.6% | 0.30 |
| HBAR | 1 | +59.0% | +44.5% | -0.21 |
| ETC | 2 | +57.6% | +40.5% | -0.18 |
| OP | 1 | +49.9% | +41.8% | 0.20 |
| ADA | 2 | +35.8% | +39.8% | 0.29 |
| CRV | 1 | +28.7% | +39.9% | -0.03 |
| BCH | 2 | +24.9% | +39.0% | 0.29 |
| FIL | 1 | +22.4% | +41.8% | 0.01 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 11 | 8 | +48.3% | +104.1% | +42.5% |
| STEADY | 6 | 3 | +33.0% | +57.6% | +39.8% |
| DEGEN | 3 | 3 | +52.5% | +113.5% | +38.4% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 13 | 10 | +48.9% | 1 |
| chop | 5 | 3 | +34.6% | 0 |
| bear | 2 | 2 | +39.1% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 3 | 0 |
| STEADY | 0 | 1 | 5 |
| VOLATILE | 2 | 9 | 0 |

### Best single engine of class

- **SHIB** | config=t_0.5 | regime=bull | hold=1d | compound=+113.5% | DD=-5.8% | hit=72.4%

### Emergent story

**Kyle_lambda_threshold** has 20 engines across 14 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=chop. WF-eligible 1; catch-tier 4.

---

## Liquidation_cascade

- Engines: 8 total, 8 pass rigorous gate, 1 WF-eligible, 1 catch-tier
- Assets covered: 7

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| LDO | 1 | +34.0% | +39.0% | 0.14 |
| ETC | 2 | +32.8% | +36.1% | -0.26 |
| XRP | 1 | +30.6% | +31.6% | 0.29 |
| LTC | 1 | +29.3% | +23.4% | 0.15 |
| APT | 1 | +17.4% | +47.6% | -0.01 |
| AVAX | 1 | +17.0% | +31.6% | -0.15 |
| BNB | 1 | +8.7% | +35.1% | 0.47 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 6 | 5 | +25.2% | +32.8% | +32.3% |
| VOLATILE | 2 | 2 | +25.7% | +34.0% | +43.3% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 5 | 5 | +24.0% | 1 |
| chop | 3 | 2 | +27.5% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| STEADY | 3 | 3 |
| VOLATILE | 2 | 0 |

### Best single engine of class

- **LDO** | config=t_1.0 | regime=bull | hold=1d | compound=+34.0% | DD=-4.5% | hit=80.0%

### Emergent story

**Liquidation_cascade** has 8 engines across 7 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 1; catch-tier 1.

---

## MACD_threshold

- Engines: 9 total, 9 pass rigorous gate, 1 WF-eligible, 7 catch-tier
- Assets covered: 3

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| LINK | 3 | +44.5% | +46.4% | 0.64 |
| SOL | 5 | +44.3% | +50.8% | 0.69 |
| XRP | 1 | +23.3% | +36.8% | 0.85 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 9 | 3 | +35.4% | +44.5% | +47.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 7 | 2 | +38.3% | 0 |
| bear | 1 | 1 | +27.2% | 0 |
| bull | 1 | 1 | +23.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| STEADY | 1 | 1 | 7 |

### Best single engine of class

- **LINK** | config=f_12_s_35_g_9 | regime=chop | hold=3d | compound=+44.5% | DD=-4.7% | hit=72.7%

### Emergent story

**MACD_threshold** has 9 engines across 3 assets. Dominant by count: regime=chop, bucket=STEADY. Highest mean compound on bucket=STEADY. Weakest regime=bull. WF-eligible 1; catch-tier 7.

---

## MA_state_EMA_above

- Engines: 19 total, 19 pass rigorous gate, 2 WF-eligible, 10 catch-tier
- Assets covered: 14

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUI | 1 | +73.0% | +39.0% | 0.24 |
| ALGO | 2 | +58.1% | +43.4% | -0.28 |
| OP | 1 | +39.1% | +45.8% | 0.27 |
| JST | 3 | +39.0% | +44.9% | 0.92 |
| XRP | 2 | +38.5% | +38.7% | 0.67 |
| HBAR | 1 | +34.9% | +27.4% | 0.19 |
| APT | 1 | +34.4% | +46.3% | 0.29 |
| LINK | 1 | +25.8% | +48.0% | 0.14 |
| UNI | 1 | +24.7% | +45.1% | 0.16 |
| PEPE | 1 | +23.3% | +56.1% | -0.22 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 7 | 7 | +32.8% | +73.0% | +42.2% |
| DEGEN | 6 | 3 | +24.1% | +39.0% | +46.5% |
| STEADY | 5 | 3 | +34.8% | +58.1% | +42.5% |
| BLUE | 1 | 1 | +13.8% | +13.8% | +46.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 10 | 9 | +23.7% | 1 |
| bull | 9 | 7 | +36.1% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 1 | 0 |
| DEGEN | 3 | 3 |
| STEADY | 3 | 2 |
| VOLATILE | 2 | 5 |

### Best single engine of class

- **SUI** | config=period_200 | regime=bull | hold=1d | compound=+73.0% | DD=-6.8% | hit=66.7%

### Emergent story

**MA_state_EMA_above** has 19 engines across 14 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=STEADY. Weakest regime=chop. WF-eligible 2; catch-tier 10.

---

## MA_state_SMA_above

- Engines: 26 total, 26 pass rigorous gate, 2 WF-eligible, 13 catch-tier
- Assets covered: 18

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| ALGO | 2 | +59.8% | +43.5% | 0.06 |
| AR | 1 | +52.3% | +49.4% | 0.07 |
| JST | 3 | +48.9% | +46.4% | 0.35 |
| PEPE | 3 | +40.1% | +55.7% | 0.29 |
| HBAR | 1 | +39.3% | +34.1% | 0.28 |
| LDO | 1 | +38.2% | +21.7% | -0.03 |
| WLD | 1 | +37.6% | +51.8% | 0.07 |
| BTC | 1 | +36.6% | +62.3% | 0.23 |
| XRP | 2 | +34.1% | +37.0% | 0.29 |
| OP | 1 | +31.2% | +49.6% | 0.10 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 9 | 5 | +30.4% | +59.8% | +41.7% |
| DEGEN | 8 | 4 | +29.8% | +48.9% | +50.4% |
| VOLATILE | 8 | 8 | +28.0% | +52.3% | +38.9% |
| BLUE | 1 | 1 | +36.6% | +36.6% | +62.3% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 16 | 12 | +27.3% | 0 |
| bull | 10 | 8 | +33.6% | 2 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 0 | 1 |
| DEGEN | 2 | 6 |
| STEADY | 5 | 4 |
| VOLATILE | 3 | 5 |

### Best single engine of class

- **ALGO** | config=period_50 | regime=bull | hold=1d | compound=+59.8% | DD=-13.0% | hit=56.8%

### Emergent story

**MA_state_SMA_above** has 26 engines across 18 assets. Dominant by count: regime=chop, bucket=STEADY. Highest mean compound on bucket=BLUE. Weakest regime=chop. WF-eligible 2; catch-tier 13.

---

## OBV_zscore

- Engines: 48 total, 48 pass rigorous gate, 8 WF-eligible, 11 catch-tier
- Assets covered: 20

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SEI | 3 | +147.7% | +40.1% | -0.13 |
| SUPER | 2 | +118.4% | +40.2% | 0.39 |
| AR | 5 | +81.1% | +46.9% | 0.05 |
| AAVE | 3 | +49.7% | +37.4% | 0.64 |
| TIA | 2 | +45.0% | +27.1% | 0.28 |
| LINK | 1 | +38.2% | +36.0% | 0.12 |
| ETC | 6 | +37.0% | +34.5% | 0.47 |
| SHIB | 1 | +30.9% | +36.6% | -0.39 |
| ADA | 4 | +30.6% | +44.6% | 0.26 |
| CHZ | 3 | +29.6% | +40.9% | 0.09 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 22 | 10 | +39.0% | +147.7% | +41.4% |
| STEADY | 17 | 6 | +24.1% | +38.2% | +39.2% |
| DEGEN | 8 | 3 | +18.1% | +30.9% | +43.5% |
| BLUE | 1 | 1 | +18.4% | +18.4% | +44.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 29 | 15 | +33.0% | 5 |
| chop | 19 | 9 | +25.0% | 3 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 1 | 0 |
| DEGEN | 1 | 7 |
| STEADY | 14 | 3 |
| VOLATILE | 13 | 9 |

### Best single engine of class

- **SEI** | config=p_30_t_1.0 | regime=chop | hold=1d | compound=+147.7% | DD=-12.1% | hit=76.2%

### Emergent story

**OBV_zscore** has 48 engines across 20 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 8; catch-tier 11.

---

## RSI_threshold

- Engines: 444 total, 444 pass rigorous gate, 28 WF-eligible, 57 catch-tier
- Assets covered: 27

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SEI | 12 | +130.3% | +38.8% | 0.40 |
| LINK | 10 | +54.9% | +43.0% | 0.28 |
| OP | 10 | +48.1% | +40.5% | 0.31 |
| HBAR | 19 | +43.0% | +35.2% | 0.29 |
| LDO | 7 | +40.6% | +39.2% | 0.29 |
| ICP | 6 | +37.4% | +40.1% | 0.49 |
| APT | 25 | +34.5% | +45.2% | 0.31 |
| XRP | 75 | +32.7% | +40.9% | 0.29 |
| DASH | 8 | +31.0% | +37.7% | 0.29 |
| ENJ | 78 | +29.2% | +29.4% | 0.54 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 221 | 14 | +20.7% | +130.3% | +36.4% |
| STEADY | 149 | 6 | +16.0% | +54.9% | +40.6% |
| DEGEN | 74 | 7 | +16.5% | +31.0% | +40.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 332 | 22 | +17.9% | 27 |
| bull | 79 | 11 | +22.7% | 1 |
| bear | 33 | 4 | +13.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 13 | 11 | 50 |
| STEADY | 17 | 13 | 119 |
| VOLATILE | 3 | 55 | 163 |

### Best single engine of class

- **SEI** | config=p_8_lo_40_hi_75 | regime=bull | hold=3d | compound=+130.3% | DD=-14.6% | hit=70.0%

### Emergent story

**RSI_threshold** has 444 engines across 27 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=bear. WF-eligible 28; catch-tier 57.

---

## VPIN_threshold

- Engines: 15 total, 15 pass rigorous gate, 3 WF-eligible, 4 catch-tier
- Assets covered: 12

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUPER | 1 | +94.0% | +44.4% | 0.14 |
| FET | 2 | +88.9% | +46.7% | 0.27 |
| LINK | 3 | +66.4% | +27.7% | 0.50 |
| ARKM | 1 | +55.2% | +49.8% | 0.28 |
| ADA | 1 | +49.3% | +48.8% | 0.28 |
| DOGE | 1 | +44.9% | +28.7% | -0.00 |
| APT | 1 | +42.8% | +33.2% | 0.80 |
| ICP | 1 | +24.6% | +37.7% | -0.09 |
| ALGO | 1 | +19.7% | +33.2% | 0.09 |
| AAVE | 1 | +13.3% | +50.2% | 0.35 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 7 | 7 | +40.4% | +94.0% | +40.1% |
| STEADY | 6 | 4 | +40.5% | +66.4% | +34.6% |
| DEGEN | 2 | 1 | +58.8% | +88.9% | +46.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 7 | 5 | +59.9% | 1 |
| chop | 6 | 6 | +26.6% | 2 |
| bear | 2 | 2 | +32.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 1 | 1 |
| STEADY | 1 | 4 | 1 |
| VOLATILE | 1 | 2 | 4 |

### Best single engine of class

- **SUPER** | config=t_0.5 | regime=bull | hold=3d | compound=+94.0% | DD=-6.0% | hit=76.9%

### Emergent story

**VPIN_threshold** has 15 engines across 12 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=chop. WF-eligible 3; catch-tier 4.

---

## VWAP_state_above

- Engines: 4 total, 4 pass rigorous gate, 0 WF-eligible, 2 catch-tier
- Assets covered: 3

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| ADA | 2 | +62.5% | +46.9% | -0.13 |
| LINK | 1 | +42.2% | +30.6% | -0.02 |
| SOL | 1 | +18.4% | +31.1% | 0.14 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 4 | 3 | +38.3% | +62.5% | +38.9% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 3 | 2 | +44.9% | 0 |
| bear | 1 | 1 | +18.4% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull |
|---|:---:|:---:|
| STEADY | 1 | 3 |

### Best single engine of class

- **ADA** | config=period_20 | regime=bull | hold=3d | compound=+62.5% | DD=-11.1% | hit=75.0%

### Emergent story

**VWAP_state_above** has 4 engines across 3 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=STEADY. Weakest regime=bear. WF-eligible 0; catch-tier 2.

---

## YZ_vol_regime

- Engines: 1 total, 1 pass rigorous gate, 0 WF-eligible, 1 catch-tier
- Assets covered: 1

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| WLD | 1 | +11.2% | +45.4% | 0.23 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 1 | 1 | +11.2% | +11.2% | +45.4% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 1 | 1 | +11.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 1 |

### Best single engine of class

- **WLD** | config=t_0.5 | regime=bull | hold=1d | compound=+11.2% | DD=-9.3% | hit=60.0%

### Emergent story

**YZ_vol_regime** has 1 engines across 1 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=?. WF-eligible 0; catch-tier 1.

---

## confluence_engines/UNI_pair_4

- Engines: 1 total, 1 pass rigorous gate, 0 WF-eligible, 1 catch-tier
- Assets covered: 1

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| UNI | 1 | +17.8% | +52.6% | 0.29 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 1 | 1 | +17.8% | +17.8% | +52.6% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 1 | 1 | +17.8% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | chop |
|---|:---:|
| VOLATILE | 1 |

### Best single engine of class

- **UNI** | config=A_MA_state_EMA_above::period_20__AND_3b__B_measure_engines/hbr_eta_buy::op_abs_gt_thr_1.0 | regime=chop | hold=1d | compound=+17.8% | DD=-4.3% | hit=75.0%

### Emergent story

**confluence_engines/UNI_pair_4** has 1 engines across 1 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=?. WF-eligible 0; catch-tier 1.

---

## measure_engines/bd_imbalance_l1

- Engines: 5 total, 5 pass rigorous gate, 1 WF-eligible, 1 catch-tier
- Assets covered: 5

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SHIB | 1 | +170.6% | +38.2% | -0.14 |
| ICP | 1 | +58.6% | +54.0% | 0.59 |
| BLUR | 1 | +27.0% | +31.8% | 0.29 |
| APT | 1 | +15.4% | +31.3% | 0.19 |
| LDO | 1 | +11.0% | +40.0% | 0.04 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 3 | 3 | +28.4% | +58.6% | +41.8% |
| DEGEN | 2 | 2 | +98.8% | +170.6% | +35.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 3 | 3 | +85.4% | 1 |
| chop | 2 | 2 | +13.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| DEGEN | 2 | 0 |
| VOLATILE | 1 | 2 |

### Best single engine of class

- **SHIB** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+170.6% | DD=-2.0% | hit=63.6%

### Emergent story

**measure_engines/bd_imbalance_l1** has 5 engines across 5 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=chop. WF-eligible 1; catch-tier 1.

---

## measure_engines/bd_imbalance_l5

- Engines: 4 total, 4 pass rigorous gate, 1 WF-eligible, 3 catch-tier
- Assets covered: 4

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUI | 1 | +49.8% | +41.0% | 0.27 |
| PEPE | 1 | +38.1% | +52.7% | 0.29 |
| FIL | 1 | +31.1% | +45.7% | 0.29 |
| ZEC | 1 | +21.7% | +51.0% | 0.45 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 3 | 3 | +30.3% | +38.1% | +49.8% |
| VOLATILE | 1 | 1 | +49.8% | +49.8% | +41.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 2 | 2 | +26.4% | 1 |
| chop | 2 | 2 | +43.9% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| DEGEN | 2 | 1 |
| VOLATILE | 0 | 1 |

### Best single engine of class

- **SUI** | config=op_lt_thr_1.0 | regime=chop | hold=1d | compound=+49.8% | DD=-2.1% | hit=81.8%

### Emergent story

**measure_engines/bd_imbalance_l5** has 4 engines across 4 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 1; catch-tier 3.

---

## measure_engines/bs_basis_delta_1d

- Engines: 9 total, 9 pass rigorous gate, 0 WF-eligible, 0 catch-tier
- Assets covered: 9

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUI | 1 | +69.8% | +36.4% | 0.25 |
| ALGO | 1 | +45.6% | +44.8% | -0.08 |
| SOL | 1 | +42.0% | +44.8% | -0.18 |
| DYDX | 1 | +27.2% | +39.5% | 0.25 |
| LDO | 1 | +26.9% | +40.7% | 0.23 |
| CHZ | 1 | +25.5% | +38.6% | -0.15 |
| DASH | 1 | +22.0% | +33.8% | 0.19 |
| ETH | 1 | +17.9% | +34.8% | 0.16 |
| TRX | 1 | +10.7% | +33.8% | 0.29 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 4 | 4 | +37.4% | +69.8% | +38.8% |
| STEADY | 3 | 3 | +32.8% | +45.6% | +41.1% |
| BLUE | 1 | 1 | +17.9% | +17.9% | +34.8% |
| DEGEN | 1 | 1 | +22.0% | +22.0% | +33.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 5 | 5 | +41.0% | 0 |
| bear | 3 | 3 | +24.0% | 0 |
| chop | 1 | 1 | +10.7% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| BLUE | 1 | 0 | 0 |
| DEGEN | 0 | 1 | 0 |
| STEADY | 0 | 2 | 1 |
| VOLATILE | 2 | 2 | 0 |

### Best single engine of class

- **SUI** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+69.8% | DD=-8.0% | hit=68.8%

### Emergent story

**measure_engines/bs_basis_delta_1d** has 9 engines across 9 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 0; catch-tier 0.

---

## measure_engines/bs_basis_z30

- Engines: 8 total, 8 pass rigorous gate, 0 WF-eligible, 5 catch-tier
- Assets covered: 6

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| ARB | 1 | +35.1% | +47.2% | -0.09 |
| ETC | 2 | +31.0% | +34.1% | 0.25 |
| AAVE | 2 | +29.1% | +47.8% | 0.27 |
| OP | 1 | +20.3% | +55.1% | -0.29 |
| SHIB | 1 | +9.7% | +58.3% | -0.22 |
| DOT | 1 | +8.4% | +36.3% | -0.31 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 4 | 3 | +26.0% | +35.1% | +49.5% |
| STEADY | 3 | 2 | +18.5% | +31.0% | +34.8% |
| DEGEN | 1 | 1 | +9.7% | +9.7% | +58.3% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 5 | 4 | +17.4% | 0 |
| bull | 2 | 2 | +33.0% | 0 |
| bear | 1 | 1 | +16.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 0 | 1 |
| STEADY | 1 | 1 | 1 |
| VOLATILE | 0 | 1 | 3 |

### Best single engine of class

- **ARB** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | hit=58.8%

### Emergent story

**measure_engines/bs_basis_z30** has 8 engines across 6 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=bear. WF-eligible 0; catch-tier 5.

---

## measure_engines/hbr_eta_buy

- Engines: 13 total, 13 pass rigorous gate, 0 WF-eligible, 3 catch-tier
- Assets covered: 11

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FLOKI | 2 | +92.6% | +52.4% | 0.28 |
| XRP | 2 | +42.7% | +35.8% | 0.27 |
| FET | 1 | +33.2% | +46.9% | 0.11 |
| BLUR | 1 | +27.3% | +26.1% | -0.03 |
| ARB | 1 | +20.2% | +12.7% | 0.27 |
| UNI | 1 | +19.0% | +42.2% | 0.26 |
| XLM | 1 | +15.6% | +37.3% | -0.03 |
| DYDX | 1 | +13.6% | +36.8% | 0.28 |
| FIL | 1 | +13.2% | +28.7% | -0.30 |
| LTC | 1 | +12.5% | +35.5% | 0.29 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 5 | 4 | +42.2% | +92.6% | +41.3% |
| STEADY | 4 | 3 | +26.3% | +42.7% | +35.8% |
| VOLATILE | 4 | 4 | +17.1% | +20.2% | +32.3% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 6 | 5 | +27.1% | 0 |
| chop | 6 | 5 | +35.0% | 0 |
| bear | 1 | 1 | +12.5% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 1 | 4 |
| STEADY | 1 | 2 | 1 |
| VOLATILE | 0 | 3 | 1 |

### Best single engine of class

- **FLOKI** | config=op_abs_gt_thr_1.0 | regime=chop | hold=3d | compound=+92.6% | DD=-1.3% | hit=72.7%

### Emergent story

**measure_engines/hbr_eta_buy** has 13 engines across 11 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 0; catch-tier 3.

---

## measure_engines/hbr_eta_total

- Engines: 13 total, 13 pass rigorous gate, 0 WF-eligible, 2 catch-tier
- Assets covered: 11

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FLOKI | 1 | +35.2% | +45.6% | 0.11 |
| BLUR | 1 | +27.3% | +28.4% | -0.03 |
| AAVE | 2 | +22.8% | +46.1% | 0.23 |
| CRV | 1 | +20.8% | +22.6% | 0.29 |
| DYDX | 1 | +19.1% | +38.9% | -0.03 |
| FIL | 1 | +18.7% | +31.6% | 0.11 |
| LTC | 1 | +12.5% | +40.0% | 0.29 |
| DASH | 1 | +12.2% | +34.7% | -0.40 |
| DOT | 2 | +11.5% | +36.1% | 0.29 |
| ZEC | 1 | +11.3% | +37.6% | 0.14 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 5 | 5 | +20.9% | +35.2% | +35.6% |
| VOLATILE | 5 | 4 | +17.1% | +22.8% | +37.2% |
| STEADY | 3 | 2 | +11.8% | +12.5% | +37.4% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 9 | 7 | +17.2% | 0 |
| bull | 3 | 3 | +19.5% | 0 |
| bear | 1 | 1 | +12.5% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 2 | 3 |
| STEADY | 1 | 0 | 2 |
| VOLATILE | 0 | 1 | 4 |

### Best single engine of class

- **FLOKI** | config=op_abs_gt_thr_1.0 | regime=chop | hold=3d | compound=+35.2% | DD=-4.4% | hit=61.5%

### Emergent story

**measure_engines/hbr_eta_total** has 13 engines across 11 assets. Dominant by count: regime=chop, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 0; catch-tier 2.

---

## measure_engines/liq_long_usd

- Engines: 3 total, 3 pass rigorous gate, 0 WF-eligible, 3 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FIL | 2 | +38.6% | +48.1% | -0.36 |
| LINK | 1 | +34.7% | +46.0% | 0.38 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 2 | 1 | +38.6% | +38.6% | +48.1% |
| STEADY | 1 | 1 | +34.7% | +34.7% | +46.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 3 | 2 | +37.3% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 2 |
| STEADY | 1 |

### Best single engine of class

- **FIL** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+38.6% | DD=-4.9% | hit=60.0%

### Emergent story

**measure_engines/liq_long_usd** has 3 engines across 2 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bull. WF-eligible 0; catch-tier 3.

---

## measure_engines/liq_long_xsec_z

- Engines: 9 total, 9 pass rigorous gate, 2 WF-eligible, 1 catch-tier
- Assets covered: 8

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FLOKI | 1 | +66.4% | +35.9% | 0.68 |
| FIL | 1 | +29.5% | +22.1% | 0.83 |
| XLM | 1 | +29.1% | +37.7% | 0.02 |
| LINK | 2 | +25.5% | +36.2% | 0.27 |
| ARB | 1 | +23.0% | +30.2% | 0.74 |
| CHZ | 1 | +22.7% | +30.4% | 0.25 |
| UNI | 1 | +20.7% | +45.9% | 0.13 |
| DASH | 1 | +20.3% | +34.4% | 0.26 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 4 | 4 | +23.9% | +29.1% | +36.1% |
| DEGEN | 3 | 3 | +38.7% | +66.4% | +30.8% |
| STEADY | 2 | 1 | +23.6% | +25.5% | +36.2% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 6 | 6 | +31.5% | 1 |
| chop | 3 | 2 | +23.4% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| DEGEN | 3 | 0 |
| STEADY | 0 | 2 |
| VOLATILE | 3 | 1 |

### Best single engine of class

- **FLOKI** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+66.4% | DD=-3.9% | hit=80.0%

### Emergent story

**measure_engines/liq_long_xsec_z** has 9 engines across 8 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=chop. WF-eligible 2; catch-tier 1.

---

## measure_engines/liq_long_z30

- Engines: 1 total, 1 pass rigorous gate, 0 WF-eligible, 0 catch-tier
- Assets covered: 1

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| PEPE | 1 | +16.0% | +41.0% | -0.23 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 1 | 1 | +16.0% | +16.0% | +41.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 1 | 1 | +16.0% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | chop |
|---|:---:|
| DEGEN | 1 |

### Best single engine of class

- **PEPE** | config=op_abs_gt_thr_1.0 | regime=chop | hold=1d | compound=+16.0% | DD=-7.3% | hit=64.3%

### Emergent story

**measure_engines/liq_long_z30** has 1 engines across 1 assets. Dominant by count: regime=chop, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=?. WF-eligible 0; catch-tier 0.

---

## measure_engines/liq_short_usd

- Engines: 3 total, 3 pass rigorous gate, 0 WF-eligible, 0 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| BNB | 2 | +23.9% | +44.1% | 0.27 |
| AVAX | 1 | +20.1% | +42.6% | 0.10 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 3 | 2 | +19.8% | +23.9% | +43.6% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 2 | 1 | +19.6% | 0 |
| bear | 1 | 1 | +20.1% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull |
|---|:---:|:---:|
| STEADY | 1 | 2 |

### Best single engine of class

- **BNB** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+23.9% | DD=-4.8% | hit=56.2%

### Emergent story

**measure_engines/liq_short_usd** has 3 engines across 2 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=STEADY. Weakest regime=bull. WF-eligible 0; catch-tier 0.

---

## measure_engines/liq_short_z30

- Engines: 3 total, 3 pass rigorous gate, 1 WF-eligible, 1 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| ADA | 2 | +55.0% | +46.5% | -0.30 |
| ETH | 1 | +17.4% | +44.6% | 0.47 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 2 | 1 | +45.2% | +55.0% | +46.5% |
| BLUE | 1 | 1 | +17.4% | +17.4% | +44.6% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 3 | 2 | +36.0% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| BLUE | 1 |
| STEADY | 2 |

### Best single engine of class

- **ADA** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+55.0% | DD=-8.0% | hit=71.4%

### Emergent story

**measure_engines/liq_short_z30** has 3 engines across 2 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=STEADY. Weakest regime=bull. WF-eligible 1; catch-tier 1.

---

## measure_engines/norm_deviation

- Engines: 12 total, 12 pass rigorous gate, 0 WF-eligible, 3 catch-tier
- Assets covered: 9

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| HBAR | 3 | +47.1% | +44.8% | 0.29 |
| ALGO | 2 | +44.8% | +37.1% | 0.23 |
| SEI | 1 | +37.7% | +7.9% | 0.21 |
| OP | 1 | +33.8% | +43.4% | 0.41 |
| SUPER | 1 | +33.2% | +45.8% | 0.26 |
| DYDX | 1 | +17.4% | +38.6% | 0.19 |
| BTC | 1 | +15.4% | +41.8% | 0.25 |
| DOT | 1 | +12.2% | +33.9% | 0.00 |
| DOGE | 1 | +10.9% | +41.4% | -0.26 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 8 | 6 | +32.0% | +47.1% | +38.9% |
| STEADY | 3 | 2 | +23.8% | +44.8% | +36.0% |
| BLUE | 1 | 1 | +15.4% | +15.4% | +41.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 10 | 8 | +28.3% | 0 |
| bull | 2 | 2 | +30.1% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 1 | 0 |
| STEADY | 1 | 2 |
| VOLATILE | 0 | 8 |

### Best single engine of class

- **HBAR** | config=op_abs_gt_thr_1.0 | regime=chop | hold=1d | compound=+47.1% | DD=-4.1% | hit=63.2%

### Emergent story

**measure_engines/norm_deviation** has 12 engines across 9 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 0; catch-tier 3.

---

## measure_engines/norm_efficiency

- Engines: 16 total, 16 pass rigorous gate, 2 WF-eligible, 6 catch-tier
- Assets covered: 11

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| HBAR | 2 | +49.7% | +62.6% | -0.07 |
| ARB | 2 | +40.5% | +44.8% | 0.17 |
| ZEC | 3 | +36.7% | +32.4% | 0.25 |
| SUPER | 1 | +34.9% | +38.6% | 0.02 |
| XRP | 1 | +33.9% | +40.2% | 0.47 |
| FIL | 1 | +33.0% | +49.5% | 0.53 |
| DOT | 1 | +30.7% | +51.8% | 0.26 |
| SUI | 1 | +23.7% | +41.8% | 0.09 |
| OP | 1 | +19.7% | +38.6% | 0.29 |
| ALGO | 1 | +14.9% | +29.9% | 0.26 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 7 | 5 | +32.3% | +49.7% | +47.7% |
| STEADY | 5 | 4 | +20.1% | +33.9% | +44.4% |
| DEGEN | 4 | 2 | +24.6% | +36.7% | +36.6% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 7 | 5 | +33.8% | 1 |
| chop | 6 | 3 | +19.9% | 1 |
| bear | 3 | 3 | +23.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 0 | 4 |
| STEADY | 1 | 2 | 2 |
| VOLATILE | 2 | 5 | 0 |

### Best single engine of class

- **HBAR** | config=op_gt_thr_1.0 | regime=bull | hold=1d | compound=+49.7% | DD=-1.8% | hit=72.7%

### Emergent story

**measure_engines/norm_efficiency** has 16 engines across 11 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 2; catch-tier 6.

---

## measure_engines/norm_flow_imbalance

- Engines: 12 total, 12 pass rigorous gate, 3 WF-eligible, 6 catch-tier
- Assets covered: 8

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FET | 3 | +95.0% | +76.1% | -0.32 |
| SUI | 1 | +67.4% | +41.7% | -0.02 |
| JST | 2 | +54.6% | +40.3% | 0.43 |
| ALGO | 2 | +44.3% | +40.5% | 0.26 |
| DOGE | 1 | +38.2% | +57.4% | -0.02 |
| AVAX | 1 | +38.1% | +52.6% | 0.15 |
| DYDX | 1 | +24.4% | +40.5% | 0.44 |
| ETC | 1 | +11.9% | +48.5% | -0.04 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 5 | 2 | +63.8% | +95.0% | +61.8% |
| STEADY | 4 | 3 | +26.5% | +44.3% | +45.5% |
| VOLATILE | 3 | 3 | +43.4% | +67.4% | +46.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 9 | 6 | +54.8% | 3 |
| bear | 2 | 2 | +11.7% | 0 |
| chop | 1 | 1 | +38.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 5 | 0 |
| STEADY | 2 | 2 | 0 |
| VOLATILE | 0 | 2 | 1 |

### Best single engine of class

- **FET** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+95.0% | DD=-7.7% | hit=65.2%

### Emergent story

**measure_engines/norm_flow_imbalance** has 12 engines across 8 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 3; catch-tier 6.

---

## measure_engines/norm_funding_momentum

- Engines: 1 total, 1 pass rigorous gate, 0 WF-eligible, 0 catch-tier
- Assets covered: 1

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| XLM | 1 | +18.7% | +34.0% | 0.29 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 1 | 1 | +18.7% | +18.7% | +34.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 1 | 1 | +18.7% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| VOLATILE | 1 |

### Best single engine of class

- **XLM** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+18.7% | DD=-8.5% | hit=58.8%

### Emergent story

**measure_engines/norm_funding_momentum** has 1 engines across 1 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=?. WF-eligible 0; catch-tier 0.

---

## measure_engines/rv_bpv_5m

- Engines: 7 total, 7 pass rigorous gate, 1 WF-eligible, 6 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SHIB | 5 | +142.5% | +48.5% | -0.33 |
| FET | 2 | +77.4% | +62.5% | 0.65 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 7 | 2 | +122.8% | +142.5% | +52.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 7 | 2 | +122.8% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 7 |

### Best single engine of class

- **SHIB** | config=op_abs_gt_thr_1.5 | regime=bull | hold=1d | compound=+142.5% | DD=-3.5% | hit=60.0%

### Emergent story

**measure_engines/rv_bpv_5m** has 7 engines across 2 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bull. WF-eligible 1; catch-tier 6.

---

## measure_engines/rv_jump_count

- Engines: 4 total, 4 pass rigorous gate, 0 WF-eligible, 4 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| OP | 2 | +31.6% | +61.5% | 0.23 |
| BCH | 2 | +25.8% | +51.5% | 0.29 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 2 | 1 | +25.8% | +25.8% | +51.5% |
| VOLATILE | 2 | 1 | +31.6% | +31.6% | +61.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 2 | 1 | +25.8% | 0 |
| chop | 2 | 1 | +31.6% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| STEADY | 2 | 0 |
| VOLATILE | 0 | 2 |

### Best single engine of class

- **OP** | config=op_abs_gt_thr_1.0 | regime=chop | hold=1d | compound=+31.6% | DD=-1.5% | hit=90.0%

### Emergent story

**measure_engines/rv_jump_count** has 4 engines across 2 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 0; catch-tier 4.

---

## measure_engines/rv_jump_frac

- Engines: 7 total, 7 pass rigorous gate, 0 WF-eligible, 3 catch-tier
- Assets covered: 5

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| HBAR | 1 | +38.7% | +43.7% | 0.26 |
| ICP | 1 | +27.0% | +45.9% | -0.26 |
| ETC | 2 | +24.1% | +41.2% | 0.06 |
| AAVE | 2 | +23.4% | +76.1% | 0.14 |
| AR | 1 | +14.3% | +41.9% | -0.07 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 5 | 4 | +25.4% | +38.7% | +56.7% |
| STEADY | 2 | 1 | +23.9% | +24.1% | +41.2% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 7 | 5 | +25.0% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | chop |
|---|:---:|
| STEADY | 2 |
| VOLATILE | 5 |

### Best single engine of class

- **HBAR** | config=op_gt_thr_1.0 | regime=chop | hold=1d | compound=+38.7% | DD=-1.9% | hit=71.4%

### Emergent story

**measure_engines/rv_jump_frac** has 7 engines across 5 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 0; catch-tier 3.

---

## measure_engines/rv_rv_5m

- Engines: 5 total, 5 pass rigorous gate, 1 WF-eligible, 3 catch-tier
- Assets covered: 3

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SHIB | 1 | +142.5% | +42.9% | -0.33 |
| AR | 2 | +109.1% | +48.5% | 0.14 |
| FET | 2 | +75.8% | +60.0% | 0.67 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 3 | 2 | +95.2% | +142.5% | +54.3% |
| VOLATILE | 2 | 1 | +107.7% | +109.1% | +48.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 5 | 3 | +100.2% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 3 |
| VOLATILE | 2 |

### Best single engine of class

- **SHIB** | config=op_gt_thr_1.0 | regime=bull | hold=1d | compound=+142.5% | DD=-3.5% | hit=60.0%

### Emergent story

**measure_engines/rv_rv_5m** has 5 engines across 3 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 1; catch-tier 3.

---

## measure_engines/stbl_total_zscore_30d

- Engines: 2 total, 2 pass rigorous gate, 0 WF-eligible, 2 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| BCH | 1 | +27.6% | +50.0% | 0.46 |
| BTC | 1 | +15.9% | +55.7% | -0.01 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| BLUE | 1 | 1 | +15.9% | +15.9% | +55.7% |
| STEADY | 1 | 1 | +27.6% | +27.6% | +50.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 2 | 2 | +21.7% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| BLUE | 1 |
| STEADY | 1 |

### Best single engine of class

- **BCH** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | hit=54.5%

### Emergent story

**measure_engines/stbl_total_zscore_30d** has 2 engines across 2 assets. Dominant by count: regime=bull, bucket=BLUE. Highest mean compound on bucket=STEADY. Weakest regime=bull. WF-eligible 0; catch-tier 2.

---

## measure_engines/te_btc_imb

- Engines: 21 total, 21 pass rigorous gate, 0 WF-eligible, 7 catch-tier
- Assets covered: 16

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| ICP | 2 | +60.3% | +45.0% | 0.18 |
| NEAR | 2 | +35.7% | +40.1% | 0.27 |
| UNI | 1 | +30.2% | +32.6% | 0.22 |
| ETH | 1 | +29.7% | +42.9% | 0.29 |
| HBAR | 1 | +28.6% | +51.0% | 0.09 |
| BNB | 2 | +27.9% | +61.8% | -0.09 |
| WLD | 1 | +25.7% | +36.6% | 0.27 |
| ZEC | 1 | +22.8% | +50.0% | 0.29 |
| AR | 1 | +22.4% | +48.5% | 0.29 |
| ENJ | 1 | +22.0% | +30.3% | 0.17 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 12 | 8 | +25.9% | +60.3% | +39.3% |
| STEADY | 6 | 5 | +16.0% | +27.9% | +48.8% |
| DEGEN | 2 | 2 | +24.3% | +25.7% | +43.3% |
| BLUE | 1 | 1 | +29.7% | +29.7% | +42.9% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 9 | 8 | +27.3% | 0 |
| bear | 6 | 6 | +19.5% | 0 |
| chop | 6 | 4 | +20.4% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| BLUE | 0 | 1 | 0 |
| DEGEN | 1 | 1 | 0 |
| STEADY | 3 | 2 | 1 |
| VOLATILE | 2 | 5 | 5 |

### Best single engine of class

- **ICP** | config=op_abs_gt_thr_1.0 | regime=bull | hold=3d | compound=+60.3% | DD=-14.1% | hit=75.0%

### Emergent story

**measure_engines/te_btc_imb** has 21 engines across 16 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=BLUE. Weakest regime=bear. WF-eligible 0; catch-tier 7.

---

## measure_engines/te_imb

- Engines: 19 total, 19 pass rigorous gate, 1 WF-eligible, 4 catch-tier
- Assets covered: 13

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| SUPER | 2 | +207.5% | +45.4% | -0.05 |
| WLD | 1 | +197.4% | +44.8% | -0.41 |
| DOGE | 3 | +57.9% | +40.2% | 0.35 |
| ADA | 1 | +52.6% | +51.4% | 0.12 |
| FET | 1 | +36.4% | +46.6% | -0.28 |
| SEI | 1 | +34.7% | +38.4% | -0.22 |
| SOL | 1 | +21.9% | +61.8% | 0.26 |
| PEPE | 1 | +21.5% | +42.4% | 0.25 |
| CHZ | 1 | +16.7% | +44.6% | 0.14 |
| CRV | 1 | +16.3% | +39.6% | 0.28 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 12 | 6 | +43.1% | +207.5% | +38.8% |
| DEGEN | 4 | 4 | +65.9% | +197.4% | +43.4% |
| STEADY | 3 | 3 | +29.0% | +52.6% | +46.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 8 | 7 | +78.6% | 1 |
| bear | 7 | 4 | +16.6% | 0 |
| chop | 4 | 4 | +30.6% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 1 | 1 | 2 |
| STEADY | 0 | 2 | 1 |
| VOLATILE | 6 | 5 | 1 |

### Best single engine of class

- **SUPER** | config=op_gt_thr_1.0 | regime=bull | hold=1d | compound=+207.5% | DD=-5.0% | hit=66.7%

### Emergent story

**measure_engines/te_imb** has 19 engines across 13 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 1; catch-tier 4.

---

## measure_engines/te_in_btc

- Engines: 14 total, 14 pass rigorous gate, 2 WF-eligible, 4 catch-tier
- Assets covered: 11

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| WLD | 1 | +236.4% | +41.7% | 0.17 |
| ICP | 2 | +58.2% | +51.4% | 0.97 |
| AR | 1 | +44.7% | +51.7% | -0.17 |
| ALGO | 1 | +29.1% | +41.2% | 0.00 |
| SEI | 1 | +27.7% | +45.5% | 0.29 |
| ENJ | 1 | +24.7% | +33.3% | 0.13 |
| NEAR | 1 | +23.2% | +36.4% | -0.35 |
| CRV | 1 | +22.8% | +18.3% | 0.49 |
| UNI | 2 | +18.2% | +37.4% | -0.05 |
| ARB | 1 | +17.3% | +33.6% | 0.15 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 10 | 8 | +30.2% | +58.2% | +39.6% |
| STEADY | 3 | 2 | +19.2% | +29.1% | +36.1% |
| DEGEN | 1 | 1 | +236.4% | +236.4% | +41.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 7 | 6 | +66.1% | 1 |
| chop | 6 | 5 | +20.2% | 1 |
| bear | 1 | 1 | +12.1% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 0 | 1 | 0 |
| STEADY | 0 | 0 | 3 |
| VOLATILE | 1 | 6 | 3 |

### Best single engine of class

- **WLD** | config=op_abs_gt_thr_1.0 | regime=bull | hold=3d | compound=+236.4% | DD=-4.3% | hit=70.0%

### Emergent story

**measure_engines/te_in_btc** has 14 engines across 11 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 2; catch-tier 4.

---

## measure_engines/wh_whale_net_usd

- Engines: 4 total, 4 pass rigorous gate, 1 WF-eligible, 4 catch-tier
- Assets covered: 3

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| PEPE | 1 | +115.1% | +45.6% | 0.18 |
| FET | 2 | +45.7% | +57.8% | 0.19 |
| APT | 1 | +41.3% | +45.8% | 0.40 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 3 | 2 | +66.7% | +115.1% | +53.7% |
| VOLATILE | 1 | 1 | +41.3% | +41.3% | +45.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 4 | 3 | +60.4% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| DEGEN | 3 |
| VOLATILE | 1 |

### Best single engine of class

- **PEPE** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+115.1% | DD=-6.8% | hit=66.7%

### Emergent story

**measure_engines/wh_whale_net_usd** has 4 engines across 3 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=DEGEN. Weakest regime=bull. WF-eligible 1; catch-tier 4.

---

## measure_engines/wh_whale_trade_count_500k

- Engines: 6 total, 6 pass rigorous gate, 0 WF-eligible, 6 catch-tier
- Assets covered: 2

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| NEAR | 4 | +104.8% | +50.9% | 0.15 |
| HBAR | 2 | +22.6% | +51.3% | 0.28 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 6 | 2 | +68.4% | +104.8% | +51.0% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 6 | 2 | +68.4% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull |
|---|:---:|
| VOLATILE | 6 |

### Best single engine of class

- **NEAR** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+104.8% | DD=-14.7% | hit=64.7%

### Emergent story

**measure_engines/wh_whale_trade_count_500k** has 6 engines across 2 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 0; catch-tier 6.

---

## measure_engines/xd_btc_return

- Engines: 12 total, 12 pass rigorous gate, 2 WF-eligible, 3 catch-tier
- Assets covered: 9

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| BLUR | 1 | +39.0% | +25.6% | 0.27 |
| CHZ | 2 | +38.0% | +42.2% | 0.61 |
| BNB | 1 | +35.6% | +42.9% | 0.15 |
| DOT | 1 | +28.2% | +45.5% | -0.26 |
| BCH | 2 | +27.2% | +36.4% | 0.11 |
| XRP | 2 | +23.8% | +33.9% | 0.57 |
| DYDX | 1 | +20.1% | +37.1% | 0.23 |
| DASH | 1 | +17.2% | +48.7% | 0.16 |
| FET | 1 | +9.8% | +59.2% | 0.26 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| STEADY | 6 | 4 | +25.1% | +35.6% | +38.2% |
| DEGEN | 3 | 3 | +22.0% | +39.0% | +44.5% |
| VOLATILE | 3 | 2 | +29.5% | +38.0% | +40.5% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 9 | 8 | +25.3% | 2 |
| chop | 3 | 3 | +25.8% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| DEGEN | 3 | 0 |
| STEADY | 5 | 1 |
| VOLATILE | 1 | 2 |

### Best single engine of class

- **BLUR** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+39.0% | DD=-2.3% | hit=83.3%

### Emergent story

**measure_engines/xd_btc_return** has 12 engines across 9 assets. Dominant by count: regime=bull, bucket=STEADY. Highest mean compound on bucket=VOLATILE. Weakest regime=bull. WF-eligible 2; catch-tier 3.

---

## measure_engines/xd_btc_volatility

- Engines: 6 total, 6 pass rigorous gate, 1 WF-eligible, 3 catch-tier
- Assets covered: 6

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| NEAR | 1 | +87.3% | +46.5% | -0.23 |
| ICP | 1 | +80.0% | +57.7% | -0.29 |
| HBAR | 1 | +55.3% | +39.8% | -0.03 |
| APT | 1 | +44.2% | +36.7% | 0.76 |
| AAVE | 1 | +11.7% | +47.4% | 0.15 |
| ETH | 1 | +8.5% | +36.4% | 0.28 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 5 | 5 | +55.7% | +87.3% | +45.6% |
| BLUE | 1 | 1 | +8.5% | +8.5% | +36.4% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 3 | 3 | +58.6% | 0 |
| chop | 3 | 3 | +37.1% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bull | chop |
|---|:---:|:---:|
| BLUE | 1 | 0 |
| VOLATILE | 2 | 3 |

### Best single engine of class

- **NEAR** | config=op_abs_gt_thr_1.0 | regime=bull | hold=3d | compound=+87.3% | DD=-3.5% | hit=70.0%

### Emergent story

**measure_engines/xd_btc_volatility** has 6 engines across 6 assets. Dominant by count: regime=bull, bucket=VOLATILE. Highest mean compound on bucket=VOLATILE. Weakest regime=chop. WF-eligible 1; catch-tier 3.

---

## measure_engines/xd_funding_spread

- Engines: 9 total, 9 pass rigorous gate, 1 WF-eligible, 5 catch-tier
- Assets covered: 8

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FET | 1 | +36.1% | +61.3% | 0.18 |
| ALGO | 1 | +35.7% | +46.2% | 0.01 |
| CHZ | 2 | +33.9% | +44.3% | 0.25 |
| ICP | 1 | +29.1% | +39.1% | 0.51 |
| HBAR | 1 | +26.6% | +54.4% | -0.21 |
| TIA | 1 | +26.1% | +26.7% | 0.06 |
| AVAX | 1 | +17.8% | +29.4% | -0.34 |
| SHIB | 1 | +12.8% | +50.0% | -0.05 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 5 | 4 | +25.1% | +33.9% | +41.8% |
| DEGEN | 2 | 2 | +24.4% | +36.1% | +55.6% |
| STEADY | 2 | 2 | +26.7% | +35.7% | +37.8% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bear | 4 | 4 | +22.6% | 0 |
| bull | 3 | 3 | +31.9% | 0 |
| chop | 2 | 2 | +20.9% | 1 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| DEGEN | 1 | 0 | 1 |
| STEADY | 1 | 1 | 0 |
| VOLATILE | 2 | 2 | 1 |

### Best single engine of class

- **FET** | config=op_abs_gt_thr_1.0 | regime=bear | hold=1d | compound=+36.1% | DD=-6.5% | hit=84.6%

### Emergent story

**measure_engines/xd_funding_spread** has 9 engines across 8 assets. Dominant by count: regime=bear, bucket=VOLATILE. Highest mean compound on bucket=STEADY. Weakest regime=chop. WF-eligible 1; catch-tier 5.

---

## measure_engines/xd_ma_distance

- Engines: 17 total, 17 pass rigorous gate, 6 WF-eligible, 3 catch-tier
- Assets covered: 13

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| FIL | 1 | +59.8% | +53.5% | 0.57 |
| FET | 1 | +59.5% | +40.4% | 0.12 |
| ALGO | 1 | +56.1% | +37.4% | 0.24 |
| CHZ | 3 | +45.2% | +47.4% | 0.37 |
| ADA | 2 | +40.0% | +36.8% | 0.55 |
| SUI | 1 | +31.5% | +46.3% | 0.00 |
| AVAX | 2 | +31.2% | +24.3% | 0.67 |
| SEI | 1 | +30.3% | +38.2% | -0.26 |
| BTC | 1 | +16.8% | +40.3% | 0.87 |
| SHIB | 1 | +15.0% | +35.8% | -0.28 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| VOLATILE | 7 | 5 | +25.1% | +45.2% | +44.1% |
| STEADY | 5 | 3 | +32.0% | +56.1% | +31.9% |
| DEGEN | 4 | 4 | +37.3% | +59.8% | +42.7% |
| BLUE | 1 | 1 | +16.8% | +16.8% | +40.3% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| chop | 9 | 6 | +28.9% | 3 |
| bull | 7 | 6 | +33.3% | 3 |
| bear | 1 | 1 | +9.2% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| BLUE | 0 | 1 | 0 |
| DEGEN | 0 | 2 | 2 |
| STEADY | 0 | 3 | 2 |
| VOLATILE | 1 | 1 | 5 |

### Best single engine of class

- **FIL** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+59.8% | DD=-4.8% | hit=66.7%

### Emergent story

**measure_engines/xd_ma_distance** has 17 engines across 13 assets. Dominant by count: regime=chop, bucket=VOLATILE. Highest mean compound on bucket=DEGEN. Weakest regime=bear. WF-eligible 6; catch-tier 3.

---

## measure_engines/xd_momentum_rank

- Engines: 18 total, 18 pass rigorous gate, 7 WF-eligible, 6 catch-tier
- Assets covered: 12

### D1. Per-asset top performers

| asset | n_engines | best_compound | mean_catch | best_stability |
|---|---:|---:|---:|---:|
| WLD | 2 | +158.4% | +41.8% | 0.17 |
| FLOKI | 1 | +141.2% | +40.8% | -0.20 |
| AVAX | 1 | +75.0% | +35.7% | 0.29 |
| FIL | 2 | +40.9% | +44.5% | 0.46 |
| AAVE | 2 | +33.8% | +42.3% | 0.82 |
| BLUR | 1 | +32.4% | +39.6% | 0.21 |
| BTC | 1 | +28.6% | +55.8% | 0.18 |
| CRV | 2 | +26.9% | +37.5% | 0.76 |
| ARB | 1 | +25.2% | +43.8% | 0.54 |
| SHIB | 1 | +21.2% | +47.6% | 0.38 |

### D2. Per-DNA-bucket

| bucket | n_engines | n_assets | mean_comp | best_comp | mean_catch |
|---|---:|---:|---:|---:|---:|
| DEGEN | 10 | 6 | +50.5% | +158.4% | +44.0% |
| VOLATILE | 6 | 4 | +26.3% | +33.8% | +41.3% |
| BLUE | 1 | 1 | +28.6% | +28.6% | +55.8% |
| STEADY | 1 | 1 | +75.0% | +75.0% | +35.7% |

### D3. Per-regime

| regime | n_engines | n_assets | mean_comp | n_WF_eligible |
|---|---:|---:|---:|---:|
| bull | 10 | 7 | +56.8% | 3 |
| chop | 7 | 5 | +25.8% | 4 |
| bear | 1 | 1 | +17.5% | 0 |

### D4. Per-bucket × per-regime grid (engine count)

| bucket | bear | bull | chop |
|---|:---:|:---:|:---:|
| BLUE | 0 | 1 | 0 |
| DEGEN | 0 | 5 | 5 |
| STEADY | 0 | 0 | 1 |
| VOLATILE | 1 | 4 | 1 |

### Best single engine of class

- **WLD** | config=op_abs_gt_thr_1.0 | regime=bull | hold=1d | compound=+158.4% | DD=-1.4% | hit=81.8%

### Emergent story

**measure_engines/xd_momentum_rank** has 18 engines across 12 assets. Dominant by count: regime=bull, bucket=DEGEN. Highest mean compound on bucket=STEADY. Weakest regime=bear. WF-eligible 7; catch-tier 6.

---
