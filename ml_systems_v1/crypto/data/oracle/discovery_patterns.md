# Discovery Scout — TRAIN+WF report (2026-05-23T01:21)

## Catalog overview
- Total engines passing rigorous TRAIN gate: **213**
- Engines basket-eligible (WF stability ∧ min-fold > 0 ∧ DD floor): **17**
- Engine families: {'measure_event': 118, 'ta_state': 59, 'ta_event': 36}
- Assets with ≥1 eligible engine: 9

## Pattern 1 — Top families by eligible-engine count

| family | n_eligible |
|---|---:|
| ta_state_MA | 4 |
| measure_xd | 3 |
| other | 3 |
| measure_bd | 2 |
| measure_norm | 2 |
| measure_wh | 1 |
| ta_DC | 1 |
| ta_MACD | 1 |

## Pattern 2 — Best per-asset stable engines (top-1 within basket)

- **ICP** | measure_event:measure_engines/bd_imbalance_l1(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 21x on TRAIN, hit 57.1%, expectancy +2.38%/fire, compound +58.6% at unit sizing, max DD -13.5%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +12.4% / +11.6% / +26.5% (stability=0.59, ShIC=0.25). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.15, t-3=-0.20, t-6=-0.18.

- **APT** | ta_event:VPIN_threshold(t_0.5) | regime=chop | mag=all | hold=1d. Fires 24x on TRAIN, hit 58.3%, expectancy +1.58%/fire, compound +42.8% at unit sizing, max DD -14.0%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +15.3% / +21.7% / +14.0% (stability=0.80, ShIC=0.26). catches 26.1% of top-25%-mover days on TRAIN.

- **CHZ** | measure_event:measure_engines/xd_btc_return(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 12x on TRAIN, hit 66.7%, expectancy +2.81%/fire, compound +38.0% at unit sizing, max DD -4.0%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +7.2% / +17.6% / +9.5% (stability=0.61, ShIC=0.22). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.02, t-3=-0.05, t-6=+0.04.

- **FIL** | measure_event:measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | regime=chop | mag=all | hold=1d. Fires 14x on TRAIN, hit 71.4%, expectancy +2.10%/fire, compound +33.0% at unit sizing, max DD -2.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +3.5% / +14.3% / +12.4% (stability=0.53, ShIC=0.16). top-25 catch rate not yet computed. Lead/lag z_lift t-1=+0.01, t-3=-0.04, t-6=-0.05.

- **LINK** | ta_state:MA_state_SMA_above(period_20) | regime=bull | mag=all | hold=1d. Fires 17x on TRAIN, hit 58.8%, expectancy +1.72%/fire, compound +30.9% at unit sizing, max DD -9.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +7.4% / +12.2% / +8.7% (stability=0.78, ShIC=0.28). top-25 catch rate not yet computed.

- **JST** | ta_state:MA_state_EMA_above(period_200) | regime=chop | mag=all | hold=1d. Fires 21x on TRAIN, hit 76.2%, expectancy +1.05%/fire, compound +23.8% at unit sizing, max DD -4.4%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +7.3% / +6.7% / +8.2% (stability=0.92, ShIC=0.26). top-25 catch rate not yet computed.

- **XRP** | ta_event:MACD_threshold(f_12_s_35_g_9) | regime=bull | mag=all | hold=3d. Fires 14x on TRAIN, hit 57.1%, expectancy +1.68%/fire, compound +23.3% at unit sizing, max DD -9.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +15.1% / +10.6% / +12.9% (stability=0.85, ShIC=0.24). top-25 catch rate not yet computed.

- **ZEC** | measure_event:measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 15x on TRAIN, hit 66.7%, expectancy +1.37%/fire, compound +21.7% at unit sizing, max DD -4.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +4.1% / +12.1% / +4.3% (stability=0.45, ShIC=0.19). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.00, t-3=+0.11, t-6=-0.03.

- **AAVE** | ta_event:VPIN_threshold(t_0.5) | regime=chop | mag=all | hold=3d. Fires 13x on TRAIN, hit 61.5%, expectancy +1.10%/fire, compound +13.3% at unit sizing, max DD -11.9%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +3.5% / +8.9% / +20.5% (stability=0.35, ShIC=0.15). catches 56.9% of top-25%-mover days on TRAIN.

## Pattern 3 — Engines with strongest WF stability (cov > 0.8)

- **JST** | ta_state:MA_state_EMA_above(period_200) | regime=chop | mag=all | hold=1d. Fires 21x on TRAIN, hit 76.2%, expectancy +1.05%/fire, compound +23.8% at unit sizing, max DD -4.4%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +7.3% / +6.7% / +8.2% (stability=0.92, ShIC=0.26). top-25 catch rate not yet computed.

- **XRP** | ta_event:MACD_threshold(f_12_s_35_g_9) | regime=bull | mag=all | hold=3d. Fires 14x on TRAIN, hit 57.1%, expectancy +1.68%/fire, compound +23.3% at unit sizing, max DD -9.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +15.1% / +10.6% / +12.9% (stability=0.85, ShIC=0.24). top-25 catch rate not yet computed.

- **APT** | ta_event:VPIN_threshold(t_0.5) | regime=chop | mag=all | hold=1d. Fires 24x on TRAIN, hit 58.3%, expectancy +1.58%/fire, compound +42.8% at unit sizing, max DD -14.0%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +15.3% / +21.7% / +14.0% (stability=0.80, ShIC=0.26). catches 26.1% of top-25%-mover days on TRAIN.

## Pattern 4 — Measure-event engines with strongest leading z_lift (t-3 or t-6)

- **APT** | measure_event:measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 13x on TRAIN, hit 46.2%, expectancy +2.99%/fire, compound +41.3% at unit sizing, max DD -8.5%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +17.0% / +18.5% / +1.9% (stability=0.40, ShIC=0.22). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.22, t-3=+0.16, t-6=+0.00.

- **ICP** | measure_event:measure_engines/bd_imbalance_l1(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 21x on TRAIN, hit 57.1%, expectancy +2.38%/fire, compound +58.6% at unit sizing, max DD -13.5%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +12.4% / +11.6% / +26.5% (stability=0.59, ShIC=0.25). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.15, t-3=-0.20, t-6=-0.18.

- **XRP** | measure_event:measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 15x on TRAIN, hit 86.7%, expectancy +2.00%/fire, compound +33.9% at unit sizing, max DD -3.3%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +2.7% / +14.9% / +13.6% (stability=0.47, ShIC=0.16). top-25 catch rate not yet computed. Lead/lag z_lift t-1=+0.02, t-3=+0.14, t-6=-0.05.

- **ZEC** | measure_event:measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 15x on TRAIN, hit 66.7%, expectancy +1.37%/fire, compound +21.7% at unit sizing, max DD -4.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +4.1% / +12.1% / +4.3% (stability=0.45, ShIC=0.19). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.00, t-3=+0.11, t-6=-0.03.

- **ICP** | measure_event:measure_engines/xd_funding_spread(op_abs_gt_thr_1.0) | regime=chop | mag=all | hold=1d. Fires 13x on TRAIN, hit 69.2%, expectancy +2.05%/fire, compound +29.1% at unit sizing, max DD -5.8%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +3.0% / +10.2% / +13.7% (stability=0.51, ShIC=0.21). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.00, t-3=-0.10, t-6=-0.04.

- **XRP** | measure_event:measure_engines/xd_btc_return(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 17x on TRAIN, hit 52.9%, expectancy +1.32%/fire, compound +23.4% at unit sizing, max DD -4.7%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +10.7% / +8.1% / +3.1% (stability=0.57, ShIC=0.21). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.06, t-3=-0.07, t-6=+0.05.

- **CHZ** | measure_event:measure_engines/xd_btc_return(op_abs_gt_thr_1.0) | regime=bull | mag=all | hold=1d. Fires 12x on TRAIN, hit 66.7%, expectancy +2.81%/fire, compound +38.0% at unit sizing, max DD -4.0%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +7.2% / +17.6% / +9.5% (stability=0.61, ShIC=0.22). top-25 catch rate not yet computed. Lead/lag z_lift t-1=-0.02, t-3=-0.05, t-6=+0.04.

- **FIL** | measure_event:measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | regime=chop | mag=all | hold=1d. Fires 14x on TRAIN, hit 71.4%, expectancy +2.10%/fire, compound +33.0% at unit sizing, max DD -2.1%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +3.5% / +14.3% / +12.4% (stability=0.53, ShIC=0.16). top-25 catch rate not yet computed. Lead/lag z_lift t-1=+0.01, t-3=-0.04, t-6=-0.05.

## Pattern 5 — Engines with highest TRUE top-25% catch rate

- **AAVE** | ta_event:VPIN_threshold(t_0.5) | regime=chop | mag=all | hold=3d. Fires 13x on TRAIN, hit 61.5%, expectancy +1.10%/fire, compound +13.3% at unit sizing, max DD -11.9%. WF folds (2018-20 sub / 2023 sub / mid-24 sub equivalents): +3.5% / +8.9% / +20.5% (stability=0.35, ShIC=0.15). catches 56.9% of top-25%-mover days on TRAIN.

## Per-asset baskets (top-5 family-diversified per asset)

### AAVE
- basket size: 1
- mean compound (TRAIN): 13.3%
- mean stability: 0.35
  - rank 1: other | VPIN_threshold(t_0.5) | regime=chop | hold=3d | compound=+13.3% | stability=0.35

### APT
- basket size: 2
- mean compound (TRAIN): 42.1%
- mean stability: 0.60
  - rank 1: other | VPIN_threshold(t_0.5) | regime=chop | hold=1d | compound=+42.8% | stability=0.80
  - rank 2: measure_wh | measure_engines/wh_whale_net_usd(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+41.3% | stability=0.40

### CHZ
- basket size: 1
- mean compound (TRAIN): 38.0%
- mean stability: 0.61
  - rank 1: measure_xd | measure_engines/xd_btc_return(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+38.0% | stability=0.61

### FIL
- basket size: 1
- mean compound (TRAIN): 33.0%
- mean stability: 0.53
  - rank 1: measure_norm | measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.0% | stability=0.53

### ICP
- basket size: 2
- mean compound (TRAIN): 43.8%
- mean stability: 0.55
  - rank 1: measure_bd | measure_engines/bd_imbalance_l1(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | stability=0.59
  - rank 2: measure_xd | measure_engines/xd_funding_spread(op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+29.1% | stability=0.51

### JST
- basket size: 1
- mean compound (TRAIN): 23.8%
- mean stability: 0.92
  - rank 1: ta_state_MA | MA_state_EMA_above(period_200) | regime=chop | hold=1d | compound=+23.8% | stability=0.92

### LINK
- basket size: 3
- mean compound (TRAIN): 38.1%
- mean stability: 0.58
  - rank 1: ta_state_MA | MA_state_SMA_above(period_20) | regime=bull | hold=1d | compound=+30.9% | stability=0.78
  - rank 2: other | VPIN_threshold(t_1.0) | regime=bull | hold=1d | compound=+54.7% | stability=0.50
  - rank 3: ta_DC | Donchian_state_above_midline(period_20) | regime=bull | hold=1d | compound=+28.8% | stability=0.45

### XRP
- basket size: 4
- mean compound (TRAIN): 26.8%
- mean stability: 0.64
  - rank 1: ta_MACD | MACD_threshold(f_12_s_35_g_9) | regime=bull | hold=3d | compound=+23.3% | stability=0.85
  - rank 2: ta_state_MA | MA_state_EMA_above(period_100) | regime=bull | hold=3d | compound=+26.7% | stability=0.67
  - rank 3: measure_xd | measure_engines/xd_btc_return(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+23.4% | stability=0.57
  - rank 4: measure_norm | measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+33.9% | stability=0.47

### ZEC
- basket size: 1
- mean compound (TRAIN): 21.7%
- mean stability: 0.45
  - rank 1: measure_bd | measure_engines/bd_imbalance_l5(op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | stability=0.45
