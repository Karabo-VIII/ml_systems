# Per-Asset DEPLOY-CANDIDATE Best Configs (TRAIN+WF, scout)

For each asset with >=1 basket-eligible engine, the BEST ENGINE under each criterion:

- **STABILITY**: highest WF cov_stability (most reliable across sub-folds)
- **COMPOUND**: highest TRAIN compound (biggest wealth-builder)
- **CATCH-RATE**: highest top-25% mover catch rate (best selector; n_fires >=30)
- **LEAD-LAG**: measure_event with strongest z_lift_t-3 (most leading)
- **COMBO**: best all-rounder = catch × compound × stability

Caveat: same engine may appear in multiple columns. NOT deploy-ready.

---

## LINK (10 eligible engines)

**Stability winner**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+30.9% | DD=-9.1% | stab=0.78 | catch=33.0%

**Compound winner**: VPIN_threshold (t_1.0) | regime=bull | hold=1d | compound=+54.7% | DD=-1.9% | stab=0.50 | catch=21.8%

**Catch-rate winner**: Donchian_state_above_midline (period_100) | regime=chop | hold=1d | compound=+25.8% | DD=-11.9% | stab=0.14 | catch=48.8%

**Lead-lag winner**: measure_engines/liq_long_usd (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+34.7% | DD=-2.9% | stab=0.38 | catch=46.0%

**All-rounder combo**: MACD_threshold (f_12_s_35_g_9) | regime=chop | hold=3d | compound=+44.5% | DD=-4.7% | stab=0.64 | catch=46.4%

---
## PEPE (8 eligible engines)

**Stability winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+28.3% | DD=-2.1% | stab=0.29 | catch=55.4%

**Compound winner**: measure_engines/wh_whale_net_usd (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+115.1% | DD=-6.8% | stab=0.18 | catch=45.6%

**Catch-rate winner**: Donchian_state_above_midline (period_55) | regime=chop | hold=1d | compound=+28.3% | DD=-2.1% | stab=0.29 | catch=57.1%

**Lead-lag winner**: measure_engines/bd_imbalance_l5 (op_gt_thr_1.0) | regime=chop | hold=1d | compound=+38.1% | DD=-1.5% | stab=0.29 | catch=52.7%

**All-rounder combo**: measure_engines/wh_whale_net_usd (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+115.1% | DD=-6.8% | stab=0.18 | catch=45.6%

---
## HBAR (7 eligible engines)

**Stability winner**: measure_engines/norm_deviation (op_abs_gt_thr_1.5) | regime=chop | hold=1d | compound=+32.4% | DD=-0.3% | stab=0.29 | catch=45.2%

**Compound winner**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=bull | hold=1d | compound=+49.7% | DD=-1.8% | stab=-0.10 | catch=66.2%

**Catch-rate winner**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=bull | hold=1d | compound=+49.7% | DD=-1.8% | stab=-0.10 | catch=66.2%

**Lead-lag winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bear | hold=1d | compound=+26.6% | DD=-4.9% | stab=-0.21 | catch=54.4%

**All-rounder combo**: measure_engines/norm_deviation (op_abs_gt_thr_1.5) | regime=chop | hold=1d | compound=+32.4% | DD=-0.3% | stab=0.29 | catch=45.2%

---
## BTC (7 eligible engines)

**Stability winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+36.6% | DD=-3.6% | stab=0.23 | catch=62.3%

**Compound winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+36.6% | DD=-3.6% | stab=0.23 | catch=62.3%

**Catch-rate winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+36.6% | DD=-3.6% | stab=0.23 | catch=62.3%

**Lead-lag winner**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+15.9% | DD=-2.3% | stab=-0.01 | catch=55.7%

**All-rounder combo**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+36.6% | DD=-3.6% | stab=0.23 | catch=62.3%

---
## SOL (6 eligible engines)

**Stability winner**: MACD_threshold (f_12_s_21_g_9) | regime=chop | hold=3d | compound=+44.3% | DD=-6.6% | stab=0.69 | catch=54.9%

**Compound winner**: MACD_threshold (f_12_s_21_g_9) | regime=chop | hold=3d | compound=+44.3% | DD=-6.6% | stab=0.69 | catch=54.9%

**Catch-rate winner**: MACD_threshold (f_12_s_21_g_9) | regime=chop | hold=3d | compound=+44.3% | DD=-6.6% | stab=0.69 | catch=54.9%

**All-rounder combo**: MACD_threshold (f_12_s_21_g_9) | regime=chop | hold=3d | compound=+44.3% | DD=-6.6% | stab=0.69 | catch=54.9%

---
## AAVE (6 eligible engines)

**Stability winner**: VPIN_threshold (t_0.5) | regime=chop | hold=3d | compound=+13.3% | DD=-11.9% | stab=0.35 | catch=50.2%

**Compound winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+29.1% | DD=-2.5% | stab=0.27 | catch=47.8%

**Catch-rate winner**: measure_engines/rv_jump_frac (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+23.4% | DD=-0.7% | stab=0.14 | catch=76.1%

**Lead-lag winner**: measure_engines/hbr_eta_total (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+11.9% | DD=-2.1% | stab=0.23 | catch=48.6%

**All-rounder combo**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+29.1% | DD=-2.5% | stab=0.27 | catch=47.8%

---
## FET (6 eligible engines)

**Stability winner**: VPIN_threshold (t_0.5) | regime=bull | hold=3d | compound=+88.9% | DD=-10.4% | stab=0.27 | catch=54.4%

**Compound winner**: VPIN_threshold (t_0.5) | regime=bull | hold=3d | compound=+88.9% | DD=-10.4% | stab=0.27 | catch=54.4%

**Catch-rate winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bear | hold=1d | compound=+36.1% | DD=-6.5% | stab=0.18 | catch=61.3%

**Lead-lag winner**: measure_engines/hbr_eta_buy (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-9.4% | stab=0.11 | catch=46.9%

**All-rounder combo**: VPIN_threshold (t_0.5) | regime=bull | hold=3d | compound=+88.9% | DD=-10.4% | stab=0.27 | catch=54.4%

---
## JST (5 eligible engines)

**Stability winner**: MA_state_EMA_above (period_200) | regime=chop | hold=1d | compound=+23.8% | DD=-4.4% | stab=0.92 | catch=42.6%

**Compound winner**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+48.9% | DD=-7.9% | stab=0.35 | catch=47.2%

**Catch-rate winner**: MA_state_SMA_above (period_100) | regime=chop | hold=1d | compound=+15.3% | DD=-11.0% | stab=0.26 | catch=48.4%

**All-rounder combo**: MA_state_EMA_above (period_200) | regime=chop | hold=1d | compound=+23.8% | DD=-4.4% | stab=0.92 | catch=42.6%

---
## OP (4 eligible engines)

**Stability winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+39.1% | DD=-7.7% | stab=0.27 | catch=45.8%

**Compound winner**: Donchian_state_above_midline (period_20) | regime=chop | hold=1d | compound=+40.4% | DD=-6.9% | stab=0.27 | catch=47.4%

**Catch-rate winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+20.3% | DD=-6.2% | stab=-0.29 | catch=55.1%

**Lead-lag winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+20.3% | DD=-6.2% | stab=-0.29 | catch=55.1%

**All-rounder combo**: Donchian_state_above_midline (period_20) | regime=chop | hold=1d | compound=+40.4% | DD=-6.9% | stab=0.27 | catch=47.4%

---
## FIL (4 eligible engines)

**Stability winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.0% | DD=-2.1% | stab=0.53 | catch=49.5%

**Compound winner**: measure_engines/liq_long_usd (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+38.6% | DD=-4.9% | stab=-0.36 | catch=48.1%

**Catch-rate winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.0% | DD=-2.1% | stab=0.53 | catch=49.5%

**Lead-lag winner**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+31.1% | DD=-11.5% | stab=0.29 | catch=45.7%

**All-rounder combo**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.0% | DD=-2.1% | stab=0.53 | catch=49.5%

---
## ADA (4 eligible engines)

**Stability winner**: VPIN_threshold (t_1.0) | regime=bull | hold=3d | compound=+49.3% | DD=-5.2% | stab=0.28 | catch=48.8%

**Compound winner**: VWAP_state_above (period_20) | regime=bull | hold=3d | compound=+62.5% | DD=-11.1% | stab=-0.22 | catch=47.4%

**Catch-rate winner**: VPIN_threshold (t_1.0) | regime=bull | hold=3d | compound=+49.3% | DD=-5.2% | stab=0.28 | catch=48.8%

**All-rounder combo**: VPIN_threshold (t_1.0) | regime=bull | hold=3d | compound=+49.3% | DD=-5.2% | stab=0.28 | catch=48.8%

---
## NEAR (4 eligible engines)

**Stability winner**: measure_engines/wh_whale_trade_count_500k (op_abs_gt_thr_1.5) | regime=bull | hold=1d | compound=+78.9% | DD=-12.4% | stab=0.15 | catch=48.9%

**Compound winner**: measure_engines/wh_whale_trade_count_500k (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+104.8% | DD=-14.7% | stab=0.03 | catch=53.7%

**Catch-rate winner**: measure_engines/wh_whale_trade_count_500k (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+104.8% | DD=-14.7% | stab=0.03 | catch=53.7%

**Lead-lag winner**: measure_engines/wh_whale_trade_count_500k (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+104.8% | DD=-14.7% | stab=0.03 | catch=53.7%

**All-rounder combo**: measure_engines/wh_whale_trade_count_500k (op_abs_gt_thr_1.5) | regime=bull | hold=1d | compound=+78.9% | DD=-12.4% | stab=0.15 | catch=48.9%

---
## XRP (4 eligible engines)

**Stability winner**: MACD_threshold (f_12_s_35_g_9) | regime=bull | hold=3d | compound=+23.3% | DD=-9.1% | stab=0.85 | catch=36.8%

**Compound winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+33.9% | DD=-3.3% | stab=0.47 | catch=40.2%

**Catch-rate winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+33.9% | DD=-3.3% | stab=0.47 | catch=40.2%

**Lead-lag winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+33.9% | DD=-3.3% | stab=0.47 | catch=40.2%

**All-rounder combo**: MACD_threshold (f_12_s_35_g_9) | regime=bull | hold=3d | compound=+23.3% | DD=-9.1% | stab=0.85 | catch=36.8%

---
## DASH (3 eligible engines)

**Stability winner**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+17.2% | DD=-3.1% | stab=0.16 | catch=48.7%

**Compound winner**: MA_state_SMA_above (period_50) | regime=bull | hold=1d | compound=+18.4% | DD=-12.3% | stab=0.14 | catch=45.1%

**Catch-rate winner**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+17.2% | DD=-3.1% | stab=0.16 | catch=48.7%

**Lead-lag winner**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+17.2% | DD=-3.1% | stab=0.16 | catch=48.7%

**All-rounder combo**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+17.2% | DD=-3.1% | stab=0.16 | catch=48.7%

---
## APT (3 eligible engines)

**Stability winner**: VPIN_threshold (t_0.5) | regime=chop | hold=1d | compound=+42.8% | DD=-14.0% | stab=0.80 | catch=33.2%

**Compound winner**: VPIN_threshold (t_0.5) | regime=chop | hold=1d | compound=+42.8% | DD=-14.0% | stab=0.80 | catch=33.2%

**Catch-rate winner**: MA_state_EMA_above (period_20) | regime=bull | hold=3d | compound=+34.4% | DD=-13.0% | stab=0.29 | catch=46.3%

**Lead-lag winner**: measure_engines/wh_whale_net_usd (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+41.3% | DD=-8.5% | stab=0.40 | catch=45.8%

**All-rounder combo**: VPIN_threshold (t_0.5) | regime=chop | hold=1d | compound=+42.8% | DD=-14.0% | stab=0.80 | catch=33.2%

---
## ICP (3 eligible engines)

**Stability winner**: measure_engines/bd_imbalance_l1 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | DD=-13.5% | stab=0.59 | catch=54.0%

**Compound winner**: measure_engines/bd_imbalance_l1 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | DD=-13.5% | stab=0.59 | catch=54.0%

**Catch-rate winner**: measure_engines/bd_imbalance_l1 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | DD=-13.5% | stab=0.59 | catch=54.0%

**Lead-lag winner**: measure_engines/bd_imbalance_l1 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | DD=-13.5% | stab=0.59 | catch=54.0%

**All-rounder combo**: measure_engines/bd_imbalance_l1 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+58.6% | DD=-13.5% | stab=0.59 | catch=54.0%

---
## FLOKI (3 eligible engines)

**Stability winner**: measure_engines/hbr_eta_buy (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+44.7% | DD=-3.3% | stab=0.28 | catch=52.4%

**Compound winner**: measure_engines/hbr_eta_buy (op_abs_gt_thr_1.0) | regime=chop | hold=3d | compound=+92.6% | DD=-1.3% | stab=-0.00 | catch=52.4%

**Catch-rate winner**: measure_engines/hbr_eta_buy (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+44.7% | DD=-3.3% | stab=0.28 | catch=52.4%

**Lead-lag winner**: measure_engines/hbr_eta_total (op_abs_gt_thr_1.0) | regime=chop | hold=3d | compound=+35.2% | DD=-4.4% | stab=0.11 | catch=45.6%

**All-rounder combo**: measure_engines/hbr_eta_buy (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+44.7% | DD=-3.3% | stab=0.28 | catch=52.4%

---
## CHZ (2 eligible engines)

**Stability winner**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+38.0% | DD=-4.0% | stab=0.61 | catch=41.7%

**Compound winner**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+38.0% | DD=-4.0% | stab=0.61 | catch=41.7%

**Catch-rate winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bear | hold=1d | compound=+10.0% | DD=-1.8% | stab=0.25 | catch=47.2%

**Lead-lag winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bear | hold=1d | compound=+10.0% | DD=-1.8% | stab=0.25 | catch=47.2%

**All-rounder combo**: measure_engines/xd_btc_return (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+38.0% | DD=-4.0% | stab=0.61 | catch=41.7%

---
## ETC (2 eligible engines)

**Stability winner**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.9% | DD=-6.3% | stab=0.25 | catch=51.9%

**Compound winner**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.9% | DD=-6.3% | stab=0.25 | catch=51.9%

**Catch-rate winner**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.9% | DD=-6.3% | stab=0.25 | catch=51.9%

**Lead-lag winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+8.1% | DD=-9.6% | stab=-0.10 | catch=48.4%

**All-rounder combo**: measure_engines/norm_efficiency (op_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.9% | DD=-6.3% | stab=0.25 | catch=51.9%

---
## DOT (2 eligible engines)

**Stability winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+30.7% | DD=-2.3% | stab=0.26 | catch=51.8%

**Compound winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+30.7% | DD=-2.3% | stab=0.26 | catch=51.8%

**Catch-rate winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+30.7% | DD=-2.3% | stab=0.26 | catch=51.8%

**Lead-lag winner**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+30.7% | DD=-2.3% | stab=0.26 | catch=51.8%

**All-rounder combo**: measure_engines/norm_efficiency (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+30.7% | DD=-2.3% | stab=0.26 | catch=51.8%

---
## SHIB (2 eligible engines)

**Stability winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.8% | DD=-4.2% | stab=-0.05 | catch=50.0%

**Compound winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.8% | DD=-4.2% | stab=-0.05 | catch=50.0%

**Catch-rate winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+9.7% | DD=-1.4% | stab=-0.22 | catch=58.3%

**Lead-lag winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+9.7% | DD=-1.4% | stab=-0.22 | catch=58.3%

**All-rounder combo**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+12.8% | DD=-4.2% | stab=-0.05 | catch=50.0%

---
## WLD (2 eligible engines)

**Stability winner**: YZ_vol_regime (t_0.5) | regime=bull | hold=1d | compound=+11.2% | DD=-9.3% | stab=0.23 | catch=45.4%

**Compound winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+37.6% | DD=-4.2% | stab=0.07 | catch=51.8%

**Catch-rate winner**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+37.6% | DD=-4.2% | stab=0.07 | catch=51.8%

**All-rounder combo**: MA_state_SMA_above (period_50) | regime=chop | hold=1d | compound=+37.6% | DD=-4.2% | stab=0.07 | catch=51.8%

---
## SUPER (1 eligible engines)

**Stability winner**: measure_engines/norm_deviation (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-5.2% | stab=0.26 | catch=45.8%

**Compound winner**: measure_engines/norm_deviation (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-5.2% | stab=0.26 | catch=45.8%

**Catch-rate winner**: measure_engines/norm_deviation (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-5.2% | stab=0.26 | catch=45.8%

**Lead-lag winner**: measure_engines/norm_deviation (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-5.2% | stab=0.26 | catch=45.8%

**All-rounder combo**: measure_engines/norm_deviation (op_abs_gt_thr_1.0) | regime=chop | hold=1d | compound=+33.2% | DD=-5.2% | stab=0.26 | catch=45.8%

---
## BCH (1 eligible engines)

**Stability winner**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | stab=0.46 | catch=50.0%

**Compound winner**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | stab=0.46 | catch=50.0%

**Catch-rate winner**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | stab=0.46 | catch=50.0%

**Lead-lag winner**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | stab=0.46 | catch=50.0%

**All-rounder combo**: measure_engines/stbl_total_zscore_30d (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+27.6% | DD=-4.2% | stab=0.46 | catch=50.0%

---
## DYDX (1 eligible engines)

**Stability winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+12.5% | DD=-2.7% | stab=-0.14 | catch=47.2%

**Compound winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+12.5% | DD=-2.7% | stab=-0.14 | catch=47.2%

**Catch-rate winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+12.5% | DD=-2.7% | stab=-0.14 | catch=47.2%

**All-rounder combo**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+12.5% | DD=-2.7% | stab=-0.14 | catch=47.2%

---
## ARB (1 eligible engines)

**Stability winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | stab=-0.09 | catch=47.2%

**Compound winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | stab=-0.09 | catch=47.2%

**Catch-rate winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | stab=-0.09 | catch=47.2%

**Lead-lag winner**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | stab=-0.09 | catch=47.2%

**All-rounder combo**: measure_engines/bs_basis_z30 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.1% | DD=-9.2% | stab=-0.09 | catch=47.2%

---
## ZEC (1 eligible engines)

**Stability winner**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | DD=-4.1% | stab=0.45 | catch=51.0%

**Compound winner**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | DD=-4.1% | stab=0.45 | catch=51.0%

**Catch-rate winner**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | DD=-4.1% | stab=0.45 | catch=51.0%

**Lead-lag winner**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | DD=-4.1% | stab=0.45 | catch=51.0%

**All-rounder combo**: measure_engines/bd_imbalance_l5 (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+21.7% | DD=-4.1% | stab=0.45 | catch=51.0%

---
## UNI (1 eligible engines)

**Stability winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+24.7% | DD=-7.0% | stab=0.16 | catch=45.1%

**Compound winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+24.7% | DD=-7.0% | stab=0.16 | catch=45.1%

**Catch-rate winner**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+24.7% | DD=-7.0% | stab=0.16 | catch=45.1%

**All-rounder combo**: MA_state_EMA_above (period_20) | regime=chop | hold=1d | compound=+24.7% | DD=-7.0% | stab=0.16 | catch=45.1%

---
## AR (1 eligible engines)

**Stability winner**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+52.3% | DD=-0.4% | stab=0.07 | catch=49.4%

**Compound winner**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+52.3% | DD=-0.4% | stab=0.07 | catch=49.4%

**Catch-rate winner**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+52.3% | DD=-0.4% | stab=0.07 | catch=49.4%

**All-rounder combo**: MA_state_SMA_above (period_20) | regime=bull | hold=1d | compound=+52.3% | DD=-0.4% | stab=0.07 | catch=49.4%

---
## ARKM (1 eligible engines)

**Stability winner**: VPIN_threshold (t_0.5) | regime=chop | hold=3d | compound=+55.2% | DD=-10.5% | stab=0.28 | catch=49.8%

**Compound winner**: VPIN_threshold (t_0.5) | regime=chop | hold=3d | compound=+55.2% | DD=-10.5% | stab=0.28 | catch=49.8%

**Catch-rate winner**: VPIN_threshold (t_0.5) | regime=chop | hold=3d | compound=+55.2% | DD=-10.5% | stab=0.28 | catch=49.8%

**All-rounder combo**: VPIN_threshold (t_0.5) | regime=chop | hold=3d | compound=+55.2% | DD=-10.5% | stab=0.28 | catch=49.8%

---
## ALGO (1 eligible engines)

**Stability winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.7% | DD=-6.4% | stab=0.01 | catch=46.2%

**Compound winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.7% | DD=-6.4% | stab=0.01 | catch=46.2%

**Catch-rate winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.7% | DD=-6.4% | stab=0.01 | catch=46.2%

**Lead-lag winner**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.7% | DD=-6.4% | stab=0.01 | catch=46.2%

**All-rounder combo**: measure_engines/xd_funding_spread (op_abs_gt_thr_1.0) | regime=bull | hold=1d | compound=+35.7% | DD=-6.4% | stab=0.01 | catch=46.2%

---