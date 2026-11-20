# BTC-Filter Conditional Fire Rules (cross-asset engine tags)

Each row is an alt engine + its recommended BTC-condition for firing.

## How to read

- **BTC_AMPLIFIED_***: engine fires BETTER when BTC also fires same-day (within +/-1 day) -- use BTC-firing as positive filter
- **BTC_STAYOUT_***: engine fires BETTER when BTC does NOT fire same-day -- use BTC-firing as STAYOUT/inhibit signal
- **BTC_NEUTRAL**: BTC condition doesn't meaningfully change return

## Tag counts

- BTC_AMPLIFIED_STRONG: 14
- BTC_NEUTRAL: 13
- BTC_AMPLIFIED_MILD: 11
- BTC_STAYOUT_STRONG: 9
- BTC_STAYOUT_MILD: 2

## BTC_AMPLIFIED rules (fire ONLY when BTC engine fires same-day)

| asset | engine | regime | total fires | with_BTC mean | without_BTC mean | lift | rule |
|---|---|---|---:|---:|---:|---:|---|
| SUPER | measure_engines/norm_deviation(op_abs_gt_thr_1.0) | chop | 35 | +2.62% | -0.29% | +2.92% | BTC_AMPLIFIED_STRONG |
| SHIB | measure_engines/bs_basis_z30(op_abs_gt_thr_1.0) | chop | 34 | +0.17% | -2.50% | +2.67% | BTC_AMPLIFIED_STRONG |
| SOL | MA_state_SMA_above(period_20) | chop | 72 | +1.37% | -0.82% | +2.20% | BTC_AMPLIFIED_STRONG |
| AAVE | measure_engines/bs_basis_z30(op_abs_gt_thr_1.0) | chop | 32 | +0.96% | -1.18% | +2.13% | BTC_AMPLIFIED_STRONG |
| AAVE | measure_engines/bs_basis_z30(op_abs_gt_thr_1.0) | chop | 32 | +0.96% | -1.18% | +2.13% | BTC_AMPLIFIED_STRONG |
| FLOKI | measure_engines/hbr_eta_total(op_abs_gt_thr_1.0) | chop | 51 | +1.85% | -0.22% | +2.07% | BTC_AMPLIFIED_STRONG |
| SOL | Donchian_state_above_midline(period_20) | chop | 75 | +1.27% | -0.66% | +1.93% | BTC_AMPLIFIED_STRONG |
| AAVE | measure_engines/hbr_eta_total(op_abs_gt_thr_1.0) | chop | 48 | +0.58% | -1.33% | +1.91% | BTC_AMPLIFIED_STRONG |
| FIL | measure_engines/norm_efficiency(op_abs_gt_thr_1.0) | chop | 35 | +0.61% | -0.79% | +1.40% | BTC_AMPLIFIED_STRONG |
| DYDX | MA_state_EMA_above(period_20) | chop | 54 | +0.58% | -0.71% | +1.29% | BTC_AMPLIFIED_STRONG |
| FLOKI | measure_engines/hbr_eta_buy(op_abs_gt_thr_1.0) | chop | 46 | +1.74% | +0.69% | +1.04% | BTC_AMPLIFIED_STRONG |
| FLOKI | measure_engines/hbr_eta_buy(op_abs_gt_thr_1.0) | chop | 46 | +1.74% | +0.69% | +1.04% | BTC_AMPLIFIED_STRONG |
| FET | measure_engines/hbr_eta_buy(op_abs_gt_thr_1.0) | chop | 58 | +0.40% | -0.63% | +1.03% | BTC_AMPLIFIED_STRONG |
| JST | MA_state_EMA_above(period_200) | chop | 62 | +0.28% | -0.72% | +1.00% | BTC_AMPLIFIED_STRONG |
| LINK | Donchian_state_above_midline(period_100) | chop | 101 | +1.05% | +0.08% | +0.97% | BTC_AMPLIFIED_MILD |
| ETC | measure_engines/norm_efficiency(op_gt_thr_1.0) | chop | 26 | +1.24% | +0.29% | +0.95% | BTC_AMPLIFIED_MILD |
| SHIB | measure_engines/xd_funding_spread(op_abs_gt_thr_1.0) | chop | 31 | +0.67% | -0.17% | +0.84% | BTC_AMPLIFIED_MILD |
| ICP | measure_engines/rv_jump_frac(op_abs_gt_thr_1.0) | chop | 25 | +0.37% | -0.44% | +0.81% | BTC_AMPLIFIED_MILD |
| LINK | MA_state_EMA_above(period_200) | chop | 103 | +0.86% | +0.08% | +0.78% | BTC_AMPLIFIED_MILD |
| LINK | MA_state_SMA_above(period_200) | chop | 103 | +0.86% | +0.08% | +0.78% | BTC_AMPLIFIED_MILD |
| AAVE | VPIN_threshold(t_0.5) | chop | 99 | +0.16% | -0.46% | +0.63% | BTC_AMPLIFIED_MILD |
| SOL | MACD_threshold(f_12_s_21_g_9) | chop | 117 | +0.77% | +0.23% | +0.54% | BTC_AMPLIFIED_MILD |
| SOL | MACD_threshold(f_8_s_35_g_9) | chop | 117 | +0.77% | +0.23% | +0.54% | BTC_AMPLIFIED_MILD |
| SOL | MACD_threshold(f_12_s_35_g_9) | chop | 117 | +0.77% | +0.23% | +0.54% | BTC_AMPLIFIED_MILD |
| SOL | MACD_threshold(f_12_s_26_g_9) | chop | 117 | +0.77% | +0.23% | +0.54% | BTC_AMPLIFIED_MILD |

## BTC_STAYOUT rules (fire ONLY when BTC engine does NOT fire same-day)

| asset | engine | regime | total fires | with_BTC mean | without_BTC mean | lift | rule |
|---|---|---|---:|---:|---:|---:|---|
| PEPE | MA_state_SMA_above(period_100) | chop | 40 | +0.20% | +5.62% | -5.42% | BTC_STAYOUT_STRONG |
| PEPE | MA_state_SMA_above(period_100) | chop | 40 | +0.20% | +5.62% | -5.42% | BTC_STAYOUT_STRONG |
| PEPE | MA_state_EMA_above(period_100) | chop | 33 | +0.21% | +5.62% | -5.41% | BTC_STAYOUT_STRONG |
| PEPE | Donchian_state_above_midline(period_100) | chop | 37 | +0.20% | +3.12% | -2.92% | BTC_STAYOUT_STRONG |
| WLD | MA_state_SMA_above(period_50) | chop | 43 | -0.08% | +2.35% | -2.43% | BTC_STAYOUT_STRONG |
| HBAR | measure_engines/norm_deviation(op_abs_gt_thr_1.5) | chop | 14 | +0.99% | +2.47% | -1.48% | BTC_STAYOUT_STRONG |
| JST | MA_state_SMA_above(period_100) | chop | 68 | +0.07% | +1.40% | -1.34% | BTC_STAYOUT_STRONG |
| JST | MA_state_EMA_above(period_100) | chop | 72 | +0.10% | +1.25% | -1.15% | BTC_STAYOUT_STRONG |
| HBAR | measure_engines/norm_deviation(op_gt_thr_1.0) | chop | 17 | +1.30% | +2.32% | -1.03% | BTC_STAYOUT_STRONG |
| OP | MA_state_SMA_above(period_20) | chop | 48 | -0.26% | +0.35% | -0.61% | BTC_STAYOUT_MILD |
| OP | MA_state_EMA_above(period_20) | chop | 49 | -0.23% | +0.35% | -0.58% | BTC_STAYOUT_MILD |