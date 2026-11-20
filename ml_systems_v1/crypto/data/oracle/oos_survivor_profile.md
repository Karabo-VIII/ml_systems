> **SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup)**: This document predates the
> canonical split-discipline gate. References to "OOS" in this file may include data
> from the canonical UNSEEN window (>=2026-01-01) per [src/split_config.py](../../src/split_config.py).
> Use this document for historical context only; deploy decisions citing UNSEEN-relevant
> claims must be re-derived from the canonical segments.

# OOS Survivor Profile (2026-05-23T13:09)

## Sample
- Total engines tested OOS: 61
- SURVIVED (ratio > 0.5): 2
- DECAYED_partial (0.3-0.5): 3
- DECAYED_heavy (0-0.3): 5
- INVERTED (ratio ≤ 0): 51

## Feature comparison: survivors vs inverters

| Feature | Survivors mean | Inverters mean | Delta | Direction |
|---|---:|---:|---:|---|
| hit_rate | 0.648 | 0.676 | -0.029 | Survivors LOWER |
| mean_pnl_pct | 2.743 | 3.811 | -1.068 | Survivors LOWER |
| n_fires | 10.600 | 13.667 | -3.067 | Survivors LOWER |
| compound_return_pct | 31.231 | 61.570 | -30.339 | Survivors LOWER |
| payoff_ratio | 2.014 | 2.881 | -0.868 | Survivors LOWER |
| wf_cov_stability | -0.000 | 0.189 | -0.190 | Survivors LOWER |
| shic_ratio | 0.232 | 0.219 | +0.014 | Survivors HIGHER |
| decay_score | -0.777 | -1.115 | +0.338 | Survivors HIGHER |
| stability_ratio | 1.900 | 1.459 | +0.441 | Survivors HIGHER |
| convexity_score | 3.229 | 5.076 | -1.847 | Survivors LOWER |
| skew_pnl | 0.444 | 0.589 | -0.146 | Survivors LOWER |
| kurt_excess_pnl | 0.379 | 1.989 | -1.610 | Survivors LOWER |

## Surviving engines (SURVIVED or DECAYED_partial)

| Asset | Class | Config | Regime | ratio | hit_rate | n_fires | stable | antifragility |
|---|---|---|---|---:|---:|---:|---|---|
| PEPE | Donchian_state_above_midline | period_55 | chop | +1.14 | 0.900 | 10 | False | CONCAVE |
| FIL | measure_engines/liq_long_usd | op_abs_gt_thr_1.0 | bull | +0.34 | 0.600 | 10 | True | CONVEX |
| FIL | measure_engines/liq_long_usd | op_gt_thr_1.0 | bull | +0.37 | 0.600 | 10 | True | CONVEX |
| FET | measure_engines/wh_whale_net_usd | op_abs_gt_thr_1.5 | bull | +0.79 | 0.538 | 13 | False | CONCAVE |
| WLD | YZ_vol_regime | t_0.5 | bull | +0.34 | 0.600 | 10 | False | SYMMETRIC |

## Per-class survival rate (ratio > 0 = positive OOS)

| class | n_tested | n_positive | pos_rate | mean_ratio |
|---|---:|---:|---:|---:|
| YZ_vol_regime | 1 | 1 | 100.0% | +0.34 |
| Donchian_state_above_midline | 1 | 1 | 100.0% | +1.14 |
| measure_engines/liq_long_usd | 3 | 2 | 66.7% | -0.37 |
| measure_engines/wh_whale_net_usd | 2 | 1 | 50.0% | +0.12 |
| measure_engines/bd_imbalance_l5 | 2 | 1 | 50.0% | -1.63 |
| measure_engines/norm_deviation | 2 | 1 | 50.0% | -0.82 |
| measure_engines/hbr_eta_buy | 3 | 1 | 33.3% | -0.56 |
| MA_state_SMA_above | 3 | 1 | 33.3% | -0.59 |
| measure_engines/norm_flow_imbalance | 4 | 1 | 25.0% | -0.48 |
| measure_engines/xd_btc_return | 1 | 0 | 0.0% | -0.33 |
| measure_engines/wh_whale_trade_count_500k | 4 | 0 | 0.0% | -0.40 |
| measure_engines/te_in_btc | 3 | 0 | 0.0% | -1.16 |
| measure_engines/te_imb | 2 | 0 | 0.0% | -0.39 |
| measure_engines/rv_rv_5m | 2 | 0 | 0.0% | -0.32 |
| measure_engines/rv_jump_frac | 1 | 0 | 0.0% | -0.83 |
| measure_engines/rv_bpv_5m | 2 | 0 | 0.0% | -0.14 |
| Distance_z_state | 4 | 0 | 0.0% | -0.85 |
| measure_engines/norm_efficiency | 2 | 0 | 0.0% | -0.63 |
| measure_engines/bd_imbalance_l1 | 2 | 0 | 0.0% | -0.55 |
| VWAP_state_above | 1 | 0 | 0.0% | -0.52 |

## Per-regime survival rate

| regime | n_tested | n_positive | pos_rate | mean_ratio |
|---|---:|---:|---:|---:|
| bear | 1 | 0 | 0.0% | -1.48 |
| bull | 44 | 6 | 13.6% | -0.82 |
| chop | 16 | 4 | 25.0% | -1.02 |

## Per-asset survival rate (assets with ≥3 tests)

| asset | n_tested | n_positive | pos_rate | mean_ratio |
|---|---:|---:|---:|---:|
| FET | 10 | 1 | 10.0% | -0.44 |
| SOL | 6 | 0 | 0.0% | -1.56 |
| PEPE | 4 | 2 | 50.0% | +0.09 |
| NEAR | 4 | 0 | 0.0% | -0.40 |
| LINK | 4 | 0 | 0.0% | -1.38 |
| ICP | 4 | 0 | 0.0% | -0.85 |
| HBAR | 3 | 1 | 33.3% | -0.40 |