# V7-SAFE Basket Sim (Red-Team Pruned) (2026-05-23T11:13)

## V7-SAFE: V7 triple-filter MINUS red-team HIGH-VULN engines

Filters applied:
- catch_tier_eligibility=True
- stable_flag=True (lifecycle decay)
- NOT in CONCAVE class (anti-fragility)
- mean_pnl_pct >= 1.5% per fire
- composite_vulnerability < median (red-team audit)

## V7-SAFE members

- APT | Distance_z_state | period_50_threshold_1.5 | bull | mean_pnl=5.16%, n_fires=10
- APT | MA_state_EMA_above | period_20 | bull | mean_pnl=2.53%, n_fires=13
- FET | measure_engines/rv_bpv_5m | op_abs_gt_thr_1.0 | bull | mean_pnl=3.43%, n_fires=18
- FET | measure_engines/rv_bpv_5m | op_gt_thr_1.0 | bull | mean_pnl=5.59%, n_fires=10
- FET | measure_engines/rv_rv_5m | op_abs_gt_thr_1.0 | bull | mean_pnl=3.38%, n_fires=18
- FET | measure_engines/rv_rv_5m | op_gt_thr_1.0 | bull | mean_pnl=4.98%, n_fires=11

## Sim results (FIXED close-derived returns, 1d hold, 24bp RT cost)

| Variant | n_eng | active_days | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|---:|
| V7_baseline_12eng | 12 | 136 | +0.822 | 3.65 | 55.9 | +180.2 |
| V7_SAFE_postRedTeam | 6 | 124 | +0.406 | 3.40 | 54.0 | +61.7 |

## Reference
- V1 32-engine basket: +0.583%/d / Sharpe 3.18 / +454% NAV
- V7 (baseline 12 engines): +0.822%/d / Sharpe 3.65 / +180% NAV
- Consensus-weighted F1 (32 engines): +0.799%/d / Sharpe 3.38 / +908% NAV
