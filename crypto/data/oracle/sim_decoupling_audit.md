# Decoupling-Basket Sim Audit (2026-05-23T10:07)

## Reconstructed MIS sizes
- J<0.15: 3 engines
- J<0.30: 11 engines
- J<0.50: **32 engines** (this is the prior claimed +0.71%/d basket)

## Sim variants on the J<0.50 basket

| Variant | n_eng | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|
| A_k3_25pct_24bp_rank_n_engines | 32 | -0.148 | -17.69 | 8.5 | -37.4 |
| B_k3_25pct_24bp_rank_n_engines_x_catch_rate | 32 | -0.145 | -16.59 | 10.1 | -36.9 |
| C_k3_25pct_5bp_maker_rank_n_engines | 32 | -0.033 | -4.26 | 34.4 | -9.9 |
| D_k3_25pct_0bp_zero_cost_GROSS | 32 | -0.003 | -0.33 | 44.5 | -0.8 |
| E_k1_100pct_24bp | 32 | -0.249 | -14.56 | 14.5 | -54.6 |
| F_k5_20pct_24bp | 32 | -0.165 | -18.33 | 6.3 | -40.7 |

## Reference

- Prior reported: +0.71%/d / Sharpe 3.46 / +699% NAV on 317 TRAIN days
- The methodology of the prior sim is NOT documented; this audit tests several variants

## Verdict

- NO variant reproduces +0.71%/d / Sharpe 3.46. Best is D_k3_25pct_0bp_zero_cost_GROSS at -0.003%/d / Sharpe -0.33
- The prior +0.71% claim CANNOT BE RECONCILED with any standard sim variant tested here
- Most likely explanations: (a) the prior sim used non-standard ranking (e.g., asset weighting by per-engine compound), (b) different TRAIN window, (c) bug in the prior inline sim