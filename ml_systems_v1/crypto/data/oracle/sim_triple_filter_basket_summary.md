# Triple-Filter Basket TRAIN Sim (2026-05-23T09:48)

## Variants (intersection of filters on catch-tier baseline)

| Variant | n_engines | n_assets | active_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V0_catch_tier_baseline | 234 | 39 | 318 | -0.174 | -19.81 | 9.4 | -42.6 | -42.6 |
| V1_not_concave | 152 | 34 | 303 | -0.176 | -19.21 | 9.6 | -41.4 | -41.4 |
| V2_not_concave_mag_ge_1.5 | 103 | 29 | 299 | -0.179 | -20.58 | 10.7 | -41.5 | -41.5 |
| V3_not_concave_mag_ge_3.0 | 49 | 16 | 292 | -0.165 | -19.37 | 9.6 | -38.3 | -38.3 |
| V4_convex_only | 28 | 15 | 273 | -0.160 | -15.82 | 13.2 | -35.4 | -35.5 |
| V5_convex_AND_stable_GOLD | 2 | 1 | 33 | -0.067 | -22.97 | 12.1 | -2.2 | -2.0 |
| V6_stable_mag_ge_1.5 | 13 | 7 | 199 | -0.127 | -15.67 | 7.0 | -22.3 | -22.4 |
| V7_triple_NOT_concave_AND_stable_AND_mag_ge_1.5 | 12 | 6 | 136 | -0.158 | -17.20 | 10.3 | -19.3 | -19.3 |

## Reference: 32-engine moderate-decoupling (J<0.50) = +0.71%/d / Sharpe 3.46 / +699% NAV

## Best Sharpe variant: **V6_stable_mag_ge_1.5** (-15.67 Sharpe, -0.127%/d, -22.3% NAV)
Comparison vs 32-eng baseline: Sharpe ratio = -4.53x, NAV ratio = -0.03x
