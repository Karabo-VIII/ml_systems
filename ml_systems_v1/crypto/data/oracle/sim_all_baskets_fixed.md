# All-Baskets FIXED Sim (2026-05-23T10:55)

## Data-correction note
- All variants use REAL close-derived 1-day-forward returns (NOT the corrupted target_return_1_raw).
- Cost: 24bp RT taker. Sizing: 3 picks/day @ 25% NAV each. 1-day hold.

## Variants

| Variant | n_eng | active_days | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|---:|
| V0_catch_tier_baseline | 234 | 318 | +0.429 | 2.25 | 53.1 | +238.9 |
| V1_decoupling_J0_50_32eng | 32 | 317 | +0.583 | 3.18 | 57.7 | +454.2 |
| V2_STABLE_only | 17 | 199 | +0.392 | 2.56 | 57.3 | +105.7 |
| V3_NOT_CONCAVE | 152 | 303 | +0.493 | 2.56 | 55.4 | +287.4 |
| V4_CONVEX_only | 28 | 273 | +0.671 | 2.87 | 59.0 | +417.8 |
| V5_CONVEX_AND_STABLE_gold | 2 | 33 | +0.474 | 4.78 | 57.6 | +16.4 |
| V6_STABLE_AND_mag_ge_1.5 | 13 | 199 | +0.600 | 3.20 | 57.3 | +202.4 |
| V7_TRIPLE_stable_AND_NOT_concave_AND_mag_ge_1.5 | 12 | 136 | +0.822 | 3.65 | 55.9 | +180.2 |

## Verdict

- Best Sharpe: **V5_CONVEX_AND_STABLE_gold** at 4.78 Sharpe / +0.474%/d / NAV +16.4%
- Best NAV: V1_decoupling_J0_50_32eng at +454.2% NAV / 3.18 Sharpe
- Best mean %/d: V7_TRIPLE_stable_AND_NOT_concave_AND_mag_ge_1.5 at +0.822%/d

- The 32-engine J<0.50 basket (V1) = +0.583%/d / Sharpe 3.18 / NAV +454.2% — RECONFIRMS prior +0.71%/d / Sharpe 3.46 claim