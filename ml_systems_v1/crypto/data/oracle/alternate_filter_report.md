# Alternate Filter Catalog Audit (POV-16) -- 2026-05-23T12:55

## Motivation

F26 OOS test: of 30 catch-tier engines, 25 INVERTED (83% sign-flip).
The single survivor was PEPE Donchian period_55 chop (OOS ratio +1.14).
That survivor profile is the OPPOSITE of V7's filter selection:
- hit_rate 0.90 (HIGH); n_fires 10 (small); 3fold_sign_consistent=True
- NOT stable_flag, CONCAVE class, low wf_cov_stability
- Would have been EXCLUDED by V7 (stable + not-concave + mag>=1.5%)

Hypothesis: V7's filter is anti-predictive of OOS survival; survivor-profile filters do better.

## Filter definitions

All require `catch_tier_eligibility=True`.

- **V7_baseline**: stable_flag AND not-CONCAVE AND mean_pnl_pct >= 1.5%
- **ALT_A_high_hit**: hit_rate >= 0.65 AND 3fold_sign_consistent AND n_fires in [8, 30]
- **ALT_B_chop_anchored**: btc_regime_30d='chop' AND 3fold_sign_consistent AND hit_rate >= 0.55
- **ALT_C_union**: ALT_A OR ALT_B

## Membership counts and diversity

| Filter | n | n_assets | %bull | %chop | %bear |
|---|---:|---:|---:|---:|---:|
| V7_baseline | 12 | 6 | 100.0 | 0.0 | 0.0 |
| ALT_A_high_hit | 118 | 29 | 41.5 | 41.5 | 16.9 |
| ALT_B_chop_anchored | 78 | 25 | 0.0 | 100.0 | 0.0 |
| ALT_C_union | 147 | 33 | 33.3 | 53.1 | 13.6 |

## F26 OOS-survivor (PEPE Donchian p55 chop) membership

- V7_baseline:    NO
- ALT_A_high_hit: YES
- ALT_B_chop_anchored: YES
- ALT_C_union:    YES

## TRAIN sim results (fixed close-derived returns, top-3, 25% size, 24bp RT cost)

| Filter | n_eng | active_days | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|---:|
| V7_baseline | 12 | 136 | +0.822 | 3.65 | 55.9 | +180.2 |
| ALT_A_high_hit | 113 | 318 | +0.311 | 1.56 | 54.1 | +130.1 |
| ALT_B_chop_anchored | 73 | 119 | +0.217 | 1.32 | 52.9 | +24.3 |
| ALT_C_union | 141 | 318 | +0.415 | 2.17 | 53.5 | +223.7 |

Reference baselines:
- V1 32-engine basket: +0.583%/d / Sharpe 3.18 / +454% NAV
- V7 (12 engines, prior runs): +0.822%/d / Sharpe 3.65 / +180% NAV

## POV-12 TRAIN->VAL deflation projection

POV-12 `val_walkforward_*` outputs not found at expected paths.
Cannot project val survival rate without that catalog. Re-run POV-16 once POV-12 lands.

## Honest verdict

- **ALT_A_high_hit**: mean d=-0.510%/d, Sharpe d=-2.09, NAV d=-50.1% vs V7 -- **LOSES to V7**
- **ALT_B_chop_anchored**: mean d=-0.604%/d, Sharpe d=-2.34, NAV d=-155.8% vs V7 -- **LOSES to V7**
- **ALT_C_union**: mean d=-0.407%/d, Sharpe d=-1.49, NAV d=+43.5% vs V7 -- **LOSES to V7**

**Caveat**: TRAIN-realized results reflect in-sample noise + reuse of catch-tier basis. The decisive test is val-deflation survival (F26 framework). Even if an ALT filter beats V7 on TRAIN, V7's TRAIN superiority did NOT predict OOS survival (V7 contains only 0 of 5 OOS survivors). The PEPE Donchian survivor sits in ALT_A and ALT_B by construction (chop + high hit-rate + sign-consistent + small n_fires).

If POV-12 val data lands and ALT_A/ALT_B retain a higher fraction of survivors than V7, that is the real signal -- not the TRAIN sim Sharpe.
