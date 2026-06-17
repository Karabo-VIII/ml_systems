# Consensus-Weighted Basket Sim (2026-05-23T11:00)

## Basket: 32-engine J<0.50 MIS, using FIXED close-derived returns, 24bp RT cost

## Strategies tested

| Strategy | active_days | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|
| CONST (Baseline F1: top-3 by n_engines, 25% sizing each) | 317 | +0.583 | 3.18 | 57.7 | +454.2 |
| PROP (Top-3 by n_engines, proportional weight by n_engines (cap at 4)) | 317 | +0.618 | 3.05 | 59.0 | +500.5 |
| TOP1_FULL_AT_HIGH_CONS (Top-1 100% if consensus>=3 else top-3 25%) | 317 | +0.799 | 3.38 | 57.4 | +908.5 |
| THR3 (Only fire when max consensus >= 3 (skip low-consensus days)) | 54 | +0.376 | 3.41 | 72.2 | +213.4 |
| THR5 (Only fire when max consensus >= 5 (high-conviction only)) | 4 | +0.042 | 1.16 | 75.0 | +13.7 |
| TOP_BY_CONS_KEEP_K3 (Top-3 by n_engines (= CONST when ranked same)) | 317 | +0.583 | 3.18 | 57.7 | +454.2 |

## Verdict

- Best Sharpe: **THR3** at Sharpe 3.41 (+0.24 vs baseline CONST)
- Mean %/d delta vs baseline: -0.208pp