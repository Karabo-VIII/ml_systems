# Minimum Regime Performance (MRP) — TRAIN-realized (2026-05-23T12:57)

## Literature context (arxiv 2604.08356)
- MRP = minimum Sharpe across distinct market regimes
- Higher MRP = more durable strategy
- Apply to V1, V7, V0 baseline using BTC regime label as the regime axis

## Per-basket per-regime breakdown

| Basket | Regime | n_days | mean %/d | Sharpe |
|---|---|---:|---:|---:|
| V1_J<0.50 | bull | 128 | +0.781 | +3.93 |
| V1_J<0.50 | chop | 101 | +0.448 | +2.73 |
| V1_J<0.50 | bear | 88 | +0.451 | +2.46 |
| V1_J<0.50 | **overall** | 317 | +0.583 | +3.18 |
| V7_TRIPLE | bull | 60 | +1.138 | +4.25 |
| V7_TRIPLE | chop | 40 | +0.662 | +3.19 |
| V7_TRIPLE | bear | 36 | +0.472 | +2.97 |
| V7_TRIPLE | **overall** | 136 | +0.822 | +3.65 |
| V0_baseline | bull | 129 | +0.638 | +2.99 |
| V0_baseline | chop | 101 | +0.269 | +1.53 |
| V0_baseline | bear | 88 | +0.308 | +1.78 |
| V0_baseline | **overall** | 318 | +0.429 | +2.25 |

## MRP comparison

| Basket | MRP (min regime Sharpe) | Overall Sharpe | MRP/overall ratio |
|---|---:|---:|---:|
| V1_J<0.50 | +2.46 | +3.18 | 0.77 |
| V7_TRIPLE | +2.97 | +3.65 | 0.81 |
| V0_baseline | +1.53 | +2.25 | 0.68 |

## Honest verdict

- Best MRP: **V7_TRIPLE** at MRP = +2.97