# Dynamic-Eligibility Gated Deploy Basket (2026-05-23T13:19)

## Methodology
- Gating rule: engine ACTIVE if trailing-30-day mean post-cost pnl > 0 AND trailing_fires >= 3
- Sim: 5-engine deploy basket (F31), top-3 picks/day at 25% sizing, per-bucket maker cost
- Compare STATIC (no gate) vs GATED (only fires when active)

## STATIC results (baseline)

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| STATIC | TRAIN_<=2024-05-15 | 225 | +0.206 | 1.35 | 52.9 | +48.8 | -17.7 |
| STATIC | VAL_2024-05-16_to_2025-03-15 | 80 | -0.029 | -0.16 | 46.2 | -5.3 | -31.5 |
| STATIC | OOS_pre_2025-03-16_to_2025-12-31 | 60 | +0.541 | 2.38 | 51.7 | +33.4 | -12.6 |
| STATIC | UNSEEN_2026-01-01_to_2026-05-19 | 26 | -0.412 | -6.54 | 42.3 | -10.3 | -10.6 |
| STATIC | FULL_POST_TRAIN | 166 | +0.117 | 0.63 | 47.6 | +13.3 | -31.5 |

## GATED results (dynamic eligibility)

| Window | n_days | mean %/d | Sharpe | hit% | NAV % | maxDD % |
|---|---:|---:|---:|---:|---:|---:|
| GATED  | TRAIN_<=2024-05-15 | 87 | +0.026 | 0.21 | 52.9 | +0.5 | -15.4 |
| GATED  | VAL_2024-05-16_to_2025-03-15 | 27 | -0.059 | -0.32 | 51.9 | -2.6 | -16.0 |
| GATED  | OOS_pre_2025-03-16_to_2025-12-31 | 20 | +0.280 | 2.11 | 45.0 | +5.3 | -8.2 |
| GATED  | UNSEEN_2026-01-01_to_2026-05-19 | 3 | -0.212 | -3.18 | 33.3 | -0.6 | -1.3 |
| GATED  | FULL_POST_TRAIN | 50 | +0.068 | 0.43 | 48.0 | +1.9 | -16.7 |

## STATIC vs GATED delta per window

| Window | STATIC mean | GATED mean | Δ mean | STATIC Sh | GATED Sh | Δ Sh |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN_<=2024-05-15 | +0.206 | +0.026 | -0.180 | 1.35 | 0.21 | -1.14 |
| VAL_2024-05-16_to_2025-03-15 | -0.029 | -0.059 | -0.030 | -0.16 | -0.32 | -0.16 |
| OOS_pre_2025-03-16_to_2025-12-31 | +0.541 | +0.280 | -0.260 | 2.38 | 2.11 | -0.27 |
| UNSEEN_2026-01-01_to_2026-05-19 | -0.412 | -0.212 | +0.200 | -6.54 | -3.18 | +3.37 |
| FULL_POST_TRAIN | +0.117 | +0.068 | -0.049 | 0.63 | 0.43 | -0.19 |

## Verdict

- UNSEEN STATIC: -0.412%/d / Sharpe -6.54
- UNSEEN GATED: -0.212%/d / Sharpe -3.18
  → Dynamic gating IMPROVES UNSEEN outcome by Sharpe Δ +3.37