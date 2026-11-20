# Decoupling-Basket Sim Audit FIXED (2026-05-23, post-data-bug correction)

## DATA BUG RETRACTION
- Prior sim audit used `target_return_1_raw` from chimera_v51 parquet.
- That column is CORRUPTED: stored values are 30-880x smaller than computed `close.pct_change().shift(-1)`.
- This script uses ACTUAL close-derived 1-day-forward returns.
- Verification of event_eval_rows: pnl_post_cost_pct values are NORMAL (abs_mean 3.32%) confirming the layer-3 oracle pipeline uses real returns.
- The PRIOR `sim_train_stable_basket_*.md`, `sim_triple_filter_basket_*.md`, `sim_decoupling_audit.md` outputs are ALL INVALID due to the data bug.

## J<0.50 32-engine basket sim variants (CORRECT returns)

| Variant | n_eng | mean %/d | Sharpe | hit% | NAV % |
|---|---:|---:|---:|---:|---:|
| A_k3_25pct_24bp_1d_hold_FIXED | 32 | +0.583 | 3.18 | 57.7 | +454.2 |
| B_k3_25pct_5bp_maker_1d_FIXED | 32 | +0.698 | 3.79 | 59.9 | +695.7 |
| C_k3_25pct_0bp_GROSS_1d | 32 | +0.728 | 3.96 | 60.6 | +775.2 |
| D_k3_25pct_24bp_3d_hold | 32 | +1.502 | 2.56 | 59.6 | +7213.5 |
| E_k3_25pct_24bp_5d_hold | 32 | +2.556 | 2.41 | 63.1 | +129397.5 |
| F_k3_25pct_24bp_7d_hold | 32 | +3.573 | 2.23 | 61.8 | +1844816.3 |
| G_k1_100pct_24bp_1d | 32 | +0.787 | 2.53 | 56.2 | +727.9 |
| H_k5_20pct_24bp_1d | 32 | +0.649 | 3.29 | 56.8 | +569.2 |

## Reference: prior session reported +0.71%/d / Sharpe 3.46 / +699% NAV

## Best Sharpe: **C_k3_25pct_0bp_GROSS_1d** at +0.728%/d / Sharpe 3.96 / NAV +775.2%
VERDICT: The original +0.71%/d basket claim is RECONFIRMED on consistent methodology.