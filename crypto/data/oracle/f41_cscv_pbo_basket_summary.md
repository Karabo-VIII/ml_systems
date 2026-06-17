# F41 basket -- CSCV BASKET-level PBO

**Full-window basket Sharpe**: 2.025, compound +262.98%

Across 924 time-block CSCV splits:

| Metric | IS-mean | IS-std | OOS-mean | OOS-std |
|---|---:|---:|---:|---:|
| Sharpe | 2.001 | 0.577 | 2.001 | 0.577 |
| Compound (%) | 95.33 | 43.94 | 95.33 | 43.94 |

**P(OOS Sharpe > 0 | IS Sharpe > 0) = 1.0000**

**Spearman(IS_Sharpe, OOS_Sharpe) = -0.9989** (p=0)
**Spearman(IS_Compound, OOS_Compound) = -1.0000** (p=0)

## Interpretation

BASKET-LEVEL OVERFIT: even the basket is no better than chance.

Compare to single-engine PBO=0.74 (anti-predictive). If basket Spearman > engine Spearman, the diversification IS the alpha.
