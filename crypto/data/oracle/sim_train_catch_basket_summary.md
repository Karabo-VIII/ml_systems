# Catch-Tier Basket — TRAIN-Realized Portfolio Simulation

Window: 2023-07-02 -> 2024-05-15 (295 trading days)

Setup: top-3 picks/day from catch-tier basket (95 engines / 30 assets), ranked by sum_catch_rate.
Sizing: 25% per pick × 3 picks = 75% NAV deployed.
Cost: 0.24% RT.

## Headline (TRAIN-only, NOT deploy)

| Metric | Value |
|---|---:|
| Mean realized %/d (post-cost) | **+0.4113%/d** |
| Median %/d | +0.4898%/d |
| Sharpe (annualized) | **+2.334** |
| Total compound NAV | **+199.30%** |
| Max DD | -31.97% |
| Positive days | 176 (59.7%) |
| Negative days | 119 |

## Honest caveats

- This is TRAIN-window backtest. OOS will deflate (per prior session, naive top-3 lost -75% on OOS).
- The catch-tier basket itself is WF-validated on catch rate, but composition behavior on OOS is untested.
- 24bp RT cost is maker-leaning; full bucket-aware cost may add 8-12bp.
- Cost on 75% deployed gives baseline drag ~0.018%/d.

## Sample 10 most-recent TRAIN days

| date | n_picks | assets | day_nav_pct_post_cost |
|---|---:|---|---:|
| 2024-05-15 | 3 | AAVE,FLOKI,SOL | -0.688% |
| 2024-05-13 | 3 | AAVE,PEPE,SOL | -0.243% |
| 2024-05-12 | 1 | HBAR | -0.313% |
| 2024-05-11 | 1 | HBAR | +0.409% |
| 2024-05-09 | 1 | FET | -0.671% |
| 2024-05-08 | 1 | FET | +1.220% |
| 2024-05-07 | 1 | FET | -1.992% |
| 2024-05-06 | 2 | FET,HBAR | -1.973% |
| 2024-05-05 | 2 | CHZ,HBAR | +0.864% |
| 2024-05-04 | 2 | CHZ,FET | +2.569% |