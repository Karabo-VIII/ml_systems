# Paper-trade ledger -- 4 frozen candidates (pre-registered live-forward track)

Emitted by `strat_prod_deploy.py`. Frozen specs (re-tuning voids the gate). Latest data as-of **2026-05-28**. Re-run weekly (`python runs/engines/STRAT_TIERS/strat_prod_deploy.py`) to append each week's decision.

| as_of | breadth | ER | CONV | SMA_ER | PARTIAL | MOM8_ER |
|---|---|---|---|---|---|---|
| 2026-05-28 | 0.38 | 0.60 | CASH | CASH | DEPLOY (65n,0.29) | CASH |

**Promotion gate (per candidate):** >=20 deployed weeks with deployed-median>0, compound>cash, compound>BH_btc over the same deployed weeks, maxDD within backtest floor. **Kill:** deployed-median<=0 over 20+ wks, compound<cash, or DD breach.