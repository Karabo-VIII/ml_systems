# V7 Independent Verification (2026-05-23T11:32)

## Methodology (DIFFERENT from sim_all_baskets_fixed.py — independent verification)

- Source returns: **event_eval_rows.pnl_post_cost_pct** (NOT close.pct_change)
- Per (asset, date) cell: MEAN pnl across firing engines (NOT top-pick close.pct_change)
- Dedup: groupby (asset, date, engine_key) take mean across def_type/side rows
- Sim: top-3 cells per date by n_engines, 25% sizing, 1d hold

## Results

- n_engines: 12
- n_active_days: 136
- mean %/d: **+0.974**
- Sharpe (annualized): **4.53**
- hit_rate: 59.6%
- total NAV: **+246.6%**
- max drawdown: -10.3%

## Comparison to sim_all_baskets_fixed.py result

- Original: V7 = +0.822%/d / Sharpe 3.65 / +180% NAV
- This independent verification: +0.974%/d / Sharpe 4.53 / +246.6% NAV

**Delta**: mean Δ = +0.152pp, Sharpe Δ = +0.88

**VERDICT: Independent verification DIVERGES**. Investigate methodology gap.