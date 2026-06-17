# Daily Operations Loop (live trading)

> What a trader does every day in live mode. Without this loop, paper-trade insights don't transfer to live and divergences accumulate silently. H18 paper-trade (commit fd0a870 2026-05-27) is the first ground-truth test of this loop in the project.

## Pre-open checks (or pre-active-bar for 24/7 markets — start of every operating session)

1. **Exchange health**: API responds, rate limit headroom > 80%, withdrawal status normal.
2. **Data feed health**: dollar-bar producer last-bar-age < 5 minutes; chimera_v51 last-touched within 24h for active assets.
3. **Position reconciliation**: exchange-side balances + positions match internal `RiskController` state.
4. **Open orders sanity**: no stale GTC orders from previous session that shouldn't be there.
5. **DD status**: per-sleeve and portfolio DD within thresholds from `RISK_PLAYBOOK.md`.
6. **Decay monitor**: status flag for every live sleeve. Any RED = halt before any new trades.
7. **News/event scan**: regulatory announcements, exchange listings/delistings, stablecoin events. Halt affected sleeves.

If ANY of 1-7 fails: halt all sleeves, investigate, resume only after fix.

## Intra-day monitoring KPIs

Check every 30-60 minutes during active periods (or every bar for high-frequency sleeves, though none are deployed):

| KPI | Threshold | Action on breach |
|---|---|---|
| Per-sleeve PnL today | within [p10, p90] of paper-trade daily PnL | Investigate, do not auto-correct |
| Per-sleeve fill rate | within [0.25, 0.50] for limit-maker orders | Below 0.25: widen quotes or switch to taker; above 0.50: tighten quotes |
| Per-sleeve slippage today | within 2x cost model | Above 2x: pause sleeve, recalibrate |
| Per-sleeve trades-today vs expected | within 50-200% of expected | Below 50%: signal may be dormant (regime); above 200%: signal storm, halve sizing |
| Cross-sleeve correlation today | < 0.7 rolling 7d | Above 0.7: trigger HRP recalc |
| API error rate | < 1% in 1h | Above 1%: investigate; above 5%: pause |
| Account margin utilization | 0% (LO+SPOT mandate, no leverage) | Any non-zero: halt immediately, mandate violation |
| Position drift from target | < 1% notional | Above 1%: reconcile or halt |

## End-of-day reconciliation

After every 24h window (UTC midnight or session end):

1. **PnL reconciliation**: sum trade-level PnL == exchange-reported PnL within 0.1%.
2. **Execution quality summary**: write `runs/execution/<sleeve_id>_daily_<utc>.json` with avg fill distance, fill rate, slippage, taker/maker ratio.
3. **Backtest-divergence test**: Anderson-Darling test of today's per-trade returns vs paper-trade distribution. `p < 0.01` for 3 consecutive days = halt sleeve for re-validation.
4. **Decay-monitor update**: recompute IC half-life on last 100 bars; halt sleeve if dropped > 50% from deploy baseline.
5. **HRP weight refresh**: recompute cross-sleeve correlation matrix on rolling 30d; update sleeve weights.
6. **Capacity log**: largest slice executed today, slippage incurred — update sleeve capacity ceiling estimate.

## Weekly review (every 7 days)

1. **Sleeve health scorecard**: per-sleeve PnL, IC, fill rate, slippage, DD vs paper-expected.
2. **Lifecycle gate check**: any sleeve eligible for stage promotion / demotion?
3. **Regime status**: REGIME_ROUTER classification of last 7 days vs deploy regime. Mismatch = re-evaluate gating.
4. **Failure-catalog update**: any newly refuted sub-hypothesis from this week added to `WEALTH_BOT_FAILURE_CATALOG.md`.
5. **CDAP rerun**: `python src/audit/check_invariants.py` exit 0 on master.

## Monthly post-mortem

1. **All-sleeves PnL attribution**: which sleeve made / lost money, by which mechanism.
2. **Backtest vs live divergence summary per sleeve**: median, p25, p75 of (live - paper) per metric.
3. **p_fill calibration update**: if 30+ trades collected, update `config/maker_cost_calibration.yaml`.
4. **Capacity curve update**: re-estimate per-sleeve capacity from monthly aggregate.
5. **What went right / wrong**: free-form to `memory/monthly_postmortem_<utc>.md`.
6. **Promote new candidates**: any sleeves from INCUBATION ready for PAPER? Any from PAPER ready for LIVE_SMALL?

## Continuous (event-driven) checks

These run on every bar via `RiskController`:

- Kill switches from `RISK_PLAYBOOK.md`.
- Position-vs-target drift.
- Mechanism falsifier ongoing check: for each trade, does the entry condition still match the mechanism claim? Halt if 5 consecutive trades violate.

## Live-paper divergence detection (the critical loop)

Every live trade is "shadowed" by what paper-trade WOULD have done at the same bar:

```python
def divergence_check(live_trade, paper_signal_at_same_bar):
    """
    For each live trade, compare against paper-trade signal at same bar.
    If live ENTERED but paper would NOT have, or vice versa: log.
    """
    if live_trade.entered != paper_signal_at_same_bar.would_enter:
        log_to('runs/divergence/<sleeve_id>.jsonl', {
            'utc': now(),
            'live_action': live_trade.action,
            'paper_action': paper_signal_at_same_bar.would_enter,
            'cause_hypothesis': 'fill_failed | data_lag | mechanism_drift | other'
        })

    # Aggregate: if > 10% of last-30d trades diverge, sleeve is drifting.
    drift_rate = len(divergences_last_30d) / len(trades_last_30d)
    if drift_rate > 0.10:
        halt_sleeve(reason='live_paper_divergence > 10%')
```

NOT YET IMPLEMENTED. Queue this as the next sleeve-adapter feature after H18 has 30+ live trades.

## Operational risk per layer

| Layer | Risk | Daily-ops mitigation |
|---|---|---|
| Exchange | API outage, withdrawal freeze | Multi-venue planned at $50K AUM; daily API-health check |
| Custody | Hot-wallet compromise | Cold-wallet ratio: keep > 80% of AUM cold (manual rebalance monthly) |
| Network | RPC failure, internet outage | Local data caching; automatic retry with exponential backoff |
| Code | Sleeve adapter bug | CDAP pre-commit gate; smoke test before any sleeve code change |
| Process | Manual error during ops | Every action logged to `runs/ops/<utc>.jsonl`; no untracked manual trades |
| Counterparty | Binance insolvency, regulatory action | Diversify venues; cap AUM per venue at 50% once multi-venue live |
| Stablecoin | USDT/USDC depeg | Halt all sleeves on depeg > 0.5%; rebalance to fiat if available |

## Tools

- `python src/strategy/risk_controller.py --live-monitor` — bar-level monitor (NOT YET BUILT, queue).
- `python src/audit/check_invariants.py` — pre-commit gate.
- `python src/audit/check_wealth_bot_claims.py runs/deploy/<sleeve>/deploy_claim.json` — claim contract.
- `runs/paper_trade/<sleeve>_decay_status.json` — current decay flag (H18 example exists).
- `runs/lifecycle/` — stage transition records.

## CDAP wiring

| Rule | Severity | Checked file |
|---|---|---|
| `trader_daily_ops_log_present` | warn | `runs/ops/` has entry for last 24h when live sleeves exist |
| `trader_eod_reconciliation_writes_json` | warn | `runs/execution/<sleeve>_daily_*.json` written daily |
| `trader_no_leverage_in_account` | critical | Account margin utilization checks in adapter code |

## Cross-references

- RISK_PLAYBOOK.md — thresholds for halt/halve actions.
- EXECUTION_PLAYBOOK.md — order-level reconciliation rules.
- LIFECYCLE.md — weekly review feeds lifecycle gate checks.
