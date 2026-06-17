# Execution Playbook (order types, slicing, fills)

> Backtests assume bar-close fills at mid-price. Real exchanges don't. This playbook closes the gap between assumed and realized execution. Until a sleeve has logged 50+ live trades, every execution number in its backtest should be treated as REPORTED, not VERIFIED.
>
> **Authority**: project empirics first (p_fill 0.21-0.40 empirical, Binance Spot taker 0.10%, dollar-bar size by asset tier). Theory references in parens.

## Order types — when to use which

| Order type | Cost (bps) | Fill probability | Use case in this project |
|---|---|---|---|
| Market taker | 10 (Binance Spot taker) | ~100% (subject to depth) | Entry on Donchian breakout (urgency), exit at max-hold (closing exposure) |
| Limit maker | -1 to 0 (rebate to 0) | 0.21-0.40 empirical | Entry on WM_DonchFilter (no urgency), exit on threshold cross with hysteresis |
| Limit IOC (immediate-or-cancel) | 10 | ~50-80% | Aggressive entry when book is thin; cancel and re-quote if no fill |
| Post-only | rebate | 0.21-0.40 | When latency adverse-selection is the bigger risk than execution risk |
| Iceberg | varies | high | NOT USED — sleeve notional too small to need it |
| Stop-market | 10 + slippage | ~100% | NOT USED — H23 refuted trail-stop pattern |

Default for this project: **limit maker for entries, market taker for exits** (asymmetric). Provenance: maker fills are dip-biased winners (adverse selection ~0.3 per `config/maker_cost_calibration.yaml`); exits prioritize certainty over price.

---

## Fill expectations (calibrated, 2026-04-22 OHLC replay)

Per bucket of (asset_tier, order_type, distance_from_mid):

| Asset tier | Order type | Distance from mid (bps) | Empirical p_fill |
|---|---|---|---|
| Tier 1 (BTC, ETH) | Limit maker | 0 (at mid) | 0.38-0.40 |
| Tier 1 | Limit maker | -2 (passive) | 0.25-0.30 |
| Tier 1 | Limit maker | -5 (very passive) | 0.15-0.20 |
| Tier 2 (SOL, BNB, XRP) | Limit maker | 0 | 0.30-0.35 |
| Tier 2 | Limit maker | -2 | 0.21-0.26 |
| Tier 3 (DOGE, ADA, AVAX, LINK, LTC) | Limit maker | 0 | 0.22-0.28 |
| Tier 3 | Limit maker | -2 | 0.13-0.18 |
| Memecoins (PEPE-class) | Limit maker | 0 | varies wildly (0.10-0.45) |
| Any | Market taker | N/A | ~1.00 |

**Critical**: backtests using MakerCostModel default `p_fill = 0.80` are 2-4x too optimistic. Real live equity expected at 50-75% of fixed-backtest equity (per CLAUDE.md MakerCostModel Invariants).

Budget for `p_fill_live ∈ [0.25, 0.50]` in sizing math. Source: `src/analysis/execution_sim.py` 2026-04-22 calibration.

---

## Slicing (order splitting)

For a given target notional $T:

| Target notional | Dollar-bar size | Slice strategy |
|---|---|---|
| < 10% of typical dollar bar | None — single order | E.g. $50 trade on BTC ($2M bar) |
| 10-50% of typical dollar bar | TWAP across 3-5 child orders over 5-15 minutes | E.g. $1000 trade on LINK ($400K bar) |
| 50-200% of typical dollar bar | VWAP across 30-60 minutes during typical-volume window | Almgren-Chriss optimal-execution |
| > 200% of typical dollar bar | Multi-day VWAP with daily-budget cap | Capacity ceiling — reconsider notional |

Current sleeves at LIVE_SMALL stage ($100-1000 per trade) are universally in the "single order" bucket. Slicing matters at LIVE_SCALE.

### TWAP slicer (simple)

```python
def twap_child_orders(total_notional, n_children, interval_seconds):
    """
    Split into n_children equal-notional limit-maker orders, spaced by
    interval_seconds. Cancel-and-replace if not filled within interval.
    """
    child = total_notional / n_children
    schedule = [(i * interval_seconds, child) for i in range(n_children)]
    return schedule
```

Sleeve adapter must handle:
- Partial fills (re-quote remainder).
- Cancel-replace if mid moves > 2 bps adverse.
- Hard timeout: if 50% un-filled after `n_children * interval_seconds`, cross spread on remainder.

### VWAP slicer (production-grade)

Estimate intraday volume profile per asset from last 30 days of dollar-bar volume. Schedule child orders proportional to expected volume. NOT YET IMPLEMENTED — defer until LIVE_SCALE actually needed.

---

## Adverse selection mechanics

Empirically: maker fills are dip-biased winners. `adverse_selection = 0.3` default in `MakerCostModel` means filled trades are systematically getting a slightly worse entry than a random-time entry would be.

Mechanism: when your bid sits at $X, you only get filled when price ticks down to $X. Conditional on fill, the next-tick distribution is more likely to keep going down than to bounce up. Adverse selection is the cost of patience.

Implications:
- Maker rebate alone does not capture the all-in execution edge.
- Real maker round-trip is approximately: `2 * (rebate) - 2 * (adverse_selection_bps)`. Net = small positive or small negative.
- The 0.10% taker assumption in CLAUDE.md is the safer model — use it as the *upper bound* on per-side cost.

---

## Slippage model (current + gap)

Current: fixed 2 bps (SPOT) / 1 bp (PERP) slippage assumption. Linear in distance from mid.

Gap: real slippage depends on order size relative to book depth, vol-of-vol, time-of-day. We don't model any of these.

Until a proper book-aware model exists, conservative budget:
- Tier 1 (BTC, ETH): 2-5 bps additional slippage on market orders.
- Tier 2: 3-8 bps.
- Tier 3: 5-15 bps.
- Memecoins: 10-50 bps (highly variable).

Add to round-trip cost when sizing.

---

## Latency considerations

Order-to-fill latency on Binance Spot: ~100-500ms typical, up to 2s in stress.

What this means for the sleeve:
- Decision interval of 64+ bars (1-2 day holding) — latency is irrelevant.
- Per-bar (dollar-bar) decisions — latency is a non-trivial fraction of bar time on tier-3 assets. Avoid this regime (per CLAUDE.md key empirical findings #1).
- Catalyst-event trades (post-listing pump, regulatory announcement): latency MATTERS. Even 500ms can flip the entry from winning to losing on a 5-second pump. Don't trade these without dedicated latency-aware execution.

---

## Multi-venue / hedge execution

Current: Binance Spot only. Single venue = single point of failure for execution.

If portfolio grows past $50K AUM, consider:
- Coinbase Pro as backup venue (lower fees but lower liquidity on alts).
- Kraken as additional Tier 1 venue.
- Multi-venue smart order router NOT in scope yet — track in queued actions.

---

## Pre-trade execution checklist (every order)

1. Pre-cancel any stale open orders on same symbol.
2. Verify available balance >= notional + buffer (2 bps for slippage).
3. Verify API rate limit headroom (orders/sec remaining > 1).
4. Place order with computed slice.
5. Set timeout on each child.
6. Log to `runs/execution/<sleeve_id>_orders.jsonl`.
7. On fill (partial or full), update position in `RiskController`.
8. If timeout reached on > 50% un-filled, escalate per slicer strategy.
9. Reconcile end-of-bar: exchange position == internal position. Halt sleeve if mismatch.

---

## Post-trade reconciliation

Every 5 minutes:
- Pull exchange-side balances + positions via API.
- Compare to internal `RiskController` state.
- Mismatch > 1% notional = halt sleeve, alert user, do not auto-correct.

Every end-of-day:
- Pull execution-quality stats from exchange (filled vs cancelled, average distance from mid, slippage).
- Write to `runs/execution/<sleeve_id>_daily_<utc>.json`.
- Update p_fill calibration if 30+ days of data collected.

---

## CDAP wiring

| Rule | Severity | What it checks |
|---|---|---|
| `trader_p_fill_budget_in_claim` | critical | Deploy claim declares `p_fill_live_budget` field |
| `trader_no_trail_stop_on_h6_class` | warn | Sleeve YAML on H6-class entries does NOT include `trail_stop: true` (H23 refutation) |
| `trader_post_trade_reconciliation_wired` | critical | Sleeve adapter has end-of-day reconciliation handler |
| `trader_rate_limit_budget_declared` | warn | Sleeve YAML declares `max_orders_per_sec` <= 4 (Binance Spot conservative) |

## Cross-references

- PRE_DEPLOY_CHECKLIST.md item 16 — exchange-side verification.
- CRYPTO_MICROSTRUCTURE.md — funding, basis, liquidations.
- DAILY_OPS.md — live monitoring + reconciliation loop.
