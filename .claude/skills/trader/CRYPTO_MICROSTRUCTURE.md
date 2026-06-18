# Crypto Microstructure (what kills crypto traders that equity traders never see)

> Crypto is not a worse stock market. It is a structurally different market with specific failure modes: funding, basis, liquidation cascades, MEV, depegs, custody risk. A trader skill that doesn't know these is incompetent in this domain.

## Funding rate (PERP-specific)

Mechanism: perpetual futures don't expire. Instead, longs pay shorts (or vice versa) every 8h to keep PERP price tethered to spot. Funding = (PERP_mid - spot_mid) / spot_mid, clipped.

Empirical for Binance USDT-perps:
- Median funding: ~0.01% / 8h = ~0.03% / day = ~10% / year (long pays short — net long crowd is the norm).
- p95 funding: 0.05% / 8h = ~0.15% / day = ~55% annualized.
- p99 funding events: 0.5%+ / 8h (squeeze events, post-news pumps).

**Why PERP is DEFERRED in this project**: 0.01% / 8h = ~3.6% / month funding cost alone exceeds typical model IC ≈ 0.03 edge. 0 of 1304 strategy configs profitable after funding. See `crypto/docs/futures_strategy_considerations.md`.

When could PERP be reconsidered:
- IC > 0.10 (Headline tier in the ARCHIVED WM ladder; IC is BANNED as a primary metric post-reset, kept as historical context). Then funding is small relative to edge.
- Tick-level signal with V20 architecture (sub-second moves outrun funding accrual).
- Funding-arbitrage strategy itself (short PERP when funding > X, long spot — cash-and-carry).

## Basis (spot-PERP arbitrage)

Basis = PERP_mid - spot_mid. Closely tied to funding (basis drives funding via the funding formula).

Trade structure: long spot + short PERP = collect funding when positive. Locked-in carry trade.
- Requires margin for the PERP short. LEV=1 mandate forbids — so this is a *theoretical* trade, not deployable here.
- At scale, basis trades are how prop shops pay for infrastructure. We are not at that scale.

Implication for the trader skill: basis is information about *crowd positioning*. When basis is wide and positive, the crowd is leveraged long — anticipate liquidation cascades on a flush.

## Liquidation cascades

When PERP shorts/longs hit auto-deleverage (ADL) thresholds, the exchange forcibly closes positions at market. This adds momentum to the move that caused the liquidation, triggering more liquidations.

Empirical pattern:
- Liquidation cascade typically lasts 5-30 minutes.
- Magnitude: 3-10% move from cascade start to peak in tier-1; 10-30% in memecoins.
- Aftermath: V-shape recovery typical within 1-4 hours.

What this means for spot sleeves (the only deployable regime here):
- During cascade: spreads widen 5-20x, slippage spikes, p_fill collapses.
- Cascade is a **regime event**: halt sleeves for 24h, do not chase.
- Cascade aftermath (1-4h post): mean-reversion opportunity, but with elevated risk.

Detection signal (informal): `liquidations_5min_usd > 50M cross-exchange` (from public liquidation feeds). Add to `RISK_PLAYBOOK.md` regime table.

## MEV (maximal extractable value) — sandwich attacks

On-chain DEX trades can be front-run by mempool searchers. Sandwich attack: searcher places a buy order in front of your buy, fills your buy at worse price, exits with a sell behind your buy. You lose the sandwich tax.

Not relevant to Binance Spot CEX trading (this project). Becomes relevant if we ever route to DEX (1inch, Uniswap). Not in scope.

## Stablecoin de-peg risk

USDT and USDC trade ~$1.00 with occasional de-pegs. Historical:
- USDT depeg 2023-03: dropped to $0.97 for ~12h.
- USDC depeg 2023-03 (SVB exposure): dropped to $0.87 for ~36h, recovered.
- Smaller depegs (UST, BUSD, etc.) — total loss when issuer fails.

For this project: all P&L denominated in USDT. A 5% USDT depeg = 5% portfolio loss on USDT-denominated balances.

Mitigations:
- Maintain < 50% in any single stablecoin once portfolio > $10K.
- Halt all sleeves on any major-stablecoin depeg > 0.5% — listed as a `RISK_PLAYBOOK.md` regime trigger.
- Rebalance to fiat (off-exchange) on sustained depeg.

## Exchange custodial risk

Binance has been the dominant CEX for crypto and is currently the venue for this project. Real risks:
- Regulatory action (US SEC has sued Binance; outcome uncertain).
- Withdrawal freezes (have occurred during regulatory events).
- Solvency (after FTX, no major exchange is presumed solvent without proof).

Mitigations:
- Hot/cold ratio: keep > 80% in cold storage off Binance once portfolio > $5K.
- Multi-venue plan at $50K+ AUM (Coinbase Pro, Kraken).
- Position sizing should NEVER assume 100% of portfolio is recoverable from Binance.

## Listing / delisting events

When Binance lists or delists an asset:
- Listing: 10-100% pump within 24h on the listing pair, fade over 7-30 days.
- Delisting: 10-50% dump within 24h of delisting announcement.

Asset universe in this project (10 assets) is currently stable. Listings/delistings of OTHER assets don't directly affect the universe but DO affect:
- Liquidity migration (volume can pull from our assets to a new listing).
- Market-wide sentiment.

For the gold-standard PEPE × MA/EMA dossier: PEPE is a memecoin, listing dynamics are non-trivial. Always check listing-event history when sizing.

## Funding-rate-driven positioning skew

When funding is sustained positive (longs paying shorts) for > 7 days:
- Crowd is leveraged long.
- Vulnerability: any 3-5% adverse move triggers cascading long liquidations.
- Trader response (for SPOT sleeves): don't add notional to longs during high-funding periods; consider exiting at threshold.

When funding is sustained negative (shorts paying longs):
- Crowd is leveraged short.
- Squeeze risk: any 3-5% favorable move triggers cascading short liquidations.
- Trader response: don't add notional to shorts; SPOT longs benefit.

## Hour-of-day / day-of-week effects

Crypto trades 24/7 but is not uniformly liquid:
- 00:00-08:00 UTC: Asia-active window, generally higher volume on tier-1.
- 12:00-20:00 UTC: US-active window, highest volume globally.
- 20:00-00:00 UTC: low-volume gap.

Implications:
- Avoid market orders during low-volume gap (slippage 2-5x typical).
- TWAP slicing should target US-active hours for large orders.
- News events typically cluster around US-active hours.

## Funding-of-leverage in Bitcoin halving cycles

BTC halving (every ~4 years) historically precedes a 12-18 month bull cycle. Current cycle: 2024-04 halving, bull-cycle peak typically 2025-Q4 to 2026-Q1.

For long-only strategies in 2026-Q2 onward:
- Awareness: we may be entering bear/sideways regime post-cycle-peak.
- Sleeves trained on 2022-2024 may not generalize to 2026+ regime.
- This is partially what W25/W26 3-MA refutation (commit 665183e) is showing: bear regime correctly blocked entries.

## CDAP wiring

| Rule | Severity | What it checks |
|---|---|---|
| `trader_lev_eq_one_invariant` | critical | No code path enables PERP for live deploy without explicit override (which doesn't exist) |
| `trader_stablecoin_concentration` | warn | Sleeve YAML declares stablecoin diversification target if `aum_usd > 10000` |
| `trader_funding_aware_for_perp_sleeves` | warn | Any PERP sleeve declaration includes `funding_aware: true` (not currently in scope) |

## Cross-references

- RISK_PLAYBOOK.md regime table — funding flip, cascade events, depegs.
- EXECUTION_PLAYBOOK.md — slicing for low-volume windows.
- DAILY_OPS.md — pre-open checks include exchange + stablecoin health.
