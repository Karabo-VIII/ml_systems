# Frontier

Experimental stack pushing toward 10x/yr (realistic: 4-10x/yr honest, 100x not plausible under SPOT/no-leverage).

## Directive

Under `/un` protocol: ship or concede per item. No interesting learning without decision.

## Isolation

All new work lives here. Zero touches to `src/{pipeline,v1-v14,agent,analysis,strategy,growth}/` unless a shared fix is warranted and approved separately. Backward compatibility preserved.

## Subtree

```
ingest/       # Public/free data (DeFiLlama, Etherscan anon, CoinGecko, KuCoin, Gate, Binance public)
features/     # Hawkes MLE, OFI, stable-flow, funding-divergence, calendar
models/       # GNN, SSL pretrain, PPO-Lagrangian, adaptive meta-labeler
strategies/   # Stable-flow overlay, funding-divergence overlay, copytrade, multi-venue listing
backtest/     # Isolated copy of fixed MtM-only simulator, frontier-specific probes
live/         # Frontier paper-trader integration
blend/        # Online-stacking adaptive weights
utils/        # Shared helpers (rate-limiters, cache, datetime)
```

## Data

- `data/frontier/{defillama,etherscan,coingecko,leaderboard,lob,social}/` — cached fetches
- `logs/frontier/` — per-item backtest + paper-trader output
- `configs/frontier/*.yaml` — per-strategy configs
- `docs/frontier/` — roadmap + results log

## Ship bar

Each item must add **≥+0.02%/day** at 5-10% alloc on 2025 OOS **OR** lift Sharpe ≥+0.15 without adding >5pp DD. Else CONCEDE.

## Tier 1 — Free-data overlays (this session starts here)

1. Stablecoin mint/flow overlay (Etherscan anon + DeFiLlama)
2. Cross-sectional funding-divergence overlay
3. Copy-trade signal from Binance public leaderboard
4. Crypto calendar effects

## Tier 2 — Universe/venue expansion

5. U200 universe expansion (freshly-listed alt ring)
6. Multi-venue listing front-run (KuCoin→Binance lag)
7. CoinGecko social velocity signals

## Tier 3 — Feature/algorithm upgrades

8. Hawkes branching-ratio MLE (Rambaldi 2024)
9. Adaptive triple-barrier meta-labeler (Hudson & Thames 2024)
10. Order-flow-imbalance from LOB depth (BTC/ETH only)

## Tier 4 — Retail-feasible DL/ML

11. GNN on cross-asset correlation graph
12. Self-supervised contrastive pretraining
13. PPO-Lagrangian with CVaR constraint
14. Online-stacking adaptive blend weights
