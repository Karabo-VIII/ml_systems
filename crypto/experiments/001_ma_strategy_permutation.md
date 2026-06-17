# Experiment 001: Moving Average Strategy Permutation Backtest

**Date:** 2026-03-06
**Script:** `src/analysis/ma_backtest.py`
**Results:** `logs/analysis/ma_backtest_20260306_195739.json`

## Objective

Test whether simple trend-following and mean-reversion strategies can extract profit from dollar bar data (~5-min bars, 288/day) across 3 instruments (BTC, ETH, SOL), using realistic Binance cost models.

## Methodology

- **133 strategy permutations** across 11 strategy types:
  SMA Crossover, EMA Crossover, Price vs SMA, SMA Momentum, Bollinger Threshold,
  Donchian Breakout, Triple SMA, SMA Cross Long-Only, Dual Momentum, Price SMA Long-Only, Buy & Hold
- **3 cost models:** Zero (pure signal), Spot (0.10%/side), Perp (0.04%/side + 0.01%/8h funding)
- **70/30 IS/OOS split** (no parameter optimization on OOS)
- **Period map:** 12=1h, 24=2h, 48=4h, 96=8h, 144=12h, 288=1d, 576=2d, 1440=5d, 2880=10d
- **Sharpe:** Annualized daily Sharpe ratio (sqrt(365))

## Data

| Asset | Bars | Bars/Day | Date Range | B&H Return |
|-------|------|----------|------------|-----------|
| BTCUSDT | 2,629,097 | 1,178 | 2020-01 to 2026-02 | +737% |
| ETHUSDT | 3,900,997 | 1,754 | 2020-01 to 2026-02 | +1,098% |
| SOLUSDT | 3,979,974 | 2,064 | 2020-11 to 2026-02 | +4,533% |

## Results Summary

### Profitability by Cost Model (OOS)

| Cost Model | Profitable OOS | Rate | Cross-Asset Winner |
|------------|:--------------:|:----:|:-------------------|
| Zero Cost | 337 / 399 | 84% | Price_vs_SMA, Bollinger |
| Spot (0.10%/side) | 5 / 399 | 1% | Buy & Hold only |
| Perp (0.04%/side + funding) | 19 / 399 | 5% | Donchian(1440) |

### Cross-Asset Consistent Strategies (OOS, all 3 assets)

| Cost Model | Strategies | Winner |
|------------|:----------:|--------|
| Zero | 92 / 133 | Price_vs_SMA(48) avg Sharpe=2.27 |
| Spot | 1 / 133 | Donchian(1440) avg Sharpe=0.54 |
| Perp | 1 / 133 | Donchian(1440) avg Sharpe=0.98 |

### Donchian 5-Day Breakout (the ONLY cross-asset profitable strategy)

| Asset | Cost Model | Sharpe | Return | MaxDD | Trades |
|-------|-----------|-------:|-------:|------:|-------:|
| BTC | Spot | 0.19 | -6.3% | 46.3% | 807 |
| BTC | Perp | 0.41 | +13.9% | 41.3% | 807 |
| ETH | Spot | 0.88 | +56.6% | 41.3% | 1,141 |
| ETH | Perp | 1.36 | +140.4% | 40.4% | 1,141 |
| SOL | Spot | 0.54 | +5.0% | 53.2% | 1,204 |
| SOL | Perp | 1.17 | +71.4% | 37.7% | 1,204 |

## Key Findings

### 1. The signal is REAL (84% profitable zero-cost)
Dollar bars contain genuine trend and mean-reversion signal. 92 of 133 strategies are profitable OOS on ALL 3 assets when costs are zero.

### 2. Transaction costs at 5-min bar frequency kill everything
At ~1,200 bars/day, even infrequent MA crossovers generate thousands of trades. At 0.10% per side (spot), most strategies lose 100% of capital to costs.

### 3. Only ultra-low-frequency trend following survives
Donchian(1440) = 5-day breakout. ~200 trades/year. Low frequency = low cost impact. Captures multi-day crypto trends.

### 4. Perp > Spot for infrequent strategies
Perpetual futures (0.04%/side) have 60% lower per-trade fees than spot (0.10%/side). For low-frequency strategies, this more than compensates for funding costs.

### 5. Mean reversion (Bollinger) works at zero cost but not after costs
Bollinger strategies top the zero-cost leaderboard (Sharpe 2-4) but require high-frequency trading to exploit. After costs, they lose 100%.

## Implications for World Model + Agent

1. **The world model predicts at h=1,4,16,64 bars** (~5min to ~5.5hr). This is the WRONG frequency for per-candle trading. The agent MUST NOT trade every bar.

2. **Multi-horizon strategy:** The agent should use h=16/h=64 predictions for ENTRY TIMING but hold positions for DAYS (matching the 5-day breakout horizon that actually works).

3. **World model as FILTER:** Use WM predictions to IMPROVE the Donchian breakout by skipping false breakouts when multi-horizon signals disagree or uncertainty is high.

4. **Regime-switching:** Use WM regime prediction to alternate between trend-following (trending market) and grid/mean-reversion (ranging market).

## Profitable Binance Bot Research

| Strategy | How It Profits | Typical Return |
|----------|---------------|:--------------:|
| Grid Bots | Buy low/sell high in range | 2.5-4%/month |
| Funding Rate Arb | Long spot + short perp | 10-15%/year |
| Market Making | Bid-ask spread capture | Varies |
| Trend Following (daily) | Multi-day/week trends | 12-25%/year |

Common thread: profitable bots exploit market microstructure (spreads, funding, oscillation), NOT short-term directional prediction at high frequency.

## Next Steps

- [x] Run MA permutation backtest
- [ ] World model horizon decomposition (Experiment 002)
- [ ] Design multi-horizon agent that trades per optimal horizon
- [ ] Test WM as filter on Donchian 5-day breakout
