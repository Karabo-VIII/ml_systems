# Oracle Z2: Queuing-Theory Capacity Report

Generated: 2026-05-23  |  TRAIN window: 2023-07-02 -> 2024-05-15

## Method

- ADV proxy = sum of dollar-bar `volume_usd` per calendar date, chimera_v51 dollar cadence
  (Binance SPOT, TRAIN window 2023-07-02 to 2024-05-15)
- 5-min slot capacity = ADV / 288 (uniform time-of-day, lower-bound assumption)
- **Daily participation cap = 5% of p25 daily ADV** (industry-standard for execution)
- **Per-fire 5m cap = 1% of 5-min slot** (framework-spec, conservative reference only)
- Pick notional = deploy_usd * 0.25 (top-3 sizing)
- Slippage = Almgren-Chriss sqrt impact: slip_bps = 100 * sqrt(participation_frac)
  (calibrated to ~50 bps at 25% participation, Kissell-Glantz crypto)

**Honest caveats:**

- No actual order book depth at scale; LOB columns sparse in chimera
- Daily volume / 288 assumes uniform intraday liquidity (false at open/close)
- Per-(asset, date) fire reconstruction uses *expected* asset weights, not actual engine fires
  (TRAIN fire log not persisted at engine-level granularity)
- Capacity at $10M is approximation only; real microstructure breaks at <$1M for thin assets

## Per-Asset ADV Distribution (TRAIN, USD)

### Cheapest 5 assets (smallest median ADV) - capacity bottlenecks

| Asset | n_days | ADV median | ADV p25 | 5% daily cap | 1% 5m cap |
|---|---:|---:|---:|---:|---:|
| JST | 319 | $1,415,847 | $557,716 | $27,886 | $19 |
| ZEC | 319 | $2,849,486 | $1,769,985 | $88,499 | $61 |
| AR | 319 | $5,339,825 | $1,995,175 | $99,759 | $69 |
| ALGO | 319 | $7,595,751 | $4,649,370 | $232,468 | $161 |
| CHZ | 319 | $9,694,595 | $3,675,422 | $183,771 | $128 |

### Richest 5 assets in V7/V1 universe

| Asset | n_days | ADV median | ADV p75 | 5% daily cap | 1% 5m cap |
|---|---:|---:|---:|---:|---:|
| FIL | 319 | $30,401,117 | $65,157,611 | $644,026 | $447 |
| ADA | 319 | $42,600,144 | $84,600,189 | $1,024,526 | $711 |
| LINK | 319 | $57,891,267 | $108,885,089 | $1,473,517 | $1,023 |
| DOGE | 319 | $83,999,633 | $212,197,879 | $1,899,877 | $1,319 |
| XRP | 319 | $190,742,962 | $283,849,828 | $6,562,711 | $4,557 |

## Capacity-Pinned Days (pressure > 5% daily ADV cap on any asset)

| Variant | $100K | $1M | $10M | $50M | $100M | $250M |
|---|---:|---:|---:|---:|---:|---:|
| V7 (136 active days) | 0 | 101 | 136 | 136 | 136 | 136 |
| V1 (317 active days) | 0 | 0 | 317 | 317 | 317 | 317 |

### Top 5 pinned assets in V7 at $10M
- FET: 136 pinned days
- AR: 136 pinned days
- APT: 123 pinned days
- FIL: 123 pinned days
- ICP: 123 pinned days

### Top 5 pinned assets in V1 at $10M
- JST: 317 pinned days
- ICP: 241 pinned days
- AR: 241 pinned days
- ZEC: 241 pinned days
- ETC: 241 pinned days

## Sharpe Degradation Table (Policy x Deploy Size)

**Reference (no capacity drag):** V7=Sharpe 3.65, V1=Sharpe 3.18 (TRAIN, FIXED)

| Variant | Policy | Deploy | mean %/d | Sharpe | NAV % | Fill avg | Slip bps |
|---|---|---:|---:|---:|---:|---:|---:|
| V1 | CAPCAP | $100K | 0.563 | 3.07 | 420.7 | 1.000 | 2.0 |
| V1 | CAPCAP | $1000K | 0.521 | 2.84 | 355.0 | 1.000 | 6.3 |
| V1 | CAPCAP | $10000K | 0.342 | 2.23 | 169.5 | 0.876 | 14.1 |
| V1 | CAPCAP | $50000K | 0.101 | 1.64 | 35.5 | 0.416 | 8.3 |
| V1 | CAPCAP | $100000K | 0.064 | 1.58 | 21.5 | 0.268 | 5.7 |
| V1 | CAPCAP | $250000K | 0.019 | 1.13 | 6.1 | 0.127 | 2.8 |
| V1 | FCFS | $100K | 0.563 | 3.07 | 420.7 | 1.000 | 2.0 |
| V1 | FCFS | $1000K | 0.521 | 2.84 | 355.0 | 1.000 | 6.3 |
| V1 | FCFS | $10000K | 0.342 | 2.23 | 169.5 | 0.876 | 14.1 |
| V1 | FCFS | $50000K | 0.101 | 1.64 | 35.5 | 0.416 | 8.3 |
| V1 | FCFS | $100000K | 0.064 | 1.58 | 21.5 | 0.268 | 5.7 |
| V1 | FCFS | $250000K | 0.019 | 1.13 | 6.1 | 0.127 | 2.8 |
| V1 | PRIORITY | $100K | 0.563 | 3.07 | 420.7 | 1.000 | 2.0 |
| V1 | PRIORITY | $1000K | 0.521 | 2.84 | 355.0 | 1.000 | 6.3 |
| V1 | PRIORITY | $10000K | 0.368 | 2.19 | 187.2 | 0.937 | 16.4 |
| V1 | PRIORITY | $50000K | 0.116 | 1.45 | 41.0 | 0.527 | 12.4 |
| V1 | PRIORITY | $100000K | 0.068 | 1.31 | 22.9 | 0.349 | 8.7 |
| V1 | PRIORITY | $250000K | 0.025 | 1.03 | 8.1 | 0.176 | 4.6 |
| V7 | CAPCAP | $100K | 0.779 | 3.47 | 164.6 | 1.000 | 4.2 |
| V7 | CAPCAP | $1000K | 0.674 | 3.09 | 130.6 | 0.975 | 12.5 |
| V7 | CAPCAP | $10000K | 0.214 | 2.57 | 32.2 | 0.413 | 9.0 |
| V7 | CAPCAP | $50000K | 0.039 | 2.29 | 5.4 | 0.090 | 2.0 |
| V7 | CAPCAP | $100000K | 0.020 | 2.29 | 2.7 | 0.045 | 1.0 |
| V7 | CAPCAP | $250000K | 0.008 | 2.29 | 1.1 | 0.018 | 0.4 |
| V7 | FCFS | $100K | 0.779 | 3.47 | 164.6 | 1.000 | 4.2 |
| V7 | FCFS | $1000K | 0.674 | 3.09 | 130.6 | 0.975 | 12.5 |
| V7 | FCFS | $10000K | 0.214 | 2.57 | 32.2 | 0.413 | 9.0 |
| V7 | FCFS | $50000K | 0.039 | 2.29 | 5.4 | 0.090 | 2.0 |
| V7 | FCFS | $100000K | 0.020 | 2.29 | 2.7 | 0.045 | 1.0 |
| V7 | FCFS | $250000K | 0.008 | 2.29 | 1.1 | 0.018 | 0.4 |
| V7 | PRIORITY | $100K | 0.779 | 3.47 | 164.6 | 1.000 | 4.2 |
| V7 | PRIORITY | $1000K | 0.687 | 3.06 | 133.7 | 1.000 | 13.4 |
| V7 | PRIORITY | $10000K | 0.284 | 2.45 | 43.8 | 0.559 | 14.4 |
| V7 | PRIORITY | $50000K | 0.052 | 2.02 | 7.2 | 0.135 | 3.7 |
| V7 | PRIORITY | $100000K | 0.026 | 2.02 | 3.6 | 0.067 | 1.8 |
| V7 | PRIORITY | $250000K | 0.010 | 2.02 | 1.4 | 0.027 | 0.7 |

## Best slot-allocation policy: **CAPCAP**

- CAPCAP: avg Sharpe across all variants/sizes = 2.37
- FCFS: avg Sharpe across all variants/sizes = 2.37
- PRIORITY: avg Sharpe across all variants/sizes = 2.24

## MAX Deploy Capital (CAPCAP policy)

### By Sharpe >= 1.0 floor

- **V7 TRIPLE FILTER:** >$250M (Sharpe ~ 2.29) (>=largest-tested)
- **V1 32-engine:** ~$250M (Sharpe ~ 1.13) (>=largest-tested)

### By economic floor (mean %/d >= 0.05% = 5 bps/d)

- **V7 TRIPLE FILTER:** ~$47.5M (mean %/d ~ 0.050%) (interp)
- **V1 32-engine:** ~$145.5M (mean %/d ~ 0.050%) (interp)

**Interpretation:**

- The CAPCAP policy is the binding constraint: once pressure exceeds 5% of daily ADV,
  fills are proportionally scaled, so additional capital adds zero return AND zero risk
  per dollar deployed. **Sharpe stabilizes** but absolute return collapses.
- The **economic max** is the more useful metric: above it, you're paying execution cost
  for negligible alpha.
- V1's 18-asset diversification provides ~3-10x higher economic capacity than V7's 6-asset basket.
- V7 is bottlenecked by AR ($5.3M median ADV) and ICP ($21M median ADV) -- the basket caps out
  on these names before the larger-cap names (FIL, APT, DOT) bind.

## 5 Most Capacity-Pinned Assets (across both baskets at $10M)

- **AR**: 377 pinned days, median ADV $5,339,825
- **ICP**: 364 pinned days, median ADV $18,554,553
- **JST**: 317 pinned days, median ADV $1,415,847
- **ZEC**: 241 pinned days, median ADV $2,849,486
- **ETC**: 241 pinned days, median ADV $21,056,923

## Verdict

- Capacity is **not binding at $100K** for either basket -- deploy can proceed at retail scale without slippage adjustment.
- V7 binds first on AR (~$1M range) due to its $5M median ADV bottleneck. V1's diversification buys ~3-10x more capacity.
- Best policy = **CAPCAP**.
- Above the per-asset 5% daily-ADV cap, Almgren-Chriss sqrt slippage compounds; Sharpe degrades from ~3.4 to ~1.0 over ~2 orders of magnitude of deploy size.

## Data outputs

- `data/oracle/queuing_capacity.parquet` (144 rows) per-asset deploy x size matrix
- `data/oracle/queuing_capacity_pressure_matrix.parquet` (6522 rows) per-(variant, date, asset)
