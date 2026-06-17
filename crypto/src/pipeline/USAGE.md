# Pipeline -- Usage Guide

## Overview

Data pipeline that transforms raw Binance trade data into enriched dollar-bar chimera parquets. Three stages: fetch raw data, build dollar bars + features + targets, inspect quality.

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_all.py` | Download raw aggTrades, funding rates, OI metrics from Binance |
| `make_dataset_legacy.py` | Build dollar bars, compute 41 features + 10 targets + regime labels |
| `inspect_dataset.py` | Quality validation with 11 strict gates + diagnostic plots |
| `sota_shared_logic_v50.py` | Feature computation engine (30 base features) |
| `inspect_raw_data.py` | Raw data integrity checks |
| `inspect_pipeline.py` | End-to-end pipeline health check |
| `config_calibrator.py` | Dollar bar size calibration |
| `data_integrity.py` | Shared data validation utilities |

## Quick Start

```powershell
# 1. Fetch raw data (all 10 assets)
python src/pipeline/fetch_all.py

# 2. Build chimera parquets (37 features + 10 targets)
python src/pipeline/make_dataset_legacy.py

# 3. Validate output quality
python src/pipeline/inspect_dataset.py --strict --plots
```

---

## Fetch All (`fetch_all.py`)

Downloads 3 data types per asset from Binance:

| Data Type | Source | Frequency |
|-----------|--------|-----------|
| aggTrades | data.binance.vision (spot) | Daily ZIP files |
| Funding Rates | data.binance.vision (futures) | 3x/day (8h intervals) |
| OI Metrics | data.binance.vision (futures) | Daily |

### Usage

```powershell
# Standard forward fetch (oldest -> newest)
python src/pipeline/fetch_all.py

# Reverse order (newest -> oldest, useful for catching up)
python src/pipeline/fetch_all.py --reverse

# Re-attempt confirmed-missing dates
python src/pipeline/fetch_all.py --recheck-missing
```

### Smart Skip List

Each asset maintains a `_fetch_manifest.json` tracking confirmed-missing dates. On restart, these are skipped to save bandwidth. Use `--recheck-missing` to force re-attempts.

### Fallback Strategy

- **aggTrades**: Direct download only (no API fallback)
- **Funding**: Download + bulk API fallback (paginated, 1000 records/call)
- **OI Metrics**: Download + API fallback + confirmation retry

### Output

```
data/raw/{SYMBOL}/
    aggTrades/*.parquet    # Daily trade files
    funding/*.parquet      # Daily funding rate files
    metrics/*.parquet      # Daily OI metric files
    _fetch_manifest.json   # Confirmed-missing registry
```

---

## Dataset Builder (`make_dataset_legacy.py`)

Two-phase pipeline that produces enriched chimera parquets.

### Phase 1: Per-Asset Processing

For each of the 10 assets:
1. Raw aggTrades -> dollar bars (grouped by cumulative $USD threshold)
2. Dollar bar OHLCV + buy/sell volume split + tick count
3. Join funding rates (24h backward tolerance)
4. Join OI metrics (48h backward tolerance)
5. Compute 34 base features via `sota_shared_logic_v50.calculate_v50_features()`
6. Compute 10 multi-horizon targets (4 raw return + 4 voladj + 2 auxiliary)
7. Compute regime labels (SMA-200 based: 0=bear, 1=neutral, 2=bull)

### Phase 2: Cross-Asset Enrichment

After all 10 assets are processed:
1. Reads all chimera files
2. Computes 7 cross-asset (XD) features per asset
3. Overwrites chimera files with 41 total features

### 41 Features Produced

**34 Base Features** (per-asset):

| # | Feature | Description |
|---|---------|-------------|
| 0 | norm_deviation | Volatility regime (EMA spread) |
| 1 | norm_fd_close | Fractional diff (stationary trend memory) |
| 2 | norm_vpin | Volume-sync probability of informed trading |
| 3 | norm_flow_imbalance | Buy/sell volume delta |
| 4 | norm_vol_cluster | Volatility of volatility |
| 5 | norm_funding | Funding rate (positioning sentiment) |
| 6 | norm_tick_count | Liquidity activity proxy |
| 7 | norm_log_volume | Absolute volume (log-scaled) |
| 8 | norm_hl_spread | Rogers-Satchell realized volatility |
| 9 | hurst_regime | Mean-reversion vs trending (R/S statistic) |
| 10 | norm_oi_change | Open interest rate of change |
| 11 | norm_return_1 | Lagged 1-bar return |
| 12 | norm_spread_bps | Effective bid-ask spread proxy |
| 13 | norm_ma_distance | SMA-200 distance |
| 14 | norm_whale | Avg trade size (institutional flow) |
| 15 | norm_efficiency | Price efficiency ratio (trending vs choppy) |
| 16 | norm_return_4 | Lagged 4-bar cumulative return |
| 17 | norm_return_16 | Lagged 16-bar cumulative return |
| 18 | norm_return_kurtosis | Rolling excess kurtosis (Tier 1) |
| 19 | norm_bar_duration | Bar duration / volume clock speed (Tier 1) |
| 20 | norm_funding_momentum | Funding rate of change (Tier 1) |
| 21 | norm_hawkes_intensity | Tick rate vs EMA self-excitation (Hawkes) |
| 22 | norm_hawkes_buy_intensity | Buy-side clustering (Hawkes) |
| 23 | norm_hawkes_sell_intensity | Sell-side clustering (Hawkes) |
| 24 | norm_hawkes_imbalance | Buy - sell clustering (Hawkes) |
| 25 | norm_momentum_accel | Trend acceleration (IC-boost) |
| 26 | norm_vol_price_corr | Volume-price correlation (IC-boost) |
| 27 | norm_vol_ratio | Volatility term structure (IC-boost) |
| 28 | norm_flow_persistence | Flow autocorrelation (IC-boost) |
| 29 | norm_oi_price_divergence | OI building while price flat (IC-boost) |
| 30 | norm_yz_volatility | Yang-Zhang volatility (SOTA, upgrades #8) |
| 31 | norm_cs_spread | Corwin-Schultz bid-ask spread (SOTA, upgrades #12) |
| 32 | norm_perm_entropy | Permutation entropy (SOTA, predictability) |
| 33 | norm_kyle_lambda | Kyle's lambda (SOTA, price impact) |

**7 Cross-Asset (XD) Features**:

| # | Feature | Description |
|---|---------|-------------|
| 34 | xd_btc_return | BTC leader signal (pass-through) |
| 35 | xd_btc_volatility | BTC risk regime (pass-through) |
| 32 | xd_funding_spread | Asset funding vs BTC/cross-mean (z-scored) |
| 33 | xd_cross_return_mean | Market breadth excl. BTC (z-scored) |
| 34 | xd_cross_vol_mean | Systemic risk excl. BTC (z-scored) |
| 35 | xd_ma_distance | Cross-sectional SMA-200 trend vs market avg (z-scored) |
| 36 | xd_momentum_rank | Cross-sectional return rank vs all peers |

### 10 Targets

| Target | Description |
|--------|-------------|
| target_return_1 | Raw next-bar return |
| target_return_4 | Raw 4-bar cumulative return |
| target_return_16 | Raw 16-bar cumulative return |
| target_return_64 | Raw 64-bar cumulative return |
| target_voladj_1 | Vol-adjusted next-bar return (symlog) -- deprecated |
| target_voladj_4 | Vol-adjusted 4-bar return (symlog) -- deprecated |
| target_voladj_16 | Vol-adjusted 16-bar return (symlog) -- deprecated |
| target_voladj_64 | Vol-adjusted 64-bar return (symlog) -- deprecated |
| target_return_50 | 50-bar risk-adjusted return |
| target_vol_20 | 20-bar forward volatility |

### Usage

```powershell
# Build all 10 assets (reads config/data_config.yaml)
python src/pipeline/make_dataset_legacy.py
```

No CLI arguments -- reads `config/data_config.yaml` for asset list and dollar bar sizes.

### Output

```
data/processed/{SYMBOL}_v50_chimera.parquet
```

Each file contains: timestamp, bar_id, OHLCV, volume_usd, buy_vol, sell_vol, tick_count, 41 features, 10 targets, regime_label.

---

## Dataset Inspector (`inspect_dataset.py`)

Comprehensive quality validator for chimera parquets.

### Usage

```powershell
# Full report (console output)
python src/pipeline/inspect_dataset.py

# With diagnostic plots
python src/pipeline/inspect_dataset.py --plots

# CI gate mode (exit code 1 on failure)
python src/pipeline/inspect_dataset.py --strict

# Quick single-asset check
python src/pipeline/inspect_dataset.py --asset btcusdt --quick
```

### 11 Strict Gates

| # | Gate | Threshold |
|---|------|-----------|
| 1 | Timestamp format | 13-digit milliseconds |
| 2 | bar_id uniqueness | No duplicates |
| 3 | OHLC geometry | high >= low |
| 4 | Base features present | All 30 columns |
| 5 | Targets present | All 6 raw targets |
| 6 | No dead base features | std > 0.01 |
| 7 | Target tail integrity | <10 zeros in last 100 of target_return_50 |
| 8 | XD features present | All 7 columns |
| 9 | No null XD values | zero nulls |
| 10 | No dead XD features | std > 0.01 |
| 11 | No leakage | |XD -> target_return_1 corr| < 0.10 |

### Diagnostic Plots (12 total)

Saved to `plots/{YYYY-MM-DD}/`:
- Bar counts, bars/day distribution, feature heatmaps
- Target distributions, price series, volume time-series
- Feature correlations, zero-value heatmaps
- XD feature stds, XD leakage correlations, XD time-series

---

## Configuration (`config/data_config.yaml`)

### 10 Assets

| Asset | Dollar Bar Size | Approx Frequency |
|-------|----------------|-------------------|
| BTCUSDT | $2,000,000 | ~5 min bars |
| ETHUSDT | $700,000 | ~5 min bars |
| SOLUSDT | $200,000 | ~5 min bars |
| BNBUSDT | $300,000 | ~5 min bars |
| XRPUSDT | $350,000 | ~5 min bars |
| DOGEUSDT | $400,000 | ~5 min bars |
| ADAUSDT | $100,000 | ~5 min bars |
| AVAXUSDT | $80,000 | ~5 min bars |
| LINKUSDT | $70,000 | ~5 min bars |
| LTCUSDT | $50,000 | ~5 min bars |

---

## Model Feature Selection

Models select features by name from their `settings.FEATURE_LIST`. Not all models use all 41 features:

| Model | Default | Feature Count Options |
|-------|---------|----------------------|
| V1.0 | 13 (fixed) | 13 base only |
| V1.1, V1.4, V1.6 | 37 | 13 / 18 / 21 / 25 / 30 / 37 |
| V1.7+ (future) | 41 | 34 / 41 (includes SOTA features) |
| V2-V9 | 37 | 13 / 18 / 30 / 37 |

### Feature Count Breakdown

- **13**: Legacy V1.0 core base features (norm_deviation through norm_spread_bps)
- **18**: 13 base + 5 extended (ma_distance, whale, efficiency, return_4, return_16)
- **21**: 18 + 3 Tier 1 (return_kurtosis, bar_duration, funding_momentum)
- **25**: 21 + 4 Hawkes (intensity, buy/sell intensity, imbalance)
- **30**: 25 + 5 IC-boost Tier 2 (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_divergence)
- **34**: 30 + 4 SOTA Tier 3 (yz_volatility, cs_spread, perm_entropy, kyle_lambda)
- **37**: Legacy 30 base + 7 cross-asset XD (backward compat, skips SOTA)
- **41**: Full (34 base + 7 cross-asset XD features)

---

## Critical Notes

- **No emoji in print()**: Windows cp1252 will crash
- **NUM_WORKERS=0**: Required for DataLoader on Windows
- **Numba JIT**: First run of frac_diff and hurst calculations compiles (slow), subsequent runs fast
- **Data-code sync**: If settings change (bins, targets, features), chimera files may need regeneration. Always run `inspect_dataset.py` after pipeline changes.
- **Raw return targets**: Default for all model training. Voladj deprecated (creates vol shortcut). Agent PnL uses raw targets.
