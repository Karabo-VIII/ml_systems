# V4 Crypto System -- Operations Guide

## System Overview

```
RESEARCH & TRAINING          ANALYSIS & VALIDATION         PRODUCTION
--------------------         ----------------------        ----------
Pipeline (data)         -->  Strategy Lab (backtest)  -->  Universe Screener
Model Training (V1-V9)  -->  Walk-Forward (validate)  -->  Fetch Universe
                             Ensemble Ablation         -->  Live Trader (paper/live)
                             Position Sizing
```

---

## Phase 1: Data Pipeline (One-Time + Periodic Refresh)

### 1A. Fetch Data
```bash
# Core 10 assets (full history from 2020)
python src/pipeline/fetch_all.py

# Core 10 assets (recent only, last 30 days)
python src/pipeline/fetch_all.py --start-date "2026-03-01"

# Top 30 assets by volume/volatility (screens + fetches in one command)
python src/pipeline/fetch_all.py --top-n 30 --start-date "2026-03-01"

# Use pre-screened universe (from universe_screener.py output)
python src/pipeline/fetch_all.py --from-screener --start-date "2026-03-01"

# Specific assets
python src/pipeline/fetch_all.py --assets SUI/USDT PEPE/USDT NEAR/USDT --start-date "2026-03-01"
```
- Downloads aggTrades, funding, OI metrics from Binance
- Saves to `data/raw/{SYMBOL}/aggTrades/*.parquet`
- Runtime: 1-2 hours (first run), 10-30 min (refresh)
- Has resume: skips already-fetched dates via manifest

### 1B. Build Chimera Files (v51 SOTA: dollar bars + 34 base + 7 XD + 80 frontier + 11 helpers)
```bash
python src/pipeline/make_dataset.py            # primary v51 builder (current)
# or for V1-V14 retraining only (legacy v50 schema, 41 features):
# python src/pipeline/make_dataset_legacy.py
```
- Phase 1: raw trades -> dollar bars -> 34 base features + targets
- Phase 2: cross-asset enrichment (+7 XD features = 41 v50 total)
- Phase 3 (v51): 80 frontier features + tick_seq + uncapped target_return_h_raw + manifest + 4 cadence views
- Output: `data/processed/<SYMBOL>/v51.parquet` (+ v51_1d/4h/1h/15m + manifest)
- Legacy output (make_dataset_legacy.py): `data/processed/{SYMBOL}_v50_chimera.parquet`
- Runtime: 5-10 hours (all 10 assets)
- Has resume: skips assets with existing output file

### 1C. Inspect Data Quality
```bash
python src/pipeline/inspect_dataset.py --asset btcusdt --quick
python src/pipeline/inspect_dataset.py --all --strict
```
- Checks: nulls, feature std, targets, regime labels, XD features
- 13 gate checks in `--strict` mode
- Runtime: <1 minute

---

## Phase 2: Model Training

### 2A. Train V1 Models (primary)
```bash
# Train all V1 variants with 13 features
python src/wm/v1/run_training.py --features 13

# Train specific feature count
python src/wm/v1/run_training.py --features 25

# Force retrain (ignore existing checkpoints)
python src/wm/v1/run_training.py --features 13 --fresh
```
- Trains V1.0, V1.1, V1.4, V1.6 sequentially
- Runtime: 12-50 hours per feature count
- Has resume: skips variants with existing best_ema checkpoint
- Checkpoints: `models/v1/v1_{variant}/base/v1_{variant}_f{N}_wm_best_ema.pt`

### 2B. Train V2-V9 Models (architectural diversity)
```bash
python src/run_version_training.py --version 4 --features 13 --only base
python src/run_version_training.py --version 3 --features 13
```
- Runtime: 12-50 hours per version
- Pre-flight checks included (settings validation)

### 2C. Run Baselines (V0 -- non-DL reference)
```bash
python src/wm/v0/v0_baseline/linear_baseline.py --features 13
python src/wm/v0/v0_baseline/nonlinear_baselines.py --features 25
python src/wm/v0/v0_baseline/nonlinear_baselines.py --features 41  # includes SOTA
```
- Feature options: 13, 18, 21, 25, 30, 34, 37, 41
- Runtime: 5-20 minutes

---

## Phase 3: Strategy Analysis & Validation

### 3A. Strategy Sweep (single OOS split)
```bash
# All strategies across all assets (price-action only)
python src/analysis/strategy_lab.py --sweep

# With WM ensemble predictions
python src/analysis/strategy_lab.py --sweep --wm-ensemble

# Specific assets
python src/analysis/strategy_lab.py --sweep --assets btcusdt ethusdt
```
- Tests 20 strategies x multiple params x 10 assets
- Runtime: 30-60 minutes (price-only), 2-3 hours (with WM)
- Has resume: incremental save per asset
- Output: `logs/analysis/strategy_lab_*.json`

### 3B. Walk-Forward Validation (THE critical gate)
```bash
# Standard 5-fold walk-forward with WM
python src/analysis/walk_forward.py --wm-ensemble --folds 5

# With position sizing analysis included
python src/analysis/walk_forward.py --wm-ensemble --folds 5 --with-sizing

# Fresh run (ignore cached results)
python src/analysis/walk_forward.py --wm-ensemble --folds 5 --fresh
```
- 5 temporal folds with 400-bar purge gaps
- Tests ALL strategies on each fold IS/OOS
- Runtime: 3-8 hours
- Has resume: incremental save per asset
- Output: `logs/analysis/walk_forward_*.json`
- **This is the deployment gate. Only strategies that pass walk-forward should be traded.**

### 3C. Ensemble Ablation (which models to use)
```bash
python src/analysis/ensemble_ablation.py --sweep
python src/analysis/ensemble_ablation.py --sweep --fresh
```
- Tests: all models, solo, leave-one-out, feature groups, architecture groups
- Runtime: 2-3 hours
- Has resume: incremental save per asset
- Output: `logs/analysis/ensemble_ablation_*.json`

### 3D. Position Sizing
```bash
python src/analysis/position_sizing.py --wm-ensemble
python src/analysis/position_sizing.py --wm-ensemble --method kelly
```
- Methods: fixed, kelly, voltarget, all
- Runtime: 15-30 minutes
- Has resume: incremental save per asset

### 3E. Learned Gating (ensemble weight optimization)
```bash
python src/analysis/train_gating.py
python src/analysis/train_gating.py --epochs 100 --lr 0.001
```
- Trains XD-conditioned gate for model weighting
- Runtime: 10-15 minutes
- Output: `models/v1/gating/gating_best.pt`, `logs/analysis/gating_training_*.json`

### 3F. Meta-Strategy Comparison
```bash
python src/analysis/meta_strategy.py --mode compare --sweep
python src/analysis/meta_strategy.py --mode vote --min-agree 3 --sweep
```
- Compares: Voting, GBT meta-model, individual strategies
- Runtime: 15-30 minutes
- Has resume: incremental save per asset

### 3G. Live Backtest with HTML Report
```bash
python src/analysis/live_backtest.py --strategy donchian --asset btcusdt
python src/analysis/live_backtest.py --strategy wm_threshold --asset btcusdt --wm-ensemble
```
- Real-time terminal replay + HTML tearsheet
- Runtime: 2-5 minutes per asset
- Output: `reports/*.html`

---

## Phase 4: Production Trading

### 4A. Screen Trading Universe
```bash
# Screen top 30 by volume/volatility
python src/prod/universe_screener.py --top 30

# Top 50 with higher volume floor
python src/prod/universe_screener.py --top 50 --min-volume 10
```
- Fetches all USDT spot tickers from Binance (public, no auth)
- Scores by: volume rank + volatility rank + spread rank
- WM-trained assets get a 20-rank bonus
- Output: `data/prod_state/universe.json`, `data/prod_state/universe_assets.txt`
- Runtime: 1-2 minutes
- **Run daily before trading session**

### 4B. Fetch Recent Data for Universe
```bash
# Fetch 30 days for screened universe
python src/prod/fetch_universe.py --from-screener --days 30

# Fetch for specific assets
python src/prod/fetch_universe.py --assets btcusdt ethusdt pepeusdt --days 7

# Fetch for default 10 assets
python src/prod/fetch_universe.py --days 30
```
- Fetches aggTrades via Binance public REST
- Saves to `data/prod/{asset}/recent_trades.parquet`
- Skips assets with data <12 hours old
- Runtime: 1-2 hours (50 assets x 30 days)
- **Run daily before trading session**

### 4C. Paper Trading (SAFE -- no real orders)
```bash
# Paper trade on 10 core assets
python src/prod/live_trader.py --paper

# Paper trade specific assets
python src/prod/live_trader.py --paper --assets btcusdt ethusdt solusdt

# With debug logging
python src/prod/live_trader.py --paper --log-level DEBUG
```
- Multi-strategy voting (2-3 strategies per asset, min_agree=2)
- Auto-maps unknown assets to nearest trained WM asset (correlation-based)
- Risk management: portfolio DD breaker, per-asset DD, kill switch
- State persistence: `data/prod_state/state_paper.json`
- Logs: `logs/prod/trader_*.log`
- **Run this first for at least 1 week before live trading**

### 4D. Live Trading -- Testnet
```bash
python src/prod/live_trader.py --live --testnet
```
- Requires `.env` with testnet API keys:
  ```
  BINANCE_TESTNET_API_KEY=your_key
  BINANCE_TESTNET_API_SECRET=your_secret
  ```
- Real orders on Binance testnet (no real money)
- All safety features active

### 4E. Live Trading -- Mainnet (REAL MONEY)
```bash
python src/prod/live_trader.py --live --mainnet
```
- Requires `.env` with mainnet API keys
- **10-second countdown before starting** (Ctrl+C to abort)
- Trade audit log: `logs/prod/trades/trade_log.jsonl`
- Equity log: `logs/prod/equity_*.jsonl`

### 4F. Emergency Controls
```bash
# Check current state
python src/prod/live_trader.py --status

# Emergency: sell all positions and halt
python src/prod/live_trader.py --liquidate

# Alternative kill switch: just create this file
touch KILL_SWITCH
```

---

## Typical Workflow Sequences

### First-Time Setup (do once)
```bash
# 1. Fetch historical data (1-2h)
python src/pipeline/fetch_all.py

# 2. Build chimera files (5-10h)
python src/pipeline/make_dataset.py

# 3. Validate data
python src/pipeline/inspect_dataset.py --all --strict

# 4. Train V1 models (days)
python src/wm/v1/run_training.py --features 13
python src/wm/v1/run_training.py --features 25

# 5. Run walk-forward validation (3-8h)
python src/analysis/walk_forward.py --wm-ensemble --folds 5

# 6. Review results
# Check logs/analysis/walk_forward_*.json
```

### Daily Operations (paper trading)
```bash
# 1. Screen universe (2 min)
python src/prod/universe_screener.py --top 30

# 2. Fetch recent data (1-2h)
python src/prod/fetch_universe.py --from-screener --days 7

# 3. Paper trade (runs continuously)
python src/prod/live_trader.py --paper
```

### Going Live
```bash
# 1. Create .env with API keys
# 2. Test on testnet first
python src/prod/live_trader.py --live --testnet

# 3. After 1 week of testnet validation
python src/prod/live_trader.py --live --mainnet
```

### Re-Validation After Changes
```bash
# After modifying strategies or adding features:
python src/analysis/walk_forward.py --wm-ensemble --folds 5 --fresh --with-sizing

# After training new models:
python src/analysis/ensemble_ablation.py --sweep --fresh
```

---

## File Locations Reference

| What | Where |
|------|-------|
| Raw trade data | `data/raw/{SYMBOL}/aggTrades/*.parquet` |
| Chimera features | `data/processed/{SYMBOL}_v50_chimera.parquet` |
| Prod data | `data/prod/{asset}/recent_trades.parquet` |
| Model checkpoints | `models/v1/v1_{variant}/base/*_best_ema.pt` |
| Gating weights | `models/v1/gating/gating_best.pt` |
| Trading state | `data/prod_state/state_*.json` |
| Universe config | `data/prod_state/universe.json` |
| Asset mappings | `data/prod_state/asset_mappings.json` |
| Trade audit log | `logs/prod/trades/trade_log.jsonl` |
| Equity history | `logs/prod/equity_*.jsonl` |
| Session logs | `logs/prod/trader_*.log` |
| Analysis results | `logs/analysis/*.json` |
| Training logs | `logs/v1/v1_{variant}/*.log` |
| API keys | `.env` (NEVER commit) |
| Kill switch | `KILL_SWITCH` (touch to halt) |

---

## Dependencies

```bash
pip install torch numpy polars matplotlib ccxt python-dotenv pyyaml scikit-learn
```

Optional (for specific features):
```bash
pip install numba     # Required: pipeline feature computation
pip install pandas_ta # Optional: alternative technical indicators
```
