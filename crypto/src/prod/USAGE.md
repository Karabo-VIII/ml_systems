# Production Trading Engine -- Usage Guide

## Architecture

```
Binance REST aggTrades (poll every 5s)
    |
    v
BarAccumulator (per-asset dollar bar construction)
    |  - Accumulates trades by dollar volume threshold
    |  - Emits completed bar when threshold crossed
    |  - Maintains rolling buffer of 2000 bars
    v
FeatureCalculator (30 base features via sota_shared_logic_v50)
    |
    v
SignalEngine (runs backtested strategies on live bars)
    |  - Same strategy classes from strategy_lab.py
    |  - Per-asset strategy assignment (from backtest results)
    |  - Detects signal changes (FLAT->LONG, LONG->FLAT)
    v
RiskManager (pre-trade risk checks)
    |  - Portfolio circuit breaker (10% max DD)
    |  - Per-asset drawdown limit (5% from entry)
    |  - Kill switch (touch KILL_SWITCH file)
    |  - Position size caps (15% per asset)
    v
Exchange (ccxt Binance SPOT, market orders)
    |  - Retry logic (3 attempts with backoff)
    |  - Precision formatting for Binance
    |  - Trade audit log (JSONL)
    v
StateManager (JSON persistence)
    - Atomic writes (temp -> rename)
    - Position tracking, equity history
    - Restart recovery
```

## Modules

| Module | File | Purpose |
|--------|------|---------|
| Config | `config.py` | All tunables, dollar thresholds from data_config.yaml |
| Exchange | `exchange.py` | Binance ccxt wrapper with retry + logging |
| BarAccumulator | `bar_accumulator.py` | Live dollar bar construction |
| FeatureCalculator | `feature_calculator.py` | 30 base features from pipeline |
| SignalEngine | `signal_engine.py` | Strategy signal generation |
| RiskManager | `risk_manager.py` | Portfolio + per-asset risk controls |
| StateManager | `state_manager.py` | Persistent state + trade log |
| LiveTrader | `live_trader.py` | Main loop tying everything together |

## Quick Start

```powershell
# 1. Paper trading (no API keys needed, safe)
python src/prod/live_trader.py --paper

# 2. Paper trading on specific assets
python src/prod/live_trader.py --paper --assets btcusdt ethusdt

# 3. Check current state
python src/prod/live_trader.py --status

# 4. Live trading on TESTNET (needs testnet API keys)
python src/prod/live_trader.py --live --testnet

# 5. Live trading on MAINNET (REAL MONEY)
python src/prod/live_trader.py --live --mainnet

# 6. Emergency liquidation
python src/prod/live_trader.py --liquidate
```

## Setup for Live Trading

### 1. Install Dependencies
```powershell
pip install ccxt python-dotenv pyyaml
```

### 2. Create .env File
```
# At project root: .env
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here

# For testnet:
BINANCE_TESTNET_API_KEY=your_testnet_key
BINANCE_TESTNET_API_SECRET=your_testnet_secret
```

### 3. Strategy Configuration
The engine uses default strategies from backtest results:
- BTC: VPIN_Trigger
- ETH: VPIN_Trigger
- SOL/BNB/ADA: FlowMomentum
- XRP: Donchian
- DOGE/LTC: HurstAdaptive
- AVAX: VPIN_Trigger
- LINK: VolBreakout

Override with `--strategy-config path/to/selected_strategies.json`

## Safety Features

1. **Default to paper mode** -- no real orders unless `--live` flag
2. **Default to testnet** -- real orders go to testnet unless `--mainnet`
3. **10-second countdown** -- mainnet mode waits 10s for abort
4. **Kill switch** -- touch `KILL_SWITCH` file at project root to halt
5. **Portfolio circuit breaker** -- 10% portfolio drawdown halts entries
6. **Per-asset stop** -- 5% drawdown from entry auto-closes position
7. **Atomic state saves** -- crash-safe JSON persistence
8. **Trade audit log** -- every order logged to `logs/prod/trades/trade_log.jsonl`

## File Locations

| Path | Purpose |
|------|---------|
| `data/prod_state/state_*.json` | Trading state (positions, equity) |
| `logs/prod/trader_*.log` | Session logs |
| `logs/prod/trades/trade_log.jsonl` | Trade audit trail |
| `logs/prod/equity_*.jsonl` | Equity snapshots |
| `KILL_SWITCH` | Emergency halt trigger (touch to create) |
| `.env` | API keys (NEVER commit) |

## Known Limitations (to be addressed)

1. **REST polling only** -- no WebSocket aggTrades yet (5s poll interval)
2. **No cross-asset features** -- XD features require synchronized bars
3. **No funding/OI data** -- live features 5,10 are zero (OK for strategies)
4. **Market orders only** -- no limit orders or stop-losses at exchange level
5. **No alerting** -- no Telegram/Slack/email notifications on trades or errors
