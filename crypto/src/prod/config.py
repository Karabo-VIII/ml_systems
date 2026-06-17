"""
Production Configuration
=========================

All tunables for live and paper trading in one place.
API keys loaded from .env file at project root.
"""
import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"


# Load dollar bar thresholds from data_config.yaml
def load_dollar_thresholds():
    """Load per-asset dollar bar thresholds from data_config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    thresholds = {}
    for asset_pair, spec in cfg.get("assets", {}).items():
        if spec.get("is_active", True):
            # "BTC/USDT" -> "btcusdt"
            key = asset_pair.replace("/", "").lower()
            thresholds[key] = spec["dollar_bar_size"]
    return thresholds


# --- Exchange ---
EXCHANGE_ID = "binance"
USE_TESTNET = True  # SAFETY: default to testnet until explicitly overridden
SPOT_MODE = True    # SPOT only (no futures)

# API keys come from .env:
#   BINANCE_API_KEY=...
#   BINANCE_API_SECRET=...
# For testnet:
#   BINANCE_TESTNET_API_KEY=...
#   BINANCE_TESTNET_API_SECRET=...

# --- Assets ---
ACTIVE_ASSETS = [
    "btcusdt", "ethusdt", "solusdt", "bnbusdt", "xrpusdt",
    "dogeusdt", "adausdt", "avaxusdt", "linkusdt", "ltcusdt",
]

# Binance WebSocket symbols (lowercase, no separator)
def ws_symbol(asset: str) -> str:
    """Convert 'btcusdt' to WebSocket stream name."""
    return asset.lower()

# ccxt symbol format
def ccxt_symbol(asset: str) -> str:
    """Convert 'btcusdt' -> 'BTC/USDT'."""
    asset = asset.upper()
    if asset.endswith("USDT"):
        base = asset[:-4]
        return f"{base}/USDT"
    return asset

# --- Bar Accumulation ---
BAR_BUFFER_SIZE = 2000       # Rolling buffer of completed bars
DOLLAR_THRESHOLDS = None     # Loaded lazily

def get_dollar_threshold(asset: str) -> float:
    """Get dollar bar threshold for an asset."""
    global DOLLAR_THRESHOLDS
    if DOLLAR_THRESHOLDS is None:
        DOLLAR_THRESHOLDS = load_dollar_thresholds()
    return DOLLAR_THRESHOLDS.get(asset.lower(), 200_000)

# --- Costs (must match backtest assumptions) ---
SPOT_FEE = 0.001       # 0.10% per side (taker)
SPOT_SLIPPAGE = 0.0002  # 0.02% estimated slippage

# --- Risk Management ---
MAX_PORTFOLIO_DRAWDOWN = 0.10   # 10% portfolio DD -> halt all new entries
MAX_ASSET_DRAWDOWN = 1.00       # DISABLED (was 0.05). Stop-losses hurt crypto per experiment log.
                                # RED TEAM: "ATR3 trailing stop destructive on 7/10 assets"
                                # "SL5 tolerable but always worse than baseline"
CAPITAL_ALLOCATION = 0.90       # Use 90% of available USDT
MAX_POSITION_PER_ASSET = 0.15   # Max 15% of portfolio per asset (10 assets = 150% max)
KILL_SWITCH_FILE = PROJECT_ROOT / "KILL_SWITCH"  # Touch this file to halt trading

# --- Polling ---
POLL_INTERVAL_SECONDS = 10   # REST poll interval (+ ~5s stagger between assets)
WS_RECONNECT_DELAY = 5      # Seconds before WebSocket reconnect attempt
WS_MAX_RECONNECTS = 50      # Max consecutive reconnects before halt

# --- Paths ---
STATE_DIR = PROJECT_ROOT / "data" / "prod_state"
LOG_DIR = PROJECT_ROOT / "logs" / "prod"
TRADE_LOG_DIR = LOG_DIR / "trades"

# --- Feature Settings ---
# Features computed on rolling bar buffer, matching pipeline
FEATURE_COUNT = 30   # Base features (no cross-asset XD in live mode)
SEQ_LEN = 96         # Sequence length for WM inference
