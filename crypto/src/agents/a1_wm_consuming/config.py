"""
Agent Configuration
===================
Hyperparameters for the trading agent, environment, and training.
Designed for RTX 4060 (8GB VRAM) with world models using ~2GB in eval.
"""

import torch
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

while not (PROJECT_ROOT / "data").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy"
AGENT_MODEL_DIR = PROJECT_ROOT / "models" / "agent"
AGENT_LOG_DIR = PROJECT_ROOT / "logs" / "agent"
AGENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Assets (must match world model settings)
# ---------------------------------------------------------------------------

ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
NUM_ASSETS = len(ASSET_LIST)
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# World model return prediction horizons
REWARD_HORIZONS = [1, 4, 16, 64]
# Active horizons for agent observations (h16/h64 reverse OOS = memorization artifacts)
ACTIVE_HORIZONS = [1, 4]

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Observation dimensions (per asset, from world model)
OBS_RETURN_PREDS = len(ACTIVE_HORIZONS)   # 2 (h1, h4 only; h16/h64 excluded)
OBS_REGIME_PROBS = 3                       # bearish / neutral / bullish
OBS_UNCERTAINTY = 1                        # posterior entropy

# Portfolio state
OBS_POSITION = 1                           # current position in asset
OBS_UNREALIZED_PNL = 1                     # unrealized P&L for asset

# Total observation dim = NUM_ASSETS * (per_asset_features) + global_features
PER_ASSET_OBS_DIM = OBS_RETURN_PREDS + OBS_REGIME_PROBS + OBS_UNCERTAINTY + OBS_POSITION + OBS_UNREALIZED_PNL
# = 2 + 3 + 1 + 1 + 1 = 8
GLOBAL_OBS_DIM = 1  # cash balance (fraction of initial capital)
TOTAL_OBS_DIM = NUM_ASSETS * PER_ASSET_OBS_DIM + GLOBAL_OBS_DIM
# = 10 * 8 + 1 = 81

# Action space: target position per asset in [-1, +1]
ACTION_DIM = NUM_ASSETS  # 10

# Trading parameters
INITIAL_CAPITAL = 10_000.0      # Starting capital (USD)
MAX_POSITION_FRAC = 0.20        # Max 20% of capital per asset
BARS_PER_HOUR = 4               # 96 bars ~ 24h => 4 bars/hour (dollar bars)
BARS_PER_DAY = BARS_PER_HOUR * 24  # ~96

# --- Perpetual futures costs (opt-in via --perp) ---
PERP_FEE_BPS = 4.0              # 0.04% taker fee (Binance Futures, consistent with strategy_lab)
PERP_SLIPPAGE_BPS = 1.0         # 1 bps slippage estimate
PERP_FUNDING_RATE_HOURLY = 0.000125  # ~0.01% per 8h (Binance Futures typical)

# --- SPOT costs (DEFAULT) ---
SPOT_FEE_BPS = 10.0             # 0.10% taker fee (Binance Spot standard)
SPOT_SLIPPAGE_BPS = 2.0         # 2 bps slippage estimate
# No funding in SPOT mode

# Active cost model (overridden by CLI --spot/--perp)
TAKER_FEE_BPS = SPOT_FEE_BPS    # Default: SPOT
MAKER_FEE_BPS = 2.0             # 0.02% maker fee (not used in current impl)
SLIPPAGE_BPS = SPOT_SLIPPAGE_BPS
FUNDING_RATE_HOURLY = 0.0       # Default: SPOT (no funding)

# Decision interval: agent decides every N bars (reduces per-candle trading noise)
# Agent still OBSERVES every bar but only EXECUTES trades every DECISION_INTERVAL
# Per-bar trading is catastrophic after costs (expected gain ~0.03% vs cost ~0.12%)
DECISION_INTERVAL = 64          # ~5.5 hours at 96 bars/day

# Portfolio-level risk
MAX_GROSS_EXPOSURE = 1.0        # Max sum(|position_frac|) across all assets

# Drawdown circuit breaker (progressive)
DD_REDUCE_THRESHOLD = 0.15      # At 15% drawdown: halve positions
DD_KILL_THRESHOLD = 0.25        # At 25% drawdown: terminate episode

# Episode parameters
EPISODE_LENGTH = 256            # Bars per episode (for training)
WARMUP_BARS = 96                # World model needs 96 bars of context
MIN_EPISODE_BARS = WARMUP_BARS + EPISODE_LENGTH  # Total bars needed per episode

# Walk-forward validation split
VAL_FRACTION = 0.20             # Last 20% for validation (50/20/20/10 split)
PURGE_GAP_BARS = 400            # Same as world model training


# ---------------------------------------------------------------------------
# Policy Network
# ---------------------------------------------------------------------------

POLICY_HIDDEN_DIM = 128         # Hidden layer size
POLICY_N_LAYERS = 3             # Number of hidden layers
POLICY_ACTIVATION = "silu"      # Activation function
POLICY_LOG_STD_MIN = -2.0       # Min log std (sigma=0.14; prevents near-deterministic collapse)
POLICY_LOG_STD_MAX = -0.5       # Max log std (sigma=0.61; was 0.5/sigma=1.65 which made actions ~uniform)
VALUE_HIDDEN_DIM = 128          # Value network hidden size
VALUE_N_LAYERS = 3              # Value network layers


# ---------------------------------------------------------------------------
# PPO Hyperparameters
# ---------------------------------------------------------------------------

# Optimization
PPO_LR = 3e-4                  # Learning rate
PPO_EPOCHS = 4                 # Epochs per PPO update
PPO_BATCH_SIZE = 64            # Minibatch size
PPO_GAMMA = 0.99               # Discount factor
PPO_GAE_LAMBDA = 0.95          # GAE lambda

# Clipping
PPO_CLIP_EPS = 0.2             # Policy clip epsilon
PPO_CLIP_VALUE = 0.2           # Value clip epsilon
PPO_MAX_GRAD_NORM = 0.5        # Gradient clipping

# Entropy
PPO_ENTROPY_COEFF = 0.01       # Entropy bonus (exploration)
PPO_VALUE_COEFF = 0.5          # Value loss coefficient

# Rollout
PPO_N_STEPS = 2048             # Steps per rollout
TOTAL_TIMESTEPS = 2_000_000    # Total training steps

# Training schedule
EVAL_EVERY = 10_000            # Evaluate on validation set every N steps
SAVE_EVERY = 50_000            # Save checkpoint every N steps
LOG_EVERY = 1_000              # Log metrics every N steps


# ---------------------------------------------------------------------------
# Reward Function
# ---------------------------------------------------------------------------

REWARD_SCALE = 100.0            # Scale raw returns (they're ~0.001 per bar)
REWARD_ASYMMETRY = 2.0          # Loss penalty multiplier (losses hurt 2x)
REWARD_COST_PENALTY = 1.0       # Transaction cost penalty multiplier
REWARD_DRAWDOWN_PENALTY = 0.5   # Per-step penalty when in drawdown
REWARD_SHARPE_WINDOW = 50       # Rolling window for Sharpe-based reward component


# ---------------------------------------------------------------------------
# Stress Augmentation
# ---------------------------------------------------------------------------

# Master switch: probability any given episode is augmented (0.0 = off)
AUGMENT_PROB = 0.15  # 15% of episodes

# Per-scenario configuration within an augmented episode.
# Probabilities are relative weights (normalized internally by StressAugmentor).
AUGMENT_SCENARIOS = {
    "flash_crash": {
        "prob": 0.30,
        "min_bars": 5,
        "max_bars": 30,
        "min_shock": -0.05,   # minimum per-bar return shock (fractional)
        "max_shock": -0.015,  # maximum (least negative) per-bar return shock
    },
    "vol_spike": {
        "prob": 0.30,
        "min_bars": 10,
        "max_bars": 50,
        "min_mult": 2.0,      # minimum return multiplier
        "max_mult": 4.0,      # maximum return multiplier
    },
    "squeeze": {
        "prob": 0.20,
        "min_bars": 8,
        "max_bars": 30,
        "momentum_bars": 10,  # bars of positive momentum before reversal
        "reversal_mult": -2.0,  # reversal magnitude relative to momentum
    },
    "cost_shock": {
        "prob": 0.20,
        "min_bars": 20,
        "max_bars": 80,
        "min_cost_mult": 2.0,  # minimum effective cost multiplier
        "max_cost_mult": 5.0,  # maximum effective cost multiplier
    },
}
