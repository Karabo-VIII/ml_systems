"""
V10 Settings -- Multi-Brain Meta-Ensemble

Architecture: Frozen V1-V9 + Learned Dynamic Router
The router is a tiny MLP (~10K params) that produces per-model weights
based on context signals (rolling IC, regime, volatility).
"""
import math
import platform
import torch
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
while not (PROJECT_ROOT / "data").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera_legacy"
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v10" / "v10"
LOG_DIR = PROJECT_ROOT / "logs" / "v10" / "v10"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0 if IS_WINDOWS else 2

# Shared constants (same as all V1-V9)
FEATURE_LIST = [
    # Base features (0-16) -- per-asset, computed in sota_shared_logic_v50
    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",
    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",
    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",
    "norm_spread_bps",
    "norm_whale", "norm_efficiency", "norm_return_4", "norm_return_16",
    # Cross-asset features (17-21) -- computed in make_dataset_legacy.py (Phase 2)
    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
    "xd_cross_return_mean", "xd_cross_vol_mean",
]
INPUT_DIM = len(FEATURE_LIST)  # 22

ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
NUM_ASSETS = len(ASSET_LIST)
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

REWARD_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]
RAW_TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]

NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# V10 Meta-Ensemble Architecture
META_N_MODELS = 9                  # V1 through V9
META_CONTEXT_DIM = 46              # 9*4 (rolling IC per model per horizon) + 3 (regime) + 1 (vol) + 6 (misc)
META_ROUTER_HIDDEN = 64            # Router MLP hidden dim
META_TEMPERATURE = 1.0             # Softmax temperature for routing weights

# Router produces per-horizon weights (different models may be better at different horizons)
META_PER_HORIZON_ROUTING = True    # If True, separate weights per horizon

# Which models to include (set False to exclude models without checkpoints)
META_MODEL_ENABLED = {i: True for i in range(1, 10)}  # V1-V9 all enabled by default

# Training
META_LR = 1e-3                     # Higher LR for tiny router
META_WEIGHT_DECAY = 1e-3
META_EPOCHS = 30
META_STEPS_PER_EPOCH = 500
META_BATCH_SIZE = 64
META_VAL_EVERY = 5
META_GRAD_CLIP = 1.0

# WM settings needed for data loading
WM_SEQ_LEN = 96

# VRAM management: run models in sequence, not parallel
# With 8GB VRAM, can hold ~2-3 models simultaneously
META_MAX_MODELS_IN_VRAM = 2        # Load/unload models to fit in VRAM
META_CACHE_PREDICTIONS = True      # Cache predictions to avoid re-running models

# Validation
GATE_IC_MIN = 0.015


def get_lr_for_epoch(epoch: int) -> float:
    """Cosine LR schedule with warmup for the router."""
    if epoch < 3:
        return 1e-5 + (META_LR - 1e-5) * ((epoch + 1) / 3)
    progress = (epoch - 3) / max(1, META_EPOCHS - 3)
    return 1e-5 + 0.5 * (META_LR - 1e-5) * (1 + math.cos(math.pi * progress))


# Cross-version invariant (CLAUDE.md): target source for all training.
# Added 2026-05-16 — was missing per wm_deep_audit framework finding.
target_prefix = "target_return"  # raw returns (voladj DEPRECATED)
