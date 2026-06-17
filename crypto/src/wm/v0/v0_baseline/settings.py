"""
V0 Settings -- Shared Constants for Linear & Non-Linear Baselines

This is the non-DL baseline. No model architecture, no GPU.
Tests whether the engineered features carry signal that any DL model
(V1-V9) must beat to justify its complexity.

Feature lists are imported from src/feature_sets.py (single source of truth
across V0-V19, post-2026-04-27 centralization). All counts in
SUPPORTED_COUNTS = [13, 18, 21, 25, 29, 30, 34, 37, 41, 46, 60, 73, 78, 81,
84, 97, 110, 121] are usable via --features <N>.
"""
import platform
import sys
from pathlib import Path

# Import central feature-set contract from src/feature_sets.py
_SRC_DIR = Path(__file__).resolve().parents[3]   # this file is at <root>/src/wm/v0/v0_baseline/ -> parents[3]=src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_21,
    FEATURE_LIST_25, FEATURE_LIST_29, FEATURE_LIST_30,
    FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    FEATURE_LIST_46, FEATURE_LIST_60, FEATURE_LIST_73,
    FEATURE_LIST_78, FEATURE_LIST_81, FEATURE_LIST_84,
    FEATURE_LIST_97, FEATURE_LIST_110, FEATURE_LIST_121,
    XD_FEATURES_7,
    get_feature_config as _central_get_feature_config,
    list_supported as _central_list_supported,
)

# =============================================================================
# PLATFORM
# =============================================================================

IS_WINDOWS = platform.system() == "Windows"

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
# This file lives at <root>/src/wm/v0/v0_baseline/settings.py.
# SCRIPT_DIR.parents = [v0/, wm/, src/, ROOT]; ROOT is parents[3].
# (Post-2026-04-29 src/wm/ migration: was parents[2] when file lived at
# <root>/src/v0/v0_baseline/.)
PROJECT_ROOT = SCRIPT_DIR.parents[3]

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v0" / "v0"
LOG_DIR = PROJECT_ROOT / "logs" / "v0" / "v0"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FEATURES -- imported from central src/feature_sets.py (single source of truth)
# =============================================================================

# Default V0 working list (full v50 = 41 features)
FEATURE_LIST = FEATURE_LIST_41
INPUT_DIM = len(FEATURE_LIST)  # 41

# Re-export get_feature_config so V0 callers have the standardized 3-tuple API
get_feature_config = _central_get_feature_config
list_supported_features = _central_list_supported

# Lazy f121 helper kept for any legacy caller; new code should use get_feature_config(121)
def get_feature_list_121():
    return FEATURE_LIST_121

# Backward compat aliases for any older code that referenced these names
FEATURE_LIST_17 = FEATURE_LIST_18    # old "17" maps to 18 (includes ma_distance)
FEATURE_LIST_20 = FEATURE_LIST_21    # old "20" maps to 21
FEATURE_LIST_22 = FEATURE_LIST_18 + XD_FEATURES_7  # extended + XD (legacy combo)

# Feature GROUP definitions (for multi-head masking comparison)
# Slices into the canonical FEATURE_LIST_41 ordering.
FEATURE_GROUPS = {
    "base13": FEATURE_LIST_41[:13],
    "extended5": FEATURE_LIST_41[13:18],
    "tier1": FEATURE_LIST_41[18:21],
    "hawkes": FEATURE_LIST_41[21:25],
    "ic_boost": FEATURE_LIST_41[25:30],
    "sota": FEATURE_LIST_41[30:34],
    "xd": FEATURE_LIST_41[34:41],
}

# Aliases (backward compat)
FEATURE_LIST_BASE = FEATURE_LIST_13
INPUT_DIM_BASE = 13


# =============================================================================
# ASSETS
# =============================================================================

ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
NUM_ASSETS = len(ASSET_LIST)
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}


# =============================================================================
# TARGETS -- Multi-horizon return predictions
# =============================================================================

REWARD_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]


# =============================================================================
# BASELINE CONFIG (non-DL)
# =============================================================================

# 4-way data split: 50/20/20/10 (train/val/oos/unseen)
TRAIN_RATIO = 0.50
VAL_RATIO = 0.20
OOS_RATIO = 0.20
UNSEEN_RATIO = 0.10

# Purge gap: prevents normalization leakage from rolling z-score (200-bar window)
PURGE_GAP_BARS = 400

# Ridge regularization search space
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]

# Shuffled IC seeds (for anti-memorization check)
N_SHUFFLE_SEEDS = 5

# WM_SEQ_LEN not used by linear baseline, but kept for import compatibility
WM_SEQ_LEN = 96

# Cross-version invariant (CLAUDE.md): target source for all training
target_prefix = "target_return"  # raw returns (voladj DEPRECATED)
