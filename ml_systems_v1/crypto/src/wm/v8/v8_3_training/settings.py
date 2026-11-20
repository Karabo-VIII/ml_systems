"""

V8.3 Settings -- Neural ODE No Dynamics Regularization Variant



Ablation: Removes dynamics smoothness regularization (lambda=0).

Tests whether the ODE learns better without explicit complexity

penalties on dh/dt magnitudes.



Hardware Target: RTX 4060 (8GB) + i9, Windows 11

"""

import math

import platform

import torch

from pathlib import Path



# =============================================================================

# INFRASTRUCTURE

# =============================================================================



IS_WINDOWS = platform.system() == "Windows"



SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR



while not (PROJECT_ROOT / "data").exists():

    if PROJECT_ROOT.parent == PROJECT_ROOT:

        PROJECT_ROOT = SCRIPT_DIR.parent.parent

        break

    PROJECT_ROOT = PROJECT_ROOT.parent



DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 migration 2026-05-25

MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v8" / "v8_3"

LOG_DIR = PROJECT_ROOT / "logs" / "v8" / "v8_3"

BASE_MODEL_DIR = MODEL_DIR / "base"

ADAPTER_MODEL_DIR = MODEL_DIR / "adapter"

ENSEMBLE_MODEL_DIR = MODEL_DIR / "ensemble"

NCL_MODEL_DIR = MODEL_DIR / "ncl"

for _d in [MODEL_DIR, LOG_DIR, BASE_MODEL_DIR, ADAPTER_MODEL_DIR, ENSEMBLE_MODEL_DIR, NCL_MODEL_DIR]:

    _d.mkdir(parents=True, exist_ok=True)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_WORKERS = 0 if IS_WINDOWS else 2





# =============================================================================

# FEATURES

# =============================================================================



FEATURE_LIST = [

    # Base features (0-16) -- per-asset, computed in sota_shared_logic_v50

    "norm_deviation", "norm_fd_close", "norm_vpin", "norm_flow_imbalance",

    "norm_vol_cluster", "norm_funding", "norm_tick_count", "norm_log_volume",

    "norm_hl_spread", "hurst_regime", "norm_oi_change", "norm_return_1",

    "norm_spread_bps",

    "norm_ma_distance",

    "norm_whale", "norm_efficiency", "norm_return_4", "norm_return_16",

    # Tier 1 features -- orthogonal replacements (V51b)

    "norm_return_kurtosis", "norm_bar_duration", "norm_funding_momentum",


    # Hawkes features (trade clustering dynamics)
    "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
    "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
    # IC-boosting features (Tier 2 -- dynamics, not levels)
    "norm_momentum_accel", "norm_vol_price_corr", "norm_vol_ratio",
    "norm_flow_persistence", "norm_oi_price_divergence",
    # SOTA features (30-33)
    "norm_yz_volatility",        # 30: Yang-Zhang vol (MVUE, overnight+RS+OC)
    "norm_cs_spread",            # 31: Corwin-Schultz bid-ask spread from H/L
    "norm_perm_entropy",         # 32: Permutation entropy (predictability/complexity)
    "norm_kyle_lambda",          # 33: Kyle's lambda (price impact per $ order flow)
    # Cross-asset features (20-24) -- computed in make_dataset_legacy.py (Phase 2)

    "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",

    "xd_cross_return_mean", "xd_cross_vol_mean",
    "xd_ma_distance", "xd_momentum_rank",

]

INPUT_DIM = len(FEATURE_LIST)  # 41





# === FEATURE SELECTION (13 / 18 / 30 / 37) ===

# 13: V1.0 compat (old base features only)

# 18: Legacy + extended base (13 + 5 extended)

# 30: All 30 base features (no XD)

# 37: Full (30 base + 7 XD) -- default

FEATURE_LIST_13 = FEATURE_LIST[:13]   # old base only (backward compat)

FEATURE_LIST_18 = FEATURE_LIST[:18]   # legacy + extended base (no tier1/hawkes/xd)

FEATURE_LIST_30 = FEATURE_LIST[:30]   # all base features (no XD)

FEATURE_LIST_37 = FEATURE_LIST        # full 37 features (default)



BASE_DIM = 34              # Features [0:BASE_DIM] are "base" (safe for posterior/recon)

XD_DROPOUT_RATE = 0.85     # SOTA-2026: was 0.7; CC-H4 anti-mem (cohort-aligned)

XD_NOISE_STD = 0.3         # Heavy noise on XD features (vs 0.02 base aug noise)

# Import frontier feature lists from the central registry (post-2026-05-25)
import sys as _sys
_SRC_DIR = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_34, FEATURE_LIST_41,
    FEATURE_LIST_127, FEATURE_LIST_133, FEATURE_LIST_154, FEATURE_LIST_161,
)



def get_feature_config(n_features: int):

    """Return (feature_list, input_dim, base_dim) for the requested feature count."""

    configs = {
        13: (FEATURE_LIST_13, 13, 13),    # legacy base only
        18: (FEATURE_LIST_18, 18, 18),    # legacy + extended base
        30: (FEATURE_LIST_30, 30, 30),    # + IC-boost (no SOTA/XD)
        34: (FEATURE_LIST_34, 34, 34),    # all base incl SOTA (no XD)
        37: (FEATURE_LIST_37, 37, 30),    # legacy full (30 base + 7 XD, backward compat)
        41: (FEATURE_LIST_41, 41, 34),    # full (34 base + 7 XD)
        127: (FEATURE_LIST_127, 127, 41),  # f121 + 6 RV/jump features
        133: (FEATURE_LIST_133, 133, 41),  # f127 + 6 TE features
        154: (FEATURE_LIST_154, 154, 41),  # f133 + 21 T2 microstructure cols
        161: (FEATURE_LIST_161, 161, 41),  # f154 + 7 extended cross-exchange signals
    }
    if n_features not in configs:
        raise ValueError(f"n_features must be one of {sorted(configs)}, got {n_features}")
    return configs[n_features]





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

# TARGETS

# =============================================================================



REWARD_HORIZONS = [1, 4, 16, 64]

# All horizons active for loss computation.

# h16/h64 act as multi-scale regularizers that prevent temporal memorization.

# Without them, ShIC declines monotonically from epoch 10 (confirmed Feb 2026 vs Mar 2026 runs).

ACTIVE_HORIZONS = [1, 4, 16, 64]

TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]

RAW_TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]





# =============================================================================

# V8 NEURAL ODE ARCHITECTURE

# =============================================================================



# Initial state encoder

WM_D_MODEL = 256



# ODE Dynamics Network

ODE_HIDDEN_LAYERS = [256, 512, 512, 256]  # MLP layers for f_theta

ODE_METHOD = "rk4"

ODE_STEP_SIZE = 1.0       # Base integration step size (interval between discrete timesteps)

ODE_SUBSTEPS = 2           # Sub-steps per interval (effective step = ODE_STEP_SIZE / ODE_SUBSTEPS)

                           # 1 = original fixed-step, 2+ = finer integration for stiff dynamics

                           # Higher = more accurate but more VRAM/compute



# RSSM Latent Space (lighter for ODE -- dynamics carry information)

RSSM_LATENT_DIM = 24

RSSM_CLASSES = 24

FLAT_DIM = RSSM_LATENT_DIM * RSSM_CLASSES  # 576



# Dynamics regularization

LAMBDA_DYNAMICS = 0.0  # V8.3: no dynamics reg (was 0.01)     # Penalize complex dynamics



# Asset Conditioning

WM_ASSET_EMB_DIM = 32



# TwoHot Symlog (return prediction) -- raw return targets

NUM_BINS = 255              # 255 bins across [-1, 1] for raw return targets

BIN_MIN = -1.0              # Raw return targets, range ~[-0.5, 0.5]

BIN_MAX = 1.0               # 255 bins in [-1, 1] -> resolution 0.0078 per bin

# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"



# SOTA Return Prediction Heads

RETURN_HEAD_DIM = 384

RETURN_HEAD_DROPOUT = 0.05

GUMBEL_TAU_START = 1.0       # Initial temperature (soft categorical, more exploration)

GUMBEL_TAU_END = 0.5         # Final temperature (sharper categorical)

GUMBEL_TAU_ANNEAL_EPOCHS = 50

GUMBEL_TAU = GUMBEL_TAU_END  # Default for inference (use final/sharp value)

DIRECT_RETURN_WEIGHT = 3.0  # Huber dominance acts as temporal regularizer (stable ShIC=0.030 in Feb runs)



# Loss initialization

LOG_VAR_INIT = [1.0, 0.0, -2.0, -2.0, -2.0, -2.0, -1.5]



# Kendall Weight Corridors (prevent reconstruction from hogging gradients)

# Root cause: Kendall optimal s_i = log(2*L_i). Easy tasks (rec) drift to high weight,

# hard tasks (returns) drift to low weight. Clamp asymmetrically to maintain return priority.

REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q: rec gets at most exp(-0.5)=0.61x weight (canonical per CLAUDE.md)

RETURN_LOG_VAR_CLAMP_MAX = -2.0

# RETURN_LOG_VAR_CLAMP_PER_H = {1: -2.0, 4: -2.0, 16: -1.0, 64: -1.0}  # h16/h64 reduced (memorize OOS)  # DISABLED: uniform clamp via RETURN_LOG_VAR_CLAMP_MAX

REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight

REGIME_FOCAL_GAMMA = 2.0



# TwoHot Focal Loss -- de-weights easy center-bin samples, up-weights tails

# TWOHOT_FOCAL_GAMMA = 1.0  # REMOVED: focal upweights temporally-clustered tail returns, accelerates memorization

# Cross-version invariant: TwoHot focal gamma is DISABLED (focal upweights
# temporally-clustered tail returns and accelerates memorization). All
# variants inherit base model's TwoHot head; this constant exists for
# `from settings import *` callers.
TWOHOT_FOCAL_GAMMA = 0.0

REGIME_HEAD_DIM = 256               # Regime head hidden dim (was 128 — capacity imbalance)



# ShIC-based early stopping (stop when memorization sets in)

SHUFFLED_IC_PATIENCE = 5            # Stop after N consecutive ShIC declines

SHUFFLED_IC_MIN_DECLINE = 0.001     # Only count as decline if drop > this from best

# SHIC_LR_REDUCE_AFTER = 3      # ATME fights first, LR reduces as fallback  # DISABLED: ShIC LR reduction removed

# SHIC_LR_REDUCE_FACTOR = 0.5   # Multiply LR by this factor per ShIC-triggered reduction  # DISABLED: ShIC LR reduction removed



# Regularization

WM_DROPOUT = 0.15

WM_FREE_NATS = 1.0

KL_ANNEAL_EPOCHS = 20



# Data Augmentation

AUG_NOISE_STD = 0.02

AUG_FEAT_DROP = 0.1





# =============================================================================

# TRAINING

# =============================================================================



WM_SEQ_LEN = 96

WM_BATCH_SIZE = 32  # Small batch = implicit gradient noise regularization (prevents ShIC decline)

WM_LR = 1e-4             # Slower for ODE stability

WM_WEIGHT_DECAY = 8e-2

WM_GRAD_CLIP = 1.0



WM_TOTAL_EPOCHS = 200

WM_STEPS_PER_EPOCH = 2000

WM_VAL_EVERY = 5

WM_PATIENCE = 40



WM_WARMUP_EPOCHS = 10    # Longer warmup for ODE stability

WM_MIN_LR = 1e-6



WM_MASK_RATIO_START = 0.10

WM_MASK_RATIO_END = 0.25

WM_MASK_RAMP_EPOCHS = 40

WM_BLOCK_SIZE_RATIO = 0.10



LOG_FREQ = 25



# EMA

EMA_DECAY = 0.995           # EMA model decay rate





# =============================================================================

# VALIDATION GATES

# =============================================================================



GATE_REC_MSE_MAX = 0.10

GATE_IC_MIN = 0.015

GATE_KL_MIN = 0.01

GATE_KL_MAX = 15.0

GATE_SHUFFLED_IC_RATIO_MIN = 0.3  # Shuffled IC / Contiguous IC (anti-memorization)

GATE_LOSS_RATIO_MAX = 2.0         # Train/Val loss ratio (overfitting detection)





# =============================================================================

# V8.x ADAPTIVE RESIDUAL ADAPTER

# =============================================================================



# Architecture -- FiLM modulation of return trunk, ~15K params

ADAPTER_FEAT_DIM = WM_D_MODEL + FLAT_DIM        # 832 (cat(h_seq, z_post))

ADAPTER_CONTEXT_DIM = 12                        # Rolling IC(4) + bias(4) + regime(3) + vol(1)

ADAPTER_BOTTLENECK = 16                         # Compressed feat representation

ADAPTER_FILM_HIDDEN = 48                        # FiLM generator hidden dim

ADAPTER_MAX_SCALE_RANGE = 0.3                   # Scale in [1-range, 1+range] = [0.7, 1.3]

ADAPTER_MAX_SHIFT_INIT = 0.01                   # Initial max shift magnitude (learnable)



# Training -- short cycles, tiny network, no EMA needed

ADAPTER_LR = 1e-3                               # Higher LR for tiny network

ADAPTER_WEIGHT_DECAY = 1e-3                     # Light regularization

ADAPTER_EPOCHS = 30                             # Short training cycles

ADAPTER_STEPS_PER_EPOCH = 500                   # Steps per adapter epoch

ADAPTER_BATCH_SIZE = 64                         # Larger batch (no base model gradients)

ADAPTER_VAL_EVERY = 5                           # Validate every N epochs

ADAPTER_GRAD_CLIP = 1.0                         # Gradient clipping



# Rolling window -- recent data for adapter training

ADAPTER_WINDOW_BARS = 500_000                   # ~2 weeks of recent bars per asset

ADAPTER_CONTEXT_LOOKBACK = 2000                 # Bars for context vector computation

ADAPTER_REPLAY_FRACTION = 0.2                   # 20% replay buffer samples in training



# Drift monitor -- tracks base model performance degradation

DRIFT_WINDOW_SIZE = 5000                        # Rolling IC window (bars)

DRIFT_WARN_RATIO = 0.5                          # Warn if IC < 50% of baseline

DRIFT_RETRAIN_RATIO = 0.3                       # Retrain if IC < 30% of baseline



# Replay buffer -- regime-balanced sample storage

REPLAY_BUFFER_SIZE = 10000                      # Max samples (balanced across 3 regimes)





# =============================================================================

# V8.E MULTI-SEED ENSEMBLE (Independent Seeds, Full Vanilla Training Each)

# =============================================================================



ENSEMBLE_N_SEEDS = 5                              # Number of independent training runs (K)

ENSEMBLE_SEEDS = [42, 1337, 2024, 7777, 31415]    # Deterministic seed list

ENSEMBLE_TOP_K = 3                                 # Use top-K seeds at inference





# =============================================================================

# V8.D MULTI-HEAD NCL DIVERSITY

# =============================================================================



DIVERSITY_N_HEADS = 5              # Number of parallel return prediction heads

DIVERSITY_NCL_LAMBDA = 0.5         # NCL diversity loss weight (higher = more diverse)

DIVERSITY_HEAD_DIM = 384           # Head dimension

DIVERSITY_HEAD_DROPOUT = 0.05      # Per-head dropout

DIVERSITY_LR = 1e-4                # Same as base model

DIVERSITY_WEIGHT_DECAY = 5e-2

DIVERSITY_TOTAL_EPOCHS = 200

DIVERSITY_STEPS_PER_EPOCH = 2000    # Cross-version invariant (must be 2000)





# =============================================================================

# LR SCHEDULE

# =============================================================================



def get_lr_for_epoch(epoch: int) -> float:

    if epoch < WM_WARMUP_EPOCHS:

        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * ((epoch + 1) / WM_WARMUP_EPOCHS)

    progress = (epoch - WM_WARMUP_EPOCHS) / max(1, WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS)

    return WM_MIN_LR + 0.5 * (WM_LR - WM_MIN_LR) * (1 + math.cos(math.pi * progress))





# =============================================================================

# ANTI-TEMPORAL-MEMORIZATION (ATME)

# =============================================================================

# Forces cross-sectional signal learning over temporal memorization.

# TEMPORAL_CTX_DROP: Obs-only posterior for return/regime heads N% of training steps.

# SEQ_SHUFFLE_PROB: Shuffles time axis of training sequences (architecture-agnostic).



TEMPORAL_CTX_DROP = 0.40            # Prob of obs-only posterior in return/regime heads

SEQ_SHUFFLE_PROB = 0.30             # Prob of shuffling a training sequence's time axis




# Cross-version invariant (CLAUDE.md): target source for all training.
# Added 2026-05-16 — was missing per wm_deep_audit framework finding.
target_prefix = "target_return"  # raw returns (voladj DEPRECATED)


# Cross-version invariant drift gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v8_3")
except ImportError:
    pass  # smoke test mode
