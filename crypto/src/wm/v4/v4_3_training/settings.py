"""

V4.3 Settings -- Mamba Narrow Expand Variant (expand=1)



Ablation: Halved expansion factor (1 vs 2). d_inner = 384 instead of 768.

Tests whether the SSM needs the full expanded inner dimension or if

a more compact representation suffices.



Hardware Target: RTX 4060 (8GB) + i9, Windows 11

"""

import torch

import math

import platform

from pathlib import Path

import sys



# ═══════════════════════════════════════════════════════════════════════════════

# INFRASTRUCTURE

# ═══════════════════════════════════════════════════════════════════════════════



SCRIPT_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = SCRIPT_DIR



# Walk up to find the project root (contains 'data/' directory)

while not (PROJECT_ROOT / "data").exists():

    if PROJECT_ROOT.parent == PROJECT_ROOT:

        # Fallback: assume relative to script

        PROJECT_ROOT = SCRIPT_DIR.parent.parent

        break

    PROJECT_ROOT = PROJECT_ROOT.parent



DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)

MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v4" / "v4_3"

LOG_DIR = PROJECT_ROOT / "logs" / "v4" / "v4_3"

BASE_MODEL_DIR = MODEL_DIR / "base"

ADAPTER_MODEL_DIR = MODEL_DIR / "adapter"

ENSEMBLE_MODEL_DIR = MODEL_DIR / "ensemble"

NCL_MODEL_DIR = MODEL_DIR / "ncl"

for _d in [MODEL_DIR, LOG_DIR, BASE_MODEL_DIR, ADAPTER_MODEL_DIR, ENSEMBLE_MODEL_DIR, NCL_MODEL_DIR]:

    _d.mkdir(parents=True, exist_ok=True)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IS_WINDOWS = platform.system() == "Windows"





# ═══════════════════════════════════════════════════════════════════════════════

# FEATURES — 22 dimensions (the world model's sensory input)

# ═══════════════════════════════════════════════════════════════════════════════



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





# ═══════════════════════════════════════════════════════════════════════════════

# ASSETS

# ═══════════════════════════════════════════════════════════════════════════════



ASSET_LIST = [

    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",

    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",

]

NUM_ASSETS = len(ASSET_LIST)

ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}





# ═══════════════════════════════════════════════════════════════════════════════

# TARGETS — Multi-horizon return predictions

# ═══════════════════════════════════════════════════════════════════════════════



# The world model predicts returns at multiple horizons simultaneously.

# This prevents myopia (overfitting to t+1 noise) and acts as a regularizer:

# a model that can predict both t+1 and t+64 must learn genuine structure.

REWARD_HORIZONS = [1, 4, 16, 64]

# All horizons active for loss computation.

# h16/h64 act as multi-scale regularizers that prevent temporal memorization.

# Without them, ShIC declines monotonically from epoch 10 (confirmed Feb 2026 vs Mar 2026 runs).

ACTIVE_HORIZONS = [1, 4, 16, 64]

TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]

RAW_TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]

# Pipeline also generates these for agent reward shaping:

#   target_return_50  (risk-adjusted, for strategic objectives)

#   target_vol_20     (volatility target, for risk management)





# ═══════════════════════════════════════════════════════════════════════════════

# WORLD MODEL ARCHITECTURE

# ═══════════════════════════════════════════════════════════════════════════════



# Mamba SSM Core

WM_D_MODEL = 384          # Hidden dimension (sweet spot for 4060 8GB)

WM_D_STATE = 16           # SSM state dimension

WM_N_LAYERS = 2           # Stacked Mamba blocks

WM_EXPAND = 1  # V4.3: narrow expand (was 2)             # Expansion factor for inner dim



# RSSM Latent Space

RSSM_LATENT_DIM = 24      # Number of categorical distributions (was 32; reduced for anti-memorization)

RSSM_CLASSES = 24          # Classes per distribution (was 32; match V1.0: 576D flat)

FLAT_DIM = RSSM_LATENT_DIM * RSSM_CLASSES  # 576



# Asset Conditioning

WM_ASSET_EMB_DIM = 32     # Learned embedding per asset



# TwoHot Symlog (return prediction) -- vol-adjusted targets with wide range

NUM_BINS = 255              # 255 bins across [-1, 1] for raw return targets

BIN_MIN = -1.0              # Vol-adjusted targets: symlog(return/vol), range ~[-3, 3]

BIN_MAX = 1.0               # 255 bins in [-1, 1] -> resolution 0.0078 per bin



# SOTA Return Prediction Heads

RETURN_HEAD_DIM = 384     # Wider return trunk (was implicit 256)

RETURN_HEAD_DROPOUT = 0.05  # Lower dropout for return heads

GUMBEL_TAU_START = 1.0       # Initial temperature (soft categorical, more exploration)

GUMBEL_TAU_END = 0.5         # Final temperature (sharper categorical)

GUMBEL_TAU_ANNEAL_EPOCHS = 50

GUMBEL_TAU = GUMBEL_TAU_END  # Default for inference (use final/sharp value)



# Direct Return Regression (bypasses TwoHot discretization bottleneck)

DIRECT_RETURN_WEIGHT = 3.0  # Huber dominance acts as temporal regularizer (stable ShIC=0.030 in Feb runs)



# Loss initialization: FORCE return learning over reconstruction

# [rec=1.0, kl=0.0, ret_1=-2.0, ret_4=-2.0, ret_16=-2.0, ret_64=-2.0, regime=-1.5]

LOG_VAR_INIT = [1.0, 0.0, -2.0, -2.0, -2.0, -2.0, -1.5]



# Kendall Weight Corridors (prevent reconstruction from hogging gradients)

# Root cause: Kendall optimal s_i = log(2*L_i). Easy tasks (rec) drift to high weight,

# hard tasks (returns) drift to low weight. Clamp asymmetrically to maintain return priority.

REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q: rec gets at most exp(-0.5)=0.61x weight (canonical per CLAUDE.md)

RETURN_LOG_VAR_CLAMP_MAX = -2.0

# RETURN_LOG_VAR_CLAMP_PER_H = {1: -2.0, 4: -2.0, 16: -1.0, 64: -1.0}  # h16/h64 reduced (memorize OOS)  # DISABLED: uniform clamp via RETURN_LOG_VAR_CLAMP_MAX

REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight

REGIME_FOCAL_GAMMA = 2.0



# TwoHot Focal Loss -- DISABLED (accelerates memorization). Plain compute_loss only.

# TWOHOT_FOCAL_GAMMA = 1.0  # REMOVED: focal upweights temporally-clustered tail returns

# Cross-version invariant: TwoHot focal gamma is DISABLED (focal upweights
# temporally-clustered tail returns and accelerates memorization). All
# variants inherit base model's TwoHot head; this constant exists for
# `from settings import *` callers.
TWOHOT_FOCAL_GAMMA = 0.0
# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"

REGIME_HEAD_DIM = 256               # Regime head hidden dim (was 128 — capacity imbalance)



# ShIC-based early stopping (stop when memorization sets in)

SHUFFLED_IC_PATIENCE = 5            # Stop after N consecutive ShIC declines

SHUFFLED_IC_MIN_DECLINE = 0.001     # Only count as decline if drop > this from best

# SHIC_LR_REDUCE_AFTER = 3      # ATME fights first, LR reduces as fallback  # DISABLED: ShIC LR reduction removed

# SHIC_LR_REDUCE_FACTOR = 0.5   # Multiply LR by this factor per ShIC-triggered reduction  # DISABLED: ShIC LR reduction removed



# Regularization

WM_DROPOUT = 0.15            # Match V1.0 (was 0.1)

WM_FREE_NATS = 1.0        # KL free nats threshold

KL_ANNEAL_EPOCHS = 40     # Longer warmup for Mamba (was 20)





# ═══════════════════════════════════════════════════════════════════════════════

# WORLD MODEL TRAINING

# ═══════════════════════════════════════════════════════════════════════════════



WM_SEQ_LEN = 96           # ~96 dollar bars ≈ 24h of BTC

WM_BATCH_SIZE = 32  # Small batch = implicit gradient noise regularization (prevents ShIC decline)

WM_LR = 2e-4              # AdamW learning rate (was 3e-4)

WM_WEIGHT_DECAY = 5e-2    # AdamW weight decay (match V1.0)

WM_GRAD_CLIP = 1.0        # Global gradient clipping



WM_TOTAL_EPOCHS = 100     # Max epochs (with early stopping)

WM_STEPS_PER_EPOCH = 2000   # Match V1 proven config (batch*steps samples per epoch)

WM_VAL_EVERY = 5          # Validate every N epochs

WM_PATIENCE = 30          # Early stopping patience (in epochs)



# Block masking curriculum (self-supervised objective)

WM_MASK_RATIO_START = 0.10

WM_MASK_RATIO_END = 0.25

WM_MASK_RAMP_EPOCHS = 40  # Linearly ramp from start to end

WM_BLOCK_SIZE_RATIO = 0.10  # Each masked block is ~10% of seq



# Data augmentation (small dataset needs augmentation)

AUG_NOISE_STD = 0.02      # Gaussian noise injection

AUG_FEAT_DROP = 0.1        # Feature dropout probability



# Workers: 0 on Windows (multiprocessing spawn issues with CUDA)

NUM_WORKERS = 0 if IS_WINDOWS else 2



# LR schedule

WM_WARMUP_EPOCHS = 5       # Linear warmup epochs

WM_MIN_LR = 1e-6           # Minimum LR for cosine schedule



# Logging

LOG_FREQ = 25             # Log every N steps within epoch



# EMA

EMA_DECAY = 0.995           # EMA model decay rate





# ═══════════════════════════════════════════════════════════════════════════════

# AGENT (Phase 2 — only used after WM passes gate)

# ═══════════════════════════════════════════════════════════════════════════════



ACTION_DIM = 3             # {Short=0, Neutral=1, Long=2}

BASE_FEE = 0.0004         # 0.04% taker fee (Binance futures)



# FIX(critical): agent.py requires RSSM_FLAT_DIM and AGENT_HIDDEN.

# RSSM_FLAT_DIM is derived from RSSM settings (same value as FLAT_DIM).

# AGENT_HIDDEN is the hidden dimension for the ActorCritic network.

RSSM_FLAT_DIM = FLAT_DIM  # Alias: RSSM_LATENT_DIM * RSSM_CLASSES = 576

AGENT_HIDDEN = 256         # Agent network hidden dimension



# Per-asset fee overrides (bps). Agents should look up their asset here.

# TODO: Integrate per-asset fees into compute_reward() — different assets have

# different spreads and liquidity. For now all use BASE_FEE as default.

# Values below are estimates; calibrate from actual fill data.

ASSET_FEE_BPS = {

    "BTCUSDT":  0.0004,   # Tight spread, deep book

    "ETHUSDT":  0.0004,   # Tight spread, deep book

    "SOLUSDT":  0.0005,   # Slightly wider

    "BNBUSDT":  0.0005,

    "XRPUSDT":  0.0005,

    "DOGEUSDT": 0.0006,   # Wider spread, thinner book

    "ADAUSDT":  0.0006,

    "AVAXUSDT": 0.0006,

    "LINKUSDT": 0.0005,

    "LTCUSDT":  0.0005,

}



DREAM_HORIZON = 15         # Steps per imagined trajectory

GAMMA = 0.99               # Discount factor

GAE_LAMBDA = 0.95          # GAE lambda



PPO_CLIP = 0.2             # PPO clipping epsilon

PPO_EPOCHS_PER_UPDATE = 4  # Mini-epochs per PPO update

AGENT_LR = 1e-4            # Agent learning rate

AGENT_GRAD_CLIP = 0.5      # Agent gradient clipping

ENTROPY_COEF = 0.01        # Entropy bonus coefficient

VALUE_COEF = 0.5           # Value loss coefficient





# ═══════════════════════════════════════════════════════════════════════════════

# V4.x ADAPTIVE RESIDUAL ADAPTER

# ═══════════════════════════════════════════════════════════════════════════════



# Architecture -- FiLM modulation of return trunk, ~15K params

# ADAPTER_FEAT_DIM = d_model(384) + RSSM flat(576) = 960

# This is cat(h_seq, z_post) which feeds into ret_trunk

ADAPTER_FEAT_DIM = WM_D_MODEL + FLAT_DIM       # 960 (cat(h_seq, z_post))

ADAPTER_CONTEXT_DIM = 12                       # Rolling IC(4) + bias(4) + regime(3) + vol(1)

ADAPTER_BOTTLENECK = 16                        # Compressed feat representation

ADAPTER_FILM_HIDDEN = 48                       # FiLM generator hidden dim

ADAPTER_MAX_SCALE_RANGE = 0.3                 # Scale in [1-range, 1+range] = [0.7, 1.3]

ADAPTER_MAX_SHIFT_INIT = 0.01                 # Initial max shift magnitude (learnable)



# Training -- short cycles, tiny network, no EMA needed

ADAPTER_LR = 1e-3                             # Higher LR for tiny network

ADAPTER_WEIGHT_DECAY = 1e-3                   # Light regularization

ADAPTER_EPOCHS = 30                            # Short training cycles

ADAPTER_STEPS_PER_EPOCH = 500                 # Steps per adapter epoch

ADAPTER_BATCH_SIZE = 64                       # Larger batch (no base model gradients)

ADAPTER_VAL_EVERY = 5                         # Validate every N epochs

ADAPTER_GRAD_CLIP = 1.0                       # Gradient clipping



# Rolling window -- recent data for adapter training

ADAPTER_WINDOW_BARS = 500_000                 # ~2 weeks of recent bars per asset

ADAPTER_CONTEXT_LOOKBACK = 2000               # Bars for context vector computation

ADAPTER_REPLAY_FRACTION = 0.2                # 20% replay buffer samples in training



# Drift monitor -- tracks base model performance degradation

DRIFT_WINDOW_SIZE = 5000                      # Rolling IC window (bars)

DRIFT_WARN_RATIO = 0.5                       # Warn if IC < 50% of baseline

DRIFT_RETRAIN_RATIO = 0.3                    # Retrain if IC < 30% of baseline



# Replay buffer -- regime-balanced sample storage

REPLAY_BUFFER_SIZE = 10000                    # Max samples (balanced across 3 regimes)





# ═══════════════════════════════════════════════════════════════════════════════

# V4.E MULTI-SEED ENSEMBLE (Independent Seeds, Full Vanilla Training Each)

# ═══════════════════════════════════════════════════════════════════════════════



# Each seed-run = full vanilla training (WM_TOTAL_EPOCHS, warmup + cosine LR).

# Different seeds -> different weight initializations -> different basins

# -> low inter-model correlation (rho~0.3-0.5) -> strong ensemble IC boost.



ENSEMBLE_N_SEEDS = 5                              # Number of independent training runs (K)

ENSEMBLE_SEEDS = [42, 1337, 2024, 7777, 31415]    # Deterministic seed list

ENSEMBLE_TOP_K = 3                                 # Use top-K seeds at inference





# ═══════════════════════════════════════════════════════════════════════════════

# V4.D MULTI-HEAD NCL DIVERSITY

# ═══════════════════════════════════════════════════════════════════════════════



DIVERSITY_N_HEADS = 5              # Number of parallel return prediction heads

DIVERSITY_NCL_LAMBDA = 0.5         # NCL diversity loss weight (higher = more diverse)

DIVERSITY_HEAD_DIM = 384           # Head dimension (matches V1)

DIVERSITY_HEAD_DROPOUT = 0.05      # Per-head dropout

DIVERSITY_LR = 3e-4                # Same as base model

DIVERSITY_WEIGHT_DECAY = 5e-2

DIVERSITY_TOTAL_EPOCHS = 100

DIVERSITY_STEPS_PER_EPOCH = 2000   # Same as base (WM_STEPS_PER_EPOCH)





# ═══════════════════════════════════════════════════════════════════════════════

# VALIDATION GATES

# ═══════════════════════════════════════════════════════════════════════════════



# World model must pass ALL of these before agent training begins

GATE_REC_MSE_MAX = 0.12    # Reconstruction quality

GATE_IC_MIN = 0.015        # Return prediction correlation

GATE_KL_MIN = 0.01         # Latent health (no collapse)

GATE_KL_MAX = 15.0         # Latent health (no explosion)

GATE_SHUFFLED_IC_RATIO_MIN = 0.3  # Shuffled IC / Contiguous IC (anti-memorization)

GATE_LOSS_RATIO_MAX = 2.0         # Train/Val loss ratio (overfitting detection)





# ═══════════════════════════════════════════════════════════════════════════════

# LR SCHEDULE (warmup + cosine)

# ═══════════════════════════════════════════════════════════════════════════════



def get_lr_for_epoch(epoch: int) -> float:

    """Linear warmup from WM_MIN_LR to WM_LR, then cosine decay back to WM_MIN_LR."""

    if epoch < WM_WARMUP_EPOCHS:

        warmup_progress = (epoch + 1) / WM_WARMUP_EPOCHS

        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * warmup_progress

    progress = (epoch - WM_WARMUP_EPOCHS) / max(1, WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS)

    return WM_MIN_LR + 0.5 * (WM_LR - WM_MIN_LR) * (1 + math.cos(math.pi * progress))





# ═══════════════════════════════════════════════════════════════════════════════

# ANTI-TEMPORAL-MEMORIZATION (ATME)

# ═══════════════════════════════════════════════════════════════════════════════

# Forces cross-sectional signal learning over temporal memorization.

# TEMPORAL_CTX_DROP: Obs-only posterior for return/regime heads N% of training steps.

# SEQ_SHUFFLE_PROB: Shuffles time axis of training sequences (architecture-agnostic).



TEMPORAL_CTX_DROP = 0.40            # Prob of obs-only posterior for return/regime heads (was 0.25; now that ATME actually blocks z_post leakage, stronger is better)

SEQ_SHUFFLE_PROB = 0.30             # Prob of shuffling a training sequence's time axis (was 0.20)




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
    _assert_canonical(globals(), version_name="v4_3")
except ImportError:
    pass  # smoke test mode
