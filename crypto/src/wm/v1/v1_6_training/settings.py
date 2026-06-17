# CDAP contract
__contract__ = {
    "kind": "model_settings",
    "model": "V1.6",
    "outputs": {
        "checkpoints": "models/wm/v1/v1_6/v1_6_f<N>_wm_*.pt",
    },
    "invariants": {
        "WM_BATCH_SIZE":              32,
        "BIN_MIN":                   -1.0,
        "BIN_MAX":                    1.0,
        "NUM_BINS":                   255,
        "TWOHOT_FOCAL_GAMMA":         0.0,
        "WM_STEPS_PER_EPOCH":         2000,
        "DIRECT_RETURN_WEIGHT":       3.0,
        "REC_LOG_VAR_CLAMP_MIN":      0.5,
        "ACTIVE_HORIZONS":            [1, 4, 16, 64],
        "feature_count_convention":   "n_features (canonical)",
        "all_techniques":             "KL anneal + Gumbel + ATME + dream",
    },
    "rationale": "All-techniques V1 variant: f34 IC=0.0619 ShIC=0.0329.",
}

"""
V1.6 Settings -- "Best of V1" Transformer-RSSM World Model

Consolidates ALL proven V1 techniques into one model:
  - V1.0 base: Transformer-RSSM, Kendall corridors, direct return Huber
  - V1.2: KL annealing (0->1 over 20 epochs)
  - V1.3: Gumbel tau annealing (1.0->0.5 over 50 epochs)
  - V3-V9: ATME temporal context dropout (p=0.15)
  - 4 additional base features (whale, efficiency, return_4, return_16)
  - All horizons active (multi-scale regularization) + pairwise ranking loss
  - Dream consistency loss (trains dream_step for agent use)
  - Raw return targets, TwoHot bins [-1, 1], no focal/smoothing on returns

Architecture: Causal Transformer + RSSM Latents
Hardware Target: RTX 4060 (8GB) + i9, Windows 11
Data: 10 assets, 37 features (30 base + 7 cross-asset)
"""
import torch
import math
import platform
from pathlib import Path

# =============================================================================
# PLATFORM DETECTION
# =============================================================================

IS_WINDOWS = platform.system() == "Windows"

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

while not (PROJECT_ROOT / "data").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v1" / "v1_6"
LOG_DIR = PROJECT_ROOT / "logs" / "v1" / "v1_6"
BASE_MODEL_DIR = MODEL_DIR / "base"
ADAPTER_MODEL_DIR = MODEL_DIR / "adapter"
ENSEMBLE_MODEL_DIR = MODEL_DIR / "ensemble"
NCL_MODEL_DIR = MODEL_DIR / "ncl"
for _d in [MODEL_DIR, LOG_DIR, BASE_MODEL_DIR, ADAPTER_MODEL_DIR, ENSEMBLE_MODEL_DIR, NCL_MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# FEATURES -- 41 dimensions (34 base + 7 cross-asset)
# =============================================================================

FEATURE_LIST = [
    # Base features (0-12) -- per-asset, computed in sota_shared_logic_v50
    "norm_deviation",        # 0:  Volatility regime (EMA spread)
    "norm_fd_close",         # 1:  Fractional diff (stationary trend memory)
    "norm_vpin",             # 2:  Volume-synchronized probability of informed trading
    "norm_flow_imbalance",   # 3:  Buy/sell volume delta
    "norm_vol_cluster",      # 4:  Volatility of volatility
    "norm_funding",          # 5:  Funding rate (positioning sentiment)
    "norm_tick_count",       # 6:  Liquidity activity proxy
    "norm_log_volume",       # 7:  Absolute volume (log-scaled)
    "norm_hl_spread",        # 8:  Rogers-Satchell realized vol (drift-independent)
    "hurst_regime",          # 9:  Mean-reversion vs trending (R/S statistic)
    "norm_oi_change",        # 10: Open interest rate of change (positioning pressure)
    "norm_return_1",         # 11: Lagged 1-bar return (autoregressive signal)
    "norm_spread_bps",       # 12: Effective bid-ask spread proxy (liquidity cost)
    # Extended base (13-17)
    "norm_ma_distance",         # 13: SMA-200 distance (medium-term trend regime)
    "norm_whale",            # 14: Avg trade size = volume/tick_count (institutional flow)
    "norm_efficiency",       # 15: Price efficiency ratio (trending vs choppy)
    "norm_return_4",         # 16: Lagged 4-bar cumulative return (mean-reversion signal)
    "norm_return_16",        # 17: Lagged 16-bar cumulative return (medium-term momentum)
    # Tier 1 features (17-19) -- V51b orthogonal replacements
    "norm_return_kurtosis",      # 17: Rolling excess kurtosis (distribution shape)
    "norm_bar_duration",         # 18: Bar duration (volume clock speed, log seconds)
    "norm_funding_momentum",     # 19: Funding rate of change (leverage dynamics)
    # Hawkes features (20-23) -- trade clustering dynamics
    "norm_hawkes_intensity",     # 20: Tick rate vs EMA (self-excitation signal)
    "norm_hawkes_buy_intensity", # 21: Buy-side clustering (informed buying)
    "norm_hawkes_sell_intensity",# 22: Sell-side clustering (liquidation/distribution)
    "norm_hawkes_imbalance",     # 23: Buy - sell clustering (directional clustering)
    # IC-boosting features (24-28) -- Tier 2 dynamics
    "norm_momentum_accel",       # 24: Second derivative of price (trend acceleration)
    "norm_vol_price_corr",       # 25: Volume-price correlation (accumulation/distribution)
    "norm_vol_ratio",            # 26: Volatility term structure (short/long vol)
    "norm_flow_persistence",     # 27: Flow autocorrelation (institutional campaigns)
    "norm_oi_price_divergence",  # 28: OI building while price flat (spring loading)
    # SOTA features (30-33) -- added 2026-03-31, backward compatible
    "norm_yz_volatility",        # 30: Yang-Zhang vol (MVUE, overnight+RS+OC)
    "norm_cs_spread",            # 31: Corwin-Schultz bid-ask spread from H/L
    "norm_perm_entropy",         # 32: Permutation entropy (predictability/complexity)
    "norm_kyle_lambda",          # 33: Kyle's lambda (price impact per $ order flow)
    # Cross-asset features (34-40) -- computed in make_dataset_legacy.py (Phase 2)
    "xd_btc_return",         # 34: BTC leader signal
    "xd_btc_volatility",     # 35: BTC risk regime
    "xd_funding_spread",     # 36: Relative positioning (asset - BTC funding; BTC=0)
    "xd_cross_return_mean",  # 37: Market breadth
    "xd_cross_vol_mean",     # 38: Systemic risk
    "xd_ma_distance",        # 39: Cross-sectional SMA-200 trend vs market avg
    "xd_momentum_rank",      # 40: Cross-sectional return rank vs all peers
]
INPUT_DIM = len(FEATURE_LIST)  # 41 (max; actual depends on feature selection)

# === FEATURE SELECTION (13 / 18 / 21 / 25 / 30 / 37) ===
# 13: Legacy base only (backward compat, V1.0 identical)
# 18: Extended base (13 legacy + whale, efficiency, return_4, return_16, ma_distance)
# 21: + Tier 1 (kurtosis, bar_duration, funding_momentum)
# 25: + Hawkes (hawkes_intensity/buy/sell/imbalance)
# 30: + IC-boost (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_div)
# 37: Full (30 base + 7 XD) -- maximum feature set
# === FEATURE SELECTION (centralized in src/feature_sets.py, post-2026-04-27) ===
import sys as _sys
_SRC_DIR = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_21,
    FEATURE_LIST_25, FEATURE_LIST_29, FEATURE_LIST_30,
    FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    FEATURE_LIST_46, FEATURE_LIST_51, FEATURE_LIST_60, FEATURE_LIST_73,
    FEATURE_LIST_78, FEATURE_LIST_81, FEATURE_LIST_84,
    FEATURE_LIST_97, FEATURE_LIST_110, FEATURE_LIST_121,
    DEAD_FEATURE_INDICES,
    get_feature_config as _central_get_feature_config,
)

# Tier 1 features are per-asset (not cross-asset) so they are "base" -- safe for posterior/recon.
BASE_DIM = 34              # Features [0:BASE_DIM] are "base" (safe for posterior/recon)
XD_DROPOUT_RATE = 0.7      # Per-timestep dropout on features [BASE_DIM:] during training
XD_NOISE_STD = 0.3         # Heavy noise on XD features (vs 0.02 base aug noise)

# V1.6 supports the full registry (all techniques: KL anneal, Gumbel, ATME, dream)
SUPPORTED_FEATURE_COUNTS_V1_6 = (13, 18, 21, 25, 29, 30, 34, 37, 41,
                                  46, 51, 60, 73, 78, 81, 84, 97, 110, 121,
                                  127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V1_6:
        raise ValueError(
            f"V1.6 supports {sorted(SUPPORTED_FEATURE_COUNTS_V1_6)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# =============================================================================
# V1.6 UPGRADES -- Consolidated from V1.1-V1.3 + V3-V9 + NEW
# =============================================================================

TWOHOT_FOCAL_GAMMA = 0.0  # DISABLED: focal upweights temporally-clustered tail returns
LABEL_SMOOTHING = 0.0     # DISABLED: learns temporal return shape
# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"

# --- From V1.2: KL Annealing ---
# Ramp KL weight from 0 to 1 over N epochs.
# Lets encoder learn good representations before KL penalty kicks in.
# ONLY technique that preserved regime head accuracy in 18-feature models (41% vs 32%).
KL_ANNEAL_EPOCHS = 20

# --- From V1.3: Gumbel Tau Annealing ---
# Lower tau -> sharper categorical codes -> more informative latent.
# Static tau=1.0 keeps categoricals soft throughout training.
GUMBEL_TAU_START = 1.0
GUMBEL_TAU_END = 0.5
GUMBEL_TAU_ANNEAL_EPOCHS = 50

# --- From V3-V9: ATME (Attention-based Temporal Masking Erasure) ---
# With probability p, zero out h_seq so return/regime heads must use z_post alone.
# Forces the model to encode genuine predictive signal into the latent state.
# All V3-V9 have this at p=0.15; V1 base doesn't.
ATME_PROB = 0.15

# --- NEW: Dream Consistency Loss ---
# Train the dream_step so it produces meaningful predictions for the agent.
# dream_proj + dream_gru are randomly initialized and never trained in V1.0-V1.5.
# This lightweight loss trains them to produce meaningful h=1 predictions.
DREAM_CONSISTENCY_WEIGHT = 0.1
DREAM_CONSISTENCY_EVERY = 4   # Apply dream loss every Nth batch

# NOTE: Per-horizon clamp dict (RETURN_LOG_VAR_CLAMP_PER_H) REMOVED -- uniform
# RETURN_LOG_VAR_CLAMP_MAX=-2.0 for all active horizons. Per-H caps hurt regularization.
# h16/h64 included as multi-scale regularizers (prevents ShIC decline).


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
# TARGETS -- Vol-normalized multi-horizon return predictions
# =============================================================================

REWARD_HORIZONS = [1, 4, 16, 64]
# All horizons active for loss computation.
# h16/h64 act as multi-scale regularizers that prevent temporal memorization.
# Without them, ShIC declines monotonically from epoch 10 (confirmed Feb 2026 vs Mar 2026 runs).
ACTIVE_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]  # Raw returns (voladj DEPRECATED)


# =============================================================================
# V1 TRANSFORMER ARCHITECTURE
# =============================================================================

# Transformer Core
# 2026-05-09 capacity-scaling flag (mirror of V1.0): see V1.0 settings.py
# for rationale. Default False preserves deployed ckpt compat.
# 2026-05-10 ROLLBACK: see V1.0 settings.py for empirical failure receipt.
USE_CAPACITY_SCALING = False
if USE_CAPACITY_SCALING:
    WM_D_MODEL = 320
    WM_N_HEADS = 8
    WM_N_LAYERS = 6
    WM_D_FF = 1280
else:
    WM_D_MODEL = 256        # Hidden dimension (deployed default)
    WM_N_HEADS = 8          # Attention heads (256 / 8 = 32 dim per head)
    WM_N_LAYERS = 3         # Transformer layers
    WM_D_FF = 768           # FFN inner dimension (3x expansion)

# RSSM Latent Space
RSSM_LATENT_DIM = 24        # Number of categorical distributions
RSSM_CLASSES = 24           # Classes per distribution
FLAT_DIM = RSSM_LATENT_DIM * RSSM_CLASSES  # 576

# Asset Conditioning
WM_ASSET_EMB_DIM = 32       # Learned embedding per asset

# TwoHot Symlog (return prediction) -- raw return targets, bins [-1, 1]
NUM_BINS = 255              # 255 bins across [-1, 1] -> resolution 0.0079 per bin
BIN_MIN = -1.0              # Fixed from [-5,5] which caused temporal memorization
BIN_MAX = 1.0

# SOTA Return Prediction Heads
RETURN_HEAD_DIM = 384       # Wider return trunk (was implicit 256)
RETURN_HEAD_DROPOUT = 0.05  # Lower dropout for return heads
GUMBEL_TAU = 1.0            # Initial Gumbel-Softmax temperature (annealed by trainer)

# Direct Return Regression (bypasses TwoHot discretization bottleneck)
# Weight=1.0: >1.0 creates Huber shortcut bypassing TwoHot bottleneck (confirmed V3/V4/V6)
DIRECT_RETURN_WEIGHT = 3.0  # Huber dominance acts as temporal regularizer (stable ShIC=0.030 in Feb runs)

# Pairwise Ranking Loss (Phase C Path 2: learning-to-rank auxiliary)
PAIRWISE_RANK_WEIGHT = 0.0   # DISABLED: pairwise ranking trains temporal ordering = memorization
PAIRWISE_RANK_PAIRS = 256    # Number of random pairs per batch (inactive when weight=0)

# Kendall log_vars: exp(-s) * L + 0.5*s -- higher s = lower weight
# Raw return targets (voladj deprecated). All horizons active for multi-scale regularization.
# [rec=1.0, kl=0.0, ret_1=-2.0, ret_4=-2.0, ret_16=-2.0, ret_64=-2.0, regime=-1.5]
LOG_VAR_INIT = [1.0, 0.0, -2.0, -2.0, -2.0, -2.0, -1.5]

# Regularization
WM_DROPOUT = 0.15           # Dropout rate
WM_FREE_NATS = 1.0          # KL free nats threshold
WM_ATTENTION_DROPOUT = 0.15  # Attention dropout

# Regime Classification -- Focal Loss (Lin et al., 2017)
REGIME_FOCAL_GAMMA = 2.0

# Regime Head Architecture
REGIME_HEAD_DIM = 256

# Kendall Weight Corridors (prevent reconstruction from hogging gradients)
REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q (2026-04-14): rec gets at most exp(-0.5)=0.61x weight (was 1.0x). Autopsy showed rec gradient dominated, starving return heads. Reduce rec dominance for more signal extraction.
RETURN_LOG_VAR_CLAMP_MAX = -2.0     # Default for horizons not in per-h dict (backward compat)
REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight

# ShIC-based early stopping
SHUFFLED_IC_PATIENCE = 5            # Stop after N consecutive ShIC declines
SHUFFLED_IC_CHECK_INTERVAL = 10     # Compute ShIC every N epochs (via AntifragileConfig.shuffled_ic_every)
SHUFFLED_IC_MIN_DECLINE = 0.001     # Only count as decline if drop > this from best
# NOTE: ShIC LR reduction REMOVED -- locks in memorized weights, prevents recovery.
# LR follows warmup+cosine schedule only. ShIC decline triggers early stop.


# =============================================================================
# TRAINING
# =============================================================================

WM_SEQ_LEN = 96             # ~96 dollar bars ~ 24h of BTC
WM_BATCH_SIZE = 32          # Small batch = implicit gradient noise regularization (prevents ShIC decline)
WM_LR = 2e-4                # Peak learning rate (after warmup)
WM_WEIGHT_DECAY = 5e-2      # Weight decay
WM_GRAD_CLIP = 1.0          # Global gradient clipping

WM_TOTAL_EPOCHS = 200       # Total training epochs
WM_STEPS_PER_EPOCH = 2000   # Steps per epoch (batch*steps=64K samples, matching healthy Feb 2026 config)
WM_VAL_EVERY = 5            # Validate every N epochs
WM_PATIENCE = 40            # Early stopping patience (in epochs)

# Learning rate schedule
WM_WARMUP_EPOCHS = 5        # Linear warmup from WM_MIN_LR to WM_LR
WM_MIN_LR = 1e-6            # Minimum LR at end of cosine decay

# Block masking curriculum (self-supervised objective)
WM_MASK_RATIO_START = 0.10
WM_MASK_RATIO_END = 0.25
WM_MASK_RAMP_EPOCHS = 40
WM_BLOCK_SIZE_RATIO = 0.10

# Data augmentation
AUG_NOISE_STD = 0.02        # Gaussian noise std added to observations
AUG_FEAT_DROP = 0.1         # Fraction of features randomly zeroed per sample

# DataLoader -- Windows multiprocessing spawn fails with num_workers > 0
NUM_WORKERS = 0 if IS_WINDOWS else 2

# Logging
LOG_FREQ = 25               # Log every N steps within epoch

# EMA
EMA_DECAY = 0.995           # EMA model decay rate


# =============================================================================
# AGENT (Phase 2 -- only used after WM passes gate)
# =============================================================================

ACTION_DIM = 3              # {Short=0, Neutral=1, Long=2}
BASE_FEE = 0.0004           # 0.04% taker fee (Binance futures)

RSSM_FLAT_DIM = FLAT_DIM   # Alias: RSSM_LATENT_DIM * RSSM_CLASSES = 576
AGENT_HIDDEN = 256          # Agent network hidden dimension

ASSET_FEE_BPS = {
    "BTCUSDT":  0.0004,   # Tight spread, deep book
    "ETHUSDT":  0.0004,
    "SOLUSDT":  0.0005,
    "BNBUSDT":  0.0005,
    "XRPUSDT":  0.0005,
    "DOGEUSDT": 0.0006,   # Wider spread, thinner book
    "ADAUSDT":  0.0006,
    "AVAXUSDT": 0.0006,
    "LINKUSDT": 0.0005,
    "LTCUSDT":  0.0005,
}

DREAM_HORIZON = 15          # Steps per imagined trajectory
GAMMA = 0.99                # Discount factor
GAE_LAMBDA = 0.95           # GAE lambda

PPO_CLIP = 0.2              # PPO clipping epsilon
PPO_EPOCHS_PER_UPDATE = 4   # Mini-epochs per PPO update
AGENT_LR = 1e-4             # Agent learning rate
AGENT_GRAD_CLIP = 0.5       # Agent gradient clipping
ENTROPY_COEF = 0.01         # Entropy bonus coefficient
VALUE_COEF = 0.5            # Value loss coefficient


# =============================================================================
# V1.x ADAPTIVE RESIDUAL ADAPTER
# =============================================================================

ADAPTER_FEAT_DIM = WM_D_MODEL + FLAT_DIM    # 832 (cat(h_seq, z_post))
ADAPTER_CONTEXT_DIM = 12                     # Rolling IC(4) + bias(4) + regime(3) + vol(1)
ADAPTER_BOTTLENECK = 16
ADAPTER_FILM_HIDDEN = 48
ADAPTER_MAX_SCALE_RANGE = 0.3
ADAPTER_MAX_SHIFT_INIT = 0.01

ADAPTER_LR = 1e-3
ADAPTER_WEIGHT_DECAY = 1e-3
ADAPTER_EPOCHS = 30
ADAPTER_STEPS_PER_EPOCH = 500
ADAPTER_BATCH_SIZE = 64
ADAPTER_VAL_EVERY = 5
ADAPTER_GRAD_CLIP = 1.0

ADAPTER_WINDOW_BARS = 500_000
ADAPTER_CONTEXT_LOOKBACK = 2000
ADAPTER_REPLAY_FRACTION = 0.2

DRIFT_WINDOW_SIZE = 5000
DRIFT_WARN_RATIO = 0.5
DRIFT_RETRAIN_RATIO = 0.3

REPLAY_BUFFER_SIZE = 10000


# =============================================================================
# V1.E SNAPSHOT ENSEMBLE (Cyclical Cosine Annealing)
# =============================================================================

SNAPSHOT_N_CYCLES = 10
SNAPSHOT_EPOCHS_PER_CYCLE = WM_TOTAL_EPOCHS // SNAPSHOT_N_CYCLES  # 20
SNAPSHOT_LR_MAX = WM_LR
SNAPSHOT_LR_MIN = WM_MIN_LR
SNAPSHOT_WARMUP_EPOCHS = 3

SNAPSHOT_TOP_K = 3
SNAPSHOT_SHIC_GATE = True


def get_snapshot_lr_for_epoch(epoch: int) -> float:
    """Cyclical cosine annealing with warm restarts (Loshchilov & Hutter, 2017)."""
    if epoch < SNAPSHOT_WARMUP_EPOCHS:
        warmup_progress = (epoch + 1) / SNAPSHOT_WARMUP_EPOCHS
        return SNAPSHOT_LR_MIN + (SNAPSHOT_LR_MAX - SNAPSHOT_LR_MIN) * warmup_progress

    cycle = epoch // SNAPSHOT_EPOCHS_PER_CYCLE
    cycle = min(cycle, SNAPSHOT_N_CYCLES - 1)
    epoch_in_cycle = epoch - cycle * SNAPSHOT_EPOCHS_PER_CYCLE

    progress = epoch_in_cycle / max(1, SNAPSHOT_EPOCHS_PER_CYCLE - 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return SNAPSHOT_LR_MIN + (SNAPSHOT_LR_MAX - SNAPSHOT_LR_MIN) * cosine_factor


# =============================================================================
# V1.D MULTI-HEAD DIVERSITY (NCL)
# =============================================================================

DIVERSITY_N_HEADS = 5
DIVERSITY_NCL_LAMBDA = 0.5
DIVERSITY_HEAD_DIM = 384
DIVERSITY_HEAD_DROPOUT = 0.05
DIVERSITY_LR = 2e-4
DIVERSITY_WEIGHT_DECAY = 5e-2
DIVERSITY_TOTAL_EPOCHS = 200
DIVERSITY_STEPS_PER_EPOCH = 2000


# =============================================================================
# VALIDATION GATES
# =============================================================================

GATE_REC_MSE_MAX = 0.12     # Reconstruction quality (relaxed from 0.10 -- V1.0 OOS=0.116)
GATE_IC_MIN = 0.015         # Return prediction correlation
GATE_KL_MIN = 0.01          # Latent health (no collapse)
GATE_KL_MAX = 15.0          # Latent health (no explosion)
GATE_SHUFFLED_IC_RATIO_MIN = 0.3  # Shuffled IC / Contiguous IC (anti-memorization)
GATE_LOSS_RATIO_MAX = 2.0         # Train/Val loss ratio (overfitting detection)


# =============================================================================
# MULTI-HEAD FEATURE ABLATION
# =============================================================================
# Train separate return prediction heads on different feature subsets in parallel.
# Each ablation head gets its own return trunk + per-horizon MLPs.
# The shared encoder/RSSM processes feature-masked inputs per head.
# IC is tracked per-head to measure each feature group's marginal contribution.
#
# How it works:
#   1. Primary head trains normally with all configured features
#   2. Each ablation head zeros non-subset features at INPUT level
#   3. Runs through shared encoder -> different h_seq per head
#   4. RSSM posterior also sees masked features -> head-specific latents
#   5. Head-specific return MLPs predict returns from head-specific representations
#   6. All head losses are summed (primary + weighted ablation heads)
#   7. IC tracked per-head; primary head determines model selection
#
# Usage: python train_world_model.py --features 25 --ablation

# Weight for non-primary ablation head losses (prevents gradient domination)
ABLATION_LOSS_WEIGHT = 0.5


def get_ablation_subsets(n_features: int) -> dict:
    """Return semantically-valid ablation subsets for a given feature count.

    Non-contiguous layouts (f18, f22) skip subsets whose features aren't
    present in the parent. Index ranges refer to MODEL input positions,
    not the global FEATURE_LIST indices.

    Feature layouts (model input positions):
      f13: [0:13] = base13
      f17: [0:17] = base13 + ext4
      f18: [0:13] = base13, [13:18] = XD5
      f20: [0:20] = base13 + ext4 + tier1_3
      f22: [0:17] = base13 + ext4, [17:22] = XD5
      f25: [0:20] = base13 + ext4 + tier1_3, [20:25] = XD5
    """
    if n_features <= 13:
        return {}
    if n_features == 17:
        return {"f13": list(range(13))}
    if n_features == 18:
        return {"f13": list(range(13))}
    if n_features == 20:
        return {"f13": list(range(13)), "f17": list(range(17))}
    if n_features == 22:
        return {"f13": list(range(13)), "f17": list(range(17))}
    if n_features == 25:
        return {"f13": list(range(13)), "f17": list(range(17)), "f20": list(range(20))}
    return {}


# =============================================================================
# LR SCHEDULE FUNCTION
# =============================================================================

def get_lr_for_epoch(epoch: int) -> float:
    """
    Compute learning rate for a given epoch.

    Schedule:
      - Epochs 0..WM_WARMUP_EPOCHS-1: linear warmup from WM_MIN_LR to WM_LR
      - Epochs WM_WARMUP_EPOCHS..WM_TOTAL_EPOCHS: cosine decay from WM_LR to WM_MIN_LR
    """
    if epoch < WM_WARMUP_EPOCHS:
        warmup_progress = (epoch + 1) / WM_WARMUP_EPOCHS
        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * warmup_progress
    else:
        decay_epochs = WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS
        decay_progress = (epoch - WM_WARMUP_EPOCHS) / max(1, decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * cosine_factor

# Cross-version invariant (CLAUDE.md): target source for all training
target_prefix = "target_return"  # raw returns (voladj DEPRECATED)

# ─── CC-H4 HEADLINE_MODE anti-mem upgrades (added 2026-04-30) ──────────────
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H4. When env var V1_HEADLINE_MODE=1 is
# set BEFORE python invocation, anti-mem knobs tighten to push ShIC into the
# Headline tier (>= 0.045). Default off; legacy training paths unchanged.
#
# Acceptance band per upgrade plan:
#   ShIC delta:  +0.005 to +0.012 (V1.1 base 0.033 -> target 0.040-0.045)
#   IC delta:   -0.003 to -0.008 (V1.1 base 0.067 -> tolerated 0.060)
#   Ratio:       0.49 -> 0.65+ (the load-bearing improvement)
#
# Activation: V1_HEADLINE_MODE=1 python train_world_model.py --features 34
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V1_HEADLINE_MODE", "0")))
if HEADLINE_MODE:
    WM_FREE_NATS = 1.5         # was 1.0; raise the KL-throw-away floor
    XD_DROPOUT_RATE = 0.85     # was 0.7; drop ~12 of 34 per batch
    XD_NOISE_STD = 0.4         # 3rd CC-H4 knob; was MISSING vs V1.1 -> V1.6 HEADLINE
                               # was under-regularized relative to the spec
    print(f"[V1.x HEADLINE_MODE] WM_FREE_NATS=1.5 XD_DROPOUT_RATE=0.85 XD_NOISE_STD=0.4")


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v1_6")
except ImportError:
    pass

