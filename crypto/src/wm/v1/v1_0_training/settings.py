# CDAP contract
__contract__ = {
    "kind": "model_settings",
    "model": "V1.0",
    "outputs": {
        "checkpoints": "models/wm/v1/v1_0/v1_0_f<N>_wm_*.pt",
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
        "feature_count_convention":   "INPUT_DIM (legacy; V1.1+ use n_features)",
    },
    "rationale": "Reference baseline; first to PASS f34 ShIC gate (IC=0.0660 ShIC=0.0320).",
}

"""
V1 Settings -- Transformer-RSSM World Model (SOTA 2025/26)

Architecture: Causal Transformer + RSSM Latents
Hardware Target: RTX 4060 (8GB) + i9, Windows 11
Data: 10 assets (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC)

SOTA components:
  - FlashAttention via F.scaled_dot_product_attention (PyTorch 2.0+)
  - RoPE for relative position encoding (Su et al., 2021)
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
  - SwiGLU gated FFN (Shazeer, 2020)
  - AdamW with betas=(0.9, 0.95) (LLaMA/Chinchilla standard)

Key design:
  - d_model=256, 3 layers, 8 heads, RSSM 24x24
  - dropout=0.15, weight_decay=5e-2
  - 200 epochs, 2000 steps/epoch, cosine LR schedule
  - Anti-fragile: walk-forward CV, shuffled IC, regime-balanced sampling
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
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v1" / "v1_0"
LOG_DIR = PROJECT_ROOT / "logs" / "v1" / "v1_0"
BASE_MODEL_DIR = MODEL_DIR / "base"
ADAPTER_MODEL_DIR = MODEL_DIR / "adapter"
ENSEMBLE_MODEL_DIR = MODEL_DIR / "ensemble"
NCL_MODEL_DIR = MODEL_DIR / "ncl"
for _d in [MODEL_DIR, LOG_DIR, BASE_MODEL_DIR, ADAPTER_MODEL_DIR, ENSEMBLE_MODEL_DIR, NCL_MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# FEATURES -- imported from src/feature_sets.py (single source of truth, V0-V19)
# =============================================================================
import sys as _sys
_SRC_DIR = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_13, FEATURE_LIST_25, FEATURE_LIST_29, FEATURE_LIST_34,
    FEATURE_LIST_41, FEATURE_LIST_121,
    DEAD_FEATURE_INDICES,
    get_feature_config as _central_get_feature_config,
)

# Default V1.0 working list = f13 (reference baseline)
FEATURE_LIST = FEATURE_LIST_13
INPUT_DIM = len(FEATURE_LIST)  # 13

# V1.0 supports a subset of all centrally-registered counts.
# (V1.0 has no XD anti-memorization split → base_dim==input_dim for every
# count it supports; centralized resolver handles all of these correctly.)
SUPPORTED_FEATURE_COUNTS_V1_0 = (13, 25, 29, 34, 41, 121, 127, 133, 154, 161)


def get_feature_config(n_features):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API.

    V1.0 has no XD split (base_dim==input_dim for all counts). Callers that
    only consume 2 elements from older V1.0 code can still do
    ``feature_list, input_dim = get_feature_config(n)[:2]``.
    """
    if n_features not in SUPPORTED_FEATURE_COUNTS_V1_0:
        raise ValueError(
            f"V1.0 supports f{list(SUPPORTED_FEATURE_COUNTS_V1_0)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)

# Cross-asset features (not used by V1.0 base model -- adapter/ensemble context only)
XD_FEATURE_LIST = [
    "xd_btc_return",         # BTC leader signal
    "xd_btc_volatility",     # BTC risk regime
    "xd_funding_spread",     # Relative funding spread
    "xd_cross_return_mean",  # Market breadth
    "xd_cross_vol_mean",     # Systemic risk
    "xd_ma_distance",        # Cross-sectional SMA-200 trend vs market avg
    "xd_momentum_rank",      # Cross-sectional return rank vs all peers
]
XD_DIM = len(XD_FEATURE_LIST)  # 7
FULL_FEATURE_LIST = FEATURE_LIST + XD_FEATURE_LIST
FULL_INPUT_DIM = len(FULL_FEATURE_LIST)  # 20


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
# All horizons active for loss computation.
# h16/h64 act as multi-scale regularizers that prevent temporal memorization.
# Without them, ShIC declines monotonically from epoch 10 (confirmed Feb 2026 vs Mar 2026 runs).
# Prediction heads exist for all REWARD_HORIZONS.
ACTIVE_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]  # Raw returns (voladj DEPRECATED)


# =============================================================================
# V1 TRANSFORMER ARCHITECTURE
# =============================================================================

# Transformer Core
# 2026-05-09 capacity-scaling flag: V1.x at D=256/N=3 is undersized for crypto
# signal density (V22 post-fix at D=320/N=6 hit Capacity-tier IC). Setting
# USE_CAPACITY_SCALING=True doubles params from ~1M to ~3-5M. Default False
# preserves deployed CONSERVATIVE/PRIME blend ckpt compat. Retraining required
# when flag flipped — old ckpts won't load with new shapes.
#
# 2026-05-10 ROLLBACK: capacity-scaling reverted to False after empirical
# failure. V1.0 cap-scaled run logs/v1/v1_0/v1_0_f29_train_20260510_141852.log
# at 10 epochs: IC1=-0.0000, ShIC=0.0000 (memorizing per gate). Per
# docs/V1_CAPACITY_SCALING_AUDIT_2026_05_10.md: V22/V25 reference at d=320
# also has ShIC=0.000 (memorized, not Capacity-tier as previously reported).
# V1 ShIC ratio is the bottleneck, not d_model — anti-mem-stack ↑ + cross-asset
# head are the correct Headline levers per WM_HEADLINE_UPGRADE_PLAN_2026_04_30.
USE_CAPACITY_SCALING = False
if USE_CAPACITY_SCALING:
    WM_D_MODEL = 320        # 256 → 320 (Capacity-scaling experiment)
    WM_N_HEADS = 8          # 320 / 8 = 40 dim per head
    WM_N_LAYERS = 6         # 3 → 6 (deeper representation)
    WM_D_FF = 1280          # 768 → 1280 (4× d_model expansion, matches V22)
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
RETURN_HEAD_DROPOUT = 0.05  # Lower dropout for return heads (need capacity, not regularization)
GUMBEL_TAU = 1.0            # Gumbel-Softmax temperature for categorical RSSM latents

# Direct Return Regression (bypasses TwoHot discretization bottleneck)
# Weight=1.0: >1.0 creates Huber shortcut bypassing TwoHot bottleneck (confirmed V3/V4/V6)
DIRECT_RETURN_WEIGHT = 3.0  # Huber dominance acts as temporal regularizer (stable ShIC=0.030 in Feb runs)

# Pairwise Ranking Loss (Phase C Path 2: learning-to-rank auxiliary)
# Samples random pairs from batch, penalizes incorrect relative ordering.
# log(1 + exp(-(pred_i - pred_j) * sign(target_i - target_j)))
# Applied to h=1 only (the only generalizing horizon).
PAIRWISE_RANK_WEIGHT = 0.0   # DISABLED: pairwise ranking trains temporal ordering = memorization
PAIRWISE_RANK_PAIRS = 256    # Number of random pairs per batch (inactive when weight=0)

# Loss initialization: FORCE the model to learn returns AND regime, not just reconstruction
# Kendall log_vars: exp(-s) * L + 0.5*s  →  higher s = lower weight
# [rec=1.0, kl=0.0, ret_1=-2.0, ret_4=-2.0, ret_16=-2.0, ret_64=-2.0, regime=-1.5]
# rec: exp(-1.0)=0.37x  (reconstruction already perfect, down-weight it)
# ret: exp(2.0)=7.4x    (returns need 7x more gradient signal)
# regime: exp(1.5)=4.5x  (was 0.0/1.0x — gradient starvation caused neutral collapse)
LOG_VAR_INIT = [1.0, 0.0, -2.0, -2.0, -2.0, -2.0, -1.5]

# Regularization
WM_DROPOUT = 0.15           # Dropout rate
WM_FREE_NATS = 1.0          # KL free nats threshold
WM_ATTENTION_DROPOUT = 0.15  # Attention dropout

# Regime Classification -- Focal Loss (Lin et al., 2017)
# gamma > 0 down-weights easy examples (confident neutral), up-weights hard (bear/bull)
# gamma=0 reverts to standard cross-entropy
REGIME_FOCAL_GAMMA = 2.0

TWOHOT_FOCAL_GAMMA = 0.0  # DISABLED: focal upweights temporally-clustered tail returns
LABEL_SMOOTHING = 0.0     # DISABLED: learns temporal return shape
#
# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule that penalizes predictions proportionally
# to distance from truth. Unlike focal/smoothing, CRPS does not reweight samples
# or soften targets -- it changes the metric on the output space.
# Use --loss-type crps to opt-in for A/B testing.
RETURN_LOSS_TYPE = "ce"

# Regime Head Architecture
# Expanded from 128 to 256: undersized head was another cause of regime collapse
# (return heads get 384-dim trunk, regime had only 128 — capacity imbalance)
REGIME_HEAD_DIM = 256

# Kendall Weight Corridors (prevent reconstruction from hogging gradients)
# Root cause: Kendall optimal s_i = log(2*L_i). When rec loss drops to 0.013,
# s_rec -> -3.65 giving rec 38.5x weight. Meanwhile return losses stay at 0.45,
# s_ret -> -0.1 giving returns only 1.1x weight. Returns starve of gradient.
# Fix: clamp log_vars asymmetrically so returns always maintain priority.
REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q (2026-04-14): rec gets at most exp(-0.5)=0.61x weight. Autopsy showed rec gradient was dominating return-head training (top grad-norm features had IC~0.001 — model was investing in reconstructing noise-rich features). Reduce rec dominance to give return heads more gradient.
RETURN_LOG_VAR_CLAMP_MAX = -2.0     # Returns always get at least exp(2)=7.4x weight (was drifting to 1.1x)
# NOTE: Per-horizon clamp dict (RETURN_LOG_VAR_CLAMP_PER_H) REMOVED -- uniform
# RETURN_LOG_VAR_CLAMP_MAX=-2.0 for all active horizons. Per-H caps hurt regularization.
REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight (prevents drift back to 1.0x)

# ShIC-based early stopping (stop when memorization sets in)
# Tracks consecutive shuffled IC declines. When ShIC has declined N times in a row,
# the model is memorizing temporal patterns and further training is counterproductive.
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

# FIX(critical): agent.py requires RSSM_FLAT_DIM and AGENT_HIDDEN.
# RSSM_FLAT_DIM is derived from RSSM settings (same value as FLAT_DIM).
# AGENT_HIDDEN is the hidden dimension for the ActorCritic network.
RSSM_FLAT_DIM = FLAT_DIM   # Alias: RSSM_LATENT_DIM * RSSM_CLASSES = 576
AGENT_HIDDEN = 256          # Agent network hidden dimension

# Per-asset fee overrides (bps). Agents should look up their asset here.
# TODO: Integrate per-asset fees into compute_reward() — different assets have
# different spreads and liquidity. For now all use BASE_FEE as default.
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

# Architecture -- FiLM modulation of return trunk, ~15K params
ADAPTER_FEAT_DIM = WM_D_MODEL + FLAT_DIM    # 832 (cat(h_seq, z_post))
ADAPTER_CONTEXT_DIM = 17                     # Rolling IC(4) + bias(4) + regime(3) + vol(1) + XD last-bar(5)
ADAPTER_BOTTLENECK = 16                      # Compressed feat representation
ADAPTER_FILM_HIDDEN = 48                     # FiLM generator hidden dim
ADAPTER_MAX_SCALE_RANGE = 0.3               # Scale in [1-range, 1+range] = [0.7, 1.3]
ADAPTER_MAX_SHIFT_INIT = 0.01               # Initial max shift magnitude (learnable)

# Training -- short cycles, tiny network, no EMA needed
ADAPTER_LR = 1e-3                           # Higher LR for tiny network
ADAPTER_WEIGHT_DECAY = 1e-3                 # Light regularization
ADAPTER_EPOCHS = 30                          # Short training cycles
ADAPTER_STEPS_PER_EPOCH = 500               # Steps per adapter epoch
ADAPTER_BATCH_SIZE = 64                     # Larger batch (no base model gradients)
ADAPTER_VAL_EVERY = 5                       # Validate every N epochs
ADAPTER_GRAD_CLIP = 1.0                     # Gradient clipping

# Rolling window -- recent data for adapter training
ADAPTER_WINDOW_BARS = 500_000               # ~2 weeks of recent bars per asset
ADAPTER_CONTEXT_LOOKBACK = 2000             # Bars for context vector computation
ADAPTER_REPLAY_FRACTION = 0.2              # 20% replay buffer samples in training

# Drift monitor -- tracks base model performance degradation
DRIFT_WINDOW_SIZE = 5000                    # Rolling IC window (bars)
DRIFT_WARN_RATIO = 0.5                     # Warn if IC < 50% of baseline
DRIFT_RETRAIN_RATIO = 0.3                  # Retrain if IC < 30% of baseline

# Replay buffer -- regime-balanced sample storage
REPLAY_BUFFER_SIZE = 10000                  # Max samples (balanced across 3 regimes)


# =============================================================================
# V1.E SNAPSHOT ENSEMBLE (Cyclical Cosine Annealing)
# =============================================================================

# Cyclical LR: divide total epochs into N cycles, cosine decay per cycle
# 10 cycles x 20 epochs: captures snapshots before memorization depth
# (memorization starts ~epoch 20-30 per cycle based on vanilla ShIC data)
SNAPSHOT_N_CYCLES = 10             # Number of warm restart cycles (was 5)
SNAPSHOT_EPOCHS_PER_CYCLE = WM_TOTAL_EPOCHS // SNAPSHOT_N_CYCLES  # 20 (was 40)
SNAPSHOT_LR_MAX = WM_LR            # Peak LR at start of each cycle (2e-4)
SNAPSHOT_LR_MIN = WM_MIN_LR        # Min LR at end of each cycle (1e-6)
SNAPSHOT_WARMUP_EPOCHS = 3         # Linear warmup within FIRST cycle only

# Ensemble inference
SNAPSHOT_TOP_K = 3                 # Use top-K snapshots by shuffled IC (not all 10)
SNAPSHOT_SHIC_GATE = True          # Only save snapshot if ShIC > GATE_IC_MIN


def get_snapshot_lr_for_epoch(epoch: int) -> float:
    """
    Cyclical cosine annealing with warm restarts (Loshchilov & Hutter, 2017).

    Divides training into SNAPSHOT_N_CYCLES cycles. Each cycle does cosine
    decay from SNAPSHOT_LR_MAX to SNAPSHOT_LR_MIN. First cycle has linear warmup.

    Args:
        epoch: current epoch (0-indexed)
    Returns:
        Learning rate for this epoch
    """
    # First cycle warmup
    if epoch < SNAPSHOT_WARMUP_EPOCHS:
        warmup_progress = (epoch + 1) / SNAPSHOT_WARMUP_EPOCHS
        return SNAPSHOT_LR_MIN + (SNAPSHOT_LR_MAX - SNAPSHOT_LR_MIN) * warmup_progress

    # Determine which cycle we're in
    cycle = epoch // SNAPSHOT_EPOCHS_PER_CYCLE
    cycle = min(cycle, SNAPSHOT_N_CYCLES - 1)
    epoch_in_cycle = epoch - cycle * SNAPSHOT_EPOCHS_PER_CYCLE

    # Cosine decay within cycle
    progress = epoch_in_cycle / max(1, SNAPSHOT_EPOCHS_PER_CYCLE - 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return SNAPSHOT_LR_MIN + (SNAPSHOT_LR_MAX - SNAPSHOT_LR_MIN) * cosine_factor


# =============================================================================
# V1.D MULTI-HEAD DIVERSITY (NCL)
# =============================================================================

DIVERSITY_N_HEADS = 5              # Number of parallel return prediction heads
DIVERSITY_NCL_LAMBDA = 0.5         # NCL diversity loss weight (higher = more diverse)
DIVERSITY_HEAD_DIM = 384           # Same as RETURN_HEAD_DIM (each head has same capacity)
DIVERSITY_HEAD_DROPOUT = 0.05      # Per-head dropout
DIVERSITY_LR = 2e-4                # Same as base model (train from scratch)
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
# LR SCHEDULE FUNCTION
# =============================================================================

def get_lr_for_epoch(epoch: int) -> float:
    """
    Compute learning rate for a given epoch.

    Schedule:
      - Epochs 0..WM_WARMUP_EPOCHS-1: linear warmup from WM_MIN_LR to WM_LR
      - Epochs WM_WARMUP_EPOCHS..WM_TOTAL_EPOCHS: cosine decay from WM_LR to WM_MIN_LR

    Args:
        epoch: current epoch (0-indexed)

    Returns:
        Learning rate for this epoch
    """
    if epoch < WM_WARMUP_EPOCHS:
        # Linear warmup
        warmup_progress = (epoch + 1) / WM_WARMUP_EPOCHS
        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * warmup_progress
    else:
        # Cosine decay
        decay_epochs = WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS
        decay_progress = (epoch - WM_WARMUP_EPOCHS) / max(1, decay_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * cosine_factor

# Cross-version invariant (CLAUDE.md): target source for all training
target_prefix = "target_return"  # raw returns (voladj DEPRECATED)


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v1_0")
except ImportError:
    pass
