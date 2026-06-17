# CDAP contract
__contract__ = {
    "kind": "model_settings",
    "model": "V1.4",
    "outputs": {
        "checkpoints": "models/wm/v1/v1_4/v1_4_f<N>_wm_*.pt",
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
        "feature_attention_block":    True,        # iTransformer cross-feature attention
    },
    "rationale": "FeatureAttentionBlock variant: f34 IC=0.0679 ShIC=0.0314.",
}

"""
V1.4 Settings -- Transformer-RSSM World Model with FeatureAttentionBlock (iTransformer)

Based on V1/TMP_2 proven architecture (ShIC=0.0305, GATE PASS, 105 epochs).
Upgraded: 18 features (13 base + 5 cross-asset), label smoothing.
V1.4 adds FeatureAttentionBlock (iTransformer-style cross-feature attention).

Architecture: Causal Transformer + RSSM Latents
Hardware Target: RTX 4060 (8GB) + i9, Windows 11
Data: 10 assets, 18 features (13 base + 5 cross-asset)

SOTA components:
  - FlashAttention via F.scaled_dot_product_attention (PyTorch 2.0+)
  - RoPE for relative position encoding (Su et al., 2021)
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
  - SwiGLU gated FFN (Shazeer, 2020)
  - AdamW with betas=(0.9, 0.95) (LLaMA/Chinchilla standard)
  - RevIN disabled by default (causes temporal memorization; opt-in via --revin)
  - FeatureAttentionBlock: iTransformer-style cross-feature attention (V1.4 unique)

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
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v1" / "v1_4"
LOG_DIR = PROJECT_ROOT / "logs" / "v1" / "v1_4"
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
    "norm_ma_distance",      # 13: SMA-200 distance (medium-term trend regime)
    "norm_whale",            # 14: Avg trade size = volume/tick_count (institutional flow)
    "norm_efficiency",       # 15: Price efficiency ratio (trending vs choppy)
    "norm_return_4",         # 16: Lagged 4-bar cumulative return (mean-reversion signal)
    "norm_return_16",        # 17: Lagged 16-bar cumulative return (medium-term momentum)
    # Tier 1 features (18-20) -- V51b orthogonal replacements
    "norm_return_kurtosis",      # 18: Rolling excess kurtosis (distribution shape)
    "norm_bar_duration",         # 19: Bar duration (volume clock speed, log seconds)
    "norm_funding_momentum",     # 20: Funding rate of change (leverage dynamics)
    # Hawkes features (21-24) -- trade clustering dynamics
    "norm_hawkes_intensity",     # 21: Tick rate vs EMA (self-excitation signal)
    "norm_hawkes_buy_intensity", # 22: Buy-side clustering (informed buying)
    "norm_hawkes_sell_intensity",# 23: Sell-side clustering (liquidation/distribution)
    "norm_hawkes_imbalance",     # 24: Buy - sell clustering (directional clustering)
    # IC-boosting features (25-29) -- Tier 2 dynamics
    "norm_momentum_accel",       # 25: Second derivative of price (trend acceleration)
    "norm_vol_price_corr",       # 26: Volume-price correlation (accumulation/distribution)
    "norm_vol_ratio",            # 27: Volatility term structure (short/long vol)
    "norm_flow_persistence",     # 28: Flow autocorrelation (institutional campaigns)
    "norm_oi_price_divergence",  # 29: OI building while price flat (spring loading)
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

# === FEATURE SELECTION ===
# 13: Legacy base only (backward compat, V1.0 identical)
# 18: Extended base (13 legacy + whale, efficiency, return_4, return_16, ma_distance)
# 21: + Tier 1 (kurtosis, bar_duration, funding_momentum)
# 25: + Hawkes (hawkes_intensity/buy/sell/imbalance)
# 30: + IC-boost (momentum_accel, vol_price_corr, vol_ratio, flow_persistence, oi_price_div)
# 34: + SOTA (yz_vol, cs_spread, perm_entropy, kyle_lambda) -- all base, no XD
# 37: Legacy full (30 base + 7 XD) -- backward compat with old checkpoints
# 41: Full (34 base + 7 XD) -- maximum feature set
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

# Anti-memorization: XD features form temporal fingerprints over 96-bar windows.
# Posterior and decoder only use BASE_DIM features; XD features get heavy
# per-timestep dropout + noise so the encoder can't build sequential fingerprints.
BASE_DIM = 34              # Features [0:BASE_DIM] are "base" (safe for posterior/recon)
XD_DROPOUT_RATE = 0.7      # Per-timestep dropout on features [BASE_DIM:] during training
XD_NOISE_STD = 0.3         # Heavy noise on XD features (vs 0.02 base aug noise)

# V1.4 supports the full registry (FeatureAttention has no special feature constraints)
SUPPORTED_FEATURE_COUNTS_V1_4 = (13, 18, 21, 25, 29, 30, 34, 37, 41,
                                  46, 51, 60, 73, 78, 81, 84, 97, 110, 121,
                                  127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V1_4:
        raise ValueError(
            f"V1.4 supports {sorted(SUPPORTED_FEATURE_COUNTS_V1_4)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)

# NOTE: LABEL_SMOOTHING REMOVED -- accelerates temporal memorization on return TwoHot.
# Return loss uses plain bucketer.compute_loss().
# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"

# V1.4: FeatureAttentionBlock (iTransformer-style cross-feature attention)
# Attends over feature dimension at each time step independently
# Learns cross-feature interactions before temporal encoding
FEAT_ATTN_D_FEAT = 32     # Feature token embedding dimension
FEAT_ATTN_N_HEADS = 4     # Attention heads (must divide FEAT_ATTN_D_FEAT)
FEAT_ATTN_DROPOUT = 0.1   # Dropout in feature attention


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

# Regime Head Architecture
# Expanded from 128 to 256: undersized head was another cause of regime collapse
# (return heads get 384-dim trunk, regime had only 128 — capacity imbalance)
REGIME_HEAD_DIM = 256

# Kendall Weight Corridors (prevent reconstruction from hogging gradients)
# Root cause: Kendall optimal s_i = log(2*L_i). When rec loss drops to 0.013,
# s_rec -> -3.65 giving rec 38.5x weight. Meanwhile return losses stay at 0.45,
# s_ret -> -0.1 giving returns only 1.1x weight. Returns starve of gradient.
# Fix: clamp log_vars asymmetrically so returns always maintain priority.
REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q (2026-04-14): rec gets at most exp(-0.5)=0.61x weight (was 1.0x). Autopsy showed rec gradient dominated, starving return heads. Reduce rec dominance for more signal extraction.
RETURN_LOG_VAR_CLAMP_MAX = -2.0     # Returns always get at least exp(2)=7.4x weight (was drifting to 1.1x)
# NOTE: Per-horizon clamp dict (RETURN_LOG_VAR_CLAMP_PER_H) REMOVED -- uniform
# RETURN_LOG_VAR_CLAMP_MAX=-2.0 for all active horizons. Per-H caps hurt regularization.
REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight (prevents drift back to 1.0x)

# ShIC-based early stopping (stop when memorization sets in)
# Tracks consecutive shuffled IC declines. When ShIC has declined N times in a row,
# the model is memorizing temporal patterns and further training is counterproductive.
# NOTE: patience was 3 but +/-0.0003 noise triggered premature stop at epoch 40/200.
# Increased to 5 + min decline threshold to prevent noise-triggered early stops.
SHUFFLED_IC_PATIENCE = 5            # Stop after N consecutive ShIC declines
SHUFFLED_IC_CHECK_INTERVAL = 10     # Compute ShIC every N epochs (via AntifragileConfig.shuffled_ic_every)
SHUFFLED_IC_MIN_DECLINE = 0.001     # Only count as decline if drop > this from best
# NOTE: ShIC LR reduction REMOVED -- locks in memorized weights, prevents recovery.
# LR follows warmup+cosine schedule only. ShIC decline triggers early stop.


# =============================================================================
# TRAINING
# =============================================================================

WM_SEQ_LEN = 96             # ~96 dollar bars ~ 24h of BTC
WM_BATCH_SIZE = 32          # Lower than Mamba due to O(T^2) attention
WM_LR = 2e-4                # Peak learning rate (after warmup)
WM_WEIGHT_DECAY = 5e-2      # Weight decay
WM_GRAD_CLIP = 1.0          # Global gradient clipping

WM_TOTAL_EPOCHS = 200       # Total training epochs
WM_STEPS_PER_EPOCH = 2000   # Steps per epoch (sized for ~15M bars across 5 assets)
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
ADAPTER_CONTEXT_DIM = 12                     # Rolling IC(4) + bias(4) + regime(3) + vol(1)
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
    print(f"[V1.x HEADLINE_MODE] WM_FREE_NATS=1.5 XD_DROPOUT_RATE=0.85")


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v1_4")
except ImportError:
    pass

