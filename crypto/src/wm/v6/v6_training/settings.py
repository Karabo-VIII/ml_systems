"""

V6 Settings -- Causal JEPA with Adversarial Time Shuffling (SOTA 2025/26)



Architecture: CausalGRU (NOT BiGRU) + JEPA + VICReg + Time Discriminator

Key Fix: Adversarial training penalizes temporal dependence in latents

Hardware Target: RTX 4060 (8GB) + i9, Windows 11



SOTA 2025/26:

  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)

  - AdamW with betas=(0.9, 0.95) (LLaMA/Chinchilla standard)

  - Adaptive residual adapter support

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

MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v6" / "v6"

LOG_DIR = PROJECT_ROOT / "logs" / "v6" / "v6"

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

# Feature subsets for --features selection
# === FEATURE SELECTION (centralized in src/feature_sets.py, post-2026-04-27) ===
import sys as _sys
_SRC_DIR = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_25, FEATURE_LIST_29,
    FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    FEATURE_LIST_51, FEATURE_LIST_121,
    DEAD_FEATURE_INDICES,
    get_feature_config as _central_get_feature_config,
)

# Anti-memorization (V1.1+ XD split): base_dim defined per-feature-count via central registry.
BASE_DIM = 34
XD_DROPOUT_RATE = 0.85  # SOTA-2026: was 0.7; CC-H4 anti-mem (V6 JEPA: constant present for parity, not consumed by XD-split path)
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_V6 = (13, 18, 29, 30, 34, 37, 41, 51, 121, 127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V6:
        raise ValueError(
            f"V6 supports {sorted(SUPPORTED_FEATURE_COUNTS_V6)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# 2026-05-25: deduplicated -- previously declared again here as a leftover from
# commit 0b5db25's rate update; both copies were identical 0.85 so behaviorally
# safe but future-bug-prone. See audit doc CRIT/LOW-5.

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

# V6 CAUSAL JEPA ARCHITECTURE

# =============================================================================



# Encoder Core
# 2026-05-07 capacity bump (was 2.23M params, below 4M iron-clad floor):
# - d_model 256->320 (wider GRU hidden, larger predictor)
# - n_layers 3->4 (deeper GRU, JEPA targets longer context)
# - d_latent 192->256 (richer prediction target embeddings)
# - predictor_layers 2->3 (stronger prediction head)
# - disc_hidden 128->192 (stronger time discriminator)
# Probed: 4.4M params, 0.7 GB peak VRAM at B=32 (well under 7.9GB free).

WM_D_MODEL = 320         # was 256

WM_D_LATENT = 256        # was 192

WM_N_LAYERS = 4          # was 3 (causal GRU layers, unidirectional)

WM_PREDICTOR_LAYERS = 3  # was 2



# Contrastive Learning

JEPA_TEMP = 0.1

JEPA_EMA_DECAY = 0.996

EMA_DECAY = 0.995            # Full-model EMA decay rate (for stable validation/ShIC)



# Adversarial Discriminator

DISC_HIDDEN = 192   # 2026-05-07: was 128 (paired with d_model bump to 320)

DISC_LAYERS = 3

LAMBDA_ADV = 0.50         # Boosted from 0.10 — at 0.10 the encoder ignored the discriminator (adv→0.004)
                          # Higher weight forces encoder to actively resist temporal memorization

DISC_GRAD_PENALTY = 0.0   # DISABLED: GP requires create_graph=True -> non-CuDNN GRU (5-10x slower)

DISC_LR_MULT = 0.5        # Slow disc learning: prevent disc from winning trivially (disc:2.4→5.5)
                          # Gives encoder time to learn adversarial resistance

DISC_LABEL_SMOOTH = 0.1   # Noisy labels for disc (real=0.9, fake=0.1) prevents disc collapse



# Temporal Jitter

TEMPORAL_JITTER = 4        # +/- timestep random shift



# Asset Conditioning

WM_ASSET_EMB_DIM = 32



# TwoHot Symlog (return prediction) -- raw return targets

NUM_BINS = 255              # 255 bins across [-1, 1] for raw return targets

BIN_MIN = -1.0              # Raw returns: symlog(return), bins in [-1, 1]

BIN_MAX = 1.0               # 255 bins in [-1, 1] -> resolution 0.0078 per bin

# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"



# SOTA Return Prediction Heads

RETURN_HEAD_DIM = 256

RETURN_HEAD_DROPOUT = 0.05

DIRECT_RETURN_WEIGHT = 3.0  # Huber dominance acts as temporal regularizer (stable ShIC=0.030 in Feb runs)



# Loss initialization for return uncertainty weights

LOG_VAR_INIT_RET = -2.0

LOG_VAR_INIT_REGIME = -1.5  # exp(1.5)=4.5x weight (was 0.0/1.0x — gradient starvation)



# Kendall Weight Corridors (prevent contrastive pretext from hogging gradients)

# JEPA variant: contrastive loss at s[0] plays the role of reconstruction.

# Same principle -- easy pretext task converges fast, starving returns of gradient.

CONTRASTIVE_LOG_VAR_CLAMP_MIN = 1.0  # Contrastive gets at most exp(-1)=0.37x weight
                                     # Reduced from 0.0 (1.0x): InfoNCE rewards temporal prediction,
                                     # which conflicts with anti-memorization. Weakening it lets
                                     # VIB KL + discriminator dominate the temporal pressure.

RETURN_LOG_VAR_CLAMP_MAX = -2.0

# RETURN_LOG_VAR_CLAMP_PER_H = {1: -2.0, 4: -2.0, 16: -1.0, 64: -1.0}  # h16/h64 reduced (memorize OOS)  # DISABLED: uniform clamp via RETURN_LOG_VAR_CLAMP_MAX

REGIME_LOG_VAR_CLAMP_MAX = -1.0       # Regime always gets at least exp(1)=2.7x weight

REGIME_FOCAL_GAMMA = 2.0



# TwoHot Focal Loss -- de-weights easy center-bin samples, up-weights tails

TWOHOT_FOCAL_GAMMA = 0.0  # MUST be 0.0 (CLAUDE.md invariant). Focal accelerates memorization.

REGIME_HEAD_DIM = 256                 # Regime head hidden dim (was 96 — capacity imbalance)



# ShIC-based early stopping (stop when memorization sets in)

SHUFFLED_IC_PATIENCE = 5              # Stop after N consecutive ShIC declines

SHUFFLED_IC_MIN_DECLINE = 0.001       # Only count as decline if drop > this from best

# SHIC_LR_REDUCE_AFTER = 2      # No ATME, reduce LR early  # DISABLED: ShIC LR reduction removed

# SHIC_LR_REDUCE_FACTOR = 0.5   # Multiply LR by this factor per ShIC-triggered reduction  # DISABLED: ShIC LR reduction removed



# VICReg

VICREG_SIM_WEIGHT = 25.0

VICREG_VAR_WEIGHT = 25.0

VICREG_COV_WEIGHT = 1.0



# Variational Information Bottleneck (VIB) -- replaces discriminator as primary
# anti-memorization defense. The discriminator failed (ShIC=0.008 after 92 epochs,
# GAN oscillation). VIB provides a hard rate constraint like RSSM's categorical
# bottleneck, but continuous. Combined with detached return path.
VIB_Z_DIM = 48              # Bottleneck dimension (d_latent=192 -> z=48 -> d_latent=192)
VIB_KL_WEIGHT = 0.01        # Beta for rate-distortion tradeoff
VIB_LOGVAR_INIT = -2.0      # Start near-deterministic, let KL anneal open it
VIB_LOGVAR_MIN = -5.0       # Prevent collapse
VIB_LOGVAR_MAX = 2.0        # Prevent explosion
KL_ANNEAL_EPOCHS = 30       # Ramp VIB KL weight from 0 to VIB_KL_WEIGHT over N epochs

# Auxiliary reconstruction

AUX_RECON_WEIGHT = 0.1



# Regularization

WM_DROPOUT = 0.15            # Match V1.0 proven baseline (was 0.22, over-regularized)

WM_WEIGHT_DECAY = 5e-2      # Match V1.0 proven baseline (was 0.12, over-regularized)

WM_FREE_NATS = 1.5  # SOTA-2026: was 1.0; CC-H4 anti-mem (KL floor raised, forces VIB to throw away more low-utility info)



# Data Augmentation

AUG_NOISE_STD = 0.02

AUG_FEAT_DROP = 0.1





# =============================================================================

# TRAINING

# =============================================================================



WM_SEQ_LEN = 96

WM_BATCH_SIZE = 32  # Small batch = implicit gradient noise regularization (was 40, match V1.0)

WM_LR = 2e-4                # Match V1.0 proven baseline (was 2.5e-4)

WM_GRAD_CLIP = 1.0



WM_TOTAL_EPOCHS = 180

WM_STEPS_PER_EPOCH = 2000    # Match V1.0 (was 300 = 6x undertraining!)

WM_VAL_EVERY = 5

WM_PATIENCE = 40



WM_WARMUP_EPOCHS = 5

WM_MIN_LR = 1e-6



WM_MASK_RATIO_START = 0.10   # Match V1.0 (was 0.15)

WM_MASK_RATIO_END = 0.25    # Match V1.0 (was 0.35, too aggressive for GRU)

WM_MASK_RAMP_EPOCHS = 40

WM_BLOCK_SIZE_RATIO = 0.10



LOG_FREQ = 25





# =============================================================================

# VALIDATION GATES

# =============================================================================



GATE_CONTRASTIVE_MIN = 0.3

GATE_IC_MIN = 0.015

GATE_EMBED_STD_MIN = 0.05

GATE_SHUFFLED_IC_RATIO_MIN = 0.3  # Shuffled IC / Contiguous IC (anti-memorization)

GATE_LOSS_RATIO_MAX = 2.0         # Train/Val loss ratio (overfitting detection)





# =============================================================================

# LR SCHEDULE

# =============================================================================



def get_lr_for_epoch(epoch: int) -> float:

    if epoch < WM_WARMUP_EPOCHS:

        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * ((epoch + 1) / WM_WARMUP_EPOCHS)

    progress = (epoch - WM_WARMUP_EPOCHS) / max(1, WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS)

    return WM_MIN_LR + 0.5 * (WM_LR - WM_MIN_LR) * (1 + math.cos(math.pi * progress))





# =============================================================================

# V6.x ADAPTIVE RESIDUAL ADAPTER

# =============================================================================



# Architecture -- FiLM modulation of return trunk, ~15K params

# V6 uses continuous latent space (d_latent=192) unlike V1's RSSM (24x24=576)

# ret_trunk input: ctx_latent [B, T, d_latent=192]

ADAPTER_FEAT_DIM = WM_D_LATENT                 # 192 (ctx_latent only, no cat like V1)

ADAPTER_CONTEXT_DIM = 12                       # Rolling IC(4) + bias(4) + regime(3) + vol(1)

ADAPTER_BOTTLENECK = 16                        # Compressed feat representation

ADAPTER_FILM_HIDDEN = 48                       # FiLM generator hidden dim

ADAPTER_MAX_SCALE_RANGE = 0.3                  # Scale in [1-range, 1+range] = [0.7, 1.3]

ADAPTER_MAX_SHIFT_INIT = 0.01                  # Initial max shift magnitude (learnable)



# Training -- short cycles, tiny network, no EMA needed

ADAPTER_LR = 1e-3                              # Higher LR for tiny network

ADAPTER_WEIGHT_DECAY = 1e-3                    # Light regularization

ADAPTER_EPOCHS = 30                            # Short training cycles

ADAPTER_STEPS_PER_EPOCH = 500                  # Steps per adapter epoch

ADAPTER_BATCH_SIZE = 64                        # Larger batch (no base model gradients)

ADAPTER_VAL_EVERY = 5                          # Validate every N epochs

ADAPTER_GRAD_CLIP = 1.0                        # Gradient clipping



# Rolling window -- recent data for adapter training

ADAPTER_WINDOW_BARS = 500_000                  # ~2 weeks of recent bars per asset

ADAPTER_CONTEXT_LOOKBACK = 2000                # Bars for context vector computation

ADAPTER_REPLAY_FRACTION = 0.2                  # 20% replay buffer samples in training



# Drift monitor -- tracks base model performance degradation

DRIFT_WINDOW_SIZE = 5000                       # Rolling IC window (bars)

DRIFT_WARN_RATIO = 0.5                         # Warn if IC < 50% of baseline

DRIFT_RETRAIN_RATIO = 0.3                      # Retrain if IC < 30% of baseline



# Replay buffer -- regime-balanced sample storage

REPLAY_BUFFER_SIZE = 10000                     # Max samples (balanced across 3 regimes)





# =============================================================================

# V6.E MULTI-SEED ENSEMBLE (Independent Seeds, Full Vanilla Training Each)

# =============================================================================



ENSEMBLE_N_SEEDS = 5                              # Number of independent training runs (K)

ENSEMBLE_SEEDS = [42, 1337, 2024, 7777, 31415]    # Deterministic seed list

ENSEMBLE_TOP_K = 3                                 # Use top-K seeds at inference





# =============================================================================

# V6.D MULTI-HEAD NCL DIVERSITY

# =============================================================================



DIVERSITY_N_HEADS = 5              # Number of parallel return prediction heads

DIVERSITY_NCL_LAMBDA = 0.5         # NCL diversity loss weight (higher = more diverse)

DIVERSITY_HEAD_DIM = 256           # V6 latent is 192, so 256 is proportional

DIVERSITY_HEAD_DROPOUT = 0.05      # Per-head dropout

DIVERSITY_LR = 3e-4                # Same as V6 base model

DIVERSITY_WEIGHT_DECAY = 5e-2

DIVERSITY_TOTAL_EPOCHS = 200

DIVERSITY_STEPS_PER_EPOCH = 2000   # Match V1.0 data exposure





# =============================================================================

# ANTI-TEMPORAL-MEMORIZATION (ATME)

# =============================================================================

# Forces cross-sectional signal learning over temporal memorization.

# TEMPORAL_CTX_DROP: N/A for JEPA (no h_seq/z_post split) — kept for API parity.

# SEQ_SHUFFLE_PROB: Shuffles time axis of training sequences (architecture-agnostic).



TEMPORAL_CTX_DROP = 0.0             # JEPA has no h_seq to drop — no-op for V6

SEQ_SHUFFLE_PROB = 0.30             # Prob of shuffling a training sequence's time axis (match V3/V9)

# ─── HEADLINE_MODE upgrades (V6-specific, added 2026-04-30) ──────────────────
# V6 JEPA + Discriminator: needs spectral-norm + R1 reg for GAN stability;
# discriminator on RESIDUAL not encoder output (V6 fix log idea).
# Per WM_HEADLINE_UPGRADE_PLAN §8.
# Activation: V6_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V6_HEADLINE_MODE", "1")))  # SOTA-2026: default ON (V6 capacity-fixed; activates disc spectral_norm + R1 + residual target). Legacy: V6_HEADLINE_MODE=0
HEADLINE_DISC_SPECTRAL_NORM = True
HEADLINE_DISC_R1_GAMMA = 10.0
HEADLINE_DISC_TARGET = "residual"   # was "encoder_output"
if HEADLINE_MODE:
    print(f"[V6 HEADLINE_MODE] disc spectral_norm + R1 + residual target")


# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output. V6 has
# .detach() on heads (encoder supervised via InfoNCE/VICReg/disc), so quantile
# loss won't disrupt JEPA's anti-mem; it adds risk-aware sizing for meta-learner.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026): per-regime auxiliary decoders.
# Adds Sharpe stability across regime shifts.
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V6 applies FiLM on ctx_latent, BEFORE VIB).
# Identity-at-init so no early disruption to JEPA training.
#   "off"           — no regime conditioning beyond classifier
#   "heads"         — CC-H6 decoder-only (default)
#   "film"          — CC-H6 + RegimeFiLM on ctx_latent
#   "multi_encoder" — TIER-3 future (not implemented)
REGIME_AWARENESS_MODE = "film"   # SOTA-2026 default


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
    _assert_canonical(globals(), version_name="v6")
except ImportError:
    pass  # smoke test mode
