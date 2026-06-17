"""

V3 Settings — WaveNet-GRU Hybrid World Model (SOTA 2025/26)



Architecture: WaveNet-style Gated Dilated Convolutions + GRU + RSSM

Key Difference: Gated activations (tanh * sigmoid) replace ReLU convolutions

               GRU replaces LSTM (fewer params, same performance)

Hardware Target: RTX 4060 (8GB) + i9, Windows 11

Data: 10 assets, 37 features (30 base + 7 cross-asset), ~30M total bars



SOTA 2025/26:

  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)

  - AdamW with betas=(0.9, 0.95) (LLaMA/Chinchilla standard)



Hardware Target: RTX 4060 (8GB) + i9, Windows 11

Target: <10 min/epoch

"""

import math

import platform

import torch

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

        PROJECT_ROOT = SCRIPT_DIR.parent

        break

    PROJECT_ROOT = PROJECT_ROOT.parent



DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)  # 2026-05-10 fix: data is under /dollar/ subdir; matches V4/V22/V25

MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v3" / "v3"

LOG_DIR = PROJECT_ROOT / "logs" / "v3" / "v3"

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

    "norm_deviation",

    "norm_fd_close",

    "norm_vpin",

    "norm_flow_imbalance",

    "norm_vol_cluster",

    "norm_funding",

    "norm_tick_count",

    "norm_log_volume",

    "norm_hl_spread",

    "hurst_regime",

    "norm_oi_change",

    "norm_return_1",

    "norm_spread_bps",

    "norm_ma_distance",

    "norm_whale",

    "norm_efficiency",

    "norm_return_4",

    "norm_return_16",

    # Tier 1 features -- orthogonal replacements (V51b)

    "norm_return_kurtosis",  # Rolling excess kurtosis (distribution shape)

    "norm_bar_duration",     # Bar duration (volume clock speed, log seconds)

    "norm_funding_momentum", # Funding rate of change (leverage dynamics)

    # Hawkes features (trade clustering dynamics)
    "norm_hawkes_intensity",      # Tick rate vs EMA (self-excitation signal)
    "norm_hawkes_buy_intensity",  # Buy-side clustering (informed buying acceleration)
    "norm_hawkes_sell_intensity", # Sell-side clustering (liquidation/distribution)
    "norm_hawkes_imbalance",     # Buy - sell clustering (directional clustering)
    # IC-boosting features (Tier 2 -- dynamics, not levels)
    "norm_momentum_accel",       # Second derivative of price (trend acceleration)
    "norm_vol_price_corr",       # Volume-price correlation (accumulation/distribution)
    "norm_vol_ratio",            # Volatility term structure (short/long vol)
    "norm_flow_persistence",     # Flow autocorrelation (institutional campaigns)
    "norm_oi_price_divergence",  # OI building while price flat (spring loading)
    # SOTA features (30-33)
    "norm_yz_volatility",        # 30: Yang-Zhang vol (MVUE, overnight+RS+OC)
    "norm_cs_spread",            # 31: Corwin-Schultz bid-ask spread from H/L
    "norm_perm_entropy",         # 32: Permutation entropy (predictability/complexity)
    "norm_kyle_lambda",          # 33: Kyle's lambda (price impact per $ order flow)

    # Cross-asset features (20-24) -- computed in make_dataset_legacy.py (Phase 2)

    "xd_btc_return",

    "xd_btc_volatility",

    "xd_funding_spread",

    "xd_cross_return_mean",

    "xd_cross_vol_mean",
    "xd_ma_distance",            # Cross-sectional SMA-200 trend vs market avg
    "xd_momentum_rank",          # Cross-sectional return rank vs all peers

]

INPUT_DIM = len(FEATURE_LIST)  # 41

# Feature subsets for --features selection
# 13: legacy base only (norm_deviation ... norm_spread_bps)
# 18: + extended (ma_distance, whale, efficiency, return_4, return_16)
# 30: + Tier 1 + Hawkes + IC-boost (all base, no XD)
# 37: Full (30 base + 7 XD)
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
XD_DROPOUT_RATE = 0.85  # SOTA-2026: was 0.7; CC-H4 anti-mem (drop ~12 of 7 XD per batch, harder cross-asset reconstruction)
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_V3 = (13, 18, 29, 30, 34, 37, 41, 51, 121, 127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V3:
        raise ValueError(
            f"V3 supports {sorted(SUPPORTED_FEATURE_COUNTS_V3)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


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

# V3 WAVENET-GRU ARCHITECTURE

# =============================================================================



# WaveNet-style Temporal Convolutions (right-sized for small dataset)

TCN_CHANNELS = [96, 128, 192, 256, 256, 256, 256]  # SOTA-2026: 7 layers (was 6); covers seq_len 256 receptive field

TCN_KERNEL_SIZE = 3

TCN_DILATIONS = [1, 2, 4, 8, 16, 32, 64]  # SOTA-2026: 7 dilations (was 6); receptive field = (3-1)*127+1 = 255 bars (matches seq_len 256)



# GRU for sequential dynamics (replaces LSTM — fewer params, same quality)

GRU_HIDDEN_DIM = 256

GRU_NUM_LAYERS = 2



# RSSM Latent Space (right-sized for small dataset)

RSSM_LATENT_DIM = 24

RSSM_CLASSES = 24

FLAT_DIM = RSSM_LATENT_DIM * RSSM_CLASSES  # 576



# Asset Conditioning

WM_ASSET_EMB_DIM = 32



# TwoHot Symlog (return prediction) -- raw return targets

NUM_BINS = 255              # 255 bins across [-1, 1] for raw return targets

BIN_MIN = -1.0              # Raw return targets in [-1, 1]

BIN_MAX = 1.0               # 255 bins in [-1, 1] -> resolution 0.0078 per bin

# Return Loss Type: "ce" (cross-entropy, default) or "crps" (CRPS, ordinal-aware).
# CRPS is a strictly proper scoring rule. Use --loss-type crps for A/B testing.
RETURN_LOSS_TYPE = "ce"



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

REC_LOG_VAR_CLAMP_MIN = 0.5        # Pattern Q (2026-04-14): rec gets at most exp(-0.5)=0.61x weight (was 1.0x). Autopsy showed rec gradient dominated, starving return heads. Reduce rec dominance for more signal extraction.

RETURN_LOG_VAR_CLAMP_MAX = -2.0

# RETURN_LOG_VAR_CLAMP_PER_H = {1: -2.0, 4: -2.0, 16: -1.0, 64: -1.0}  # h16/h64 reduced (memorize OOS)  # DISABLED: uniform clamp via RETURN_LOG_VAR_CLAMP_MAX

REGIME_LOG_VAR_CLAMP_MAX = -1.0     # Regime always gets at least exp(1)=2.7x weight

REGIME_FOCAL_GAMMA = 2.0



# TwoHot Focal Loss -- de-weights easy center-bin samples, up-weights tails

TWOHOT_FOCAL_GAMMA = 0.0  # DISABLED: focal upweights temporally-clustered tail returns, accelerates memorization

REGIME_HEAD_DIM = 256               # Regime head hidden dim (was 128 — capacity imbalance)



# ShIC-based early stopping (stop when memorization sets in)

SHUFFLED_IC_PATIENCE = 5            # Stop after N consecutive ShIC declines

SHUFFLED_IC_MIN_DECLINE = 0.001     # Only count as decline if drop > this from best

# SHIC_LR_REDUCE_AFTER = 3      # ATME fights first, LR reduces as fallback  # DISABLED: ShIC LR reduction removed

# SHIC_LR_REDUCE_FACTOR = 0.5   # Multiply LR by this factor per ShIC-triggered reduction  # DISABLED: ShIC LR reduction removed



# Regularization (stronger for small dataset)

WM_DROPOUT = 0.15

TCN_DROPOUT = 0.2

WM_FREE_NATS = 1.5  # SOTA-2026: was 1.0; KL floor raised per CC-H4 anti-mem (forces categorical bottleneck to throw away more low-utility info)

KL_ANNEAL_EPOCHS = 20            # Linearly ramp KL weight from 0 to 1 over N epochs

# Variational Information Bottleneck (VIB) -- clean-variant memorization defense.
# V3-clean stripped RSSM, leaving no compression pressure. 4.7M params memorized
# IC=0.28 at epoch 5 with ShIC=0.0002. VIB is the non-recurrent analog of RSSM:
# project h_seq to (mu,logvar) of a small latent z, sample z~N(mu,sigma), KL to
# N(0,I). Bottleneck dim << d_model forces the encoder to allocate limited bits
# to features that predict returns, not temporal autocorrelation shortcuts.
VIB_Z_DIM = 32                   # Latent bottleneck dimension (d_model=256 -> 32 = 8x compression)
VIB_KL_WEIGHT = 0.05             # Max KL weight (annealed via KL_ANNEAL_EPOCHS)
VIB_LOGVAR_INIT = -4.0           # Initial logvar bias (std~0.135 at t=0, mild noise)
VIB_LOGVAR_MIN = -6.0            # Clamp floor (prevents numerical collapse)
VIB_LOGVAR_MAX = 2.0             # Clamp ceiling (prevents exploding noise)

WM_WEIGHT_DECAY = 5e-2



# Data Augmentation

AUG_NOISE_STD = 0.02           # Gaussian noise std added to input features

AUG_FEAT_DROP = 0.1            # Probability of zeroing out entire feature columns





# =============================================================================

# TRAINING

# =============================================================================



WM_SEQ_LEN = 256  # SOTA-2026: was 96; extended for 7-layer TCN receptive field [1,2,4,8,16,32,64] = 255 bars

WM_BATCH_SIZE = 32  # Small batch = implicit gradient noise regularization (was 48, violates invariant)

WM_LR = 2e-4              # Match V1.0 (was 3e-4)

WM_GRAD_CLIP = 1.0



WM_TOTAL_EPOCHS = 200

WM_STEPS_PER_EPOCH = 2000

WM_VAL_EVERY = 5

WM_PATIENCE = 40



# LR Schedule: Linear warmup + cosine decay

WM_WARMUP_EPOCHS = 5

WM_MIN_LR = 1e-6



WM_MASK_RATIO_START = 0.10

WM_MASK_RATIO_END = 0.25

WM_MASK_RAMP_EPOCHS = 40

WM_BLOCK_SIZE_RATIO = 0.10



LOG_FREQ = 25



# EMA

EMA_DECAY = 0.995           # EMA model decay rate





# =============================================================================

# LR SCHEDULE FUNCTION

# =============================================================================



def get_lr_for_epoch(epoch: int) -> float:

    """Linear warmup for WM_WARMUP_EPOCHS, then cosine decay to WM_MIN_LR."""

    if epoch < WM_WARMUP_EPOCHS:

        # Linear warmup from WM_MIN_LR to WM_LR

        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * ((epoch + 1) / WM_WARMUP_EPOCHS)

    else:

        # Cosine decay from WM_LR to WM_MIN_LR

        progress = (epoch - WM_WARMUP_EPOCHS) / max(1, WM_TOTAL_EPOCHS - WM_WARMUP_EPOCHS)

        progress = min(progress, 1.0)

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        return WM_MIN_LR + (WM_LR - WM_MIN_LR) * cosine_decay





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

# V3.x ADAPTIVE RESIDUAL ADAPTER

# =============================================================================



# Architecture -- FiLM modulation of return trunk, ~15K params

ADAPTER_FEAT_DIM = GRU_HIDDEN_DIM + FLAT_DIM  # 832 (cat(h_seq, z_post))

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





# =============================================================================

# V3.E MULTI-SEED ENSEMBLE (Independent Seeds, Full Vanilla Training Each)

# =============================================================================



# Each seed-run = full vanilla training (WM_TOTAL_EPOCHS, warmup + cosine LR).

# Different seeds -> different weight initializations -> different basins

# -> low inter-model correlation (rho~0.3-0.5) -> strong ensemble IC boost.



ENSEMBLE_N_SEEDS = 5                              # Number of independent training runs (K)

ENSEMBLE_SEEDS = [42, 1337, 2024, 7777, 31415]    # Deterministic seed list

ENSEMBLE_TOP_K = 3                                 # Use top-K seeds at inference





# =============================================================================

# V3.D MULTI-HEAD NCL DIVERSITY

# =============================================================================



DIVERSITY_N_HEADS = 5              # Number of parallel return prediction heads

DIVERSITY_NCL_LAMBDA = 0.5         # NCL diversity loss weight (higher = more diverse)

DIVERSITY_HEAD_DIM = 384           # Head dimension (matches V1)

DIVERSITY_HEAD_DROPOUT = 0.05      # Per-head dropout

DIVERSITY_LR = 3e-4                # Same as base model

DIVERSITY_WEIGHT_DECAY = 5e-2

DIVERSITY_TOTAL_EPOCHS = 200

DIVERSITY_STEPS_PER_EPOCH = 2000   # Same as base





# =============================================================================

# ANTI-TEMPORAL-MEMORIZATION (ATME)

# =============================================================================

# Forces cross-sectional signal learning over temporal memorization.

# TEMPORAL_CTX_DROP: Obs-only posterior for return/regime heads N% of training steps.

# SEQ_SHUFFLE_PROB: Shuffles time axis of training sequences (architecture-agnostic).



TEMPORAL_CTX_DROP = 0.15            # SOTA-2026: per-sample ATME (was 0.40 batch-level; canonical per CLAUDE.md). World_model.py now applies mask per-sample, not per-batch.

SEQ_SHUFFLE_PROB = 0.30             # Prob of shuffling a training sequence's time axis

# === Forecast head (2026-05-10 generalization fix; same pattern as V4/V8/V22/V25) ===
# Predict obs[t+h] from h_seq[t] (WaveNet+GRU state). Anchors h_seq to feature-faithful
# future prediction. V4 probe-validated: +88% train_IC. V3 was AutopsyMode-queued
# for a memorization issue in --clean variant; the forecast-head anchor is the
# same root-cause fix that resolved V22/V25 memorization. Operator should still
# run AutopsyMode probe to confirm anchor lifts the trajectory; see
# docs/V8_V3_QUEUED_2026_05_10.md for run commands.
USE_FORECAST_HEAD = True
FORECAST_WEIGHT = 0.5

# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output for
# downstream meta-learner risk-aware sizing. Imports from
# src/wm/_shared/headline_components.py (QuantileHeads + quantile_loss).
# Default ON since V3 not yet trained — adds ~50K params but legacy TwoHot
# remains primary head; impact: tradeable Sharpe lift in fat-tail regimes,
# negligible raw-IC effect.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1   # small weight — auxiliary, not primary

# CC-H6 regime-conditional heads (SOTA-2026): per-regime auxiliary decoders.
# Training: per-sample CE on the head matching that sample's regime label.
# Inference: soft-blend by predicted regime probabilities. Adds Sharpe
# stability across regime shifts (+0.05 Sharpe in regime-shift windows
# per WM_HEADLINE_UPGRADE_PLAN §0 CC-H6).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1     # auxiliary; primary return loss still TwoHot

# Regime-awareness depth (SOTA-2026; 2026-05-16 architectural extension).
# Three tiers, in increasing strength:
#   "off"           -- no regime conditioning beyond the regime classifier
#   "heads"         -- CC-H6 decoder-only conditioning (default; cheap; +50K params)
#   "film"          -- CC-H6 + Regime-FiLM modulator on the encoder output
#                       (cheap +1.5K params; conditions ENCODER not just heads;
#                        identity-at-init so safe to add to a planned cold-start)
#   "multi_encoder" -- TIER-3 future: N copies of encoder routed by regime
#                       gate (MoE-style; expensive; not yet implemented)
# Recommended default: "heads" for first SOTA training; "film" once
# CC-H6 baseline ships ShIC > 0.030; "multi_encoder" reserved for V3-MoE
# / V4-MoE as new top-level versions.
REGIME_AWARENESS_MODE = "film"   # SOTA-2026 default: CC-H6 + FiLM (V3 not yet trained)

# ─── HEADLINE_MODE upgrades (V3-specific, added 2026-04-30) ──────────────────
# Per WM_HEADLINE_UPGRADE_PLAN §6: extend dilations + raise seq_len.
# Activation: V3_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V3_HEADLINE_MODE", "1")))  # SOTA-2026: default ON. Legacy fallback: V3_HEADLINE_MODE=0
HEADLINE_DILATIONS = [1, 2, 4, 8, 16, 32, 64]   # was [1, 2, 4, 8]
HEADLINE_SEQ_LEN = 256                           # was 96
HEADLINE_FREE_NATS = 1.5
HEADLINE_XD_DROPOUT = 0.85
if HEADLINE_MODE:
    print(f"[V3 HEADLINE_MODE] dilations -> {HEADLINE_DILATIONS}, seq_len -> {HEADLINE_SEQ_LEN}")
    # Wire HEADLINE values into the canonical training constants.
    # TCN_DILATIONS / WM_SEQ_LEN are already at the headline values (set above),
    # so this branch is a no-op in practice but makes the contract explicit:
    # any future change to HEADLINE_DILATIONS/SEQ_LEN automatically propagates.
    TCN_DILATIONS = HEADLINE_DILATIONS
    WM_SEQ_LEN = HEADLINE_SEQ_LEN
    WM_FREE_NATS = HEADLINE_FREE_NATS
    XD_DROPOUT_RATE = HEADLINE_XD_DROPOUT


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
    _assert_canonical(globals(), version_name="v3")
except ImportError:
    pass  # smoke test mode
