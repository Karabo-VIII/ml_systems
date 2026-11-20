"""
V11 Settings -- Microstructure Feature Extractor
=================================================

No RSSM. No reconstruction. No dream step.
WaveNet-TCN encoder + time-shuffle discriminator + regime-gated experts.

Design principles:
  1. Every parameter serves return prediction. No wasted capacity.
  2. Anti-memorization is structural (discriminator), not stochastic (dropout).
  3. Dollar-bar multi-scale patterns via dilated causal convolutions.
  4. Regime specialization via hard Hurst gating (not learned router).
"""
import torch
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # 2026-05-03 fix: was 4 .parents (=src/), need 5 (=repo root)
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 migration 2026-05-25
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v11" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v11"

# ── Device ───────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = True

# ── Feature Configuration ────────────────────────────────────────────────────

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
XD_DROPOUT_RATE = 0.85  # SOTA-2026: was 0.7; CC-H4 anti-mem (cohort-aligned)
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_V11 = (13, 25, 29, 34, 41, 51, 121, 127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V11:
        raise ValueError(
            f"V11 supports {sorted(SUPPORTED_FEATURE_COUNTS_V11)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# Default: f34 (V11's traditional default; XD features not needed since
# V11 has a discriminator + WaveNet for cross-asset signal)
FEATURE_LIST = FEATURE_LIST_34
INPUT_DIM = len(FEATURE_LIST)

# ── Assets ───────────────────────────────────────────────────────────────────

NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# ── Architecture ─────────────────────────────────────────────────────────────

# WaveNet TCN encoder
WM_D_MODEL = 256
WAVENET_CHANNELS = [96, 128, 192, 256, 256, 256]   # 6 layers: full 96-bar coverage
WAVENET_DILATIONS = [1, 2, 4, 8, 16, 32]            # Receptive field: 127 bars
WAVENET_KERNEL = 3
WAVENET_DROPOUT = 0.10

# Asset embedding
WM_ASSET_EMB_DIM = 32

# Regime experts (2 experts: trending vs mean-reverting)
NUM_EXPERTS = 2
EXPERT_TRENDING_DILATIONS = [4, 8]     # Long-range for momentum/breakout
EXPERT_REVERTING_DILATIONS = [1, 2]    # Short-range for mean-reversion
EXPERT_D_MODEL = 128                    # Smaller per-expert
EXPERT_DROPOUT = 0.10
HURST_GATE_THRESHOLD = 0.1             # hurst_regime > 0.1 = trending

# Post-encoder feature attention
FEAT_ATTN_D = 64
FEAT_ATTN_HEADS = 4
FEAT_ATTN_DROPOUT = 0.10

# Return prediction
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# Active horizons (h1 and h4 ONLY -- h16/h64 don't generalize)
ACTIVE_HORIZONS = [1, 4, 16, 64]  # CLAUDE.md invariant (was [1,4] -- fixed 2026-04-27 audit; missing h16/h64 caused ShIC decay)
REWARD_HORIZONS = [1, 4, 16, 64]  # All for data loading, only ACTIVE used in loss

# ── Time-Shuffle Discriminator ───────────────────────────────────────────────

DISC_HIDDEN = 128
DISC_LAYERS = 3
DISC_LR = 1e-4
DISC_GRAD_PENALTY = 10.0               # WGAN-GP lambda
DISC_WEIGHT = 0.3                       # RAISED from 0.1 — adversarial was too weak (ShIC=0 at ep9)
DISC_UPDATE_FREQ = 1                    # Update disc every N batches

# 2026-05-09 V11 upgrade: discriminator warmup + freeze cadence.
# Hypothesis: V11's currently-low IC is partly due to discriminator pulling
# generator into sign-flipped solutions early in training. Warmup keeps
# discriminator weak for first N steps so generator can establish correct
# direction. Freeze-cadence prevents discriminator from over-fitting once
# it's identified the temporal-shuffle distinction.
DISC_WARMUP_STEPS = 1000        # Generator-only training first 1000 steps
DISC_FREEZE_AFTER_STEPS = 0     # 0 = never freeze; > 0 = freeze disc after step N

# ── Variational Information Bottleneck (ADDED 2026-04-22 SOTA upgrade) ──────
# V11 was no-RSSM, no-VIB → pure memorization via WaveNet receptive field.
# ATME zeroing alone is insufficient (30% samples zeroed, 70% memorized).
# VIB forces compressed stochastic bottleneck between h_seq and return_trunk.
VIB_Z_DIM = 32                         # Bottleneck dimension (d_model=256 → 32 = 8x compression)
VIB_KL_WEIGHT = 0.05                   # KL weight on z distribution
VIB_KL_ANNEAL_EPOCHS = 20              # Linearly anneal KL from 0 to VIB_KL_WEIGHT over N epochs
VIB_LOGVAR_INIT = -1.0                 # Initial logvar bias (prevents collapse)
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === RECON ANCHOR (keystone, 2026-06-11) ====================================
# V11 had a REAL Gaussian VIB (to_mu/to_logvar/reparam, KL already in `total`)
# but recon=torch.zeros stub + rec/kl reported 0.0 + NO recon term -> the
# bottleneck had no input-reconstruction pressure (the heads could route around
# z; the V22/V25 memorization trap: high contiguous IC, ShIC~0). RECON_WEIGHT>0
# turns on the masked recon-MSE term that forces the 32-dim VIB latent to retain
# input-reconstructable structure. Unit weight (1.0) matches the V12 donor's
# fixed unit-weight recon + the V13 graft. The missing HALF of the anchor
# (recon + VIB KL together).
RECON_WEIGHT = 1.0

# ── Anti-Memorization ────────────────────────────────────────────────────────

ATME_PROB = 0.30                        # 30% temporal context drop (aggressive)
TEMPORAL_CTX_DROP = 0.30                # Alias

# ── Training ─────────────────────────────────────────────────────────────────

WM_SEQ_LEN = 96
WM_BATCH_SIZE = 32
WM_STEPS_PER_EPOCH = 2000
DIVERSITY_STEPS_PER_EPOCH = 2000  # CLAUDE.md invariant (V11 has no NCL variant but constant required)
WM_LR = 3e-4
WM_WEIGHT_DECAY = 1e-4
WM_GRAD_CLIP = 1.0
WM_DROPOUT = 0.15
WM_WARMUP_STEPS = 500
WM_COSINE_PERIOD = 50
WM_EPOCHS = 150

# Direct return weight (Huber dominance regularizer)
DIRECT_RETURN_WEIGHT = 3.0

# Regime head
REGIME_HEAD_DIM = 128
REGIME_FOCAL_GAMMA = 2.0

# Masking (token-level random, not block)
WM_MASK_RATIO = 0.25                    # 25% random token masking
WM_BLOCK_MASK = False                   # Random tokens, not contiguous blocks

# Data split
TRAIN_RATIO = 0.50
VAL_RATIO = 0.20
OOS_RATIO = 0.20
UNSEEN_RATIO = 0.10

# ShIC monitoring
SHUFFLED_IC_PATIENCE = 5
SHUFFLED_IC_CHECK_INTERVAL = 10

# Misc
LOG_FREQ = 100
NUM_WORKERS = 0
EMA_DECAY = 0.999

# Focal / smoothing (DISABLED -- accelerate memorization)
TWOHOT_FOCAL_GAMMA = 0.0
LABEL_SMOOTHING = 0.0

# Pairwise ranking (DISABLED -- trains temporal ordering)
PAIRWISE_RANK_WEIGHT = 0.0

# Target prefix
target_prefix = "target_return"

# ─── HEADLINE_MODE upgrades (V11-specific, added 2026-04-30) ─────────────────
# V11 = V3 + V6 + V9. Per WM_HEADLINE_UPGRADE_PLAN §12: drop V9 MoE
# component (use 1 expert); inherit V3+V6 with their headline upgrades.
# Activation: V11_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V11_HEADLINE_MODE", "1")))  # SOTA-2026: default ON (V9 MoE leak dropped; V3+V6 headline knobs ON). Legacy: V11_HEADLINE_MODE=0
HEADLINE_MOE_EXPERTS = 1                # was 3 (V9-style, leaks); 1 = drop MoE
HEADLINE_DISC_SPECTRAL_NORM = True
HEADLINE_DILATIONS = [1, 2, 4, 8, 16, 32, 64]   # match V3-Headline
if HEADLINE_MODE:
    print(f"[V11 HEADLINE_MODE] MoE experts -> 1 (drop V9 leak); V3+V6 headline knobs")


# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026): per-regime auxiliary decoders.
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V11: FiLM on h_seq AFTER feat_attn, BEFORE VIB).
REGIME_AWARENESS_MODE = "film"   # SOTA-2026 default


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v11")
except ImportError:
    pass

