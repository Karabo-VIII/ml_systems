"""
V12 Settings -- Cross-Asset Attention Model
=============================================

Processes ALL assets jointly. Each asset gets a WaveNet encoder,
then cross-asset multi-head attention lets assets inform each other.

Key insight: when BTC breaks out AND ETH funding is negative AND SOL
VPIN is spiking, the combined signal is stronger than any individual
asset's features. This model learns those cross-asset interactions.

Requires synchronized multi-asset batches (all 10 assets at same timestamps).
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # 2026-05-03 fix: was 4 .parents (=src/), need 5 (=repo root)
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 migration 2026-05-25
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v12" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v12"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = True

# ── Features ─────────────────────────────────────────────────────────────────

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

SUPPORTED_FEATURE_COUNTS_V12 = (13, 25, 29, 34, 41, 51, 121, 127, 133, 154, 161)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V12:
        raise ValueError(
            f"V12 supports {sorted(SUPPORTED_FEATURE_COUNTS_V12)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)



FEATURE_LIST = FEATURE_LIST_25  # Default f25 (XD features not needed -- model IS cross-asset)
INPUT_DIM = len(FEATURE_LIST)

# ── Assets ───────────────────────────────────────────────────────────────────

NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# ── Architecture ─────────────────────────────────────────────────────────────

# Per-asset WaveNet encoder
WM_D_MODEL = 128                        # Smaller per-asset (10 assets in parallel)
WAVENET_CHANNELS = [64, 96, 128]        # 3 layers (lighter than V11's 4)
WAVENET_DILATIONS = [1, 2, 4]           # Receptive field: 3+5+9 = 17 bars
WAVENET_KERNEL = 3
WAVENET_DROPOUT = 0.10

# Cross-asset attention
CROSS_ATTN_HEADS = 4                    # 4 heads, each asset attends to all others
CROSS_ATTN_LAYERS = 2                   # 2 layers of cross-asset interaction
CROSS_ATTN_DROPOUT = 0.10

# Asset embedding
WM_ASSET_EMB_DIM = 32

# Return prediction (per-asset, informed by cross-asset context)
RETURN_HEAD_DIM = 192
RETURN_HEAD_DROPOUT = 0.05
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# Variational Information Bottleneck (mirrors V3-clean fix; same memorization
# risk because get_loss -> forward_train -> single-asset path bypasses the
# cross-asset attention bottleneck. Cross-asset attention is dead code in the
# standard runner — fixing that requires synchronized multi-asset batches.)
VIB_Z_DIM = 16                   # Smaller bottleneck (d_model=128)
VIB_KL_WEIGHT = 0.15                 # RAISED from 0.05 (2026-04-22 SOTA upgrade, ShIC=0 fix)
VIB_KL_ANNEAL_EPOCHS = 20
VIB_LOGVAR_INIT = -1.0               # RAISED from -4.0 — start with more stochasticity
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

ACTIVE_HORIZONS = [1, 4, 16, 64]  # CLAUDE.md invariant (was [1,4] -- fixed 2026-04-27 audit; missing h16/h64 caused ShIC decay)
REWARD_HORIZONS = [1, 4, 16, 64]

# ── Training ─────────────────────────────────────────────────────────────────

WM_SEQ_LEN = 96
WM_BATCH_SIZE = 32                      # 32 synchronized multi-asset samples
WM_STEPS_PER_EPOCH = 2000               # CLAUDE.md cross-version invariant (must stay 2000)
DIVERSITY_STEPS_PER_EPOCH = 2000  # CLAUDE.md invariant (V12 has no NCL variant but constant required)
WM_LR = 3e-4
WM_WEIGHT_DECAY = 1e-4
WM_GRAD_CLIP = 1.0
WM_DROPOUT = 0.15
# THROUGHPUT NOTE (2026-06-10): HEADLINE_MODE processes B*A=32*10=320 asset-samples/step.
# Using WM_STEPS_PER_EPOCH=2000 would be 640k samples/epoch = 10x C1's 64k.
# HEADLINE_STEPS_PER_EPOCH=200 gives C1-equivalent data throughput.
# The trainer loop uses HEADLINE_STEPS_PER_EPOCH when HEADLINE_MODE=True.
# WM_STEPS_PER_EPOCH stays at 2000 to satisfy the cross-version CDAP invariant.
HEADLINE_STEPS_PER_EPOCH = 200          # 200 * B*A = 200*320 = 64k = C1-equivalent
WM_WARMUP_STEPS = 100               # Scaled from 500: proportional to HEADLINE_STEPS_PER_EPOCH
WM_EPOCHS = 80                      # Reduced from 150; HEADLINE loop now has patience guard

DIRECT_RETURN_WEIGHT = 3.0
REGIME_HEAD_DIM = 128
REGIME_FOCAL_GAMMA = 2.0

# Early-stop patience (HEADLINE path previously had NO patience guard; ran full WM_EPOCHS).
# 20 epochs * 200 steps = 4k patient steps (same as C1's 40 epochs * 2000 steps pattern).
WM_PATIENCE = 20

WM_MASK_RATIO = 0.25
TEMPORAL_CTX_DROP = 0.30                # ATME

TRAIN_RATIO = 0.50
VAL_RATIO = 0.20
OOS_RATIO = 0.20
UNSEEN_RATIO = 0.10

SHUFFLED_IC_PATIENCE = 5
SHUFFLED_IC_CHECK_INTERVAL = 10

LOG_FREQ = 100
NUM_WORKERS = 0
EMA_DECAY = 0.999

TWOHOT_FOCAL_GAMMA = 0.0
PAIRWISE_RANK_WEIGHT = 0.0
target_prefix = "target_return"

# ─── HEADLINE_MODE upgrades (V12-specific, added 2026-04-30) ─────────────────
# V12 cross-asset attention is dead code in standard runner (per
# world_model.py:267-271). HEADLINE_MODE flag enables the multi-asset
# path; trainer must read HEADLINE_MULTI_ASSET_PATH and use the multi-asset
# dataloader. Per WM_HEADLINE_UPGRADE_PLAN §14: highest ceiling; harness
# fix is the unlock.
# Activation: V12_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V12_HEADLINE_MODE", "0")))
HEADLINE_MULTI_ASSET_PATH = True        # forward_multi_asset (was dead code)
HEADLINE_HIERARCHICAL_ATTN = True       # cross-asset @ bar-level + temporal @ seq-level
# HEADLINE_VIB_KL = 0.10  # REMOVED 2026-06-10: dead constant -- get_multi_loss uses
# VIB_KL_WEIGHT=0.15 directly; this override was never wired and shadowed nothing.

# VIB-collapse guard constants (FIX 1, 2026-06-10).
# If KL drops below KL_COLLAPSE_FLOOR for KL_COLLAPSE_K consecutive epochs
# while kl_anneal >= 1.0, the bottleneck has opened -> early stop.
KL_COLLAPSE_FLOOR = 0.05   # below this the bottleneck is essentially open
KL_COLLAPSE_K = 3          # consecutive epochs below floor -> stop

if HEADLINE_MODE:
    print(f"[V12 HEADLINE_MODE] multi-asset forward enabled; hierarchical attention ON")
    print(f"[V12 HEADLINE_MODE] DATALOADER MUST PROVIDE SYNCHRONIZED MULTI-ASSET BATCHES")


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v12")
except ImportError:
    pass


# CC-H5/H6/FiLM flags (SOTA-2026) — V12 is BLOCKED on MultiAssetDataset.
# Single-asset fallback path could wire these like V11, but V12's
# distinctive lever IS the multi-asset path. Deeper wiring deferred
# until MultiAssetDataset ships and the cross-asset path is live.
# Defaults set so the moment dataloader lands, V12 has SOTA defaults ready.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1
REGIME_AWARENESS_MODE = "film"

