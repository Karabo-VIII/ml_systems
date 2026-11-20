"""
V13 Settings -- Temporal Fusion Transformer (TFT)
===================================================

Google's time-series SOTA with variable selection networks.
Learns which features matter at each timestep (not equal weighting).

Key innovation: per-timestep learned feature gates
  "at this bar, VPIN and flow matter; ignore the rest"

Architecture:
  Variable Selection Network -> GRN encoding -> Temporal Self-Attention
  -> Gated Residual Network decoding -> Multi-horizon heads

No RSSM, no reconstruction, no dream. Same get_loss interface as V1.x.
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # 2026-05-03 fix: was 4 .parents (=src/), need 5 (=repo root)
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)  # 2026-05-10 fix: data is under /dollar/ subdir; matches V4/V22/V25
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v13" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v13"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = True

# Features
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

SUPPORTED_FEATURE_COUNTS_V13 = (13, 25, 29, 34, 41, 51, 121, 127, 133, 154, 161)

# ─── V13 STATUS: UNFROZEN 2026-04-29 ──────────────────────────────────────────
# V13 (TFT — Temporal Fusion Transformer) was previously marked
# FROZEN/deprecated due to settings drift after the 2026-04-27 feature-set
# centralization pass. As of 2026-04-29 the architecture block is restored
# (sourced from commit 3c54d26 "V11-V14 SOTA surgical fixes: VIB +
# hard-top-k") so `python -m src.wm.v13.v13_training.train_world_model`
# imports cleanly. ACTIVE_HORIZONS / WM_BATCH_SIZE etc. follow the current
# cross-version invariants from config/_invariants.yaml (NOT the [1, 4]
# horizons of 3c54d26 -- those were superseded).
# Pre-train edge validation pending; do not ship without IC > 0 walk-forward.

FEATURE_LIST = FEATURE_LIST_25
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V13:
        raise ValueError(
            f"V13 supports {sorted(SUPPORTED_FEATURE_COUNTS_V13)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)



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

# Cross-version training invariants (added 2026-04-29 -- previously the v13
# settings.py was deliberately slim and `from settings import *` left these
# undefined, which broke train_world_model at runtime). Source of truth:
# config/_invariants.yaml::cross_version_constants.
WM_BATCH_SIZE = 32
WM_STEPS_PER_EPOCH = 2000
DIVERSITY_STEPS_PER_EPOCH = 2000
DIRECT_RETURN_WEIGHT = 3.0
BIN_MIN = -1.0
BIN_MAX = 1.0
NUM_BINS = 255
ACTIVE_HORIZONS = [1, 4, 16, 64]
REWARD_HORIZONS = [1, 4, 16, 64]

# ─── Architecture block (restored from 3c54d26, unfrozen 2026-04-29) ─────────
# 10-asset universe (matches V1.x; legacy chimera is u10).
NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# TFT architecture
# 2026-05-07 capacity bump (was 2.21M params, below 4M iron-clad floor):
# - WM_D_MODEL 256->320 (wider attention working dim)
# - TFT_N_LAYERS 2->6 (Google reference TFT uses 1-4 LSTM + multi-head attn;
#   crypto-bar regime benefits from deeper stack to compose multi-scale patterns;
#   first try at 4 layers landed at 3.37M, still below floor)
# - TFT_GRN_HIDDEN 256->512 (richer gated residual capacity)
# - TFT_VSN_HIDDEN 64->96 (stronger feature-selection gates)
# Anti-memorization: VSN_TOP_K=8 hard gate + ATME 0.30 retained (no RSSM/VIB
# in V13 design; future work could add a VIB on h_seq before return heads).
WM_D_MODEL = 320           # was 256
TFT_N_HEADS = 4
TFT_N_LAYERS = 6           # was 2
TFT_DROPOUT = 0.10
TFT_VSN_HIDDEN = 96        # was 64 (Variable Selection Network hidden dim)
TFT_GRN_HIDDEN = 512       # was 256 (Gated Residual Network hidden dim)
VSN_TOP_K = 8              # Hard top-k feature gate (ShIC=0 fix)
WM_ASSET_EMB_DIM = 32

# Heads
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
REGIME_HEAD_DIM = 128

# === TFT-native quantile loss (2026-05-10 cohort upgrade) ===
# Replace TwoHot CE return head (255 bins, designed for V1.x RSSM family) with
# TFT-native pinball loss on quantile predictions. TFT was designed for quantile
# regression in the original Google paper; the inherited TwoHot CE leaves signal
# on the table and adds a discretization bottleneck. Pinball loss computes one
# loss per quantile: L_q(y, y_pred) = max(q*(y - y_pred), (q-1)*(y - y_pred))).
# IC computation uses the median quantile (q=0.5) prediction.
# Existing V13 ckpts will have shape mismatch on return_heads — strict=False
# load + fresh training. V13 had no live ckpts at session start.
# 2026-05-10 EMPIRICAL REVERT: 10-ep gauntlet (b71wkkp1x) Ep 1 VAL showed
#   ic1=-0.0019 ic4=-0.0026 ic16=-0.0001 ic64=+0.0048  -- IC≈0 across all
# horizons. Ep 2 r1=0.000 (pinball loss collapsed to zero). Mechanism:
# pinball loss on median quantile (q=0.5) incentivizes predicting the
# empirical median of the target distribution; for centered crypto returns
# the median is ≈ 0; model collapses to "predict zero" which minimizes
# pinball but gives zero IC by construction. TwoHot CE doesn't have this
# issue (CE incentivizes mass at the correct bin, not at the median).
# The cohort plan's "+0.05-0.08 IC lift from quantile loss" projection
# empirically does NOT hold for crypto returns. Reverted to TwoHot CE.
# Re-enable only if a different quantile loss formulation (e.g. CRPS,
# weighted quantile loss with non-uniform quantiles) is implemented.
USE_QUANTILE_LOSS = False
QUANTILES = (0.1, 0.5, 0.9)        # if re-enabled
QUANTILE_LOSS_WEIGHT = 1.0          # if re-enabled

# Training schedule
WM_SEQ_LEN = 96
WM_LR = 3e-4
WM_WEIGHT_DECAY = 1e-4
WM_GRAD_CLIP = 1.0
WM_DROPOUT = 0.15
WM_WARMUP_STEPS = 500
WM_EPOCHS = 150
WM_MASK_RATIO = 0.25
# Round-4: ATME drop reduced 0.30 -> 0.15 to match CLAUDE.md cross-version
# invariant (per-sample 0.15 is the V1.6/RSSM standard). The 0.30 batch rate
# from 3c54d26 was empirically tuned for a TFT without bottleneck; with VIB
# now in place the per-sample 0.15 is the right anti-fragile pairing.
TEMPORAL_CTX_DROP = 0.15

# === VIB (Variational Information Bottleneck) — round-4 anti-fragile fix ===
# V13 was the only V-version with NO bottleneck (pre-round-4) — an outlier
# from CLAUDE.md's anti-fragile mandate (RSSM in V1-V8, VIB in V11/V14).
# VSN provides feature selection but no temporal-capacity constraint, so the
# TFT was effectively unbounded. VIB adds the missing stochastic compression.
VIB_Z_DIM = 32
VIB_KL_WEIGHT = 0.05
VIB_KL_ANNEAL_EPOCHS = 20
VIB_LOGVAR_INIT = -1.0
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === RECON ANCHOR (keystone, 2026-06-10) ====================================
# V13 had a VIB latent but recon=torch.zeros stub + rec=0.0 + RECON_WEIGHT
# effectively absent -> the bottleneck was a pass-through (no input-
# reconstruction pressure, so the heads could route around z). RECON_WEIGHT>0
# turns on the masked recon-MSE term that forces the 32-dim VIB latent to retain
# input-reconstructable structure. Unit weight (1.0) matches the V12 donor's
# fixed unit-weight recon (the single-asset MSE, not Kendall-weighted). This is
# the missing HALF of the anti-fragile anchor (recon + VIB KL together).
RECON_WEIGHT = 1.0

# ─── HEADLINE_MODE upgrades (V13-specific, added 2026-04-30) ─────────────────
# V13 TFT: VSN_TOP_K bump from 8 to 12-16; cross-asset VSN layer.
# Per WM_HEADLINE_UPGRADE_PLAN §15.
# Activation: V13_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V13_HEADLINE_MODE", "1")))  # SOTA-2026: default ON (VSN_TOP_K → 16; cross-asset VSN ON). Legacy: V13_HEADLINE_MODE=0
HEADLINE_VSN_TOP_K = 16              # was 8; less aggressive bottleneck
HEADLINE_CROSS_ASSET_VSN = True      # new: asset-level variable selection
if HEADLINE_MODE:
    # Override the existing VSN_TOP_K
    VSN_TOP_K = HEADLINE_VSN_TOP_K
    print(f"[V13 HEADLINE_MODE] VSN_TOP_K -> {VSN_TOP_K}; cross-asset VSN ON")


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v13")
except ImportError:
    pass


# CC-H5 quantile heads (SOTA-2026): TFT-native quantile architecture would
# REPLACE TwoHot but USE_QUANTILE_LOSS was REVERTED 2026-05-10 due to
# regression. Keep auxiliary q-heads as CC-H5 add-on (q05..q95 alongside
# TwoHot, NOT replacing it). Safe; if TFT's quantile output proves better,
# the trainer can flip primary path later.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V13: FiLM after VSN+GRN encoder, before attention).
REGIME_AWARENESS_MODE = "film"

