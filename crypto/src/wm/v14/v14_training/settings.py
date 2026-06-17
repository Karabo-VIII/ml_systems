"""
V14 Settings -- Diffusion Return Distribution Model
=====================================================

Generates the FULL DISTRIBUTION of possible returns, not a point estimate.
Position sizing from distribution SHAPE, not just mean.

Key innovation: "returns could be [-2%, -1%, +0.5%, +3%, +5%] with these
probabilities" -> informed position sizing based on risk/reward ratio.

Architecture:
  WaveNet encoder -> condition embedding
  Diffusion denoiser: learns P(return | condition) via iterative denoising
  Inference: sample N return scenarios, compute mean/std/percentiles

Dual path: TwoHot (VIB-bottlenecked) + Diffusion denoiser (direct conditioning).
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]  # 2026-05-03 fix: was 4 .parents (=src/), need 5 (=repo root)
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 migration 2026-05-25
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v14" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v14"

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

SUPPORTED_FEATURE_COUNTS_V14 = (13, 25, 29, 34, 41, 51, 121, 127, 133, 154, 161)

# ─── V14 STATUS: UNFROZEN 2026-04-29 ──────────────────────────────────────────
# V14 (Diffusion Return Distribution) was previously marked FROZEN/deprecated
# due to settings drift after the 2026-04-27 centralization pass. As of
# 2026-04-29 the architecture block is restored (sourced from commit 3c54d26
# "V11-V14 SOTA surgical fixes: VIB + hard-top-k") so
# `python -m src.wm.v14.v14_training.train_world_model` imports cleanly.
# Pre-train edge validation pending; do not ship without IC > 0 walk-forward.

FEATURE_LIST = FEATURE_LIST_25
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V14:
        raise ValueError(
            f"V14 supports {sorted(SUPPORTED_FEATURE_COUNTS_V14)}; got f{n_features}"
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

# Cross-version training invariants (added 2026-04-29 -- previously the v14
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

# Diffusion architecture
# 2026-05-07 capacity + receptive-field bumps (was 3.04M params with WaveNet
# receptive field of only 31 bars for a 96-bar seq -- half the temporal context
# was invisible to the conditioning encoder):
# - WAVENET_DILATIONS [1,2,4,8] -> [1,2,4,8,16,32] (receptive field 127 bars
#   covers the full 96-bar seq, matches V3/V11)
# - WAVENET_CHANNELS extended 4 -> 6 layers
# - DENOISER_LAYERS 3 -> 6 (matches diffusion SOTA depth)
# - DENOISER_HIDDEN 256 -> 320 (richer denoiser representation)
# Probed: ~5.3M params, ~1.1 GB peak VRAM at B=32.
WM_D_MODEL = 256
DIFFUSION_STEPS = 100              # Denoising steps during training
DIFFUSION_INFERENCE_STEPS = 50     # DDIM-style inference
DIFFUSION_BETA_START = 1e-4
DIFFUSION_BETA_END = 0.02
DIFFUSION_N_SAMPLES = 32           # Return samples per prediction
DENOISER_HIDDEN = 320              # was 256
DENOISER_LAYERS = 6                # was 3
WM_ASSET_EMB_DIM = 32

# WaveNet conditioning encoder (extended to cover full 96-bar receptive field)
WAVENET_CHANNELS = [96, 128, 192, 256, 256, 256]   # was [96,128,192,256] (4 layers)
WAVENET_DILATIONS = [1, 2, 4, 8, 16, 32]           # was [1,2,4,8] (RF 31 bars; now 127 bars)
WAVENET_KERNEL = 3
WAVENET_DROPOUT = 0.10

# Heads
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
REGIME_HEAD_DIM = 128

# Training schedule
WM_SEQ_LEN = 96
WM_LR = 3e-4
WM_WEIGHT_DECAY = 1e-4
WM_GRAD_CLIP = 1.0
WM_DROPOUT = 0.15
WM_WARMUP_STEPS = 500
WM_EPOCHS = 150
WM_MASK_RATIO = 0.25
# TEMPORAL_CTX_DROP intentionally NOT defined for V14 -- diffusion noise
# injection is the regularizer; VIB on TwoHot path handles return memorization.

# Variational Information Bottleneck on TwoHot path (mirrors V3-clean fix).
# 2026-04-22 SOTA upgrade: KL=0.10 (was 0.05), logvar_init=-1 (was -4) to
# fix ShIC=0 collapse at epoch 9.
VIB_Z_DIM = 32
VIB_KL_WEIGHT = 0.10
VIB_KL_ANNEAL_EPOCHS = 20
VIB_LOGVAR_INIT = -1.0
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === RECON ANCHOR (keystone, 2026-06-11) ====================================
# V14 had a REAL Gaussian VIB on the TwoHot path (the ShIC-validated path) but
# recon=torch.zeros stub + rec=0.0 + NO recon term -> the bottleneck had no
# input-reconstruction pressure (the TwoHot heads could route around z; the
# V22/V25 memorization trap: high contiguous IC, ShIC~0). RECON_WEIGHT>0 turns
# on the masked recon-MSE that forces the 32-dim VIB latent to retain input-
# reconstructable structure. Unit weight (1.0) matches the V12 donor + V13 graft.
# The missing HALF of the anchor (recon + VIB KL together).
#
# SCOPE -- this anchors the TWOHOT path ONLY (where ShIC is measured).
# TODO (DDPM-path bottleneck, SEPARATE LARGER CHANGE, deferred 2026-06-11):
#   The headline DDPM denoiser reads `condition` UNBOTTLENECKED
#   (world_model.py forward_train ~:345 and the loss/sample paths ~:604) by
#   design (stable diffusion training). The denoiser therefore still has the
#   memorization escape-hatch the TwoHot path no longer has. Routing the denoiser
#   off the VIB bottleneck (feat) instead of `condition` -- AND/OR adding a recon
#   anchor on the diffusion condition -- is a separate, higher-risk change that
#   must be validated against the diffusion training stability (it changes the
#   conditioning manifold the denoiser was tuned on). NOT done in this graft.
RECON_WEIGHT = 1.0

# ─── HEADLINE_MODE upgrades (V14-specific, added 2026-04-30) ─────────────────
# V14 Diffusion: reduce inference steps (50 -> 10 via DDIM); fewer samples
# (32 -> 8) -> 5x faster inference. Per WM_HEADLINE_UPGRADE_PLAN §16.
# The Headline argument for V14 is distributional Sharpe via meta-learner
# accepting q05/q50/q95 input -- not raw IC.
# Activation: V14_HEADLINE_MODE=1 python ... train_world_model.py
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V14_HEADLINE_MODE", "1")))  # 2026-05-21: re-enabled (DPM-Solver++ 2M wired in world_model.py:_sample_dpmpp_2m). At K=10-15 the multistep order-2 solver approximates DDPM K=50 quality. Set V14_HEADLINE_MODE=0 for legacy DDPM K=50 sampling.
HEADLINE_DIFFUSION_INFERENCE_STEPS = 15     # 15 steps = best quality/speed balance for DPM-Solver++ 2M; K=10 also acceptable
HEADLINE_DIFFUSION_N_SAMPLES = 8            # was 32
HEADLINE_USE_DDIM = True                    # WIRED 2026-05-21: dispatches sample_returns to _sample_dpmpp_2m
HEADLINE_CFG_SCALE = 1.5                    # DEAD FLAG: no unconditional path implemented (V14 is conditional-only)
HEADLINE_QUANTILE_HEAD = True               # output q05/q50/q95 to meta-learner (wired via _shared)
if HEADLINE_MODE:
    DIFFUSION_INFERENCE_STEPS = HEADLINE_DIFFUSION_INFERENCE_STEPS
    DIFFUSION_N_SAMPLES = HEADLINE_DIFFUSION_N_SAMPLES
    print(f"[V14 HEADLINE_MODE] DPM-Solver++ 2M sampler at K={DIFFUSION_INFERENCE_STEPS} steps x {DIFFUSION_N_SAMPLES} samples per bar")


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v14")
except ImportError:
    pass


# CC-H5 quantile heads (SOTA-2026): NATIVE FIT for V14 since DDPM already
# outputs distributional predictions via N_SAMPLES. Quantile-heads add an
# explicit q05/q50/q95 head consumed by meta-learner without requiring the
# expensive DDPM sampling at inference time. The HEADLINE_QUANTILE_HEAD flag
# at §16 of WM_HEADLINE_UPGRADE_PLAN aligns with this.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V14: FiLM after WaveNet condition encoder, before DDPM denoiser).
REGIME_AWARENESS_MODE = "film"

