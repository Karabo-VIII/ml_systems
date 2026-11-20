"""V23 Settings — xLSTM (Beck et al., NeurIPS 2024).

Source: Beck, M. et al. "xLSTM: Extended Long Short-Term Memory."
NeurIPS 2024. arXiv:2405.04517.

Why V23 specifically:
  Recurrent SOTA alternative to V6's GRU JEPA. xLSTM closes the capacity
  gap with transformers via:
    - Exponential gating (replaces sigmoid; allows revising past memory)
    - Matrix memory (mLSTM block — parallel formulation, transformer-like throughput)
    - Stabilized state via normalization on cell state
  Compute is linear in T, cheap to train.

CLAUDE.md cross-version invariants enforced (see V22 settings).
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v23" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v23"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = True

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

BASE_DIM = 34
XD_DROPOUT_RATE = 0.85  # SOTA-2026: was 0.7; CC-H4 anti-mem
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_V23 = (13, 18, 25, 29, 34, 37, 41, 51, 121, 127, 133, 154, 161)
FEATURE_LIST = FEATURE_LIST_29
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V23:
        raise ValueError(
            f"V23 supports {sorted(SUPPORTED_FEATURE_COUNTS_V23)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# === Assets ===
NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# === xLSTM architecture ===
# Sized for iron-clad floor + VRAM budget. The binding constraint is mLSTM's
# matrix C state [B, d_v, d_v] stored per-timestep (96) for backward.
# At d_model=256/d_value=256: C = 8.4MB * 3 mLSTM layers * 96 steps ≈ 2.4GB
# At d_model=320/d_value=128: C = 2.1MB * 3 mLSTM layers * 96 steps ≈ 600MB
# The smaller d_value (paper standard ~64-128) is a key/value bottleneck.
# Probed: d=320 / dv=128 / L=6 alt → 4.4M params, peak VRAM ~4.5GB.
WM_D_MODEL = 320             # was 256 (capacity bump for iron-clad floor)
WM_N_LAYERS = 8              # Alternating sLSTM + mLSTM (paper Table 1 default; bumped 6->8 for >4M floor)
BLOCK_PATTERN = "alternate"
WM_DROPOUT = 0.15
WM_ASSET_EMB_DIM = 32
MLSTM_DV = 128               # Value dim for mLSTM (key/value bottleneck, VRAM cap)

# === VIB (Variational Information Bottleneck) — round-4 anti-fragile fix ===
# xLSTM's mLSTM matrix memory has unbounded storage capacity (paper §3.2:
# "matrix memory and a covariance update rule to enhance storage capacity").
# Without a stochastic compression bottleneck, the model can memorize training-
# set covariance directly. VIB adds the bottleneck on h_seq before return heads,
# matching V11/V14 pattern.
VIB_Z_DIM = 32
# RAISED 0.05 -> 0.15 (2026-06-10 anchor completion). 0.15 is V12's proven VIB KL
# weight (V12 raised it from 0.05 for the same ShIC=0 memorization fix). V23's HEADLINE
# graft anchors the latent with recon (RECON_WEIGHT=0.5) AND the VIB rate-cap; at 0.05
# the bottleneck barely constrains capacity (recon alone is a weaker anchor). get_loss
# gates the KL term as `VIB_KL_WEIGHT * kl_anneal` (world_model.py:520), so this is the
# live knob -- the KL_WEIGHT_INITIAL/FINAL below are legacy RSSM constants NOT consumed
# by V23's get_loss (the VIB bottleneck is the V23 KL path).
VIB_KL_WEIGHT = 0.15
VIB_KL_ANNEAL_EPOCHS = 20
VIB_LOGVAR_INIT = -1.0
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === Output heads ===
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
REGIME_HEAD_DIM = 128

# === TwoHot Symlog ===
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# === Cross-version invariants ===
ACTIVE_HORIZONS = [1, 4, 16, 64]
REWARD_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]
WM_SEQ_LEN = 96
WM_BATCH_SIZE = 32
WM_STEPS_PER_EPOCH = 2000
DIVERSITY_STEPS_PER_EPOCH = 2000
DIRECT_RETURN_WEIGHT = 3.0
TWOHOT_FOCAL_GAMMA = 0.0
target_prefix = "target_return"
TEMPORAL_CTX_DROP = 0.15

# === Training schedule ===
WM_LR = 3e-4
WM_WEIGHT_DECAY = 1e-4
WM_GRAD_CLIP = 1.0
WM_WARMUP_STEPS = 500
WM_EPOCHS = 150
WM_MASK_RATIO = 0.25
SHUFFLED_IC_PATIENCE = 5
SHUFFLED_IC_CHECK_INTERVAL = 10
LOG_FREQ = 100
NUM_WORKERS = 0
EMA_DECAY = 0.999

TRAIN_RATIO = 0.50
VAL_RATIO = 0.20
OOS_RATIO = 0.20
UNSEEN_RATIO = 0.10

REC_LOG_VAR_CLAMP_MIN = 0.5
# Reconstruction anchor weight (HEADLINE graft, 2026-06-10). V23 had a REAL VIB on
# h_seq but recon was a torch.zeros stub at RECON_WEIGHT=0 -> the bottleneck was a
# pass-through label-fit (the V22/V25 memorization trap: high contiguous IC, ShIC~0).
# A masked recon-MSE off the bottlenecked VIB latent (world_model.recon_decoder) forces
# the latent to retain input-reconstructable structure. Set >0 to engage the anchor;
# 0.0 leaves the base path byte-for-byte unchanged (decoder not run, rec=0.0). 0.5 is a
# regularizing weight (recon is an anchor, not the primary objective).
RECON_WEIGHT = 0.5
# NOTE: KL_WEIGHT_INITIAL/FINAL/KL_ANNEAL_EPOCHS are LEGACY RSSM-template constants and
# are NOT consumed by V23's get_loss. The V23 KL bottleneck is the VIB path, gated by
# VIB_KL_WEIGHT (=0.15) * kl_anneal in world_model.get_loss (line ~520). Do not read these
# zeros as "the bottleneck is OFF" -- the VIB rate-cap is ON via VIB_KL_WEIGHT above.
KL_WEIGHT_INITIAL = 0.0
KL_WEIGHT_FINAL = 0.0
KL_ANNEAL_EPOCHS = 0
PAIRWISE_RANK_WEIGHT = 0.0


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v23")
except ImportError:
    pass


# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V23 xLSTM: FiLM after the xLSTM stack, before heads).
REGIME_AWARENESS_MODE = "film"
