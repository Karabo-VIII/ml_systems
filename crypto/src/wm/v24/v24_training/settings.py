"""V24 Settings — TimesNet (Wu et al., ICLR 2023).

Source: Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., Long, M. "TimesNet:
Temporal 2D-Variation Modeling for General Time Series Analysis."
ICLR 2023. arXiv:2210.02186.

Why V24 specifically:
  Crypto has strong cyclical structure (8h funding, 24h UTC, 7d weekly).
  TimesNet detects the dominant periods via FFT, reshapes 1D series into
  a 2D tensor (rows=intra-period position, cols=cycle index), then applies
  inception-style 2D convolutions to capture both relations directly.

CLAUDE.md cross-version invariants enforced (see V22/V23 settings).
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v24" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v24"
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

SUPPORTED_FEATURE_COUNTS_V24 = (13, 18, 25, 29, 34, 37, 41, 51, 121, 127, 133, 154, 161)
FEATURE_LIST = FEATURE_LIST_29
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V24:
        raise ValueError(
            f"V24 supports {sorted(SUPPORTED_FEATURE_COUNTS_V24)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# === Assets ===
NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# === TimesNet architecture ===
# Sized for iron-clad floor. Inception 2D conv params scale linearly with
# inception_channels^2 * num_kernels. Bumped channels 32->96 for capacity.
# Probed: d=256 / blocks=4 / inception=96 / k_sizes=(1,3,5,7) → 4-5M params.
WM_D_MODEL = 256
N_BLOCKS = 6                 # Stacked TimesBlocks (paper default 2-3, we use 6 for >4M iron-clad floor)
TOP_K_PERIODS = 4            # FFT top-K (paper default 3-5)
INCEPTION_CHANNELS = 128     # 2D conv channels (was 32; capacity bump)
INCEPTION_KERNELS = (1, 3, 5, 7)   # Multi-scale 2D kernels
WM_DROPOUT = 0.15
WM_ASSET_EMB_DIM = 32

# === VIB (Variational Information Bottleneck) — round-4 anti-fragile fix ===
# TimesNet's FFT period detection is regime-dependent: detected periods may
# memorize bull-cycle / weekend-effect patterns and fail to generalize OOS.
# VIB adds stochastic compression on h_seq before return heads, matching the
# V11/V14 pattern.
VIB_Z_DIM = 32
VIB_KL_WEIGHT = 0.05
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
# 2026-05-29: supervise only the last bar of each window. TimesNet's 2D-inception
# is non-causal (symmetric padding), so per-bar supervision leaks future-cycle info
# into mid-window predictions; the last bar's window is entirely its past -> causal.
USE_LAST_BAR_SUPERVISION = True
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
RECON_WEIGHT = 0.0
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
    _assert_canonical(globals(), version_name="v24")
except ImportError:
    pass


# CC-H5 quantile heads (SOTA-2026): TimesNet's FFT-period discovery is a
# natural fit for distributional output — periods in FFT carry uncertainty
# that quantile heads can express explicitly.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V24 TimesNet: FiLM after TimesBlock stack).
REGIME_AWARENESS_MODE = "film"
