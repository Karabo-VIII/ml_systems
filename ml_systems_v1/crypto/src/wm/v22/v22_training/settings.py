"""V22 Settings — iTransformer (channel-tokenized cross-feature attention).

Source: Liu et al. ICLR 2024, "iTransformer: Inverted Transformers Are Effective
for Time Series Forecasting" (arXiv:2310.06625).

🟢 STRUCTURAL CAUSALITY FIX SHIPPED 2026-05-21 (Path A — Timer-XL/TimesFM pattern).
The IC=+0.21 / ShIC=0.0 memorization came from the iTransformer [F→T] projection:
each output bar depended on the full 96-bar window via cross-feature attention,
and the per-bar loss supervised bars t<T-1 that structurally saw future bars.

Fix: USE_LAST_BAR_SUPERVISION=True (default). Encoder processes 96 bars of
CONTEXT; only the LAST bar's prediction is supervised. Inference protocol is
identical (model is fed last 96 bars, asked for next return) so deployment
behavior is unchanged. Sources: Timer-XL (ICLR 2025), TimesFM (Das et al
arXiv:2310.10688). See settings.py USE_LAST_BAR_SUPERVISION docstring below.

Why V22 specifically:
  V12's cross-asset attention is structurally blocked because dollar-bar data
  is not synchronized across assets. iTransformer INVERTS the tokenization
  so each feature is a token; cross-feature self-attention runs without any
  timestamp-synchronization requirement. This is the cleanest architectural
  fix for V12's design issue.

CLAUDE.md cross-version invariants enforced (NUM_BINS=255, BIN_MIN/MAX=±1,
WM_BATCH_SIZE=32, WM_STEPS_PER_EPOCH=2000, ACTIVE_HORIZONS=[1,4,16,64],
target_prefix='target_return', TWOHOT_FOCAL_GAMMA=0.0, DIRECT_RETURN_WEIGHT=3.0,
TEMPORAL_CTX_DROP=0.15 per-sample).
"""
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v22" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v22"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_WINDOWS = True

# === FEATURES (centralized in src/feature_sets.py, post-2026-04-27) ===
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
XD_DROPOUT_RATE = 0.85  # SOTA-2026: was 0.7; CC-H4 anti-mem (cohort-aligned)
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_V22 = (13, 18, 25, 29, 34, 37, 41, 51, 121, 127, 133, 154, 161)

FEATURE_LIST = FEATURE_LIST_29
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V22:
        raise ValueError(
            f"V22 supports {sorted(SUPPORTED_FEATURE_COUNTS_V22)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# === Assets (10-asset universe, matches V1.x / chimera_legacy) ===
NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# === iTransformer architecture ===
# Round-4 literature correction: paper (Liu et al. ICLR 2024) tests
# L ∈ {2, 3, 4} and d_model ∈ {256, 512}. The paper explicitly states
# "the number of Transformer blocks and hidden dimension are not essentially
# favored to be as large as possible in iTransformer." Round-3 used L=6
# which is OVER paper range. Reverting to L=4 (top of paper range).
# Param count: ~5.5M at d=320/L=4 (still above 4M iron-clad floor).
WM_D_MODEL = 320             # Token embedding dim (paper range {256, 512})
WM_N_HEADS = 8               # Attention heads (d_model / n_heads = 40 dim/head)
WM_N_LAYERS = 4              # was 6 (round-3) → 4 (round-4, paper top-range)
WM_DROPOUT = 0.15
WM_ASSET_EMB_DIM = 32
USE_ASSET_TOKEN = True       # Prepend asset token to feature tokens

# === VIB (Variational Information Bottleneck) — round-9 rate-budget upgrade ===
# Round-9 (2026-05-08): real-data probe at β=0.20 STILL showed V22 memorizing
# hard (BestIC=0.76, ShIC/IC=0.01). Diagnosis: prediction loss dominates KL by
# ~150x; β tuning can't bind. SOLUTION: switch to V25's RateBudgetVIB —
# auto-tunes β via Lagrangian to hit a fixed bits-per-timestep target, so
# bottleneck binds REGARDLESS of prediction loss magnitude.
# Plus F1 (round-9): inverted-embedding dropout + noise to break 96-bar
# pattern memorization. Plus F3 (round-9): asset-token random drop 30%
# to break "BTC-prior memorization."
VIB_Z_DIM = 32
VIB_KL_WEIGHT = 0.05         # initial β (rate-budget will auto-tune)
VIB_TARGET_RATE_NATS = 4.0   # target ~5.8 bits/timestep cap on I(X;Z)
VIB_BETA_LR = 1e-3           # Lagrangian update rate
VIB_BETA_MIN = 1e-4
VIB_BETA_MAX = 1.0
VIB_KL_ANNEAL_EPOCHS = 20

# Round-9 F1: inverted-embedding regularization (KEPT as additional defense)
INV_EMBED_INPUT_DROPOUT = 0.20    # dropout on patch-embedded input
INV_EMBED_INPUT_NOISE = 0.10       # Gaussian noise std on input

# Round-9 F3: asset-token random-drop probability during training
ASSET_TOKEN_DROP_PROB = 0.30

# Round-10 PATCH-EMBEDDING (per other-instance V22 root-cause diagnosis):
# Replace nn.Linear(seq_len=96, d_model=320) — 30,720 params/feature × 30 features
# = 921,600 params memorization vector — with PatchTST-style patch embedding.
# Per-patch Linear(PATCH_LEN=12, PATCH_DIM=40) shared across 8 patches per
# feature: 520 params/feature × 30 features = 15,600 params (60× reduction).
# Each patch sees only 12 bars, can't memorize 96-bar templates. Per
# Nie et al. ICLR 2023 (arXiv:2211.14730) PatchTST.
USE_PATCH_EMBEDDING = True
PATCH_LEN = 12              # bars per patch

# 2026-05-09 H3 root-cause fix: cross-feature attention defaults OFF.
# Without forecast-loss supervision (V22 has none — recon is dummy zeros),
# cross-feature attention learns sign-flipped representations on real data.
# Empirical proof (V22 12-epoch + ShIC):
#   ON  : ic1=-0.10  ic16=-0.39  ic64=-0.53  ShIC=0.0001  val_loss=35.8
#   OFF : ic1=+0.21  ic16=+0.64  ic64=+0.60  ShIC=0.0000  val_loss=23.3
# Re-enable only when a forecast-head supervision is added (see
# docs/V22_V25_FORECAST_HEAD_PROPOSAL_2026_05_09.md).
USE_CROSS_FEAT_ATTN = False
# Note: WM_SEQ_LEN=96 and WM_D_MODEL=320 are defined later but constants here
# are evaluated at import time. Use literals consistent with above.
N_PATCHES = 96 // PATCH_LEN     # 96 / 12 = 8 patches per feature
PATCH_DIM = 320 // N_PATCHES    # 320 / 8 = 40-dim per patch token

# Round-10 SPECTRAL NORMALIZATION on embedding (Fix #3 from other instance):
# Bounds largest singular value of embedding Linear weight to <= 1.
# Limits Lipschitz constant -> limits expressive memorization capacity.
USE_SPECTRAL_NORM_EMBED = True

# Round-10 INPUT VIB (Fix #2 from other instance):
# Move VIB UPSTREAM — between embedding and transformer layers.
# Constrains embedding output BEFORE transformer can amplify memorized templates.
# Replaces the post-encoder VIB (which fired AFTER memorization had already
# happened in the inverted-attention layers).
USE_INPUT_VIB = True
INPUT_VIB_TARGET_RATE_NATS = 4.0   # same as output VIB target
VIB_LOGVAR_INIT = -1.0       # Mild initial stochasticity (V11/V14 standard)
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === Output heads ===
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
REGIME_HEAD_DIM = 128

# === TwoHot Symlog (CLAUDE.md cross-version invariants) ===
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# === ACTIVE_HORIZONS / REWARD_HORIZONS (CLAUDE.md cross-version invariant) ===
ACTIVE_HORIZONS = [1, 4, 16, 64]
REWARD_HORIZONS = [1, 4, 16, 64]
TARGET_COLUMNS = [f"target_return_{h}" for h in REWARD_HORIZONS]

# === Anti-fragile invariants (CLAUDE.md) ===
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

# === Splits (CLAUDE.md 50/20/20/10) ===
TRAIN_RATIO = 0.50
VAL_RATIO = 0.20
OOS_RATIO = 0.20
UNSEEN_RATIO = 0.10

# === Reconstruction weights (Pattern Q) — V22 has no reconstruction head, kept for trainer compat ===
REC_LOG_VAR_CLAMP_MIN = 0.5
RECON_WEIGHT = 0.0
KL_WEIGHT_INITIAL = 0.0
KL_WEIGHT_FINAL = 0.0
KL_ANNEAL_EPOCHS = 0
PAIRWISE_RANK_WEIGHT = 0.0

# === Forecast head (2026-05-10 attempt; REVERTED after empirical regression) ===
# History:
#   2026-05-10 a: Wired forecast head per docs/V22_V25_FORECAST_HEAD_PROPOSAL_2026_05_09.md
#               (USE_FORECAST_HEAD=True, FORECAST_WEIGHT=0.5).
#   2026-05-10 b: 3-epoch CUDA validation showed SIGN-FLIP REGRESSION:
#                   Ep 1: ic1=-0.151  IC(-pred)=+0.151 (sign-flipped)
#                   Ep 2: ic1=-0.178  IC(-pred)=+0.178 (worse)
#                   Ep 3: ic1=-0.184  IC(-pred)=+0.184 (stable sign-flip)
#                   Pred std=0.00055 vs real std=0.00138 (collapsed; ps/rs=0.40x)
#                   val_loss climbing 43 → 51 → 55 (training getting WORSE)
#               Pre-fix V22 (no forecast head, cross_feat_attn=False): ic1=+0.21
# Verdict: forecast head ALONE doesn't compose with iTransformer cross_feat_attn=False.
# The proposal explicitly recommended re-enabling cross_feat_attn AT THE SAME TIME
# as forecast head; we tested forecast-only (one-knob-at-a-time) and got the
# regression. Two paths forward (next session):
#   Path A: enable both USE_CROSS_FEAT_ATTN=True + USE_FORECAST_HEAD=True (proposal's
#           full spec; cross_feat_attn lets feature interaction supervise sign,
#           forecast loss anchors encoder)
#   Path B: add a proper reconstruction decoder (mirror V1.x style) — replaces the
#           recon=torch.zeros placeholder with a real decoder + RECON_WEIGHT > 0
# Reverted to USE_FORECAST_HEAD=False; V22 returns to +0.21/ShIC=0 baseline.
# See docs/V22_V25_VALIDATION_FINDINGS_2026_05_10.md.
USE_FORECAST_HEAD = False
FORECAST_WEIGHT = 0.5         # weight if re-enabled; tune in [0.1, 1.0]

# 2026-05-16: gate the period_emb add-site (mirrors V25 USE_PERIOD_EMB=False).
# period_emb is hard-coded position information; without an encoder anchor it
# contributes to temporal memorization the same way RevIN did. Default-False
# matches V25's empirically-justified ablation. Re-enable only as part of a
# joint experiment that pairs it with a working anchor.
USE_PERIOD_EMB = False

# 2026-05-21: SOTA causality fix (Timer-XL ICLR 2025 / TimesFM Das et al. pattern).
# iTransformer's [F→T] projection is structurally non-causal: bar t's output
# depends on the entire 96-bar window via mean-of-feature-tokens. With per-bar
# supervision (legacy default), bars t<T-1 leak future bars through the
# attention representation — root cause of the IC=+0.21 / ShIC=0.0 memorization.
#
# The SOTA fix (Timer-XL "encoder + decoder-style next-token supervision"):
# supervise ONLY the last bar of each window. The encoder processes 96 bars
# of CONTEXT and predicts the NEXT bar's return — by construction the last
# bar's prediction has no future-bar leak. ShIC-safe.
#
# Trade-off: 96x fewer supervised positions per sample. The training data
# already uses overlapping windows (window-step=1 via AntifragileDataset),
# so the effective supervision count is preserved.
#
# Default True. Set False only for legacy ablation reproduction.
USE_LAST_BAR_SUPERVISION = True


# ── Cross-version invariant drift gate (2026-05-16) ─────────────────────
# Opt-in import; raises InvariantDriftError if any canonical constant
# diverges from src/wm/_shared/invariants.py. Wrapped in try/except so
# stand-alone smoke tests outside the project tree still import settings.
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v22")
except ImportError:
    pass  # smoke test mode


# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V22: FiLM after bar_proj, before period_emb).
REGIME_AWARENESS_MODE = "film"
