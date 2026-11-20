"""V25 Settings — Frontier Crypto WM (first-principles synthesis).

Round-6 commit (2026-05-07). Designed under the unconstrained-default-synthesis
protocol (memory/feedback_unconstrained_default_synthesis.md). No paper-aligned
configurations; every choice traces to a first-principles regime argument.

🟢 STRUCTURAL CAUSALITY FIX SHIPPED 2026-05-21 (Path A — Timer-XL/TimesFM pattern,
inherited from V22 fix). Same fix as V22: USE_LAST_BAR_SUPERVISION=True
(default). Encoder processes 96 bars of CONTEXT; only the LAST bar's
prediction is supervised. Inference protocol unchanged. Sources: Timer-XL
(ICLR 2025), TimesFM (Das et al arXiv:2310.10688).

Five-component stack (each with a regime-specific justification):

  1. Channel-tokenized cross-feature attention (cross-asset without sync)
  2. Hard-coded crypto period embeddings (8h / 24h / 7d / 30d cycles)
  3. Hurst-regime conditioned FFN (per-bar bull/sideways/bear gating)
  4. Rate-budget VIB (auto-tuned β to bits-per-timestep target)
  5. Tail-adaptive Huber + adversarial regime upweighting

CLAUDE.md cross-version invariants enforced.
"""
import math
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "chimera" / "dollar"  # v51 (migrated 2026-05-17 per connector_integrity_crawler A1)
MODEL_DIR = PROJECT_ROOT / "models" / "wm" / "v25" / "base"
LOG_DIR = PROJECT_ROOT / "logs" / "v25"
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

SUPPORTED_FEATURE_COUNTS_V25 = (13, 18, 25, 29, 34, 37, 41, 51, 121, 127, 133, 154, 161)
FEATURE_LIST = FEATURE_LIST_29
INPUT_DIM = len(FEATURE_LIST)


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_V25:
        raise ValueError(
            f"V25 supports {sorted(SUPPORTED_FEATURE_COUNTS_V25)}; got f{n_features}"
        )
    return _central_get_feature_config(n_features)


# === Assets ===
NUM_ASSETS = 10
ASSET_LIST = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT",
]
ASSET_TO_IDX = {name: idx for idx, name in enumerate(ASSET_LIST)}

# === V25 Frontier architecture ===
# d_model: 320 (matches V22 round-5; depth+regime-gating amplifies effective capacity).
# n_layers: 6 (UN-constrained: paper says ≤4, but we have rate-budget VIB +
#              regime gating that constrains effective capacity regardless of
#              depth, making the depth-vs-overfit trade-off paper-irrelevant).
WM_D_MODEL = 320
WM_N_HEADS = 8
WM_N_LAYERS = 6
# 2026-05-10 fix: dropout up 0.15 -> 0.25 (V25-specific; cross-version
# invariants do not include WM_DROPOUT). Higher dropout pressures the model
# to find redundant signal vs memorize a single path.
WM_DROPOUT = 0.25
WM_ASSET_EMB_DIM = 32
USE_ASSET_TOKEN = True

# Phase 14.7 fix (2026-05-09): Patch embedding replaces Linear(seq_len, d_model)
# memorization vector. Same fix as V22 (per V22_MEMORIZATION_ROOT_CAUSE_2026_05_08
# + NON_V1_MODELS_CRITICAL_AUDIT_2026_05_08). Each feature's 96-bar window splits
# into N_PATCHES × PATCH_LEN bars; shared Linear(PATCH_LEN, PATCH_DIM) per patch.
# Memorization capacity: 30,720 → 520 params per feature (60× reduction).
USE_PATCH_EMBEDDING = True
PATCH_LEN = 12                       # 12-bar patches
N_PATCHES = 96 // PATCH_LEN          # 8 patches per feature
PATCH_DIM = WM_D_MODEL // N_PATCHES  # 40-dim per patch token; preserves d_model
EMBED_SPECTRAL_NORM = True           # bound largest singular value of embed Linear

# Upstream VIB — DISABLED post-Phase-14.8 root-cause probe.
#
# History: Phase 14.7 added input_vib BEFORE the transformer layers per
# V22_MEMORIZATION_ROOT_CAUSE recommendation ("move VIB upstream"). Probe
# revealed two issues:
#   (1) z_dim was 320 (=WM_D_MODEL) → false bottleneck (no compression).
#       Fixed to 32 in initial Phase 14.8 patch.
#   (2) Even with z_dim=32, logvar_init=-1.0 injects σ≈0.6 noise BEFORE
#       transformer layers — destroys signal during training.
#
# V25 negative-IC probe results (200 steps, synthetic +1 correlation target):
#   Full V25 + my fixes:   IC +0.031 (model can't learn even trivial linear task)
#   patches off:           IC +0.107
#   input_vib off:         IC +0.622  ← culprit
#   both off:              IC +0.893
#
# Decision: DEFAULT input_vib OFF. PatchTST patches alone are the load-bearing
# memorization fix (per Cell D probe). Upstream-VIB was speculative; user can
# re-enable for experiments via flag override.
USE_INPUT_VIB = False                  # was True in Phase 14.7; DISABLED Phase 14.8
INPUT_VIB_Z_DIM = 32                   # if re-enabled, use real compression z_dim
INPUT_VIB_TARGET_RATE_NATS = 4.0       # bar-level VIB target rate

# 2026-05-09 H3 root-cause fix: cross-feature attention defaults OFF.
# Identical bug to V22 — without forecast-loss supervision, cross-feature
# attention learns sign-flipped representations. V22 empirical proof
# (12-epoch + ShIC):
#   ON  : ic1=-0.10  ic16=-0.39  ic64=-0.53  ShIC=0.0001  val_loss=35.8
#   OFF : ic1=+0.21  ic16=+0.64  ic64=+0.60  ShIC=0.0000  val_loss=23.3
# V25 inherits the same architectural family. Re-enable only when a
# forecast-head supervision is added — see
# docs/V22_V25_FORECAST_HEAD_PROPOSAL_2026_05_09.md.
USE_CROSS_FEAT_ATTN = False

# Round-9 F1: input regularization on patches (defensive layer)
INV_EMBED_INPUT_DROPOUT = 0.20
INV_EMBED_INPUT_NOISE = 0.10

# === Hard-coded crypto period embeddings ===
# Periods specified in BARS for our 96-bar dollar-bar window. The dollar-bar
# regime varies — at full liquidity, ~96 bars covers ~24h for BTC, more for
# illiquid alts. We use multi-scale periods that span typical regimes.
# These are KNOWN exogenous cycles, not discovery problems.
#
#   8 bars  ≈ funding cycle (8h on Binance perp at ~1h cadence)
#   24 bars ≈ daily UTC cycle
#   96 bars ≈ full window (weekly approximation at 4h cadence)
#   168 bars ≈ weekly at 1h cadence (with padding via embedding lookup)
PERIOD_BARS = (8, 24, 96, 168)
# 2026-05-10 ablation α: PERIOD_AMP_INIT=0.0 disables the per-T deterministic
# scalar that gets added to every sample. With ic1=+0.32 contiguous but
# ShIC=+0.0001, the period scalar is the prime suspect for the position
# channel: on shuffled data, period_per_t becomes pure noise relative to
# bar order, but the model has learned to rely on it.
PERIOD_AMP_INIT = 0.0                 # was 0.1; pinned to 0 to ablate the channel
# 2026-05-10 ablation γ: full period_emb ablation. AMP_INIT=0 only zeros the
# initial scalar — period_emb.proj still trains and can revive position info
# via gradient pressure if any downstream layer benefits. USE_PERIOD_EMB=False
# bypasses the entire call site (h_seq stays unmodified). Toggle ON only
# after positional supervision (calendar/funding-cycle) is added explicitly.
USE_PERIOD_EMB = False                # was implicit-True; now full-ablated by default

# === Regime gating ===
# 3-way regime: bull / sideways / bear, detected per-bar from h_seq.
N_REGIMES = 3
REGIME_GATE_HIDDEN = 64
# 2026-05-10 fix: regime-conditioned FFN is the prime suspect for memorization
# at ShIC=0. Each layer has 3 specialist FFNs weighted by a soft regime gate;
# without supervision, the gate learns to memorize sequence position via the
# regime distribution. Disabling reverts to a SINGLE vanilla FFN per layer.
# Toggle ON only after a properly-supervised regime classifier exists.
USE_REGIME_FFN = False
# 2026-05-10 variant β: spectral norm on the rank-collapsing matrices.
# Cross-instance probe (docs/V25_MEMORIZATION_DIAGNOSIS_2026_05_10.md) shows
# proj.weight at sr_frac=0.035-0.05 (3-5% of capacity) and the active
# regime_ffn[0] Linears at sr_frac=0.026-0.048. Both are primary memorization
# channels surviving the regime-FFN-off fix. Mirroring patch_embed treatment.
PROJ_SPECTRAL_NORM = True
ACTIVE_FFN_SPECTRAL_NORM = True

# === Rate-budget VIB ===
# Information-theoretic: target X nats/timestep cap on I(input; z). β auto-tuned
# via Lagrangian to hit the rate target. No β cargo-cult.
VIB_Z_DIM = 32
VIB_TARGET_RATE_NATS = 4.0           # ~5.8 bits/timestep cap (information-rich
                                     # but not unbounded; gives anti-memo cap)
VIB_BETA_INIT = 0.05
VIB_BETA_LR = 1e-3                   # Lagrangian update rate
VIB_BETA_MIN = 1e-4
VIB_BETA_MAX = 1.0
VIB_LOGVAR_INIT = -1.0
VIB_LOGVAR_MIN = -6.0
VIB_LOGVAR_MAX = 2.0

# === Tail-adaptive Huber ===
# For crypto's heavy-tailed returns (kurtosis 5-15), standard Huber under-weights
# tails. We upweight |target| > tail_threshold * std(target) to ensure tail
# events are actually fit, not averaged out.
HUBER_DELTA = 0.5
TAIL_THRESHOLD_SIGMA = 2.0           # σ-multiplier for "tail" classification
TAIL_WEIGHT = 2.5                    # Multiplicative loss weight for tail samples

# === Adversarial regime training ===
# Per-batch, identify the regime distribution; upweight loss for samples in
# the worst-quintile regime occurrence. Trains against worst-case regime,
# not average. Anti-fragile by construction.
ADVERSARIAL_REGIME_WEIGHT = 1.5      # Multiplicative weight on worst-quintile regime
ADVERSARIAL_REGIME_QUANTILE = 0.20   # Worst 20% regime distribution

# === Output heads ===
RETURN_HEAD_DIM = 256
RETURN_HEAD_DROPOUT = 0.05
REGIME_HEAD_DIM = 128

# === TwoHot Symlog (CLAUDE.md invariant) ===
NUM_BINS = 255
BIN_MIN = -1.0
BIN_MAX = 1.0

# === Cross-version invariants (CLAUDE.md) ===
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
RECON_WEIGHT = 0.0
KL_WEIGHT_INITIAL = 0.0
KL_WEIGHT_FINAL = 0.0
KL_ANNEAL_EPOCHS = 0
PAIRWISE_RANK_WEIGHT = 0.0

# === Forecast head (2026-05-10 attempt; PRECAUTIONARY revert pending V25 validation) ===
# Same fix as V22 (mirrors V22 settings.py). V25 inherits the no-anchor problem
# from the iTransformer base.
#
# 2026-05-10 V22 EMPIRICAL REGRESSION:
#   V22 with same wiring + USE_FORECAST_HEAD=True ran 3-ep CUDA validation
#   and produced SIGN-FLIP REGRESSION (ic1 -0.15 → -0.18 across epochs;
#   IC(-pred) +0.18; pred std collapsed to 0.18x of real). V22 was reverted
#   to USE_FORECAST_HEAD=False.
#
# V25 has additional defenses V22 lacks (spectral norms on proj +
# regime_ffn[0], dropout 0.25, period_emb full-ablated, tail-Huber + adv
# regime), which COULD prevent the same sign-flip basin. But V25 has the
# SAME architectural family (iTransformer with cross_feat_attn=False) and
# SAME structural root cause (no proper reconstruction decoder; recon is
# torch.zeros placeholder). Risk of same regression is real.
#
# Precautionary revert: USE_FORECAST_HEAD=False. Re-enable for empirical
# A/B testing; if 3-ep validation shows clean positive IC1, ship. If
# sign-flips, the fix is the same as V22 (Path A: enable cross_feat_attn=True
# alongside; Path B: add proper recon decoder). See
# docs/V22_V25_VALIDATION_FINDINGS_2026_05_10.md.
USE_FORECAST_HEAD = False
FORECAST_WEIGHT = 0.1         # weight if re-enabled; lower than V22 default (0.5) due to deeper net

# 2026-05-21: SOTA causality fix (Timer-XL ICLR 2025 / TimesFM Das et al pattern,
# same as V22). Supervise only the LAST bar of each window. The encoder still
# processes 96 bars of CONTEXT; only the last bar's prediction is supervised.
# By construction, the last bar has no future-bar leak. Inference protocol is
# unchanged (model is fed last 96 bars, asked for next return) so deployment
# behavior is identical. Default True; set False only for legacy reproduction.
USE_LAST_BAR_SUPERVISION = True

# === Diagnostic richness (axis 3 — info-max) ===
# Per-epoch autopsy log includes: regime distribution, KL trajectory,
# β trajectory, tail-vs-mean loss split, per-regime IC, ShIC.
AUTOPSY_PER_REGIME_BREAKDOWN = True
AUTOPSY_BETA_LOG_EVERY_EPOCH = True
AUTOPSY_RATE_TARGET_TRACKING = True


# Cross-version invariant gate (see src/wm/_shared/invariants.py)
try:
    import sys as _sys
    from pathlib import Path as _Path
    _shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in _sys.path:
        _sys.path.insert(0, _shared_path)
    from invariants import assert_canonical as _assert_canonical
    _assert_canonical(globals(), version_name="v25")
except ImportError:
    pass


# CC-H5 quantile heads (SOTA-2026): auxiliary distributional output.
# V25 has rich existing regime infrastructure (regime_ffn ModuleList);
# CC-H5/H6 are COMPLEMENTARY auxiliary heads, not replacements.
USE_QUANTILE_HEADS = True
QUANTILE_LOSS_WEIGHT = 0.1

# CC-H6 regime-conditional heads (SOTA-2026).
# Note: V25 already has regime_ffn[3] internal to the encoder (gated by
# USE_REGIME_FFN flag, default False per V25 deep audit). CC-H6 is a
# SEPARATE decoder-side mechanism — adds per-regime return decoders
# AFTER the encoder. Not redundant with regime_ffn (which is encoder-side).
USE_REGIME_COND_HEADS = True
REGIME_COND_WEIGHT = 0.1

# Regime-awareness depth (V25: FiLM after bar_proj, before VIB).
REGIME_AWARENESS_MODE = "film"
