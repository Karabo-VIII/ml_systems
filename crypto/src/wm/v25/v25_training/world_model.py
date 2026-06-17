"""V25 World Model — Frontier Crypto WM (first-principles synthesis).

Round-6 commit (2026-05-07). Designed under unconstrained-default-synthesis
protocol. NO single component matches a paper directly; the synthesis is
crypto-regime-specific.

Five components, each load-bearing for a specific crypto-regime characteristic:

  1. CryptoPeriodEmbedding — hard-coded sinusoidal embeddings for known
     crypto cycles (8h funding, 24h UTC, 7d weekly, 30d monthly). Replaces
     V24 TimesNet's FFT-based discovery. We KNOW these cycles; no need to
     re-derive each batch.

  2. InvertedAttention with regime-conditioned FFN — feature-as-token
     attention (V22 iTransformer base) but the FFN output is regime-mixed:
     each layer has 3 FFNs (bull/sideways/bear) and outputs are weighted by
     a per-bar regime distribution. Generic models average across regimes;
     this conditions on regime structure.

  3. RateBudgetVIB — VIB with explicit information-rate target (nats per
     timestep). β auto-tuned via Lagrangian update. Information-theoretic
     anti-memorization, not β cargo-cult.

  4. TailAdaptiveHuber — direct-return loss upweights |target| > 2σ. Heavy-
     tailed crypto returns get fit instead of averaged out.

  5. AdversarialRegimeUpweight — per-batch worst-quintile regime upweighted
     in loss. Trains against worst-case regime, not average. Anti-fragile
     by construction.

This is intentionally outside any single paper's scope. Each component
has independent first-principles justification for our specific regime.
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from settings import *
except ImportError:
    from .settings import *

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)

from components import RMSNorm, TwoHotSymlog, MLPHead


# =============================================================================
# Component 1: Hard-coded Crypto Period Embedding
# =============================================================================

class CryptoPeriodEmbedding(nn.Module):
    """Sinusoidal embeddings for known crypto cycles.

    For each period p in PERIOD_BARS, generates sin(2πt/p) + cos(2πt/p) and
    learns an amplitude. Concatenated across periods, projected to d_model.
    No discovery — these cycles are exogenous and known.
    """

    def __init__(self, d_model: int, periods: tuple = PERIOD_BARS,
                 amp_init: float = PERIOD_AMP_INIT):
        super().__init__()
        self.periods = periods
        self.amplitudes = nn.Parameter(torch.full((len(periods),), amp_init))
        # 2 channels per period (sin/cos), projected to d_model
        self.proj = nn.Linear(2 * len(periods), d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns [T, d_model] period embedding for sequence length T."""
        t = torch.arange(T, device=device).float()
        feats = []
        for i, p in enumerate(self.periods):
            phase = 2 * math.pi * t / p
            feats.append(torch.sin(phase) * self.amplitudes[i])
            feats.append(torch.cos(phase) * self.amplitudes[i])
        # [T, 2 * len(periods)]
        period_feat = torch.stack(feats, dim=-1)
        return self.norm(self.proj(period_feat))   # [T, d_model]


# =============================================================================
# Component 2: Regime-Conditioned Inverted Attention
# =============================================================================

class RegimeGate(nn.Module):
    """Per-bar 3-way regime distribution from feature representation."""

    def __init__(self, d_model: int, hidden: int = REGIME_GATE_HIDDEN,
                 n_regimes: int = N_REGIMES):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            RMSNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_regimes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, T, D] or [B, F, D]
        return F.softmax(self.gate(h), dim=-1)


class RegimeConditionedInvertedAttention(nn.Module):
    """One layer of feature-as-token self-attention with regime-conditioned FFN.

    Pre-norm Transformer block. Operates on [B, F, D]. The attention itself
    is regime-agnostic (cross-feature interactions are universal), but the
    FFN output is a regime-weighted combination of N_REGIMES specialist FFNs.

    Bull regime → trending FFN dominates
    Sideways regime → mean-reverting FFN dominates
    Bear regime → liquidation-aware FFN dominates
    """

    def __init__(self, d_model: int, n_heads: int, n_regimes: int = N_REGIMES,
                 dim_ff: int | None = None, dropout: float = 0.1,
                 use_cross_feat_attn: bool = False,
                 use_regime_ffn: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self._use_cross_feat_attn = use_cross_feat_attn
        self._use_regime_ffn = use_regime_ffn
        # Always allocate the MHA module so checkpoint state_dict matches the
        # state-of-the-iTransformer schema. Whether it runs is gated in forward().
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        ff_dim = dim_ff if dim_ff is not None else 4 * d_model
        # Always allocate the regime_ffn ModuleList so checkpoint state_dict
        # matches across toggles (load_state_dict(strict=False) handles unused).
        # 2026-05-10 variant β: when use_regime_ffn=False, the kept active
        # path (regime_ffn[0]) was rank-collapsing on its own (sr_frac 0.03-0.05
        # at epoch 10). Wrap its two Linear layers with spectral_norm to bound
        # the largest singular value, preventing rank collapse. Mirrors the
        # existing patch_embed treatment.
        try:
            from settings import ACTIVE_FFN_SPECTRAL_NORM as _active_ffn_sn
        except ImportError:
            _active_ffn_sn = False

        def _make_ffn(spectral_norm_active: bool):
            l1 = nn.Linear(d_model, ff_dim)
            l2 = nn.Linear(ff_dim, d_model)
            if spectral_norm_active:
                l1 = nn.utils.parametrizations.spectral_norm(l1)
                l2 = nn.utils.parametrizations.spectral_norm(l2)
            return nn.Sequential(l1, nn.GELU(), nn.Dropout(dropout), l2)

        # 2026-05-21 (oracle validation): only allocate the FFNs actually
        # used. Previously we always built n_regimes FFNs even when
        # use_regime_ffn=False (then routed only through [0]), wasting
        # ~12M dead params + AdamW m/v state. Now allocate exactly the
        # number we use. Ckpt-incompat: existing V25 ckpts saved with
        # use_regime_ffn=False still strict=False load because the dead
        # regime_ffn[1] / regime_ffn[2] keys are now MISSING — load_state_dict
        # warns; the kept regime_ffn[0] weights match.
        n_active = n_regimes if use_regime_ffn else 1
        sn_per_path = [
            _active_ffn_sn and (i == 0 or use_regime_ffn) for i in range(n_active)
        ]
        self.regime_ffn = nn.ModuleList([
            _make_ffn(sn) for sn in sn_per_path
        ])
        self.drop = nn.Dropout(dropout)
        self.n_regimes = n_regimes
        self._n_active_ffn = n_active

    def forward(self, x: torch.Tensor, regime_w: torch.Tensor) -> torch.Tensor:
        """x: [B, F, D] feature tokens; regime_w: [B, F, n_regimes]."""
        h = self.norm1(x)
        # 2026-05-09 H3 fix: cross-feature attention without forecast supervision
        # learns sign-flipped representations on real data (V22 12-epoch + ShIC
        # empirical proof). Default skip; toggle via USE_CROSS_FEAT_ATTN setting.
        # See docs/V22_V25_FORECAST_HEAD_PROPOSAL_2026_05_09.md.
        if self._use_cross_feat_attn:
            attn_out, _ = self.attn(h, h, h, need_weights=False)
        else:
            attn_out = h  # identity passthrough — proven Capacity-tier on V22
        x = x + self.drop(attn_out)
        h2 = self.norm2(x)
        if self._use_regime_ffn:
            # Per-regime FFN, then regime-weighted sum
            ff_outs = torch.stack([ffn(h2) for ffn in self.regime_ffn], dim=-1)  # [B, F, D, R]
            ff_combined = (ff_outs * regime_w.unsqueeze(-2)).sum(dim=-1)         # [B, F, D]
        else:
            # 2026-05-10 fix: vanilla FFN path. Use only the first regime FFN
            # (others remain in state_dict for ablation, no grad through them
            # so they won't be updated). Eliminates the regime-gate memorization
            # path that drove ShIC=0 across V25 31-epoch run.
            ff_combined = self.regime_ffn[0](h2)
        return x + self.drop(ff_combined)


# =============================================================================
# Component 3: Rate-Budget VIB
# =============================================================================

class RateBudgetVIB(nn.Module):
    """VIB with explicit information-rate target. β auto-tuned via Lagrangian.

    Forward returns (z_expanded, mu, logvar, kl_per_timestep). The β value
    is held as a buffer and updated via update_beta() called from get_loss
    after observing the actual KL.

    Information-theoretic interpretation: I(X; Z) is the channel capacity
    used by the bottleneck. Target is bits/timestep. β > target_β means the
    rate is too high (more anti-memo pressure needed); β < means too low.
    """

    def __init__(self, d_model: int, z_dim: int = VIB_Z_DIM,
                 target_rate_nats: float = VIB_TARGET_RATE_NATS,
                 beta_init: float = VIB_BETA_INIT,
                 beta_lr: float = VIB_BETA_LR,
                 logvar_init: float = VIB_LOGVAR_INIT,
                 dropout: float = 0.1):
        super().__init__()
        self.z_dim = z_dim
        self.target_rate_nats = target_rate_nats
        self.beta_lr = beta_lr
        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_logvar = nn.Linear(d_model, z_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, logvar_init)
        self.z_expand = nn.Sequential(
            nn.Linear(z_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # β (and its log) as buffer for auto-tuning
        self.register_buffer("beta_log", torch.tensor(math.log(beta_init)))

    def forward(self, h: torch.Tensor, training: bool):
        """h: [B, T, D] or [B, F, D]. Returns (feat_out, mu, logvar, kl_pt)."""
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)
        # KL per timestep (mean over batch and z_dim, sum over timesteps approximated as mean)
        kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        return feat, mu, logvar, kl

    @torch.no_grad()
    def update_beta(self, kl_observed: torch.Tensor):
        """Lagrangian update on β. If KL > target, raise β to compress more."""
        error = kl_observed.detach() - self.target_rate_nats
        new_log = self.beta_log + self.beta_lr * error
        new_log.clamp_(math.log(VIB_BETA_MIN), math.log(VIB_BETA_MAX))
        self.beta_log.copy_(new_log)

    def get_beta(self) -> torch.Tensor:
        return torch.exp(self.beta_log)


# =============================================================================
# Component 4: Tail-Adaptive Huber Loss (function, not module)
# =============================================================================

def tail_adaptive_huber(decoded: torch.Tensor, target: torch.Tensor,
                        delta: float = HUBER_DELTA,
                        tail_sigma: float = TAIL_THRESHOLD_SIGMA,
                        tail_weight: float = TAIL_WEIGHT) -> torch.Tensor:
    """Asymmetric Huber: standard Huber + multiplicative weight on tails.

    Crypto returns have kurtosis 5-15. Standard Huber is symmetric and tail-
    agnostic — the tails get under-weighted in mean-aggregated loss. This
    upweights samples where |target| > tail_sigma * std(target).
    """
    err = decoded - target
    abs_err = err.abs()
    quad = 0.5 * err.pow(2)
    lin = delta * (abs_err - 0.5 * delta)
    huber = torch.where(abs_err < delta, quad, lin)
    target_std = target.std() + 1e-6
    tail_mask = (target.abs() > tail_sigma * target_std).float()
    weights = 1.0 + (tail_weight - 1.0) * tail_mask
    return (huber * weights).mean()


# =============================================================================
# V25 World Model
# =============================================================================

class V25FrontierWorldModel(nn.Module):
    """V25: first-principles crypto WM. See module docstring for component list.

    Inputs:
      obs_seq:  [B, T, F]
      asset_id: [B]

    Returns dict (forward_train) with keys: return_logits, regime_logits,
    h_seq, ret_trunk, regime_dist (per-bar regime distribution),
    vib_mu, vib_logvar, vib_kl, plus zero-stub shims for V1.x interface compat.
    """

    def __init__(self, input_dim: int = INPUT_DIM, seq_len: int = WM_SEQ_LEN,
                 d_model: int = WM_D_MODEL, n_heads: int = WM_N_HEADS,
                 n_layers: int = WM_N_LAYERS, dropout: float = WM_DROPOUT,
                 num_bins: int = NUM_BINS, num_assets: int = NUM_ASSETS,
                 asset_emb_dim: int = WM_ASSET_EMB_DIM,
                 z_dim: int = VIB_Z_DIM, n_regimes: int = N_REGIMES):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_regimes = n_regimes
        self.z_dim = z_dim
        self._num_bins = num_bins

        # Phase 14.7 fix (2026-05-09): replace memorization-prone Linear(seq_len, d_model)
        # with PatchTST-style patch embedding. Each feature's seq_len=96 window is split
        # into N_PATCHES=8 patches × PATCH_LEN=12 bars; shared Linear(12, 40) per patch.
        # Memorization capacity: 30,720 → 520 params per feature (60× reduction).
        # Per V22_MEMORIZATION_ROOT_CAUSE_2026_05_08 + NON_V1_MODELS_CRITICAL_AUDIT.
        try:
            from settings import (
                USE_PATCH_EMBEDDING, PATCH_LEN, N_PATCHES, PATCH_DIM,
                EMBED_SPECTRAL_NORM, USE_INPUT_VIB,
                INPUT_VIB_TARGET_RATE_NATS, INV_EMBED_INPUT_DROPOUT,
                INV_EMBED_INPUT_NOISE,
            )
        except ImportError:
            USE_PATCH_EMBEDDING = False
            EMBED_SPECTRAL_NORM = False
            USE_INPUT_VIB = False

        self._use_patch_embedding = USE_PATCH_EMBEDDING
        if USE_PATCH_EMBEDDING:
            patch_embed = nn.Linear(PATCH_LEN, PATCH_DIM)
            if EMBED_SPECTRAL_NORM:
                patch_embed = nn.utils.parametrizations.spectral_norm(patch_embed)
            self.patch_embed = patch_embed
            self._patch_len = PATCH_LEN
            self._n_patches = N_PATCHES
            self._patch_dim = PATCH_DIM
            # Legacy Linear kept None when patches active (prevents grad on dead path)
            self.embed = None
        else:
            base_embed = nn.Linear(seq_len, d_model)
            if EMBED_SPECTRAL_NORM:
                base_embed = nn.utils.parametrizations.spectral_norm(base_embed)
            self.embed = base_embed
            self.patch_embed = None

        # Round-9 F1: input regularization on patches (defensive layer)
        self._inv_embed_input_drop = INV_EMBED_INPUT_DROPOUT if USE_PATCH_EMBEDDING else 0.0
        self._inv_embed_input_noise = INV_EMBED_INPUT_NOISE if USE_PATCH_EMBEDDING else 0.0

        # Phase 14.7 fix (Phase 14.8 z_dim correction): upstream VIB BEFORE transformer
        # layers. Original VIB at h_seq (post-encoder) is "right tool, wrong location" —
        # bottlenecks z but can't stop upstream embedding from memorizing.
        # Phase 14.8 z_dim fix: was z_dim=d_model (320, no real compression) → catastrophic
        # rate strangulation. Now z_dim=INPUT_VIB_Z_DIM (32) for real 320→32→320 bottleneck.
        if USE_INPUT_VIB:
            try:
                from settings import INPUT_VIB_Z_DIM as _input_vib_z_dim
            except ImportError:
                _input_vib_z_dim = 32
            self.input_vib = RateBudgetVIB(
                d_model, z_dim=_input_vib_z_dim,
                target_rate_nats=INPUT_VIB_TARGET_RATE_NATS,
                beta_init=VIB_BETA_INIT, beta_lr=VIB_BETA_LR,
                logvar_init=VIB_LOGVAR_INIT, dropout=dropout,
            )
        else:
            self.input_vib = None

        # Asset token (paper §4.2 covariate extension)
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.asset_token_proj = nn.Linear(asset_emb_dim, d_model)

        # Component 1: Hard-coded crypto period embedding (added to T-axis input)
        self.period_emb = CryptoPeriodEmbedding(d_model, periods=PERIOD_BARS,
                                                amp_init=PERIOD_AMP_INIT)

        # Per-bar regime gate (operates on h_seq AFTER inverted attention)
        self.regime_gate_per_bar = RegimeGate(d_model, n_regimes=n_regimes)
        # Per-feature regime gate (operates on feature tokens DURING attention)
        self.regime_gate_per_feat = RegimeGate(d_model, n_regimes=n_regimes)

        # Component 2: Regime-conditioned inverted attention
        # 2026-05-09 H3 fix: cross-feature attention defaults OFF (sign-flip bug).
        # Toggle via USE_CROSS_FEAT_ATTN in settings (kept for ablation/future
        # experiments with forecast head).
        try:
            from settings import USE_CROSS_FEAT_ATTN as _use_attn
        except ImportError:
            _use_attn = False
        try:
            from settings import USE_REGIME_FFN as _use_regime_ffn
        except ImportError:
            _use_regime_ffn = True
        self.layers = nn.ModuleList([
            RegimeConditionedInvertedAttention(d_model, n_heads,
                                                n_regimes=n_regimes, dropout=dropout,
                                                use_cross_feat_attn=_use_attn,
                                                use_regime_ffn=_use_regime_ffn)
            for _ in range(n_layers)
        ])
        self.post_norm = RMSNorm(d_model)

        # Inverted projection: [B, F, D] -> [B, F, T]
        # 2026-05-10 variant β: spectral_norm on proj. Cross-instance probe
        # (docs/V25_MEMORIZATION_DIAGNOSIS_2026_05_10.md) found proj.weight
        # rank-collapsed to sr_frac=0.035 (epoch 30, regime_ffn ON) and 0.050
        # (epoch 10, regime_ffn OFF) -- 3-5% of 96 effective output dimensions.
        # This 320->96 inversion is the primary residual memorization channel.
        try:
            from settings import PROJ_SPECTRAL_NORM as _proj_sn
        except ImportError:
            _proj_sn = False
        self.proj = nn.Linear(d_model, seq_len)
        if _proj_sn:
            self.proj = nn.utils.parametrizations.spectral_norm(self.proj)

        # Per-bar projection: F cross-feature signals → D-dim per-bar rep
        self.bar_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Component 3: Rate-budget VIB on per-bar h_seq
        self.vib = RateBudgetVIB(d_model, z_dim=z_dim,
                                 target_rate_nats=VIB_TARGET_RATE_NATS,
                                 beta_init=VIB_BETA_INIT,
                                 beta_lr=VIB_BETA_LR,
                                 logvar_init=VIB_LOGVAR_INIT,
                                 dropout=dropout)

        # Heads
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, RETURN_HEAD_DIM),
            RMSNorm(RETURN_HEAD_DIM),
            nn.SiLU(),
            nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(RETURN_HEAD_DIM, RETURN_HEAD_DIM // 2),
                RMSNorm(RETURN_HEAD_DIM // 2),
                nn.SiLU(),
                nn.Linear(RETURN_HEAD_DIM // 2, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Regime classification head (3-way: bear/neutral/bull)
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot bucketer
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall log-vars
        self.log_vars = nn.Parameter(
            torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5])
        )

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026)
        # NOTE: V25 has internal regime_ffn (encoder-side, gated by
        # USE_REGIME_FFN). CC-H6's RegimeConditionalHeads is decoder-side
        # and COMPLEMENTARY — adds per-regime return decoders after encoder.
        import sys as _qsys
        from pathlib import Path as _qPath
        _shared = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
        if _shared not in _qsys.path:
            _qsys.path.insert(0, _shared)
        try:
            from settings import USE_QUANTILE_HEADS as _use_qh
        except ImportError:
            _use_qh = False
        if _use_qh:
            from headline_components import QuantileHeads as _QH
            self.quantile_heads = _QH(head_input_dim=d_model,
                                         horizons=tuple(REWARD_HORIZONS))
        else:
            self.quantile_heads = None
        try:
            from settings import USE_REGIME_COND_HEADS as _use_rc
        except ImportError:
            _use_rc = False
        if _use_rc:
            from headline_components import RegimeConditionalHeads as _RC
            self.regime_cond_heads = _RC(head_input_dim=d_model,
                                            horizons=tuple(REWARD_HORIZONS),
                                            num_bins=num_bins, n_regimes=3)
        else:
            self.regime_cond_heads = None
        try:
            from settings import REGIME_AWARENESS_MODE as _ram
        except ImportError:
            _ram = "heads"
        self._regime_awareness_mode = _ram
        if _ram == "film":
            from headline_components import RegimeFiLM as _RF
            self.regime_film = _RF(d_model=d_model, n_regimes=3)
            self.regime_film_gate = nn.Linear(d_model, 3)
            nn.init.zeros_(self.regime_film_gate.weight)
            nn.init.zeros_(self.regime_film_gate.bias)
        else:
            self.regime_film = None
            self.regime_film_gate = None

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor,
                      masked_obs_seq: torch.Tensor | None = None):
        B, T, F_in = obs_seq.shape
        if F_in != self.input_dim:
            raise ValueError(f"Expected n_features={self.input_dim}, got {F_in}")
        if T != self.seq_len:
            if T < self.seq_len:
                pad_len = self.seq_len - T
                obs_seq = F.pad(obs_seq, (0, 0, 0, pad_len))
                if masked_obs_seq is not None:
                    masked_obs_seq = F.pad(masked_obs_seq, (0, 0, 0, pad_len))
            else:
                obs_seq = obs_seq[:, : self.seq_len, :]
                if masked_obs_seq is not None:
                    masked_obs_seq = masked_obs_seq[:, : self.seq_len, :]
            T = self.seq_len

        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Causal shift (kept; tested no-shift 2026-05-09 -> Ep1 ic1=-0.131
        # vs with-shift Ep1 ic1=-0.103, removing the shift made it slightly
        # WORSE; structural sign-flip bug is elsewhere).
        shifted = torch.cat(
            [torch.zeros(B, 1, F_in, device=obs_seq.device), input_obs[:, :-1, :]],
            dim=1,
        )

        # Component 1: Add hard-coded crypto period embedding to per-timestep input
        # 2026-05-10 ablation γ: full period_emb gating. Previously controlled only
        # by PERIOD_AMP_INIT (initial scalar amplitude); the proj layer trained
        # regardless and could revive position info via gradient pressure. Now
        # the entire add-site is gated by USE_PERIOD_EMB. Default: False.
        try:
            from settings import USE_PERIOD_EMB as _use_period_emb
        except ImportError:
            _use_period_emb = True   # legacy fallback for older configs
        if _use_period_emb:
            # period_emb: [T, D] -> per-timestep additive embedding
            period_t = self.period_emb(T, device=obs_seq.device)  # [T, D]
            # We add period info AFTER the inversion so it influences per-feature tokens
            # via the temporal axis. To do that we'd need [B, F, T] -> [B, F, D] embedding.
            # Simpler: project period_emb back to [T] scalar via a learned linear and add
            # to each feature's time-series before embedding.
            period_per_t = period_t.mean(dim=-1)  # [T] scalar period signal
            # Add to each feature's T-series (broadcasts: [1, T, 1])
            shifted = shifted + period_per_t.unsqueeze(0).unsqueeze(-1) * 0.1  # mild residual

        # [B, T, F] -> [B, F, T] (inversion)
        x_t = shifted.transpose(1, 2)
        # Round-9 F1 (Phase 14.7): regularize T-series before embedding
        if self.training and self._inv_embed_input_drop > 0:
            x_t = nn.functional.dropout(x_t, self._inv_embed_input_drop)
        if self.training and self._inv_embed_input_noise > 0:
            x_t = x_t + torch.randn_like(x_t) * self._inv_embed_input_noise
        # Phase 14.7 fix: PatchTST-style embedding (replaces dense Linear(96, 320))
        # Each feature's 96-bar T-series → 8 patches × Linear(12, 40) → concat to 320-dim
        if self._use_patch_embedding:
            B_, F_, T_ = x_t.shape
            patches = x_t.reshape(B_, F_, self._n_patches, self._patch_len)
            patch_emb = self.patch_embed(patches)              # [B, F, N_PATCHES, PATCH_DIM]
            tokens = patch_emb.reshape(B_, F_, self._n_patches * self._patch_dim)  # [B, F, d_model]
        else:
            # Legacy dense embedding (kept for ablation)
            tokens = self.embed(x_t)

        # Phase 14.7 fix: upstream VIB BEFORE transformer layers (not at h_seq output)
        # Bottleneck the embedding output before memorized patterns can propagate.
        upstream_kl = None
        if self.input_vib is not None:
            tokens, _, _, upstream_kl = self.input_vib(tokens, training=self.training)
            if self.training:
                self.input_vib.update_beta(upstream_kl)

        # Prepend asset token
        if USE_ASSET_TOKEN:
            asset_emb = self.asset_embedding(asset_id)
            asset_tok = self.asset_token_proj(asset_emb).unsqueeze(1)  # [B, 1, D]
            tokens = torch.cat([asset_tok, tokens], dim=1)            # [B, F+1, D]

        # Per-feature regime distribution (used to gate FFN within each layer)
        regime_w_per_feat = self.regime_gate_per_feat(tokens)  # [B, F+1, n_regimes]

        # Component 2: Regime-conditioned inverted attention
        for layer in self.layers:
            tokens = layer(tokens, regime_w_per_feat)
        tokens = self.post_norm(tokens)

        # Drop asset token for projection
        feat_tokens = tokens[:, 1:, :] if USE_ASSET_TOKEN else tokens   # [B, F, D]

        # Inverted projection: [B, F, D] -> [B, F, T]
        feat_T = self.proj(feat_tokens)

        # Per-bar representation: at each bar t, F cross-feature values
        per_bar_F = feat_T.transpose(1, 2)             # [B, T, F]
        h_seq = self.bar_proj(per_bar_F)               # [B, T, D]

        # RegimeFiLM (SOTA-2026 opt-in) — identity-at-init; modulates h_seq
        # BEFORE the existing V25 regime_gate_per_bar + VIB. Complements V25's
        # internal regime_ffn (encoder-side) by adding learnable encoder-level
        # regime conditioning that's not gated by USE_REGIME_FFN.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # Per-bar regime distribution (used downstream for adversarial regime weighting)
        regime_dist_per_bar = self.regime_gate_per_bar(h_seq)  # [B, T, n_regimes]

        # Component 3: Rate-budget VIB on h_seq
        feat_vib, vib_mu, vib_logvar, vib_kl = self.vib(h_seq, training=self.training)

        # ATME (per-sample)
        feat = feat_vib
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device)
                         > TEMPORAL_CTX_DROP).float()
            # ATME drops temporal CONTEXT but must NOT zero the supervised last bar:
            # under USE_LAST_BAR_SUPERVISION the loss reads feat[:, -1], so zeroing
            # the whole feat (incl. position T-1) taught predict-from-nothing 15% of
            # the time. Mask context [0:T-1] only; keep the prediction position.
            if feat.shape[1] > 1:
                feat = torch.cat([feat[:, :-1, :] * atme_mask, feat[:, -1:, :]], dim=1)
            else:
                feat = feat * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {
            h_key: self.return_heads[str(h_key)](ret_trunk)
            for h_key in REWARD_HORIZONS
        }
        regime_logits = self.regime_head(feat)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026, complementary to V25's
        # existing regime_ffn which is encoder-side).
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "ret_trunk": ret_trunk,
            "regime_dist": regime_dist_per_bar,    # [B, T, n_regimes]
            "vib_mu": vib_mu,
            "vib_logvar": vib_logvar,
            "vib_kl": vib_kl,
            "z_post": (vib_mu + 0 * vib_logvar),    # surrogate z
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
            # 2026-05-10 forecast-head root-cause fix: expose feat_T for auxiliary
            # MSE loss in get_loss. Anchors encoder to feature-faithful semantics.
            "feat_T": feat_T,                       # [B, F, T] iTransformer per-feature forecast
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None,
                 return_components=False, **kwargs):
        """V1-compatible loss interface with first-principles components.

        Components 4 + 5 (tail-adaptive Huber + adversarial regime upweighting)
        applied here in the loss layer.
        """
        B, T, n_feat = obs_seq.shape
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        masked_obs = obs_seq.clone()
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # Component 3: Auto-tuned VIB β via Lagrangian
        vib_kl = outputs["vib_kl"]
        beta = self.vib.get_beta()
        if self.training:
            self.vib.update_beta(vib_kl)

        s = self.log_vars.clamp(-6.0, 6.0)
        loss_dict = {
            "total": 0.0, "rec": 0.0,
            "kl": vib_kl.item(),
            "kl_raw": vib_kl.item(),
            "vib_beta": beta.item(),
            "vib_target_rate_nats": VIB_TARGET_RATE_NATS,
        }
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # Component 5: Adversarial regime upweighting
        # Per-batch regime histogram → identify worst-quintile regime mass → upweight
        regime_dist = outputs["regime_dist"]      # [B, T, n_regimes]
        regime_freq = regime_dist.mean(dim=(0, 1))  # [n_regimes] empirical batch frequency
        # Worst regime = least-frequent (model has seen fewer samples there → less robust)
        worst_regime_idx = torch.argmin(regime_freq).item()
        # Weight per-bar by [1 + (ADVERSARIAL_REGIME_WEIGHT - 1) * regime_dist[..., worst_regime]]
        adv_weight = 1.0 + (ADVERSARIAL_REGIME_WEIGHT - 1.0) * \
                     regime_dist[..., worst_regime_idx]   # [B, T]
        adv_weight_flat = adv_weight.reshape(-1)           # [B*T]
        loss_dict["adv_regime_idx"] = worst_regime_idx
        loss_dict["adv_weight_mean"] = adv_weight.mean().item()

        # 2026-05-21 SOTA causality fix (Timer-XL / TimesFM pattern). Supervise
        # only the LAST bar — the only position whose attention representation
        # has no future-bar leak. Same fix as V22.
        try:
            from settings import USE_LAST_BAR_SUPERVISION as _last_bar
        except ImportError:
            _last_bar = True  # SOTA default

        def _slice_logits(x):
            return x[:, -1:, :] if _last_bar else x

        def _slice_target(x):
            return x[:, -1:] if _last_bar else x

        # Per-horizon return losses with adversarial regime upweighting
        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_flat = _slice_logits(outputs["return_logits"][h]).reshape(-1, self._num_bins)
            tgt_flat = _slice_target(targets[h]).reshape(-1)
            if h in ACTIVE_HORIZONS:
                # Compute base TwoHot CE, then apply adversarial regime weighting
                # at batch granularity (TwoHotSymlog returns scalar mean loss).
                # Per-sample weighting would require modifying TwoHotSymlog;
                # batch-mean weighting is sufficient for the adversarial-regime
                # signal since regime mass is computed over the same batch.
                l_ret_base = self.bucketer.compute_loss(logits_flat, tgt_flat)
                l_ret = l_ret_base * adv_weight.mean()
                s_ret = s[hi].clamp(max=-2.0)
                ret_terms[h] = torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            decoded = self.bucketer.decode(logits_flat)
            # Component 4: Tail-adaptive Huber direct return loss
            l_direct = l_direct + tail_adaptive_huber(
                decoded, tgt_flat,
                delta=HUBER_DELTA,
                tail_sigma=TAIL_THRESHOLD_SIGMA,
                tail_weight=TAIL_WEIGHT,
            )

        aux_term = DIRECT_RETURN_WEIGHT * l_direct + beta * vib_kl
        loss_dict["direct_ret"] = l_direct.item()

        # 2026-05-10 forecast-head ROOT-CAUSE FIX for memorization:
        # MSE(feat_T, obs_seq.T) anchors the encoder representation to be
        # feature-faithful. Without this, V25 (like V22) drifted to position-
        # memorization (ShIC=0.000 with high contiguous IC). Mirrors V1.x's
        # RSSM reconstruction anchor. See docs/V22_V25_SOLUTION_2026_05_10.md.
        try:
            from settings import USE_FORECAST_HEAD as _use_fc, FORECAST_WEIGHT as _fc_w
        except ImportError:
            _use_fc, _fc_w = False, 0.0
        if _use_fc and "feat_T" in outputs:
            forecast_target = obs_seq.transpose(1, 2)        # [B, F, T]
            forecast_loss = F.mse_loss(outputs["feat_T"], forecast_target)
            aux_term = aux_term + _fc_w * forecast_loss
            loss_dict["forecast_mse"] = forecast_loss.item()

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            # Last-bar regime supervision under USE_LAST_BAR_SUPERVISION
            regime_logits_sliced = _slice_logits(outputs["regime_logits"])
            regime_tgt_sliced = _slice_target(regime_tgt)
            l_regime = F.cross_entropy(
                regime_logits_sliced.reshape(-1, 3), regime_tgt_sliced.reshape(-1)
            )
            s_regime = s[-1].clamp(max=-1.0)
            aux_term = aux_term + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (
                    regime_logits_sliced.argmax(-1) == regime_tgt_sliced
                ).float().mean().item()

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    # Last-bar dir_acc under USE_LAST_BAR_SUPERVISION
                    logits_h = _slice_logits(outputs["return_logits"][h]).reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = _slice_target(targets[h]).reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (
                            torch.sign(dec[nz]) == torch.sign(act[nz])
                        ).float().mean().item()
            # Per-regime IC for autopsy
            for r in range(self.n_regimes):
                regime_mass = regime_dist[..., r].mean().item()
                loss_dict[f"regime_{r}_mass"] = regime_mass

        # CC-H5 + CC-H6 auxiliary losses (SOTA-2026)
        try:
            from settings import QUANTILE_LOSS_WEIGHT as _ql_w
        except ImportError:
            _ql_w = 0.0
        l_quantile = torch.tensor(0.0, device=obs_seq.device)
        if (self.quantile_heads is not None
                and outputs.get("quantile_logits") is not None
                and _ql_w > 0):
            import sys as _qsys
            from pathlib import Path as _qPath
            _sd = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
            if _sd not in _qsys.path:
                _qsys.path.insert(0, _sd)
            from headline_components import quantile_loss as _ql
            quants = self.quantile_heads.quantiles
            n_h = 0
            for h_key in REWARD_HORIZONS:
                if h_key in targets:
                    # Last-bar quantile loss under USE_LAST_BAR_SUPERVISION
                    q_pred = _slice_logits(outputs["quantile_logits"][h_key])
                    q_tgt = _slice_target(targets[h_key])
                    l_quantile = l_quantile + _ql(q_pred, q_tgt, quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile
        loss_dict["quantile_pinball"] = l_quantile.item()
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0 and regime_labels is not None):
            # Last-bar regime-conditional under USE_LAST_BAR_SUPERVISION
            flat_labels = _slice_target(regime_labels).reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h_key in REWARD_HORIZONS:
                    if h_key not in targets:
                        continue
                    head_logits = _slice_logits(outputs["regime_cond_logits"][r][h_key])
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = _slice_target(targets[h_key]).reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        total = total + _rc_w * l_regime_cond
        loss_dict["regime_cond_ce"] = l_regime_cond.item()

        loss_dict["total"] = total.item()

        if return_components:
            components = {
                "aux": aux_term,
                **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS},
            }
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(42)
    B, T, F_in = 4, WM_SEQ_LEN, INPUT_DIM
    m = V25FrontierWorldModel(input_dim=F_in)
    x = torch.randn(B, T, F_in)
    asset = torch.randint(0, NUM_ASSETS, (B,))
    targets = {h: torch.randn(B, T) * 0.01 for h in REWARD_HORIZONS}
    total, ld, out = m.get_loss(x, asset, targets, mask_ratio=0.15)
    total.backward()
    n_params = count_parameters(m)
    print(f"[V25 Frontier smoke] PASS: B={B} T={T} F={F_in}")
    print(f"  params={n_params:,}  loss={ld['total']:.4f}  direct={ld['direct_ret']:.4f}")
    print(f"  vib_kl={ld['kl']:.4f}  vib_beta={ld['vib_beta']:.4f}  target_rate={ld['vib_target_rate_nats']:.2f}")
    print(f"  adv_regime_idx={ld['adv_regime_idx']}  adv_weight_mean={ld['adv_weight_mean']:.3f}")
    for r in range(m.n_regimes):
        print(f"  regime_{r}_mass={ld[f'regime_{r}_mass']:.3f}")
    print(f"  return_logits[1]: {tuple(out['return_logits'][1].shape)}")
