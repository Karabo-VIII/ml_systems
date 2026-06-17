"""
V11 World Model -- Microstructure Feature Extractor
=====================================================

No RSSM. No reconstruction. No dream step.
Every parameter serves return prediction.

Architecture:
  Input [B, T, F] + asset_emb
    -> WaveNet-TCN (multi-scale dilated causal convolutions)
    -> Regime-gated experts (trending TCN vs reverting TCN, Hurst-gated)
    -> Post-encoder feature attention (cross-feature interaction on temporal reps)
    -> Return trunk -> per-horizon heads (h=1, h=4 only)
    -> Regime head

Anti-memorization:
  - Time-shuffle discriminator (learned adversary on encoder output)
  - ATME (30% temporal context zeroing in return heads)
  - Random token masking (25%, not block)

Training interface matches V1.x:
  - get_loss(obs, asset, targets, ...) -> (loss, loss_dict, outputs)
  - forward_train(obs, asset) -> dict with return_logits, regime_logits, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

# Import shared components from V1.0
_v1_components = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_components not in sys.path:
    sys.path.insert(0, _v1_components)

from components import RMSNorm, TwoHotSymlog, SwiGLU, MLPHead


# =============================================================================
# WaveNet Causal TCN
# =============================================================================

class CausalConv1d(nn.Module):
    """Causal convolution: pad left, no future leakage."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        # x: [B, C, T]
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class WaveNetBlock(nn.Module):
    """Gated dilated causal convolution with residual + skip connections.

    Phase 14.7 fix: post-residual RMSNorm (was missing — caused grad
    explosion 306K in B=32 audit probe). Aligns with V3's WaveNetBlock.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        # Post-residual RMSNorm: bounds activation magnitude through stack depth
        self.norm = RMSNorm(channels)

    def forward(self, x):
        # x: [B, C, T]
        # fp32 for gated activation — tanh/sigmoid backward overflows fp16
        _f, _g = self.filter_conv(x), self.gate_conv(x)
        h = torch.tanh(_f.float()) * torch.sigmoid(_g.float())
        h = h.to(x.dtype)
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        # Apply RMSNorm on [B, T, C] then transpose back
        residual = self.norm(residual.transpose(1, 2)).transpose(1, 2)
        return residual, skip


class WaveNetTCN(nn.Module):
    """Multi-scale WaveNet TCN with skip aggregation.

    Each layer operates at a different dilation, capturing patterns from
    3-bar (dilation=1) to 17-bar (dilation=8) timescales.
    All skip connections project to the final channel size for clean summation.
    """

    def __init__(self, in_dim, channels, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = channels[-1]

        self.blocks = nn.ModuleList()
        self.channel_transitions = nn.ModuleList()
        self.skip_projs = nn.ModuleList()  # Project each skip to out_dim

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            if in_ch != ch:
                self.channel_transitions.append(nn.Conv1d(in_ch, ch, 1))
            else:
                self.channel_transitions.append(None)
            self.blocks.append(WaveNetBlock(ch, kernel_size, dil, dropout))
            # Skip projection to final channel size
            if ch != self.out_dim:
                self.skip_projs.append(nn.Conv1d(ch, self.out_dim, 1))
            else:
                self.skip_projs.append(None)

        self.output_norm = nn.GroupNorm(8, self.out_dim)

    def forward(self, x):
        """x: [B, T, F] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.input_proj(x)

        skip_sum = 0
        n_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if self.channel_transitions[i] is not None:
                x = self.channel_transitions[i](x)
            x, skip = block(x)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i](skip)
            skip_sum = skip_sum + skip

        # Phase 14.7 fix: scale skip-sum by 1/sqrt(N) to keep variance bounded
        # as stack depth grows. Pre-fix grad_max=306K in B=32 probe.
        if n_blocks > 1:
            skip_sum = skip_sum / (n_blocks ** 0.5)
        x = self.output_norm(x + skip_sum)
        return x.transpose(1, 2)  # [B, T, out_dim]


# =============================================================================
# Regime-Gated Experts
# =============================================================================

class ExpertTCN(nn.Module):
    """Small TCN expert for one regime type."""

    def __init__(self, in_dim, out_dim, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Conv1d(in_dim, out_dim, 1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(out_dim, kernel_size, d, dropout) for d in dilations
        ])
        self.proj_out = nn.Conv1d(out_dim, out_dim, 1)
        self.norm = nn.GroupNorm(min(8, out_dim), out_dim)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)
        x = self.proj_in(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(self.proj_out(x))
        return x.transpose(1, 2)


class RegimeGatedExperts(nn.Module):
    """Up to two regime experts gated by Hurst exponent.

    Trending expert (Hurst > threshold): large dilations for momentum.
    Reverting expert (Hurst <= threshold): small dilations for mean-reversion.
    Soft blending based on Hurst distance from threshold.

    2026-05-21: now honors `n_experts` to gate the V9 MoE pattern. When
    n_experts=1, only the trending expert is built and used (no gating,
    no reverting branch). The settings.HEADLINE_MOE_EXPERTS flag is wired
    via MicrostructureWorldModel.__init__. Previously hardcoded to 2 —
    the flag was decorative.
    """

    def __init__(self, in_dim, expert_dim, trending_dilations, reverting_dilations,
                 hurst_threshold=0.1, dropout=0.1, n_experts: int = 2):
        super().__init__()
        if n_experts not in (1, 2):
            raise ValueError(f"n_experts must be 1 or 2, got {n_experts}")
        self.n_experts = n_experts
        self.expert_trending = ExpertTCN(in_dim, expert_dim, trending_dilations, dropout=dropout)
        if n_experts == 2:
            self.expert_reverting = ExpertTCN(in_dim, expert_dim, reverting_dilations, dropout=dropout)
        else:
            self.expert_reverting = None  # not built; saves params + compute
        self.merge_proj = nn.Linear(expert_dim, in_dim)
        self.merge_norm = RMSNorm(in_dim)
        self.hurst_threshold = hurst_threshold

    def forward(self, x, hurst_feature):
        """
        x: [B, T, D] encoder output
        hurst_feature: [B, T] hurst regime values (used only when n_experts==2)
        Returns: [B, T, D] (same shape as input, residual added)
        """
        if self.n_experts == 1:
            # No gating; trending expert handles all regimes.
            h_out = self.expert_trending(x)  # [B, T, expert_dim]
        else:
            # Soft gating: sigmoid centered on threshold, scaled for sharpness.
            # hurst > threshold -> gate -> 1.0 (trending dominates).
            gate = torch.sigmoid(5.0 * (hurst_feature.unsqueeze(-1) - self.hurst_threshold))
            h_trend = self.expert_trending(x)
            h_revert = self.expert_reverting(x)
            h_out = gate * h_trend + (1.0 - gate) * h_revert
        projected = self.merge_proj(h_out)
        return self.merge_norm(x + projected)  # Residual connection


# =============================================================================
# Post-Encoder Feature Attention
# =============================================================================

class TemporalFeatureAttention(nn.Module):
    """Cross-feature attention on temporal representations.

    After the WaveNet extracts temporal patterns at each bar, this module
    lets features interact: "VPIN spike pattern + positive flow pattern
    at this timestep = breakout signal."

    Unlike V1.4's pre-encoder FeatureAttention (which can't see temporal
    patterns), this operates on the encoder OUTPUT where each feature-
    channel carries temporal context.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, D]"""
        B, T, D = x.shape
        # Self-attention over the D dimension at each timestep
        # Reshape: treat each timestep independently
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, D/H]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        x = residual + self.dropout(self.out_proj(attn))

        # FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# Time-Shuffle Discriminator
# =============================================================================

class TimeShuffleDiscriminator(nn.Module):
    """Classifies whether a sequence of latent representations is
    in temporal order or time-shuffled.

    The encoder is trained adversarially to fool this discriminator,
    forcing it to produce temporal-invariant representations.

    2026-05-21: spectral_norm parametrization (Miyato et al ICLR 2018
    arXiv:1802.05957) is now wired via the `spectral_norm` constructor
    arg, threaded from settings.HEADLINE_DISC_SPECTRAL_NORM. Bounding
    the discriminator's largest singular value to 1 enforces the
    1-Lipschitz constraint, which is the standard SNGAN stabilizer for
    adversarial training. Previously the flag was a no-op.
    """

    def __init__(self, input_dim, hidden_dim=128, n_layers=3, dropout=0.15,
                 spectral_norm: bool = False):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(n_layers):
            lin = nn.Linear(in_d, hidden_dim)
            if spectral_norm:
                lin = nn.utils.parametrizations.spectral_norm(lin)
            layers.append(lin)
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        out_lin = nn.Linear(hidden_dim, 1)
        if spectral_norm:
            out_lin = nn.utils.parametrizations.spectral_norm(out_lin)
        layers.append(out_lin)
        self.net = nn.Sequential(*layers)

    def forward(self, h_seq):
        """
        h_seq: [B, T, D] -- sequence of encoder outputs
        Returns: [B, 1] -- probability of being temporally coherent
        """
        # Use variance of consecutive differences as temporal structure signal
        diffs = h_seq[:, 1:, :] - h_seq[:, :-1, :]  # [B, T-1, D]
        stats = torch.cat([
            diffs.mean(dim=1),          # Mean diff [B, D]
            diffs.std(dim=1),           # Std diff [B, D]
            h_seq.mean(dim=1),          # Mean repr [B, D]
        ], dim=-1)  # [B, 3*D]
        return self.net(stats)


# =============================================================================
# V11 World Model
# =============================================================================

class MicrostructureWorldModel(nn.Module):
    """V11: Production-grade microstructure feature extractor.

    No RSSM, no reconstruction, no dream step.
    Every parameter serves return prediction.
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Asset embedding
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        # Input projection (features + asset embedding -> d_model)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # WaveNet TCN encoder (multi-scale temporal patterns)
        self.wavenet = WaveNetTCN(
            in_dim=d_model,
            channels=WAVENET_CHANNELS,
            dilations=WAVENET_DILATIONS,
            kernel_size=WAVENET_KERNEL,
            dropout=WAVENET_DROPOUT,
        )

        # Regime-gated experts
        # HEADLINE_MOE_EXPERTS: 1 = single-expert (drops V9 MoE leak),
        # 2 = legacy two-expert with Hurst gate. Wired 2026-05-21
        # (was decorative flag).
        try:
            from settings import HEADLINE_MOE_EXPERTS as _moe_n
        except ImportError:
            _moe_n = 2
        self.regime_experts = RegimeGatedExperts(
            in_dim=d_model,
            expert_dim=EXPERT_D_MODEL,
            trending_dilations=EXPERT_TRENDING_DILATIONS,
            reverting_dilations=EXPERT_REVERTING_DILATIONS,
            hurst_threshold=HURST_GATE_THRESHOLD,
            dropout=EXPERT_DROPOUT,
            n_experts=int(_moe_n),
        )

        # Post-encoder feature attention
        self.feat_attn = TemporalFeatureAttention(
            d_model=d_model,
            n_heads=FEAT_ATTN_HEADS,
            dropout=FEAT_ATTN_DROPOUT,
        )

        # VIB bottleneck (SOTA upgrade 2026-04-22: fixes ShIC=0 at ep9)
        # Replaces the no-op ATME by forcing stochastic compression of h_seq.
        self.z_dim = VIB_Z_DIM
        self.to_mu = nn.Linear(d_model, self.z_dim)
        self.to_logvar = nn.Linear(d_model, self.z_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, VIB_LOGVAR_INIT)
        self.z_expand = nn.Sequential(
            nn.Linear(self.z_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── RECON ANCHOR (keystone, 2026-06-11; donor V12:222/434/702-710) ───
        # Reconstruction decoder OFF the bottlenecked VIB latent (z_expand(z), the
        # d_model `feat`). Mirrors the V12 donor + the V13/V23 grafts: decode the
        # bottlenecked feature back to the INPUT features [B,T,input_dim]. V11's
        # forward_train had `recon=torch.zeros` (a stub) and get_loss reported
        # `rec=0.0`/`kl=0.0` with NO recon term in `total` -- so the VIB bottleneck
        # had no input-reconstruction pressure (the heads could route around z; the
        # V22/V25 memorization trap: high contiguous IC, ShIC~0). The recon term
        # forces the 32-dim VIB latent to retain input-reconstructable structure
        # (not pure label-fit); the VIB KL (already computed + already in `total`,
        # get_loss line ~646) caps its capacity. Together they make the bottleneck
        # REAL -- the missing HALF of the anchor. RNG-neutral construction (mirrors
        # the V23 graft): snapshot RNG before building the decoder + restore after,
        # and skip recon_decoder in _init_weights, so the pre-graft base params see
        # the exact same RNG draws.
        _rng_state = torch.get_rng_state()
        self.recon_decoder = nn.Sequential(
            SwiGLU(d_model, RETURN_HEAD_DIM, dim_out=RETURN_HEAD_DIM, dropout=dropout),
            RMSNorm(RETURN_HEAD_DIM),
            nn.Linear(RETURN_HEAD_DIM, input_dim),
        )
        torch.set_rng_state(_rng_state)

        # Return prediction trunk + per-horizon heads
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, RETURN_HEAD_DIM),
            RMSNorm(RETURN_HEAD_DIM),
            nn.SiLU(),
            nn.Dropout(RETURN_HEAD_DROPOUT),
        )

        ret_mid = RETURN_HEAD_DIM // 2
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(RETURN_HEAD_DIM, ret_mid),
                RMSNorm(ret_mid),
                nn.SiLU(),
                nn.Linear(ret_mid, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Regime head
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot bucketer
        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Time-shuffle discriminator (separate optimizer). spectral_norm
        # wired 2026-05-21 from settings.HEADLINE_DISC_SPECTRAL_NORM (was
        # dead flag previously). Mitigates the unbounded-D failure mode
        # at the bumped DISC_WEIGHT=0.3.
        try:
            from settings import HEADLINE_DISC_SPECTRAL_NORM as _disc_sn
        except ImportError:
            _disc_sn = False
        self.discriminator = TimeShuffleDiscriminator(
            input_dim=d_model * 3,  # mean_diff + std_diff + mean_repr
            hidden_dim=DISC_HIDDEN,
            n_layers=DISC_LAYERS,
            spectral_norm=_disc_sn,
        )

        # Kendall log_vars for multi-task balancing
        # [ret_1, ret_4, ret_16, ret_64, regime]
        n_loss_terms = len(REWARD_HORIZONS) + 1
        self.log_vars = nn.Parameter(torch.tensor(
            [-2.0] * len(REWARD_HORIZONS) + [-1.5]  # Returns weighted high, regime medium
        ))

        # Hurst feature index (for regime gating)
        self._hurst_idx = 9  # hurst_regime is feature #9 in all FEATURE_LISTs

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026 auxiliary heads / encoder cond)
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
        """Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
            # Skip the recon-anchor decoder (2026-06-11 graft): it keeps its
            # construction-time init (nn.Linear xavier-via-this-loop would otherwise
            # consume RNG draws). Excluding it here keeps this loop's RNG draws
            # byte-for-byte identical to the pre-graft model so every base param
            # below it sees the same RNG state.
            if name.startswith("recon_decoder"):
                continue
            if "weight" in name and param.dim() >= 2:
                if "conv" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="linear")
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """Forward pass for training and inference.

        Args:
            obs_seq: [B, T, F] observation sequence
            asset_id: [B] asset indices
            masked_obs_seq: [B, T, F] masked version (optional)

        Returns dict compatible with V1.x interface:
            return_logits: {h: [B, T, NUM_BINS]}
            regime_logits: [B, T, 3]
            h_seq: [B, T, d_model]
        """
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Asset embedding broadcast over time
        asset_emb = self.asset_embedding(asset_id)  # [B, emb_dim]
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, emb_dim]

        # Causal shift: prepend zeros, drop last (prevent current-bar leakage)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device), input_obs[:, :-1, :]], dim=1)
        enc_input = torch.cat([shifted, asset_emb], dim=-1)  # [B, T, F+emb]

        # Encode — WaveNet in fp32 to prevent GN:inf from residual+skip
        # gradient accumulation in fp16 backward (same fix as V3)
        h = self.obs_encoder(enc_input)       # [B, T, d_model]
        with torch.amp.autocast("cuda", enabled=False):
            h = self.wavenet(h.float())        # [B, T, d_model] fp32

        # Regime-gated experts — read Hurst from the SHIFTED+MASKED sequence,
        # NOT raw obs_seq. Previously this used obs_seq[:, :, _hurst_idx], i.e.
        # the unshifted current-bar Hurst, while everything else in `h` was
        # computed from `shifted = [0, obs[:-1]]`. The gate at bar t was
        # therefore seeing bar t's Hurst (look-ahead by one bar) before the
        # encoder was allowed to. Fixed 2026-05-21 RED-team audit.
        hurst = shifted[:, :, self._hurst_idx] if n_feat > self._hurst_idx else torch.zeros(B, T, device=obs_seq.device)
        h = self.regime_experts(h, hurst)      # [B, T, d_model]

        # Post-encoder feature attention
        h = self.feat_attn(h)                  # [B, T, d_model]

        h_seq = h  # Save for discriminator and outputs

        # RegimeFiLM (SOTA-2026 opt-in): conditions h_seq BEFORE VIB.
        # Identity-at-init so safe to add to a planned cold-start.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # VIB bottleneck (SOTA upgrade): stochastic compression of h_seq
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)  # [B, T, d_model] — replaces raw h_seq

        # ── RECON ANCHOR (keystone, 2026-06-11) ──────────────────────────────
        # Decode the bottlenecked latent back to the INPUT features [B,T,input_dim].
        # Decoded off `feat` (the clean bottlenecked feature, BEFORE the ATME drop
        # below) so the reconstruction target is the un-dropped representation --
        # mirrors the V12 donor (decode off z_expand(z)) + the V13 graft. The masked
        # recon-MSE in get_loss compares this vs the causally-shifted clean input.
        recon = self.recon_decoder(feat)        # [B, T, input_dim]

        # ATME on bottlenecked features (still useful as additional dropout)
        if self.training and ATME_PROB > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h.device) > ATME_PROB).float()
            feat = feat * atme_mask

        # Return predictions (from VIB bottleneck, not h_seq)
        ret_trunk = self.return_trunk(feat)    # [B, T, RETURN_HEAD_DIM]
        return_logits = {}
        for h_key in REWARD_HORIZONS:
            return_logits[h_key] = self.return_heads[str(h_key)](ret_trunk)

        # Regime prediction reads h_seq directly (regime is deterministic context)
        regime_logits = self.regime_head(h_seq)  # [B, T, 3]

        # CC-H5 quantile heads (SOTA-2026, auxiliary) — operate on feat (post-VIB)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)

        # CC-H6 regime-conditional heads (SOTA-2026, auxiliary)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "vib_mu": mu, "vib_logvar": logvar, "z_post": z,
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
            # V1.x compatibility keys
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": recon,  # [B, T, input_dim] -- REAL recon anchor (was torch.zeros stub)
            "ret_trunk": ret_trunk,
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False,
                 regime_labels=None, **kwargs):
        """Compute training loss.

        Returns: (total_loss, loss_dict, outputs) -- V1.x compatible 3-tuple.
        """
        B, T, n_feat = obs_seq.shape
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        # Token masking (random, not block)
        masked_obs = obs_seq.clone()
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()

        # Forward pass
        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # ── VIB KL loss (SOTA upgrade: stochastic bottleneck) ────────────
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        kl_anneal = kwargs.get("kl_anneal", 1.0)  # Training loop anneals
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        # ── Return losses (TwoHot CE + Huber direct) ─────────────────────
        s = self.log_vars.clamp(-6.0, 6.0)

        total = kl_weight * vib_kl
        loss_dict = {"total": 0.0, "vib_kl": vib_kl.item(), "vib_kl_weight": kl_weight}

        # ── RECON ANCHOR: reconstruction MSE (keystone, 2026-06-11) ──────────
        # recon[b,t,:] (decoded off the bottlenecked latent at position t) must
        # reconstruct the CLEAN feature the encoder consumed at that position. The
        # encoder consumes `shifted` = causal-shift of the (possibly mask_ratio-
        # dropped) input (forward_train line ~536), so the recon TARGET is the
        # causal-shift of the CLEAN obs_seq (standard masked-autoencoding: rebuild
        # the true input from a possibly-masked view). NO LOOK-AHEAD: position t
        # reconstructs obs[t-1] (a PAST bar), never a future one. RECON_WEIGHT>0
        # makes the VIB bottleneck REAL -- the latent must carry input-
        # reconstructable structure, so the heads can no longer route around z.
        # Together with the VIB KL (already in `total` above) this is the anchor
        # V11 was missing. Mirrors the V13 graft exactly.
        recon = outputs["recon"]                                 # [B, T, input_dim]
        recon_target = torch.cat(
            [torch.zeros(B, 1, n_feat, device=obs_seq.device), obs_seq[:, :-1, :]],
            dim=1)                                               # clean causal shift
        l_rec = F.mse_loss(recon, recon_target)
        total = total + RECON_WEIGHT * l_rec

        l_direct_total = torch.tensor(0.0, device=obs_seq.device)

        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue

            logits = outputs["return_logits"][h]
            tgt = targets[h]

            # Flatten for TwoHot
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tgt_flat = tgt.reshape(-1)

            # TwoHot cross-entropy
            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)  # Returns at least 7.4x weight
                total = total + torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()

            # Direct return Huber (all horizons for regularization)
            decoded = self.bucketer.decode(logits_flat)
            l_huber = F.huber_loss(decoded, tgt_flat, reduction="mean", delta=0.5)
            l_direct_total = l_direct_total + l_huber

        total = total + DIRECT_RETURN_WEIGHT * l_direct_total
        loss_dict["direct_ret"] = l_direct_total.item()

        # ── Regime loss ──────────────────────────────────────────────────
        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(
                outputs["regime_logits"].reshape(-1, 3),
                regime_tgt.reshape(-1),
            )
            s_regime = s[-1].clamp(max=-1.0)
            total = total + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()

            # Regime accuracy
            with torch.no_grad():
                pred_reg = outputs["regime_logits"].argmax(dim=-1)
                regime_acc = (pred_reg == regime_tgt).float().mean()
                loss_dict["regime_acc"] = regime_acc.item()

        # ── Adversarial loss (encoder vs discriminator) ──────────────────
        if self.training:
            h_seq = outputs["h_seq"]

            # Discriminator loss (train disc to distinguish real vs shuffled)
            with torch.no_grad():
                # Shuffle temporal order
                idx = torch.randperm(T, device=h_seq.device)
                h_shuffled = h_seq[:, idx, :]

            # Real = 1, Shuffled = 0
            d_real = self.discriminator(h_seq.detach())
            d_fake = self.discriminator(h_shuffled.detach())
            l_disc = -(torch.mean(d_real) - torch.mean(d_fake))  # WGAN

            # Gradient penalty
            alpha = torch.rand(B, 1, 1, device=h_seq.device)
            interp = (alpha * h_seq.detach() + (1 - alpha) * h_shuffled.detach()).requires_grad_(True)
            d_interp = self.discriminator(interp)
            grad = torch.autograd.grad(
                outputs=d_interp, inputs=interp,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True, retain_graph=True,
            )[0]
            gp = ((grad.norm(2, dim=[1, 2]) - 1) ** 2).mean()
            l_disc = l_disc + DISC_GRAD_PENALTY * gp

            loss_dict["disc"] = l_disc.item()

            # Adversarial loss on encoder (fool discriminator)
            d_enc = self.discriminator(h_seq)
            d_enc = torch.clamp(d_enc, -10.0, 10.0)
            l_adv = -torch.mean(d_enc)
            total = total + DISC_WEIGHT * l_adv
            loss_dict["adv"] = l_adv.item()

            # Store disc loss for separate optimizer step
            outputs["_disc_loss"] = l_disc

        # ── Directional accuracy tracking ────────────────────────────────
        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = outputs["return_logits"][h].reshape(-1, NUM_BINS)
                    decoded = self.bucketer.decode(logits_h)
                    actual = targets[h].reshape(-1)
                    nonzero = torch.abs(actual) > 1e-6
                    if nonzero.sum() > 50:
                        correct = (torch.sign(decoded[nonzero]) == torch.sign(actual[nonzero])).float()
                        loss_dict["dir_acc_%d" % h] = correct.mean().item()

        # CC-H5 quantile loss (SOTA-2026, auxiliary)
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
            for h in REWARD_HORIZONS:
                if h in targets:
                    q_pred = outputs["quantile_logits"][h]
                    l_quantile = l_quantile + _ql(q_pred, targets[h], quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile
        loss_dict["quantile_pinball"] = l_quantile.item()

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary)
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0 and regime_labels is not None):
            flat_labels = regime_labels.reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h in REWARD_HORIZONS:
                    if h not in targets:
                        continue
                    head_logits = outputs["regime_cond_logits"][r][h]
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = targets[h].reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        total = total + _rc_w * l_regime_cond
        loss_dict["regime_cond_ce"] = l_regime_cond.item()

        # Reconstruction + VIB KL anchor (2026-06-11 graft): report the REAL
        # values (were hardcoded 0.0). recon term is in `total` (RECON_WEIGHT*l_rec
        # above); VIB KL is in `total` (kl_weight*vib_kl at the top of get_loss).
        # No dream loss (V11 has no RSSM).
        loss_dict["rec"] = l_rec.item()
        loss_dict["kl"] = vib_kl.item()
        loss_dict["total"] = total.item()

        return total, loss_dict, outputs


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disc = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    return total, total - disc, disc
