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
    """Gated dilated causal convolution with residual + skip connections."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, C, T]
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
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
        for i, block in enumerate(self.blocks):
            if self.channel_transitions[i] is not None:
                x = self.channel_transitions[i](x)
            x, skip = block(x)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i](skip)
            skip_sum = skip_sum + skip

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
    """Two regime experts gated by Hurst exponent.

    Trending expert (Hurst > threshold): large dilations for momentum.
    Reverting expert (Hurst <= threshold): small dilations for mean-reversion.
    Soft blending based on Hurst distance from threshold.
    """

    def __init__(self, in_dim, expert_dim, trending_dilations, reverting_dilations,
                 hurst_threshold=0.1, dropout=0.1):
        super().__init__()
        self.expert_trending = ExpertTCN(in_dim, expert_dim, trending_dilations, dropout=dropout)
        self.expert_reverting = ExpertTCN(in_dim, expert_dim, reverting_dilations, dropout=dropout)
        self.merge_proj = nn.Linear(expert_dim, in_dim)
        self.merge_norm = RMSNorm(in_dim)
        self.hurst_threshold = hurst_threshold

    def forward(self, x, hurst_feature):
        """
        x: [B, T, D] encoder output
        hurst_feature: [B, T] hurst regime values (from input features)
        Returns: [B, T, D] (same shape as input, residual added)
        """
        # Soft gating: sigmoid centered on threshold, scaled for sharpness
        # hurst > threshold -> gate -> 1.0 (trending expert dominates)
        # hurst < threshold -> gate -> 0.0 (reverting expert dominates)
        gate = torch.sigmoid(5.0 * (hurst_feature.unsqueeze(-1) - self.hurst_threshold))

        h_trend = self.expert_trending(x)    # [B, T, expert_dim]
        h_revert = self.expert_reverting(x)  # [B, T, expert_dim]

        mixed = gate * h_trend + (1.0 - gate) * h_revert  # [B, T, expert_dim]
        projected = self.merge_proj(mixed)
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
    """

    def __init__(self, input_dim, hidden_dim=128, n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        in_d = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
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


# Frontier ceiling-breaking components
import sys as _sys
_frontier_path = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'components')
if _frontier_path not in _sys.path:
    _sys.path.insert(0, _frontier_path)
from frontier_mixin import FrontierLossMixin


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
        self.regime_experts = RegimeGatedExperts(
            in_dim=d_model,
            expert_dim=EXPERT_D_MODEL,
            trending_dilations=EXPERT_TRENDING_DILATIONS,
            reverting_dilations=EXPERT_REVERTING_DILATIONS,
            hurst_threshold=HURST_GATE_THRESHOLD,
            dropout=EXPERT_DROPOUT,
        )

        # Post-encoder feature attention
        self.feat_attn = TemporalFeatureAttention(
            d_model=d_model,
            n_heads=FEAT_ATTN_HEADS,
            dropout=FEAT_ATTN_DROPOUT,
        )

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

        # Time-shuffle discriminator (separate optimizer)
        self.discriminator = TimeShuffleDiscriminator(
            input_dim=d_model * 3,  # mean_diff + std_diff + mean_repr
            hidden_dim=DISC_HIDDEN,
            n_layers=DISC_LAYERS,
        )

        # Kendall log_vars for multi-task balancing
        # [ret_1, ret_4, ret_16, ret_64, regime]
        n_loss_terms = len(REWARD_HORIZONS) + 1
        self.log_vars = nn.Parameter(torch.tensor(
            [-2.0] * len(REWARD_HORIZONS) + [-1.5]  # Returns weighted high, regime medium
        ))

        # Hurst feature index (for regime gating)
        self._hurst_idx = 9  # hurst_regime is feature #9 in all FEATURE_LISTs

        self._init_weights()
        # Frontier ceiling-breaking heads
        self.init_frontier = FrontierLossMixin.init_frontier
        self.add_frontier_losses = FrontierLossMixin.add_frontier_losses
        FrontierLossMixin.init_frontier(self, d_model=self.d_model if hasattr(self, "d_model") else 256)

    def _init_weights(self):
        """Xavier/Kaiming initialization."""
        for name, param in self.named_parameters():
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

        # Encode
        h = self.obs_encoder(enc_input)       # [B, T, d_model]
        h = self.wavenet(h)                    # [B, T, d_model]

        # Regime-gated experts (use Hurst feature from ORIGINAL obs, not masked)
        hurst = obs_seq[:, :, self._hurst_idx] if n_feat > self._hurst_idx else torch.zeros(B, T, device=obs_seq.device)
        h = self.regime_experts(h, hurst)      # [B, T, d_model]

        # Post-encoder feature attention
        h = self.feat_attn(h)                  # [B, T, d_model]

        h_seq = h  # Save for discriminator and outputs

        # ATME: zero temporal context for some sequences (anti-memorization)
        feat = h_seq
        if self.training and ATME_PROB > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h.device) > ATME_PROB).float()
            feat = h_seq * atme_mask  # Zero entire sequences with prob ATME_PROB

        # Return predictions
        ret_trunk = self.return_trunk(feat)    # [B, T, RETURN_HEAD_DIM]
        return_logits = {}
        for h_key in REWARD_HORIZONS:
            return_logits[h_key] = self.return_heads[str(h_key)](ret_trunk)

        # Regime prediction
        regime_logits = self.regime_head(h_seq)  # [B, T, 3]

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            # V1.x compatibility keys (unused but expected by some callers)
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
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

        # ── Return losses (TwoHot CE + Huber direct) ─────────────────────
        s = self.log_vars.clamp(-6.0, 6.0)

        total = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {"total": 0.0}

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
            l_adv = -torch.mean(d_enc)  # Encoder wants disc to say "real" for everything
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

        # No reconstruction loss. No KL loss. No dream loss.
        loss_dict["rec"] = 0.0
        loss_dict["kl"] = 0.0
        
        # Frontier ceiling-breaking losses
        if hasattr(self, 'add_frontier_losses'):
            regime = targets.get('regime_label') if isinstance(targets, dict) else None
            total, loss_dict = FrontierLossMixin.add_frontier_losses(
                self, total, loss_dict, outputs, targets, obs_seq, regime)

        loss_dict["total"] = total.item()


        return total, loss_dict, outputs


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disc = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    return total, total - disc, disc
