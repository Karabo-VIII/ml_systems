"""
V3-Clean: WaveNet-Direct Microstructure Model
================================================

Stripped version: WaveNet-TCN -> direct return prediction.
No GRU, no RSSM, no reconstruction, no dream step.

WaveNet is the best-motivated encoder for dollar-bar data:
  - Dilated causal convolutions capture multi-scale temporal patterns
  - Dilations [1,2,4,8] = receptive fields [3,5,9,17] bars
  - Gated activation (tanh * sigmoid) for selective feature propagation
  - Multi-scale skip aggregation preserves scale-specific information

Uses V1.0 components (TwoHotSymlog, SwiGLU, MLPHead, RMSNorm).
Same training interface as V1.x: get_loss() returns (loss, dict, outputs).

Training: python src/wm/v3/v3_training/train_world_model.py --features 25 --clean
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_components = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_components not in sys.path:
    sys.path.insert(0, _v1_components)

from components import RMSNorm, TwoHotSymlog, SwiGLU, MLPHead


# =============================================================================
# WaveNet Causal TCN (from V11, shared architecture)
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class WaveNetBlock(nn.Module):
    """ModernTCN-style gated causal convolution with depthwise separable conv.

    ICLR 2024 (Luo & Wang): depthwise Conv1d (groups=channels) + pointwise 1x1.
    Allows kernel_size=13 at same param cost as standard kernel_size=3.
    Wider receptive field per layer = captures longer-range patterns.
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        # Depthwise separable: DWConv(groups=C) + pointwise 1x1
        # Larger kernel (13) feasible because DWConv has C params per position, not C^2
        dw_kernel = max(kernel_size, 13)  # ModernTCN uses large kernels
        self.filter_dw = CausalConv1d(channels, channels, dw_kernel, dilation)
        self.filter_pw = nn.Conv1d(channels, channels, 1)  # Pointwise mixing
        self.gate_dw = CausalConv1d(channels, channels, dw_kernel, dilation)
        self.gate_pw = nn.Conv1d(channels, channels, 1)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Depthwise -> pointwise for both filter and gate branches
        f = self.filter_pw(self.filter_dw(x))
        g = self.gate_pw(self.gate_dw(x))
        h = torch.tanh(f) * torch.sigmoid(g)
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        return residual, skip


class WaveNetTCN(nn.Module):
    """Multi-scale WaveNet with skip aggregation."""

    def __init__(self, in_dim, channels, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = channels[-1]

        self.blocks = nn.ModuleList()
        self.channel_transitions = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.channel_transitions.append(
                nn.Conv1d(in_ch, ch, 1) if in_ch != ch else None
            )
            self.blocks.append(WaveNetBlock(ch, kernel_size, dil, dropout))
            self.skip_projs.append(
                nn.Conv1d(ch, self.out_dim, 1) if ch != self.out_dim else None
            )

        self.output_norm = nn.GroupNorm(8, self.out_dim)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)
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
        return x.transpose(1, 2)


# =============================================================================
# V3-Clean World Model
# =============================================================================


# Frontier ceiling-breaking components
import sys as _sys
_frontier_path = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'components')
if _frontier_path not in _sys.path:
    _sys.path.insert(0, _frontier_path)
from frontier_mixin import FrontierLossMixin


class WaveNetCleanModel(nn.Module):
    """V3-Clean: WaveNet-Direct. No RSSM, no reconstruction, no dream.

    Architecture:
        obs + asset_emb -> causal shift -> Linear -> WaveNet-TCN
        -> ATME (30%) -> return trunk -> h=1, h=4 heads
        -> regime head
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=GRU_HIDDEN_DIM,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Asset embedding
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        # Input projection
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # WaveNet TCN (the CORE of V3)
        self.wavenet = WaveNetTCN(
            in_dim=d_model,
            channels=TCN_CHANNELS,
            dilations=TCN_DILATIONS,
            kernel_size=TCN_KERNEL_SIZE,
            dropout=TCN_DROPOUT,
        )

        # Return prediction
        head_dim = RETURN_HEAD_DIM
        ret_mid = head_dim // 2
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, head_dim),
            RMSNorm(head_dim),
            nn.SiLU(),
            nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, ret_mid),
                RMSNorm(ret_mid),
                nn.SiLU(),
                nn.Linear(ret_mid, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Regime head
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot
        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall log_vars: [ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(torch.tensor(
            [-2.0] * len(REWARD_HORIZONS) + [-1.5]
        ))

        self._init_weights()
        # Frontier ceiling-breaking heads
        self.init_frontier = FrontierLossMixin.init_frontier
        self.add_frontier_losses = FrontierLossMixin.add_frontier_losses
        FrontierLossMixin.init_frontier(self, d_model=self.d_model if hasattr(self, "d_model") else 256)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "conv" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="linear")
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Asset embedding
        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)

        # Causal shift
        shifted = torch.cat([
            torch.zeros(B, 1, n_feat, device=obs_seq.device),
            input_obs[:, :-1, :]
        ], dim=1)
        enc_input = torch.cat([shifted, asset_emb], dim=-1)

        # Encode
        h = self.obs_encoder(enc_input)
        h_seq = self.wavenet(h)

        # ATME: zero temporal context for 30% of sequences
        feat = h_seq
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device) > TEMPORAL_CTX_DROP).float()
            feat = h_seq * atme_mask

        # Return predictions
        ret_trunk = self.return_trunk(feat)
        return_logits = {}
        for h_key in REWARD_HORIZONS:
            return_logits[h_key] = self.return_heads[str(h_key)](ret_trunk)

        # Regime
        regime_logits = self.regime_head(h_seq)

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "ret_trunk": ret_trunk,
            # V1.x compat stubs
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False,
                 regime_labels=None, **kwargs):
        B, T, n_feat = obs_seq.shape

        # Sync bucketer device
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        # Token masking
        masked_obs = obs_seq.clone()
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        s = self.log_vars.clamp(-6.0, 6.0)
        total = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {"total": 0.0}
        l_direct_total = torch.tensor(0.0, device=obs_seq.device)

        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits = outputs["return_logits"][h]
            tgt = targets[h]
            logits_flat = logits.reshape(-1, logits.shape[-1])
            tgt_flat = tgt.reshape(-1)

            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                total = total + torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()

            decoded = self.bucketer.decode(logits_flat)
            l_huber = F.huber_loss(decoded, tgt_flat, reduction="mean", delta=0.5)
            l_direct_total = l_direct_total + l_huber

        total = total + DIRECT_RETURN_WEIGHT * l_direct_total
        loss_dict["direct_ret"] = l_direct_total.item()

        # Regime loss
        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(
                outputs["regime_logits"].reshape(-1, 3),
                regime_tgt.reshape(-1),
            )
            s_regime = s[-1].clamp(max=-1.0)
            total = total + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                pred_reg = outputs["regime_logits"].argmax(dim=-1)
                loss_dict["regime_acc"] = (pred_reg == regime_tgt).float().mean().item()

        # Directional accuracy
        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                    decoded = self.bucketer.decode(logits_h)
                    actual = targets[h].reshape(-1)
                    nonzero = torch.abs(actual) > 1e-6
                    if nonzero.sum() > 50:
                        correct = (torch.sign(decoded[nonzero]) == torch.sign(actual[nonzero])).float()
                        loss_dict["dir_acc_%d" % h] = correct.mean().item()

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
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
