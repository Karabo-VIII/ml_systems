"""
V12: Cross-Asset Attention Model
==================================

Processes ALL 10 assets jointly at each timestep.

Architecture:
  Per-asset: WaveNet encoder -> per-asset hidden state [D]
  Cross-asset: Multi-head attention over 10 asset states
               Each asset attends to all others at same timestep
  Output: per-asset return prediction informed by full market state

Key insight: "BTC broke out AND ETH funding negative AND SOL VPIN spiking"
is a stronger signal than any single asset's features provide.

The attention weights are interpretable: shows which assets drive each
prediction (e.g., DOGE prediction: 80% BTC, 10% ETH, 10% self).

Training interface: same get_loss/forward_train as V1.x.
Requires synchronized multi-asset batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, SwiGLU, MLPHead


# =============================================================================
# WaveNet (reused from V11, lighter config)
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        return residual, skip


class LightWaveNet(nn.Module):
    """Lightweight WaveNet for per-asset encoding."""

    def __init__(self, in_dim, channels, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = channels[-1]
        self.blocks = nn.ModuleList()
        self.ch_trans = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.ch_trans.append(nn.Conv1d(in_ch, ch, 1) if in_ch != ch else None)
            self.blocks.append(WaveNetBlock(ch, kernel_size, dil, dropout))
            self.skip_projs.append(nn.Conv1d(ch, self.out_dim, 1) if ch != self.out_dim else None)

        self.output_norm = nn.GroupNorm(min(8, self.out_dim), self.out_dim)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        skip_sum = 0
        for i, block in enumerate(self.blocks):
            if self.ch_trans[i] is not None:
                x = self.ch_trans[i](x)
            x, skip = block(x)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i](skip)
            skip_sum = skip_sum + skip
        x = self.output_norm(x + skip_sum)
        return x.transpose(1, 2)


# =============================================================================
# Cross-Asset Attention
# =============================================================================

class CrossAssetAttention(nn.Module):
    """Multi-head attention across assets at each timestep.

    Input: [B, N_assets, T, D] -- N_assets hidden states
    At each timestep t, each asset attends to all other assets at time t.
    Output: [B, N_assets, T, D] -- cross-asset informed representations
    """

    def __init__(self, d_model, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm2": RMSNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 3),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 3, d_model),
                ),
                "drop": nn.Dropout(dropout),
            }))

    def forward(self, x):
        """x: [B, A, T, D] -> [B, A, T, D]

        Reshapes to [B*T, A, D], runs attention over A dimension,
        then reshapes back. Each timestep is independent.
        """
        B, A, T, D = x.shape
        # Reshape: treat each (batch, timestep) as an independent attention problem
        x = x.permute(0, 2, 1, 3).reshape(B * T, A, D)  # [B*T, A, D]

        for layer in self.layers:
            h = layer["norm1"](x)
            h, _ = layer["attn"](h, h, h)  # No causal mask needed -- all assets are at same time
            x = x + layer["drop"](h)
            x = x + layer["drop"](layer["ffn"](layer["norm2"](x)))

        return x.reshape(B, T, A, D).permute(0, 2, 1, 3)  # [B, A, T, D]


# =============================================================================
# V12 World Model
# =============================================================================


# Frontier ceiling-breaking components
import sys as _sys
_frontier_path = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'components')
if _frontier_path not in _sys.path:
    _sys.path.insert(0, _frontier_path)
from frontier_mixin import FrontierLossMixin


class CrossAssetWorldModel(nn.Module):
    """V12: Cross-Asset Attention. Processes 10 assets jointly.

    Per-asset: obs -> causal_shift -> WaveNet -> per-asset hidden [D]
    Cross-asset: attention over 10 hidden states at each timestep
    Output: per-asset return prediction, informed by all other assets
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_assets = num_assets

        # Shared per-asset encoder (same weights for all assets)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model), nn.SiLU(), nn.Dropout(dropout),
        )
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        self.wavenet = LightWaveNet(
            d_model, WAVENET_CHANNELS, WAVENET_DILATIONS,
            WAVENET_KERNEL, WAVENET_DROPOUT,
        )

        # Cross-asset attention
        self.cross_attn = CrossAssetAttention(
            d_model, CROSS_ATTN_HEADS, CROSS_ATTN_LAYERS, CROSS_ATTN_DROPOUT,
        )

        # Per-asset return prediction (shared weights, conditioned by asset embedding)
        head_dim = RETURN_HEAD_DIM
        ret_mid = head_dim // 2
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, head_dim), RMSNorm(head_dim),
            nn.SiLU(), nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, ret_mid), RMSNorm(ret_mid),
                nn.SiLU(), nn.Linear(ret_mid, num_bins),
            )
            for h in REWARD_HORIZONS
        })
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot
        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall: [ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

        self._init_weights()
        # Frontier ceiling-breaking heads
        self.init_frontier = FrontierLossMixin.init_frontier
        self.add_frontier_losses = FrontierLossMixin.add_frontier_losses
        FrontierLossMixin.init_frontier(self, d_model=self.d_model if hasattr(self, "d_model") else 256)

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward_single_asset(self, obs_seq, asset_id):
        """Encode a single asset. Used for per-asset processing before cross-attention.

        Args:
            obs_seq: [B, T, F]
            asset_id: [B] or scalar

        Returns: [B, T, D]
        """
        B, T, n_feat = obs_seq.shape
        if isinstance(asset_id, int):
            asset_id = torch.full((B,), asset_id, dtype=torch.long, device=obs_seq.device)

        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             obs_seq[:, :-1, :]], dim=1)
        h = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))
        h = self.wavenet(h)
        return h

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """Forward pass for single-asset (V1-compatible interface).

        For single-asset evaluation, acts like V11 without cross-asset attention.
        For multi-asset training, use forward_multi_asset instead.
        """
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq
        h_seq = self.forward_single_asset(input_obs, asset_id)

        # ATME
        feat = h_seq
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device) > TEMPORAL_CTX_DROP).float()
            feat = h_seq * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {h: self.return_heads[str(h)](ret_trunk) for h in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "ret_trunk": ret_trunk,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
        }

    def forward_multi_asset(self, multi_obs, multi_asset_ids):
        """Forward pass with cross-asset attention.

        Args:
            multi_obs: [B, A, T, F] -- A assets, synchronized timestamps
            multi_asset_ids: [B, A] -- asset indices

        Returns: dict with per-asset predictions
        """
        B, A, T, F = multi_obs.shape

        # Encode each asset independently
        all_h = []
        for a in range(A):
            h_a = self.forward_single_asset(multi_obs[:, a], multi_asset_ids[:, a])
            all_h.append(h_a)
        h_stack = torch.stack(all_h, dim=1)  # [B, A, T, D]

        # Cross-asset attention
        h_cross = self.cross_attn(h_stack)  # [B, A, T, D]

        # ATME on cross-asset output
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, 1, device=h_cross.device) > TEMPORAL_CTX_DROP).float()
            h_cross = h_cross * atme_mask

        # Per-asset predictions
        all_return_logits = {h: [] for h in REWARD_HORIZONS}
        all_regime_logits = []

        for a in range(A):
            feat_a = h_cross[:, a]  # [B, T, D]
            rt = self.return_trunk(feat_a)
            for h in REWARD_HORIZONS:
                all_return_logits[h].append(self.return_heads[str(h)](rt))
            all_regime_logits.append(self.regime_head(feat_a))

        # Stack: [B, A, T, bins] for each horizon
        return {
            "return_logits": {h: torch.stack(v, dim=1) for h, v in all_return_logits.items()},
            "regime_logits": torch.stack(all_regime_logits, dim=1),
            "h_seq": h_cross,  # [B, A, T, D]
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None, **kwargs):
        """Single-asset loss (V1-compatible). For multi-asset, use get_multi_loss."""
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
        s = self.log_vars.clamp(-6.0, 6.0)
        total = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {"total": 0.0}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_flat = outputs["return_logits"][h].reshape(-1, self._num_bins)
            tgt_flat = targets[h].reshape(-1)
            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                total = total + torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            decoded = self.bucketer.decode(logits_flat)
            l_direct = l_direct + F.huber_loss(decoded, tgt_flat, reduction="mean", delta=0.5)

        total = total + DIRECT_RETURN_WEIGHT * l_direct
        loss_dict["direct_ret"] = l_direct.item()

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(outputs["regime_logits"].reshape(-1, 3), regime_tgt.reshape(-1))
            s_regime = s[-1].clamp(max=-1.0)
            total = total + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (outputs["regime_logits"].argmax(-1) == regime_tgt).float().mean().item()

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = targets[h].reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (torch.sign(dec[nz]) == torch.sign(act[nz])).float().mean().item()

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
