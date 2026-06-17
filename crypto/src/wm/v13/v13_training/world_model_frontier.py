"""
V13: Temporal Fusion Transformer (TFT)
========================================

Google's time-series SOTA adapted for dollar-bar crypto.

Key innovation: Variable Selection Networks (VSN) learn per-timestep
which features matter. Instead of treating all 25 features equally,
the model learns: "at this bar, VPIN and flow matter; ignore the rest."

Architecture:
  1. Variable Selection Network: [B,T,F] -> soft feature gates [B,T,F] x features
  2. GRN encoding: gated residual networks process selected features
  3. Temporal self-attention: interpretable multi-head attention
  4. GRN decoding: decode to return predictions

No RSSM, no reconstruction, no dream. Same get_loss interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, MLPHead


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT. Gated skip connection + ELU + dropout."""

    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1, context_dim=0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(residual + gate * out)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns per-timestep feature importance weights.

    Input: [B, T, F] features
    Output: [B, T, D] weighted combination, [B, T, F] weights (for interpretability)
    """

    def __init__(self, n_features, d_model, hidden_dim, dropout=0.1, context_dim=0):
        super().__init__()
        self.n_features = n_features
        # Per-feature GRN transforms
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, d_model, dropout, context_dim)
            for _ in range(n_features)
        ])
        # Softmax gate over features
        self.gate_grn = GatedResidualNetwork(
            n_features, hidden_dim, n_features, dropout, context_dim
        )

    def forward(self, x, context=None):
        """x: [B, T, F] -> ([B, T, D], [B, T, F] weights)"""
        B, T, F = x.shape
        # Compute feature weights
        weights = torch.softmax(self.gate_grn(x, context), dim=-1)  # [B, T, F]

        # Transform each feature independently
        transformed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]  # [B, T, 1]
            ctx = context if context is not None else None
            transformed.append(self.feature_grns[i](feat_i, ctx))  # [B, T, D]

        # Stack and weight
        stacked = torch.stack(transformed, dim=2)  # [B, T, F, D]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        return weighted, weights


class InterpretableAttention(nn.Module):
    """Multi-head attention that exposes attention weights for interpretability."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, D], attn_weights [B, H, T, T]"""
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return residual + out, attn_weights



# Frontier ceiling-breaking components
import sys as _sys
_frontier_path = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'components')
if _frontier_path not in _sys.path:
    _sys.path.insert(0, _frontier_path)
from frontier_mixin import FrontierLossMixin


class TFTWorldModel(nn.Module):
    """V13: Temporal Fusion Transformer. Variable selection + interpretable attention."""

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Asset embedding (used as context for VSN)
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(
            input_dim, d_model, TFT_VSN_HIDDEN, dropout, context_dim=asset_emb_dim
        )

        # Temporal encoding (position + GRN)
        self.temporal_grn = GatedResidualNetwork(d_model, TFT_GRN_HIDDEN, d_model, dropout)

        # Interpretable attention layers
        self.attn_layers = nn.ModuleList([
            InterpretableAttention(d_model, TFT_N_HEADS, dropout)
            for _ in range(TFT_N_LAYERS)
        ])

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(d_model, TFT_GRN_HIDDEN, d_model, dropout)

        # Return heads
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

        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

        # Frontier components
        self.add_frontier_losses = FrontierLossMixin.add_frontier_losses
        FrontierLossMixin.init_frontier(self, d_model=d_model)

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Causal shift
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             input_obs[:, :-1, :]], dim=1)

        # Asset context for VSN
        asset_ctx = self.asset_embedding(asset_id)  # [B, emb]
        asset_ctx_t = asset_ctx.unsqueeze(1).expand(-1, T, -1)  # [B, T, emb]

        # Variable Selection: learns which features matter per timestep
        selected, feature_weights = self.vsn(shifted, asset_ctx_t)  # [B,T,D], [B,T,F]

        # Temporal GRN
        h = self.temporal_grn(selected)

        # Interpretable attention
        all_attn = []
        for attn_layer in self.attn_layers:
            h, attn_w = attn_layer(h)
            all_attn.append(attn_w)

        h_seq = self.post_attn_grn(h)

        # ATME
        feat = h_seq
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device) > TEMPORAL_CTX_DROP).float()
            feat = h_seq * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {h_key: self.return_heads[str(h_key)](ret_trunk) for h_key in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "ret_trunk": ret_trunk,
            "feature_weights": feature_weights,  # [B, T, F] for interpretability
            "attention_weights": all_attn,        # List of [B, H, T, T]
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None, **kwargs):
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

        # Feature selection entropy (track how many features the VSN actually uses)
        with torch.no_grad():
            fw = outputs["feature_weights"]  # [B, T, F]
            ent = -(fw * torch.log(fw + 1e-8)).sum(dim=-1).mean()
            loss_dict["vsn_entropy"] = ent.item()  # High = uniform, low = selective

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
