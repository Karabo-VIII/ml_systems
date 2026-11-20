"""
V6-Clean: Transformer + Time-Shuffle Discriminator
=====================================================

Stripped version: CausalTransformer -> adversarial temporal erasure.
No JEPA, no VICReg, no EMA target encoder, no reconstruction, no RSSM.

V6's unique innovation: the time-shuffle discriminator forces the encoder
to produce temporal-invariant representations. Unlike ATME (stochastic
dropout), the discriminator is a LEARNED adversary that finds and
eliminates the most temporally-informative directions in latent space.

Uses V1.0 CausalTransformerBlock (3 layers, d=256, 8 heads).
Same training interface: get_loss() returns 3-tuple (not V6-old's 4-tuple).
Discriminator loss stored in outputs["_disc_loss"] for separate optimizer.
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

from components import (
    RMSNorm, TwoHotSymlog, SwiGLU, MLPHead,
    CausalTransformerBlock, RotaryEmbedding,
)


class TimeShuffleDiscriminator(nn.Module):
    """Classifies temporal order vs shuffled. Encoder trained to fool it."""

    def __init__(self, input_dim, hidden_dim=DISC_HIDDEN, n_layers=3, dropout=0.15):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_d, hidden_dim), nn.LeakyReLU(0.2), nn.Dropout(dropout)])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, h_seq):
        """h_seq: [B, T, D] -> [B, 1]"""
        diffs = h_seq[:, 1:, :] - h_seq[:, :-1, :]
        stats = torch.cat([diffs.mean(dim=1), diffs.std(dim=1), h_seq.mean(dim=1)], dim=-1)
        return self.net(stats)


class TransformerDiscriminatorModel(nn.Module):
    """V6-Clean: Transformer encoder + adversarial temporal erasure.

    3-layer causal Transformer (same as V1.0 backbone) with time-shuffle
    discriminator. No JEPA, no VICReg, no RSSM, no reconstruction.
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 n_heads=8, n_layers=WM_N_LAYERS,
                 d_ff=None, num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 3

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

        # RoPE
        self.rotary_emb = RotaryEmbedding(d_model // n_heads, max_len=1024)

        # Causal Transformer (V1.0's proven backbone)
        self.transformer_layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Return prediction
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

        # Regime head
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # Time-shuffle discriminator (V6's unique innovation)
        self.discriminator = TimeShuffleDiscriminator(d_model * 3)

        # TwoHot
        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall: [ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                if "qkv_proj" in name or "out_proj" in name:
                    nn.init.xavier_uniform_(param)
                elif "w1" in name or "w2" in name or "w_gate" in name:
                    nn.init.kaiming_normal_(param, nonlinearity="linear")
                elif "w3" in name or "w_down" in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                else:
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             input_obs[:, :-1, :]], dim=1)
        h = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))

        # Transformer
        for layer in self.transformer_layers:
            h = layer(h, rotary_emb=self.rotary_emb)

        h_seq = h

        # ATME (30% temporal context zeroing -- stacks with discriminator)
        feat = h_seq
        if self.training:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device) > 0.30).float()
            feat = h_seq * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {h_key: self.return_heads[str(h_key)](ret_trunk) for h_key in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "ret_trunk": ret_trunk,
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

        # Adversarial temporal erasure (V6's core innovation)
        if self.training:
            h_seq = outputs["h_seq"]
            idx = torch.randperm(T, device=h_seq.device)
            h_shuffled = h_seq[:, idx, :]

            d_real = self.discriminator(h_seq.detach())
            d_fake = self.discriminator(h_shuffled.detach())
            l_disc = -(torch.mean(d_real) - torch.mean(d_fake))

            alpha = torch.rand(B, 1, 1, device=h_seq.device)
            interp = (alpha * h_seq.detach() + (1 - alpha) * h_shuffled.detach()).requires_grad_(True)
            d_interp = self.discriminator(interp)
            grad = torch.autograd.grad(d_interp, interp, torch.ones_like(d_interp),
                                        create_graph=True, retain_graph=True)[0]
            gp = ((grad.norm(2, dim=[1, 2]) - 1) ** 2).mean()
            l_disc = l_disc + DISC_GRAD_PENALTY * gp
            loss_dict["disc"] = l_disc.item()
            outputs["_disc_loss"] = l_disc

            d_enc = self.discriminator(h_seq)
            l_adv = -torch.mean(d_enc)
            total = total + LAMBDA_ADV * l_adv
            loss_dict["adv"] = l_adv.item()

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
        loss_dict["total"] = total.item()
        return total, loss_dict, outputs


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    disc = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    return total, total - disc, disc
