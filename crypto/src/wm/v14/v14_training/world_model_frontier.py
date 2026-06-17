"""
V14: Diffusion Return Distribution Model
==========================================

Generates the FULL DISTRIBUTION of possible returns, not a point estimate.

Instead of predicting E[return] = +0.3%, predicts:
  "returns could be [-2%, -0.5%, +0.3%, +1%, +3%] with these probabilities"

Position sizing from distribution SHAPE:
  - High mean, low variance -> full size (confident directional)
  - High mean, high variance -> half size (right direction but risky)
  - Bimodal (crash or rally) -> small size or skip

Architecture:
  1. WaveNet encoder -> condition embedding c [B, T, D]
  2. Diffusion denoiser: takes noisy return r_t + condition c
     -> predicts noise epsilon (standard DDPM)
  3. Training: add noise to actual returns, learn to denoise
  4. Inference: start from pure noise, denoise N steps -> return samples

Also produces TwoHot logits for V1-compatible interface.
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

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, MLPHead


# =============================================================================
# WaveNet Encoder (condition extractor, same as V11/V12)
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
        self.norm = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        return residual, skip


class ConditionEncoder(nn.Module):
    """WaveNet encoder that produces condition embeddings for denoiser."""

    def __init__(self, in_dim, d_model, channels, dilations, kernel=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = d_model
        self.blocks = nn.ModuleList()
        self.ch_trans = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        out_ch = channels[-1]

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.ch_trans.append(nn.Conv1d(in_ch, ch, 1) if in_ch != ch else None)
            self.blocks.append(WaveNetBlock(ch, kernel, dil, dropout))
            self.skip_projs.append(nn.Conv1d(ch, out_ch, 1) if ch != out_ch else None)

        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.Conv1d(out_ch, d_model, 1),
        )

    def forward(self, x):
        """x: [B, T, F] -> [B, T, D]"""
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
        x = self.output_proj(x + skip_sum)
        return x.transpose(1, 2)


# =============================================================================
# Diffusion Denoiser
# =============================================================================

class SinusoidalTimeEmb(nn.Module):
    """Sinusoidal time embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """t: [B] float in [0, 1] -> [B, dim]"""
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ReturnDenoiser(nn.Module):
    """Denoises return values conditioned on encoder output.

    Input: noisy return r_t [B, T, 1] + condition c [B, T, D] + timestep t [B]
    Output: predicted noise epsilon [B, T, 1]
    """

    def __init__(self, cond_dim, hidden_dim=256, n_layers=3, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalTimeEmb(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)

        # Input: noisy_return(1) + condition(cond_dim) -> hidden
        self.input_proj = nn.Linear(1 + cond_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm": nn.LayerNorm(hidden_dim),
                "fc": nn.Linear(hidden_dim, hidden_dim),
                "act": nn.SiLU(),
                "drop": nn.Dropout(dropout),
            }))

        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_return, condition, timestep):
        """
        noisy_return: [B, T, 1]
        condition: [B, T, D]
        timestep: [B] float in [0, 1]
        Returns: [B, T, 1] predicted noise
        """
        # Time embedding
        t_emb = self.time_proj(F.silu(self.time_emb(timestep)))  # [B, H]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, H]

        # Combine inputs
        h = self.input_proj(torch.cat([noisy_return, condition], dim=-1))  # [B, T, H]
        h = h + t_emb  # Add time conditioning

        for layer in self.layers:
            residual = h
            h = layer["norm"](h)
            h = layer["act"](layer["fc"](h))
            h = layer["drop"](h)
            h = residual + h

        return self.output_proj(h)  # [B, T, 1]


# =============================================================================
# V14 World Model
# =============================================================================


# Frontier ceiling-breaking components
import sys as _sys
_frontier_path = str(__import__('pathlib').Path(__file__).resolve().parent.parent.parent / 'components')
if _frontier_path not in _sys.path:
    _sys.path.insert(0, _frontier_path)
from frontier_mixin import FrontierLossMixin


class DiffusionWorldModel(nn.Module):
    """V14: Diffusion Return Distribution.

    Trains: add noise to actual returns -> learn to denoise
    Inference: start from noise -> denoise -> return samples -> statistics
    Also produces TwoHot logits for V1-compatible evaluation.
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

        # Obs encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model), nn.SiLU(), nn.Dropout(dropout),
        )

        # Condition encoder (WaveNet)
        self.condition_encoder = ConditionEncoder(
            d_model, d_model, WAVENET_CHANNELS, WAVENET_DILATIONS,
            WAVENET_KERNEL, WAVENET_DROPOUT,
        )

        # Per-horizon denoisers
        self.denoisers = nn.ModuleDict({
            str(h): ReturnDenoiser(d_model, DENOISER_HIDDEN, DENOISER_LAYERS, dropout)
            for h in ACTIVE_HORIZONS
        })

        # Diffusion schedule (linear beta)
        steps = DIFFUSION_STEPS
        betas = torch.linspace(DIFFUSION_BETA_START, DIFFUSION_BETA_END, steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1 - alpha_cumprod))

        # TwoHot return heads (for V1-compatible evaluation)
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

    def _q_sample(self, x_0, t, noise=None):
        """Forward diffusion: add noise to clean data at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise, noise

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             input_obs[:, :-1, :]], dim=1)
        h = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))
        condition = self.condition_encoder(h)

        h_seq = condition

        # ATME (V14 base uses VIB + diffusion noise; frontier clean path uses ATME)
        feat = h_seq
        _atme_prob = 0.30
        if self.training and _atme_prob > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device) > _atme_prob).float()
            feat = h_seq * atme_mask

        # TwoHot return predictions (V1-compatible)
        ret_trunk = self.return_trunk(feat)
        return_logits = {h_key: self.return_heads[str(h_key)](ret_trunk) for h_key in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "condition": condition, "ret_trunk": ret_trunk,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
        }

    @torch.no_grad()
    def sample_returns(self, condition, horizon=1, n_samples=None):
        """Generate return distribution samples via reverse diffusion.

        Args:
            condition: [B, T, D] from encoder
            horizon: which horizon to sample
            n_samples: number of samples (default from settings)

        Returns: [B, T, n_samples] return samples
        """
        n_samples = n_samples or DIFFUSION_N_SAMPLES
        B, T, D = condition.shape
        denoiser = self.denoisers[str(horizon)]

        all_samples = []
        for _ in range(n_samples):
            # Start from pure noise
            x = torch.randn(B, T, 1, device=condition.device)

            # Reverse diffusion (DDPM)
            for t in reversed(range(DIFFUSION_INFERENCE_STEPS)):
                t_tensor = torch.full((B,), t / DIFFUSION_STEPS, device=condition.device)
                noise_pred = denoiser(x, condition, t_tensor)

                # DDPM update step
                alpha_t = self.alphas[t]
                alpha_bar_t = self.alpha_cumprod[t]
                beta_t = self.betas[t]

                x = (1 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * noise_pred
                )

                # Add noise (except at t=0)
                if t > 0:
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)

            all_samples.append(x.squeeze(-1))  # [B, T]

        return torch.stack(all_samples, dim=-1)  # [B, T, n_samples]

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
        condition = outputs["condition"]

        s = self.log_vars.clamp(-6.0, 6.0)
        total = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {"total": 0.0}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # TwoHot losses (standard, V1-compatible)
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

        # Diffusion denoising losses (per active horizon)
        l_diffusion = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h not in targets:
                continue
            denoiser = self.denoisers[str(h)]
            x_0 = targets[h].unsqueeze(-1)  # [B, T, 1]

            # Random diffusion timestep per sample
            t = torch.randint(0, DIFFUSION_STEPS, (B,), device=obs_seq.device)
            noisy, noise = self._q_sample(x_0, t)
            t_float = t.float() / DIFFUSION_STEPS

            # Predict noise
            noise_pred = denoiser(noisy, condition, t_float)
            l_diff_h = F.mse_loss(noise_pred, noise)
            l_diffusion = l_diffusion + l_diff_h
            loss_dict["diff_%d" % h] = l_diff_h.item()

        total = total + l_diffusion  # Equal weight with TwoHot
        loss_dict["diffusion"] = l_diffusion.item()

        # Regime
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
