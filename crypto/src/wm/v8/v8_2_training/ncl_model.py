"""
V8.2.D Diversity World Model -- Multi-Head NCL Architecture

Adds K parallel return prediction paths to V8's NeuralODE-RSSM backbone.
Each head has its own return_trunk + per-horizon return_heads.
Trained with Negative Correlation Learning (NCL) to force diverse predictions.

At inference, predictions are averaged across all K heads for improved IC.
IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))

Architecture: Neural ODE encoder (continuous-time dynamics) + RSSM latent + K diverse return heads

Reference: Liu & Yao (1999) "Ensemble Learning via Negative Correlation Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from components import TwoHotSymlog, SwiGLU, MLPHead, ODEDynamics, RK4Solver, RMSNorm
from settings import *


class ReturnHead(nn.Module):
    """Single return prediction path: trunk + per-horizon heads."""

    def __init__(
        self,
        input_dim: int,
        head_dim: int = DIVERSITY_HEAD_DIM,
        num_bins: int = NUM_BINS,
        dropout: float = DIVERSITY_HEAD_DROPOUT,
        horizons: list = None,
    ):
        super().__init__()
        self.horizons = horizons or REWARD_HORIZONS

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            RMSNorm(head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                RMSNorm(head_dim // 2),
                nn.SiLU(),
                nn.Linear(head_dim // 2, num_bins),
            )
            for h in self.horizons
        })

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, T, input_dim]
        Returns:
            dict of {horizon: [B, T, num_bins]}
        """
        trunk_out = self.trunk(feat)
        return {h: self.heads[str(h)](trunk_out) for h in self.horizons}


class DiversityWorldModel(nn.Module):
    """
    V8.D: Multi-Head Diversity World Model.

    Same backbone as V8 (Neural ODE encoder + RSSM), but with K parallel
    return prediction paths trained with NCL diversity loss.

    The backbone can be:
    - Trained from scratch (mode='full')
    - Frozen from V8.2.1 checkpoint (mode='frozen_backbone')
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = WM_D_MODEL,
        ode_hidden_layers: list = None,
        latent_dim: int = RSSM_LATENT_DIM,
        classes: int = RSSM_CLASSES,
        num_bins: int = NUM_BINS,
        num_assets: int = NUM_ASSETS,
        asset_emb_dim: int = WM_ASSET_EMB_DIM,
        dropout: float = WM_DROPOUT,
        n_diversity_heads: int = DIVERSITY_N_HEADS,
        ncl_lambda: float = DIVERSITY_NCL_LAMBDA,
    ):
        super().__init__()

        if ode_hidden_layers is None:
            ode_hidden_layers = ODE_HIDDEN_LAYERS

        self.d_model = d_model
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes
        self.n_diversity_heads = n_diversity_heads
        self.ncl_lambda = ncl_lambda

        # -- Shared Backbone (same as V8 NeuralODEWorldModel) --
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Project asset-conditioned obs back to INPUT_DIM for ODE conditioning
        self.obs_proj = nn.Linear(input_dim + asset_emb_dim, input_dim)

        # Neural ODE Core
        self.dynamics = ODEDynamics(d_model, ode_hidden_layers, dropout, input_dim)
        self.solver = RK4Solver(self.dynamics, substeps=ODE_SUBSTEPS)

        # RSSM Latent Heads
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        self.posterior_head = MLPHead(d_model + input_dim, 256, self.flat_dim, dropout)

        head_input_dim = d_model + self.flat_dim

        # Decoder (shared, same as V8)
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, input_dim),
        )

        # Regime head (shared, same as V8)
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # -- K Diverse Return Heads --
        self.diversity_heads = nn.ModuleList([
            ReturnHead(head_input_dim, DIVERSITY_HEAD_DIM, num_bins, DIVERSITY_HEAD_DROPOUT)
            for _ in range(n_diversity_heads)
        ])

        # Loss balancing (same as V8)
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # TwoHot encoder
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def load_backbone_from_v8(self, v8_state_dict: dict, freeze: bool = False):
        """
        Load backbone weights from a trained V8.2 checkpoint.
        Maps V8's single return_trunk/return_heads to the first diversity head.

        Args:
            v8_state_dict: V8 NeuralODEWorldModel state_dict
            freeze: if True, freeze backbone parameters
        """
        own_state = self.state_dict()
        loaded = 0
        for name, param in v8_state_dict.items():
            # Skip V8's return_trunk and return_heads (we have diversity heads)
            if name.startswith("return_trunk") or name.startswith("return_heads"):
                # Map V8's return path to first diversity head
                if name.startswith("return_trunk."):
                    new_name = name.replace("return_trunk.", "diversity_heads.0.trunk.")
                elif name.startswith("return_heads."):
                    new_name = name.replace("return_heads.", "diversity_heads.0.heads.")
                else:
                    continue
                if new_name in own_state and own_state[new_name].shape == param.shape:
                    own_state[new_name].copy_(param)
                    loaded += 1
            elif name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1

        self.load_state_dict(own_state)
        print(f"  [OK] Loaded {loaded} params from V8.2.1 checkpoint")

        if freeze:
            # Freeze everything except diversity heads
            for name, p in self.named_parameters():
                if not name.startswith("diversity_heads"):
                    p.requires_grad = False
            print("  [OK] Backbone frozen, only diversity heads trainable")

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        shape = logits.shape
        reshaped = logits.view(*shape[:-1], self.latent_dim, self.classes)
        tau = getattr(self, '_gumbel_tau', GUMBEL_TAU)
        z = F.gumbel_softmax(reshaped, tau=tau, hard=True, dim=-1)
        return z.view(*shape)

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None, temporal_ctx_drop=0.0):
        """
        Forward pass producing K sets of return predictions.

        Returns dict with:
            - Standard V8 outputs (recon, regime, prior/post logits, etc.)
            - 'return_logits': averaged logits across all K heads
            - 'all_return_logits': list of K dicts, each {horizon: [B,T,NUM_BINS]}
            - 'ret_trunk': from first head (for adapter compatibility)
            - 'obs_for_ode': projected observations used for ODE conditioning
            - 't': time span tensor
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)
        encoded = self.initial_encoder(enc_input)

        # 2. Take h0 = encoded[:, 0, :] as initial state
        h0 = encoded[:, 0, :]

        # 3. Prepare obs for ODE conditioning — MASKED input_obs. Fixed 2026-05-21.
        obs_with_asset = torch.cat([input_obs, asset_emb], dim=-1)
        obs_for_ode = self.obs_proj(obs_with_asset)

        # 4. Time span
        t = torch.arange(T, device=obs_seq.device, dtype=torch.float32)

        # 5. Solve ODE: integrate continuous-time dynamics
        h_seq = self.solver(h0, obs_for_ode, t)

        # 6. RSSM: Prior + Posterior reads MASKED input_obs. Fixed 2026-05-21.
        prior_logits = self.prior_head(h_seq)
        post_input = torch.cat([h_seq, input_obs], dim=-1)
        post_logits = self.posterior_head(post_input)
        z_post = self._get_stoch_state(post_logits)

        # 7. Feature vector for heads
        feat = torch.cat([h_seq, z_post], dim=-1)

        recon = self.decoder(feat)  # reconstruction always uses full temporal context

        # ATME: temporal context dropout — force return/regime heads to use z_post only
        if self.training and temporal_ctx_drop > 0 and torch.rand(1).item() < temporal_ctx_drop:
            feat_heads = torch.cat([torch.zeros_like(h_seq), z_post], dim=-1)
        else:
            feat_heads = feat

        regime_logits = self.regime_head(feat_heads)

        # K diverse return predictions
        all_return_logits = []
        for head in self.diversity_heads:
            head_logits = head(feat_heads)
            all_return_logits.append(head_logits)

        # Average for primary output
        avg_return_logits = {}
        for h in REWARD_HORIZONS:
            stacked = torch.stack([rl[h] for rl in all_return_logits])
            avg_return_logits[h] = stacked.mean(dim=0)

        return {
            "recon": recon,
            "return_logits": avg_return_logits,
            "all_return_logits": all_return_logits,
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": self.diversity_heads[0].trunk(feat_heads),
            "obs_for_ode": obs_for_ode,
            "t": t,
        }

    def dynamics_regularization(
        self,
        h_seq: torch.Tensor,
        obs_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Penalize complex dynamics by computing ||f(h_t, t, obs_t)||^2
        at randomly sampled timesteps.
        """
        B, T, D = h_seq.shape
        num_samples = min(8, T)
        indices = torch.randperm(T, device=h_seq.device)[:num_samples]
        total_norm_sq = torch.tensor(0.0, device=h_seq.device)
        for idx in indices:
            h_t = h_seq[:, idx, :]
            obs_t = obs_seq[:, idx, :]
            t_val = t[idx]
            dh_dt = self.dynamics(h_t, t_val, obs_t)
            total_norm_sq = total_norm_sq + (dh_dt ** 2).mean()
        return total_norm_sq / num_samples

    def get_loss(self, obs_seq, asset_id, target_returns, mask_ratio=0.15, block_mask=True,
                 kl_anneal=1.0, gumbel_tau=GUMBEL_TAU, temporal_ctx_drop=0.0):
        """
        Compute loss with NCL diversity penalty + dynamics regularization.

        Total = standard_loss + NCL_penalty + dynamics_regularization
        """
        B, T, _ = obs_seq.shape
        self._gumbel_tau = gumbel_tau

        # Apply masking (same as V8)
        masked_obs = obs_seq.clone()
        if mask_ratio > 0:
            if block_mask:
                block_size = max(4, int(T * WM_BLOCK_SIZE_RATIO))
                num_blocks = max(1, int((T * mask_ratio) / block_size))
                for b in range(B):
                    for _ in range(num_blocks):
                        start = torch.randint(0, max(1, T - block_size), (1,)).item()
                        masked_obs[b, start:start + block_size] = 0.0
            else:
                mask = torch.rand(B, T, device=obs_seq.device) < mask_ratio
                masked_obs[mask.unsqueeze(-1).expand_as(obs_seq)] = 0.0

        outputs = self.forward_train(obs_seq, asset_id, masked_obs, temporal_ctx_drop=temporal_ctx_drop)

        # -- Standard V8 losses --
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)

        prior = outputs["prior_logits"].view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].view(-1, self.latent_dim, self.classes)
        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior)
        ).mean()
        kl_raw = l_kl.item()
        l_kl = torch.max(l_kl, torch.tensor(WM_FREE_NATS, device=obs_seq.device))

        # -- Per-head return losses + NCL --
        all_head_errors = {h: [] for h in REWARD_HORIZONS}
        horizon_losses = {}

        for h in ACTIVE_HORIZONS:
            horizon_losses[h] = torch.tensor(0.0, device=obs_seq.device)
            if h not in target_returns:
                continue

            targets = target_returns[h].reshape(-1)

            for k, head_logits in enumerate(outputs["all_return_logits"]):
                logits_k = head_logits[h].reshape(-1, NUM_BINS)
                l_k = self.bucketer.compute_loss(logits_k, targets)
                horizon_losses[h] = horizon_losses[h] + l_k / self.n_diversity_heads

                pred_k = self.bucketer.decode(logits_k)
                error_k = pred_k - targets.detach()
                all_head_errors[h].append(error_k)

        # NCL diversity penalty (Liu & Yao, 1999)
        l_ncl = torch.tensor(0.0, device=obs_seq.device)
        n_ncl_horizons = 0
        for h in REWARD_HORIZONS:
            if len(all_head_errors[h]) < 2:
                continue
            n_ncl_horizons += 1
            errors = torch.stack(all_head_errors[h])  # [K, N]
            total_error = errors.sum(dim=0)  # [N]
            for k in range(self.n_diversity_heads):
                others_error = (total_error - errors[k]).detach()
                l_ncl = l_ncl + (errors[k] * others_error).mean()
        if n_ncl_horizons > 0:
            l_ncl = l_ncl / (self.n_diversity_heads * n_ncl_horizons)

        # Regime loss
        ret_1 = target_returns.get(1, torch.zeros(B, T, device=obs_seq.device))
        with torch.no_grad():
            ret_std = ret_1.std() + 1e-6
            regime_labels = torch.ones_like(ret_1, dtype=torch.long)
            regime_labels[ret_1 > ret_std * 0.5] = 2
            regime_labels[ret_1 < -ret_std * 0.5] = 0

        regime_logits_flat = outputs["regime_logits"].reshape(-1, 3)
        regime_labels_flat = regime_labels.reshape(-1)
        ce_per_sample = F.cross_entropy(
            regime_logits_flat, regime_labels_flat, reduction="none"
        )
        p_t = torch.exp(-ce_per_sample)
        focal_weight = (1.0 - p_t) ** REGIME_FOCAL_GAMMA
        l_regime = (focal_weight * ce_per_sample).mean()

        with torch.no_grad():
            regime_preds = regime_logits_flat.argmax(dim=-1)
            regime_acc = (regime_preds == regime_labels_flat).float().mean().item()

        # Direct return regression (averaged predictions)
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                decoded = self.bucketer.decode(outputs["return_logits"][h])
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )

        # Dynamics regularization (V8-specific)
        l_dynamics = self.dynamics_regularization(
            outputs["h_seq"], outputs["obs_for_ode"], outputs["t"]
        )

        # -- Total loss with uncertainty weighting + Kendall corridors --
        s = self.log_vars.clamp(-6.0, 6.0)

        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        total = total + torch.exp(-s[1]) * kl_anneal * l_kl + 0.5 * s[1]

        for i, h in enumerate(REWARD_HORIZONS):
            idx = 2 + i
            if h not in ACTIVE_HORIZONS:
                continue
            clamp_max = RETURN_LOG_VAR_CLAMP_MAX
            s_ret = s[idx].clamp(max=clamp_max)
            total = total + torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret

        regime_idx = 2 + len(REWARD_HORIZONS)
        s_regime = s[regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        total = total + torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        # Dynamics regularization (not uncertainty-weighted, it's a penalty)
        total = total + LAMBDA_DYNAMICS * l_dynamics

        # NCL penalty (not uncertainty-weighted, it's a regularizer)
        total = total + self.ncl_lambda * l_ncl

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "dynamics_reg": l_dynamics.item(),
            "ncl": l_ncl.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, outputs

    @torch.no_grad()
    def encode_sequence(self, obs_seq, asset_id):
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = DiversityWorldModel().to(DEVICE)
    print(f"V8.2.D Diversity Model Parameters: {count_parameters(model):,}")
    print(f"  Diversity heads: {model.n_diversity_heads}")
    print(f"  NCL lambda: {model.ncl_lambda}")

    head_params = sum(p.numel() for p in model.diversity_heads.parameters())
    backbone_params = count_parameters(model) - head_params
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Diversity head params: {head_params:,} ({head_params // model.n_diversity_heads:,} per head)")

    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, outputs = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"NCL penalty: {loss_dict['ncl']:.4f}")
    print(f"Dynamics reg: {loss_dict['dynamics_reg']:.4f}")
    print(f"Breakdown: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")

    all_rl = outputs["all_return_logits"]
    for k in range(len(all_rl)):
        pred = model.bucketer.decode(all_rl[k][1])
        print(f"  Head {k} mean pred: {pred.mean().item():.6f}")

    print("\n  Testing V8.2 backbone loading...")
    from world_model import NeuralODEWorldModel
    v8 = NeuralODEWorldModel().to(DEVICE)
    model2 = DiversityWorldModel().to(DEVICE)
    model2.load_backbone_from_v8(v8.state_dict(), freeze=False)

    loss2, _, _ = model2.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"  Loss after V8 load: {loss2.item():.4f}")

    print("\n[OK] V8.2.D Diversity model sanity check passed.")
