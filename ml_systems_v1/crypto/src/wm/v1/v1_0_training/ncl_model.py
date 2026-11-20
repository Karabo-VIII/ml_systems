"""
V1.D Diversity World Model -- Multi-Head NCL Architecture

Adds K parallel return prediction paths to V1's shared Transformer-RSSM backbone.
Each head has its own return_trunk + per-horizon return_heads.
Trained with Negative Correlation Learning (NCL) to force diverse predictions.

At inference, predictions are averaged across all K heads for improved IC.
IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))

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

from components import (
    CausalTransformerBlock,
    RotaryEmbedding,
    RMSNorm,
    TwoHotSymlog,
    SwiGLU,
    MLPHead,
)
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
    V1.D: Multi-Head Diversity World Model.

    Same backbone as V1 (Transformer-RSSM), but with K parallel
    return prediction paths trained with NCL diversity loss.

    The backbone can be:
    - Trained from scratch (mode='full')
    - Frozen from V1 checkpoint (mode='frozen_backbone')
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = WM_D_MODEL,
        n_heads: int = WM_N_HEADS,
        n_layers: int = WM_N_LAYERS,
        d_ff: int = WM_D_FF,
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

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes  # 576
        self.n_layers = n_layers
        self.n_diversity_heads = n_diversity_heads
        self.ncl_lambda = ncl_lambda

        # -- Shared Backbone (same as V1) --

        # 1. Observation Encoder
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.rotary_emb = RotaryEmbedding(d_model // n_heads, max_len=1024)

        # 2. Transformer Core (stacked causal blocks)
        self.transformer_layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 3. RSSM Latent Heads
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        self.posterior_head = MLPHead(
            d_model + input_dim, 256, self.flat_dim, dropout
        )

        # 4. Output Heads
        head_input_dim = d_model + self.flat_dim  # 256 + 576 = 832

        # Decoder (shared, same as V1)
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, input_dim),
        )

        # Regime head (shared, same as V1)
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # Dream step projection (project combined state to d_model for Transformer input)
        self.dream_proj = nn.Linear(head_input_dim, d_model)

        # -- K Diverse Return Heads --
        self.diversity_heads = nn.ModuleList([
            ReturnHead(head_input_dim, DIVERSITY_HEAD_DIM, num_bins, DIVERSITY_HEAD_DROPOUT)
            for _ in range(n_diversity_heads)
        ])

        # Loss balancing (same as V1)
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # TwoHot encoder
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # Regime EMA buffer (non-persistent, momentum=0.99)
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with appropriate strategies per layer type."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "qkv_proj" in name or "out_proj" in name:
                    nn.init.xavier_uniform_(module.weight)
                elif "w1" in name or "w2" in name or "w_gate" in name or "w_up" in name:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                elif "w3" in name or "w_down" in name:
                    nn.init.xavier_uniform_(module.weight, gain=0.5)
                else:
                    nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def load_backbone_from_v1(self, v1_state_dict: dict, freeze: bool = False):
        """
        Load backbone weights from a trained V1 checkpoint.
        Maps V1's single return_trunk/return_heads to the first diversity head.

        Args:
            v1_state_dict: V1 TransformerWorldModel state_dict
            freeze: if True, freeze backbone parameters
        """
        own_state = self.state_dict()
        loaded = 0
        for name, param in v1_state_dict.items():
            # Map V1's return_trunk/return_heads to first diversity head
            if name.startswith("return_trunk.") or name.startswith("return_heads."):
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
        print(f"  [OK] Loaded {loaded} params from V1 checkpoint")

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
            - Standard V1 outputs (recon, regime, prior/post logits, etc.)
            - 'return_logits': averaged logits across all K heads
            - 'all_return_logits': list of K dicts, each {horizon: [B,T,NUM_BINS]}
            - 'ret_trunk': from first head (for adapter compatibility)
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)
        obs_emb = self.obs_encoder(enc_input)

        # 2. Causal shift: predict t from t-1
        obs_emb_shifted = torch.cat([
            torch.zeros(B, 1, self.d_model, device=obs_seq.device),
            obs_emb[:, :-1, :],
        ], dim=1)

        # 3. Transformer core (RoPE applied internally per layer)
        h_seq = obs_emb_shifted
        for layer in self.transformer_layers:
            h_seq = layer(h_seq, rotary_emb=self.rotary_emb)

        # 4. RSSM: Prior and Posterior
        prior_logits = self.prior_head(h_seq)
        post_input = torch.cat([h_seq, obs_seq], dim=-1)
        post_logits = self.posterior_head(post_input)
        z_post = self._get_stoch_state(post_logits)

        # 5. Decode from combined features
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, d_model + flat_dim]

        recon = self.decoder(feat)

        # ATME: temporal context dropout -- force return/regime heads to use z_post only
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
        }

    def get_loss(self, obs_seq, asset_id, target_returns, mask_ratio=0.15, block_mask=True,
                 kl_anneal=1.0, gumbel_tau=GUMBEL_TAU, label_smoothing=0.0, temporal_ctx_drop=0.0):
        """
        Compute loss with NCL diversity penalty.

        Total = standard_loss + NCL_penalty

        NCL penalty forces each head's prediction errors to be negatively
        correlated with other heads' errors, promoting diversity.
        """
        B, T, _ = obs_seq.shape
        self._gumbel_tau = gumbel_tau

        # Apply masking (same as V1)
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

        # -- Standard V1 losses --
        l_rec = F.mse_loss(outputs["recon"], obs_seq)

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

        for h in REWARD_HORIZONS:
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

        # Regime loss with EMA ret_std (non-persistent buffer)
        ret_1 = target_returns.get(1, torch.zeros(B, T, device=obs_seq.device))
        with torch.no_grad():
            batch_ret_std = ret_1.std() + 1e-6
            if self.training:
                self._regime_ret_std_ema.mul_(0.99).add_(batch_ret_std, alpha=0.01)
            ret_std = self._regime_ret_std_ema
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

        # -- Total loss with uncertainty weighting + Kendall corridors --
        s = self.log_vars.clamp(-6.0, 6.0)

        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        total = total + torch.exp(-s[1]) * kl_anneal * l_kl + 0.5 * s[1]

        for i, h in enumerate(REWARD_HORIZONS):
            idx = 2 + i
            if h not in ACTIVE_HORIZONS:
                continue
            s_ret = s[idx].clamp(max=RETURN_LOG_VAR_CLAMP_MAX)
            total = total + torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret

        regime_idx = 2 + len(REWARD_HORIZONS)
        s_regime = s[regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        total = total + torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        # Add NCL penalty (not uncertainty-weighted, it's a regularizer)
        total = total + self.ncl_lambda * l_ncl

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "ncl": l_ncl.item(),
        }
        for h in REWARD_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, outputs

    @torch.no_grad()
    def encode_sequence(self, obs_seq, asset_id):
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in REWARD_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds

    @torch.no_grad()
    def dream_step(self, h_prev, z_prev):
        """
        One-step imagination using prior (no observation).
        V1 Transformer is stateless -- no hidden state to pass.

        Args:
            h_prev: [B, d_model] -- previous Transformer output
            z_prev: [B, flat_dim] -- previous stochastic latent

        Returns:
            h_next: [B, d_model]
            z_next: [B, flat_dim]
            pred_returns: dict {horizon: [B]}
        """
        combined = torch.cat([h_prev, z_prev], dim=-1)
        h_next = self.dream_proj(combined)

        prior_logits = self.prior_head(h_next)
        z_next = self._get_stoch_state(prior_logits)

        feat = torch.cat([h_next, z_next], dim=-1)
        all_preds = {h: [] for h in REWARD_HORIZONS}
        for head in self.diversity_heads:
            head_logits = head(feat)
            for h in REWARD_HORIZONS:
                all_preds[h].append(self.bucketer.decode(head_logits[h]))

        pred_returns = {}
        for h in REWARD_HORIZONS:
            pred_returns[h] = torch.stack(all_preds[h]).mean(dim=0)

        return h_next, z_next, pred_returns


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = DiversityWorldModel().to(DEVICE)
    print(f"V1.D Diversity Model Parameters: {count_parameters(model):,}")
    print(f"  Diversity heads: {model.n_diversity_heads}")
    print(f"  NCL lambda: {model.ncl_lambda}")

    # Count per-head parameters
    head_params = sum(p.numel() for p in model.diversity_heads.parameters())
    backbone_params = count_parameters(model) - head_params
    print(f"  Backbone params: {backbone_params:,}")
    print(f"  Diversity head params: {head_params:,} ({head_params // model.n_diversity_heads:,} per head)")

    # Test forward
    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, outputs = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"NCL penalty: {loss_dict['ncl']:.4f}")
    print(f"Breakdown: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")

    # Verify K heads produce different predictions
    all_rl = outputs["all_return_logits"]
    for k in range(len(all_rl)):
        pred = model.bucketer.decode(all_rl[k][1])
        print(f"  Head {k} mean pred: {pred.mean().item():.6f}")

    # Test V1 backbone loading
    print("\n  Testing V1 backbone loading...")
    from world_model import TransformerWorldModel
    v1 = TransformerWorldModel().to(DEVICE)
    model2 = DiversityWorldModel().to(DEVICE)
    model2.load_backbone_from_v1(v1.state_dict(), freeze=False)

    loss2, _, _ = model2.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"  Loss after V1 load: {loss2.item():.4f}")

    # Test dream step
    h_seq, z_post, ret_preds = model.encode_sequence(obs, asset)
    h_last = h_seq[:, -1, :]
    z_last = z_post[:, -1, :]
    h_dream, z_dream, dream_rets = model.dream_step(h_last, z_last)
    print(f"  Dream step shapes: h={h_dream.shape}, z={z_dream.shape}")

    print("\n[OK] V1.D Diversity model sanity check passed.")
