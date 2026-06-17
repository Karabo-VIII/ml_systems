"""
V6.2.D Diversity World Model -- Multi-Head NCL Architecture (CausalJEPA variant)

Adds K parallel return prediction paths to V6's shared CausalJEPA backbone.
Each head has its own return_trunk + per-horizon return_heads.
Trained with Negative Correlation Learning (NCL) to force diverse predictions.

At inference, predictions are averaged across all K heads for improved IC.
IC_ensemble = IC_single * sqrt(K / (1 + (K-1)*rho))

V6-specific (vs V2.D):
  - CausalJEPAWorldModel backbone (CausalGRU + contrastive + TimeDiscriminator)
  - Head input dim is d_latent (192) instead of d_model+flat_dim
  - No KL loss, no Gumbel tau (JEPA has no stochastic latent)
  - Has contrastive (InfoNCE) + VICReg + adversarial losses
  - get_loss() returns 4-tuple: (total, loss_dict, l_disc, outputs)
  - TimeDiscriminator present: trained with separate discriminator optimizer
  - Dual EMA: target encoder (JEPA_EMA_DECAY per step) + full model (EMA_DECAY)

Reference: Liu & Yao (1999) "Ensemble Learning via Negative Correlation Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from components import (
    RMSNorm,
    CausalGRUEncoder,
    PredictorNetwork,
    TimeDiscriminator,
    InfoNCELoss,
    VICRegLoss,
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
    V6.D: Multi-Head Diversity World Model (CausalJEPA variant).

    Same backbone as V6 (CausalGRU + contrastive JEPA + VICReg + TimeDiscriminator),
    but with K parallel return prediction paths trained with NCL diversity loss.

    The backbone can be:
    - Trained from scratch (mode='full')
    - Loaded from V6.2.1 checkpoint (load_backbone_from_v6())

    CRITICAL V6 differences from V2:
    - TimeDiscriminator is part of backbone; get_loss returns 4-tuple
    - Dual optimizer pattern: main encoder + disc optimizer (handled in train_ncl.py)
    - l_disc is returned separately for the discriminator backward pass
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = WM_D_MODEL,
        d_latent: int = WM_D_LATENT,
        n_layers: int = WM_N_LAYERS,
        num_bins: int = NUM_BINS,
        num_assets: int = NUM_ASSETS,
        asset_emb_dim: int = WM_ASSET_EMB_DIM,
        dropout: float = WM_DROPOUT,
        n_diversity_heads: int = DIVERSITY_N_HEADS,
        ncl_lambda: float = DIVERSITY_NCL_LAMBDA,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_diversity_heads = n_diversity_heads
        self.ncl_lambda = ncl_lambda

        # -- Shared Backbone (same as V6) --
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Context encoder (online, trained via backprop)
        # CRITICAL: CausalGRU (unidirectional) -- no future leakage
        self.context_encoder = CausalGRUEncoder(d_model, d_model, n_layers, dropout)

        # Target encoder (EMA-updated, no gradients)
        self.target_encoder = CausalGRUEncoder(d_model, d_model, n_layers, dropout)
        self._copy_context_to_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Latent projectors
        self.context_latent_proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            RMSNorm(d_latent),
        )
        self.target_latent_proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            RMSNorm(d_latent),
        )
        self.target_latent_proj.load_state_dict(self.context_latent_proj.state_dict())
        for p in self.target_latent_proj.parameters():
            p.requires_grad = False

        # Predictor network
        self.predictor = PredictorNetwork(d_latent, d_latent, WM_PREDICTOR_LAYERS, dropout)

        # Time Discriminator (V6-specific adversarial component)
        self.discriminator = TimeDiscriminator(d_latent, DISC_HIDDEN, DISC_LAYERS)

        # Contrastive + VICReg losses
        self.contrastive_loss_fn = InfoNCELoss(JEPA_TEMP)
        self.vicreg_loss_fn = VICRegLoss()

        # Head input dim: V6 feeds ctx_latent (d_latent=192)
        head_input_dim = d_latent

        # Regime head (shared)
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # Auxiliary reconstruction head (shared)
        self.recon_head = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

        # -- K Diverse Return Heads --
        self.diversity_heads = nn.ModuleList([
            ReturnHead(head_input_dim, DIVERSITY_HEAD_DIM, num_bins, DIVERSITY_HEAD_DROPOUT)
            for _ in range(n_diversity_heads)
        ])

        # Loss balancing (same structure as V6)
        # Terms: contrastive, vicreg, recon, ret_1, ret_4, ret_16, ret_64, regime
        log_var_init = [0.0, 0.0, 0.0]  # contrastive, vicreg, recon
        log_var_init += [LOG_VAR_INIT_RET] * len(REWARD_HORIZONS)  # ret_1..64
        log_var_init += [LOG_VAR_INIT_REGIME]  # regime
        self.log_vars = nn.Parameter(torch.tensor(log_var_init, dtype=torch.float32))

        # TwoHot encoder
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                if hasattr(module, 'weight'):
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

        # Re-sync target encoder + target latent projector after init
        self._copy_context_to_target()
        self.target_latent_proj.load_state_dict(self.context_latent_proj.state_dict())

    def _copy_context_to_target(self):
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = JEPA_EMA_DECAY):
        """EMA update of target encoder + target latent projector."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)
        for param_q, param_k in zip(
            self.context_latent_proj.parameters(), self.target_latent_proj.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

    def load_backbone_from_v6(self, v6_state_dict: dict, freeze: bool = False):
        """
        Load backbone weights from a trained V6.2 checkpoint.
        Maps V6's single return_trunk/return_heads to the first diversity head.

        Args:
            v6_state_dict: V6 CausalJEPAWorldModel state_dict
            freeze: if True, freeze backbone parameters (except diversity heads)
        """
        own_state = self.state_dict()
        loaded = 0
        for name, param in v6_state_dict.items():
            # Map V6's return path to first diversity head
            if name.startswith("return_trunk."):
                new_name = name.replace("return_trunk.", "diversity_heads.0.trunk.")
                if new_name in own_state and own_state[new_name].shape == param.shape:
                    own_state[new_name].copy_(param)
                    loaded += 1
            elif name.startswith("return_heads."):
                new_name = name.replace("return_heads.", "diversity_heads.0.heads.")
                if new_name in own_state and own_state[new_name].shape == param.shape:
                    own_state[new_name].copy_(param)
                    loaded += 1
            elif name in own_state and own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1

        self.load_state_dict(own_state)
        print(f"  [OK] Loaded {loaded} params from V6.2.1 checkpoint")

        if freeze:
            # Freeze everything except diversity heads
            for name, p in self.named_parameters():
                if not name.startswith("diversity_heads"):
                    p.requires_grad = False
            print("  [OK] Backbone frozen, only diversity heads trainable")

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """
        Forward pass producing K sets of return predictions.

        Returns dict with:
            - Standard V6 outputs (recon, regime, ctx/tgt/pred latent)
            - 'return_logits': averaged logits across all K heads
            - 'all_return_logits': list of K dicts, each {horizon: [B,T,NUM_BINS]}
            - 'ret_trunk': from first head (for adapter compatibility)
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Shared backbone (same as V6)
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)

        # Context branch
        ctx_input = torch.cat([input_obs, asset_emb], dim=-1)
        ctx_emb = self.obs_proj(ctx_input)
        ctx_hidden = self.context_encoder(ctx_emb)
        ctx_latent = self.context_latent_proj(ctx_hidden)

        # Target branch (EMA, no grad)
        with torch.no_grad():
            tgt_input = torch.cat([obs_seq, asset_emb], dim=-1)
            tgt_emb = self.obs_proj(tgt_input)
            tgt_hidden = self.target_encoder(tgt_emb)
            tgt_latent = self.target_latent_proj(tgt_hidden)

        # Predictor
        pred_latent = self.predictor(ctx_latent)

        # Shared heads
        regime_logits = self.regime_head(ctx_latent)
        recon = self.recon_head(ctx_latent)

        # K diverse return predictions
        all_return_logits = []
        for head in self.diversity_heads:
            head_logits = head(ctx_latent)
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
            "ctx_latent": ctx_latent,
            "tgt_latent": tgt_latent,
            "pred_latent": pred_latent,
            "ret_trunk": self.diversity_heads[0].trunk(ctx_latent),
        }

    def get_loss(self, obs_seq, asset_id, target_returns, mask_ratio=0.15,
                 block_mask=True):
        """
        Compute loss with NCL diversity penalty (CausalJEPA variant).

        CRITICAL: Returns 4-tuple like V6's CausalJEPAWorldModel:
            (total_loss, loss_dict, l_disc, outputs)

        total_loss: encoder + return + NCL losses (does NOT include discriminator loss)
        l_disc: discriminator loss (optimized with separate disc_optimizer)

        NCL penalty forces each head's prediction errors to be negatively
        correlated with other heads' errors, promoting diversity.
        """
        B, T, _ = obs_seq.shape

        # Apply masking (same as V6)
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

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # -- 1. Contrastive loss (InfoNCE, per-timestep) --
        l_contrastive, contrastive_acc = self.contrastive_loss_fn(
            outputs["pred_latent"], outputs["tgt_latent"]
        )

        # -- 2. VICReg loss (collapse prevention) --
        ctx_flat = outputs["ctx_latent"].reshape(-1, self.d_latent)
        tgt_flat = outputs["tgt_latent"].reshape(-1, self.d_latent)
        l_vicreg = self.vicreg_loss_fn(
            ctx_flat, tgt_flat,
            sim_w=VICREG_SIM_WEIGHT,
            var_w=VICREG_VAR_WEIGHT,
            cov_w=VICREG_COV_WEIGHT,
        )

        # -- 3. Auxiliary reconstruction loss --
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_recon = F.mse_loss(outputs["recon"], recon_target)

        # -- 4. Adversarial loss (V6-specific TimeDiscriminator) --
        # Discriminator loss: distinguish real (temporally coherent) from shuffled
        real_score = self.discriminator(outputs["ctx_latent"].detach())  # [B]

        perm = torch.randperm(T, device=obs_seq.device)
        shuffled_latent = outputs["ctx_latent"][:, perm, :].detach()  # [B, T, D]
        fake_score = self.discriminator(shuffled_latent)  # [B]

        # Binary cross-entropy for discriminator
        l_disc = -torch.log(real_score + 1e-6).mean() - torch.log(1.0 - fake_score + 1e-6).mean()

        # Gradient penalty (WGAN-GP) for discriminator stability
        if DISC_GRAD_PENALTY > 0:
            alpha = torch.rand(B, 1, 1, device=obs_seq.device)
            interpolated = (
                alpha * outputs["ctx_latent"].detach() + (1 - alpha) * shuffled_latent
            )
            interpolated.requires_grad_(True)
            d_interp = self.discriminator(interpolated)
            grad_outputs = torch.ones_like(d_interp)
            gradients = torch.autograd.grad(
                outputs=d_interp, inputs=interpolated,
                grad_outputs=grad_outputs, create_graph=True, retain_graph=True
            )[0]
            grad_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
            l_disc = l_disc + DISC_GRAD_PENALTY * grad_penalty

        # Adversarial encoder loss: encoder wants to fool discriminator
        encoder_score = self.discriminator(outputs["ctx_latent"])  # WITH gradients to encoder
        l_adv = -torch.log(1.0 - encoder_score + 1e-6).mean()

        # -- 5. Per-head return losses + NCL --
        all_head_errors = {h: [] for h in REWARD_HORIZONS}
        horizon_losses = {}

        for h in ACTIVE_HORIZONS:
            horizon_losses[h] = torch.tensor(0.0, device=obs_seq.device)
            if h not in target_returns:
                continue

            targets = target_returns[h].reshape(-1)

            for k, head_logits in enumerate(outputs["all_return_logits"]):
                logits_k = head_logits[h].reshape(-1, NUM_BINS)
                # Per-head TwoHot loss
                l_k = self.bucketer.compute_loss(logits_k, targets)
                horizon_losses[h] = horizon_losses[h] + l_k / self.n_diversity_heads

                # Decode predictions for NCL error computation
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
                others_error = (total_error - errors[k]).detach()  # stop grad on others
                l_ncl = l_ncl + (errors[k] * others_error).mean()
        if n_ncl_horizons > 0:
            l_ncl = l_ncl / (self.n_diversity_heads * n_ncl_horizons)

        # -- 6. Regime classification loss --
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

        # -- 7. Direct return regression (averaged predictions) --
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                decoded = self.bucketer.decode(outputs["return_logits"][h])
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )

        # -- Uncertainty-weighted total loss (V6 corridor pattern) --
        s = self.log_vars.clamp(-6.0, 6.0)
        idx = 0

        # Contrastive: clamp from below
        s_con = s[idx].clamp(min=CONTRASTIVE_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_con) * l_contrastive + 0.5 * s_con
        idx += 1

        # VICReg (fixed weight)
        total = total + 0.1 * l_vicreg
        idx += 1

        # Auxiliary reconstruction (fixed small weight)
        total = total + AUX_RECON_WEIGHT * l_recon
        idx += 1

        # Return horizons: clamp from above
        for i, h in enumerate(REWARD_HORIZONS):
            if h not in ACTIVE_HORIZONS:
                idx += 1
                continue
            clamp_max = RETURN_LOG_VAR_CLAMP_MAX
            s_ret = s[idx].clamp(max=clamp_max)
            total = total + torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret
            idx += 1

        # Regime: clamped from above
        s_regime = s[idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        total = total + torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        # Adversarial encoder loss (fixed weight LAMBDA_ADV)
        total = total + LAMBDA_ADV * l_adv

        # Direct return regression
        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        # NCL penalty (not uncertainty-weighted, it's a regularizer)
        total = total + self.ncl_lambda * l_ncl

        loss_dict = {
            "total": total.item(),
            "contrastive": l_contrastive.item(),
            "contrastive_acc": contrastive_acc.item(),
            "vicreg": l_vicreg.item(),
            "recon": l_recon.item(),
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "adv": l_adv.item(),
            "disc": l_disc.item(),
            "direct_ret": l_direct_return.item(),
            "ncl": l_ncl.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        # CRITICAL: return 4-tuple matching V6's CausalJEPAWorldModel.get_loss()
        return total, loss_dict, l_disc, outputs

    @torch.no_grad()
    def encode_sequence(self, obs_seq, asset_id):
        """Encode and return averaged return predictions across K heads.

        Returns 2-tuple: (ctx_latent, return_preds) matching V6's encode_sequence.
        """
        self.eval()
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["ctx_latent"], return_preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = DiversityWorldModel().to(DEVICE)
    print(f"V6.2.D Diversity Model Parameters: {count_parameters(model):,}")
    print(f"  Diversity heads: {model.n_diversity_heads}")
    print(f"  NCL lambda: {model.ncl_lambda}")

    # Count per-head parameters
    head_params = sum(p.numel() for p in model.diversity_heads.parameters())
    disc_params = sum(p.numel() for p in model.discriminator.parameters())
    backbone_params = count_parameters(model) - head_params
    print(f"  Backbone params (incl. discriminator): {backbone_params:,}")
    print(f"  Discriminator params: {disc_params:,}")
    print(f"  Diversity head params: {head_params:,} ({head_params // model.n_diversity_heads:,} per head)")

    # Test forward
    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, l_disc, outputs = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Disc Loss: {l_disc.item():.4f}")
    print(f"NCL penalty: {loss_dict['ncl']:.4f}")
    print(f"Adv: {loss_dict['adv']:.4f}")
    print(f"Breakdown: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")

    # Verify K heads produce different predictions
    all_rl = outputs["all_return_logits"]
    for k in range(len(all_rl)):
        pred = model.bucketer.decode(all_rl[k][1])
        print(f"  Head {k} mean pred: {pred.mean().item():.6f}")

    # Test V6.2 backbone loading
    print("\n  Testing V6.2 backbone loading...")
    from world_model import CausalJEPAWorldModel
    v6 = CausalJEPAWorldModel().to(DEVICE)
    model2 = DiversityWorldModel().to(DEVICE)
    model2.load_backbone_from_v6(v6.state_dict(), freeze=False)

    loss2, _, _, _ = model2.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"  Loss after V6 load: {loss2.item():.4f}")

    # Test EMA update
    model.update_target_encoder()

    # Test encode_sequence (returns 2-tuple for JEPA)
    ctx_latent, return_preds = model.encode_sequence(obs, asset)
    print(f"  ctx_latent shape: {ctx_latent.shape}")
    print(f"  return_preds keys: {list(return_preds.keys())}")

    print("\n[OK] V6.2.D Diversity model sanity check passed.")
