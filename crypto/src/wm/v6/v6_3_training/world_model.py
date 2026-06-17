"""
V6.3 World Model -- Causal JEPA + Adversarial Time Shuffling + Anti-Memorization

Architecture:
  - Context Encoder: CausalGRU (NOT BiGRU) for encoding observations (online, trained)
  - Target Encoder: EMA-updated copy of context encoder (momentum encoder)
  - Predictor Network: Predicts future target embeddings from context
  - Latent Projector: Maps encoder output to latent space
  - Time Discriminator: Adversarial classifier penalizing temporal dependence in latents
    (UNCHANGED -- operates in latent space, unaffected by feature-space anti-memorization)
  - Prediction Heads: Multi-horizon returns, regime classification
  - Auxiliary Reconstruction Head: Light decoder (base features only, NOT XD features)

Key Fix vs V6:
  V6.3 adds feature-decoupled anti-memorization:
    1. Decoder reconstructs only base features [0:base_dim] -- XD features are never
       a reconstruction target, preventing the model from memorizing XD patterns.
    2. XD features get per-timestep dropout (70%) + heavy noise during training --
       forces the model to learn cross-sectional signal that survives without XD context.
    3. TimeDiscriminator is UNCHANGED -- it operates in latent space and is not
       affected by feature-space augmentation.

Losses:
  1. InfoNCE (per-timestep, memory-safe): Contrastive alignment of predicted vs target
  2. VICReg: Prevents representation collapse (variance + covariance regularization)
  3. Auxiliary reconstruction (MSE, base features only) -- do NOT reconstruct XD
  4. Adversarial loss (encoder fools discriminator) + Discriminator loss (separate opt)
  5. Multi-horizon return prediction (TwoHot Symlog, 255 bins, [-1, 1])
  6. Direct Huber return loss (bypasses TwoHot discretization bottleneck)
  7. Regime classification (bull/neutral/bear)
  8. Uncertainty-weighted balancing (Kendall et al.)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .components import (
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
    from .settings import *
except ImportError:
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


class CausalJEPAWorldModel(nn.Module):
    """V6.3: Causal JEPA + Adversarial Time Shuffling + Anti-Memorization world model.

    Anti-memorization changes vs V6:
      - Decoder outputs only base_dim features (no XD reconstruction target)
      - XD features [base_dim:] get per-timestep dropout + noise during training
      - TimeDiscriminator is unchanged (operates in latent space)
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
        base_dim: int = BASE_DIM,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent
        self.base_dim = base_dim

        # ── Asset conditioning ───────────────────────────────────────────────
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)

        # ── Observation projection ───────────────────────────────────────────
        self.obs_proj = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Target observation projection (EMA-updated, no gradients) ────────
        # CRITICAL FIX: target branch must use its own EMA-updated obs_proj.
        self.target_obs_proj = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Context encoder (online, trained via backprop) ───────────────────
        # CRITICAL: CausalGRU, NOT BiGRU -- prevents temporal overfitting
        self.context_encoder = CausalGRUEncoder(d_model, d_model, n_layers, dropout)

        # ── Target encoder (EMA-updated, no gradients) ───────────────────────
        self.target_encoder = CausalGRUEncoder(d_model, d_model, n_layers, dropout)

        # ── Latent projectors ────────────────────────────────────────────────
        self.context_latent_proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            RMSNorm(d_latent),
        )
        self.target_latent_proj = nn.Sequential(
            nn.Linear(d_model, d_latent),
            RMSNorm(d_latent),
        )
        # Copy context projector weights to target projector
        self.target_latent_proj.load_state_dict(self.context_latent_proj.state_dict())
        for p in self.target_latent_proj.parameters():
            p.requires_grad = False

        # ── Predictor network (predicts future embeddings) ───────────────────
        self.predictor = PredictorNetwork(d_latent, d_latent, WM_PREDICTOR_LAYERS, dropout)

        # ── Time Discriminator (NEW for V6) ──────────────────────────────────
        self.discriminator = TimeDiscriminator(d_latent, DISC_HIDDEN, DISC_LAYERS)

        # ── Contrastive + VICReg losses ──────────────────────────────────────
        self.contrastive_loss_fn = InfoNCELoss(JEPA_TEMP)
        self.vicreg_loss_fn = VICRegLoss()

        # ── Prediction heads ─────────────────────────────────────────────────
        head_input_dim = d_latent

        # Multi-Horizon Return Heads -- SOTA: wider trunk + deeper per-horizon MLPs
        ret_dim = RETURN_HEAD_DIM
        ret_drop = RETURN_HEAD_DROPOUT
        self.return_trunk = nn.Sequential(
            nn.Linear(head_input_dim, ret_dim),
            RMSNorm(ret_dim),
            nn.SiLU(),
            nn.Dropout(ret_drop),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(ret_dim, ret_dim // 2),
                RMSNorm(ret_dim // 2),
                nn.SiLU(),
                nn.Linear(ret_dim // 2, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Regime classification head
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # ── Auxiliary reconstruction head (helps with small data) ────────────
        # Light decoder: latent -> base features only (no asset embedding)
        # Reconstruction head (base features only -- do NOT reconstruct XD features)
        self.recon_head = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, base_dim),
        )

        # ── Loss balancing (uncertainty-weighted) ────────────────────────────
        # Terms: contrastive, vicreg, recon, ret_1, ret_4, ret_16, ret_64, regime
        # Initialize to FORCE return learning: contrastive=0, vicreg=0, recon=0,
        # then ret horizons at LOG_VAR_INIT_RET (-2.0 -> 7.4x weight), regime=0
        log_var_init = [0.0, 0.0, 0.0]  # contrastive, vicreg, recon
        log_var_init += [LOG_VAR_INIT_RET] * len(REWARD_HORIZONS)  # ret_1..64
        log_var_init += [LOG_VAR_INIT_REGIME]  # regime
        self.log_vars = nn.Parameter(torch.tensor(log_var_init, dtype=torch.float32))

        # ── TwoHot encoder ───────────────────────────────────────────────────
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # EMA of regime ret_std for stable regime labels across batches
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        # ── Initialize target encoder as exact copy, then freeze ─────────────
        self._copy_context_to_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_obs_proj.parameters():
            p.requires_grad = False

        # ── Initialize weights ───────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training start."""
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

        # Re-sync target encoder + target projector after init
        self._copy_context_to_target()
        self.target_latent_proj.load_state_dict(self.context_latent_proj.state_dict())

    def _copy_context_to_target(self):
        """Copy context encoder weights to target encoder (hard copy)."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
        for param_q, param_k in zip(
            self.obs_proj.parameters(), self.target_obs_proj.parameters()
        ):
            param_k.data.copy_(param_q.data)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = JEPA_EMA_DECAY):
        """
        EMA update of target encoder + target latent projector from online counterparts.

        CRITICAL: Must be called EVERY step after optimizer.step(), not just every epoch.
        """
        # Update target obs_proj
        for param_q, param_k in zip(
            self.obs_proj.parameters(), self.target_obs_proj.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

        # Update target encoder
        for param_q, param_k in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

        # Update target latent projector
        for param_q, param_k in zip(
            self.context_latent_proj.parameters(), self.target_latent_proj.parameters()
        ):
            param_k.data.mul_(momentum).add_(param_q.data, alpha=1.0 - momentum)

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        masked_obs_seq: torch.Tensor = None,
    ):
        """
        Full forward pass for training.

        Args:
            obs_seq: [B, T, input_dim] - original (unmasked) observation sequence
            asset_id: [B] - asset indices
            masked_obs_seq: [B, T, input_dim] - masked version for context encoder (optional)

        Returns:
            dict with ctx_latent, tgt_latent, pred_latent, return_logits, regime_logits, recon
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # 1. Encode observations with asset embedding
        asset_emb = self.asset_embedding(asset_id)            # [B, asset_emb_dim]
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, asset_emb_dim]

        # Context branch (online encoder -- receives masked input)
        ctx_input = torch.cat([input_obs, asset_emb], dim=-1)  # [B, T, input_dim + asset_emb_dim]
        ctx_emb = self.obs_proj(ctx_input)                      # [B, T, d_model]
        ctx_hidden = self.context_encoder(ctx_emb)              # [B, T, d_model]
        ctx_latent = self.context_latent_proj(ctx_hidden)       # [B, T, d_latent]

        # Target branch (EMA encoder -- receives UNMASKED input)
        with torch.no_grad():
            tgt_input = torch.cat([obs_seq, asset_emb], dim=-1)
            tgt_emb = self.target_obs_proj(tgt_input)
            tgt_hidden = self.target_encoder(tgt_emb)
            tgt_latent = self.target_latent_proj(tgt_hidden)    # [B, T, d_latent]

        # 2. Predict future embeddings from context
        pred_latent = self.predictor(ctx_latent)                # [B, T, d_latent]

        # 3. Prediction heads (use context latent)
        ret_trunk_out = self.return_trunk(ctx_latent)
        return_logits = {}
        for h in REWARD_HORIZONS:
            return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        regime_logits = self.regime_head(ctx_latent)

        # 4. Auxiliary reconstruction (from context latent)
        recon = self.recon_head(ctx_latent)                     # [B, T, input_dim]

        return {
            "ctx_latent": ctx_latent,
            "tgt_latent": tgt_latent,
            "pred_latent": pred_latent,
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "recon": recon,
            "ret_trunk": ret_trunk_out,  # Exposed for adapter
        }

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        regime_labels: torch.Tensor = None,
    ):
        """
        Compute training loss with all objectives including adversarial.

        Loss components:
          1. InfoNCE contrastive loss (per-timestep, memory-safe)
          2. VICReg loss (collapse prevention)
          3. Auxiliary reconstruction MSE (small weight)
          4. Adversarial loss (encoder fools discriminator) + discriminator loss
          5. Multi-horizon return prediction (TwoHot Symlog)
          6. Direct Huber return loss
          7. Regime classification
          8. Uncertainty-weighted total

        Returns:
            (total_loss, loss_dict, l_disc) where:
              - total_loss includes all encoder losses (incl. adversarial encoder loss)
              - loss_dict has per-component values for logging
              - l_disc is the discriminator loss (optimized with separate optimizer)
        """
        B, T, _ = obs_seq.shape

        # ── XD anti-memorization augmentation (training only) ────────────────
        # Per-timestep dropout + heavy noise on XD features prevents the model
        # from building sequential temporal fingerprints over 96-bar windows.
        # TimeDiscriminator is UNCHANGED -- it operates in latent space.
        if self.training and self.base_dim < self.input_dim:
            xd_count = self.input_dim - self.base_dim
            xd_mask = (torch.rand(B, T, xd_count, device=obs_seq.device) > XD_DROPOUT_RATE).float()
            obs_seq = obs_seq.clone()
            obs_seq[:, :, self.base_dim:] *= xd_mask
            obs_seq[:, :, self.base_dim:] += (
                torch.randn(B, T, xd_count, device=obs_seq.device) * XD_NOISE_STD
            )

        # ── 1. Apply block masking ───────────────────────────────────────────
        masked_obs = obs_seq.clone()
        if mask_ratio > 0:
            if block_mask:
                block_size = max(4, int(T * WM_BLOCK_SIZE_RATIO))
                num_blocks = max(1, int((T * mask_ratio) / block_size))
                for b in range(B):
                    for _ in range(num_blocks):
                        start = torch.randint(0, max(1, T - block_size), (1,)).item()
                        masked_obs[b, start : start + block_size] = 0.0
            else:
                mask = torch.rand(B, T, device=obs_seq.device) < mask_ratio
                masked_obs[mask.unsqueeze(-1).expand_as(obs_seq)] = 0.0

        # ── 2. Forward pass ──────────────────────────────────────────────────
        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # ── 3. InfoNCE contrastive loss (per-timestep) ───────────────────────
        l_contrastive, contrastive_acc = self.contrastive_loss_fn(
            outputs["pred_latent"], outputs["tgt_latent"]
        )

        # ── 4. VICReg loss (collapse prevention) ─────────────────────────────
        # Flatten [B, T, D] -> [B*T, D] for VICReg computation
        ctx_flat = outputs["ctx_latent"].reshape(-1, self.d_latent)
        tgt_flat = outputs["tgt_latent"].reshape(-1, self.d_latent)
        l_vicreg = self.vicreg_loss_fn(
            ctx_flat, tgt_flat,
            sim_w=VICREG_SIM_WEIGHT,
            var_w=VICREG_VAR_WEIGHT,
            cov_w=VICREG_COV_WEIGHT,
        )

        # ── 5. Auxiliary reconstruction loss (base features only) ────────────
        l_recon = F.mse_loss(outputs["recon"], obs_seq[:, :, :self.base_dim])

        # ── 6. Adversarial loss (NEW for V6) ─────────────────────────────────
        # Discriminator loss: distinguish real (temporally coherent) from shuffled
        real_score = self.discriminator(outputs["ctx_latent"].detach())  # [B]

        # Create time-shuffled version
        perm = torch.randperm(T, device=obs_seq.device)
        shuffled_latent = outputs["ctx_latent"][:, perm, :].detach()  # [B, T, D]
        fake_score = self.discriminator(shuffled_latent)  # [B]

        # Binary cross-entropy for discriminator (wants to separate real vs fake)
        l_disc = -torch.log(real_score + 1e-6).mean() - torch.log(1.0 - fake_score + 1e-6).mean()

        # Gradient penalty (WGAN-GP) for discriminator stability
        if DISC_GRAD_PENALTY > 0:
            alpha = torch.rand(B, 1, 1, device=obs_seq.device)
            interpolated = alpha * outputs["ctx_latent"].detach() + (1 - alpha) * shuffled_latent
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
        # (make real sequences look like shuffled to the discriminator)
        encoder_score = self.discriminator(outputs["ctx_latent"])  # [B] -- WITH gradients to encoder
        l_adv = -torch.log(1.0 - encoder_score + 1e-6).mean()

        # ── 7. Multi-Horizon Return Losses ───────────────────────────────────
        horizon_losses = {}
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                logits = outputs["return_logits"][h].reshape(-1, NUM_BINS)
                targets = target_returns[h].reshape(-1)
                if RETURN_LOSS_TYPE == "crps":
                    horizon_losses[h] = self.bucketer.compute_crps_loss(logits, targets)
                else:
                    horizon_losses[h] = self.bucketer.compute_loss(logits, targets)
            else:
                horizon_losses[h] = torch.tensor(0.0, device=obs_seq.device)

        # ── 8. Regime classification loss ────────────────────────────────────
        if regime_labels is None:
            # Fallback: noisy return-based labels (backward compat)
            ret_1 = target_returns.get(1, torch.zeros(B, T, device=obs_seq.device))
            with torch.no_grad():
                batch_std = ret_1.std() + 1e-6
                if self.training:
                    self._regime_ret_std_ema = 0.99 * self._regime_ret_std_ema + 0.01 * batch_std
                ret_std = self._regime_ret_std_ema
                regime_labels = torch.ones_like(ret_1, dtype=torch.long)   # neutral
                regime_labels[ret_1 > ret_std * 0.5] = 2                   # bull
                regime_labels[ret_1 < -ret_std * 0.5] = 0                  # bear

        regime_logits_flat = outputs["regime_logits"].reshape(-1, 3)
        regime_labels_flat = regime_labels.reshape(-1)

        ce_per_sample = F.cross_entropy(
            regime_logits_flat,
            regime_labels_flat,
            reduction="none",
        )
        p_t = torch.exp(-ce_per_sample)
        focal_weight = (1.0 - p_t) ** REGIME_FOCAL_GAMMA
        l_regime = (focal_weight * ce_per_sample).mean()

        with torch.no_grad():
            regime_preds = regime_logits_flat.argmax(dim=-1)
            regime_acc = (regime_preds == regime_labels_flat).float().mean().item()

        # ── 9. Uncertainty-weighted total loss ───────────────────────────────
        # With asymmetric corridors to prevent contrastive from hogging gradients
        s = self.log_vars.clamp(-6.0, 6.0)
        idx = 0

        # Contrastive: clamp from below -- easy pretext task shouldn't steal gradient
        s_con = s[idx].clamp(min=CONTRASTIVE_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_con) * l_contrastive + 0.5 * s_con
        idx += 1

        # VICReg (fixed weight, not uncertainty-weighted -- it's a regularizer)
        total = total + 0.1 * l_vicreg
        idx += 1

        # Auxiliary reconstruction (fixed small weight)
        total = total + AUX_RECON_WEIGHT * l_recon
        idx += 1

        # Return horizons: clamp from above -- hard task must maintain gradient priority
        for i, h in enumerate(REWARD_HORIZONS):
            if h not in ACTIVE_HORIZONS:
                idx += 1
                continue
            clamp_max = RETURN_LOG_VAR_CLAMP_MAX
            s_ret = s[idx].clamp(max=clamp_max)
            total = total + torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret
            idx += 1

        # Regime: clamped from above to maintain minimum gradient priority
        s_regime = s[idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        total = total + torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        # Adversarial encoder loss (V6.3: scheduled weight, set by training loop)
        _adv_wt = getattr(self, "current_lambda_adv", LAMBDA_ADV)
        total = total + _adv_wt * l_adv

        # ── 10. Direct Return Regression Loss (SOTA) ─────────────────────────
        # Bypasses TwoHot discretization bottleneck with smooth Huber gradients
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                decoded = self.bucketer.decode(outputs["return_logits"][h])
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )
        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        # ── Build loss dict for logging ──────────────────────────────────────
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
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, l_disc, outputs

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """
        Encode a sequence and return latent representations + return predictions.
        Used for downstream tasks (agent training, evaluation).

        Args:
            obs_seq: [B, T, input_dim]
            asset_id: [B]

        Returns:
            (ctx_latent, return_preds) where:
              ctx_latent: [B, T, d_latent]
              return_preds: dict {horizon: [B, T]}
        """
        self.eval()

        B, T, _ = obs_seq.shape
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)

        ctx_input = torch.cat([obs_seq, asset_emb], dim=-1)
        ctx_emb = self.obs_proj(ctx_input)
        ctx_hidden = self.context_encoder(ctx_emb)
        ctx_latent = self.context_latent_proj(ctx_hidden)

        # Return predictions
        ret_trunk_out = self.return_trunk(ctx_latent)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(self.return_heads[str(h)](ret_trunk_out))

        return ctx_latent, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    # Test with 18 features (full)
    model_18 = CausalJEPAWorldModel(input_dim=18, base_dim=13).to(DEVICE)
    total_params = count_parameters(model_18)
    print(f"V6.3 Causal JEPA+Adversarial+AntiMem World Model Parameters: {total_params:,}")

    B, T = 4, 96
    obs_18 = torch.randn(B, T, 18).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, l_disc, _ = model_18.get_loss(obs_18, asset, targets, mask_ratio=0.15)
    print(f"[18f] Total Loss: {loss.item():.4f}")
    print(f"[18f] Disc Loss:  {l_disc.item():.4f}")
    print(f"[18f] Recon shape matches base_dim: recon={model_18.recon_head[-1].out_features}")

    # Test EMA update
    model_18.update_target_encoder()

    # Test encode_sequence
    latent, ret_preds = model_18.encode_sequence(obs_18, asset)
    print(f"[18f] Latent shape: {latent.shape}")
    print(f"[18f] Return pred shapes: { {h: v.shape for h, v in ret_preds.items()} }")

    # Test with 14 features (base only)
    model_13 = CausalJEPAWorldModel(input_dim=13, base_dim=13).to(DEVICE)
    obs_14 = torch.randn(B, T, 14).to(DEVICE)
    loss_13, loss_dict_13, l_disc_13, _ = model_13.get_loss(obs_14, asset, targets, mask_ratio=0.15)
    print(f"[13f] Total Loss: {loss_13.item():.4f}")
    print(f"[13f] Disc Loss:  {l_disc_13.item():.4f}")

    param_mb = total_params * 4 / 1024 / 1024
    print(f"\nParameter memory: {param_mb:.1f} MB")
    print(f"InfoNCE memory per step: {B * B * 4 * T / 1024 / 1024:.2f} MB (per-timestep)")

    print("\nV6.3 Causal JEPA+Adversarial+AntiMem world model sanity check passed.")
