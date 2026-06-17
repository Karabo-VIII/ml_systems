"""
V8.1 World Model -- Neural ODE with Continuous-Time Dynamics + XD Anti-Memorization

Based on V8 Neural ODE architecture with surgical anti-memorization defenses:
  - base_dim=13: Posterior/decoder restricted to base features only
  - XD dropout (70%) + noise (0.3 std) on cross-asset features
  - Prevents temporal fingerprint shortcut through XD feature sequences

Architecture:
  - Obs Encoder: Linear projection + asset embedding
  - Neural ODE Core: dh/dt = f_theta(h, t, obs_t) solved via RK4
  - RSSM Latents: Prior/Posterior categorical distributions
    - Posterior uses d_model + base_dim (not XD) to prevent temporal fingerprint shortcut
  - Heads: Reconstruction (base_dim only), Multi-Horizon Returns (TwoHot), Regime classification

Supports --features 13|18 for ablation testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

try:
    from components import TwoHotSymlog, SwiGLU, MLPHead, ODEDynamics, RK4Solver, RMSNorm
    from settings import *
except ImportError:
    from v8_1_training.components import TwoHotSymlog, SwiGLU, MLPHead, ODEDynamics, RK4Solver, RMSNorm
    from v8_1_training.settings import *


class NeuralODEWorldModel(nn.Module):
    """
    The V8.1 World Model -- Neural ODE with continuous-time market dynamics + XD Anti-Memorization.

    Inputs:
        obs_seq:  [B, T, INPUT_DIM]  -- feature sequences
        asset_id: [B]                -- integer asset index (0-4)

    Training targets:
        target_returns: dict of {horizon: [B, T] tensor} for horizons [1, 4, 16, 64]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        base_dim: int = BASE_DIM,
        d_model: int = WM_D_MODEL,
        ode_hidden_layers: list = None,
        latent_dim: int = RSSM_LATENT_DIM,
        classes: int = RSSM_CLASSES,
        num_bins: int = NUM_BINS,
        num_assets: int = NUM_ASSETS,
        asset_emb_dim: int = WM_ASSET_EMB_DIM,
        dropout: float = WM_DROPOUT,
    ):
        super().__init__()

        if ode_hidden_layers is None:
            ode_hidden_layers = ODE_HIDDEN_LAYERS

        self.d_model = d_model
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes  # 576

        # == Encoding ===========================================================
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.initial_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Project asset-conditioned obs back to INPUT_DIM for ODE conditioning
        self.obs_proj = nn.Linear(input_dim + asset_emb_dim, input_dim)

        # == Neural ODE Core ====================================================
        self.dynamics = ODEDynamics(d_model, ode_hidden_layers, dropout, input_dim)
        self.solver = RK4Solver(self.dynamics, substeps=ODE_SUBSTEPS)

        # == RSSM Latent Heads ==================================================
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        # Posterior uses only base features (not XD) to prevent temporal fingerprint shortcut
        self.posterior_head = MLPHead(d_model + base_dim, 256, self.flat_dim, dropout)

        # == Output Heads =======================================================
        head_input_dim = d_model + self.flat_dim

        # Reconstruction head (base features only -- do NOT reconstruct XD features)
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, base_dim),
        )

        # Multi-Horizon Return Heads -- SOTA: wider trunk + deeper per-horizon MLPs
        ret_dim = RETURN_HEAD_DIM       # 384
        ret_drop = RETURN_HEAD_DROPOUT  # 0.05
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

        # Regime classification: {bearish=0, neutral=1, bullish=2}
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # == Loss Balancing =====================================================
        # Kendall uncertainty weighting -- initialize to force return learning
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # == TwoHot encoder =====================================================
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # EMA of regime ret_std for stable regime labels across batches
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        # == Initialize weights =================================================
        self._init_weights()

    def _init_weights(self):
        """
        Apply proper weight initialization:
          - Xavier uniform for Linear layers
          - Normal(0, 0.02) for Embedding layers
          - Ones for RMSNorm (no bias)
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latent using Gumbel-Softmax."""
        shape = logits.shape
        reshaped = logits.view(*shape[:-1], self.latent_dim, self.classes)
        tau = getattr(self, '_gumbel_tau', GUMBEL_TAU)
        z = F.gumbel_softmax(reshaped, tau=tau, hard=True, dim=-1)
        return z.view(*shape)

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        masked_obs_seq: torch.Tensor = None,
        temporal_ctx_drop: float = 0.0,
    ):
        """
        Full forward pass for training.

        Args:
            obs_seq:        [B, T, INPUT_DIM] -- original observations (for posterior)
            asset_id:       [B] -- integer asset index
            masked_obs_seq: [B, T, INPUT_DIM] -- masked observations (for encoder input)

        Returns: dict with recon, return_logits, regime_logits, prior/post logits,
                 h_seq, z_post, obs_for_ode, t
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)               # [B, emb]
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)     # [B, T, emb]
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)    # [B, T, INPUT_DIM + emb]
        encoded = self.initial_encoder(enc_input)                 # [B, T, D]

        # 2. Take h0 = encoded[:, 0, :] as initial state
        h0 = encoded[:, 0, :]  # [B, D]

        # 3. Prepare obs for ODE conditioning — reads MASKED input_obs not raw
        # obs_seq. Fixed 2026-05-21 RED-team audit (block-mask was near no-op).
        obs_with_asset = torch.cat([input_obs, asset_emb], dim=-1)  # [B, T, INPUT_DIM + emb]
        obs_for_ode = self.obs_proj(obs_with_asset)                 # [B, T, INPUT_DIM]

        # 4. Time span
        t = torch.arange(T, device=obs_seq.device, dtype=torch.float32)

        # 5. Solve ODE: integrate continuous-time dynamics
        h_seq = self.solver(h0, obs_for_ode, t)  # [B, T, D]

        # 6. RSSM: Prior from h_seq, Posterior reads MASKED input_obs base
        # features. Fixed 2026-05-21 RED-team audit.
        prior_logits = self.prior_head(h_seq)
        post_input = torch.cat([h_seq, input_obs[:, :, :self.base_dim]], dim=-1)
        post_logits = self.posterior_head(post_input)

        # 7. Sample posterior latent
        z_post = self._get_stoch_state(post_logits)

        # 8. Feature vector for heads
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, D + flat_dim]

        # 9. Heads
        recon = self.decoder(feat)  # reconstruction always uses full temporal context

        # ATME: temporal context dropout — force return/regime heads to use z_post only
        if self.training and temporal_ctx_drop > 0 and torch.rand(1).item() < temporal_ctx_drop:
            feat_heads = torch.cat([torch.zeros_like(h_seq), z_post], dim=-1)
        else:
            feat_heads = feat

        regime_logits = self.regime_head(feat_heads)

        # Multi-horizon return predictions
        ret_trunk_out = self.return_trunk(feat_heads)
        return_logits = {}
        for h in REWARD_HORIZONS:
            return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        return {
            "recon": recon,
            "return_logits": return_logits,   # dict: {horizon: [B, T, NUM_BINS]}
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": ret_trunk_out,       # [B, T, RETURN_HEAD_DIM] -- for adapter
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

        Args:
            h_seq:   [B, T, D] trajectory of hidden states
            obs_seq: [B, T, INPUT_DIM] obs for ODE conditioning
            t:       [T] time points

        Returns: scalar mean ||f||^2
        """
        B, T, D = h_seq.shape
        num_samples = min(8, T)

        # Sample random timestep indices
        indices = torch.randperm(T, device=h_seq.device)[:num_samples]

        total_norm_sq = torch.tensor(0.0, device=h_seq.device)
        for idx in indices:
            h_t = h_seq[:, idx, :]       # [B, D]
            obs_t = obs_seq[:, idx, :]   # [B, INPUT_DIM]
            t_val = t[idx]               # scalar

            dh_dt = self.dynamics(h_t, t_val, obs_t)  # [B, D]
            total_norm_sq = total_norm_sq + (dh_dt ** 2).mean()

        return total_norm_sq / num_samples

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        kl_anneal: float = 1.0,
        gumbel_tau: float = GUMBEL_TAU,
        temporal_ctx_drop: float = 0.0,
        regime_labels: torch.Tensor = None,
    ):
        """
        Compute training loss with block masking, multi-horizon targets,
        and dynamics regularization.

        Args:
            obs_seq:        [B, T, INPUT_DIM]
            asset_id:       [B]
            target_returns: dict of {int_horizon: [B, T]} tensors
            mask_ratio:     fraction of timesteps to mask
            block_mask:     if True, mask contiguous blocks
            gumbel_tau:     Gumbel-Softmax temperature (annealed during training)
        """
        B, T, _ = obs_seq.shape

        # Set Gumbel tau for _get_stoch_state (used in forward_train)
        self._gumbel_tau = gumbel_tau

        # -- XD anti-memorization augmentation (training only) -----------------
        if self.training and self.base_dim < self.input_dim:
            xd_count = self.input_dim - self.base_dim
            xd_mask = (torch.rand(B, T, xd_count, device=obs_seq.device) > XD_DROPOUT_RATE).float()
            obs_seq = obs_seq.clone()
            obs_seq[:, :, self.base_dim:] *= xd_mask
            obs_seq[:, :, self.base_dim:] += (
                torch.randn(B, T, xd_count, device=obs_seq.device) * XD_NOISE_STD
            )

        # == Apply block masking ================================================
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

        # == Forward pass =======================================================
        outputs = self.forward_train(obs_seq, asset_id, masked_obs, temporal_ctx_drop=temporal_ctx_drop)

        # == Reconstruction loss (base features only, NOT XD) ===================
        l_rec = F.mse_loss(outputs["recon"], obs_seq[:, :, :self.base_dim])

        # == KL Divergence ======================================================
        prior = outputs["prior_logits"].view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].view(-1, self.latent_dim, self.classes)

        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior),
        ).mean()
        l_kl = torch.max(l_kl, torch.tensor(WM_FREE_NATS, device=obs_seq.device))

        # == Multi-Horizon Return Losses ========================================
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

        # == Regime classification loss =========================================
        if regime_labels is None:
            # Fallback: noisy return-based labels (backward compat)
            ret_1 = target_returns.get(1, torch.zeros(B, T, device=obs_seq.device))
            with torch.no_grad():
                batch_std = ret_1.std() + 1e-6
                if self.training:
                    self._regime_ret_std_ema = 0.99 * self._regime_ret_std_ema + 0.01 * batch_std
                ret_std = self._regime_ret_std_ema
                regime_labels = torch.ones_like(ret_1, dtype=torch.long)
                regime_labels[ret_1 > ret_std * 0.5] = 2
                regime_labels[ret_1 < -ret_std * 0.5] = 0

        ce_per_sample = F.cross_entropy(
            outputs["regime_logits"].reshape(-1, 3),
            regime_labels.reshape(-1),
            reduction="none",
        )
        p_t = torch.exp(-ce_per_sample)
        focal_weight = (1.0 - p_t) ** REGIME_FOCAL_GAMMA
        l_regime = (focal_weight * ce_per_sample).mean()

        with torch.no_grad():
            regime_preds = outputs["regime_logits"].reshape(-1, 3).argmax(dim=-1)
            regime_acc = (regime_preds == regime_labels.reshape(-1)).float().mean().item()

        # == Uncertainty-weighted total loss ====================================
        # Kendall et al. (2018): L = sum_i (exp(-s_i) * L_i + 0.5 * s_i)
        s = self.log_vars.clamp(-6.0, 6.0)

        # Rec: clamp from below -- easy task shouldn't steal gradient from returns
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        # KL: unconstrained, with annealing
        total = total + torch.exp(-s[1]) * l_kl * kl_anneal + 0.5 * s[1]

        # Returns: clamp from above -- hard task must maintain gradient priority
        for i, h in enumerate(REWARD_HORIZONS):
            idx = 2 + i
            if h not in ACTIVE_HORIZONS:
                continue
            clamp_max = RETURN_LOG_VAR_CLAMP_MAX
            s_ret = s[idx].clamp(max=clamp_max)
            total = total + torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret

        # Regime: clamped from above to maintain minimum gradient priority
        regime_idx = 2 + len(REWARD_HORIZONS)
        s_regime = s[regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        total = total + torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        # == Direct Return Regression Loss (SOTA) ==============================
        # Bypasses TwoHot discretization bottleneck with smooth Huber gradients
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                decoded = self.bucketer.decode(outputs["return_logits"][h])
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )
        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        # == Dynamics Regularization (NEW for V8) ===============================
        # Penalize ||f(h,t,obs)||^2 to encourage smooth, simple dynamics
        l_dynamics = self.dynamics_regularization(
            outputs["h_seq"], outputs["obs_for_ode"], outputs["t"]
        )
        total = total + LAMBDA_DYNAMICS * l_dynamics

        # == Build loss dict ====================================================
        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "dynamics_reg": l_dynamics.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, outputs

    # == Inference Methods ======================================================

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """
        Encode a sequence and return hidden states + posterior latents.
        Returns: h_seq, z_seq, return_preds (dict by horizon)
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    n_feat = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    feat_list, input_dim, base_dim = get_feature_config(n_feat)
    print(f"Device: {DEVICE} | features={n_feat} | input_dim={input_dim} | base_dim={base_dim}")

    model = NeuralODEWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
    print(f"V8.1 Neural ODE World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, input_dim).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"Loss: {loss.item():.4f}")
    print(f"Breakdown: {loss_dict}")

    # Test encode
    h_seq, z_seq, preds = model.encode_sequence(obs, asset)
    print(f"Hidden: {h_seq.shape}, Latent: {z_seq.shape}")
    for h_val, p in preds.items():
        print(f"  Return t+{h_val}: {p.shape}")

    print("V8.1 Neural ODE world model sanity check passed.")
