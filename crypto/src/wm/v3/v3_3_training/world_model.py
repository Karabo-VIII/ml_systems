"""
V3.3 World Model — WaveNet-GRU Hybrid with RSSM + XD Anti-Memorization

Based on V3 WaveNet-GRU architecture with surgical anti-memorization defenses:
  - base_dim=13: Posterior/decoder restricted to base features only
  - XD dropout (70%) + noise (0.3 std) on cross-asset features
  - Prevents temporal fingerprint shortcut through XD feature sequences

Architecture:
  1. Obs Encoder: Linear(input_dim+32 -> 96) + RMSNorm + SiLU
  2. WaveNet TCN: 4 gated dilated causal conv layers, dilations [1,2,4,8]
  3. MultiScaleAggregator: Combine skip connections from all 4 scales
  4. CausalGRU: 2-layer GRU for sequential dynamics
  5. RSSM Latents: Prior/Posterior (24x24 categorical) -- posterior uses base_dim only
  6. Heads: Reconstruction (base_dim), Multi-Horizon Returns, Regime

Supports --features 13|18 for ablation testing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

try:
    from components import (
        RMSNorm,
        WaveNetTCN,
        MultiScaleAggregator,
        CausalGRU,
        TwoHotSymlog,
        SwiGLU,
        MLPHead,
    )
    from settings import *
except ImportError:
    from v3_3_training.components import (
        RMSNorm,
        WaveNetTCN,
        MultiScaleAggregator,
        CausalGRU,
        TwoHotSymlog,
        SwiGLU,
        MLPHead,
    )
    from v3_3_training.settings import *


class WaveNetGRUWorldModel(nn.Module):
    """V3.3: WaveNet-GRU Hybrid World Model with RSSM latents + XD Anti-Memorization."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        base_dim: int = BASE_DIM,
        tcn_channels: list = None,
        tcn_kernel: int = TCN_KERNEL_SIZE,
        tcn_dilations: list = None,
        gru_hidden: int = GRU_HIDDEN_DIM,
        gru_layers: int = GRU_NUM_LAYERS,
        latent_dim: int = RSSM_LATENT_DIM,
        classes: int = RSSM_CLASSES,
        num_bins: int = NUM_BINS,
        num_assets: int = NUM_ASSETS,
        asset_emb_dim: int = WM_ASSET_EMB_DIM,
        dropout: float = WM_DROPOUT,
        tcn_dropout: float = TCN_DROPOUT,
    ):
        super().__init__()

        if tcn_channels is None:
            tcn_channels = TCN_CHANNELS
        if tcn_dilations is None:
            tcn_dilations = TCN_DILATIONS

        self.gru_hidden = gru_hidden
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes  # 576

        # =====================================================================
        # 1. Observation Encoder
        # =====================================================================
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, tcn_channels[0]),
            RMSNorm(tcn_channels[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # =====================================================================
        # 2. WaveNet TCN (gated dilated causal convolutions)
        # =====================================================================
        self.wavenet = WaveNetTCN(
            input_dim=tcn_channels[0],
            channels=tcn_channels,
            kernel_size=tcn_kernel,
            dilations=tcn_dilations,
            dropout=tcn_dropout,
        )

        # =====================================================================
        # 3. Multi-Scale Aggregator
        # =====================================================================
        self.aggregator = MultiScaleAggregator(
            channels=tcn_channels[-1],
            out_channels=tcn_channels[-1],
        )

        # =====================================================================
        # 4. Causal GRU for sequential dynamics (V3.3: bypassed when USE_GRU=False)
        # =====================================================================
        _use_gru = globals().get("USE_GRU", True)
        self.use_gru = _use_gru
        if _use_gru:
            self.gru = CausalGRU(
                input_dim=tcn_channels[-1],
                hidden_dim=gru_hidden,
                num_layers=gru_layers,
                dropout=dropout,
            )

        # =====================================================================
        # 5. RSSM Latent Heads
        # =====================================================================
        self.prior_head = MLPHead(gru_hidden, 256, self.flat_dim, dropout)
        # Posterior uses only base features (not XD) to prevent temporal fingerprint shortcut
        self.posterior_head = MLPHead(
            gru_hidden + base_dim, 256, self.flat_dim, dropout
        )

        # =====================================================================
        # 6. Output Heads
        # =====================================================================
        head_input_dim = gru_hidden + self.flat_dim

        # Reconstruction head (base features only -- do NOT reconstruct XD features)
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, base_dim),
        )

        # Multi-Horizon Return Heads — SOTA: wider trunk + deeper per-horizon MLPs
        ret_dim = RETURN_HEAD_DIM  # 384
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

        # Regime classification head (3 classes: bearish/neutral/bullish)
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # =====================================================================
        # Loss balancing (uncertainty-weighted)
        # =====================================================================
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # Dream step projection (projects combined state to GRU input dim)
        self.dream_proj = nn.Linear(head_input_dim, tcn_channels[-1])

        # TwoHot encoder
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # EMA of regime ret_std for stable regime labels across batches
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        # Apply weight initialization
        self._init_weights()

    # =========================================================================
    # WEIGHT INITIALIZATION
    # =========================================================================

    def _init_weights(self):
        """Initialize weights with appropriate strategies per layer type."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.GRU):
                for param_name, param in module.named_parameters():
                    if "weight_ih" in param_name:
                        nn.init.xavier_uniform_(param)
                    elif "weight_hh" in param_name:
                        nn.init.orthogonal_(param)
                    elif "bias" in param_name:
                        nn.init.zeros_(param)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    # =========================================================================
    # RSSM STOCHASTIC STATE
    # =========================================================================

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latent using Gumbel-Softmax.

        AMP-safe: Gumbel noise saturates to inf in fp16. Force fp32.
        """
        shape = logits.shape
        in_dtype = logits.dtype
        reshaped = logits.view(*shape[:-1], self.latent_dim, self.classes)
        tau = getattr(self, '_gumbel_tau', GUMBEL_TAU)
        with torch.amp.autocast("cuda", enabled=False):
            z = F.gumbel_softmax(reshaped.float(), tau=tau, hard=True, dim=-1)
        return z.to(in_dtype).view(*shape)

    # =========================================================================
    # FORWARD PASS (TRAINING)
    # =========================================================================

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
            obs_seq: [B, T, INPUT_DIM] — raw observations
            asset_id: [B] — asset indices
            masked_obs_seq: [B, T, INPUT_DIM] — masked observations (optional)
            temporal_ctx_drop: prob of zeroing h_seq in return/regime heads (ATME)

        Returns:
            dict with recon, return_logits, regime_logits, prior/post logits, h_seq, z_post
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, emb_dim]
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)  # [B, T, 13+32]
        obs_emb = self.obs_encoder(enc_input)  # [B, T, 96]

        # 2. Causal shift: predict t from t-1 (applied BEFORE WaveNet to prevent
        #    target observation leaking into temporal convolution receptive field)
        obs_emb_shifted = torch.cat([
            torch.zeros(B, 1, obs_emb.size(2), device=obs_seq.device),
            obs_emb[:, :-1, :],
        ], dim=1)

        # 3. WaveNet TCN for multi-scale temporal features
        tcn_out, skips = self.wavenet(obs_emb_shifted)  # [B, T, 256], list of [B, 256, T]

        # 4. Multi-scale aggregation of skip connections
        agg_out = self.aggregator(skips)  # [B, T, 256]

        # 5. GRU for sequential dynamics (V3.3: bypassed when USE_GRU=False)
        if self.use_gru:
            h_seq, _ = self.gru(agg_out)  # [B, T, gru_hidden]
        else:
            h_seq = agg_out  # [B, T, tcn_channels[-1]] (same dim as gru_hidden)

        # 6. RSSM: Prior and Posterior
        prior_logits = self.prior_head(h_seq)
        # Posterior reads MASKED input_obs base features. Fixed 2026-05-21
        # RED-team audit — block-mask MAE leak through posterior path.
        post_input = torch.cat([h_seq, input_obs[:, :, :self.base_dim]], dim=-1)
        post_logits = self.posterior_head(post_input)
        z_post = self._get_stoch_state(post_logits)

        # 7. Decode from combined features
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, gru_hidden + flat_dim]

        recon = self.decoder(feat)  # reconstruction always uses full temporal context

        # ATME: anti-temporal-memorization via obs-only posterior
        if self.training and temporal_ctx_drop > 0 and torch.rand(1).item() < temporal_ctx_drop:
            # Full ATME: obs-only posterior blocks temporal leakage through z_post
            post_input_obs = torch.cat([torch.zeros_like(h_seq), obs_seq[:, :, :self.base_dim]], dim=-1)
            post_logits_obs = self.posterior_head(post_input_obs)
            z_post_obs = self._get_stoch_state(post_logits_obs)
            feat_heads = torch.cat([torch.zeros_like(h_seq), z_post_obs], dim=-1)
        else:
            # Normal: h_seq.detach() — GRU learns from recon/KL only, return heads
            # READ temporal features but can't OPTIMIZE GRU for memorization
            feat_heads = torch.cat([h_seq.detach(), z_post], dim=-1)

        regime_logits = self.regime_head(feat_heads)

        # Multi-horizon return predictions
        ret_trunk_out = self.return_trunk(feat_heads)
        return_logits = {}
        for h in REWARD_HORIZONS:
            return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        return {
            "recon": recon,
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": ret_trunk_out,
        }

    # =========================================================================
    # LOSS COMPUTATION
    # =========================================================================

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
        Compute training loss with block masking and multi-horizon targets.

        Args:
            obs_seq: [B, T, INPUT_DIM]
            asset_id: [B]
            target_returns: dict {horizon: [B, T]}
            mask_ratio: fraction of timesteps to mask
            block_mask: use contiguous block masking vs random
            gumbel_tau: Gumbel-Softmax temperature (annealed during training)

        Returns:
            total_loss: scalar
            loss_dict: dict of individual loss components
        """
        B, T, _ = obs_seq.shape
        self._gumbel_tau = gumbel_tau

        # -- XD anti-memorization augmentation (training only) -----------------
        # Per-timestep dropout + heavy noise on XD features prevents the model
        # from building sequential temporal fingerprints over 96-bar windows.
        if self.training and self.base_dim < self.input_dim:
            xd_count = self.input_dim - self.base_dim
            xd_mask = (torch.rand(B, T, xd_count, device=obs_seq.device) > XD_DROPOUT_RATE).float()
            obs_seq = obs_seq.clone()
            obs_seq[:, :, self.base_dim:] *= xd_mask
            obs_seq[:, :, self.base_dim:] += (
                torch.randn(B, T, xd_count, device=obs_seq.device) * XD_NOISE_STD
            )

        # Apply masking
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

        # Forward pass
        outputs = self.forward_train(obs_seq, asset_id, masked_obs, temporal_ctx_drop=temporal_ctx_drop)

        # Reconstruction loss (base features only, NOT XD)
        l_rec = F.mse_loss(outputs["recon"], obs_seq[:, :, :self.base_dim])

        # KL Divergence (categorical RSSM)
        prior = outputs["prior_logits"].view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].view(-1, self.latent_dim, self.classes)

        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior),
        ).mean()
        l_kl = torch.max(l_kl, torch.tensor(WM_FREE_NATS, device=obs_seq.device))

        # Multi-Horizon Return Losses
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

        # Regime classification loss
        if regime_labels is None:
            # Fallback: noisy return-based labels (backward compat)
            ret_1 = target_returns.get(1, torch.zeros(B, T, device=obs_seq.device))
            with torch.no_grad():
                batch_std = ret_1.std() + 1e-6
                if self.training:
                    self._regime_ret_std_ema = 0.99 * self._regime_ret_std_ema + 0.01 * batch_std
                ret_std = self._regime_ret_std_ema
                regime_labels = torch.ones_like(ret_1, dtype=torch.long)  # neutral
                regime_labels[ret_1 > ret_std * 0.5] = 2   # bullish
                regime_labels[ret_1 < -ret_std * 0.5] = 0  # bearish

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

        # Uncertainty-weighted total loss (Kendall et al. 2018)
        # With asymmetric corridors to prevent reconstruction from hogging gradients
        s = self.log_vars.clamp(-6.0, 6.0)

        # Rec: clamp from below -- easy task shouldn't steal gradient from returns
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        total = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        # KL: unconstrained (with annealing)
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

        # ── Direct Return Regression Loss (SOTA) ────────────────────────────
        # Bypasses TwoHot discretization bottleneck with smooth Huber gradients
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                decoded = self.bucketer.decode(outputs["return_logits"][h])
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )
        total = total + DIRECT_RETURN_WEIGHT * l_direct_return

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, outputs

    # =========================================================================
    # INFERENCE / ENCODING
    # =========================================================================

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """
        Encode a sequence and return hidden states + posterior latents.

        Args:
            obs_seq: [B, T, INPUT_DIM]
            asset_id: [B]

        Returns:
            h_seq: [B, T, gru_hidden]
            z_post: [B, T, flat_dim]
            return_preds: dict {horizon: [B, T]}
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds

    # =========================================================================
    # DREAM STEP (for agent dreaming / imagination)
    # =========================================================================

    @torch.no_grad()
    def dream_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        gru_hidden: torch.Tensor = None,
    ):
        """
        One-step imagination using prior (no observation).

        Given previous deterministic state and stochastic latent,
        predict next state using only the prior (no posterior correction).

        Args:
            h_prev: [B, gru_hidden] — previous GRU output
            z_prev: [B, flat_dim] — previous stochastic latent
            gru_hidden: [num_layers, B, gru_hidden] — GRU hidden state

        Returns:
            h_next: [B, gru_hidden]
            z_next: [B, flat_dim]
            gru_hidden: [num_layers, B, gru_hidden]
            return_preds: dict {horizon: [B]}
        """
        # Combine previous state and project to GRU input dimension
        combined = torch.cat([h_prev, z_prev], dim=-1)  # [B, gru_hidden + flat_dim]
        gru_input = self.dream_proj(combined)  # [B, tcn_channels[-1]=256]

        # GRU step
        gru_input = gru_input.unsqueeze(1)  # [B, 1, 256]
        h_next, gru_hidden = self.gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)  # [B, gru_hidden]

        # Prior-only latent (no observation for posterior)
        prior_logits = self.prior_head(h_next)
        z_next = self._get_stoch_state(prior_logits)

        # Predict returns from dreamed state
        feat = torch.cat([h_next, z_next], dim=-1)
        ret_trunk_out = self.return_trunk(feat)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(
                self.return_heads[str(h)](ret_trunk_out)
            )

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    n_feat = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    feat_list, input_dim, base_dim = get_feature_config(n_feat)
    print(f"Device: {DEVICE} | features={n_feat} | input_dim={input_dim} | base_dim={base_dim}")

    model = WaveNetGRUWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
    print(f"V3.3 WaveNet-GRU World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, input_dim).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"Loss: {loss.item():.4f}")
    print(f"Breakdown: {loss_dict}")

    # Test dream step
    h_seq, z_post, ret_preds = model.encode_sequence(obs, asset)
    h_last = h_seq[:, -1, :]
    z_last = z_post[:, -1, :]
    h_dream, z_dream, _, dream_rets = model.dream_step(h_last, z_last)
    print(f"Dream step output shapes: h={h_dream.shape}, z={z_dream.shape}")
    print(f"Dream return preds: { {h: v.shape for h, v in dream_rets.items()} }")

    print("V3.3 WaveNet-GRU world model sanity check passed.")
