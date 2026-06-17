"""
V1.4 World Model -- Transformer-RSSM with Multi-Horizon Prediction + FeatureAttentionBlock

Based on V1/TMP_2 proven architecture (ShIC=0.0305, GATE PASS).
Extended to 22 features (17 base + 5 cross-asset), ACTIVE_HORIZONS, pairwise ranking.
V1.4 adds FeatureAttentionBlock (iTransformer-style cross-feature attention).

Architecture:
  - FeatureAttentionBlock: iTransformer-style cross-feature attention (Liu et al., 2023)
  - Obs Encoder: Linear projection + asset embedding -> d_model=256
  - RoPE: Rotary Position Embedding (Su et al., 2021)
  - RMSNorm: Root Mean Square normalization (Zhang & Sennrich, 2019)
  - FlashAttention: F.scaled_dot_product_attention (PyTorch 2.0+)
  - Transformer Core: 3-layer causal self-attention (d_ff=768, 8 heads, SwiGLU)
  - RSSM Latents: free-nats constrained categorical distributions (24x24=576 flat)
  - Return Heads: Multi-horizon TwoHot (255 bins) + direct Huber regression
  - Regime Head: Focal loss classification (bear/neutral/bull)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from components import (
    CausalTransformerBlock,
    RotaryEmbedding,
    RMSNorm,
    TwoHotSymlog,
    SwiGLU,
    MLPHead,
    FeatureAttentionBlock,
)
from settings import *


class TransformerWorldModel(nn.Module):
    """
    V1.4: Transformer-based World Model with RSSM latents + FeatureAttentionBlock.

    Inputs:
        obs_seq:  [B, T, INPUT_DIM]  -- feature sequences (18 features: 13 base + 5 XD)
        asset_id: [B]                -- integer asset index (0-9, 10 assets)

    Training targets:
        target_returns: dict of {horizon: [B, T] tensor} for horizons [1, 4, 16, 64]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        base_dim: int = BASE_DIM,
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
    ):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes  # 24*24 = 576
        self.n_layers = n_layers

        # -- Encoding ----------------------------------------------------------
        # Feature attention: cross-feature interactions before temporal encoding
        self.feature_attn = FeatureAttentionBlock(
            n_features=input_dim,
            d_feat=FEAT_ATTN_D_FEAT,
            n_heads=FEAT_ATTN_N_HEADS,
            dropout=FEAT_ATTN_DROPOUT,
        )

        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.rotary_emb = RotaryEmbedding(d_model // n_heads, max_len=1024)

        # -- Transformer Core (stacked causal blocks) --------------------------
        self.transformer_layers = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # -- RSSM Latent Heads -------------------------------------------------
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        # Posterior uses only base features (not XD) to prevent temporal fingerprint shortcut
        self.posterior_head = MLPHead(d_model + base_dim, 256, self.flat_dim, dropout)

        # -- Output Heads ------------------------------------------------------
        head_input_dim = d_model + self.flat_dim  # 256 + 576 = 832

        # Reconstruction head (base features only -- do NOT reconstruct XD features)
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, base_dim),
        )

        # Return prediction heads (TwoHot 255 bins)
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

        # Frontier-ML upgrade hooks (set via apply_v1_upgrades). Default OFF.
        self._use_mtp = False
        self.mtp_head = None

        # Regime classification: {bearish=0, neutral=1, bullish=2}
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # -- Loss Balancing (learned log-variance) -----------------------------
        # 7 entries: [rec, kl, ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # -- TwoHot encoder ---------------------------------------------------
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # EMA of regime ret_std for stable regime labels across batches
        # persistent=True so it survives checkpoint save/load (avoids reset to 1.0 on resume)
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=True)

        # Pre-allocated free-nats threshold (avoids creating new CUDA tensor every forward pass)
        self.register_buffer('_free_nats', torch.tensor(WM_FREE_NATS), persistent=False)

        # -- Dream step (agent imagination) ------------------------------------
        # Transformer is stateless -- dream_gru provides recurrent state evolution
        # for multi-step imagination in agent's DREAM mode.
        self.dream_proj = nn.Linear(d_model + self.flat_dim, d_model)
        self.dream_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

        # -- Initialize weights ------------------------------------------------
        self._init_weights()

    def _init_weights(self):
        """Apply proper weight initialization."""
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

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latent using Gumbel-Softmax (straight-through).

        AMP-safe: Gumbel noise saturates to inf in fp16 when uniform~0.
        Force fp32 around sampling; cast back.
        """
        shape = logits.shape
        in_dtype = logits.dtype
        reshaped = logits.view(*shape[:-1], self.latent_dim, self.classes)
        with torch.amp.autocast("cuda", enabled=False):
            z = F.gumbel_softmax(reshaped.float(), tau=GUMBEL_TAU, hard=True, dim=-1)
        return z.to(in_dtype).view(*shape)

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        masked_obs_seq: torch.Tensor = None,
    ):
        """
        Full forward pass for training.

        Args:
            obs_seq:        [B, T, INPUT_DIM] -- clean observations
            asset_id:       [B] -- integer asset indices
            masked_obs_seq: [B, T, INPUT_DIM] -- masked observations (for encoder input)

        Returns:
            Dict with all outputs needed for loss computation
        """
        B, T, _ = obs_seq.shape

        # Apply cross-feature attention to encoder input (learns feature interactions)
        input_obs = self.feature_attn(masked_obs_seq if masked_obs_seq is not None else obs_seq)

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)
        obs_emb = self.obs_encoder(enc_input)

        # 2. Causal shift (predict t from t-1)
        obs_emb_shifted = torch.cat([
            torch.zeros(B, 1, self.d_model, device=obs_seq.device),
            obs_emb[:, :-1, :]
        ], dim=1)

        # 3. Transformer core (RoPE applied internally per layer)
        h_seq = obs_emb_shifted
        for layer in self.transformer_layers:
            h_seq = layer(h_seq, rotary_emb=self.rotary_emb)

        # 4. RSSM: Prior and Posterior
        prior_logits = self.prior_head(h_seq)
        # Posterior sees only base features -- XD temporal fingerprint shortcut removed
        post_input = torch.cat([h_seq, obs_seq[:, :, :self.base_dim]], dim=-1)
        post_logits = self.posterior_head(post_input)
        z_post = self._get_stoch_state(post_logits)

        # 5. Combined features -> decode
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, d_model + flat_dim]

        recon = self.decoder(feat)
        regime_logits = self.regime_head(feat)

        # Multi-horizon return predictions via shared trunk
        ret_trunk_out = self.return_trunk(feat)
        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk_out)
            for h in REWARD_HORIZONS:
                return_logits[h] = mtp_out[f"h{h}"]
        elif getattr(self, "_use_mdn", False):
            for h in REWARD_HORIZONS:
                return_logits[h] = ret_trunk_out
        else:
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
            "obs_enriched": input_obs,
        }

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        regime_labels: torch.Tensor = None,
        return_components: bool = False,
    ):
        """
        Compute training loss.

        Args:
            return_components: when True, returns 4-tuple (total, loss_dict,
                outputs, components). components dict has per-task tensors
                {aux, ret_1, ret_4, ret_16, ret_64} with Kendall log-var
                weighting baked in. Sum equals total. Used by PCGrad.

        Returns:
            (total_loss, loss_dict, outputs)
        """
        B, T, _ = obs_seq.shape

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

        # -- Apply masking -----------------------------------------------------
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

        # -- Forward pass ------------------------------------------------------
        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # -- Reconstruction loss (base features only, NOT XD) --------------------
        l_rec = F.mse_loss(outputs["recon"], obs_seq[:, :, :self.base_dim])

        # -- KL Divergence with free-nats (max formulation) --------------------
        # CRITICAL: Use max(kl, free_nats), NOT clamp(kl - free_nats, min=0).
        # The clamp formulation allows KL=0 which causes Kendall log_var drift
        # toward -infinity (weight explosion). The max formulation ensures KL
        # always contributes to the Kendall weighting, keeping weights stable.
        #
        # AMP-safe: D.Categorical(logits=fp16) is unstable for |logit| > ~15.
        # Force fp32 explicitly.
        with torch.amp.autocast("cuda", enabled=False):
            prior = outputs["prior_logits"].float().view(-1, self.latent_dim, self.classes)
            post = outputs["post_logits"].float().view(-1, self.latent_dim, self.classes)
            l_kl = D.kl_divergence(
                D.Categorical(logits=post),
                D.Categorical(logits=prior)
            ).mean()
        kl_raw = l_kl.item()

        l_kl = torch.max(l_kl, self._free_nats)

        # -- Multi-Horizon Return Losses (ACTIVE_HORIZONS only) -----------------
        # MDN-aware: head.log_prob(trunk, target) instead of TwoHot CE.
        use_mdn = getattr(self, "_use_mdn", False)
        ret_trunk_for_mdn = outputs.get("ret_trunk") if use_mdn else None
        horizon_losses = {}
        for h in REWARD_HORIZONS:
            if h not in ACTIVE_HORIZONS:  # was missing: inactive horizons still got loss
                horizon_losses[h] = torch.tensor(0.0, device=obs_seq.device)
                continue
            if h in target_returns:
                if use_mdn:
                    head = self.return_heads[str(h)]
                    horizon_losses[h] = -head.log_prob(ret_trunk_for_mdn, target_returns[h]).mean()
                else:
                    logits = outputs["return_logits"][h].reshape(-1, NUM_BINS)
                    targets = target_returns[h].reshape(-1)
                    if RETURN_LOSS_TYPE == "crps":
                        horizon_losses[h] = self.bucketer.compute_crps_loss(logits, targets)
                    else:
                        horizon_losses[h] = self.bucketer.compute_loss(logits, targets)
            else:
                horizon_losses[h] = torch.tensor(0.0, device=obs_seq.device)

        # -- Regime classification loss (Focal Loss) ---------------------------
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

        # -- Direct Return Regression Loss (MDN-aware) -----------------------
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        decoded_h1 = None
        for h in ACTIVE_HORIZONS:  # was REWARD_HORIZONS: inactive horizons got Huber grad (matches V3/V4/V8)
            if h in target_returns:
                if use_mdn:
                    decoded = self.return_heads[str(h)].expectation(ret_trunk_for_mdn)
                else:
                    decoded = self.bucketer.decode(outputs["return_logits"][h])
                if h == 1:
                    decoded_h1 = decoded
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )

        # -- Pairwise Ranking Loss (h=1 only) ---------------------------------
        l_pairwise = torch.tensor(0.0, device=obs_seq.device)
        if PAIRWISE_RANK_WEIGHT > 0 and 1 in target_returns:
            pred_h1 = decoded_h1 if decoded_h1 is not None else (
                self.return_heads[str(1)].expectation(ret_trunk_for_mdn) if use_mdn
                else self.bucketer.decode(outputs["return_logits"][1])
            )
            real_h1 = target_returns[1]
            flat_pred = pred_h1.reshape(-1)
            flat_real = real_h1.reshape(-1)
            n = flat_pred.shape[0]
            if n > 1:
                n_pairs = min(PAIRWISE_RANK_PAIRS, n * (n - 1) // 2)
                idx_i = torch.randint(0, n, (n_pairs,), device=obs_seq.device)
                idx_j = torch.randint(0, n, (n_pairs,), device=obs_seq.device)
                pred_diff = flat_pred[idx_i] - flat_pred[idx_j]
                real_sign = torch.sign(flat_real[idx_i] - flat_real[idx_j])
                l_pairwise = F.softplus(-pred_diff * real_sign).mean()

        # -- Total loss (decomposed for PCGrad) --------------------------------
        s = self.log_vars.clamp(-6.0, 6.0)
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        rec_term = torch.exp(-s_rec) * l_rec + 0.5 * s_rec
        kl_term = torch.exp(-s[1]) * l_kl + 0.5 * s[1]

        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for i, h in enumerate(REWARD_HORIZONS):
            idx = 2 + i
            if h not in ACTIVE_HORIZONS:
                continue
            s_ret = s[idx].clamp(max=RETURN_LOG_VAR_CLAMP_MAX)
            ret_terms[h] = torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret

        regime_idx = 2 + len(REWARD_HORIZONS)
        s_regime = s[regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        regime_term = torch.exp(-s_regime) * l_regime + 0.5 * s_regime
        direct_term = DIRECT_RETURN_WEIGHT * l_direct_return
        pairwise_term = (PAIRWISE_RANK_WEIGHT * l_pairwise) if PAIRWISE_RANK_WEIGHT > 0 \
            else torch.tensor(0.0, device=obs_seq.device)

        aux_term = rec_term + kl_term + regime_term + direct_term + pairwise_term
        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "pairwise": l_pairwise.item(),
        }
        for h in REWARD_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        if return_components:
            components = {
                "aux": aux_term,
                **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS},
            }
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs

    # -- Inference Methods -----------------------------------------------------

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """Encode a sequence and return hidden states + posterior latents."""
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        if getattr(self, "_use_mdn", False):
            ret_trunk = outputs.get("ret_trunk")
            for h in REWARD_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk)
        else:
            for h in REWARD_HORIZONS:
                return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds

    @torch.no_grad()
    def dream_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        gru_hidden: torch.Tensor = None,
    ):
        """
        One-step imagination using dream GRU (no observation).

        Transformer is stateless, so dream_gru provides recurrent state
        evolution for multi-step imagination in agent's DREAM mode.

        Args:
            h_prev:     [B, d_model] last hidden state
            z_prev:     [B, flat_dim] last latent state
            gru_hidden: [1, B, d_model] dream GRU hidden state (or None)

        Returns:
            h_next:       [B, d_model]
            z_next:       [B, flat_dim] sampled from prior
            gru_hidden:   [1, B, d_model]
            return_preds: dict of {horizon: [B]} scalar predictions
        """
        # Project combined state and evolve via dream GRU
        combined = torch.cat([h_prev, z_prev], dim=-1)
        gru_input = self.dream_proj(combined).unsqueeze(1)  # [B, 1, d_model]
        h_next, gru_hidden = self.dream_gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)  # [B, d_model]

        # Sample next latent from prior conditioned on new hidden state
        prior_logits = self.prior_head(h_next)
        z_next = self._get_stoch_state(prior_logits)

        # Decode from combined features
        feat = torch.cat([h_next, z_next], dim=-1)
        ret_trunk_out = self.return_trunk(feat)
        return_preds = {}
        if getattr(self, "_use_mdn", False):
            for h in REWARD_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk_out)
        elif self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk_out)
            for h in REWARD_HORIZONS:
                return_preds[h] = self.bucketer.decode(mtp_out[f"h{h}"])
        else:
            for h in REWARD_HORIZONS:
                logits = self.return_heads[str(h)](ret_trunk_out)
                return_preds[h] = self.bucketer.decode(logits)

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Architecture: d_model={WM_D_MODEL}, n_layers={WM_N_LAYERS}, "
          f"n_heads={WM_N_HEADS}, d_ff={WM_D_FF}")
    print(f"RSSM: {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"Features: {INPUT_DIM} ({len(FEATURE_LIST)} in FEATURE_LIST)")

    model = TransformerWorldModel().to(DEVICE)
    print(f"V1.4 World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, 96
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, outputs = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"Loss: {loss.item():.4f}")
    for k, v in sorted(loss_dict.items()):
        print(f"  {k}: {v:.4f}")

    # Test encode
    h, z, preds = model.encode_sequence(obs, asset)
    print(f"Hidden: {h.shape}, Latent: {z.shape}")
    for h_val, p in preds.items():
        print(f"  Return t+{h_val}: {p.shape}")

    # Test dream_step
    h_last = h[:, -1, :]
    z_last = z[:, -1, :]
    h_next, z_next, _, dream_rets = model.dream_step(h_last, z_last)
    print(f"Dream h: {h_next.shape}, z: {z_next.shape}")
    for h_val, p in dream_rets.items():
        print(f"  Dream return t+{h_val}: {p.shape}")

    # Test with both feature counts
    for f_count in [13, 18]:
        m = TransformerWorldModel(input_dim=f_count).to(DEVICE)
        obs_f = torch.randn(B, T, f_count).to(DEVICE)
        loss_f, _, _ = m.get_loss(obs_f, asset, targets)
        print(f"  input_dim={f_count}: loss={loss_f.item():.4f}, params={count_parameters(m):,}")

    print("[OK] V1.4 world model sanity check passed.")
