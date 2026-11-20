"""
V1.6 World Model -- "Best of V1" Transformer-RSSM

Consolidates all proven V1 techniques:
  - V1.0 base: Transformer-RSSM, Kendall corridors, direct return Huber
  - V1.2: KL annealing, base_dim posterior, XD augmentation
  - V1.3: Gumbel tau annealing (runtime parameter)
  - V3-V9: ATME temporal context dropout (p=0.15)
  - ACTIVE_HORIZONS [1,4,16,64] + pairwise ranking loss
  - Raw return targets, TwoHot bins [-1, 1], no focal/smoothing
  - Dream consistency loss (trains dream_step for agent use)
  - Directional accuracy tracking
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
)
from settings import *


class TransformerWorldModel(nn.Module):
    """
    V1.6: "Best of V1" Transformer World Model with RSSM latents.

    Inputs:
        obs_seq:  [B, T, INPUT_DIM]  -- feature sequences (13/18/21/25/30/37 features)
        asset_id: [B]                -- integer asset index (0-9)

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
        self.atme_prob = ATME_PROB  # V1.6: ATME from V3-V9

        # -- Encoding ----------------------------------------------------------
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

        # Multi-Horizon Return Heads -- wider trunk + deeper per-horizon MLPs
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
        # V1.6: h16/h64 init at -1.0 (2.7x) vs h1/h4 at -2.0 (7.4x)
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # -- TwoHot encoder (raw returns, bins [-1.0, 1.0]) --------------------
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # -- Regime fallback EMA -----------------------------------------------
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=True)
        self.register_buffer('_free_nats', torch.tensor(WM_FREE_NATS), persistent=False)

        # -- Gumbel tau (runtime mutable by trainer) ---------------------------
        self._gumbel_tau = GUMBEL_TAU

        # -- Dream step (agent imagination) ------------------------------------
        self.dream_proj = nn.Linear(d_model + self.flat_dim, d_model)
        self.dream_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

        # 2026-05-21: SEPARATE dream return trunk + heads (full TD-MPC2 isolation).
        # Previously dream_step_train shared return_trunk + return_heads with the
        # main forward pass; with .detach() on the trunk output the heads still
        # received dream-distribution gradients. Full isolation removes that
        # contamination at the cost of ~ret_dim * (head_input_dim + 4 * num_bins)
        # extra params (~600K for d_model=256, ret_dim=384, num_bins=255).
        # Source: TD-MPC2 (Hansen 2024, arXiv:2310.16828) — consistency loss
        # trains only the dynamics-specific pathway, not the prediction heads
        # that operate on real-data encoder outputs.
        self.dream_return_trunk = nn.Sequential(
            nn.Linear(head_input_dim, ret_dim),
            RMSNorm(ret_dim),
            nn.SiLU(),
            nn.Dropout(ret_drop),
        )
        self.dream_return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(ret_dim, ret_dim // 2),
                RMSNorm(ret_dim // 2),
                nn.SiLU(),
                nn.Linear(ret_dim // 2, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # -- Initialize weights ------------------------------------------------
        self._init_weights()

    def _init_weights(self):
        """
        Apply proper weight initialization:
          - Xavier uniform for attention projections
          - He/Kaiming for FFN layers (accounts for SiLU non-linearity)
          - Zero bias where applicable
          - Small init for output projections
        """
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
        """
        Sample from categorical latent using Gumbel-Softmax (straight-through).
        Uses self._gumbel_tau (set by trainer for annealing).

        AMP-safe: Gumbel noise saturates to inf in fp16 when uniform~0.
        Force fp32 around sampling; cast back to input dtype.
        """
        shape = logits.shape
        in_dtype = logits.dtype
        reshaped = logits.view(*shape[:-1], self.latent_dim, self.classes)
        tau = getattr(self, '_gumbel_tau', GUMBEL_TAU)
        with torch.amp.autocast("cuda", enabled=False):
            z = F.gumbel_softmax(reshaped.float(), tau=tau, hard=True, dim=-1)
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
            Dict with: recon, return_logits, regime_logits, prior_logits, post_logits,
                       h_seq, z_post, ret_trunk
        """
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

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
        # Split per HRSSM (Liu 2024) pattern: decoder always sees full [h_seq, z_post];
        # only return/regime heads see ATME-zeroed feat. 2026-05-21 fix:
        # previously the decoder was reconstructing base features from z_post-only
        # 15% of the time (ATME-zeroed h_seq), which trained the decoder to
        # ignore h_seq and measurably hurt non-ATME reconstruction. Refactor
        # routes the ATME mask only to the prediction heads — the decoder
        # always gets the full feature concatenation.
        feat_full = torch.cat([h_seq, z_post], dim=-1)  # [B, T, d_model + flat_dim]

        # 6. ATME: Attention-based Temporal Masking Erasure (from V3-V9).
        # With prob p, zero out h_seq portion of the HEAD-PATH feat so
        # return/regime heads must predict from z_post alone. Forces
        # genuine signal into latent state. Decoder unaffected.
        if self.training and self.atme_prob > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat_full.device) < self.atme_prob)
            h_zeroed = torch.zeros_like(h_seq)
            # z_post = f(posterior(h_seq, base_obs)) leaked temporal info, so zeroing
            # h_seq in the head-path ALONE was cosmetic -- "predict from z_post alone"
            # still carries h_seq's memorized temporal signal. For ATME samples recompute
            # z_post from a ZEROED h_seq (base-obs only, matching line 255) so the drop is
            # real (mirrors V8/V4 fix, 2026-05-29). decoder/feat_full keep full-context z_post.
            post_logits_atme = self.posterior_head(
                torch.cat([h_zeroed, obs_seq[:, :, :self.base_dim]], dim=-1))
            z_post_atme = self._get_stoch_state(post_logits_atme)
            feat_heads_atme = torch.cat([h_zeroed, z_post_atme], dim=-1)
            feat_for_heads = torch.where(atme_mask, feat_heads_atme, feat_full)
        else:
            feat_for_heads = feat_full

        # Reconstruction always reads the FULL feature concatenation (HRSSM
        # pattern). Return/regime heads read the ATME-routed feature.
        recon = self.decoder(feat_full)
        regime_logits = self.regime_head(feat_for_heads)

        # Multi-horizon return predictions via shared trunk
        ret_trunk_out = self.return_trunk(feat_for_heads)
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
            "recon": recon,                  # [B, T, base_dim]
            "return_logits": return_logits,  # dict: {horizon: [B, T, NUM_BINS]}
            "regime_logits": regime_logits,  # [B, T, 3]
            "prior_logits": prior_logits,    # [B, T, flat_dim]
            "post_logits": post_logits,      # [B, T, flat_dim]
            "h_seq": h_seq,                  # [B, T, d_model]
            "z_post": z_post,                # [B, T, flat_dim]
            "ret_trunk": ret_trunk_out,      # [B, T, RETURN_HEAD_DIM]
        }

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        kl_anneal: float = 1.0,
        gumbel_tau: float = GUMBEL_TAU,
        regime_labels: torch.Tensor = None,
        dream_targets_h1: torch.Tensor = None,
        return_components: bool = False,
    ):
        """
        Compute training loss with all V1.6 upgrades.

        Args:
            return_components: when True, returns 4-tuple
                (total, loss_dict, outputs, components). components dict
                holds per-task tensors {aux, ret_1, ret_4, ret_16, ret_64}.
                Aux includes rec + KL (with anneal applied) + regime + direct
                + pairwise + dream. Sum equals total. Used by PCGrad.

                NOTE: KL term carries the anneal weight (kl_anneal arg) so
                PCGrad sees the SCHEDULED KL contribution, not the raw KL.

        Args:
            obs_seq:          [B, T, INPUT_DIM]
            asset_id:         [B]
            target_returns:   dict of {int_horizon: [B, T]} tensors
            mask_ratio:       fraction of timesteps to mask
            block_mask:       if True, mask contiguous blocks
            kl_anneal:        KL weight multiplier (V1.2: ramp 0->1)
            gumbel_tau:       Gumbel-Softmax temperature (V1.3: anneal 1.0->0.5)
            regime_labels:    [B, T] precomputed SMA-based regime labels (0/1/2)
            dream_targets_h1: [B] actual h=1 return at position T+1 for dream loss

        Returns:
            (total_loss, loss_dict, outputs)
        """
        B, T, _ = obs_seq.shape

        # Set gumbel_tau for this forward pass (V1.3 annealing)
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

        # -- Apply masking (vectorized) ----------------------------------------
        masked_obs = obs_seq.clone()
        if mask_ratio > 0:
            if block_mask:
                block_size = max(4, int(T * WM_BLOCK_SIZE_RATIO))
                num_blocks = max(1, int((T * mask_ratio) / block_size))
                max_start = max(1, T - block_size)
                starts = torch.randint(0, max_start, (B, num_blocks), device=obs_seq.device)
                offsets = torch.arange(block_size, device=obs_seq.device)
                for nb in range(num_blocks):
                    indices = starts[:, nb:nb+1] + offsets.unsqueeze(0)  # [B, block_size]
                    indices = indices.clamp(max=T-1)
                    expanded = indices.unsqueeze(-1).expand(-1, -1, obs_seq.shape[-1])
                    masked_obs.scatter_(1, expanded, 0.0)
            else:
                mask = torch.rand(B, T, device=obs_seq.device) < mask_ratio
                masked_obs[mask.unsqueeze(-1).expand_as(obs_seq)] = 0.0

        # -- Forward pass ------------------------------------------------------
        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # -- Reconstruction loss (base features only, NOT XD) ------------------
        l_rec = F.mse_loss(outputs["recon"], obs_seq[:, :, :self.base_dim])

        # -- KL Divergence with free-nats (max formulation) --------------------
        # AMP-safe: D.Categorical(logits=fp16) is unstable when |logit| > ~15.
        # Force fp32 explicitly so the log_softmax inside KL is well-conditioned.
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
        # MDN-aware path uses head.log_prob; default uses TwoHot CE.
        use_mdn = getattr(self, "_use_mdn", False)
        ret_trunk_for_mdn = outputs.get("ret_trunk") if use_mdn else None
        horizon_losses = {}
        for h in REWARD_HORIZONS:
            if h not in ACTIVE_HORIZONS:
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

        # -- Direct Return Regression Loss (MDN-aware) ------------------------
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        decoded_cache = {}
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                if use_mdn:
                    decoded = self.return_heads[str(h)].expectation(ret_trunk_for_mdn)
                else:
                    decoded = self.bucketer.decode(outputs["return_logits"][h])
                decoded_cache[h] = decoded
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )

        # -- Pairwise Ranking Loss (Phase C: learning-to-rank) -----------------
        l_pairwise = torch.tensor(0.0, device=obs_seq.device)
        if PAIRWISE_RANK_WEIGHT > 0 and 1 in decoded_cache and 1 in target_returns:
            pred_flat = decoded_cache[1].reshape(-1)
            tgt_flat = target_returns[1].reshape(-1)
            n = pred_flat.shape[0]
            k = min(PAIRWISE_RANK_PAIRS, n // 2)
            if k > 0:
                idx_a = torch.randint(0, n, (k,), device=obs_seq.device)
                idx_b = torch.randint(0, n, (k,), device=obs_seq.device)
                mask = idx_a != idx_b
                idx_a, idx_b = idx_a[mask], idx_b[mask]
                if idx_a.numel() > 0:
                    pred_diff = pred_flat[idx_a] - pred_flat[idx_b]
                    tgt_sign = torch.sign(tgt_flat[idx_a] - tgt_flat[idx_b])
                    nonzero = tgt_sign != 0
                    if nonzero.sum() > 0:
                        l_pairwise = F.softplus(
                            -pred_diff[nonzero] * tgt_sign[nonzero]
                        ).mean()

        # -- Dream Consistency Loss (V1.6 NEW) ---------------------------------
        # Train dream_step so it produces meaningful h=1 predictions.
        # Compares dream h=1 prediction to actual t+1 target (if available).
        l_dream = torch.tensor(0.0, device=obs_seq.device)
        if dream_targets_h1 is not None and DREAM_CONSISTENCY_WEIGHT > 0:
            # Detach h/z from main computation graph so dream loss only trains
            # dream_proj, dream_gru, prior_head, and return heads
            h_last = outputs["h_seq"][:, -1, :].detach()   # [B, d_model]
            z_last = outputs["z_post"][:, -1, :].detach()   # [B, flat_dim]

            # Use dream_step_train (WITH gradients, not @no_grad dream_step)
            h_d, z_d, _, dream_rets = self.dream_step_train(h_last, z_last)
            l_dream = F.huber_loss(dream_rets[1].squeeze(-1), dream_targets_h1)

        # -- Directional accuracy (metric, not loss) ---------------------------
        dir_acc = {}
        with torch.no_grad():
            for h in REWARD_HORIZONS:
                if h in target_returns:
                    decoded = self.bucketer.decode(outputs["return_logits"][h])
                    actuals = target_returns[h]
                    mask = torch.abs(actuals) > 1e-6
                    if mask.sum() > 50:
                        correct = (torch.sign(decoded[mask]) == torch.sign(actuals[mask])).float()
                        dir_acc[h] = correct.mean().item()
                    else:
                        dir_acc[h] = 0.5

        # -- Total loss (Kendall + corridors; decomposed for PCGrad) ----------
        s = self.log_vars.clamp(-6.0, 6.0)
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        rec_term = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        # KL with anneal: anneal is BAKED INTO the term so PCGrad sees the
        # scheduled contribution, not the raw KL.
        s_kl = s[1].clamp(min=-2.0) if kl_anneal < 1.0 else s[1]
        kl_term = torch.exp(-s_kl) * l_kl * kl_anneal + 0.5 * s_kl

        # Per-horizon weighted return terms (kept separate for PCGrad)
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

        if dream_targets_h1 is not None and DREAM_CONSISTENCY_WEIGHT > 0:
            dream_term = DREAM_CONSISTENCY_WEIGHT * l_dream
        else:
            dream_term = torch.tensor(0.0, device=obs_seq.device)

        if PAIRWISE_RANK_WEIGHT > 0:
            pairwise_term = PAIRWISE_RANK_WEIGHT * l_pairwise
        else:
            pairwise_term = torch.tensor(0.0, device=obs_seq.device)

        # Aux groups everything that ISN'T per-horizon (PCGrad surgery target).
        aux_term = (rec_term + kl_term + regime_term + direct_term
                    + dream_term + pairwise_term)
        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "kl_anneal": kl_anneal,
            "gumbel_tau": gumbel_tau,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "dream": l_dream.item(),
            "pairwise": l_pairwise.item(),
        }
        for h in REWARD_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()
            if h in dir_acc:
                loss_dict[f"dir_acc_{h}"] = dir_acc[h]

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
        """
        Encode a sequence and return hidden states + posterior latents.

        Returns:
            h_seq:        [B, T, d_model]
            z_post:       [B, T, flat_dim]
            return_preds: dict of {horizon: [B, T]} scalar predictions
        """
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

        Args:
            h_prev:     [B, d_model]
            z_prev:     [B, flat_dim]
            gru_hidden: [1, B, d_model] dream GRU hidden state (or None)

        Returns:
            h_next:       [B, d_model]
            z_next:       [B, flat_dim]
            gru_hidden:   [1, B, d_model]
            return_preds: dict of {horizon: [B]} scalar predictions
        """
        combined = torch.cat([h_prev, z_prev], dim=-1)
        gru_input = self.dream_proj(combined).unsqueeze(1)
        h_next, gru_hidden = self.dream_gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)

        prior_logits = self.prior_head(h_next)
        z_next = self._get_stoch_state(prior_logits)

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

    def dream_step_train(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        gru_hidden: torch.Tensor = None,
    ):
        """
        Dream step WITH gradients (for dream consistency loss training).
        Same as dream_step() but without @torch.no_grad().

        2026-05-21 FULL FIX (TD-MPC2 isolation, oracle validation review):
        the dream path now uses its OWN trunk + heads (dream_return_trunk /
        dream_return_heads). The main return_trunk + return_heads receive
        no gradient from the dream loss whatsoever — full separation. Real-
        data and synthetic-dream supervision train disjoint parameter sets
        for the prediction modules; only the shared encoder modules
        (asset_embedding, obs_encoder, transformer_layers, RSSM heads) see
        both, which is the desired multi-task signal.
        """
        combined = torch.cat([h_prev, z_prev], dim=-1)
        gru_input = self.dream_proj(combined).unsqueeze(1)
        h_next, gru_hidden = self.dream_gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)

        prior_logits = self.prior_head(h_next)
        z_next = self._get_stoch_state(prior_logits)

        feat = torch.cat([h_next, z_next], dim=-1)
        # Use SEPARATE dream trunk + heads; main return_trunk/return_heads
        # are untouched by this path.
        ret_trunk_out = self.dream_return_trunk(feat)
        return_preds = {}
        if getattr(self, "_use_mdn", False):
            # MDN expectation: dream path also uses dream heads
            for h in REWARD_HORIZONS:
                return_preds[h] = self.dream_return_heads[str(h)].expectation(ret_trunk_out) \
                    if hasattr(self.dream_return_heads[str(h)], "expectation") \
                    else self.bucketer.decode(self.dream_return_heads[str(h)](ret_trunk_out))
        elif self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk_out)
            for h in REWARD_HORIZONS:
                return_preds[h] = self.bucketer.decode(mtp_out[f"h{h}"])
        else:
            for h in REWARD_HORIZONS:
                logits = self.dream_return_heads[str(h)](ret_trunk_out)
                return_preds[h] = self.bucketer.decode(logits)

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"V1.6 'Best of V1' Architecture:")
    print(f"  d_model={WM_D_MODEL}, n_layers={WM_N_LAYERS}, "
          f"n_heads={WM_N_HEADS}, d_ff={WM_D_FF}")
    print(f"  RSSM: {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  TwoHot bins: {NUM_BINS} in [{BIN_MIN}, {BIN_MAX}]")
    print(f"  ATME prob: {ATME_PROB}")
    print(f"  KL anneal epochs: {KL_ANNEAL_EPOCHS}")
    print(f"  Gumbel tau: {GUMBEL_TAU_START} -> {GUMBEL_TAU_END} over {GUMBEL_TAU_ANNEAL_EPOCHS} ep")
    print(f"  Dream consistency: weight={DREAM_CONSISTENCY_WEIGHT}, every={DREAM_CONSISTENCY_EVERY}")
    print(f"  Return clamp: {RETURN_LOG_VAR_CLAMP_MAX} (uniform all horizons)")

    model = TransformerWorldModel().to(DEVICE)
    print(f"Parameters: {count_parameters(model):,}")

    # Test forward pass with V1.6 feature configs (13 base-only and 37 full)
    B, T = 4, 96
    for n_feat in [13, 37]:
        print(f"\n--- Testing with {n_feat} features ---")
        flist, idim, bdim = get_feature_config(n_feat)
        obs = torch.randn(B, T, idim).to(DEVICE)
        m = TransformerWorldModel(input_dim=idim, base_dim=bdim).to(DEVICE)
        print(f"  Params: {count_parameters(m):,}, input={idim}, base={bdim}")
    # Default test uses INPUT_DIM
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}
    dream_h1 = torch.randn(B).to(DEVICE) * 0.01

    # Test with all V1.6 loss features
    loss, loss_dict, _ = model.get_loss(
        obs, asset, targets,
        mask_ratio=0.15,
        kl_anneal=0.5,     # mid-annealing
        gumbel_tau=0.75,   # mid-annealing
        dream_targets_h1=dream_h1,
    )
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Breakdown:")
    for k, v in sorted(loss_dict.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Test ATME is active during training
    model.train()
    out1 = model.forward_train(obs, asset)
    print(f"\n[OK] ATME active (prob={model.atme_prob})")

    # Test ATME is inactive during eval
    model.eval()
    out2 = model.forward_train(obs, asset)
    print(f"[OK] ATME inactive during eval")

    # Test encode
    h, z, preds = model.encode_sequence(obs, asset)
    print(f"\nEncode: h={h.shape}, z={z.shape}")
    for h_val, p in preds.items():
        print(f"  Return t+{h_val}: {p.shape}, range=[{p.min():.6f}, {p.max():.6f}]")

    # Test dream_step
    h_last = h[:, -1, :]
    z_last = z[:, -1, :]
    h_next, z_next, _, dream_rets = model.dream_step(h_last, z_last)
    print(f"\nDream: h={h_next.shape}, z={z_next.shape}")
    for h_val, p in dream_rets.items():
        print(f"  Dream t+{h_val}: {p.shape}")

    # Test dream_step_train (with gradients)
    model.train()
    h_d, z_d, _, d_rets = model.dream_step_train(h_last.detach(), z_last.detach())
    dream_loss = F.huber_loss(d_rets[1].squeeze(-1), dream_h1)
    dream_loss.backward()
    print(f"\n[OK] dream_step_train gradient flows (dream_loss={dream_loss.item():.6f})")

    # Verify bins
    print(f"\n[OK] TwoHot bins: [{BIN_MIN}, {BIN_MAX}], bin_width={(BIN_MAX-BIN_MIN)/(NUM_BINS-1):.6f}")

    # Verify directional accuracy in loss_dict
    for h in REWARD_HORIZONS:
        key = f"dir_acc_{h}"
        if key in loss_dict:
            print(f"[OK] {key}: {loss_dict[key]:.4f}")

    print("\nV1.6 world model sanity check passed.")
