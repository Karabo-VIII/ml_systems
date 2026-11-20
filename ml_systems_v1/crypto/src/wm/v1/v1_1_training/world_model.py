"""
V1.1 World Model -- Transformer-RSSM with Multi-Horizon Prediction (25-Feature)

Based on V1/TMP_2 proven architecture (ShIC=0.0305, GATE PASS).
25 features (20 base + 5 cross-asset), selectable via --features 13/17/18/20/22/25.

Architecture:
  - Obs Encoder: Linear projection + asset embedding -> d_model=256
  - RoPE: Rotary Position Embedding (Su et al., 2021)
  - RMSNorm: Root Mean Square normalization (Zhang & Sennrich, 2019)
  - FlashAttention: F.scaled_dot_product_attention (PyTorch 2.0+)
  - Transformer Core: 3-layer causal self-attention (d_ff=768, 8 heads, SwiGLU)
  - RSSM Latents: free-nats constrained categorical distributions (24x24=576 flat)
  - Return Heads: Multi-horizon TwoHot (255 bins) + direct Huber regression
  - Regime Head: Focal loss classification (bear/neutral/bull)
  - Ablation Heads (optional): Per-subset return MLPs for feature contribution analysis
  - VSN (optional, V1_VSN=1): Variable Selection Network -- per-timestep feature gating
    before the obs_encoder. Learnable sigmoid gate g_t = sigmoid(W * x_t), x'_t = g_t * x_t.
    Causal by construction (uses only x_t, no future). Regularizes low-SNR input features.
    Gate weights are exposed via get_vsn_weights() for operator inspection.
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
from settings import get_ablation_subsets

# ==============================================================================
# VARIABLE SELECTION NETWORK (VSN) -- generalized to src/wm/_shared (2026-06-10)
# ==============================================================================
# The per-timestep causal feature gate used to live inline here. It is now a
# shared, model-agnostic lever so every WM version reuses the SAME implementation.
# Import mirrors the other _shared imports (try package path, fall back to the
# _shared dir on sys.path). Behaviour is byte-for-byte identical to the old inline
# class -- the V1_VSN flag path is unchanged.
try:
    from wm._shared.variable_selection import VariableSelectionNetwork
except ImportError:
    import sys as _sys_vsn
    from pathlib import Path as _Path_vsn
    _shared_dir_vsn = str(_Path_vsn(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_dir_vsn not in _sys_vsn.path:
        _sys_vsn.path.insert(0, _shared_dir_vsn)
    from variable_selection import VariableSelectionNetwork


class TransformerWorldModel(nn.Module):
    """
    V1.1: Transformer-based World Model with RSSM latents.

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
        ablation_subsets: dict = None,
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

        # Reconstruction head (base features only — do NOT reconstruct XD features)
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

        # Frontier-ML upgrade hooks (set via apply_v1_upgrades helper).
        # Default OFF: behavior identical to pre-upgrade V1.1.
        self._use_mtp = False
        self.mtp_head = None  # populated when use_mtp=True via apply_v1_upgrades

        # Headline-tier hooks (set via apply_headline_upgrades). Default OFF: base V1.1 unchanged.
        self._use_multires = False;    self.multires_encoder = None      # CC-H1
        self._use_linattn = False;     self.linattn_block = None         # CC-H2
        self._use_quantile = False;    self.quantile_heads = None        # CC-H5 (aux)
        self._use_regime_cond = False; self.regime_cond_heads = None     # CC-H6 (aux)
        self._use_dream = False                                          # CC-H7 (trainer aux loss)

        # -- VSN (Variable Selection Network, V1_VSN flag) ---------------------
        # Flag-gated: constructed ONLY when env var V1_VSN="1" at model init time.
        # When OFF (default): module not present -> forward path byte-for-byte unchanged.
        # When ON: VariableSelectionNetwork sits BEFORE obs_encoder, gates [B,T,input_dim].
        # Combinable with V1_FORWARD_REGIME (both ON = full world-class candidate).
        import os as _os_vsn
        self._use_vsn = _os_vsn.environ.get("V1_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None  # not constructed; no parameters, no side effects

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

        # -- Multi-Head Feature Ablation (optional) ----------------------------
        # Each ablation head gets its own return trunk + per-horizon MLPs.
        # The shared encoder/RSSM processes feature-masked inputs per head.
        self.ablation_subsets = ablation_subsets or {}
        if self.ablation_subsets:
            self.ablation_trunks = nn.ModuleDict()
            self.ablation_return_heads = nn.ModuleDict()
            for name in self.ablation_subsets:
                self.ablation_trunks[name] = nn.Sequential(
                    nn.Linear(head_input_dim, ret_dim),
                    RMSNorm(ret_dim),
                    nn.SiLU(),
                    nn.Dropout(ret_drop),
                )
                self.ablation_return_heads[name] = nn.ModuleDict({
                    str(h): nn.Sequential(
                        nn.Linear(ret_dim, ret_dim // 2),
                        RMSNorm(ret_dim // 2),
                        nn.SiLU(),
                        nn.Linear(ret_dim // 2, num_bins),
                    )
                    for h in REWARD_HORIZONS
                })
            # Register feature masks as non-persistent buffers
            for name, indices in self.ablation_subsets.items():
                mask = torch.zeros(input_dim)
                mask[indices] = 1.0
                self.register_buffer(f'_abl_mask_{name}', mask, persistent=False)

        # -- Initialize weights ------------------------------------------------
        self._init_weights()

    def _init_weights(self):
        """Apply proper weight initialization."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # FIX 3 (2026-06-10): skip gate_proj here; its neutral-start init
                # (std=0.01, bias=0 -> gates ~0.5) is set in VariableSelectionNetwork.__init__
                # and must not be overwritten by xavier_uniform (std~0.156 would push
                # initial gates away from neutral, defeating the design intent).
                if "gate_proj" in name:
                    continue
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

        AMP-safe: gumbel_softmax samples Gumbel noise (-log(-log(uniform)))
        which saturates to inf in fp16 when uniform is near 0. Force fp32
        around sampling; cast back. Plausible root cause of the documented
        V1.1 torch.compile NaN collapse with f13 (CLAUDE.md cohort note).
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
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # VSN: per-timestep feature gate applied BEFORE obs_encoder (causal -- uses only x_t).
        # When V1_VSN=0 (default): self.vsn is None, branch skipped, path identical to base.
        # When V1_VSN=1: input_obs is replaced with gate * input_obs (same shape).
        # The gate is differentiable -> gradients flow back through the gate weights.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)
        obs_emb = self.obs_encoder(enc_input)
        if self._use_multires and self.multires_encoder is not None:
            # CC-H1: 1/4/16-bar causal context as a RESIDUAL (multires.fuse is zero-init ->
            # starts == base obs_emb, keeps downstream well-conditioned, then learns). A full
            # replacement fed ~0 into the net and exploded grads through the zero-var LayerNorm.
            obs_emb = obs_emb + self.multires_encoder(input_obs, asset_id)

        # 2. Causal shift (predict t from t-1)
        obs_emb_shifted = torch.cat([
            torch.zeros(B, 1, self.d_model, device=obs_seq.device),
            obs_emb[:, :-1, :]
        ], dim=1)

        # 3. Transformer core (RoPE applied internally per layer)
        h_seq = obs_emb_shifted
        for layer in self.transformer_layers:
            h_seq = layer(h_seq, rotary_emb=self.rotary_emb)
        if self._use_linattn and self.linattn_block is not None:
            h_seq = self.linattn_block(h_seq)                      # CC-H2: linear-attn long-context refine

        # 4. RSSM: Prior and Posterior
        prior_logits = self.prior_head(h_seq)
        # Posterior sees only base features — XD temporal fingerprint shortcut removed
        base_obs = obs_seq if self.base_dim == self.input_dim else obs_seq[:, :, :self.base_dim]
        post_input = torch.cat([h_seq, base_obs], dim=-1)
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
            # MTP causal-chain head produces all horizons in one pass
            mtp_out = self.mtp_head(ret_trunk_out)  # (B, T, NUM_BINS) per horizon
            for h in REWARD_HORIZONS:
                return_logits[h] = mtp_out[f"h{h}"]
        elif getattr(self, "_use_mdn", False):
            # MDN: heads return parameter dicts, not bin logits. We don't call
            # them here; get_loss accesses ret_trunk_out via outputs["ret_trunk"]
            # and computes log_prob directly. Store trunk in return_logits slot
            # to keep the dict shape stable for backward-compat callers that
            # expect outputs["return_logits"][h] to exist.
            for h in REWARD_HORIZONS:
                return_logits[h] = ret_trunk_out  # placeholder; MDN-aware paths use ret_trunk
        else:
            for h in REWARD_HORIZONS:
                return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        out = {
            "recon": recon,
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": ret_trunk_out,
        }
        # Headline aux heads (CC-H5 quantile, CC-H6 regime-conditional) -- computed on feat,
        # supervised by auxiliary losses in get_loss when the flags are set. Default OFF.
        if self._use_quantile and self.quantile_heads is not None:
            out["quantile"] = self.quantile_heads(feat)                  # {h: [B,T,n_quantiles]}
        if self._use_regime_cond and self.regime_cond_heads is not None:
            regime_probs = torch.softmax(regime_logits, dim=-1)          # [B,T,3]
            out["regime_cond"] = self.regime_cond_heads.soft_blend(
                self.regime_cond_heads(feat), regime_probs)              # {h: [B,T,num_bins]}
        # Forward-regime head (Layer-C wiring, 2026-06-10). OFF by default:
        # _use_forward_regime is False until attach_forward_regime_head() is called.
        # When OFF: no 'forward_regime' key in out -- base path byte-for-byte unchanged.
        if getattr(self, "_use_forward_regime", False) and getattr(self, "forward_regime_head", None) is not None:
            out["forward_regime"] = self.forward_regime_head(feat)       # {bear_logits,trend_logits,move_logits}
        return out

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
            return_components: when True, returns 4-tuple
                (total, loss_dict, outputs, components) where `components` is
                a dict with keys {'aux', 'ret_1', 'ret_4', 'ret_16', 'ret_64'}.
                Each value is a SCALAR TENSOR with the Kendall log-var weighting
                baked in. Sum equals `total`. Used by PCGrad gradient surgery.

        Returns:
            (total_loss, loss_dict, outputs)            when return_components=False
            (total_loss, loss_dict, outputs, components) when return_components=True
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

        # -- Reconstruction loss (base features only, NOT XD) --------------------
        recon_target = obs_seq if self.base_dim == self.input_dim else obs_seq[:, :, :self.base_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)

        # -- KL Divergence with free-nats (max formulation) --------------------
        # CRITICAL: Use max(kl, free_nats), NOT clamp(kl - free_nats, min=0).
        # The clamp formulation allows KL=0 which causes Kendall log_var drift
        # toward -infinity (weight explosion). The max formulation ensures KL
        # always contributes to the Kendall weighting, keeping weights stable.
        #
        # AMP-safe: torch.distributions.Categorical calls F.log_softmax which
        # is unstable under fp16 when |logit| > ~15. Force fp32 explicitly.
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
        # MDN path uses NLL on log_prob directly from trunk; bin-bucketer path
        # (default + MTP) uses TwoHot CE on output logits.
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
                    targets = target_returns[h]
                    log_p = head.log_prob(ret_trunk_for_mdn, targets)
                    horizon_losses[h] = -log_p.mean()
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

        # -- Direct Return Regression Loss ------------------------------------
        # MDN-aware: use head.expectation(trunk) instead of bucketer.decode(logits).
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        decoded_cache = {}
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                if use_mdn:
                    head = self.return_heads[str(h)]
                    decoded = head.expectation(ret_trunk_for_mdn)
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

        # -- Total loss --------------------------------------------------------
        # Kendall uncertainty weighting with corridors
        # log_vars: [rec, kl, ret_1, ret_4, ret_16, ret_64, regime] (7 entries)
        s = self.log_vars.clamp(-6.0, 6.0)

        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        rec_term = torch.exp(-s_rec) * l_rec + 0.5 * s_rec

        # KL (free-nats max formulation, Kendall-weighted)
        kl_term = torch.exp(-s[1]) * l_kl + 0.5 * s[1]

        # Per-horizon weighted return terms (kept separate for PCGrad)
        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for i, h in enumerate(REWARD_HORIZONS):
            idx = 2 + i
            if h not in ACTIVE_HORIZONS:
                continue
            s_ret = s[idx].clamp(max=RETURN_LOG_VAR_CLAMP_MAX)
            ret_terms[h] = torch.exp(-s_ret) * horizon_losses[h] + 0.5 * s_ret

        # Regime
        regime_idx = 2 + len(REWARD_HORIZONS)
        s_regime = s[regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX)
        regime_term = torch.exp(-s_regime) * l_regime + 0.5 * s_regime

        # Direct return regression + pairwise (fixed weights)
        direct_term = DIRECT_RETURN_WEIGHT * l_direct_return
        pairwise_term = (PAIRWISE_RANK_WEIGHT * l_pairwise) if PAIRWISE_RANK_WEIGHT > 0 \
            else torch.tensor(0.0, device=obs_seq.device)

        # Aux = everything that ISN'T per-horizon (shared across horizon heads).
        # PCGrad applies surgery across [aux, ret_1, ret_4, ret_16, ret_64].
        aux_term = rec_term + kl_term + regime_term + direct_term + pairwise_term

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        # -- Headline aux losses (CC-H5 quantile / CC-H6 regime-cond / CC-H7 dream) -------
        # Guarded; default OFF -> total unchanged. Fixed small weights (tunable in settings).
        _haux_log = {}
        if self._use_quantile and "quantile" in outputs:
            from wm._shared.headline_components import quantile_loss as _qloss
            ql = torch.tensor(0.0, device=obs_seq.device)
            for h in ACTIVE_HORIZONS:
                if h in target_returns and h in outputs["quantile"]:
                    p = outputs["quantile"][h]
                    ql = ql + _qloss(p.reshape(-1, p.shape[-1]), target_returns[h].reshape(-1))
            total = total + 0.30 * ql
            _haux_log["q_aux"] = ql.item()
        if self._use_regime_cond and "regime_cond" in outputs:
            rc = torch.tensor(0.0, device=obs_seq.device)
            for h in ACTIVE_HORIZONS:
                if h in target_returns and h in outputs["regime_cond"]:
                    rc = rc + self.bucketer.compute_loss(
                        outputs["regime_cond"][h].reshape(-1, NUM_BINS), target_returns[h].reshape(-1))
            total = total + 0.30 * rc
            _haux_log["rc_aux"] = rc.item()
        if self._use_dream and hasattr(self, "dream_step"):
            dl = self._headline_dream_loss(outputs, target_returns)
            total = total + 0.20 * dl
            _haux_log["dream_aux"] = dl.item()
        # Forward-regime aux loss (Layer-C wiring, 2026-06-10).
        # Guarded: only fires when head is attached (_use_forward_regime=True) and
        # the trainer supplied forward_regime_labels. Default OFF -> total unchanged.
        # Weight 0.10 (small fixed; head is additive supervision, not the primary objective).
        # Trainer builds labels via src/wm/_shared/regime_targets.py:
        #   bear_lab = forward_bear_label(close, K=64, dd_thresh=0.05)  -- float32 (N,)
        #   trend_lab = forward_trend_label(close, K=64)                -- float32 (N,)
        #   move_lab  = move_onset_label(close, a=1, b=64)              -- float32 (N,)
        # then packs them into targets["forward_regime_labels"] = {"bear": [B,T], ...}
        # NaN rows (last K bars per series) are auto-masked by forward_regime_aux_loss.
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = target_returns.get("forward_regime_labels", {})
            if _fr_labels:
                import sys as _frsys
                from pathlib import Path as _frPath
                _shared_dir = str(_frPath(__file__).resolve().parent.parent.parent / "_shared")
                if _shared_dir not in _frsys.path:
                    _frsys.path.insert(0, _shared_dir)
                from forward_regime_head import forward_regime_aux_loss as _fr_aux_loss
                fr_loss = _fr_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * fr_loss
                _haux_log["fr_aux"] = fr_loss.item()

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
        loss_dict.update(_haux_log)

        if return_components:
            components = {
                "aux": aux_term,
                **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS},
            }
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs

    # -- Multi-Head Feature Ablation -------------------------------------------

    def ablation_forward(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
    ):
        """
        Run ablation heads with feature-masked inputs.

        Each head zeros non-subset features at the input level, runs through
        the shared encoder/RSSM, then predicts returns via head-specific MLPs.

        Args:
            obs_seq:        [B, T, input_dim] -- observations (pre-augmentation)
            asset_id:       [B] -- asset indices
            target_returns: dict of {h: [B, T]} targets

        Returns:
            dict of {head_name: {"loss": Tensor, "losses": {h: float},
                     "return_logits": {h: Tensor}}}
        """
        results = {}
        for name in self.ablation_subsets:
            feat_mask = getattr(self, f'_abl_mask_{name}')
            masked_obs = obs_seq * feat_mask

            # Run through shared encoder/RSSM with masked input
            outputs = self.forward_train(masked_obs, asset_id)
            feat = torch.cat([outputs["h_seq"], outputs["z_post"]], dim=-1)

            # Head-specific return predictions
            trunk_out = self.ablation_trunks[name](feat)
            head_loss = torch.tensor(0.0, device=obs_seq.device)
            head_losses = {}
            return_logits = {}

            for h in ACTIVE_HORIZONS:
                if h in target_returns:
                    logits = self.ablation_return_heads[name][str(h)](trunk_out)
                    return_logits[h] = logits
                    loss_h = self.bucketer.compute_loss(
                        logits.reshape(-1, NUM_BINS),
                        target_returns[h].reshape(-1),
                    )
                    head_losses[h] = loss_h.item()
                    head_loss = head_loss + loss_h

            results[name] = {
                "loss": head_loss,
                "losses": head_losses,
                "return_logits": return_logits,
            }

        return results

    def _headline_dream_loss(self, outputs, target_returns):
        """CC-H7 dream-rollout aux (GRAD-enabled; dream_step is no_grad so unusable for loss).
        From the penultimate state, roll one dream step through dream_proj/dream_gru/prior and
        supervise the h=1 return logits against the realized last-step target -> regularizes the
        latent dynamics for imagination/rollout. Safe: returns 0 if seq too short or h=1 absent."""
        h_seq, z_post = outputs["h_seq"], outputs["z_post"]
        if h_seq.shape[1] < 2 or 1 not in target_returns:
            return torch.tensor(0.0, device=h_seq.device)
        combined = torch.cat([h_seq[:, -2, :], z_post[:, -2, :]], dim=-1)
        gru_in = self.dream_proj(combined).unsqueeze(1)
        h_next, _ = self.dream_gru(gru_in)
        h_next = h_next.squeeze(1)
        z_next = self._get_stoch_state(self.prior_head(h_next))
        feat = torch.cat([h_next, z_next], dim=-1)
        ret_logits = self.return_heads["1"](self.return_trunk(feat))
        tgt = target_returns[1][:, -1].reshape(-1)
        return self.bucketer.compute_loss(ret_logits.reshape(-1, NUM_BINS), tgt)

    # -- VSN Inspection --------------------------------------------------------

    @torch.no_grad()
    def get_vsn_weights(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Return VSN gate activations for feature-selection inspection.

        Args:
            obs_seq: [B, T, input_dim] or [T, input_dim] -- raw features

        Returns:
            gate: [B, T, input_dim] float in (0, 1) -- higher = feature selected.
                  Averaged over time and batch dimensions for operator inspection:
                  use gate.mean(dim=(0,1)) to get a [input_dim] per-feature weight.

        Raises:
            RuntimeError: if V1_VSN is not enabled (vsn is None).
        """
        if self.vsn is None:
            raise RuntimeError(
                "get_vsn_weights() called but V1_VSN is not enabled. "
                "Set V1_VSN=1 before constructing the model."
            )
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(0)
        return self.vsn.get_weights(obs_seq)

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
    print(f"Base dim: {BASE_DIM} (posterior/decoder use base features only)")

    model = TransformerWorldModel(input_dim=INPUT_DIM, base_dim=BASE_DIM).to(DEVICE)
    print(f"V1.1 World Model Parameters: {count_parameters(model):,}")
    print(f"  Posterior head input: d_model({WM_D_MODEL}) + base_dim({BASE_DIM}) = {WM_D_MODEL + BASE_DIM}")
    print(f"  Decoder output: {BASE_DIM}")
    print(f"  Recon shape: [B, T, {BASE_DIM}]")

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

    # Test ablation heads
    print("\n--- Testing Ablation Mode ---")
    abl_subsets = {
        "f13": list(range(13)),
        "f17": list(range(17)),
    }
    abl_model = TransformerWorldModel(
        input_dim=INPUT_DIM, base_dim=BASE_DIM, ablation_subsets=abl_subsets,
    ).to(DEVICE)
    abl_params = count_parameters(abl_model)
    base_params = count_parameters(model)
    print(f"Ablation Model Parameters: {abl_params:,} (+{abl_params - base_params:,} for ablation heads)")

    loss, loss_dict, outputs = abl_model.get_loss(obs, asset, targets, mask_ratio=0.15)
    abl_results = abl_model.ablation_forward(obs, asset, targets)
    for name, abl in abl_results.items():
        print(f"  Ablation head '{name}': loss={abl['loss'].item():.4f}, "
              f"horizons={list(abl['losses'].keys())}")

    print("[OK] V1.1 world model sanity check passed.")
