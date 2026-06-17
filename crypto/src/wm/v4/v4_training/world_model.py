"""
V4 World Model — Mamba-3 RSSM Hybrid with Multi-Horizon Prediction

Architecture:
  - Obs Encoder: Linear projection + asset embedding
  - Mamba-3 Core: 2-layer stacked SSM (complex-valued, SSD, trapezoidal)
  - RSSM Latents: Prior/Posterior categorical distributions (24x24 bottleneck)
  - Heads: Reconstruction, Multi-Horizon Returns (TwoHot), Regime classification

Mamba-3 innovations (ICLR 2026):
  - Complex-valued dynamics via data-dependent RoPE on B/C
  - Trapezoidal discretization (second-order accurate state update)
  - QK-Norm on B/C for training stability
  - SSD chunk-based parallel scan (replaces sequential for-loop)

Anti-memorization:
  - RSSM categorical bottleneck (24x24 = 576 states, ~9.2 bits/timestep hard cap)
  - Per-sample ATME (15% probability, obs-only posterior + h_seq.detach())
  - SSM forward in fp32 (autocast disabled for complex-valued state stability)
  - nan_to_num on KL logits (D.Categorical protection)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

try:
    from components import Mamba3Block, TwoHotSymlog, SwiGLU, MLPHead, RMSNorm
    from settings import *
except ImportError:
    from v4_training.components import Mamba3Block, TwoHotSymlog, SwiGLU, MLPHead, RMSNorm
    from v4_training.settings import *

# Round-7 frontier components (shared)
import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from frontier_components import tail_adaptive_huber  # noqa: E402

# ==============================================================================
# VARIABLE SELECTION NETWORK (VSN) -- shared lever, V4_VSN flag (2026-06-10)
# ==============================================================================
# Generalised to src/wm/_shared (same implementation as V1.1 inline class).
# V4_VSN=1 enables the gate; V4_VSN=0 (default) = base path byte-for-byte unchanged.
try:
    from wm._shared.variable_selection import VariableSelectionNetwork
except ImportError:
    import sys as _sys_vsn
    from pathlib import Path as _Path_vsn
    _shared_dir_vsn = str(_Path_vsn(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_dir_vsn not in _sys_vsn.path:
        _sys_vsn.path.insert(0, _shared_dir_vsn)
    from variable_selection import VariableSelectionNetwork


class MambaWorldModel(nn.Module):
    """
    The V4 World Model.

    Inputs:
        obs_seq:  [B, T, INPUT_DIM]  — feature sequences
        asset_id: [B]                — integer asset index (0-4)

    Training targets:
        target_returns: dict of {horizon: [B, T] tensor} for horizons [1, 4, 16, 64]
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = WM_D_MODEL,
        d_state: int = WM_D_STATE,
        n_layers: int = WM_N_LAYERS,
        expand: int = WM_EXPAND,
        headdim: int = WM_HEADDIM,
        chunk_size: int = WM_CHUNK_SIZE,
        latent_dim: int = RSSM_LATENT_DIM,
        classes: int = RSSM_CLASSES,
        num_bins: int = NUM_BINS,
        num_assets: int = NUM_ASSETS,
        asset_emb_dim: int = WM_ASSET_EMB_DIM,
        dropout: float = WM_DROPOUT,
    ):
        super().__init__()

        self.d_model = d_model
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes
        self.n_layers = n_layers

        # ── VSN (Variable Selection Network, V4_VSN flag) ─────────────────────
        # Activated by env var V4_VSN="1" at model-init time.
        # When OFF (default): module not constructed -> forward path byte-for-byte unchanged.
        # When ON: VariableSelectionNetwork sits BEFORE obs_encoder, gates [B,T,input_dim].
        # Combinable with V4_FORWARD_REGIME (both ON = full world-class candidate).
        import os as _os_vsn
        self._use_vsn = _os_vsn.environ.get("V4_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None  # not constructed; no parameters, no side effects

        # ── Encoding ──────────────────────────────────────────────────────────
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Mamba-3 Core (stacked) ────────────────────────────────────────────
        self.mamba_layers = nn.ModuleList([
            Mamba3Block(d_model, d_state=d_state, expand=expand,
                        headdim=headdim, chunk_size=chunk_size, dropout=dropout)
            for _ in range(n_layers)
        ])
        # Post-SSM norm: prevents magnitude explosion through residual connections.
        # SSD output grows ~4x per layer. Without this norm, h_seq reaches max=300+
        # after 2 layers and 34 training steps -> decoder SwiGLU overflows fp16.
        self.post_ssm_norm = RMSNorm(d_model)

        # ── RSSM Latent Heads ─────────────────────────────────────────────────
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        self.posterior_head = MLPHead(d_model + input_dim, 256, self.flat_dim, dropout)

        # ── Output Heads ──────────────────────────────────────────────────────
        head_input_dim = d_model + self.flat_dim

        # Reconstruction
        self.decoder = nn.Sequential(
            # FIX: Explicitly project back down to 256 to match RMSNorm
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, input_dim),
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

        # Frontier-ML upgrade hooks (set via apply_v1_upgrades). Default OFF.
        self._use_mtp = False
        self.mtp_head = None

        # Forward-regime head (V4_FORWARD_REGIME flag, Layer-C wiring). Default OFF.
        # attach_forward_regime_head() sets _use_forward_regime=True and attaches
        # self.forward_regime_head. Until then the base path is byte-for-byte unchanged.
        self._use_forward_regime = False
        self.forward_regime_head = None

        # Regime classification: {bearish=0, neutral=1, bullish=2}
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # ── Loss Balancing ────────────────────────────────────────────────────
        # SOTA: Initialize to FORCE return learning over reconstruction
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # ── TwoHot encoder ────────────────────────────────────────────────────
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # EMA of regime ret_std for stable regime labels across batches
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        # ── Dream step (agent imagination) ────────────────────────────────
        # Mamba is stateless -- dream_gru provides recurrent state evolution
        self.dream_proj = nn.Linear(d_model + self.flat_dim, d_model)
        self.dream_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

        # ── Forecast heads (2026-05-10 generalization fix) ────────────────
        # Predict obs[t+h] from h_seq[t]. Anchors h_seq to feature-faithful
        # future prediction (Mamba's natural objective). Empirical winner per
        # scripts/v4_diag/probe_v4_options.py: D (forecast + atme=0.20) lifted
        # train_IC@h1 from 0.0176 (baseline) to 0.0330 (+88%).
        try:
            from settings import USE_FORECAST_HEAD as _use_fc
        except ImportError:
            _use_fc = False
        if _use_fc:
            self.forecast_heads = nn.ModuleDict({
                str(h): nn.Linear(d_model, input_dim) for h in REWARD_HORIZONS
            })
        else:
            self.forecast_heads = None

        # CC-H5 quantile heads (SOTA-2026, auxiliary). Same pattern as V3.
        try:
            from settings import USE_QUANTILE_HEADS as _use_qh
        except ImportError:
            _use_qh = False
        # Resolve _shared once (also used for CC-H6).
        import sys as _qsys
        from pathlib import Path as _qPath
        _shared = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
        if _shared not in _qsys.path:
            _qsys.path.insert(0, _shared)
        if _use_qh:
            from headline_components import QuantileHeads as _QH
            self.quantile_heads = _QH(
                head_input_dim=head_input_dim,
                horizons=tuple(REWARD_HORIZONS),
            )
        else:
            self.quantile_heads = None

        # CC-H6 regime-conditional heads (SOTA-2026, auxiliary).
        try:
            from settings import USE_REGIME_COND_HEADS as _use_rc
        except ImportError:
            _use_rc = False
        if _use_rc:
            from headline_components import RegimeConditionalHeads as _RC
            self.regime_cond_heads = _RC(
                head_input_dim=head_input_dim,
                horizons=tuple(REWARD_HORIZONS),
                num_bins=num_bins,
                n_regimes=3,
            )
        else:
            self.regime_cond_heads = None

        # Regime-awareness encoder conditioning ("film" mode adds FiLM
        # modulator on h_seq before heads; identity-at-init).
        try:
            from settings import REGIME_AWARENESS_MODE as _ram
        except ImportError:
            _ram = "heads"
        self._regime_awareness_mode = _ram
        if _ram == "film":
            from headline_components import RegimeFiLM as _RF
            self.regime_film = _RF(d_model=d_model, n_regimes=3)
            self.regime_film_gate = nn.Linear(d_model, 3)
            nn.init.zeros_(self.regime_film_gate.weight)
            nn.init.zeros_(self.regime_film_gate.bias)
        else:
            self.regime_film = None
            self.regime_film_gate = None

        # ── Weight initialization ────────────────────────────────────────────
        # Tag the VSN's gate_proj BEFORE apply() so _init_weights can skip it.
        # This preserves the neutral-start init set in VariableSelectionNetwork.__init__:
        # std=0.01, bias=0 -> sigmoid(~0) = 0.5 (all features half-open).
        if self.vsn is not None:
            self.vsn.gate_proj._vsn_gate_proj = True
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights. SSM params (A_log/D/dt_bias/B_bias/C_bias) are nn.Parameters, not nn.Modules, so they are not affected by apply()."""
        if isinstance(module, nn.Linear):
            # Skip gate_proj: VSN neutral-start init (std=0.01, bias=0 -> gates ~0.5)
            # must not be overwritten. self.apply() does not pass the qualified name, so
            # we use a sentinel attr set on the gate_proj Linear after VSN construction
            # (see bottom of __init__: self._tag_vsn_gate() call).
            if getattr(module, "_vsn_gate_proj", False):
                return
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latent using Gumbel-Softmax.

        AMP-safe: Gumbel noise saturates to inf in fp16. Force fp32.
        """
        in_dtype = logits.dtype
        shape = logits.shape
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
        temporal_ctx_drop: float = 0.0,
    ):
        """Full forward pass for training."""
        # FIX: unpack to '_' instead of 'D' to avoid shadowing torch.distributions
        B, T, _ = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # VSN: per-timestep feature gate applied BEFORE obs_encoder (causal -- uses only x_t).
        # When V4_VSN=0 (default): self.vsn is None, branch skipped, path identical to base.
        # When V4_VSN=1: input_obs is replaced with gate * input_obs (same shape).
        # The gate is differentiable -> gradients flow back through the gate weights.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

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

        # 3. Mamba-3 core (fp32 for complex-valued state stability)
        with torch.amp.autocast("cuda", enabled=False):
            h_seq = obs_emb_shifted.float()
            for layer in self.mamba_layers:
                h_seq = layer(h_seq)
            h_seq = self.post_ssm_norm(h_seq)

        # Regime-FiLM modulator (SOTA-2026 opt-in): conditions h_seq on
        # regime BEFORE posterior + heads. h_seq-only gate; identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # 4. RSSM: Prior and Posterior
        prior_logits = self.prior_head(h_seq)

        # Per-sample ATME (V1.6-class): ~15% of samples get obs-only posterior
        # Key insight: z_post = f(posterior(h_seq, obs_seq)), so zeroing h_seq in
        # feat_heads alone is cosmetic — temporal info already leaked through z_post.
        # Fix: ATME samples zero h_seq in posterior input; normal samples detach h_seq.
        if self.training and temporal_ctx_drop > 0:
            atme_mask = (torch.rand(B, 1, 1, device=obs_seq.device) < temporal_ctx_drop).float()
            # ATME samples: obs-only posterior (h_seq zeroed)
            # Normal samples: full posterior (h_seq included)
            # Posterior reads MASKED input_obs not raw obs_seq (block-mask MAE
            # leak fix — 2026-05-21 RED-team audit).
            h_seq_for_post = h_seq * (1.0 - atme_mask)
            post_input = torch.cat([h_seq_for_post, input_obs], dim=-1)
            post_logits = self.posterior_head(post_input)
            z_post = self._get_stoch_state(post_logits)

            # feat_heads: h_seq always detached during training, zero for ATME samples
            h_seq_for_heads = h_seq.detach() * (1.0 - atme_mask)
            feat_heads = torch.cat([h_seq_for_heads, z_post], dim=-1)
        else:
            # Eval mode: full temporal context (no detach). Note posterior here
            # reads raw obs_seq because eval-time has no masking — input_obs ==
            # obs_seq when masked_obs_seq is None (set at line 258).
            post_input = torch.cat([h_seq, input_obs], dim=-1)
            post_logits = self.posterior_head(post_input)
            z_post = self._get_stoch_state(post_logits)
            feat_heads = torch.cat([h_seq, z_post], dim=-1)

        # CC-H3 cross-asset hook (SOTA-2026): no-op until MultiAssetDataset
        # provides N_assets-per-batch synchronized slices. Wired via
        # `wrap_with_cross_asset_head` from src/wm/_shared/headline_components.
        # Until then, single-asset training has N_assets=1 and the head
        # degenerates to self-attention on one token = identity.
        if hasattr(self, "_cross_asset_head"):
            feat_heads = self._cross_asset_head(feat_heads)

        # 5. Decode (reconstruction uses full temporal context, no ATME)
        feat = torch.cat([h_seq, z_post], dim=-1)
        recon = self.decoder(feat)

        regime_logits = self.regime_head(feat_heads)

        # Multi-horizon return predictions
        ret_trunk_out = self.return_trunk(feat_heads)
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

        # Forecast heads (added 2026-05-10) — predict obs[t+h] from h_seq[t]
        forecast_logits = None
        if self.forecast_heads is not None:
            forecast_logits = {
                int(h): self.forecast_heads[str(h)](h_seq) for h in REWARD_HORIZONS
            }

        # CC-H5 quantile heads (SOTA-2026): auxiliary distributional output.
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat_heads)

        # CC-H6 regime-conditional heads (SOTA-2026): all-regime outputs.
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat_heads)

        # Forward-regime head (V4_FORWARD_REGIME flag, Layer-C wiring, 2026-06-10).
        # OFF by default: _use_forward_regime is False until attach_forward_regime_head()
        # is called from the trainer. When OFF: key absent from output dict -- base path
        # byte-for-byte unchanged. When ON: reads feat (= cat(h_seq, z_post)), causal
        # (feat is assembled from past-only h_seq from the SSM forward pass).
        # The HEAD reads the post-encoder feat; its LABELS use future bars only at
        # target-construction time (never as model inputs) -- no look-ahead.
        if getattr(self, "_use_forward_regime", False) and self.forward_regime_head is not None:
            feat_for_fr = torch.cat([h_seq, z_post], dim=-1)   # same as `feat` above
            out_forward_regime = self.forward_regime_head(feat_for_fr)
        else:
            out_forward_regime = None

        out = {
            "recon": recon,
            "return_logits": return_logits,  # dict: {horizon: [B, T, NUM_BINS]}
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": ret_trunk_out,  # Expose for adapter modulation
            "forecast_logits": forecast_logits,  # dict: {horizon: [B, T, INPUT_DIM]} or None
            "quantile_logits": quantile_logits,  # dict: {horizon: [B, T, n_quantiles]} or None
            "regime_cond_logits": regime_cond_logits,  # {regime_idx: {h: [B, T, num_bins]}} or None
        }
        if out_forward_regime is not None:
            out["forward_regime"] = out_forward_regime  # {bear_logits,trend_logits,move_logits}
        return out

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,  # {horizon: [B, T] tensor}
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        kl_anneal: float = 1.0,
        gumbel_tau: float = GUMBEL_TAU,
        temporal_ctx_drop: float = 0.0,
        regime_labels: torch.Tensor = None,
        return_components: bool = False,
    ):
        """
        Compute training loss with block masking and multi-horizon targets.

        Args:
            return_components: when True, returns 4-tuple (total, loss_dict,
                outputs, components). components dict has per-task tensors
                {aux, ret_1, ret_4, ret_16, ret_64} for PCGrad surgery.

        Args:
            obs_seq:        [B, T, INPUT_DIM]
            asset_id:       [B]
            target_returns: dict of {int_horizon: [B, T]} tensors
            mask_ratio:     fraction of timesteps to mask
            block_mask:     if True, mask contiguous blocks
            gumbel_tau:     Gumbel-Softmax temperature (annealed during training)
        """
        # FIX: unpack to '_' instead of 'D' to avoid shadowing torch.distributions
        B, T, _ = obs_seq.shape
        self._gumbel_tau = gumbel_tau

        # ── XD anti-memorization augmentation (training only) ─────────────────
        # Per-timestep dropout + heavy noise on XD features [BASE_DIM:] prevents
        # the model from building sequential temporal fingerprints over 96-bar
        # windows.  Skipped when input_dim == BASE_DIM (base-only feature sets).
        if self.training and BASE_DIM < obs_seq.shape[-1]:
            xd_count = obs_seq.shape[-1] - BASE_DIM
            xd_mask = (torch.rand(B, T, xd_count, device=obs_seq.device) > XD_DROPOUT_RATE).float()
            obs_seq = obs_seq.clone()
            obs_seq[:, :, BASE_DIM:] *= xd_mask
            obs_seq[:, :, BASE_DIM:] += (
                torch.randn(B, T, xd_count, device=obs_seq.device) * XD_NOISE_STD
            )

        # ── Apply masking ─────────────────────────────────────────────────────
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

        # ── Forward pass ──────────────────────────────────────────────────────
        outputs = self.forward_train(obs_seq, asset_id, masked_obs, temporal_ctx_drop=temporal_ctx_drop)

        # ── Reconstruction loss ───────────────────────────────────────────────
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)

        # ── KL Divergence ─────────────────────────────────────────────────────
        prior = outputs["prior_logits"].float().view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].float().view(-1, self.latent_dim, self.classes)
        # nan_to_num protection: D.Categorical crashes on NaN/inf logits
        prior = torch.nan_to_num(prior, nan=0.0, posinf=10.0, neginf=-10.0)
        post = torch.nan_to_num(post, nan=0.0, posinf=10.0, neginf=-10.0)

        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior)
        ).mean()
        kl_raw = l_kl.item()
        l_kl = torch.max(l_kl, torch.tensor(WM_FREE_NATS, device=obs_seq.device))

        # ── Multi-Horizon Return Losses (MDN-aware) ──────────────────────────
        use_mdn = getattr(self, "_use_mdn", False)
        ret_trunk_for_mdn = outputs.get("ret_trunk") if use_mdn else None
        horizon_losses = {}
        for h in ACTIVE_HORIZONS:
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

        # ── Regime classification loss ────────────────────────────────────────
        # Use t+1 returns to define regime
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

        # ── Uncertainty-weighted total (decomposed for PCGrad) ──────────────
        s = self.log_vars.clamp(-6.0, 6.0)
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        rec_term = torch.exp(-s_rec) * l_rec + 0.5 * s_rec
        kl_term = torch.exp(-s[1]) * l_kl * kl_anneal + 0.5 * s[1]

        # Per-horizon weighted return terms (separate for PCGrad)
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

        # Direct return (MDN-aware)
        l_direct_return = torch.tensor(0.0, device=obs_seq.device)
        for h in ACTIVE_HORIZONS:
            if h in target_returns:
                if use_mdn:
                    decoded = self.return_heads[str(h)].expectation(ret_trunk_for_mdn)
                else:
                    decoded = self.bucketer.decode(outputs["return_logits"][h])
                # Round-7: tail-adaptive Huber for crypto fat-tail returns
                l_direct_return = l_direct_return + tail_adaptive_huber(
                    decoded.reshape(-1), target_returns[h].reshape(-1),
                    delta=0.5, tail_sigma=2.0, tail_weight=2.5
                )
        direct_term = DIRECT_RETURN_WEIGHT * l_direct_return

        # ── Forecast head loss (2026-05-10 generalization fix) ─────────────
        # Predict obs[t+h] from h_seq[t]. Anchors h_seq to feature-faithful
        # future prediction. Empirical winner: D (forecast + atme=0.20) lifted
        # train_IC@h1 from 0.0176 (baseline) to 0.0330 (+88%) on 400-step
        # real-data probe. See scripts/v4_diag/probe_v4_options.py.
        try:
            from settings import USE_FORECAST_HEAD as _use_fc, FORECAST_WEIGHT as _fc_w
        except ImportError:
            _use_fc, _fc_w = False, 0.0
        l_forecast = torch.tensor(0.0, device=obs_seq.device)
        if _use_fc and outputs.get("forecast_logits") is not None:
            for h in REWARD_HORIZONS:
                if h >= T:
                    continue
                fc_pred = outputs["forecast_logits"][h][:, :-h, :]   # [B, T-h, F]
                fc_tgt = obs_seq[:, h:, :]                           # [B, T-h, F]
                l_forecast = l_forecast + F.mse_loss(fc_pred, fc_tgt)
            l_forecast = l_forecast / len(REWARD_HORIZONS)
        forecast_term = _fc_w * l_forecast

        # CC-H5 quantile loss (SOTA-2026, auxiliary).
        try:
            from settings import QUANTILE_LOSS_WEIGHT as _ql_w
        except ImportError:
            _ql_w = 0.0
        l_quantile = torch.tensor(0.0, device=obs_seq.device)
        if (self.quantile_heads is not None
                and outputs.get("quantile_logits") is not None
                and _ql_w > 0):
            import sys as _qsys
            from pathlib import Path as _qPath
            _sd = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
            if _sd not in _qsys.path:
                _qsys.path.insert(0, _sd)
            from headline_components import quantile_loss as _ql
            quants = self.quantile_heads.quantiles
            n_h = 0
            for h in REWARD_HORIZONS:
                if h in target_returns:
                    q_pred = outputs["quantile_logits"][h]
                    q_tgt = target_returns[h]
                    l_quantile = l_quantile + _ql(q_pred, q_tgt, quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        quantile_term = _ql_w * l_quantile

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary).
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0):
            flat_labels = regime_labels.reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h in REWARD_HORIZONS:
                    if h not in target_returns:
                        continue
                    head_logits = outputs["regime_cond_logits"][r][h]
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = target_returns[h].reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        regime_cond_term = _rc_w * l_regime_cond

        aux_term = (rec_term + kl_term + regime_term + direct_term
                     + forecast_term + quantile_term + regime_cond_term)
        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        # ── Forward-regime aux loss (V4_FORWARD_REGIME flag, Layer-C wiring) ───
        # Guarded: only fires when head is attached (_use_forward_regime=True) AND
        # the trainer supplied targets["forward_regime_labels"]. Default OFF ->
        # total unchanged. Weight 0.10 (additive; does not interact with log_vars).
        # NaN rows (last K bars per series) are auto-masked by forward_regime_aux_loss.
        l_fr_aux = torch.tensor(0.0, device=obs_seq.device)
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = target_returns.get("forward_regime_labels", {})
            if _fr_labels:
                import sys as _frsys
                from pathlib import Path as _frPath
                _shared_dir_fr = str(_frPath(__file__).resolve().parent.parent.parent / "_shared")
                if _shared_dir_fr not in _frsys.path:
                    _frsys.path.insert(0, _shared_dir_fr)
                from forward_regime_head import forward_regime_aux_loss as _fr_aux_loss
                l_fr_aux = _fr_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * l_fr_aux

        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "forecast_mse": l_forecast.item(),
            "quantile_pinball": l_quantile.item(),
            "regime_cond_ce": l_regime_cond.item(),
            "fr_aux": l_fr_aux.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        if return_components:
            components = {"aux": aux_term, **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS}}
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs

    # ── Inference Methods ─────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """
        Encode a sequence and return hidden states + posterior latents.
        Returns: h_seq, z_seq, return_preds (dict by horizon)
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        if getattr(self, "_use_mdn", False):
            ret_trunk = outputs.get("ret_trunk")
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk)
        else:
            for h in ACTIVE_HORIZONS:
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

        Mamba is stateless, so dream_gru provides recurrent state evolution
        for multi-step imagination in agent's DREAM mode.

        Returns: h_next, z_next, gru_hidden, return_preds
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
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk_out)
        elif self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk_out)
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.bucketer.decode(mtp_out[f"h{h}"])
        else:
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.bucketer.decode(self.return_heads[str(h)](ret_trunk_out))

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = MambaWorldModel().to(DEVICE)
    print(f"World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, 96
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"Loss: {loss.item():.4f}")
    print(f"Breakdown: {loss_dict}")

    # Test encode
    h, z, preds = model.encode_sequence(obs, asset)
    print(f"Hidden: {h.shape}, Latent: {z.shape}")
    for h_val, p in preds.items():
        print(f"  Return t+{h_val}: {p.shape}")

    print("V4 Mamba-3 world model sanity check passed.")