"""
V8 World Model -- Neural ODE with Continuous-Time Dynamics

Architecture:
  - Obs Encoder: Linear projection + asset embedding
  - Neural ODE Core: dh/dt = f_theta(h, t, obs_t) solved via RK4
  - RSSM Latents: Prior/Posterior categorical distributions
  - Heads: Reconstruction, Multi-Horizon Returns (TwoHot), Regime classification

Key design decisions:
  - Continuous-time dynamics via Neural ODE (not discrete RNN/Transformer steps)
  - RK4 fixed-step integrator for stable, efficient integration
  - Observation-conditioned dynamics (ODE sees current obs at each step)
  - Sinusoidal time encoding for temporal awareness
  - RSSM latent space for stochastic state modeling
  - Dynamics regularization penalizes ||f(h,t,obs)||^2 for smooth trajectories
  - Multi-horizon return heads at [1, 4, 16, 64] bars prevent myopia
  - Uncertainty-weighted loss (Kendall) + direct Huber return loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

try:
    from components import TwoHotSymlog, SwiGLU, MLPHead, ODEDynamics, RK4Solver, RMSNorm
    from settings import *
except ImportError:
    from .components import TwoHotSymlog, SwiGLU, MLPHead, ODEDynamics, RK4Solver, RMSNorm
    from .settings import *

# ==============================================================================
# SHARED LEVERS -- Variable Selection Network + Forward-Regime Head
# (generalized in src/wm/_shared/; same try/except pattern as V1.1)
# ==============================================================================
try:
    from wm._shared.variable_selection import VariableSelectionNetwork
except ImportError:
    import sys as _sys_vsn
    from pathlib import Path as _Path_vsn
    _shared_dir_vsn = str(_Path_vsn(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_dir_vsn not in _sys_vsn.path:
        _sys_vsn.path.insert(0, _shared_dir_vsn)
    from variable_selection import VariableSelectionNetwork


class NeuralODEWorldModel(nn.Module):
    """
    The V8 World Model -- Neural ODE with continuous-time market dynamics.

    Inputs:
        obs_seq:  [B, T, INPUT_DIM]  -- feature sequences
        asset_id: [B]                -- integer asset index (0-4)

    Training targets:
        target_returns: dict of {horizon: [B, T] tensor} for horizons [1, 4, 16, 64]
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
    ):
        super().__init__()

        if ode_hidden_layers is None:
            ode_hidden_layers = ODE_HIDDEN_LAYERS

        self.d_model = d_model
        self.input_dim = input_dim
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
        # ODE_METHOD is read from settings (rk4|rk2|euler). Default rk2 since
        # 2026-04-27 audit (2x speedup; see settings.py ODE_METHOD comment).
        try:
            _method = ODE_METHOD
        except NameError:
            _method = "rk4"
        self.solver = RK4Solver(self.dynamics, substeps=ODE_SUBSTEPS, method=_method)

        # == RSSM Latent Heads ==================================================
        self.prior_head = MLPHead(d_model, 256, self.flat_dim, dropout)
        self.posterior_head = MLPHead(d_model + input_dim, 256, self.flat_dim, dropout)

        # == Output Heads =======================================================
        head_input_dim = d_model + self.flat_dim

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            SwiGLU(head_input_dim, 256, dim_out=256, dropout=dropout),
            RMSNorm(256),
            nn.Linear(256, input_dim),
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

        # == VSN (Variable Selection Network, V8_VSN flag) =====================
        # Flag-gated: constructed ONLY when env var V8_VSN="1" at model init time.
        # When OFF (default): module not present -> forward path byte-for-byte unchanged.
        # When ON: VariableSelectionNetwork sits BEFORE initial_encoder, gates [B,T,input_dim].
        # Combinable with V8_FORWARD_REGIME (both ON = full world-class candidate).
        import os as _os_vsn
        self._use_vsn = _os_vsn.environ.get("V8_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None  # not constructed; no parameters, no side effects

        # == Forward-Regime Head (V8_FORWARD_REGIME flag) ======================
        # Default OFF: _use_forward_regime=False, forward_regime_head=None.
        # Attach via attach_forward_regime_head() (called in trainer when flag ON).
        # The guarded block in forward_train reads feat=cat(h_seq,z_post) -- the
        # SAME fused feature the existing return/regime heads read; no new input
        # plumbing. OFF by default so base path is byte-for-byte unchanged.
        self._use_forward_regime = False
        self.forward_regime_head = None

        # == Dream step (agent imagination) =====================================
        # Neural ODE is stateless -- dream_gru provides recurrent state evolution
        self.dream_proj = nn.Linear(d_model + self.flat_dim, d_model)
        self.dream_gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)

        # == Forecast heads (2026-05-10 generalization fix) =====================
        # Predict obs[t+h] from h_seq[t] (ODE-integrated state). Anchors h_seq to
        # feature-faithful future prediction — Neural ODE's natural objective.
        # Same pattern as V4/V22/V25; V4 probe-validated +88% train_IC lift.
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

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026 auxiliary heads / encoder cond)
        import sys as _qsys
        from pathlib import Path as _qPath
        _shared = str(_qPath(__file__).resolve().parent.parent.parent / "_shared")
        if _shared not in _qsys.path:
            _qsys.path.insert(0, _shared)
        try:
            from settings import USE_QUANTILE_HEADS as _use_qh
        except ImportError:
            _use_qh = False
        if _use_qh:
            from headline_components import QuantileHeads as _QH
            self.quantile_heads = _QH(head_input_dim=head_input_dim,
                                         horizons=tuple(REWARD_HORIZONS))
        else:
            self.quantile_heads = None
        try:
            from settings import USE_REGIME_COND_HEADS as _use_rc
        except ImportError:
            _use_rc = False
        if _use_rc:
            from headline_components import RegimeConditionalHeads as _RC
            self.regime_cond_heads = _RC(head_input_dim=head_input_dim,
                                            horizons=tuple(REWARD_HORIZONS),
                                            num_bins=num_bins, n_regimes=3)
        else:
            self.regime_cond_heads = None
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
                # Skip gate_proj: its neutral-start init (std=0.01, bias=0 -> gates ~0.5)
                # is set in VariableSelectionNetwork.__init__ and must NOT be overwritten by
                # xavier_uniform (mirrors the V1.1 _init_weights guard, 2026-06-10).
                if "gate_proj" in name:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def _get_stoch_state(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical latent using Gumbel-Softmax (fp32-safe)."""
        orig_dtype = logits.dtype
        shape = logits.shape
        reshaped = logits.float().view(*shape[:-1], self.latent_dim, self.classes)
        tau = getattr(self, '_gumbel_tau', GUMBEL_TAU)
        z = F.gumbel_softmax(reshaped, tau=tau, hard=True, dim=-1)
        return z.view(*shape).to(orig_dtype)

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

        # VSN: per-timestep feature gate applied BEFORE initial_encoder (causal -- uses only x_t).
        # When V8_VSN=0 (default): self.vsn is None, branch skipped, path identical to base.
        # When V8_VSN=1: input_obs is replaced with gate * input_obs (same shape).
        # Gate is differentiable -> gradients flow back through the VSN weights.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

        # 1. Encode observations + asset embedding
        asset_emb = self.asset_embedding(asset_id)               # [B, emb]
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)     # [B, T, emb]
        enc_input = torch.cat([input_obs, asset_emb], dim=-1)    # [B, T, INPUT_DIM + emb]
        encoded = self.initial_encoder(enc_input)                 # [B, T, D]

        # 2. Take h0 = encoded[:, 0, :] as initial state
        h0 = encoded[:, 0, :]  # [B, D]

        # 3. Prepare obs for ODE conditioning
        # ODE reads MASKED input_obs not raw obs_seq. Previously the ODE
        # received the unmasked sequence at every step, so block-masking
        # only affected h0 (the initial state) — masking was a near-no-op
        # for V8. Fixed 2026-05-21 RED-team audit.
        obs_with_asset = torch.cat([input_obs, asset_emb], dim=-1)  # [B, T, INPUT_DIM + emb]
        obs_for_ode = self.obs_proj(obs_with_asset)                  # [B, T, INPUT_DIM]

        # 4. Time span
        t = torch.arange(T, device=obs_seq.device, dtype=torch.float32)

        # 5. Solve ODE in bf16 — upgraded from fp32 (2026-04-16).
        # RK4 does 4 evals per timestep, accumulating over 96*4=384 evaluations.
        # fp16 overflowed (V3 GN:inf pattern). bf16 has the SAME dynamic range
        # as fp32 (1e38) with ~2x compute density on Ampere+. No scaler needed
        # for bf16 backward. Preserves stability while eliminating the ~70%
        # of forward-pass time that fp32 was costing us.
        # cache_enabled=False is CRITICAL: without it, the bf16 cast of
        # dynamics.net weights persists in the autocast weight cache and
        # is re-used when dynamics_regularization runs under the outer
        # fp16 autocast, producing "mat1 Half vs mat2 BFloat16" errors.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            h_seq = self.solver(h0.to(torch.bfloat16), obs_for_ode.to(torch.bfloat16), t)
        # Return to fp32 so downstream heads (under outer fp16 autocast) see a
        # well-defined dtype and the scaler path matches the original training.
        h_seq = h_seq.float()

        # RegimeFiLM (SOTA-2026 opt-in): conditions ODE-integrated h_seq on
        # regime BEFORE RSSM. Identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # 6. RSSM: Prior from h_seq, Posterior from cat(h_seq, input_obs)
        # Posterior reads MASKED input_obs not raw obs_seq. Fixed 2026-05-21.
        # Clamp logits to ±10 — prevents KL inf (same fix as V3).
        prior_logits = self.prior_head(h_seq).clamp(-10, 10)
        post_input = torch.cat([h_seq, input_obs], dim=-1)
        post_logits = self.posterior_head(post_input).clamp(-10, 10)

        # 7. Sample posterior latent
        z_post = self._get_stoch_state(post_logits)

        # 8. Feature vector for heads
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, D + flat_dim]

        # 9. Heads
        recon = self.decoder(feat)  # reconstruction always uses full temporal context

        # ATME: per-sample anti-temporal-memorization (V1.6-class)
        # h_seq.detach() in normal path: return heads READ ODE dynamics
        # but can't OPTIMIZE the ODE integrator for memorization.
        feat_heads = torch.cat([h_seq.detach(), z_post], dim=-1)
        if self.training and temporal_ctx_drop > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device) < temporal_ctx_drop)
            # z_post = f(posterior(h_seq, obs)) leaked temporal info, so zeroing h_seq
            # in feat_heads ALONE was cosmetic. For ATME samples recompute z_post from a
            # ZEROED h_seq (obs-only) so the temporal drop is real (mirrors V4 fix,
            # 2026-05-29). recon (feat) keeps the full-context z_post -- unchanged.
            h_zeroed = torch.zeros_like(h_seq)
            post_logits_atme = self.posterior_head(
                torch.cat([h_zeroed, input_obs], dim=-1)).clamp(-10, 10)
            z_post_atme = self._get_stoch_state(post_logits_atme)
            feat_atme = torch.cat([h_zeroed, z_post_atme], dim=-1)
            feat_heads = torch.where(atme_mask, feat_atme, feat_heads)

        regime_logits = self.regime_head(feat_heads)

        # Multi-horizon return predictions
        ret_trunk_out = self.return_trunk(feat_heads)
        return_logits = {}
        for h in REWARD_HORIZONS:
            return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        # Forecast heads (added 2026-05-10) -- predict obs[t+h] from h_seq[t]
        # 2026-05-10 V8 fix: bf16 ODE solver can emit NaN/inf h_seq values
        # that cascade through forecast_heads into NaN losses → NaN gradients
        # → entire training collapses to 100% NaN. Defensive cast + nan_to_num.
        forecast_logits = None
        if self.forecast_heads is not None:
            h_seq_fc = torch.nan_to_num(h_seq.float(), nan=0.0, posinf=10.0, neginf=-10.0)
            forecast_logits = {
                int(h): self.forecast_heads[str(h)](h_seq_fc) for h in REWARD_HORIZONS
            }

        # CC-H5 quantile heads (SOTA-2026, auxiliary)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat_heads)

        # CC-H6 regime-conditional heads (SOTA-2026, auxiliary)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat_heads)

        out = {
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
            "forecast_logits": forecast_logits,  # dict: {h: [B, T, INPUT_DIM]} or None
            "quantile_logits": quantile_logits,  # dict: {h: [B, T, n_quantiles]} or None
            "regime_cond_logits": regime_cond_logits,  # {regime: {h: logits}} or None
        }
        # Forward-regime head (V8_FORWARD_REGIME flag, 2026-06-10).
        # OFF by default: _use_forward_regime=False until attach_forward_regime_head() is called.
        # When OFF: no 'forward_regime' key in out -- base path byte-for-byte unchanged.
        # When ON: feat = cat(h_seq, z_post) is the SAME fused feature the existing heads use;
        #          the head reads PAST-ENCODED state -> no look-ahead (labels are future-only,
        #          computed at target-construction time; head inputs are causal encoder outputs).
        if getattr(self, "_use_forward_regime", False) and self.forward_regime_head is not None:
            out["forward_regime"] = self.forward_regime_head(feat)  # {bear_logits,trend_logits,move_logits}
        return out

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

            # Pre-encode time into [4] features so we take the fast path
            # in ODEDynamics.forward. Use h_t.dtype to keep all inputs to
            # the Linear consistent (h, time_feat, obs_t all same dtype).
            time_feat = self.dynamics.encode_time(
                t_val, device=h_t.device, dtype=h_t.dtype)
            # Make obs_t match so cat doesn't mix fp16+fp32 dtypes
            obs_t = obs_t.to(h_t.dtype)
            dh_dt = self.dynamics(h_t, time_feat, obs_t)  # [B, D]
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

        # == XD anti-memorization augmentation (training only) ==================
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

        # == Reconstruction loss ================================================
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)

        # == KL Divergence ======================================================
        # fp32 + nan_to_num + clamp — D.Categorical crashes on ANY NaN/inf in logits
        prior = outputs["prior_logits"].float().view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].float().view(-1, self.latent_dim, self.classes)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
        post = torch.nan_to_num(post, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)

        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior),
        ).mean()
        kl_raw = l_kl.item()
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
        # nan_to_num guard (2026-05-29): the ODE trajectory can emit NaN/inf on
        # instability; the forecast path sanitizes it but this regularization got
        # the RAW h_seq -> NaN loss -> NaN grad -> silent training crash.
        _h_seq_safe = torch.nan_to_num(outputs["h_seq"], nan=0.0, posinf=0.0, neginf=0.0)
        # Guard obs_for_ode symmetrically (the masking path can zero/propagate NaN);
        # t is torch.arange -> always finite, no guard needed.
        _obs_safe = torch.nan_to_num(outputs["obs_for_ode"], nan=0.0, posinf=0.0, neginf=0.0)
        l_dynamics = self.dynamics_regularization(
            _h_seq_safe, _obs_safe, outputs["t"]
        )
        total = total + LAMBDA_DYNAMICS * l_dynamics

        # == Forecast head loss (2026-05-10 generalization fix) =================
        # MSE(forecast_logits[h][:t-h], obs_seq[t+h:]) per horizon. Anchors
        # ODE-integrated state to feature-faithful future prediction. Same
        # pattern as V4/V22/V25; V4 probe lifted train_IC +88%.
        try:
            from settings import USE_FORECAST_HEAD as _use_fc, FORECAST_WEIGHT as _fc_w
        except ImportError:
            _use_fc, _fc_w = False, 0.0
        l_forecast = torch.tensor(0.0, device=obs_seq.device)
        if _use_fc and outputs.get("forecast_logits") is not None:
            T_obs = obs_seq.shape[1]
            for h in REWARD_HORIZONS:
                if h >= T_obs:
                    continue
                fc_pred = outputs["forecast_logits"][h][:, :-h, :]   # [B, T-h, F]
                fc_tgt = obs_seq[:, h:, :]                           # [B, T-h, F]
                # 2026-05-10 V8 NaN guard: bf16 ODE solver can produce NaN
                # h_seq values that survive nan_to_num through forecast_heads;
                # clamp predictions to a sensible range before MSE.
                fc_pred = torch.nan_to_num(fc_pred, nan=0.0, posinf=10.0, neginf=-10.0)
                fc_pred = fc_pred.clamp(-10.0, 10.0)
                fc_loss = F.mse_loss(fc_pred, fc_tgt)
                if torch.isfinite(fc_loss):
                    l_forecast = l_forecast + fc_loss
            l_forecast = l_forecast / len(REWARD_HORIZONS)
            # Final safety: zero out if still NaN/inf
            if not torch.isfinite(l_forecast):
                l_forecast = torch.tensor(0.0, device=obs_seq.device)
        total = total + _fc_w * l_forecast

        # CC-H5 quantile loss (SOTA-2026, auxiliary)
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
                    l_quantile = l_quantile + _ql(q_pred, target_returns[h], quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary)
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
        total = total + _rc_w * l_regime_cond

        # == Forward-regime aux loss (V8_FORWARD_REGIME flag, 2026-06-10) =======
        # Guarded: only fires when head is attached (_use_forward_regime=True) and
        # the trainer supplied forward_regime_labels. Default OFF -> total unchanged.
        # Weight 0.10 (small fixed; head is additive supervision, not primary objective).
        # Labels built by trainer from src/wm/_shared/regime_targets.py:
        #   bear_lab  = forward_bear_label(close, K=64, dd_thresh=0.05)
        #   trend_lab = forward_trend_label(close, K=64)
        #   move_lab  = move_onset_label(close, a=1, b=64)
        # packed into targets["forward_regime_labels"] = {"bear": [B,T], ...}
        # NaN rows (last K bars per series) are auto-masked by forward_regime_aux_loss.
        l_forward_regime = torch.tensor(0.0, device=obs_seq.device)
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = target_returns.get("forward_regime_labels", {})
            if _fr_labels:
                import sys as _frsys
                from pathlib import Path as _frPath
                _shared_dir_fr = str(_frPath(__file__).resolve().parent.parent.parent / "_shared")
                if _shared_dir_fr not in _frsys.path:
                    _frsys.path.insert(0, _shared_dir_fr)
                from forward_regime_head import forward_regime_aux_loss as _fr_aux_loss
                l_forward_regime = _fr_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * l_forward_regime

        # == Build loss dict ====================================================
        loss_dict = {
            "total": total.item(),
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "kl_raw": kl_raw,
            "regime": l_regime.item(),
            "regime_acc": regime_acc,
            "direct_ret": l_direct_return.item(),
            "dynamics_reg": l_dynamics.item(),
            "forecast_mse": l_forecast.item(),
            "quantile_pinball": l_quantile.item(),
            "regime_cond_ce": l_regime_cond.item(),
            "fr_aux": l_forward_regime.item(),
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

    # == Dream Step (agent imagination) ========================================

    @torch.no_grad()
    def dream_step(
        self,
        h_prev: torch.Tensor,
        z_prev: torch.Tensor,
        gru_hidden: torch.Tensor = None,
    ):
        """
        One-step imagination using dream GRU (no observation).

        Neural ODE is stateless in the dream context, so dream_gru provides
        recurrent state evolution for multi-step imagination.

        Returns: h_next, z_next, gru_hidden, return_preds
        """
        combined = torch.cat([h_prev, z_prev], dim=-1)
        gru_input = self.dream_proj(combined).unsqueeze(1)
        h_next, gru_hidden = self.dream_gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)

        prior_logits = self.prior_head(h_next).clamp(-10, 10)
        z_next = self._get_stoch_state(prior_logits)
        feat = torch.cat([h_next, z_next], dim=-1)

        ret_trunk_out = self.return_trunk(feat)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(self.return_heads[str(h)](ret_trunk_out))

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = NeuralODEWorldModel().to(DEVICE)
    print(f"V8 Neural ODE World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, 96
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
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

    print("V8 Neural ODE world model sanity check passed.")
