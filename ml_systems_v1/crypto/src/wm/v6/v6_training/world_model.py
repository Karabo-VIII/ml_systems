"""
V6 World Model -- Causal JEPA with Adversarial Time Shuffling

Architecture:
  - Context Encoder: CausalGRU (NOT BiGRU) for encoding observations (online, trained)
  - Target Encoder: EMA-updated copy of context encoder (momentum encoder)
  - Predictor Network: Predicts future target embeddings from context
  - Latent Projector: Maps encoder output to latent space
  - Time Discriminator: Adversarial classifier penalizing temporal dependence in latents
  - Prediction Heads: Multi-horizon returns, regime classification
  - Auxiliary Reconstruction Head: Light decoder for input reconstruction

Key Fix vs V2:
  V2's BiGRU caused catastrophic temporal overfitting because the bidirectional
  encoder could peek at future timesteps, learning spurious temporal correlations
  that don't exist at inference time.

  V6 fixes this with two mechanisms:
    1. CausalGRU (unidirectional) -- representations at time t only depend on t' <= t
    2. Time Discriminator -- adversarially penalizes any remaining temporal dependence
       in the latent space by training the encoder to produce latents that are
       indistinguishable from time-shuffled versions

Losses:
  1. InfoNCE (per-timestep, memory-safe): Contrastive alignment of predicted vs target
  2. VICReg: Prevents representation collapse (variance + covariance regularization)
  3. Auxiliary reconstruction (MSE, small weight) -- helps with small data
  4. Adversarial loss (encoder fools discriminator) + Discriminator loss (separate opt)
  5. Multi-horizon return prediction (TwoHot Symlog, 255 bins, [-1, 1])
  6. Direct Huber return loss (bypasses TwoHot discretization bottleneck)
  7. Regime classification (bull/neutral/bear)
  8. Uncertainty-weighted balancing (Kendall et al.)

World-class levers (flag-gated, default OFF -- base is byte-for-byte unchanged):
  V6_VSN=1             -- Variable Selection Network: per-timestep causal feature gate
                          applied BEFORE obs_proj. Gate g_t = sigmoid(W * x_t),
                          x'_t = g_t * x_t. Causal, no look-ahead.
  V6_FORWARD_REGIME=1  -- Forward regime/move-onset head on vib_expand(z_vib) (feat).
                          Three binary/3-class MLP heads (bear / trend / move), supervised
                          by forward labels from regime_targets.py.  aux weight 0.10.
"""
import os as _os_levers
import sys as _sys_levers
from pathlib import Path as _Path_levers
import torch
import torch.nn as nn
import torch.nn.functional as F

# Support both script (`python v6_training/train_world_model.py`) and package
# (`from v6.v6_training.world_model import ...`) imports.
try:
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
except ImportError:
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

# ==============================================================================
# SHARED LEVERS -- VSN + ForwardRegimeHead (imported from src/wm/_shared/)
# ==============================================================================
# Each import uses a two-stage try/except identical to V1.1 so both script
# and package invocations resolve correctly.  The _shared dir is added to
# sys.path only once and only if needed.

def _ensure_shared_on_path():
    _shared = str(_Path_levers(__file__).resolve().parent.parent.parent / "_shared")
    if _shared not in _sys_levers.path:
        _sys_levers.path.insert(0, _shared)

try:
    from wm._shared.variable_selection import VariableSelectionNetwork
except ImportError:
    _ensure_shared_on_path()
    from variable_selection import VariableSelectionNetwork

try:
    from wm._shared.forward_regime_head import (
        ForwardRegimeHead as _ForwardRegimeHead,
        forward_regime_aux_loss as _forward_regime_aux_loss,
    )
except ImportError:
    _ensure_shared_on_path()
    from forward_regime_head import (
        ForwardRegimeHead as _ForwardRegimeHead,
        forward_regime_aux_loss as _forward_regime_aux_loss,
    )


class CausalJEPAWorldModel(nn.Module):
    """V6: Causal JEPA + Adversarial Time Shuffling world model."""

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
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.d_latent = d_latent

        # ── Asset conditioning ───────────────────────────────────────────────
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)

        # ── Observation projection (online, trained) ──────────────────────────
        self.obs_proj = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── Target observation projection (EMA-updated, no gradients) ────────
        # CRITICAL FIX: target branch must use its own EMA-updated obs_proj,
        # not the online obs_proj. Sharing obs_proj defeats JEPA's slow-moving
        # target assumption because obs_proj updates every gradient step.
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

        # ── Variational Information Bottleneck (VIB) ─────────────────────────
        # Replaces discriminator as primary anti-memorization defense.
        # ctx_latent [d_latent] -> VIB [z_dim] -> expand back [d_latent]
        self.vib_z_dim = VIB_Z_DIM
        self.vib_mu = nn.Linear(d_latent, VIB_Z_DIM)
        self.vib_logvar = nn.Linear(d_latent, VIB_Z_DIM)
        nn.init.zeros_(self.vib_logvar.weight)
        nn.init.constant_(self.vib_logvar.bias, VIB_LOGVAR_INIT)
        self.vib_expand = nn.Sequential(
            nn.Linear(VIB_Z_DIM, d_latent),
            RMSNorm(d_latent),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

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
        # Light decoder: latent -> input features (no asset embedding)
        self.recon_head = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
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

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026 auxiliary heads).
        # Imports the shared modules; opt-in via settings flags.
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

        # RegimeFiLM on ctx_latent (BEFORE VIB) — identity-at-init.
        # JEPA's encoder is supervised by InfoNCE/VICReg/disc; FiLM here adds
        # regime-aware modulation that gets gradient signal via the VIB path.
        try:
            from settings import REGIME_AWARENESS_MODE as _ram
        except ImportError:
            _ram = "heads"
        self._regime_awareness_mode = _ram
        if _ram == "film":
            from headline_components import RegimeFiLM as _RF
            self.regime_film = _RF(d_model=d_latent, n_regimes=3)
            self.regime_film_gate = nn.Linear(d_latent, 3)
            nn.init.zeros_(self.regime_film_gate.weight)
            nn.init.zeros_(self.regime_film_gate.bias)
        else:
            self.regime_film = None
            self.regime_film_gate = None

        # EMA of regime ret_std for stable regime labels across batches
        self.register_buffer('_regime_ret_std_ema', torch.tensor(1.0), persistent=False)

        # ── Dream step (agent imagination) ────────────────────────────────
        self.dream_proj = nn.Linear(d_model + d_latent, d_model)

        # ── World-class lever: VSN (V6_VSN=1, default OFF) ───────────────────
        # Per-timestep causal feature gate applied BEFORE obs_proj.
        # When OFF (default): module not constructed; forward path byte-for-byte
        # identical to base. When ON: VariableSelectionNetwork sits before
        # obs_proj, gates [B,T,input_dim] with g_t = sigmoid(W*x_t)*x_t.
        # Causal by construction (gate at t depends only on x_t).
        self._use_vsn = _os_levers.environ.get("V6_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None  # not constructed; no parameters, no side effects

        # ── World-class lever: Forward-regime head (V6_FORWARD_REGIME=1, OFF) ─
        # Three small MLP heads (bear / trend / move) on feat = vib_expand(z_vib)
        # [B, T, d_latent].  Attached here at __init__ time; labels are supplied
        # by the trainer at training time via targets["forward_regime_labels"].
        # When OFF (default): _use_forward_regime=False, head=None; the guarded
        # block in forward_train/get_loss is a no-op -> base unchanged.
        self._use_forward_regime = _os_levers.environ.get("V6_FORWARD_REGIME", "0") == "1"
        if self._use_forward_regime:
            # feat_dim for V6 = d_latent (the post-VIB representation used by all
            # existing heads; NOT d_model + flat_dim which is V1's RSSM fused feat).
            self.forward_regime_head = _ForwardRegimeHead(d_latent, hidden=128, dropout=dropout)
        else:
            self.forward_regime_head = None  # not constructed; no parameters

        # ── Initialize target encoder + target obs_proj as exact copies, then freeze ──
        self._copy_context_to_target()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_obs_proj.parameters():
            p.requires_grad = False

        # ── Initialize weights ───────────────────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training start."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # VSN neutral-start guard: gate_proj uses std=0.01 / bias=0 init
                # set in VariableSelectionNetwork.__init__; xavier_uniform here
                # would overwrite it and push initial gates away from 0.5.
                # Identical guard as V1.1 _init_weights (CLAUDE.md pattern).
                if "gate_proj" in name:
                    continue
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
        """Copy context encoder + obs_proj weights to target counterparts (hard copy)."""
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
        EMA update of target obs_proj + encoder + latent projector from online counterparts.

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

        # VSN: per-timestep feature gate BEFORE obs_proj (causal -- uses only x_t).
        # When V6_VSN=0 (default): self.vsn is None, branch skipped, path unchanged.
        # When V6_VSN=1: input_obs is replaced with gate * input_obs (same shape).
        # The gate is differentiable -> gradients flow back through the gate weights.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

        # 1. Encode observations with asset embedding
        asset_emb = self.asset_embedding(asset_id)            # [B, asset_emb_dim]
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, asset_emb_dim]

        # Context branch (online encoder -- receives masked input)
        ctx_input = torch.cat([input_obs, asset_emb], dim=-1)  # [B, T, input_dim + asset_emb_dim]
        ctx_emb = self.obs_proj(ctx_input)                      # [B, T, d_model]
        ctx_hidden = self.context_encoder(ctx_emb)              # [B, T, d_model]
        ctx_latent = self.context_latent_proj(ctx_hidden)       # [B, T, d_latent]

        # RegimeFiLM (SOTA-2026 opt-in): modulate ctx_latent BEFORE VIB.
        # Identity-at-init so safe to enable on a planned cold-start.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(ctx_latent)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            ctx_latent = self.regime_film(ctx_latent, regime_probs_film)

        # Target branch (EMA obs_proj + EMA encoder -- receives UNMASKED input)
        with torch.no_grad():
            tgt_input = torch.cat([obs_seq, asset_emb], dim=-1)
            tgt_emb = self.target_obs_proj(tgt_input)
            tgt_hidden = self.target_encoder(tgt_emb)
            tgt_latent = self.target_latent_proj(tgt_hidden)    # [B, T, d_latent]

        # 2. Variational Information Bottleneck (VIB)
        # Hard rate constraint: limits temporal info that reaches downstream heads.
        # ctx_latent [192] -> z [48] -> feat [192]
        vib_mu = self.vib_mu(ctx_latent)
        vib_logvar = self.vib_logvar(ctx_latent).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * vib_logvar)
            z_vib = vib_mu + std * torch.randn_like(vib_mu)
        else:
            z_vib = vib_mu
        feat = self.vib_expand(z_vib)  # [B, T, d_latent]

        # 3. Predict future embeddings (through VIB -- KL-constrained)
        pred_latent = self.predictor(feat)                      # [B, T, d_latent]

        # 4. Prediction heads -- DETACHED from encoder
        # Return loss (7.4x dominant) can't drive GRU temporal memorization.
        # Encoder gets gradient only from: InfoNCE, VICReg, recon, disc, regime.
        ret_trunk_out = self.return_trunk(feat.detach())
        return_logits = {}
        for h in REWARD_HORIZONS:
            return_logits[h] = self.return_heads[str(h)](ret_trunk_out)

        regime_logits = self.regime_head(feat.detach())

        # CC-H5 quantile heads (SOTA-2026, auxiliary).
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat.detach())

        # CC-H6 regime-conditional heads (SOTA-2026, auxiliary).
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat.detach())

        # 5. Auxiliary reconstruction (through VIB, gives encoder gradient)
        recon = self.recon_head(feat)                           # [B, T, input_dim]

        out = {
            "ctx_latent": ctx_latent,
            "feat": feat,           # Post-VIB representation
            "vib_mu": vib_mu,
            "vib_logvar": vib_logvar,
            "tgt_latent": tgt_latent,
            "pred_latent": pred_latent,
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
            "recon": recon,
            "ret_trunk": ret_trunk_out,  # Exposed for adapter
        }

        # Forward-regime head (V6_FORWARD_REGIME lever, 2026-06-10).
        # OFF by default: _use_forward_regime is False; block is a no-op.
        # When ON: head reads feat (post-VIB, d_latent) -- PAST observations only,
        # NO look-ahead. Future bars are used only to BUILD the labels (target-
        # construction time, in the trainer), never as inputs here.
        if getattr(self, "_use_forward_regime", False) and self.forward_regime_head is not None:
            out["forward_regime"] = self.forward_regime_head(feat)  # {bear/trend/move logits}

        return out

    def get_loss(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor,
        target_returns: dict,
        mask_ratio: float = 0.15,
        block_mask: bool = True,
        regime_labels: torch.Tensor = None,
        **kwargs,
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

        # ── 3b. VIB KL loss ──────────────────────────────────────────────────
        vib_mu = outputs["vib_mu"]
        vib_logvar = outputs["vib_logvar"]
        l_vib_kl = (-0.5 * (1 + vib_logvar - vib_mu.pow(2) - vib_logvar.exp())).mean()
        kl_anneal = kwargs.get("kl_anneal", 1.0)
        vib_kl_weight = VIB_KL_WEIGHT * kl_anneal

        # ── 4. VICReg loss (collapse prevention) ─────────────────────────────
        # Use post-VIB feat for VICReg (ensures VIB output doesn't collapse)
        ctx_flat = outputs["feat"].reshape(-1, self.d_latent)
        tgt_flat = outputs["tgt_latent"].reshape(-1, self.d_latent)
        l_vicreg = self.vicreg_loss_fn(
            ctx_flat, tgt_flat,
            sim_w=VICREG_SIM_WEIGHT,
            var_w=VICREG_VAR_WEIGHT,
            cov_w=VICREG_COV_WEIGHT,
        )

        # ── 5. Auxiliary reconstruction loss ─────────────────────────────────
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_recon = F.mse_loss(outputs["recon"], recon_target)

        # ── 6. Adversarial loss ──────────────────────────────────────────────
        # Discriminator targets pre-VIB ctx_latent (the encoder residual) per
        # HEADLINE_MODE spec — temporal memorization originates upstream of
        # the VIB bottleneck, so adversarial pressure must hit ctx_latent
        # not the compressed feat. All four refs (real/fake/grad-pen/enc-adv)
        # must use the same representation to keep D's positives and
        # negatives drawn from the same distribution.
        disc_target = outputs["ctx_latent"]
        real_score = self.discriminator(disc_target.detach())  # [B]

        # Create time-shuffled version
        perm = torch.randperm(T, device=obs_seq.device)
        shuffled_latent = disc_target[:, perm, :].detach()  # [B, T, D]
        fake_score = self.discriminator(shuffled_latent)  # [B]

        # Binary cross-entropy with label smoothing (prevents disc from winning trivially)
        smooth = getattr(self, '_disc_label_smooth', DISC_LABEL_SMOOTH) if hasattr(self, '_disc_label_smooth') else DISC_LABEL_SMOOTH
        real_target = 1.0 - smooth   # 0.9 instead of 1.0
        fake_target = smooth          # 0.1 instead of 0.0
        l_disc = -real_target * torch.log(real_score + 1e-6).mean() - (1 - fake_target) * torch.log(1.0 - fake_score + 1e-6).mean()

        # Gradient penalty (WGAN-GP) for discriminator stability
        if DISC_GRAD_PENALTY > 0:
            alpha = torch.rand(B, 1, 1, device=obs_seq.device)
            interpolated = alpha * disc_target.detach() + (1 - alpha) * shuffled_latent
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
        # — gradients flow through ctx_latent (pre-VIB) to the encoder.
        encoder_score = self.discriminator(disc_target)  # [B]
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

        # VIB KL loss (hard rate constraint on temporal encoding)
        total = total + vib_kl_weight * l_vib_kl

        # Adversarial encoder loss (fixed weight LAMBDA_ADV, secondary regularizer)
        total = total + LAMBDA_ADV * l_adv

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

        # ── CC-H5 quantile loss (SOTA-2026, auxiliary) ───────────────────────
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

        # ── CC-H6 regime-conditional CE (SOTA-2026, auxiliary) ───────────────
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

        # ── Forward-regime aux loss (V6_FORWARD_REGIME lever) ────────────────
        # Guarded: fires ONLY when head is attached (_use_forward_regime=True) and
        # the trainer supplied forward_regime_labels in kwargs.  Default OFF ->
        # total unchanged (no new keys in loss_dict, no parameter updates).
        # Weight 0.10 (additive, small fixed; head supervision does not disrupt
        # the VIB/InfoNCE/Huber primary objectives).
        l_forward_regime = torch.tensor(0.0, device=obs_seq.device)
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = kwargs.get("forward_regime_labels", {})
            if _fr_labels:
                l_forward_regime = _forward_regime_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * l_forward_regime

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
            "kl": l_vib_kl.item(),
            "kl_raw": l_vib_kl.item(),
            "kl_weight": vib_kl_weight,
            "quantile_pinball": l_quantile.item(),
            "regime_cond_ce": l_regime_cond.item(),
            "forward_regime": l_forward_regime.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        return total, loss_dict, l_disc, outputs

    @torch.no_grad()
    def encode_sequence(self, obs_seq: torch.Tensor, asset_id: torch.Tensor):
        """
        Encode a sequence and return hidden states + latent representations + return predictions.
        Compatible with agent environment's 3-tuple interface.

        Args:
            obs_seq: [B, T, input_dim]
            asset_id: [B]

        Returns:
            h_seq:        [B, T, d_model] GRU hidden states
            z_post:       [B, T, d_latent] projected latents (continuous, not RSSM)
            return_preds: dict {horizon: [B, T]}
        """
        self.eval()

        B, T, _ = obs_seq.shape
        asset_emb = self.asset_embedding(asset_id)
        asset_emb = asset_emb.unsqueeze(1).expand(-1, T, -1)

        ctx_input = torch.cat([obs_seq, asset_emb], dim=-1)
        ctx_emb = self.obs_proj(ctx_input)
        ctx_hidden = self.context_encoder(ctx_emb)         # [B, T, d_model]
        ctx_latent = self.context_latent_proj(ctx_hidden)   # [B, T, d_latent]

        # RegimeFiLM BEFORE VIB, matching forward_train (only active when
        # REGIME_AWARENESS_MODE="film"). FIX 2026-05-29: inference previously
        # skipped this, so when FiLM was on, train/inference diverged.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_probs_film = torch.softmax(self.regime_film_gate(ctx_latent), dim=-1)
            ctx_latent = self.regime_film(ctx_latent, regime_probs_film)

        # Apply the VIB DETERMINISTICALLY (mu, no sampling) before the return
        # trunk, to match the TRAINING representation. FIX 2026-05-29: this path
        # previously fed raw pre-VIB ctx_latent to return_trunk, but the trunk was
        # trained on post-VIB `feat = vib_expand(z)` (forward_train) -> a
        # train/inference distribution mismatch that produced degraded/garbage IC.
        feat = self.vib_expand(self.vib_mu(ctx_latent))      # [B, T, d_latent]

        # Return predictions
        ret_trunk_out = self.return_trunk(feat)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(self.return_heads[str(h)](ret_trunk_out))

        return ctx_hidden, ctx_latent, return_preds

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
        One-step imagination using the context encoder's GRU (no observation).

        JEPA uses continuous latent space (no RSSM prior/posterior split).
        Reuses the CausalGRU for state evolution.

        Args:
            h_prev:     [B, d_model] previous GRU hidden output
            z_prev:     [B, d_latent] previous latent (continuous)
            gru_hidden: [n_layers, B, d_model] GRU hidden state (or None)

        Returns:
            h_next:       [B, d_model]
            z_next:       [B, d_latent]
            gru_hidden:   [n_layers, B, d_model]
            return_preds: dict {horizon: [B]}
        """
        combined = torch.cat([h_prev, z_prev], dim=-1)
        gru_input = self.dream_proj(combined).unsqueeze(1)  # [B, 1, d_model]

        h_next, gru_hidden = self.context_encoder.gru(gru_input, gru_hidden)
        h_next = h_next.squeeze(1)  # [B, d_model]

        z_next = self.context_latent_proj(h_next.unsqueeze(1)).squeeze(1)  # [B, d_latent]

        # Apply the VIB deterministically (mu) before the return trunk, matching
        # the training representation (same fix as encode_sequence) -- dream/agent
        # imagination previously fed raw pre-VIB z_next to the return heads.
        feat_next = self.vib_expand(self.vib_mu(z_next))
        ret_trunk_out = self.return_trunk(feat_next)
        return_preds = {}
        for h in ACTIVE_HORIZONS:
            return_preds[h] = self.bucketer.decode(self.return_heads[str(h)](ret_trunk_out))

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = CausalJEPAWorldModel().to(DEVICE)
    total_params = count_parameters(model)
    print(f"V6 Causal JEPA+Adversarial World Model Parameters: {total_params:,}")

    # Quick sanity check: test forward pass
    B, T = 4, 96
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
    asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)
    targets = {h: torch.randn(B, T).to(DEVICE) * 0.01 for h in REWARD_HORIZONS}

    loss, loss_dict, l_disc, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Disc Loss:  {l_disc.item():.4f}")
    print(f"Breakdown:  {loss_dict}")

    # Test EMA update
    model.update_target_encoder()

    # Test encode_sequence (returns 3 values: h_seq, latent, return_preds)
    h_seq, latent, ret_preds = model.encode_sequence(obs, asset)
    print(f"Hidden shape: {h_seq.shape}, Latent shape: {latent.shape}")
    print(f"Return pred shapes: { {h: v.shape for h, v in ret_preds.items()} }")

    # Memory estimate
    param_mb = total_params * 4 / 1024 / 1024
    print(f"\nParameter memory: {param_mb:.1f} MB")
    print(f"InfoNCE memory per step: {B * B * 4 * T / 1024 / 1024:.2f} MB (per-timestep)")

    print("\nV6 Causal JEPA+Adversarial world model sanity check passed.")
