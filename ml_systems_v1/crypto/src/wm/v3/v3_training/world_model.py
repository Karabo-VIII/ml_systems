"""
V3 World Model — WaveNet-GRU Hybrid with RSSM (SOTA 2025/26)

Architecture:
  1. Obs Encoder: Linear(13+32 -> 96) + RMSNorm + SiLU
  2. WaveNet TCN: 4 gated dilated causal conv layers, dilations [1,2,4,8]
  3. MultiScaleAggregator: Combine skip connections from all 4 scales
  4. CausalGRU: 2-layer GRU for sequential dynamics
  5. RSSM Latents: Prior/Posterior (24x24 categorical)
  6. Heads: Reconstruction, Multi-Horizon Returns, Regime

Key differences from V1/V2:
  - WaveNet gated activations (tanh * sigmoid) instead of ReLU convolutions
  - GRU replaces LSTM (fewer params, same quality)
  - Multi-scale skip aggregation for richer temporal features
  - Right-sized for ~2,100 training sequences (smaller channels/latents)
  - Proper weight initialization
  - dream_step() for agent dreaming

SOTA 2025/26:
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
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
    from v3_training.components import (
        RMSNorm,
        WaveNetTCN,
        MultiScaleAggregator,
        CausalGRU,
        TwoHotSymlog,
        SwiGLU,
        MLPHead,
    )
    from v3_training.settings import *

# ==============================================================================
# VARIABLE SELECTION NETWORK (VSN) -- shared lever (src/wm/_shared/variable_selection.py)
# Wired behind V3_VSN env flag (default "0" = OFF). When OFF: module not constructed,
# base path byte-for-byte unchanged. When ON: per-timestep sigmoid gate before obs_encoder.
# Mirrors V1.1 wiring exactly (same flag idiom; same _init_weights guard for gate_proj).
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


class WaveNetGRUWorldModel(nn.Module):
    """V3: WaveNet-GRU Hybrid World Model with RSSM latents."""

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
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
        self.latent_dim = latent_dim
        self.classes = classes
        self.flat_dim = latent_dim * classes  # 576
        # d_model alias: shared lever attach_forward_regime_head uses model.d_model + model.flat_dim
        # to compute feat_dim; V3's equivalent is gru_hidden (h_seq dim = 256 in both skip_gru paths).
        self.d_model = gru_hidden

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
        # 4. Causal GRU for sequential dynamics
        # =====================================================================
        self.gru = CausalGRU(
            input_dim=tcn_channels[-1],
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout,
        )
        # V3 UPGRADE: bypass GRU by default. WaveNet already captures
        # multi-scale temporal dependencies; GRU is redundant and adds
        # memorization risk. Set to False to re-enable GRU for ablation.
        self.skip_gru = True

        # =====================================================================
        # 5. RSSM Latent Heads
        # =====================================================================
        self.prior_head = MLPHead(gru_hidden, 256, self.flat_dim, dropout)
        self.posterior_head = MLPHead(
            gru_hidden + input_dim, 256, self.flat_dim, dropout
        )

        # =====================================================================
        # 6. Output Heads
        # =====================================================================
        head_input_dim = gru_hidden + self.flat_dim

        # Reconstruction head
        self.decoder = nn.Sequential(
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

        # Regime classification head (3 classes: bearish/neutral/bullish)
        self.regime_head = MLPHead(head_input_dim, REGIME_HEAD_DIM, 3, dropout)

        # -- VSN (Variable Selection Network, V3_VSN flag) ---------------------
        # Flag-gated: constructed ONLY when env var V3_VSN="1" at model init.
        # When OFF (default): module not present -> forward path byte-for-byte unchanged.
        # When ON: VariableSelectionNetwork gates [B,T,input_dim] BEFORE obs_encoder.
        # Combinable with V3_FORWARD_REGIME (both ON = full world-class candidate).
        import os as _os_vsn
        self._use_vsn = _os_vsn.environ.get("V3_VSN", "0") == "1"
        if self._use_vsn:
            self.vsn = VariableSelectionNetwork(input_dim)
        else:
            self.vsn = None  # not constructed; no parameters, no side effects

        # -- Forward-regime head (V3_FORWARD_REGIME flag) ----------------------
        # Default OFF: _use_forward_regime=False, head=None -> no output key,
        # no parameters, no loss contribution. Wired in forward_train + get_loss
        # via guarded getattr checks (identical pattern to V1.1).
        # Attach via attach_forward_regime_head (src/wm/_shared/forward_regime_head.py)
        # or set V3_FORWARD_REGIME=1 in the trainer after model construction.
        self._use_forward_regime = False
        self.forward_regime_head = None

        # =====================================================================
        # Loss balancing (uncertainty-weighted)
        # =====================================================================
        self.log_vars = nn.Parameter(torch.tensor(LOG_VAR_INIT, dtype=torch.float32))

        # Dream step projection (projects combined state to GRU input dim)
        self.dream_proj = nn.Linear(head_input_dim, tcn_channels[-1])

        # Forecast heads (2026-05-10 generalization fix; same pattern as V4/V8/V22/V25)
        # Predict obs[t+h] from h_seq[t] (WaveNet+GRU state). Anchors h_seq to
        # feature-faithful future prediction. V4 probe-validated: +88% train_IC.
        try:
            from settings import USE_FORECAST_HEAD as _use_fc
        except ImportError:
            _use_fc = False
        if _use_fc:
            self.forecast_heads = nn.ModuleDict({
                str(h): nn.Linear(gru_hidden, input_dim) for h in REWARD_HORIZONS
            })
        else:
            self.forecast_heads = None

        # CC-H5 quantile heads (SOTA-2026, auxiliary — legacy TwoHot heads
        # remain primary). Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H5: explicit
        # q05/q10/q25/q50/q75/q90/q95 per horizon for risk-aware sizing in
        # downstream meta-learner. Opt-in via USE_QUANTILE_HEADS flag (settings).
        try:
            from settings import USE_QUANTILE_HEADS as _use_qh
        except ImportError:
            _use_qh = False
        # Resolve _shared path once (also used for CC-H6).
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

        # CC-H6 regime-conditional heads (SOTA-2026, auxiliary): per-regime
        # decoders (bear/neutral/bull). At training, per-sample CE uses the
        # head matching that sample's regime label. At inference, soft-blend
        # by predicted regime probabilities. Adds Sharpe stability across
        # regime shifts (per plan §0, +0.003-0.008 IC but +0.05 Sharpe in
        # regime-shift windows). Opt-in via USE_REGIME_COND_HEADS.
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

        # Regime-awareness encoder conditioning (SOTA-2026 +1 over CC-H6).
        # "film" mode adds a RegimeFiLM modulator on h_seq before heads.
        # Identity-at-init (zero scale/shift) so safe to add to a planned
        # cold-start without disrupting early-training dynamics.
        try:
            from settings import REGIME_AWARENESS_MODE as _ram
        except ImportError:
            _ram = "heads"
        self._regime_awareness_mode = _ram
        if _ram == "film":
            from headline_components import RegimeFiLM as _RF
            self.regime_film = _RF(d_model=gru_hidden, n_regimes=3)
            # h_seq-only regime gate that drives FiLM. Cannot reuse the
            # main regime_head (which reads feat = [h_seq, z_post]) because
            # z_post hasn't been computed at the FiLM site (z_post depends
            # on h_seq, so we'd have a chicken-and-egg). This small gate
            # is auxiliary; identity-at-init FiLM means early training is
            # undisrupted. The gate gets its training signal via gradient
            # flow through the modulated h_seq into downstream heads.
            self.regime_film_gate = nn.Linear(gru_hidden, 3)
            # Init the gate to ~uniform softmax (zero weight + bias)
            nn.init.zeros_(self.regime_film_gate.weight)
            nn.init.zeros_(self.regime_film_gate.bias)
        else:
            self.regime_film = None
            self.regime_film_gate = None

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
        """Initialize weights with appropriate strategies per layer type.

        Phase 14.7 fix: differentiate Conv1d init by role.
          - WaveNet filter/gate convs use tanh/sigmoid gating → xavier (gain 1.0)
          - WaveNet residual/skip 1x1 projections use small init (std=0.02)
            so residual additions don't compound across N blocks.
          - Other Conv1d (input_proj, channel_projs) use kaiming-relu.
        Pre-fix: all Conv1d used kaiming-relu, producing 6.7e12 grad_max in
        B=32 audit probe.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # VSN gate_proj has neutral-start init set in VariableSelectionNetwork.__init__
                # (std=0.01, bias=0 -> sigmoid~0.5). Must NOT be overwritten here.
                if "gate_proj" in name:
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                # Role-based init for Conv1d
                if "conv_filter" in name or "conv_gate" in name:
                    # Gated tanh/sigmoid path — xavier matches gain 1.0
                    nn.init.xavier_uniform_(module.weight)
                elif "residual" in name or "skip" in name:
                    # 1x1 projections — small init prevents residual amplification
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    # input_proj, channel_projs, aggregator convs — kaiming-relu
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

        # VSN: per-timestep causal feature gate applied BEFORE obs_encoder.
        # When V3_VSN=0 (default): self.vsn is None, branch skipped, path identical to base.
        # When V3_VSN=1: input_obs is replaced with gate * input_obs (same shape).
        # Gate is causal -- sigmoid(W * x_t) uses only x_t, no future timesteps.
        if self._use_vsn and self.vsn is not None:
            input_obs = self.vsn(input_obs)

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
        # Run WaveNet + aggregator in fp32 — the 4-layer residual+skip chain
        # accumulates gradient magnitudes that overflow fp16 backward by epoch 21,
        # causing GN:inf and freezing the model. Disabling autocast here forces
        # both forward AND backward through WaveNet to use fp32.
        # The RSSM, return heads, and loss stay in fp16 for speed.
        # Ref: PyTorch AMP docs recommend autocast(enabled=False) for layers with
        # known fp16 gradient instability (https://docs.pytorch.org/docs/stable/amp.html)
        with torch.amp.autocast("cuda", enabled=False):
            obs_emb_fp32 = obs_emb_shifted.float()
            tcn_out, skips = self.wavenet(obs_emb_fp32)
            agg_out = self.aggregator(skips)  # [B, T, 256] in fp32

        # 5. GRU for sequential dynamics (bypassed by default -- WaveNet is sufficient)
        if self.skip_gru:
            h_seq = agg_out  # Direct WaveNet output to RSSM
        else:
            h_seq, _ = self.gru(agg_out)  # [B, T, gru_hidden]

        # Regime-FiLM modulator (SOTA-2026 opt-in): conditions h_seq on
        # regime BEFORE posterior + heads. Identity-at-init (zero scale +
        # zero shift) so safe to add without disrupting early-training
        # dynamics. Regime probs come from a small h_seq-only gate (cannot
        # reuse the main regime_head which depends on z_post).
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)   # [B, T, 3]
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # 6. RSSM: Prior and Posterior
        # Posterior reads `input_obs` (the MASKED observations), NOT raw `obs_seq`.
        # Using unmasked obs_seq lets the posterior trivially copy missing values
        # back through z_post, defeating the block-masking pretext objective.
        # Fixed 2026-05-21 per pre-retrain RED-team audit.
        prior_logits = self.prior_head(h_seq)
        post_input = torch.cat([h_seq, input_obs], dim=-1)
        post_logits = self.posterior_head(post_input)
        z_post = self._get_stoch_state(post_logits)

        # 7. Decode from combined features
        feat = torch.cat([h_seq, z_post], dim=-1)  # [B, T, gru_hidden + flat_dim]

        recon = self.decoder(feat)  # reconstruction always uses full temporal context

        # ATME: anti-temporal-memorization via obs-only posterior. SOTA-2026:
        # per-sample mask (was batch-level) — each sample independently rolled
        # so a single batch contains both ATME and normal paths. Per CLAUDE.md
        # canonical TEMPORAL_CTX_DROP=0.15 for RSSM-class. Cost: +1 posterior
        # forward pass (need both paths to mix per-sample).
        if self.training and temporal_ctx_drop > 0:
            B = h_seq.shape[0]
            # Per-sample dice: True = use ATME (obs-only) path for this sample
            sample_mask = (torch.rand(B, 1, 1, device=h_seq.device)
                            < temporal_ctx_drop).float()
            # ATME path: obs-only posterior blocks temporal leakage through z_post.
            # FIX 2026-05-29: use input_obs (the MASKED obs) not raw obs_seq -- the
            # block-masking pretext was defeated by feeding unmasked obs here.
            post_input_obs = torch.cat([torch.zeros_like(h_seq), input_obs], dim=-1)
            post_logits_obs = self.posterior_head(post_input_obs)
            z_post_obs = self._get_stoch_state(post_logits_obs)
            feat_heads_atme = torch.cat([torch.zeros_like(h_seq), z_post_obs], dim=-1)
            # Normal path: h_seq.detach() — GRU learns from recon/KL only, return
            # heads READ temporal features but can't OPTIMIZE GRU for memorization
            feat_heads_norm = torch.cat([h_seq.detach(), z_post], dim=-1)
            # Mix per-sample (broadcast [B,1,1] over [B,T,D])
            feat_heads = (sample_mask * feat_heads_atme
                            + (1.0 - sample_mask) * feat_heads_norm)
        else:
            # Inference / temporal_ctx_drop=0: pure normal path.
            feat_heads = torch.cat([h_seq.detach(), z_post], dim=-1)

        # CC-H3 cross-asset hook: no-op until MultiAssetDataset is wired
        # (single-asset training has N_assets=1 per batch, so the head's
        # cross-asset attention is a self-attention on a 1-token sequence
        # = identity). When the dataloader provides synchronized N-asset
        # batches, this hook lifts h_seq[t] across the asset dimension.
        if hasattr(self, "_cross_asset_head"):
            feat_heads = self._cross_asset_head(feat_heads)

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

        # Forecast heads (added 2026-05-10) -- predict obs[t+h] from h_seq[t]
        forecast_logits = None
        if self.forecast_heads is not None:
            forecast_logits = {
                int(h): self.forecast_heads[str(h)](h_seq) for h in REWARD_HORIZONS
            }

        # CC-H5 quantile heads (SOTA-2026): per-horizon distributional output.
        # Auxiliary to TwoHot; consumed by strategy-side meta-learner.
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat_heads)

        # CC-H6 regime-conditional heads (SOTA-2026): all-regime outputs
        # for training (loss picks per-sample regime) + soft-blend at inference.
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat_heads)

        out = {
            "recon": recon,
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "prior_logits": prior_logits,
            "post_logits": post_logits,
            "h_seq": h_seq,
            "z_post": z_post,
            "ret_trunk": ret_trunk_out,
            "forecast_logits": forecast_logits,
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }

        # Forward-regime head (V3_FORWARD_REGIME flag, 2026-06-10).
        # OFF by default: _use_forward_regime=False -> block is no-op, key absent from out,
        # base path byte-for-byte unchanged.
        # ON (after attach_forward_regime_head is called in trainer): reads feat_heads
        # (= cat(h_seq, z_post) post-ATME, same tensor all other heads read), emits
        # {bear_logits, trend_logits, move_logits}. Input is feat_heads not feat (ATME
        # path is active during training; the head benefits from the same anti-memorization).
        # No look-ahead: head reads the ENCODED feature; labels are built at TARGET-CONSTRUCTION
        # time in regime_targets.py and packed into targets["forward_regime_labels"] by collate.
        if getattr(self, "_use_forward_regime", False) and getattr(self, "forward_regime_head", None) is not None:
            out["forward_regime"] = self.forward_regime_head(feat_heads)

        return out

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
        return_components: bool = False,
    ):
        """
        Compute training loss with block masking and multi-horizon targets.

        Args:
            return_components: when True, returns 4-tuple (total, loss_dict,
                outputs, components). components dict has per-task tensors
                {aux, ret_1, ret_4, ret_16, ret_64} for PCGrad.

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

        # Reconstruction loss
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)

        # KL Divergence (categorical RSSM)
        # nan_to_num: sporadic NaN from WaveNet fp16 path must not crash D.Categorical
        prior = outputs["prior_logits"].float().view(-1, self.latent_dim, self.classes)
        post = outputs["post_logits"].float().view(-1, self.latent_dim, self.classes)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=10.0, neginf=-10.0)
        post = torch.nan_to_num(post, nan=0.0, posinf=10.0, neginf=-10.0)

        l_kl = D.kl_divergence(
            D.Categorical(logits=post),
            D.Categorical(logits=prior),
        ).mean()
        kl_raw = l_kl.item()
        l_kl = torch.max(l_kl, torch.tensor(WM_FREE_NATS, device=obs_seq.device))

        # Multi-Horizon Return Losses (MDN-aware)
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

        # Uncertainty-weighted total (decomposed for PCGrad)
        s = self.log_vars.clamp(-6.0, 6.0)
        s_rec = s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN)
        rec_term = torch.exp(-s_rec) * l_rec + 0.5 * s_rec
        kl_term = torch.exp(-s[1]) * l_kl * kl_anneal + 0.5 * s[1]

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
                l_direct_return = l_direct_return + F.huber_loss(
                    decoded.reshape(-1), target_returns[h].reshape(-1)
                )
        direct_term = DIRECT_RETURN_WEIGHT * l_direct_return

        # Forecast head loss (2026-05-10 generalization fix)
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
                fc_pred = outputs["forecast_logits"][h][:, :-h, :]
                fc_tgt = obs_seq[:, h:, :]
                l_forecast = l_forecast + F.mse_loss(fc_pred, fc_tgt)
            l_forecast = l_forecast / len(REWARD_HORIZONS)
        forecast_term = _fc_w * l_forecast

        # CC-H5 quantile loss (SOTA-2026, auxiliary): pinball regression
        # on q05..q95 per horizon. Imported from _shared/headline_components.
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
                    q_pred = outputs["quantile_logits"][h]   # [B, T, Q]
                    q_tgt = target_returns[h]                 # [B, T]
                    l_quantile = l_quantile + _ql(q_pred, q_tgt, quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        quantile_term = _ql_w * l_quantile

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary): per-sample CE
        # using the head matching that sample's regime label.
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

        # Forward-regime aux loss (V3_FORWARD_REGIME flag, 2026-06-10).
        # Guarded: only fires when head is attached (_use_forward_regime=True) AND
        # trainer supplied forward_regime_labels. Default OFF -> total unchanged.
        # Weight 0.10 (small fixed, additive; head is auxiliary supervision).
        # Labels built by src/wm/_shared/regime_targets.py (forward-only, NaN tail masked).
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
        forward_regime_term = 0.10 * l_forward_regime

        aux_term = (rec_term + kl_term + regime_term + direct_term
                     + forecast_term + quantile_term + regime_cond_term
                     + forward_regime_term)
        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

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
            "forward_regime": l_forward_regime.item(),
        }
        for h in ACTIVE_HORIZONS:
            loss_dict[f"ret_{h}"] = horizon_losses[h].item()

        if return_components:
            components = {"aux": aux_term, **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS}}
            return total, loss_dict, outputs, components
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
        if getattr(self, "_use_mdn", False):
            ret_trunk = outputs.get("ret_trunk")
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk)
        else:
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
        if getattr(self, "_use_mdn", False):
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.return_heads[str(h)].expectation(ret_trunk_out)
        elif self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk_out)
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.bucketer.decode(mtp_out[f"h{h}"])
        else:
            for h in ACTIVE_HORIZONS:
                return_preds[h] = self.bucketer.decode(
                    self.return_heads[str(h)](ret_trunk_out)
                )

        return h_next, z_next, gru_hidden, return_preds


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = WaveNetGRUWorldModel().to(DEVICE)
    print(f"V3 WaveNet-GRU World Model Parameters: {count_parameters(model):,}")

    # Test forward pass
    B, T = 4, WM_SEQ_LEN
    obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
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

    print("V3 WaveNet-GRU world model sanity check passed.")
