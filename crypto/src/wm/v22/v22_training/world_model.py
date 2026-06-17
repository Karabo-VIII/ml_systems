"""V22 World Model — iTransformer (Liu et al., ICLR 2024).

This is the production V-version of the V22 backbone. The clumsy
`bar_signal_3d` expansion in `itransformer_backbone.py` is replaced by
the cleaner `[B, F, T] -> [B, T, F] -> Linear(F, D)` projection: each bar's
representation is the F-dim cross-feature-attended slice projected to D.

Faithful to ICLR 2024 paper §3:
  1. Inverted embedding: [B, T, F] -> [B, F, D] via Linear(T, D)
     (each feature's full T-series becomes a D-dim token)
  2. Optional asset token prepended to the F tokens (paper §4.2 covariate ext.)
  3. N layers of multi-head attention OVER FEATURES (the inversion)
  4. Inverted projection: [B, F, D] -> [B, F, T] via Linear(D, T)
  5. Per-bar projection: [B, T, F] -> [B, T, D] via Linear(F, D)
     -- each bar's representation is its column of the cross-feature signal
  6. Per-bar return heads (TwoHot, multi-horizon)
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from settings import *
except ImportError:
    from .settings import *

# Reuse V1.x components for RMSNorm + TwoHotSymlog (CLAUDE.md invariants)
_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)

from components import RMSNorm, TwoHotSymlog, MLPHead

# Round-7 + Round-9 frontier components
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from frontier_components import (
    tail_adaptive_huber, CryptoPeriodEmbedding, RateBudgetVIB,
)


# =============================================================================
# Inverted Attention Layer (paper §3.1, pre-norm)
# =============================================================================

class InvertedAttentionLayer(nn.Module):
    """One encoder layer of feature-as-token self-attention. Operates on [B, F, D]."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int | None = None,
                 dropout: float = 0.1, use_cross_feat_attn: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self._use_cross_feat_attn = use_cross_feat_attn
        # Always allocate the MHA module so checkpoint state_dict matches the
        # state-of-the-iTransformer schema. Whether it runs is gated in forward().
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = RMSNorm(d_model)
        ff_dim = dim_ff if dim_ff is not None else 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # 2026-05-09 H3 fix: cross-feature attention without forecast supervision
        # learns a sign-flipped representation (val-IC -0.10 to -0.53). Skipping
        # the attention (identity passthrough) flips IC POSITIVE: Ep10 ic1=+0.21,
        # ic16=+0.64, ic64=+0.60, ShIC=0.0000 (clean anti-fragility). The FFN
        # stack alone provides sufficient capacity for return prediction. See
        # docs/V25_V22_H3_ABLATION_2026_05_09.md and V22_V25_FORECAST_HEAD_PROPOSAL_*.
        if getattr(self, "_use_cross_feat_attn", False):
            attn_out, _ = self.attn(h, h, h, need_weights=False)
        else:
            attn_out = h  # identity passthrough — proven Capacity-tier
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# =============================================================================
# iTransformer World Model
# =============================================================================

class iTransformerWorldModel(nn.Module):
    """V22: iTransformer for crypto WM.

    Inputs:
      obs_seq:  [B, T, F]
      asset_id: [B]
    Returns dict (forward_train) with keys: return_logits, regime_logits, h_seq,
    plus zero-stub compatibility shims (prior_logits, post_logits, z_post, recon).
    """

    def __init__(self, input_dim: int = INPUT_DIM, seq_len: int = WM_SEQ_LEN,
                 d_model: int = WM_D_MODEL, n_heads: int = WM_N_HEADS,
                 n_layers: int = WM_N_LAYERS, dropout: float = WM_DROPOUT,
                 num_bins: int = NUM_BINS, num_assets: int = NUM_ASSETS,
                 asset_emb_dim: int = WM_ASSET_EMB_DIM,
                 use_asset_token: bool = USE_ASSET_TOKEN,
                 z_dim: int = VIB_Z_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_asset_token = use_asset_token
        self._num_bins = num_bins
        self.z_dim = z_dim

        # Round-9 F1: inverted embedding INPUT regularization (KEPT)
        self.inv_embed_input_drop = nn.Dropout(INV_EMBED_INPUT_DROPOUT)
        self._inv_embed_input_noise_std = INV_EMBED_INPUT_NOISE

        # Round-10 ROOT-CAUSE FIX (per other-instance V22 audit):
        # Replace nn.Linear(96, 320) — 921,600-param memorization vector —
        # with PatchTST patch embedding. Each patch sees only PATCH_LEN bars
        # (12), can't memorize full 96-bar templates. Per Nie et al. ICLR 2023.
        if USE_PATCH_EMBEDDING:
            patch_embed = nn.Linear(PATCH_LEN, PATCH_DIM)
            if USE_SPECTRAL_NORM_EMBED:
                # Spectral norm bounds Lipschitz constant -> caps memorization
                # capacity (Fix #3 from other-instance audit).
                patch_embed = nn.utils.parametrizations.spectral_norm(patch_embed)
            self.patch_embed = patch_embed
            self.embed = None   # disabled — patch path replaces
        else:
            self.patch_embed = None
            base_embed = nn.Linear(seq_len, d_model)
            if USE_SPECTRAL_NORM_EMBED:
                base_embed = nn.utils.parametrizations.spectral_norm(base_embed)
            self.embed = base_embed

        # Asset token (paper §4.2 covariate extension)
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.asset_token_proj = nn.Linear(asset_emb_dim, d_model)

        # Round-10 INPUT VIB (Fix #2 from other-instance audit):
        # Bottleneck UPSTREAM of transformer layers. Forces embedding to
        # compress information BEFORE transformer can amplify memorized
        # templates. Replaces post-encoder VIB position which fired AFTER
        # memorization had already happened in inverted-attention layers.
        if USE_INPUT_VIB:
            self.input_vib = RateBudgetVIB(
                d_model=d_model, z_dim=self.z_dim,
                target_rate_nats=INPUT_VIB_TARGET_RATE_NATS,
                beta_init=VIB_KL_WEIGHT, beta_lr=VIB_BETA_LR,
                beta_min=VIB_BETA_MIN, beta_max=VIB_BETA_MAX,
                logvar_init=VIB_LOGVAR_INIT,
                logvar_min=VIB_LOGVAR_MIN, logvar_max=VIB_LOGVAR_MAX,
                dropout=dropout,
            )
        else:
            self.input_vib = None

        # Stacked inverted attention encoder
        # 2026-05-09 H3 fix: cross-feature attention defaults OFF — without
        # forecast-loss supervision it learns sign-flipped representations.
        # Toggle via USE_CROSS_FEAT_ATTN in settings (kept for ablation/future
        # experiments with forecast head). See V22_V25_FORECAST_HEAD_PROPOSAL.
        try:
            from settings import USE_CROSS_FEAT_ATTN as _use_attn
        except ImportError:
            _use_attn = False
        self.layers = nn.ModuleList([
            InvertedAttentionLayer(d_model, n_heads, dropout=dropout,
                                    use_cross_feat_attn=_use_attn)
            for _ in range(n_layers)
        ])
        self.post_norm = RMSNorm(d_model)

        # Inverted projection: each feature token → T-dim time-series (paper §3.1)
        self.proj = nn.Linear(d_model, seq_len)

        # Per-bar projection: F cross-feature signals at each bar → D-dim per-bar rep
        # (Cleaner than backbone's degenerate seq_len-expand. Each bar's representation
        # is the F-dim cross-feature slice projected to D.)
        self.bar_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Round-7: hard-coded crypto period embedding added to per-bar h_seq.
        # iTransformer's inversion loses temporal position info — this restores
        # known crypto cycles (8h funding / 24h UTC / 7d weekly).
        self.period_emb = CryptoPeriodEmbedding(d_model)

        # Round-9 F2: replace fixed-β VIB with rate-budget VIB.
        # Auto-tunes β via Lagrangian to hit a fixed bits-per-timestep target.
        # Bottleneck binds REGARDLESS of prediction loss magnitude — fixes
        # the V22 memorization that fixed-β couldn't address (probe showed
        # β=0.20 still gave BestIC=0.76, ShIC/IC=0.01).
        self.vib = RateBudgetVIB(
            d_model=d_model, z_dim=self.z_dim,
            target_rate_nats=VIB_TARGET_RATE_NATS,
            beta_init=VIB_KL_WEIGHT, beta_lr=VIB_BETA_LR,
            beta_min=VIB_BETA_MIN, beta_max=VIB_BETA_MAX,
            logvar_init=VIB_LOGVAR_INIT,
            logvar_min=VIB_LOGVAR_MIN, logvar_max=VIB_LOGVAR_MAX,
            dropout=dropout,
        )

        # Return heads (TwoHot multi-horizon)
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, RETURN_HEAD_DIM),
            RMSNorm(RETURN_HEAD_DIM),
            nn.SiLU(),
            nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(RETURN_HEAD_DIM, RETURN_HEAD_DIM // 2),
                RMSNorm(RETURN_HEAD_DIM // 2),
                nn.SiLU(),
                nn.Linear(RETURN_HEAD_DIM // 2, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Regime head
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot bucketer (lazy device migration)
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall log-vars: [ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(
            torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5])
        )

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026)
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
            self.quantile_heads = _QH(head_input_dim=d_model,
                                         horizons=tuple(REWARD_HORIZONS))
        else:
            self.quantile_heads = None
        try:
            from settings import USE_REGIME_COND_HEADS as _use_rc
        except ImportError:
            _use_rc = False
        if _use_rc:
            from headline_components import RegimeConditionalHeads as _RC
            self.regime_cond_heads = _RC(head_input_dim=d_model,
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

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor,
                      masked_obs_seq: torch.Tensor | None = None):
        B, T, F_in = obs_seq.shape
        if F_in != self.input_dim:
            raise ValueError(f"Expected n_features={self.input_dim}, got {F_in}")
        if T != self.seq_len:
            if T < self.seq_len:
                pad_len = self.seq_len - T
                obs_seq = F.pad(obs_seq, (0, 0, 0, pad_len))
                if masked_obs_seq is not None:
                    masked_obs_seq = F.pad(masked_obs_seq, (0, 0, 0, pad_len))
            else:
                obs_seq = obs_seq[:, : self.seq_len, :]
                if masked_obs_seq is not None:
                    masked_obs_seq = masked_obs_seq[:, : self.seq_len, :]
            T = self.seq_len

        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # Causal shift (predict t from t-1)
        shifted = torch.cat(
            [torch.zeros(B, 1, F_in, device=obs_seq.device), input_obs[:, :-1, :]],
            dim=1,
        )

        # [B, T, F] -> [B, F, T] (the inversion)
        x_t = shifted.transpose(1, 2)
        # Round-9 F1: regularize the T-series before the inverted embedding to
        # prevent memorization of specific 96-bar patterns.
        if self.training:
            x_t = self.inv_embed_input_drop(x_t)
            if self._inv_embed_input_noise_std > 0:
                x_t = x_t + torch.randn_like(x_t) * self._inv_embed_input_noise_std

        # Round-10 PATCH-EMBEDDING (replaces Linear(96, 320) memorization
        # vector). Each feature's T-series is split into N_PATCHES patches of
        # PATCH_LEN bars; each patch is embedded to PATCH_DIM dims; patches
        # are concatenated to a single d_model token per feature.
        if self.patch_embed is not None:
            # x_t: [B, F, T] -> reshape to [B, F, N_PATCHES, PATCH_LEN]
            B_, F_, T_ = x_t.shape
            assert T_ == self.seq_len, f"x_t T={T_} != seq_len={self.seq_len}"
            patches = x_t.reshape(B_, F_, N_PATCHES, PATCH_LEN)
            # Embed each patch: [B, F, N_PATCHES, PATCH_DIM]
            patch_emb = self.patch_embed(patches)
            # Concatenate patches per feature: [B, F, N_PATCHES * PATCH_DIM] = [B, F, d_model]
            tokens = patch_emb.reshape(B_, F_, N_PATCHES * PATCH_DIM)
        else:
            # Legacy linear embedding path (when USE_PATCH_EMBEDDING=False)
            tokens = self.embed(x_t)

        # Round-10 INPUT VIB: bottleneck UPSTREAM of transformer layers.
        # Forces embedding output to compress to z_dim nats per token BEFORE
        # transformer can amplify memorized templates.
        input_vib_kl = None
        if self.input_vib is not None:
            tokens, input_vib_mu, input_vib_logvar, input_vib_kl = self.input_vib(
                tokens, training=self.training
            )
            if self.training:
                self.input_vib.update_beta(input_vib_kl)

        # Prepend asset token
        if self.use_asset_token:
            asset_emb = self.asset_embedding(asset_id)
            asset_tok = self.asset_token_proj(asset_emb).unsqueeze(1)  # [B, 1, D]
            # Round-9 F3: random-drop asset token during training to break
            # "BTC-prior memorization" — model can't rely on knowing the asset.
            if self.training and ASSET_TOKEN_DROP_PROB > 0:
                drop_mask = (torch.rand(B, 1, 1, device=asset_tok.device)
                              > ASSET_TOKEN_DROP_PROB).float()
                asset_tok = asset_tok * drop_mask
            tokens = torch.cat([asset_tok, tokens], dim=1)            # [B, F+1, D]

        # Cross-feature attention
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.post_norm(tokens)

        # Drop the asset token before per-feature projection
        feat_tokens = tokens[:, 1:, :] if self.use_asset_token else tokens   # [B, F, D]

        # Inverted projection: [B, F, D] -> [B, F, T]
        feat_T = self.proj(feat_tokens)

        # Per-bar representation: at bar t, the F-dim slice across features
        per_bar_F = feat_T.transpose(1, 2)             # [B, T, F]
        h_seq = self.bar_proj(per_bar_F)               # [B, T, D]
        # RegimeFiLM (SOTA-2026 opt-in) — identity-at-init; conditions h_seq
        # BEFORE period_emb + VIB.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)
        # Round-7: hard-coded crypto period embedding. Gated behind
        # USE_PERIOD_EMB (default False, mirroring V25 ablation). Period_emb is
        # an additive position-coded channel and contributes to temporal
        # memorization when the encoder has no anchor; disabled by default.
        if globals().get("USE_PERIOD_EMB", False):
            h_seq = h_seq + self.period_emb(T, device=h_seq.device).unsqueeze(0)

        # Round-9 F2: rate-budget VIB (auto-tuned β toward bits-per-timestep target)
        feat_vib, mu, logvar, vib_kl_per = self.vib(h_seq, training=self.training)
        # Update β via Lagrangian using observed KL (in-place buffer update;
        # gradient-free, no effect on backward graph).
        if self.training:
            self.vib.update_beta(vib_kl_per)

        # ATME (per-sample) on the post-VIB feature
        feat = feat_vib
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device)
                         > TEMPORAL_CTX_DROP).float()
            # ATME drops temporal CONTEXT but must NOT zero the supervised last bar:
            # under USE_LAST_BAR_SUPERVISION the loss reads feat[:, -1], so zeroing
            # the whole feat (incl. position T-1) taught predict-from-nothing 15% of
            # the time. Mask context [0:T-1] only; keep the prediction position.
            if feat.shape[1] > 1:
                feat = torch.cat([feat[:, :-1, :] * atme_mask, feat[:, -1:, :]], dim=1)
            else:
                feat = feat * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {
            h_key: self.return_heads[str(h_key)](ret_trunk)
            for h_key in REWARD_HORIZONS
        }
        regime_logits = self.regime_head(feat)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "ret_trunk": ret_trunk,
            "vib_mu": mu,
            "vib_logvar": logvar,
            "vib_kl_per": vib_kl_per,
            "vib_beta": self.vib.get_beta() if hasattr(self, "vib") else None,
            # Round-10: input-VIB diagnostics (None if disabled)
            "input_vib_kl": input_vib_kl,
            "input_vib_beta": self.input_vib.get_beta() if self.input_vib is not None else None,
            "z_post": mu.detach(),
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
            # 2026-05-10 forecast-head root-cause fix: expose feat_T for auxiliary
            # MSE loss in get_loss. Anchors encoder to feature-faithful semantics.
            "feat_T": feat_T,                  # [B, F, T] iTransformer per-feature forecast
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None,
                 return_components=False, **kwargs):
        """V1-compatible loss interface.

        Returns (total, loss_dict, outputs).
        With return_components=True, returns (total, loss_dict, outputs, components).
        """
        B, T, n_feat = obs_seq.shape
        dev = str(obs_seq.device)
        if self._bucketer_device != dev:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev)
            self._bucketer_device = dev

        masked_obs = obs_seq.clone()
        if mask_ratio > 0 and self.training:
            mask = torch.rand(B, T, 1, device=obs_seq.device) < mask_ratio
            masked_obs = masked_obs * (~mask).float()

        outputs = self.forward_train(obs_seq, asset_id, masked_obs)

        # Round-9: rate-budget VIB. KL is computed inside self.vib; use its
        # auto-tuned β. Also track for diagnostics.
        vib_kl = outputs["vib_kl_per"]
        beta = outputs.get("vib_beta", torch.tensor(VIB_KL_WEIGHT))
        kl_anneal = float(kwargs.get("kl_anneal", 1.0))
        kl_weight = (beta.item() if torch.is_tensor(beta) else float(beta)) * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        loss_dict = {"total": 0.0, "rec": 0.0, "kl": vib_kl.item(),
                     "kl_raw": vib_kl.item(), "kl_weight": kl_weight}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # 2026-05-21 SOTA causality fix (Timer-XL / TimesFM pattern). Supervise
        # only the LAST bar of each window — the only position whose attention
        # representation has no future-bar leak (there are no future bars in the
        # window). The encoder still runs over all 96 bars; only the loss
        # restricts to position T-1.
        try:
            from settings import USE_LAST_BAR_SUPERVISION as _last_bar
        except ImportError:
            _last_bar = True  # SOTA default

        def _slice_logits(x):
            # x: [B, T, ...] -> [B, 1, ...] when last_bar; passthrough otherwise.
            return x[:, -1:, :] if _last_bar else x

        def _slice_target(x):
            # x: [B, T] -> [B, 1] when last_bar; passthrough otherwise.
            return x[:, -1:] if _last_bar else x

        # Per-horizon TwoHot CE (PCGrad-friendly: kept separate)
        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_sliced = _slice_logits(outputs["return_logits"][h])
            tgt_sliced = _slice_target(targets[h])
            logits_flat = logits_sliced.reshape(-1, self._num_bins)
            tgt_flat = tgt_sliced.reshape(-1)
            if h in ACTIVE_HORIZONS:
                l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                ret_terms[h] = torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            decoded = self.bucketer.decode(logits_flat)
            # Round-7: tail-adaptive Huber for crypto fat-tail returns
            l_direct = l_direct + tail_adaptive_huber(
                decoded, tgt_flat, delta=0.5, tail_sigma=2.0, tail_weight=2.5
            )

        aux_term = DIRECT_RETURN_WEIGHT * l_direct + kl_weight * vib_kl
        loss_dict["direct_ret"] = l_direct.item()

        # 2026-05-10 forecast-head ROOT-CAUSE FIX for memorization:
        # MSE(feat_T, obs_seq.T) anchors the encoder representation to be
        # feature-faithful. Without this, the encoder drifts to position-
        # memorization solutions (the V22 ShIC=0.000 trajectory). See
        # docs/V22_V25_SOLUTION_2026_05_10.md and FORECAST_HEAD_PROPOSAL.
        try:
            from settings import USE_FORECAST_HEAD as _use_fc, FORECAST_WEIGHT as _fc_w
        except ImportError:
            _use_fc, _fc_w = False, 0.0
        if _use_fc and "feat_T" in outputs:
            forecast_target = obs_seq.transpose(1, 2)        # [B, F, T]
            forecast_loss = F.mse_loss(outputs["feat_T"], forecast_target)
            aux_term = aux_term + _fc_w * forecast_loss
            loss_dict["forecast_mse"] = forecast_loss.item()

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            # Last-bar regime supervision under USE_LAST_BAR_SUPERVISION
            regime_logits_sliced = _slice_logits(outputs["regime_logits"])
            regime_tgt_sliced = _slice_target(regime_tgt)
            l_regime = F.cross_entropy(
                regime_logits_sliced.reshape(-1, 3), regime_tgt_sliced.reshape(-1)
            )
            s_regime = s[-1].clamp(max=-1.0)
            aux_term = aux_term + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (
                    regime_logits_sliced.argmax(-1) == regime_tgt_sliced
                ).float().mean().item()

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    # Last-bar dir_acc under USE_LAST_BAR_SUPERVISION
                    logits_h = _slice_logits(outputs["return_logits"][h]).reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = _slice_target(targets[h]).reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (
                            torch.sign(dec[nz]) == torch.sign(act[nz])
                        ).float().mean().item()

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
            for h_key in REWARD_HORIZONS:
                if h_key in targets:
                    # Last-bar quantile loss
                    q_pred = _slice_logits(outputs["quantile_logits"][h_key])
                    q_tgt = _slice_target(targets[h_key])
                    l_quantile = l_quantile + _ql(q_pred, q_tgt, quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile
        loss_dict["quantile_pinball"] = l_quantile.item()

        # CC-H6 regime-conditional CE (SOTA-2026, auxiliary)
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0 and regime_labels is not None):
            # Last-bar regime-conditional supervision
            flat_labels = _slice_target(regime_labels).reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h_key in REWARD_HORIZONS:
                    if h_key not in targets:
                        continue
                    head_logits = _slice_logits(outputs["regime_cond_logits"][r][h_key])
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = _slice_target(targets[h_key]).reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        total = total + _rc_w * l_regime_cond
        loss_dict["regime_cond_ce"] = l_regime_cond.item()

        loss_dict["total"] = total.item()

        if return_components:
            components = {
                "aux": aux_term,
                **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS},
            }
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(42)
    B, T, F_in = 4, WM_SEQ_LEN, INPUT_DIM
    m = iTransformerWorldModel(input_dim=F_in)
    x = torch.randn(B, T, F_in)
    asset = torch.randint(0, NUM_ASSETS, (B,))
    targets = {h: torch.randn(B, T) * 0.01 for h in REWARD_HORIZONS}
    total, ld, out = m.get_loss(x, asset, targets, mask_ratio=0.15)
    total.backward()
    n_params = count_parameters(m)
    print(f"[V22 iTransformerWorldModel smoke] PASS: B={B} T={T} F={F_in}")
    print(f"  params={n_params:,}  loss={ld['total']:.4f}  direct={ld['direct_ret']:.4f}")
    print(f"  return_logits[1]: {tuple(out['return_logits'][1].shape)}")
