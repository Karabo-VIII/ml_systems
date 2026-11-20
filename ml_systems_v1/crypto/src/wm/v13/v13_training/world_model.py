"""
V13: Temporal Fusion Transformer (TFT)
========================================

Google's time-series SOTA adapted for dollar-bar crypto.

Key innovation: Variable Selection Networks (VSN) learn per-timestep
which features matter. Instead of treating all 25 features equally,
the model learns: "at this bar, VPIN and flow matter; ignore the rest."

Architecture:
  1. Variable Selection Network: [B,T,F] -> soft feature gates [B,T,F] x features
  2. GRN encoding: gated residual networks process selected features
  3. Temporal self-attention: interpretable multi-head attention
  4. GRN decoding: decode to return predictions

No RSSM, no reconstruction, no dream. Same get_loss interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, MLPHead, SwiGLU

# Round-7 frontier components (shared across V-versions)
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from frontier_components import tail_adaptive_huber, CryptoPeriodEmbedding

# Shared, model-agnostic levers (reused, not reinvented). Import mirrors V1.1:
# try the package path, fall back to the _shared dir already on sys.path above.
# These are CONSTRUCTED only behind their flags (see __init__); importing the
# class is side-effect-free, so the base model is unchanged when the flags are OFF.
try:
    from wm._shared.variable_selection import VariableSelectionNetwork as SharedVSN
except ImportError:
    from variable_selection import VariableSelectionNetwork as SharedVSN


class GatedResidualNetwork(nn.Module):
    """GRN: core building block of TFT. Gated skip connection + ELU + dropout."""

    def __init__(self, input_dim, hidden_dim, output_dim=None, dropout=0.1, context_dim=0):
        super().__init__()
        output_dim = output_dim or input_dim
        self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        # RMSNorm instead of LayerNorm: drops the learned bias degree of
        # freedom that the GRN's gated affine path can exploit for
        # temporal memorization. Matches project-wide RMSNorm convention
        # already in use elsewhere in this file (lines 197/212/230).
        self.norm = RMSNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else None

    def forward(self, x, context=None):
        residual = self.skip(x) if self.skip else x
        if context is not None:
            x = torch.cat([x, context], dim=-1)
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        out = self.fc2(h)
        gate = torch.sigmoid(self.gate(h))
        return self.norm(residual + gate * out)


class VariableSelectionNetwork(nn.Module):
    """VSN: learns per-timestep feature importance weights.

    Input: [B, T, F] features
    Output: [B, T, D] weighted combination, [B, T, F] weights (for interpretability)
    """

    def __init__(self, n_features, d_model, hidden_dim, dropout=0.1, context_dim=0):
        super().__init__()
        self.n_features = n_features
        # Per-feature GRN transforms
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(1, hidden_dim, d_model, dropout, context_dim)
            for _ in range(n_features)
        ])
        # Softmax gate over features
        self.gate_grn = GatedResidualNetwork(
            n_features, hidden_dim, n_features, dropout, context_dim
        )

    def forward(self, x, context=None):
        """x: [B, T, F] -> ([B, T, D], [B, T, F] weights)

        SOTA upgrade 2026-04-22: hard top-k feature selection on the softmax
        weights. Forces the model to pick at most VSN_TOP_K features per
        bar, zeroing the rest. This is the anti-memorization primitive —
        without hard selection, all F features bleed through and the model
        memorizes via the un-compressed path.
        """
        B, T, F_dim = x.shape
        # Compute feature weights (soft)
        logits = self.gate_grn(x, context)  # [B, T, F]

        # Hard top-k: keep only top-K features, zero the rest
        top_k = getattr(self, "top_k", None) or min(8, F_dim)
        if top_k < F_dim:
            topk_vals, topk_idx = logits.topk(top_k, dim=-1)
            mask = torch.zeros_like(logits).scatter_(-1, topk_idx, 1.0)
            # Apply mask BEFORE softmax so non-top-k get -inf and fully zeroed
            logits = logits.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # [B, T, F]

        # Transform each feature independently
        transformed = []
        for i in range(self.n_features):
            feat_i = x[:, :, i:i+1]  # [B, T, 1]
            ctx = context if context is not None else None
            transformed.append(self.feature_grns[i](feat_i, ctx))  # [B, T, D]

        # Stack and weight (only top-K features contribute due to softmax mask)
        stacked = torch.stack(transformed, dim=2)  # [B, T, F, D]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=2)  # [B, T, D]

        return weighted, weights


class InterpretableAttention(nn.Module):
    """Multi-head attention that exposes attention weights for interpretability."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        # RMSNorm (was nn.LayerNorm before 2026-05-21). Same fix the 2026-05-16
        # GRN refactor applied at line 57; the attention block was missed. The
        # learnable bias on nn.LayerNorm gave attention a free DOF for temporal
        # memorization.
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, D], attn_weights [B, H, T, T]"""
        B, T, D = x.shape
        residual = x
        x = self.norm(x)

        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return residual + out, attn_weights


class TFTWorldModel(nn.Module):
    """V13: Temporal Fusion Transformer. Variable selection + interpretable attention."""

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # flat_dim=0: V13's heads read `feat` (post-VIB, dim=d_model) directly --
        # there is no separate flattened latent appended (unlike V1.1's
        # cat(h_seq, z_post)). Declaring flat_dim=0 lets the SHARED
        # attach_forward_regime_head() compute feat_dim = d_model + 0 = d_model,
        # which matches the `feat` the forward head reads.
        self.flat_dim = 0

        # Asset embedding (used as context for VSN)
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        # Variable Selection Network (SOTA upgrade: hard top-k selection)
        self.vsn = VariableSelectionNetwork(
            input_dim, d_model, TFT_VSN_HIDDEN, dropout, context_dim=asset_emb_dim
        )
        # Set hard top-k: only VSN_TOP_K features per bar survive softmax gate.
        # Read the import*-bound module global (not a runtime __import__("settings"),
        # which could resolve a SHADOWED settings module under meta-ensemble loading
        # of multiple versions and silently change the gate).
        self.vsn.top_k = globals().get("VSN_TOP_K", min(8, input_dim))

        # Temporal encoding (position + GRN)
        self.temporal_grn = GatedResidualNetwork(d_model, TFT_GRN_HIDDEN, d_model, dropout)

        # Interpretable attention layers
        self.attn_layers = nn.ModuleList([
            InterpretableAttention(d_model, TFT_N_HEADS, dropout)
            for _ in range(TFT_N_LAYERS)
        ])

        # Post-attention GRN
        self.post_attn_grn = GatedResidualNetwork(d_model, TFT_GRN_HIDDEN, d_model, dropout)

        # Return heads
        head_dim = RETURN_HEAD_DIM
        ret_mid = head_dim // 2
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, head_dim), RMSNorm(head_dim),
            nn.SiLU(), nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        # 2026-05-10 TFT-native quantile heads (replaces TwoHot CE):
        # Output len(QUANTILES) values per head instead of num_bins. Pinball
        # loss in get_loss; IC decode uses median (q=0.5).
        try:
            from settings import USE_QUANTILE_LOSS as _use_q, QUANTILES as _quantiles
        except ImportError:
            _use_q, _quantiles = False, (0.5,)
        self._use_quantile_loss = _use_q
        self._quantiles = tuple(_quantiles)
        n_outputs = len(self._quantiles) if _use_q else num_bins
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, ret_mid), RMSNorm(ret_mid),
                nn.SiLU(), nn.Linear(ret_mid, n_outputs),
            )
            for h in REWARD_HORIZONS
        })
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # VIB bottleneck on h_seq (round-4 anti-fragile fix). V13 was the only
        # V-version pre-round-4 with no Gaussian/categorical bottleneck. VSN
        # is feature-selection, not capacity constraint; TFT post-attn_grn is
        # unbounded. VIB adds the missing stochastic compression.
        self.z_dim = VIB_Z_DIM
        self.to_mu = nn.Linear(d_model, self.z_dim)
        self.to_logvar = nn.Linear(d_model, self.z_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, VIB_LOGVAR_INIT)
        self.z_expand = nn.Sequential(
            nn.Linear(self.z_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # ── RECON ANCHOR (keystone, 2026-06-10) ──────────────────────────────
        # Reconstruction decoder OFF the bottlenecked VIB latent (z_expand(z), the
        # d_model `feat_vib`). Mirrors the V12 donor (v12 world_model.py:222): decode
        # the bottlenecked feature back to the INPUT features [B,T,input_dim]. V13's
        # forward_train had `recon=torch.zeros` (a stub) and get_loss had `rec=0.0`
        # with RECON_WEIGHT effectively absent -- so the VIB bottleneck was a
        # pass-through (heads could route around z; no input-reconstruction pressure
        # constrained the latent). The recon term forces the 32-dim VIB latent to
        # retain input-reconstructable structure (not pure label-fit); the VIB KL
        # (already computed, line ~394) caps its capacity. Together they make the
        # bottleneck REAL -- the missing HALF of the anchor.
        self.recon_decoder = nn.Sequential(
            SwiGLU(d_model, RETURN_HEAD_DIM, dim_out=RETURN_HEAD_DIM, dropout=dropout),
            RMSNorm(RETURN_HEAD_DIM),
            nn.Linear(RETURN_HEAD_DIM, input_dim),
        )

        # ── SHARED LEVER: VSN (V13_VSN flag, default OFF) ────────────────────
        # V13 already has its OWN TFT VariableSelectionNetwork (self.vsn, line ~183)
        # that does per-timestep top-k feature selection INSIDE the TFT encoder.
        # The shared causal input-gate is therefore REDUNDANT with V13's native VSN
        # for the input-selection role -- so this lever is OFF by default AND, even
        # when ON, sits as a cheap PRE-gate before the native VSN (a second, simpler
        # sigmoid gate). Constructed ONLY when V13_VSN="1" at init; when OFF the
        # module is not present -> forward path byte-for-byte unchanged.
        import os as _os_vsn
        self._use_shared_vsn = _os_vsn.environ.get("V13_VSN", "0") == "1"
        if self._use_shared_vsn:
            self.shared_vsn = SharedVSN(input_dim)
        else:
            self.shared_vsn = None  # not constructed; no parameters, no side effects

        # ── SHARED LEVER: forward-regime/move head (V13_FORWARD_REGIME flag) ──
        # OFF by default: _use_forward_regime is False and forward_regime_head is None
        # until attach_forward_regime_head() is called by the trainer. When OFF the
        # guarded block in forward_train is a no-op (no 'forward_regime' key) and the
        # guarded aux-loss block in get_loss never fires -> base path unchanged.
        self._use_forward_regime = False
        self.forward_regime_head = None

        # Round-7: hard-coded crypto period embedding added to h_seq before VIB.
        # TFT's interpretable attention captures cross-bar relationships but does
        # not encode known periodicities (8h funding, 24h UTC, 7d weekly). The
        # period embedding makes these structural cycles explicit.
        self.period_emb = CryptoPeriodEmbedding(d_model)

        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

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

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        # SHARED VSN pre-gate (V13_VSN flag): per-timestep causal feature gate applied
        # BEFORE the causal shift / native TFT VSN. When V13_VSN=0 (default):
        # self.shared_vsn is None, branch skipped, path identical to base. When ON:
        # input_obs is replaced with gate * input_obs (same shape). The gate uses only
        # x_t (causal, no look-ahead) and is differentiable -> grad flows to gate weights.
        if self._use_shared_vsn and self.shared_vsn is not None:
            input_obs = self.shared_vsn(input_obs)

        # Causal shift
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             input_obs[:, :-1, :]], dim=1)

        # Asset context for VSN
        asset_ctx = self.asset_embedding(asset_id)  # [B, emb]
        asset_ctx_t = asset_ctx.unsqueeze(1).expand(-1, T, -1)  # [B, T, emb]

        # Variable Selection: learns which features matter per timestep
        selected, feature_weights = self.vsn(shifted, asset_ctx_t)  # [B,T,D], [B,T,F]

        # Temporal GRN
        h = self.temporal_grn(selected)

        # Interpretable attention
        all_attn = []
        for attn_layer in self.attn_layers:
            h, attn_w = attn_layer(h)
            all_attn.append(attn_w)

        h_seq = self.post_attn_grn(h)
        # RegimeFiLM (SOTA-2026 opt-in): condition h_seq on regime BEFORE VIB.
        # Identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)
        # Round-7: add hard-coded crypto period embedding (8h/24h/7d cycles)
        h_seq = h_seq + self.period_emb(T, device=h_seq.device).unsqueeze(0)

        # VIB bottleneck (round-4 anti-fragile fix). VSN is feature selection,
        # not capacity constraint — TFT post-attn was unbounded pre-round-4.
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat_vib = self.z_expand(z)

        # ── RECON ANCHOR (keystone, 2026-06-10) ──────────────────────────────
        # Decode the bottlenecked latent back to the INPUT features [B,T,input_dim].
        # Decoded off feat_vib (the clean bottlenecked feature, BEFORE ATME) so the
        # reconstruction target is the un-dropped representation -- mirrors the V12
        # donor (decode off z_expand(z)). The masked recon-MSE in get_loss compares
        # recon vs the (causally shifted) input the bottleneck must reconstruct.
        recon = self.recon_decoder(feat_vib)               # [B, T, input_dim]

        # ATME on the post-VIB feature (per-sample 0.15, CLAUDE.md invariant)
        feat = feat_vib
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device) > TEMPORAL_CTX_DROP).float()
            feat = feat * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {h_key: self.return_heads[str(h_key)](ret_trunk) for h_key in REWARD_HORIZONS}
        regime_logits = self.regime_head(feat)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        out = {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "ret_trunk": ret_trunk,
            "feature_weights": feature_weights,  # [B, T, F] for interpretability
            "attention_weights": all_attn,        # List of [B, H, T, T]
            "vib_mu": mu, "vib_logvar": logvar,
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": z,
            # RECON ANCHOR: real reconstruction [B,T,F] (was torch.zeros stub).
            "recon": recon,
        }
        # SHARED LEVER: forward-regime head. OFF by default -- _use_forward_regime
        # is False until attach_forward_regime_head() is called, so this block is a
        # no-op and out has no 'forward_regime' key -> base path byte-for-byte
        # unchanged. The head reads `feat` (post-VIB, the SAME bottlenecked feature
        # the return/regime heads read; causal, no look-ahead).
        if getattr(self, "_use_forward_regime", False) and getattr(self, "forward_regime_head", None) is not None:
            out["forward_regime"] = self.forward_regime_head(feat)  # {bear_logits,trend_logits,move_logits}
        return out

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None, **kwargs):
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

        # VIB KL term (round-4 anti-fragile bottleneck regularization)
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        kl_anneal = float(kwargs.get("kl_anneal", 1.0))
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        total = kl_weight * vib_kl
        loss_dict = {"total": 0.0, "kl": vib_kl.item(), "kl_raw": vib_kl.item(),
                     "kl_weight": kl_weight}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # ── RECON ANCHOR: reconstruction MSE (keystone, 2026-06-10) ──────────
        # recon[b,t,:] (decoded off the bottlenecked latent at position t) must
        # reconstruct the CLEAN feature the encoder consumed at that position. The
        # encoder consumes `shifted` = causal-shift of the (possibly mask_ratio-
        # dropped) input, so the recon TARGET is the causal-shift of the CLEAN
        # obs_seq (standard masked-autoencoding: reconstruct the true input from a
        # possibly-masked view). No look-ahead: position t reconstructs obs[t-1]
        # (a PAST bar), never a future one. RECON_WEIGHT>0 makes the VIB bottleneck
        # REAL -- the latent must carry input-reconstructable structure, so the
        # heads can no longer route around z. This is the HALF V13 was missing.
        recon = outputs["recon"]                                 # [B, T, F]
        recon_target = torch.cat(
            [torch.zeros(B, 1, n_feat, device=obs_seq.device), obs_seq[:, :-1, :]],
            dim=1)                                               # clean causal shift
        l_rec = F.mse_loss(recon, recon_target)
        total = total + RECON_WEIGHT * l_rec

        # 2026-05-10 TFT-native quantile loss path
        if self._use_quantile_loss:
            try:
                from settings import QUANTILE_LOSS_WEIGHT as _q_w
            except ImportError:
                _q_w = 1.0
            q_tensor = torch.tensor(self._quantiles, device=obs_seq.device)  # [n_q]
            median_idx = self._quantiles.index(0.5) if 0.5 in self._quantiles else len(self._quantiles) // 2
            for hi, h in enumerate(REWARD_HORIZONS):
                if h not in targets:
                    continue
                pred_q = outputs["return_logits"][h]               # [B, T, n_q]
                tgt = targets[h].unsqueeze(-1)                     # [B, T, 1]
                err = tgt - pred_q                                 # [B, T, n_q]
                # Pinball: max(q*err, (q-1)*err)
                pinball = torch.maximum(q_tensor * err, (q_tensor - 1.0) * err)
                if h in ACTIVE_HORIZONS:
                    l_ret = pinball.mean()
                    s_ret = s[hi].clamp(max=-2.0)
                    total = total + torch.exp(-s_ret) * (_q_w * l_ret) + s_ret
                    loss_dict["ret_%d" % h] = l_ret.item()
                # Direct return regression on median quantile
                decoded = pred_q[..., median_idx].reshape(-1)      # [B*T]
                tgt_flat = targets[h].reshape(-1)
                l_direct = l_direct + tail_adaptive_huber(
                    decoded, tgt_flat, delta=0.5, tail_sigma=2.0, tail_weight=2.5
                )
        else:
            for hi, h in enumerate(REWARD_HORIZONS):
                if h not in targets:
                    continue
                logits_flat = outputs["return_logits"][h].reshape(-1, self._num_bins)
                tgt_flat = targets[h].reshape(-1)
                if h in ACTIVE_HORIZONS:
                    l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                    s_ret = s[hi].clamp(max=-2.0)
                    total = total + torch.exp(-s_ret) * l_ret + s_ret
                    loss_dict["ret_%d" % h] = l_ret.item()
                decoded = self.bucketer.decode(logits_flat)
                # Round-7: tail-adaptive Huber upweights crypto's heavy-tail samples
                l_direct = l_direct + tail_adaptive_huber(decoded, tgt_flat,
                                                          delta=0.5, tail_sigma=2.0,
                                                          tail_weight=2.5)

        total = total + DIRECT_RETURN_WEIGHT * l_direct
        loss_dict["direct_ret"] = l_direct.item()

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(outputs["regime_logits"].reshape(-1, 3), regime_tgt.reshape(-1))
            s_regime = s[-1].clamp(max=-1.0)
            total = total + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (outputs["regime_logits"].argmax(-1) == regime_tgt).float().mean().item()

        # Feature selection entropy (track how many features the VSN actually uses)
        with torch.no_grad():
            fw = outputs["feature_weights"]  # [B, T, F]
            ent = -(fw * torch.log(fw + 1e-8)).sum(dim=-1).mean()
            loss_dict["vsn_entropy"] = ent.item()  # High = uniform, low = selective

        with torch.no_grad():
            if self._use_quantile_loss:
                median_idx = self._quantiles.index(0.5) if 0.5 in self._quantiles else len(self._quantiles) // 2
                for h in ACTIVE_HORIZONS:
                    if h in targets:
                        pred_q = outputs["return_logits"][h]                     # [B, T, n_q]
                        dec = pred_q[..., median_idx].reshape(-1)
                        act = targets[h].reshape(-1)
                        nz = torch.abs(act) > 1e-6
                        if nz.sum() > 50:
                            loss_dict["dir_acc_%d" % h] = (torch.sign(dec[nz]) == torch.sign(act[nz])).float().mean().item()
            else:
                for h in ACTIVE_HORIZONS:
                    if h in targets:
                        logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                        dec = self.bucketer.decode(logits_h)
                        act = targets[h].reshape(-1)
                        nz = torch.abs(act) > 1e-6
                        if nz.sum() > 50:
                            loss_dict["dir_acc_%d" % h] = (torch.sign(dec[nz]) == torch.sign(act[nz])).float().mean().item()

        # RECON ANCHOR: real reconstruction MSE (was hard-coded 0.0 stub).
        loss_dict["rec"] = l_rec.item()

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
                    q_pred = outputs["quantile_logits"][h_key]
                    l_quantile = l_quantile + _ql(q_pred, targets[h_key], quants)
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
            flat_labels = regime_labels.reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h_key in REWARD_HORIZONS:
                    if h_key not in targets:
                        continue
                    head_logits = outputs["regime_cond_logits"][r][h_key]
                    flat_logits = head_logits.reshape(-1, NUM_BINS)
                    flat_targets = targets[h_key].reshape(-1)
                    l_per = self.bucketer.compute_loss(
                        flat_logits[mask_r], flat_targets[mask_r])
                    l_regime_cond = l_regime_cond + l_per
                    n_terms += 1
            if n_terms > 0:
                l_regime_cond = l_regime_cond / n_terms
        total = total + _rc_w * l_regime_cond
        loss_dict["regime_cond_ce"] = l_regime_cond.item()

        # ── SHARED LEVER: forward-regime aux loss (V13_FORWARD_REGIME) ───────
        # Guarded: only fires when the head is attached (_use_forward_regime=True,
        # set by attach_forward_regime_head() in the trainer) AND the trainer
        # supplied forward_regime_labels in `targets`. Default OFF -> total unchanged.
        # Weight 0.10 (small fixed; additive supervision, NOT the primary objective).
        # Labels are built in the trainer via src/wm/_shared/regime_targets.py using
        # FUTURE bars only at TARGET-CONSTRUCTION time (never as model inputs); NaN
        # tail rows are auto-masked inside forward_regime_aux_loss.
        if getattr(self, "_use_forward_regime", False) and "forward_regime" in outputs:
            _fr_labels = targets.get("forward_regime_labels", {})
            if _fr_labels:
                import sys as _frsys
                from pathlib import Path as _frPath
                _shared_dir = str(_frPath(__file__).resolve().parent.parent.parent / "_shared")
                if _shared_dir not in _frsys.path:
                    _frsys.path.insert(0, _shared_dir)
                from forward_regime_head import forward_regime_aux_loss as _fr_aux_loss
                fr_loss = _fr_aux_loss(outputs, _fr_labels)
                total = total + 0.10 * fr_loss
                loss_dict["fr_aux"] = fr_loss.item()

        # round-4: don't overwrite VIB kl; preserve from earlier in this method
        loss_dict["total"] = total.item()
        return total, loss_dict, outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
