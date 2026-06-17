"""
V12: Cross-Asset Attention Model
==================================

Processes ALL 10 assets jointly at each timestep.

Architecture:
  Per-asset: WaveNet encoder -> per-asset hidden state [D]
  Cross-asset: Multi-head attention over 10 asset states
               Each asset attends to all others at same timestep
  Output: per-asset return prediction informed by full market state

Key insight: "BTC broke out AND ETH funding negative AND SOL VPIN spiking"
is a stronger signal than any single asset's features provide.

The attention weights are interpretable: shows which assets drive each
prediction (e.g., DOGE prediction: 80% BTC, 10% ETH, 10% self).

Training interface: same get_loss/forward_train as V1.x.
Requires synchronized multi-asset batches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

_v1_comp = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_comp not in sys.path:
    sys.path.insert(0, _v1_comp)

from components import RMSNorm, TwoHotSymlog, SwiGLU, MLPHead


# =============================================================================
# WaveNet (reused from V11, lighter config)
# =============================================================================

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class WaveNetBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.filter_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(channels, channels, kernel_size, dilation)
        self.residual_proj = nn.Conv1d(channels, channels, 1)
        self.skip_proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # fp32 for gated activation — tanh/sigmoid backward overflows fp16
        _f, _g = self.filter_conv(x), self.gate_conv(x)
        h = torch.tanh(_f.float()) * torch.sigmoid(_g.float())
        h = h.to(x.dtype)
        h = self.dropout(h)
        skip = self.skip_proj(h)
        residual = self.residual_proj(h) + x
        return residual, skip


class LightWaveNet(nn.Module):
    """Lightweight WaveNet for per-asset encoding."""

    def __init__(self, in_dim, channels, dilations, kernel_size=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, channels[0], 1)
        self.out_dim = channels[-1]
        self.blocks = nn.ModuleList()
        self.ch_trans = nn.ModuleList()
        self.skip_projs = nn.ModuleList()

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            self.ch_trans.append(nn.Conv1d(in_ch, ch, 1) if in_ch != ch else None)
            self.blocks.append(WaveNetBlock(ch, kernel_size, dil, dropout))
            self.skip_projs.append(nn.Conv1d(ch, self.out_dim, 1) if ch != self.out_dim else None)

        self.output_norm = nn.GroupNorm(min(8, self.out_dim), self.out_dim)

    def forward(self, x):
        """x: [B, T, D] -> [B, T, out_dim]"""
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        skip_sum = 0
        for i, block in enumerate(self.blocks):
            if self.ch_trans[i] is not None:
                x = self.ch_trans[i](x)
            x, skip = block(x)
            if self.skip_projs[i] is not None:
                skip = self.skip_projs[i](skip)
            skip_sum = skip_sum + skip
        x = self.output_norm(x + skip_sum)
        return x.transpose(1, 2)


# =============================================================================
# Cross-Asset Attention
# =============================================================================

class CrossAssetAttention(nn.Module):
    """Multi-head attention across assets at each timestep.

    Input: [B, N_assets, T, D] -- N_assets hidden states
    At each timestep t, each asset attends to all other assets at time t.
    Output: [B, N_assets, T, D] -- cross-asset informed representations
    """

    def __init__(self, d_model, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "norm2": RMSNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 3),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 3, d_model),
                ),
                "drop": nn.Dropout(dropout),
            }))

    def forward(self, x):
        """x: [B, A, T, D] -> [B, A, T, D]

        Reshapes to [B*T, A, D], runs attention over A dimension,
        then reshapes back. Each timestep is independent.
        """
        B, A, T, D = x.shape
        # Reshape: treat each (batch, timestep) as an independent attention problem
        x = x.permute(0, 2, 1, 3).reshape(B * T, A, D)  # [B*T, A, D]

        for layer in self.layers:
            h = layer["norm1"](x)
            h, _ = layer["attn"](h, h, h)  # No causal mask needed -- all assets are at same time
            x = x + layer["drop"](h)
            x = x + layer["drop"](layer["ffn"](layer["norm2"](x)))

        return x.reshape(B, T, A, D).permute(0, 2, 1, 3)  # [B, A, T, D]


# =============================================================================
# V12 World Model
# =============================================================================

class CrossAssetWorldModel(nn.Module):
    """V12: Cross-Asset Attention. Processes 10 assets jointly.

    Per-asset: obs -> causal_shift -> WaveNet -> per-asset hidden [D]
    Cross-asset: attention over 10 hidden states at each timestep
    Output: per-asset return prediction, informed by all other assets
    """

    def __init__(self, input_dim=INPUT_DIM, d_model=WM_D_MODEL,
                 num_bins=NUM_BINS, num_assets=NUM_ASSETS,
                 asset_emb_dim=WM_ASSET_EMB_DIM, dropout=WM_DROPOUT):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_assets = num_assets
        self.z_dim = VIB_Z_DIM

        # Shared per-asset encoder (same weights for all assets)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model), nn.SiLU(), nn.Dropout(dropout),
        )
        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        nn.init.normal_(self.asset_embedding.weight, 0, 0.02)

        self.wavenet = LightWaveNet(
            d_model, WAVENET_CHANNELS, WAVENET_DILATIONS,
            WAVENET_KERNEL, WAVENET_DROPOUT,
        )

        # Cross-asset attention (DEAD CODE in single-asset training path;
        # only fires when forward_multi_asset is called explicitly)
        self.cross_attn = CrossAssetAttention(
            d_model, CROSS_ATTN_HEADS, CROSS_ATTN_LAYERS, CROSS_ATTN_DROPOUT,
        )

        # Variational Information Bottleneck — same fix as V3-clean (5f25f97
        # context). Without VIB, single-asset forward_train memorizes via the
        # WaveNet receptive field; ATME (h_seq * mask) is a no-op channel drop.
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

        # Cross-asset reconstruction decoder (HEADLINE anchor, 2026-06-10).
        # Mirrors the V1.1 single-asset RSSM decoder: decode the bottlenecked
        # latent (z_expand(z), d_model) back to the INPUT features [.,F]. The
        # multi-asset path (forward_multi_asset) had NO recon + NO VIB KL (hard
        # kl=0.0), so it was the V22/V25 memorization trap (high contiguous IC,
        # ShIC~0). Reconstruction forces the 16-dim VIB latent to retain
        # input-reconstructable structure (not pure label-fit); VIB KL caps its
        # capacity -> the bottleneck the cross-asset path was missing.
        # Decode target = ALL input_dim features (V12 has no XD split: f25 IS
        # the cross-asset signal, BASE_DIM is the f34 single-asset convention).
        self.recon_decoder = nn.Sequential(
            SwiGLU(d_model, RETURN_HEAD_DIM, dim_out=RETURN_HEAD_DIM, dropout=dropout),
            RMSNorm(RETURN_HEAD_DIM),
            nn.Linear(RETURN_HEAD_DIM, input_dim),
        )

        # Per-asset return prediction (shared weights, conditioned by asset embedding)
        head_dim = RETURN_HEAD_DIM
        ret_mid = head_dim // 2
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, head_dim), RMSNorm(head_dim),
            nn.SiLU(), nn.Dropout(RETURN_HEAD_DROPOUT),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, ret_mid), RMSNorm(ret_mid),
                nn.SiLU(), nn.Linear(ret_mid, num_bins),
            )
            for h in REWARD_HORIZONS
        })

        # Frontier-ML upgrade hooks (set via apply_v1_upgrades). Default OFF.
        self._use_mtp = False
        self.mtp_head = None
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        # TwoHot
        self._num_bins = num_bins
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"

        # Kendall: [ret_1, ret_4, ret_16, ret_64, regime]
        self.log_vars = nn.Parameter(torch.tensor([-2.0] * len(REWARD_HORIZONS) + [-1.5]))

        # CC-H5 + CC-H6 + RegimeFiLM (SOTA-2026 auxiliary heads / encoder cond)
        # NOTE: V12's primary architecture is multi-asset (forward_multi_asset);
        # the single-asset fallback wires these heads on per-asset h_seq. When
        # MultiAssetDataset lands, the multi-asset path will also benefit.
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
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward_single_asset(self, obs_seq, asset_id):
        """Encode a single asset. Used for per-asset processing before cross-attention.

        Args:
            obs_seq: [B, T, F]
            asset_id: [B] or scalar

        Returns: [B, T, D]
        """
        B, T, n_feat = obs_seq.shape
        if isinstance(asset_id, int):
            asset_id = torch.full((B,), asset_id, dtype=torch.long, device=obs_seq.device)

        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        shifted = torch.cat([torch.zeros(B, 1, n_feat, device=obs_seq.device),
                             obs_seq[:, :-1, :]], dim=1)
        h = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))
        # fp32 for WaveNet: Conv1d backward + skip accumulation overflow in fp16
        with torch.amp.autocast("cuda", enabled=False):
            h = self.wavenet(h.float())
        return h

    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """Forward pass for single-asset (V1-compatible interface).

        VIB-bottlenecked. Cross-asset attention is bypassed here (dead code in
        the standard runner). For real multi-asset training, use forward_multi_asset.
        """
        B, T, n_feat = obs_seq.shape
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq
        h_seq = self.forward_single_asset(input_obs, asset_id)

        # RegimeFiLM (SOTA-2026 opt-in) — identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # Variational Information Bottleneck (replaces no-op ATME)
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)

        ret_trunk = self.return_trunk(feat)
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(ret_trunk)
            return_logits = {h: mtp_out[f"h{h}"] for h in REWARD_HORIZONS}
        elif getattr(self, "_use_mdn", False):
            return_logits = {h: ret_trunk for h in REWARD_HORIZONS}
        else:
            return_logits = {h: self.return_heads[str(h)](ret_trunk) for h in REWARD_HORIZONS}
        regime_logits = self.regime_head(h_seq)

        # CC-H5 + CC-H6 auxiliary heads (SOTA-2026)
        quantile_logits = None
        if self.quantile_heads is not None:
            quantile_logits = self.quantile_heads(feat)
        regime_cond_logits = None
        if self.regime_cond_heads is not None:
            regime_cond_logits = self.regime_cond_heads(feat)

        return {
            "return_logits": return_logits, "regime_logits": regime_logits,
            "h_seq": h_seq, "ret_trunk": ret_trunk,
            "vib_mu": mu, "vib_logvar": logvar, "z_post": z,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }

    def forward_multi_asset(self, multi_obs, multi_asset_ids):
        """Forward pass with cross-asset attention.

        Args:
            multi_obs: [B, A, T, F] -- A assets, synchronized timestamps
            multi_asset_ids: [B, A] -- asset indices

        Returns: dict with per-asset predictions
        """
        B, A, T, F = multi_obs.shape

        # Encode each asset independently
        all_h = []
        for a in range(A):
            h_a = self.forward_single_asset(multi_obs[:, a], multi_asset_ids[:, a])
            all_h.append(h_a)
        h_stack = torch.stack(all_h, dim=1)  # [B, A, T, D]

        # Cross-asset attention
        h_cross = self.cross_attn(h_stack)  # [B, A, T, D]

        # ATME on cross-asset output
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, 1, device=h_cross.device) > TEMPORAL_CTX_DROP).float()
            h_cross = h_cross * atme_mask

        # ── Variational Information Bottleneck (HEADLINE anchor, 2026-06-10) ──
        # The cross-asset path previously fed h_cross STRAIGHT into the return
        # heads (no bottleneck, no recon, hard kl=0.0) -> the V22/V25 trap. Mirror
        # the single-asset forward_train VIB: reparameterize h_cross -> (mu,logvar)
        # -> z (16-dim), expand back to d_model. Capacity of z is the bottleneck;
        # the KL term (computed in get_multi_loss) forces compression toward the
        # N(0,1) prior. Causality: h_cross is built from forward_single_asset's
        # causal shift (obs[:, :-1]) per asset, and cross_attn mixes only over the
        # ASSET axis at a fixed t -> no temporal look-ahead is introduced here.
        mu = self.to_mu(h_cross)                                   # [B, A, T, z_dim]
        logvar = self.to_logvar(h_cross).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)                                    # [B, A, T, D] (bottlenecked)

        # Reconstruction: decode the bottlenecked latent back to input features.
        # recon[b,a,t,:] reconstructs multi_obs[b,a,t,:]; masking in get_multi_loss
        # zeros absent-asset slots. Forces z to retain reconstructable structure.
        recon = self.recon_decoder(feat)                          # [B, A, T, F]

        # Per-asset predictions (now read the BOTTLENECKED feat, not raw h_cross)
        all_return_logits = {h: [] for h in REWARD_HORIZONS}
        all_regime_logits = []

        for a in range(A):
            feat_a = feat[:, a]  # [B, T, D] -- bottlenecked
            rt = self.return_trunk(feat_a)
            for h in REWARD_HORIZONS:
                all_return_logits[h].append(self.return_heads[str(h)](rt))
            all_regime_logits.append(self.regime_head(feat_a))

        # Stack: [B, A, T, bins] for each horizon
        return {
            "return_logits": {h: torch.stack(v, dim=1) for h, v in all_return_logits.items()},
            "regime_logits": torch.stack(all_regime_logits, dim=1),
            "h_seq": h_cross,        # [B, A, T, D] -- pre-bottleneck (for ShIC/diagnostics)
            "vib_mu": mu,            # [B, A, T, z_dim]
            "vib_logvar": logvar,    # [B, A, T, z_dim]
            "recon": recon,          # [B, A, T, F]
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None,
                 return_components=False, **kwargs):
        """Single-asset loss (V1-compatible). For multi-asset, use get_multi_loss.

        Args:
            return_components: when True, returns 4-tuple
                (total, loss_dict, outputs, components). components has
                {aux, ret_1, ret_4, ret_16, ret_64} for PCGrad.
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

        # MDN-aware: head.log_prob instead of TwoHot CE; head.expectation instead
        # of bucketer.decode for the direct-return term.
        use_mdn = getattr(self, "_use_mdn", False)
        ret_trunk_for_mdn = outputs.get("ret_trunk") if use_mdn else None

        # VIB KL term
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean()
        # default 1.0 (was 0.0): validate() calls get_loss WITHOUT kl_anneal, so the
        # val total excluded the KL term -> checkpoint selection could pick a
        # VIB-collapsed model. Training passes kl_anneal explicitly, so unaffected.
        kl_anneal = kwargs.get("kl_anneal", 1.0)
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        loss_dict = {"total": 0.0, "kl": vib_kl.item(), "kl_raw": vib_kl.item(),
                     "kl_weight": kl_weight, "rec": 0.0}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        # Per-horizon weighted return terms (kept separate for PCGrad)
        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            if h in ACTIVE_HORIZONS:
                if use_mdn:
                    head = self.return_heads[str(h)]
                    l_ret = -head.log_prob(ret_trunk_for_mdn, targets[h]).mean()
                else:
                    logits_flat = outputs["return_logits"][h].reshape(-1, self._num_bins)
                    tgt_flat = targets[h].reshape(-1)
                    l_ret = self.bucketer.compute_loss(logits_flat, tgt_flat)
                s_ret = s[hi].clamp(max=-2.0)
                ret_terms[h] = torch.exp(-s_ret) * l_ret + s_ret
                loss_dict["ret_%d" % h] = l_ret.item()
            # Direct return regression (MDN-aware)
            if use_mdn:
                decoded = self.return_heads[str(h)].expectation(ret_trunk_for_mdn).reshape(-1)
            else:
                logits_flat2 = outputs["return_logits"][h].reshape(-1, self._num_bins)
                decoded = self.bucketer.decode(logits_flat2)
            l_direct = l_direct + F.huber_loss(
                decoded, targets[h].reshape(-1), reduction="mean", delta=0.5,
            )

        # Aux groups everything that ISN'T per-horizon (PCGrad surgery target).
        aux_term = kl_weight * vib_kl + DIRECT_RETURN_WEIGHT * l_direct
        loss_dict["direct_ret"] = l_direct.item()

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(outputs["regime_logits"].reshape(-1, 3), regime_tgt.reshape(-1))
            s_regime = s[-1].clamp(max=-1.0)
            aux_term = aux_term + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (outputs["regime_logits"].argmax(-1) == regime_tgt).float().mean().item()

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    if use_mdn:
                        dec = self.return_heads[str(h)].expectation(ret_trunk_for_mdn).reshape(-1)
                    else:
                        logits_h = outputs["return_logits"][h].reshape(-1, self._num_bins)
                        dec = self.bucketer.decode(logits_h)
                    act = targets[h].reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (torch.sign(dec[nz]) == torch.sign(act[nz])).float().mean().item()

        loss_dict["rec"] = 0.0

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

        loss_dict["total"] = total.item()

        if return_components:
            components = {"aux": aux_term, **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS}}
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs


    def get_multi_loss(self, multi_obs, multi_asset_ids, targets, mask,
                       return_components=False, kl_anneal=1.0):
        """Multi-asset loss for HEADLINE_MODE training (Layer-C wiring, 2026-06-10).

        Args:
            multi_obs:        [B, A, T, F]   -- synchronized multi-asset observations
            multi_asset_ids:  [B, A]          -- asset index per slot
            targets:          {h: [B, A, T]} -- per-horizon return targets
            mask:             [B, A, T] bool  -- True = real data, False = absent/padded
            return_components: when True, return 4-tuple like get_loss
            kl_anneal:        VIB KL anneal factor in [0,1] (trainer ramps 0->1 over
                              VIB_KL_ANNEAL_EPOCHS). Default 1.0 so validate() (which
                              omits it) still includes the full KL in checkpoint
                              selection -- mirrors the single-asset get_loss default.

        Loss contract:
          - forward_multi_asset -> return_logits {h: [B, A, T, bins]}, recon [B,A,T,F],
            vib_mu/vib_logvar [B,A,T,z_dim]  (the HEADLINE anchor, 2026-06-10).
          - flatten [B, A, T] -> [B*A*T] for TwoHot CE + Huber (same as single-asset)
          - mask_flat (float 0/1) MULTIPLIES per-sample losses before .sum() / mask_sum
            so absent-asset (mask=False) slots contribute ZERO to gradients.
          - recon MSE + VIB KL are likewise masked (broadcast mask over F / z_dim) so
            absent assets contribute 0 to BOTH terms.
          - Kendall log-var weighting mirrors the single-asset path.

        TwoHot CE per-sample: uses encode() to get (idx_floor, idx_ceil, w_f, w_c)
        then computes w_f * CE(floor) + w_c * CE(ceil) per slot -- same arithmetic as
        TwoHotSymlog.compute_loss(), but with reduction="none" for masking.

        ANCHOR (2026-06-10): recon + VIB KL close the memorization gap. Without
        them the cross-asset path was the V22/V25 trap (high contiguous IC,
        ShIC~0). recon forces the 16-dim VIB latent to retain reconstructable
        structure; the KL caps its capacity -> a real information bottleneck.
        """
        B, A, T, _F = multi_obs.shape
        dev = multi_obs.device
        dev_str = str(dev)
        if self._bucketer_device != dev_str:
            self.bucketer = TwoHotSymlog(self._num_bins, BIN_MIN, BIN_MAX, dev_str)
            self._bucketer_device = dev_str

        outputs = self.forward_multi_asset(multi_obs, multi_asset_ids)
        # outputs["return_logits"][h] shape: [B, A, T, bins]

        # Flatten mask to [B*A*T] float weights. mask_sum guards against /0.
        mask_flat = mask.reshape(-1).float()          # [B*A*T]
        mask_sum = mask_flat.sum().clamp(min=1.0)     # scalar, never zero

        s = self.log_vars.clamp(-6.0, 6.0)
        l_direct = torch.tensor(0.0, device=dev)
        ret_terms = {h: torch.tensor(0.0, device=dev) for h in REWARD_HORIZONS}
        loss_dict: Dict = {"total": 0.0, "rec": 0.0, "kl": 0.0, "kl_raw": 0.0,
                           "direct_ret": 0.0}

        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets or h not in ACTIVE_HORIZONS:
                continue
            logits_h = outputs["return_logits"][h]          # [B, A, T, bins]
            logits_flat = logits_h.reshape(-1, self._num_bins)  # [B*A*T, bins]
            tgt_flat = targets[h].reshape(-1)               # [B*A*T]

            # TwoHot CE per-sample (reduction="none" inline, avoids needing a new method).
            # Replicates TwoHotSymlog.compute_loss() with per-sample output for masking.
            idx_f, idx_c, w_f, w_c = self.bucketer.encode(tgt_flat)
            ce_f = F.cross_entropy(logits_flat, idx_f, reduction="none")  # [B*A*T]
            ce_c = F.cross_entropy(logits_flat, idx_c, reduction="none")  # [B*A*T]
            ce_per = w_f * ce_f + w_c * ce_c                              # [B*A*T]
            # Apply mask: absent slots (mask=False) contribute 0 to the sum.
            l_ret = (ce_per * mask_flat).sum() / mask_sum

            s_ret = s[hi].clamp(max=-2.0)
            ret_terms[h] = torch.exp(-s_ret) * l_ret + s_ret
            loss_dict["ret_%d" % h] = l_ret.item()

            # Direct Huber, masked
            decoded = self.bucketer.decode(logits_flat)          # [B*A*T]
            huber_per = F.huber_loss(decoded, tgt_flat, reduction="none", delta=0.5)
            l_direct = l_direct + (huber_per * mask_flat).sum() / mask_sum

        loss_dict["direct_ret"] = l_direct.item()

        # ── ANCHOR: masked reconstruction MSE (HEADLINE, 2026-06-10) ──────────
        # recon [B,A,T,F] vs multi_obs [B,A,T,F], masked so absent assets (and
        # padded tail bars) contribute 0. mask4 broadcasts [B,A,T] over F.
        mask4 = mask.unsqueeze(-1).float()                    # [B,A,T,1]
        recon = outputs["recon"]                               # [B,A,T,F]
        recon_sq = (recon - multi_obs).pow(2) * mask4          # [B,A,T,F]
        # Denominator = number of UNMASKED feature-elements (mask_sum * F).
        recon_denom = (mask4.sum() * _F).clamp(min=1.0)
        l_rec = recon_sq.sum() / recon_denom

        # ── ANCHOR: masked VIB KL (free-floating; replaces hard kl=0.0) ──────
        # KL(N(mu,sigma) || N(0,1)) per latent element, masked + annealed. This
        # is the capacity cap that makes the bottleneck REAL (kl=0.0 made it a
        # pass-through). Element-wise KL = -0.5(1 + logvar - mu^2 - exp(logvar)).
        mu = outputs["vib_mu"]                                 # [B,A,T,z_dim]
        logvar = outputs["vib_logvar"]                         # [B,A,T,z_dim]
        kl_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # [B,A,T,z_dim]
        kl_elem = kl_elem * mask4                              # absent assets -> 0
        kl_denom = (mask4.sum() * self.z_dim).clamp(min=1.0)
        vib_kl = kl_elem.sum() / kl_denom                     # per-element mean KL
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        # aux = direct-return Huber + recon + VIB KL (the anchor terms).
        # rec uses Kendall-free fixed weight DIRECT_RETURN_WEIGHT-class scaling via
        # VIB_KL_WEIGHT for KL; recon is unit-weighted (MSE, same as V1.1 pre-Kendall
        # baseline -- the single-asset path Kendall-weights it, but the multi-asset
        # log_vars vector has no recon/kl slots, so a fixed unit weight is the
        # faithful minimal change). Watch loss_dict["rec"]/["kl"] in training.
        aux_term = (DIRECT_RETURN_WEIGHT * l_direct
                    + l_rec
                    + kl_weight * vib_kl)
        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        loss_dict["rec"] = l_rec.item()
        loss_dict["kl"] = vib_kl.item()
        loss_dict["kl_raw"] = vib_kl.item()
        loss_dict["kl_weight"] = kl_weight
        loss_dict["total"] = total.item()

        if return_components:
            components = {"aux": aux_term, **{f"ret_{h}": ret_terms[h] for h in REWARD_HORIZONS}}
            return total, loss_dict, outputs, components
        return total, loss_dict, outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
