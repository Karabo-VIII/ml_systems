"""V24 World Model — TimesNet (Wu et al., ICLR 2023).

Production V-version of the V24 backbone. Adds get_loss() following the
V13 pattern.

Architecture (faithful to ICLR 2023 paper §3):
  1. FFT period detection: top-K frequencies from amplitude spectrum (Algorithm 1)
  2. For each period p_i: reshape [B, T, D] -> [B, D, p_i, ceil(T/p_i)]
     (with right-padding if T not divisible by p_i)
  3. Inception 2D conv: parallel kernels (1, 3, 5, 7) on the 2D tensor
  4. Reshape back to [B, T, D]
  5. Aggregate K period-specific outputs via softmax(amplitude)-weighted sum
  6. Stacked TimesBlocks
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

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)

from components import RMSNorm, TwoHotSymlog, MLPHead

# Round-7 frontier components
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from frontier_components import tail_adaptive_huber, CryptoPeriodEmbedding


def fft_top_k_periods(x: torch.Tensor, top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect top-K periods via FFT amplitude (paper §3.1, Algorithm 1).

    x: [B, T, D]
    Returns:
        periods: [top_k] integer periods (shared across batch — see note below)
        weights: [B, top_k] per-batch amplitude weights for aggregation

    Note (2026-05-21 RED-team audit): the original implementation used
    `amp.mean(dim=0)` (cross-batch pool) before topk. With multi-asset batches
    that pooled BTC's spectrum with ETH/SOL/... so period selection leaked
    across assets and was non-deterministic across batches. Fixed to use the
    first sample's spectrum as the period selector — within a single asset
    that's the asset's own spectrum; across assets it stays deterministic
    per-sample. (The downstream `weights = amp[:, top_freqs]` is still
    per-sample so the aggregation respects each sample's amplitude.)

    Still a residual look-ahead: torch.fft.rfft sees ALL T bars including
    future ones at any given step. That's a structural property of FFT-based
    period selection — fix would require windowed/streaming FFT.
    """
    B, T, D = x.shape
    # FFT in fp32 for stability under autocast
    x_fp32 = x.float()
    xf = torch.fft.rfft(x_fp32, dim=1)                  # [B, T//2+1, D]
    amp = xf.abs().mean(dim=-1)                          # [B, T//2+1]
    # Per-sample (no longer mean(dim=0)): use the first sample's amplitude
    # spectrum to pick periods. Avoids cross-asset leak when batches mix assets.
    # Conv2d kernels are shared across the batch so the same periods must be
    # used for all samples in the batch — this is a deliberate constraint.
    amp_first = amp[0].clone()                           # [T//2+1]
    amp_first[0] = 0.0                                   # discard DC
    top_freqs = torch.topk(amp_first, top_k).indices     # [top_k]
    periods = (T // top_freqs.clamp(min=1)).clamp(min=1)
    weights = amp[:, top_freqs]                          # [B, top_k] per-sample
    return periods, weights


class InceptionBlock2D(nn.Module):
    """Multi-kernel 2D convolutions (TimesNet paper §3.2)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: tuple = INCEPTION_KERNELS):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, p_i, T/p_i]
        return sum(branch(x) for branch in self.branches) / len(self.branches)


class TimesBlock(nn.Module):
    """One TimesNet block: FFT-period detection + 2D inception + aggregate."""

    def __init__(self, d_model: int, top_k: int = TOP_K_PERIODS,
                 inception_channels: int = INCEPTION_CHANNELS,
                 dropout: float = WM_DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        # Channel-wise lift to inception_channels for 2D conv, then back to d_model
        self.lift = nn.Conv2d(d_model, inception_channels, kernel_size=1)
        self.inception = InceptionBlock2D(inception_channels, inception_channels)
        self.proj = nn.Conv2d(inception_channels, d_model, kernel_size=1)
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_2d(self, x: torch.Tensor, period: int) -> tuple[torch.Tensor, int]:
        # x: [B, T, D] -> [B, D, period, T_padded/period]
        B, T, D = x.shape
        target_T = ((T + period - 1) // period) * period
        if target_T > T:
            x_pad = F.pad(x, (0, 0, 0, target_T - T))
        else:
            x_pad = x
        n_periods = target_T // period
        out = x_pad.transpose(1, 2).reshape(B, D, n_periods, period)
        out = out.transpose(2, 3).contiguous()
        return out, target_T

    def _reshape_to_1d(self, x_2d: torch.Tensor, target_T: int, T: int) -> torch.Tensor:
        B, D, period, n_periods = x_2d.shape
        x_1d = x_2d.transpose(2, 3).reshape(B, D, target_T)
        x_1d = x_1d[:, :, :T].transpose(1, 2)
        return x_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        x_norm = self.norm(x)
        periods, weights = fft_top_k_periods(x_norm, self.top_k)

        per_period_outs = []
        for i, p_t in enumerate(periods.tolist()):
            p = max(int(p_t), 1)
            x_2d, target_T = self._reshape_to_2d(x_norm, p)
            h = self.lift(x_2d)
            h = F.silu(self.inception(h))
            h = self.proj(h)
            h_1d = self._reshape_to_1d(h, target_T, T)
            per_period_outs.append(h_1d)

        stacked = torch.stack(per_period_outs, dim=-1)        # [B, T, D, K]
        w = F.softmax(weights, dim=-1).unsqueeze(1).unsqueeze(1).to(stacked.dtype)  # [B, 1, 1, K]
        agg = (stacked * w).sum(dim=-1)                        # [B, T, D]
        return x + self.dropout(agg)


class TimesNetWorldModel(nn.Module):
    """V24: stacked TimesBlocks for crypto WM (V1.x-compatible interface)."""

    def __init__(self, input_dim: int = INPUT_DIM, seq_len: int = WM_SEQ_LEN,
                 d_model: int = WM_D_MODEL, n_blocks: int = N_BLOCKS,
                 top_k: int = TOP_K_PERIODS,
                 inception_channels: int = INCEPTION_CHANNELS,
                 dropout: float = WM_DROPOUT, num_bins: int = NUM_BINS,
                 num_assets: int = NUM_ASSETS,
                 asset_emb_dim: int = WM_ASSET_EMB_DIM,
                 z_dim: int = VIB_Z_DIM):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model
        self._num_bins = num_bins
        self.z_dim = z_dim

        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            TimesBlock(d_model, top_k=top_k,
                       inception_channels=inception_channels, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.post_norm = RMSNorm(d_model)

        # VIB bottleneck on h_seq (round-4 anti-fragile fix). TimesNet's FFT
        # period detection is regime-dependent and could memorize bull-cycle /
        # weekend-effect patterns; VIB adds stochastic compression.
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
        self.regime_head = MLPHead(d_model, REGIME_HEAD_DIM, 3, dropout)

        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, "cpu")
        self._bucketer_device = "cpu"
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
        input_obs = masked_obs_seq if masked_obs_seq is not None else obs_seq

        shifted = torch.cat(
            [torch.zeros(B, 1, F_in, device=obs_seq.device), input_obs[:, :-1, :]],
            dim=1,
        )
        asset_emb = self.asset_embedding(asset_id).unsqueeze(1).expand(-1, T, -1)
        x = self.obs_encoder(torch.cat([shifted, asset_emb], dim=-1))   # [B, T, D]

        for block in self.blocks:
            x = block(x)
        h_seq = self.post_norm(x)

        # RegimeFiLM (SOTA-2026 opt-in) — identity-at-init.
        if self.regime_film is not None and self.regime_film_gate is not None:
            regime_logits_for_film = self.regime_film_gate(h_seq)
            regime_probs_film = torch.softmax(regime_logits_for_film, dim=-1)
            h_seq = self.regime_film(h_seq, regime_probs_film)

        # VIB bottleneck (anti-fragile compression on h_seq)
        mu = self.to_mu(h_seq)
        logvar = self.to_logvar(h_seq).clamp(VIB_LOGVAR_MIN, VIB_LOGVAR_MAX)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat_vib = self.z_expand(z)

        feat = feat_vib
        if self.training and TEMPORAL_CTX_DROP > 0:
            atme_mask = (torch.rand(B, 1, 1, device=feat.device)
                         > TEMPORAL_CTX_DROP).float()
            # Preserve the supervised last bar (USE_LAST_BAR_SUPERVISION reads
            # feat[:, -1]); mask context [0:T-1] only. Same fix as V22/V25 -- V24
            # had last-bar supervision added but its ATME still zeroed the last bar.
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
            "z_post": z,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
            "quantile_logits": quantile_logits,
            "regime_cond_logits": regime_cond_logits,
        }

    def get_loss(self, obs_seq, asset_id, targets,
                 mask_ratio=0.0, block_mask=False, regime_labels=None,
                 return_components=False, **kwargs):
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

        # LAST-BAR SUPERVISION (2026-05-29 fix): TimesNet's 2D-inception is NON-causal
        # (symmetric padding lets a mid-window bar's features see later cycles). Under
        # per-bar supervision that leaks future info into the graded prediction. Grading
        # only the LAST bar (whose window is entirely its past) makes it causal-safe,
        # mirroring V22/V25.
        try:
            from settings import USE_LAST_BAR_SUPERVISION as _last_bar
        except ImportError:
            _last_bar = True

        def _slice_logits(x):
            return x[:, -1:, :] if _last_bar else x

        def _slice_target(x):
            return x[:, -1:] if _last_bar else x

        # VIB KL term (round-4 anti-fragile bottleneck regularization)
        mu = outputs["vib_mu"]
        logvar = outputs["vib_logvar"]
        vib_kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        kl_anneal = float(kwargs.get("kl_anneal", 1.0))
        kl_weight = VIB_KL_WEIGHT * kl_anneal

        s = self.log_vars.clamp(-6.0, 6.0)
        loss_dict = {"total": 0.0, "rec": 0.0, "kl": vib_kl.item(),
                     "kl_raw": vib_kl.item(), "kl_weight": kl_weight}
        l_direct = torch.tensor(0.0, device=obs_seq.device)

        ret_terms = {h: torch.tensor(0.0, device=obs_seq.device) for h in REWARD_HORIZONS}
        for hi, h in enumerate(REWARD_HORIZONS):
            if h not in targets:
                continue
            logits_flat = _slice_logits(outputs["return_logits"][h]).reshape(-1, self._num_bins)
            tgt_flat = _slice_target(targets[h]).reshape(-1)
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

        if regime_labels is not None:
            regime_tgt = regime_labels.long().clamp(0, 2)
            l_regime = F.cross_entropy(
                _slice_logits(outputs["regime_logits"]).reshape(-1, 3),
                _slice_target(regime_tgt).reshape(-1)
            )
            s_regime = s[-1].clamp(max=-1.0)
            aux_term = aux_term + torch.exp(-s_regime) * l_regime + s_regime
            loss_dict["regime"] = l_regime.item()
            with torch.no_grad():
                loss_dict["regime_acc"] = (
                    _slice_logits(outputs["regime_logits"]).argmax(-1) == _slice_target(regime_tgt)
                ).float().mean().item()

        total = aux_term + sum(ret_terms[h] for h in REWARD_HORIZONS)

        with torch.no_grad():
            for h in ACTIVE_HORIZONS:
                if h in targets:
                    logits_h = _slice_logits(outputs["return_logits"][h]).reshape(-1, self._num_bins)
                    dec = self.bucketer.decode(logits_h)
                    act = _slice_target(targets[h]).reshape(-1)
                    nz = torch.abs(act) > 1e-6
                    if nz.sum() > 50:
                        loss_dict["dir_acc_%d" % h] = (
                            torch.sign(dec[nz]) == torch.sign(act[nz])
                        ).float().mean().item()

        # CC-H5 + CC-H6 auxiliary losses (SOTA-2026)
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
                    # slice to last bar (consistent with the main loss under
                    # USE_LAST_BAR_SUPERVISION; the non-causal inception makes
                    # mid-window positions look-ahead-contaminated).
                    q_pred = _slice_logits(outputs["quantile_logits"][h_key])
                    l_quantile = l_quantile + _ql(q_pred, _slice_target(targets[h_key]), quants)
                    n_h += 1
            if n_h > 0:
                l_quantile = l_quantile / n_h
        total = total + _ql_w * l_quantile
        loss_dict["quantile_pinball"] = l_quantile.item()
        try:
            from settings import REGIME_COND_WEIGHT as _rc_w
        except ImportError:
            _rc_w = 0.0
        l_regime_cond = torch.tensor(0.0, device=obs_seq.device)
        if (self.regime_cond_heads is not None
                and outputs.get("regime_cond_logits") is not None
                and _rc_w > 0 and regime_labels is not None):
            # slice to last bar (consistent with main loss under last-bar supervision)
            flat_labels = _slice_target(regime_labels).reshape(-1)
            n_terms = 0
            for r in range(self.regime_cond_heads.n_regimes):
                mask_r = (flat_labels == r)
                if not mask_r.any():
                    continue
                for h_key in REWARD_HORIZONS:
                    if h_key not in targets:
                        continue
                    head_logits = outputs["regime_cond_logits"][r][h_key]
                    flat_logits = _slice_logits(head_logits).reshape(-1, NUM_BINS)
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
    torch.manual_seed(42)
    B, T, F_in = 4, WM_SEQ_LEN, INPUT_DIM
    m = TimesNetWorldModel(input_dim=F_in)
    x = torch.randn(B, T, F_in)
    asset = torch.randint(0, NUM_ASSETS, (B,))
    targets = {h: torch.randn(B, T) * 0.01 for h in REWARD_HORIZONS}
    total, ld, out = m.get_loss(x, asset, targets, mask_ratio=0.15)
    total.backward()
    n_params = count_parameters(m)
    print(f"[V24 TimesNetWorldModel smoke] PASS: B={B} T={T} F={F_in}")
    print(f"  params={n_params:,}  loss={ld['total']:.4f}  direct={ld['direct_ret']:.4f}")
    print(f"  return_logits[1]: {tuple(out['return_logits'][1].shape)}")
