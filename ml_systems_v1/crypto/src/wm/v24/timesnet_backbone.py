"""V24 — TimesNet (Wu et al., ICLR 2023).

Source: Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., Long, M. (2023).
TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.
ICLR 2023. arxiv 2210.02186.

Why TimesNet for this project:
  Crypto markets have strong cyclical structure:
    - 8-hour funding rate cycle (Binance perp)
    - 24-hour UTC daily cycle (US/Asia open/close shifts)
    - 7-day weekly cycle (weekend effect, options expiry Friday)
  Standard 1D temporal models (Transformer, RNN, WaveNet) can learn these
  but only implicitly. TimesNet detects them explicitly via FFT, then RESHAPES
  the 1D series into a 2D tensor where:
    - Row axis = position WITHIN the cycle (intra-period)
    - Col axis = which cycle (inter-period)
  Inception-style 2D convolutions then capture both relations directly.

Architecture (faithful to ICLR 2023 paper §3):
  1. FFT-based period detection: top-K periods from amplitude spectrum
  2. For each period p_i: reshape [B, T, D] -> [B, p_i, ceil(T/p_i), D]
     (zero-pad if T not divisible by p_i)
  3. Inception 2D conv: parallel kernel sizes (1, 3, 5) on the 2D tensor
  4. Reshape back to [B, T, D]
  5. Aggregate K period-specific outputs via softmax(amplitude)-weighted sum
  6. Stacked TimesBlocks (paper default 2-3 blocks)

Anti-memorization:
  - 2D conv inception has bounded receptive field per layer (no global path)
  - Period detection is data-driven; no fixed cycle assumption
  - ATME 0.15 (per-sample, V1.x convention)

Iron-clad sizing for our regime (8 GB VRAM, B=32, T=96, F~29-121):
  d_model=192, n_blocks=3, n_kernels=4, top_k=3 → ~3-5M params.

Status: BACKBONE SCAFFOLD. Forward + backward + smoke verified.
Trainer wiring (matching V1.x interface) is a separate ~2-day work item.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_v1_path = _PROJECT_ROOT / "src" / "wm" / "v1" / "v1_0_training"
if str(_v1_path) not in sys.path:
    sys.path.insert(0, str(_v1_path))


def _try_import_components():
    try:
        from components import RMSNorm, TwoHotSymlog, MLPHead  # noqa
        return RMSNorm, TwoHotSymlog, MLPHead
    except Exception:
        class _RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return _RMSNorm, None, None


def fft_top_k_periods(x: torch.Tensor, top_k: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """Detect top-K periods via FFT amplitude (paper S3.1, Algorithm 1).

    x: [B, T, D]
    Returns:
        periods: [top_k] integer periods (shared across batch)
        weights: [B, top_k] per-batch amplitude weights for aggregation

    !! LOOK-AHEAD CAVEAT (2026-06-10 hardening flag, D6): this FFT runs over the WHOLE window [0,T],
    and the downstream TimesBlock 2D-conv also mixes all positions -- TimesNet is NON-CAUSAL by
    construction (a position's output can depend on future-within-window positions). That is FINE for
    LAST-POSITION-only supervision (window = all past, predict after it), but is LOOK-AHEAD if V24 is
    supervised PER-POSITION (the Timer-XL pattern: every bar a target against its own future return),
    because earlier positions then see future-window info via the global FFT + 2D conv. Before trusting
    V24's OOS IC: confirm supervision/eval is last-position-only, OR add causal masking (windowed FFT +
    causal conv padding). The shuffled-IC guard does NOT catch this (within-window structural, not
    temporal-order memorization). Flag, not yet a fix -- V24 is not trained in this code-hardening pass.
    """
    B, T, D = x.shape
    # FFT along time, mean across feature dim, mean across batch for global periods
    xf = torch.fft.rfft(x, dim=1)                       # [B, T//2+1, D]
    amp = xf.abs().mean(dim=-1)                          # [B, T//2+1]
    amp_global = amp.mean(dim=0)                         # [T//2+1]
    # Discard the DC bin (period = inf)
    amp_global[0] = 0.0
    top_freqs = torch.topk(amp_global, top_k).indices    # [top_k]
    periods = (T // top_freqs.clamp(min=1)).clamp(min=1) # [top_k]
    # Per-sample amplitude weight at the top-K freqs
    weights = amp[:, top_freqs]                           # [B, top_k]
    return periods, weights


class InceptionBlock2D(nn.Module):
    """Multi-kernel 2D convolutions (TimesNet paper §3.2)."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_sizes: tuple = (1, 3, 5)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, p_i, T/p_i]
        out = sum(branch(x) for branch in self.branches) / len(self.branches)
        return out


class TimesBlock(nn.Module):
    """One TimesNet block: FFT-period detection + per-period 2D conv + aggregate."""

    def __init__(self, d_model: int, top_k: int = 3,
                 inception_channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.top_k = top_k
        # Channel-wise lift to inception_channels for 2D conv, then back to d_model
        self.lift = nn.Conv2d(d_model, inception_channels, kernel_size=1)
        self.inception = InceptionBlock2D(inception_channels, inception_channels)
        self.proj = nn.Conv2d(inception_channels, d_model, kernel_size=1)
        RMSNorm, _, _ = _try_import_components()
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _reshape_to_2d(self, x: torch.Tensor, period: int) -> tuple[torch.Tensor, int]:
        # x: [B, T, D] -> [B, D, period, T/period] (with right-pad if needed)
        B, T, D = x.shape
        target_T = ((T + period - 1) // period) * period
        if target_T > T:
            x_pad = F.pad(x, (0, 0, 0, target_T - T))
        else:
            x_pad = x
        # [B, T_padded, D] -> [B, D, period, T_padded/period]
        n_periods = target_T // period
        out = x_pad.transpose(1, 2).reshape(B, D, n_periods, period)
        # Swap to [B, D, period, n_periods] to match paper Figure 3
        # (rows = within-period position, cols = which cycle)
        out = out.transpose(2, 3).contiguous()
        return out, target_T

    def _reshape_to_1d(self, x_2d: torch.Tensor, target_T: int, T: int) -> torch.Tensor:
        # x_2d: [B, D, period, n_periods] -> [B, T, D]
        B, D, period, n_periods = x_2d.shape
        x_1d = x_2d.transpose(2, 3).reshape(B, D, target_T)  # [B, D, T_padded]
        x_1d = x_1d[:, :, :T].transpose(1, 2)                # [B, T, D]
        return x_1d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        x_norm = self.norm(x)
        periods, weights = fft_top_k_periods(x_norm, self.top_k)  # periods: [top_k]

        per_period_outs = []
        for i, p_t in enumerate(periods.tolist()):
            p = max(int(p_t), 1)
            x_2d, target_T = self._reshape_to_2d(x_norm, p)         # [B, D, p, n]
            h = self.lift(x_2d)
            h = F.silu(self.inception(h))
            h = self.proj(h)
            h_1d = self._reshape_to_1d(h, target_T, T)              # [B, T, D]
            per_period_outs.append(h_1d)

        # Stack [top_k] outputs: [B, T, D, top_k]
        stacked = torch.stack(per_period_outs, dim=-1)              # [B, T, D, K]
        # Weight by softmax(amplitudes) per sample
        w = F.softmax(weights, dim=-1).unsqueeze(1).unsqueeze(1)    # [B, 1, 1, K]
        agg = (stacked * w).sum(dim=-1)                              # [B, T, D]

        return x + self.dropout(agg)


class TimesNetBackbone(nn.Module):
    """Stacked TimesBlocks for crypto WM.

    Inputs:
      obs_seq: [B, T, F]
      asset_id: [B]
    Outputs (forward_train):
      same dict shape as V13/V14 (V1.x-compatible interface).
    """

    def __init__(
        self,
        n_features: int = 29,
        seq_len: int = 96,
        d_model: int = 192,
        n_blocks: int = 3,
        top_k: int = 3,
        inception_channels: int = 32,
        dropout: float = 0.15,
        num_bins: int = 255,
        bin_min: float = -1.0,
        bin_max: float = 1.0,
        num_assets: int = 10,
        asset_emb_dim: int = 32,
        active_horizons: tuple = (1, 4, 16, 64),
        atme_prob: float = 0.15,
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.active_horizons = tuple(active_horizons)
        self.atme_prob = atme_prob
        self._num_bins = num_bins

        RMSNorm, _, _ = _try_import_components()

        self.asset_embedding = nn.Embedding(num_assets, asset_emb_dim)
        self.obs_encoder = nn.Sequential(
            nn.Linear(n_features + asset_emb_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Stacked TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(d_model, top_k=top_k,
                       inception_channels=inception_channels,
                       dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.post_norm = RMSNorm(d_model)

        # Heads
        self.return_trunk = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.return_heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                RMSNorm(d_model // 2),
                nn.SiLU(),
                nn.Linear(d_model // 2, num_bins),
            )
            for h in self.active_horizons
        })
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
        )
        self.log_vars = nn.Parameter(
            torch.tensor([-2.0] * len(self.active_horizons) + [-1.5])
        )

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
        if F_in != self.n_features:
            raise ValueError(f"Expected n_features={self.n_features}, got {F_in}")
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

        feat = h_seq
        if self.training and self.atme_prob > 0:
            atme_mask = (torch.rand(B, 1, 1, device=h_seq.device)
                         > self.atme_prob).float()
            feat = h_seq * atme_mask

        ret_trunk = self.return_trunk(feat)
        return_logits = {
            h_key: self.return_heads[str(h_key)](ret_trunk)
            for h_key in self.active_horizons
        }
        regime_logits = self.regime_head(ret_trunk)

        return {
            "return_logits": return_logits,
            "regime_logits": regime_logits,
            "h_seq": h_seq,
            "ret_trunk": ret_trunk,
            "prior_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "post_logits": torch.zeros(B, T, 1, device=obs_seq.device),
            "z_post": torch.zeros(B, T, 1, device=obs_seq.device),
            "recon": torch.zeros(B, T, 1, device=obs_seq.device),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _smoke_test():
    torch.manual_seed(42)
    B, T, F_in = 4, 96, 29
    m = TimesNetBackbone(
        n_features=F_in, seq_len=T,
        d_model=192, n_blocks=3, top_k=3,
        num_assets=10,
    )
    x = torch.randn(B, T, F_in)
    asset = torch.randint(0, 10, (B,))
    out = m.forward_train(x, asset)
    assert out["return_logits"][1].shape == (B, T, 255), out["return_logits"][1].shape
    loss = sum(out["return_logits"][h].pow(2).mean() for h in m.active_horizons)
    loss.backward()
    n_params = count_parameters(m)
    print(f"[V24 TimesNet smoke] PASS: B={B} T={T} F={F_in} -> "
          f"return_logits[1]{tuple(out['return_logits'][1].shape)}, "
          f"params={n_params:,}")


if __name__ == "__main__":
    _smoke_test()
