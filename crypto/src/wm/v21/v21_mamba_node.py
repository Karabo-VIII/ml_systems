"""V21 — Mamba + Latent NODE hybrid backbone.

Per B005 §3 + MODE arXiv 2601.00920: the modern frontier for continuous-
time forecasting on irregular time intervals (dollar bars vary in time).

Design:
- Mamba-3 backbone (reuse V4's Mamba3Block) for primary temporal modeling
- A small Latent NODE module that operates on the Mamba's hidden state
  to produce a continuous-time correction proportional to bar duration
- Multi-horizon TwoHot heads matching V1.x

The NODE residual provides explicit dt-conditioning: each bar carries
its duration in seconds, and the NODE integrates the latent dynamics
over that time interval. This addresses the dollar-bar's irregularity:
a "low-vol day with one bar" should evolve the latent more than "five
high-vol bars in a minute."

This is a SCAFFOLD (forward + backward verified). Production training
requires per-bar duration features in chimera_legacy (already there as
norm_bar_duration).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import Mamba3Block, RMSNorm, TwoHotSymlog  # noqa: E402


class LatentNODEBlock(nn.Module):
    """Small Latent Neural ODE: dx/dt = f(x) parameterized by 2-layer MLP.

    Solves x(t+dt) = x(t) + integral_0^dt f(x) ds
    Approximation: Euler step with `n_steps` substeps for stability.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, n_steps: int = 4):
        super().__init__()
        self.n_steps = n_steps
        self.f = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model),
        )
        self.norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model); dt: (B, T) bar durations.

        Returns x updated with NODE-correction over each bar's duration.
        """
        # Euler integration with n_steps substeps over each bar
        # Note: dt is per-bar; we treat dt as a time-rescaling factor on f.
        x_norm = self.norm(x)
        # Reshape dt for broadcasting: (B, T, 1)
        dt_b = dt.unsqueeze(-1)
        sub_dt = dt_b / max(1, self.n_steps)
        x_curr = x_norm
        for _ in range(self.n_steps):
            dx = self.f(x_curr)
            x_curr = x_curr + sub_dt * dx
        return x + (x_curr - x_norm)  # residual: NODE correction


class V21Backbone(nn.Module):
    """V21 = Mamba-3 backbone + Latent NODE residual + multi-horizon heads."""

    def __init__(
        self,
        n_features: int = 34,
        d_model: int = 256,
        d_state: int = 32,
        n_mamba_layers: int = 6,
        node_hidden: int = 64,
        node_n_steps: int = 4,
        dropout: float = 0.1,
        num_bins: int = 255,
        horizons: tuple = (1, 4, 16, 64),
    ):
        super().__init__()
        self.d_model = d_model
        self.horizons = tuple(horizons)
        self.num_bins = num_bins

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.mamba_layers = nn.ModuleList([
            Mamba3Block(d_model=d_model, d_state=d_state, expand=2, headdim=64,
                          chunk_size=16, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])
        self.mamba_norm = RMSNorm(d_model)

        # Single NODE block at the end of the stack (residual correction)
        self.node = LatentNODEBlock(d_model, hidden_dim=node_hidden,
                                      n_steps=node_n_steps)
        self.norm_out = RMSNorm(d_model)

        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, num_bins) for h in self.horizons
        })

        self._use_mtp = False
        self.mtp_head = None
        self._use_mdn = False
        self.bucketer = TwoHotSymlog(num_bins, -1.0, 1.0,
                                     "cuda" if torch.cuda.is_available() else "cpu")

    def forward_train(
        self,
        obs_seq: torch.Tensor,
        asset_id: torch.Tensor = None,
        bar_durations: torch.Tensor = None,
    ) -> dict:
        """obs_seq: (B, T, F); bar_durations: (B, T) seconds per bar (optional).

        If bar_durations is None, treat each bar as unit duration (1.0).
        Otherwise pass the actual seconds-per-bar to the NODE block.
        """
        h = self.input_proj(obs_seq)
        for block in self.mamba_layers:
            h = block(h)
        h = self.mamba_norm(h)

        if bar_durations is None:
            dt = torch.ones(obs_seq.shape[0], obs_seq.shape[1],
                            device=obs_seq.device, dtype=h.dtype)
        else:
            dt = bar_durations.to(h.dtype)
        h = self.node(h, dt)
        h = self.norm_out(h)
        h_pool = h[:, -1, :]

        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(h_pool)
            for hi in self.horizons:
                return_logits[hi] = mtp_out[f"h{hi}"]
        else:
            for hi in self.horizons:
                return_logits[hi] = self.return_heads[f"h{hi}"](h_pool)

        return {
            "return_logits": return_logits,
            "h_pool": h_pool,
            "ret_trunk": h_pool,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke():
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if DEV == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30)
    model = V21Backbone(n_features=34, d_model=256, n_mamba_layers=4).to(DEV)
    print(f"[v21-mamba-node] params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    B, T = 4, 96
    obs = torch.randn(B, T, 34, device=DEV)
    bar_dur = torch.rand(B, T, device=DEV) * 0.5 + 0.5  # 0.5 - 1.0 sec equivalent
    out = model.forward_train(obs, bar_durations=bar_dur)
    for k, v in out["return_logits"].items():
        assert v.shape == (B, model.num_bins)
    print(f"[v21-mamba-node] return_logits + bar-dur conditioning OK")
    loss = sum(v.float().pow(2).mean() for v in out["return_logits"].values())
    loss.backward()
    print("[v21-mamba-node] backward OK")
    print("[v21-mamba-node] PASS smoke")


if __name__ == "__main__":
    smoke()
