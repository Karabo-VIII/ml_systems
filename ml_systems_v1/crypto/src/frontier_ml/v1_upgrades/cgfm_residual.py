"""Conditional Guided Flow Matching residual head (B007 E3 / §3.4).

VERIFIED source: arXiv 2507.07192, "Conditional Guided Flow Matching" --
"a novel model-agnostic framework that extends flow matching by integrating
outputs from an auxiliary predictive model. This enables learning from the
probabilistic structure of prediction residuals."

Why this matters for V1.x:
    - V1.x emits a 255-bin TwoHot distribution over symlog-transformed returns.
    - Tail probability is bin-discretization-error limited.
    - CGFM is a distributional refinement on top of the point predictor:
        residual e = y - y_hat
        learn the conditional density p(e | h, y_hat)  via flow matching
    - At inference, sample r ~ p_residual, return y_hat + r as the full
      conditional distribution.

Mechanism:
    Rectified-flow training (Liu et al. 2022):
        t ~ U[0, 1]
        z ~ N(0, I)                              # base sample
        x_t = (1 - t) * z + t * residual         # linear interpolant
        v_target = residual - z                  # constant velocity
        loss = E[ || v_theta(x_t, t, condition) - v_target ||^2 ]

    Inference (Euler ODE solve, n_steps):
        z ~ N(0, I)
        x = z
        for k in range(n_steps):
            t = k / n_steps
            x = x + (1 / n_steps) * v_theta(x, t, condition)
        residual_pred = x

    Final prediction:
        y_dist = point_pred + residual_pred       # samples from p(y | h)

Architecture:
    v_theta is a small MLP: [residual_dim + 1 (time) + cond_dim] -> residual_dim.
    For scalar return (residual_dim=1): MLP-3, hidden=128, ~50K params.

Usage:
    cgfm = CGFMResidualHead(cond_dim=256+255+1, residual_dim=1, hidden=128)
    # Training (one V1.x batch):
    point_pred, h_state, bin_probs = v1_model.encode_for_cgfm(obs, asset)
    cond = torch.cat([h_state, bin_probs, point_pred.unsqueeze(-1)], dim=-1)
    loss = cgfm.train_step(target=y, point_pred=point_pred, cond=cond)
    # Inference:
    samples = cgfm.sample(point_pred=point_pred, cond=cond, n_samples=128, n_steps=20)
    crps = cgfm.crps(samples, y)
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sinusoidal_time_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal positional encoding for continuous time t in [0, 1].

    t shape (B,) -> (B, dim).
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device).float() / max(half - 1, 1)
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if emb.shape[-1] < dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class ResidualFlowMLP(nn.Module):
    """Vector-field network v_theta(x, t, c) for flow matching."""

    def __init__(
        self,
        residual_dim: int = 1,
        cond_dim: int = 512,
        hidden: int = 128,
        n_layers: int = 3,
        time_emb_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.residual_dim = residual_dim
        self.cond_dim = cond_dim
        self.time_emb_dim = time_emb_dim
        in_dim = residual_dim + time_emb_dim + cond_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.SiLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, residual_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, residual_dim); t: (B,); cond: (B, cond_dim)."""
        t_emb = _sinusoidal_time_embed(t, self.time_emb_dim)  # (B, time_emb_dim)
        z = torch.cat([x, t_emb, cond], dim=-1)
        return self.net(z)


class CGFMResidualHead(nn.Module):
    """Conditional Guided Flow Matching residual head.

    Trains a vector field over residuals (y - point_pred) conditioned on
    auxiliary signals (typically the V1.x latent + bin-prob vector + point_pred).
    """

    def __init__(
        self,
        cond_dim: int,
        residual_dim: int = 1,
        hidden: int = 128,
        n_layers: int = 3,
        time_emb_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.flow = ResidualFlowMLP(
            residual_dim=residual_dim, cond_dim=cond_dim,
            hidden=hidden, n_layers=n_layers,
            time_emb_dim=time_emb_dim, dropout=dropout,
        )
        self.residual_dim = residual_dim

    def fm_loss(
        self,
        target: torch.Tensor,           # (B, residual_dim) -- the residual r = y - y_hat
        cond: torch.Tensor,             # (B, cond_dim)
    ) -> torch.Tensor:
        """Rectified-flow training objective. Pass r (NOT y); caller subtracts y_hat."""
        if target.dim() == 1:
            target = target.unsqueeze(-1)
        B = target.shape[0]
        device = target.device
        t = torch.rand(B, device=device)                          # (B,)
        z = torch.randn_like(target)                              # base sample
        x_t = (1.0 - t.unsqueeze(-1)) * z + t.unsqueeze(-1) * target
        v_target = target - z
        v_pred = self.flow(x_t, t, cond)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,             # (B, cond_dim)
        n_samples: int = 64,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Sample residuals from the learned flow. Returns (B, n_samples)."""
        B = cond.shape[0]
        device = cond.device
        # Replicate condition across samples
        cond_rep = cond.unsqueeze(1).expand(B, n_samples, cond.shape[-1]).reshape(B * n_samples, -1)
        z = torch.randn(B * n_samples, self.residual_dim, device=device)
        x = z
        dt = 1.0 / n_steps
        for k in range(n_steps):
            t_k = torch.full((B * n_samples,), k * dt, device=device)
            v = self.flow(x, t_k, cond_rep)
            x = x + dt * v
        return x.reshape(B, n_samples, self.residual_dim).squeeze(-1)

    @torch.no_grad()
    def crps(self, samples: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Empirical CRPS of an ensemble of residual+y_hat samples vs realized y.

        samples shape: (B, n_samples) representing the conditional predictive distribution.
        y_true shape: (B,).
        Returns scalar CRPS averaged over batch.
        """
        if samples.dim() == 3:
            samples = samples.squeeze(-1)
        B, M = samples.shape
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        a = (samples - y_true.unsqueeze(-1)).abs().mean(dim=-1)             # (B,)
        # E|X - X'| via permutation pairing (unbiased estimator)
        perm = torch.randperm(M, device=samples.device)
        b = (samples - samples[:, perm]).abs().mean(dim=-1)
        return (a - 0.5 * b).mean()


def smoke():
    """Verify CGFM can capture a simple conditional residual distribution.

    Setup: y = 2*c + e, e ~ N(0, sigma(c)^2) where sigma depends on c.
    Point predictor knows the mean (2*c) but mis-specifies the variance.
    CGFM should learn the heteroscedastic residual.
    """
    torch.manual_seed(0)
    cond_dim = 4
    head = CGFMResidualHead(cond_dim=cond_dim, residual_dim=1, hidden=128, n_layers=3)
    optim = torch.optim.AdamW(head.parameters(), lr=2e-3)

    # Generate data: c ~ U[-1, 1], y = 2*c[0] + sigma(c[0]) * z, sigma in [0.1, 0.5]
    N = 8192
    c = torch.rand(N, cond_dim) * 2 - 1
    sigma = 0.1 + 0.4 * (c[:, 0].abs())  # heteroscedastic
    y = 2.0 * c[:, 0] + sigma * torch.randn(N)
    y_hat = 2.0 * c[:, 0]                # known-mean point predictor
    r = y - y_hat                         # residuals to model

    # Train CGFM on residuals
    losses = []
    for it in range(3000):
        idx = torch.randint(0, N, (512,))
        loss = head.fm_loss(target=r[idx].unsqueeze(-1), cond=c[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % 500 == 0:
            losses.append(loss.item())
    print(f"[cgfm] training losses (every 500 it): {losses}")

    # Eval: sample under known cond, compute CRPS vs Gaussian baseline.
    head.eval()
    test_idx = torch.randperm(N)[:1024]
    samples_r = head.sample(cond=c[test_idx], n_samples=64, n_steps=20)
    samples_y = y_hat[test_idx].unsqueeze(-1) + samples_r
    crps_cgfm = head.crps(samples_y, y[test_idx]).item()

    # Baseline: Gaussian samples around y_hat with global empirical std
    sigma_global = r.std().item()
    z = torch.randn(1024, 64)
    samples_y_baseline = y_hat[test_idx].unsqueeze(-1) + sigma_global * z
    crps_baseline = head.crps(samples_y_baseline, y[test_idx]).item()

    # Sanity: per-cond predictive stddev tracks |c[:, 0]|
    pred_std = samples_r.std(dim=-1)
    rho = torch.corrcoef(torch.stack([pred_std, sigma[test_idx]]))[0, 1].item()

    print(f"[cgfm] CRPS  CGFM={crps_cgfm:.4f}  Gaussian-baseline={crps_baseline:.4f}")
    print(f"[cgfm] corr(pred_std, true_sigma) = {rho:.4f}")
    assert losses[-1] < losses[0] * 0.6, f"CGFM failed to descend: {losses}"
    assert crps_cgfm < crps_baseline * 1.05, (
        f"CGFM CRPS {crps_cgfm} not competitive with Gaussian baseline {crps_baseline}"
    )
    assert rho > 0.3, f"CGFM did not capture heteroscedasticity (rho={rho})"
    print("[cgfm] PASS smoke")


if __name__ == "__main__":
    smoke()
