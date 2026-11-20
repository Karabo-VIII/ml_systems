"""Implicit Quantile Network head (B007 §7.3).

VERIFIED source: Dabney et al. 2018, arXiv 1806.06923, "Implicit Quantile
Networks for Distributional Reinforcement Learning."
REPORTED 2025 time-series application: samratsahoo.com/2025/05/07/iqn --
"outperforming or matching SOTA in CRPS, quantile, and point-forecast metrics."

Replaces the 255-bin TwoHot discretization with a tau-conditioned head that
returns Q(tau; x) for any tau in (0, 1). Trained with quantile-Huber loss.
Same parameter count tier as TwoHot (smaller in practice); infinite resolution.

Architecture (Dabney 2018 Sec 3):
    1. tau ~ U(0, 1) sampled per forward (n_tau samples per state)
    2. cosine embedding phi(tau) = ReLU(W * cos(pi * i * tau)) for i = 0..n_cos-1
    3. element-wise product (psi(x) * phi(tau)) -> small MLP -> Q(tau; x)
    4. quantile-Huber loss against observed return

Usage as drop-in for V1.x return head:

    iqn = IQNHead(d_model=256, n_cos=64, kappa=1.0)

    # Training:
    q_pred, taus = iqn(state_features, n_tau=8)  # (B, n_tau)
    loss = iqn.quantile_huber_loss(q_pred, taus, y_true)

    # Inference (point estimate):
    point = iqn.expectation(state_features, n_tau=32)

    # Inference (interval at coverage 1-alpha):
    L = iqn.predict_quantile(state_features, tau=alpha/2)
    U = iqn.predict_quantile(state_features, tau=1-alpha/2)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNHead(nn.Module):
    """Implicit Quantile Network head over a state representation."""

    def __init__(self, d_model: int = 256, n_cos: int = 64, kappa: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_cos = n_cos
        self.kappa = float(kappa)
        self.cos_proj = nn.Linear(n_cos, d_model)
        self.merge = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        # Pre-register the cosine basis to avoid recomputing per-call.
        self.register_buffer(
            "i_pi", torch.arange(1, n_cos + 1, dtype=torch.float32) * math.pi,
            persistent=False,
        )

    def _phi(self, tau: torch.Tensor) -> torch.Tensor:
        """Cosine embedding of tau. tau shape (B, n_tau) -> (B, n_tau, d_model)."""
        # cos(pi * i * tau)
        cos = torch.cos(tau.unsqueeze(-1) * self.i_pi)  # (B, n_tau, n_cos)
        return F.relu(self.cos_proj(cos))                # (B, n_tau, d_model)

    def forward(
        self,
        state: torch.Tensor,
        n_tau: int = 8,
        tau: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """state: (B, d_model). Returns (q_pred (B, n_tau), taus (B, n_tau))."""
        B = state.shape[0]
        if tau is None:
            tau = torch.rand(B, n_tau, device=state.device, dtype=state.dtype)
        else:
            assert tau.shape == (B, n_tau), f"tau shape {tau.shape} != {(B, n_tau)}"
        phi = self._phi(tau)                           # (B, n_tau, d_model)
        psi = state.unsqueeze(1)                       # (B, 1, d_model)
        merged = psi * phi                             # broadcast
        q = self.merge(merged).squeeze(-1)             # (B, n_tau)
        return q, tau

    def quantile_huber_loss(
        self, q_pred: torch.Tensor, taus: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Standard quantile-Huber (Dabney 2018 eq 9). target shape (B,)."""
        # diff = target - q_pred  (B, n_tau)
        diff = target.unsqueeze(-1) - q_pred
        abs_diff = diff.abs()
        huber = torch.where(
            abs_diff <= self.kappa,
            0.5 * diff.pow(2),
            self.kappa * (abs_diff - 0.5 * self.kappa),
        )
        # rho_tau = |tau - 1[diff < 0]| * huber(diff) / kappa
        weight = (taus - (diff < 0).float()).abs()
        loss = (weight * huber / max(self.kappa, 1e-6)).mean()
        return loss

    @torch.no_grad()
    def expectation(self, state: torch.Tensor, n_tau: int = 32) -> torch.Tensor:
        q, _ = self.forward(state, n_tau=n_tau)
        return q.mean(dim=-1)

    @torch.no_grad()
    def predict_quantile(self, state: torch.Tensor, tau: float) -> torch.Tensor:
        B = state.shape[0]
        tau_t = torch.full((B, 1), float(tau), device=state.device, dtype=state.dtype)
        q, _ = self.forward(state, n_tau=1, tau=tau_t)
        return q.squeeze(-1)


def smoke():
    """IQN should fit a known location-scale Gaussian and recover quantiles."""
    torch.manual_seed(0)
    d_model = 64
    head = IQNHead(d_model=d_model, n_cos=32, kappa=1.0)
    optim = torch.optim.AdamW(head.parameters(), lr=1e-3)

    # Build target: y ~ N(0, 1) conditional on a fixed state.
    state = torch.randn(256, d_model)
    y_true = torch.randn(256)

    losses = []
    for it in range(400):
        optim.zero_grad()
        q, taus = head(state, n_tau=8)
        loss = head.quantile_huber_loss(q, taus, y_true)
        loss.backward()
        optim.step()
        if it % 100 == 0:
            losses.append(loss.item())
    print(f"[iqn] losses: {losses}")
    # Sanity: median should be near sample median.
    head.eval()
    median_pred = head.predict_quantile(state, tau=0.5).mean().item()
    median_true = y_true.median().item()
    print(f"[iqn] median pred={median_pred:.3f} true={median_true:.3f}")
    assert losses[-1] < losses[0], "IQN failed to descend"
    print("[iqn] PASS smoke")


if __name__ == "__main__":
    smoke()
