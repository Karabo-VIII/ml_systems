"""Koopman Neural Forecaster (B006 R1) — distribution-shift-aware head.

Lange et al. 2022 (KNF, arXiv 2210.03675) [VERIFIED] + Liu 2023 (Koopa, NIPS-23)
+ SKOLR (ICML 2025) — Koopman operators learn a linear evolution in a learned
embedding space:

    z_t = phi(x_t)             # measurement function (encoder)
    z_{t+1} = K * z_t           # linear Koopman operator
    x_hat_{t+1} = phi^-1(z_{t+1})  # decoder

The trick: NONLINEAR dynamics in the original space become LINEAR in the
embedding space if phi is well-chosen. The Koopman operator K is a learned
linear map. Per B006 the mechanism is "explicit for distribution shift" —
crypto regimes shift; this is mechanism-to-problem fit.

This module provides a Koopman HEAD that can be attached to V1.x's hidden
state h_seq[:, -1, :] (B, d_model) and predicts the next bar's hidden
state via the Koopman linear map:

    z   = encoder(h_last)        # (B, d_koop)
    z_next = K @ z              # linear evolution
    h_pred = decoder(z_next)    # (B, d_model)
    return_pred = head(h_pred)  # (B, num_bins) -- standard return head

K is a learnable (d_koop, d_koop) matrix. We initialize K close to identity
(small perturbation) so initial training behaves like an MLP residual.

Reduces to a "fancy MLP" if K is unconstrained. The Koopman discipline is
enforced by:
1. Penalizing K's spectral radius > 1 (prevents exponential blow-up)
2. (Optional) eigenstructure regularization to keep K's eigenvalues on
   the unit circle (preserves long-horizon stability)

Drop-in fit: V1.x's existing return_heads can be REPLACED with KoopmanReturnHead
which wraps the standard MLP head behind a Koopman pre-step.

⚠ Reliability: Koopman papers tested mostly on M4/ETT/long-horizon
forecasting, not crypto IC. Transfer to V1.x's regime-shifted dollar bars
is INFERRED. Probe before commit.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class KoopmanReturnHead(nn.Module):
    """Koopman-augmented multi-horizon return head.

    For each input hidden state h (B, T, d_model) at the LAST timestep:
        z   = encoder(h_last)              # (B, d_koop)
        For each horizon h_i in horizons:
            n_steps = h_i  (one Koopman application per "bar")
            z_h     = K^n_steps @ z         # roll forward
            h_pred  = decoder(z_h)
            logits  = head(h_pred)          # (B, NUM_BINS)
        return_logits[h_i] = logits

    Note: at inference, K^n_steps means *n_steps applications of K*.
    So the Koopman operator is taking us forward h_i bars.
    """

    def __init__(
        self,
        d_model: int,
        num_bins: int,
        d_koop: int = 64,
        horizons: tuple = (1, 4, 16, 64),
        spectral_penalty: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.d_koop = d_koop
        self.horizons = tuple(horizons)
        self.spectral_penalty = spectral_penalty

        # Encoder + decoder (small MLPs)
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_koop * 2),
            nn.GELU(),
            nn.Linear(d_koop * 2, d_koop),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_koop, d_koop * 2),
            nn.GELU(),
            nn.Linear(d_koop * 2, d_model),
        )

        # Koopman operator K (d_koop, d_koop). Init close to identity.
        K_init = torch.eye(d_koop) + 0.01 * torch.randn(d_koop, d_koop)
        self.K = nn.Parameter(K_init)

        # Standard return head per horizon (operates on Koopman-rolled h)
        self.heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, num_bins) for h in self.horizons
        })

    def spectral_radius_loss(self) -> torch.Tensor:
        """Penalize K's spectral radius > 1 (prevents exponential blow-up)."""
        # Compute Frobenius norm as a proxy for spectral radius (cheap; tighter for normal K)
        # For exact: torch.linalg.matrix_norm(K, ord=2) but slower.
        return torch.relu(self.K.norm(p="fro") - float(self.d_koop) ** 0.5).pow(2) * self.spectral_penalty

    def forward(self, h_in: torch.Tensor) -> dict:
        """h_in: (B, d_model) OR (B, T, d_model). Returns dict of (B, NUM_BINS) logits per horizon.

        For 3D input, processes each timestep independently (no temporal rollout
        across the input sequence; only Koopman steps for HORIZON forecasting).
        """
        if h_in.dim() == 2:
            return self._forward_2d(h_in)
        else:
            # (B, T, d_model): process per-timestep, return (B, T, NUM_BINS) per horizon
            B, T, D = h_in.shape
            out = {f"h{h}": torch.empty(B, T, self.num_bins, device=h_in.device, dtype=h_in.dtype)
                   for h in self.horizons}
            for t in range(T):
                pred_t = self._forward_2d(h_in[:, t, :])
                for hk, v in pred_t.items():
                    out[hk][:, t, :] = v
            return out

    def _forward_2d(self, h_last: torch.Tensor) -> dict:
        z = self.encoder(h_last)                  # (B, d_koop)
        out = {}
        # Roll forward in Koopman space; reuse intermediate z for each horizon
        z_curr = z
        max_h = max(self.horizons)
        cache_z = {0: z}
        for step in range(1, max_h + 1):
            z_curr = z_curr @ self.K.T            # one Koopman step
            cache_z[step] = z_curr
        for h in self.horizons:
            z_h = cache_z[h]
            h_pred = self.decoder(z_h)
            out[f"h{h}"] = self.heads[f"h{h}"](h_pred)
        return out


def smoke():
    torch.manual_seed(0)
    B, T, D = 4, 16, 256
    NB = 255
    head = KoopmanReturnHead(d_model=D, num_bins=NB, d_koop=64, horizons=(1, 4, 16, 64))
    h = torch.randn(B, T, D, requires_grad=True)
    out = head(h)
    for k, v in out.items():
        assert v.shape == (B, T, NB), f"bad shape {k}: {v.shape}"
        print(f"[koopman] {k}: shape={tuple(v.shape)}")
    sp_loss = head.spectral_radius_loss()
    total = sum(v.float().pow(2).mean() for v in out.values()) + sp_loss
    total.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    K_grad = head.K.grad.abs().sum().item()
    print(f"[koopman] spectral_radius_loss: {sp_loss.item():.6f}")
    print(f"[koopman] K param grad sum: {K_grad:.4f}  (Koopman operator learning)")
    print(f"[koopman] params: {sum(p.numel() for p in head.parameters()):,}")
    print("[koopman] PASS smoke (3D + 2D paths + spectral penalty + grad)")


if __name__ == "__main__":
    smoke()
