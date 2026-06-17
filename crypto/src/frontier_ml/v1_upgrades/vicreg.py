"""VICReg — Variance / Invariance / Covariance regularization (B005 R1).

Bardes et al. 2021 (arXiv 2105.04906) → C-JEPA (arXiv 2410.19560) which
specifically integrates VICReg into a JEPA framework to prevent the
EMA-target-encoder COLLAPSE that plain I-JEPA suffers from.

Three terms applied to embedding tensor z (B, D):

    L_var(z)  = mean over dim of  max(0, gamma - sqrt(Var(z) + eps))
                Penalizes per-dim std falling below threshold gamma.
                Forces the embedding distribution to MAINTAIN spread.
                Default gamma = 1.0 (per VICReg paper).

    L_inv(z_anchor, z_pred) = MSE(z_anchor, z_pred)
                Predicted-vs-target invariance term.
                The standard JEPA InfoNCE / regression loss already
                contains an invariance term; VICReg's L_inv is reused for
                consistency.

    L_cov(z)  = sum of squared off-diagonal entries of Cov(z) / D
                Decorrelates dimensions; prevents redundant features.

Total: L = lambda_inv * L_inv + lambda_var * L_var + lambda_cov * L_cov

Default weights from VICReg paper: lambda_var=25, lambda_inv=25, lambda_cov=1.

V6 (JEPA + Discriminator) failure mode (per WM_FINDINGS): ShIC declines
0.0236 -> 0.0204 -> 0.0201 mid-training. C-JEPA paper attributes this
exact pattern to "EMA-target encoder collapse." Adding VICReg's variance
+ covariance terms to V6's existing InfoNCE loss is the prescribed fix.

Usage:

    vicreg = VICReg(lambda_var=25.0, lambda_cov=1.0, gamma=1.0)
    # In V6 training step:
    z_anchor  = encoder(x_anchor)        # (B, D)
    z_pred    = predictor(z_anchor)      # (B, D)
    z_target  = ema_encoder(x_target)    # (B, D)
    extra = vicreg(z_pred, z_target, z_anchor)
    total_loss = base_jepa_loss + extra["total"]

Returns dict with components for logging.

Reference impl: github.com/facebookresearch/vicreg.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VICReg(nn.Module):
    """VICReg regularization layer."""

    def __init__(
        self,
        lambda_inv: float = 25.0,
        lambda_var: float = 25.0,
        lambda_cov: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.lambda_inv = lambda_inv
        self.lambda_var = lambda_var
        self.lambda_cov = lambda_cov
        self.gamma = gamma
        self.eps = eps

    @staticmethod
    def variance_loss(z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        # z: (B, D)
        std = torch.sqrt(z.var(dim=0) + eps)
        return F.relu(gamma - std).mean()

    @staticmethod
    def covariance_loss(z: torch.Tensor) -> torch.Tensor:
        # z: (B, D)
        B, D = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)
        cov = (z_centered.T @ z_centered) / max(1, B - 1)
        # Sum of squared off-diagonals, normalized by D
        off_diag = cov - torch.diag(torch.diag(cov))
        return (off_diag.pow(2).sum()) / D

    def forward(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        z_anchor: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """z_pred / z_target: (B, D); z_anchor optional for additional variance regularization."""
        # Invariance: predictor matches target
        l_inv = F.mse_loss(z_pred, z_target.detach())
        # Variance: applied to predictor AND (optionally) anchor
        l_var = self.variance_loss(z_pred, gamma=self.gamma, eps=self.eps)
        if z_anchor is not None:
            l_var = 0.5 * (l_var + self.variance_loss(z_anchor, gamma=self.gamma, eps=self.eps))
        # Covariance: applied to predictor (decorrelate)
        l_cov = self.covariance_loss(z_pred)
        if z_anchor is not None:
            l_cov = 0.5 * (l_cov + self.covariance_loss(z_anchor))

        total = (self.lambda_inv * l_inv
                  + self.lambda_var * l_var
                  + self.lambda_cov * l_cov)
        return {
            "total": total,
            "l_inv": l_inv,
            "l_var": l_var,
            "l_cov": l_cov,
        }


def smoke():
    """Verify VICReg components on collapsed vs spread embeddings."""
    torch.manual_seed(0)
    vic = VICReg()
    B, D = 32, 64

    # Healthy: spread, decorrelated embeddings
    z_a = torch.randn(B, D)
    z_p = z_a + 0.01 * torch.randn(B, D)
    z_t = z_a + 0.01 * torch.randn(B, D)
    out_h = vic(z_p, z_t, z_a)
    print(f"[vicreg] healthy: total={out_h['total'].item():.4f} "
          f"inv={out_h['l_inv'].item():.6f} "
          f"var={out_h['l_var'].item():.6f} "
          f"cov={out_h['l_cov'].item():.6f}")

    # Collapsed: all rows identical (bad — VICReg should signal high variance penalty)
    z_collapse = torch.zeros(B, D) + 0.001 * torch.randn(B, D)  # near-zero variance
    out_c = vic(z_collapse, z_collapse, z_collapse)
    print(f"[vicreg] collapsed: total={out_c['total'].item():.4f} "
          f"var={out_c['l_var'].item():.4f}  "
          f"(>> healthy var; signals collapse)")
    assert out_c["l_var"].item() > out_h["l_var"].item() * 5, \
        "VICReg variance term must heavily penalize collapsed embeddings"

    # High-correlation: cov term should dominate
    z_corr = torch.randn(B, 4).repeat(1, D // 4)        # repeated -> very high cov
    out_r = vic(z_corr, z_corr, z_corr)
    print(f"[vicreg] high-corr: total={out_r['total'].item():.4f} "
          f"cov={out_r['l_cov'].item():.4f}  "
          f"(>> healthy cov; signals redundancy)")
    print("[vicreg] PASS smoke")


if __name__ == "__main__":
    smoke()
