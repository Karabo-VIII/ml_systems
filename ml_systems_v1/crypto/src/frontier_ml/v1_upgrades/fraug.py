"""FrAug — Frequency-domain augmentation for time-series (B003 R2).

Chen et al. 2023 (arXiv 2302.09292). Approximate ~0 marginal training
cost; targets generalization specifically (lifts ShIC harder than IC by
construction since it forces invariance to small spectral perturbations).

Mechanism:
    1. Real-FFT input feature sequence over the time dimension
    2. Randomly mask K frequency components (zero them)
    3. Inverse-FFT back to time domain
    4. Pass the augmented sequence to the model

Augmentation is applied stochastically per training batch with
probability `p_aug` (default 0.5). At inference, FrAug is a no-op.

Three modes from the paper:
    - low_pass   : zero high-frequency components
    - high_pass  : zero low-frequency components
    - random     : zero random K% of components
The default is `random` because it doesn't bias the model to any specific
frequency band.

Usage:

    fraug = FrAug(mask_ratio=0.1, mode="random", p_aug=0.5)
    x_aug = fraug(x)   # x: (B, T, F); returns (B, T, F)

Compute cost: 1 FFT + 1 IFFT per batch on the time dimension. Negligible
vs the model forward pass.

Drop-in: insert AT INPUT of the V1.x trainer's data path (after RevIN if
present, before the encoder). Only applied during training (controlled
by torch.is_grad_enabled() OR an explicit flag).
"""
from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class FrAug(nn.Module):
    """Frequency-domain stochastic mask augmentation."""

    def __init__(
        self,
        mask_ratio: float = 0.10,
        mode: Literal["random", "low_pass", "high_pass"] = "random",
        p_aug: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")
        if mode not in ("random", "low_pass", "high_pass"):
            raise ValueError(f"mode must be one of random / low_pass / high_pass")
        self.mask_ratio = mask_ratio
        self.mode = mode
        self.p_aug = p_aug

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F)  -> augmented (B, T, F).

        Augmentation is stochastic: with probability (1 - p_aug) returns x
        unchanged. At eval time (model.eval() AND p_aug already 0) no-op.
        """
        if not self.training:
            return x
        if torch.rand(1).item() > self.p_aug:
            return x
        # Real-FFT along time axis (T)
        # rfft returns (B, T//2 + 1, F) complex
        Xf = torch.fft.rfft(x, dim=1)
        n_freq = Xf.size(1)
        n_mask = int(round(n_freq * self.mask_ratio))
        if n_mask <= 0:
            return x

        if self.mode == "random":
            # Per-sample independent random mask
            B = Xf.size(0)
            mask = torch.ones(B, n_freq, 1, device=Xf.device, dtype=torch.bool)
            for b in range(B):
                idx = torch.randperm(n_freq, device=Xf.device)[:n_mask]
                mask[b, idx, 0] = False
            Xf = Xf * mask
        elif self.mode == "low_pass":
            # Zero the highest-frequency n_mask components
            Xf[:, -n_mask:, :] = 0
        elif self.mode == "high_pass":
            # Zero the lowest-frequency n_mask components (keep DC + high)
            Xf[:, :n_mask, :] = 0
        # IFFT back to time domain
        T = x.size(1)
        x_aug = torch.fft.irfft(Xf, n=T, dim=1)
        return x_aug


def smoke():
    """Verify FrAug shapes + that augmented tensor differs from input under p_aug=1."""
    torch.manual_seed(0)
    B, T, F = 4, 96, 34
    x = torch.randn(B, T, F)

    aug = FrAug(mask_ratio=0.20, mode="random", p_aug=1.0)
    aug.train()
    x_a = aug(x)
    print(f"[fraug] input  shape: {tuple(x.shape)}")
    print(f"[fraug] output shape: {tuple(x_a.shape)}")
    assert x_a.shape == x.shape
    diff = (x_a - x).abs().mean().item()
    print(f"[fraug] mean abs delta: {diff:.4f}  (expect > 0)")
    assert diff > 0.0, "augmented tensor must differ from input"

    # Eval mode -> no-op
    aug.eval()
    x_b = aug(x)
    eq_diff = (x_b - x).abs().max().item()
    print(f"[fraug] eval-mode delta: {eq_diff:.6f} (expect 0)")
    assert eq_diff == 0.0, "FrAug must be no-op at eval"

    # low_pass / high_pass modes
    for mode in ("low_pass", "high_pass"):
        a2 = FrAug(mask_ratio=0.30, mode=mode, p_aug=1.0)
        a2.train()
        out = a2(x)
        d = (out - x).abs().mean().item()
        print(f"[fraug] mode={mode:10s} delta={d:.4f}")
        assert d > 0.0
    print("[fraug] PASS smoke")


if __name__ == "__main__":
    smoke()
