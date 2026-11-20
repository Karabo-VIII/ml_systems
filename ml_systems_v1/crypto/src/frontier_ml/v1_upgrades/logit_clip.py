"""LogitClip: bounded-norm logits to mitigate noisy-label memorization (B007 §5.2).

REPORTED source: arXiv 2212.04055, "Mitigating Memorization of Noisy Labels by
Clipping the Model Prediction." Clamps the logit-vector L2 norm to an upper
bound; reduces over-confidence in noisy regions, slows memorization.

Direct fit for our 255-bin TwoHot classifier head and any other classification
output (e.g. regime gate logits).

Usage:

    clip = LogitClip(tau=1.0)
    logits_clipped = clip(logits)
    # Use logits_clipped for the loss (CE / TwoHot); leave inference unaffected
    # via the default behaviour (only clip when training=True).

Cost: ~0 (one norm + scale per forward pass).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LogitClip(nn.Module):
    """Scale logits down so that ||logits||_2 <= tau (per row).

    tau: maximum allowed L2 norm of the logit vector.
    apply_at_inference: if False, identity at eval time.
    eps: numerical floor.
    """

    def __init__(self, tau: float = 1.0, apply_at_inference: bool = False, eps: float = 1e-6):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = float(tau)
        self.apply_at_inference = apply_at_inference
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if (not self.training) and (not self.apply_at_inference):
            return logits
        # Per-row L2 norm; rescale rows whose norm > tau.
        flat = logits.reshape(-1, logits.shape[-1])
        norm = flat.norm(p=2, dim=-1, keepdim=True)
        scale = torch.clamp(self.tau / (norm + self.eps), max=1.0)
        clipped = flat * scale
        return clipped.reshape_as(logits)


def smoke():
    torch.manual_seed(0)
    clip = LogitClip(tau=2.0)
    clip.train()
    x = torch.randn(32, 255) * 5.0  # large norms
    y = clip(x)
    norms = y.norm(p=2, dim=-1)
    print(f"[logit_clip] post-clip norms: max={norms.max():.4f} target={clip.tau}")
    assert norms.max().item() <= clip.tau + 1e-3, "logit norm exceeds tau"

    clip.eval()
    y_eval = clip(x)
    assert torch.allclose(y_eval, x), "should be identity at eval"
    print("[logit_clip] PASS smoke")


if __name__ == "__main__":
    smoke()
