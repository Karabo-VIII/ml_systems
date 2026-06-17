"""MTP — Multi-Token Prediction sequential head (B002 R1).

DeepSeek-V3 (arXiv 2412.19437) introduced MTP: instead of independent
heads predicting the next token, the next-2 token, the next-4 token,
etc. -- predict each in a CAUSAL chain where the t+1 prediction informs
the t+2 prediction informs the t+4 prediction etc. The shared causal
chain extracts more signal from each backbone forward pass.

Our V1.x has 4 horizon heads predicting return at h ∈ {1, 4, 16, 64}
INDEPENDENTLY from the same backbone hidden state. MTP replaces this
with a sequential causal chain:

    h_seq = backbone(x)                    # (B, T, d_model)
    h_last = h_seq[:, -1, :]               # (B, d_model)
    z_1 = head_z(h_last)                   # (B, NUM_BINS)  -- raw logits at h=1
    e_1 = embed(decode(z_1))               # (B, d_emb)     -- embed of predicted return
    z_4 = head_z(h_last + e_1)             # (B, NUM_BINS)  -- conditioned on h=1 prediction
    ...

Drop-in replacement for the V1.x `return_heads` dict. Same forward
signature: takes a (B, d_model) hidden state, returns a dict
{f"h{h}": logits at horizon h}.

Compute cost: ~4× the cost of a single head's forward, but heads are
tiny (~50K params each) so the absolute cost is negligible (~0% of
backbone forward).

The CAUSAL property: each prediction at horizon h is informed by all
predictions at horizons < h. Loss can still be computed independently
per horizon (sum of TwoHot CE), but the gradient flow induces shared
structure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPHead(nn.Module):
    """Sequential causal-chain multi-horizon head.

    Replaces a `nn.ModuleDict({"h1": Linear, "h4": Linear, ...})` style
    independent head with a chain of TwoHot heads conditioned on the
    embedding of the prior horizon's predicted distribution.

    Forward returns:
        return_logits: {f"h{h}" : (B, NUM_BINS)} same shape as before

    The chain is:
        z_1 = head(h_pool)
        e_1 = z_to_embed(z_1)         # learned embedding of softmax(z_1)
        z_4 = head(h_pool + e_1)
        e_4 = z_to_embed(z_4)
        z_16 = head(h_pool + e_4)
        ...
    """

    def __init__(
        self,
        d_model: int,
        num_bins: int,
        horizons: Tuple[int, ...] = (1, 4, 16, 64),
        share_head: bool = True,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.horizons = tuple(horizons)
        self.share_head = share_head
        self.embed_dim = embed_dim or d_model

        if share_head:
            # One head reused at every step (typical MTP)
            self.head = nn.Linear(d_model, num_bins)
        else:
            # Independent heads -- variant for experiments
            self.heads = nn.ModuleDict({
                f"h{h}": nn.Linear(d_model, num_bins) for h in horizons
            })

        # Distribution embedding: turns (B, NUM_BINS) prob vector -> (B, d_model)
        # so it can be added to the hidden state for the next step.
        self.z_to_embed = nn.Sequential(
            nn.Linear(num_bins, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, d_model),
        )

    def _head_at(self, idx: int, h: int) -> nn.Module:
        if self.share_head:
            return self.head
        return self.heads[f"h{h}"]

    def forward(self, h_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Handles both pooled (B, d_model) and sequential (B, T, d_model) inputs.

        For (B, d_model)    output is dict {f"h{h}": (B, NUM_BINS)}
        For (B, T, d_model) output is dict {f"h{h}": (B, T, NUM_BINS)}

        The causal chain operates element-wise; each (b, t) position has its
        own h1 -> h4 -> h16 -> h64 chain.
        """
        out: Dict[str, torch.Tensor] = {}
        h = h_in
        for i, hi in enumerate(self.horizons):
            head = self._head_at(i, hi)
            z = head(h)                                 # last-dim NUM_BINS
            out[f"h{hi}"] = z
            if i < len(self.horizons) - 1:
                p = F.softmax(z, dim=-1)
                e = self.z_to_embed(p)                  # last-dim d_model
                h = h_in + e
        return out


def smoke():
    """Verify shapes + gradient flow on both (B, d) and (B, T, d) inputs."""
    torch.manual_seed(0)
    B, T, d, NB = 4, 96, 256, 255

    # 2D smoke (foundation use)
    h_pool = torch.randn(B, d, requires_grad=True)
    mtp = MTPHead(d_model=d, num_bins=NB, horizons=(1, 4, 16, 64))
    out = mtp(h_pool)
    for k, v in out.items():
        assert v.shape == (B, NB), f"2D bad shape {k}: {v.shape}"
    loss = sum(v.float().pow(2).mean() for v in out.values())
    loss.backward()
    z_emb_grad_sum = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in mtp.z_to_embed.parameters()
    )
    assert z_emb_grad_sum > 0.0
    print(f"[mtp] 2D PASS: keys {list(out.keys())}, z_to_embed grad sum {z_emb_grad_sum:.2f}")

    # 3D smoke (V1.x sequential use)
    mtp2 = MTPHead(d_model=d, num_bins=NB, horizons=(1, 4, 16, 64))
    h_seq = torch.randn(B, T, d, requires_grad=True)
    out2 = mtp2(h_seq)
    for k, v in out2.items():
        assert v.shape == (B, T, NB), f"3D bad shape {k}: {v.shape}"
    loss2 = sum(v.float().pow(2).mean() for v in out2.values())
    loss2.backward()
    z_emb2 = sum(
        (p.grad.abs().sum().item() if p.grad is not None else 0.0)
        for p in mtp2.z_to_embed.parameters()
    )
    assert z_emb2 > 0.0
    print(f"[mtp] 3D PASS: keys {list(out2.keys())}, z_to_embed grad sum {z_emb2:.2f}")
    print("[mtp] PASS smoke (both 2D + 3D)")


if __name__ == "__main__":
    smoke()
