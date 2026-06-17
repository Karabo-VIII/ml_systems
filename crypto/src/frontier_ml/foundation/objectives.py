"""Pretraining objectives — causal multi-horizon TwoHot + lead-lag contrastive.

Per LITERATURE.md Hole 2 closure: drop MSM + adversarial, keep two terms.

Loss = w_horizon * sum_h L_twohot(logits_h, target_h)
     + w_contrastive * L_contrastive(emb_anchor, emb_pos, emb_neg)

Default weights: w_horizon=1.0, w_contrastive=0.1 (matching JEPA ratio
and PLAN.md proportional weighting).

Reuses TwoHotSymlog from src/wm/v4/v4_training/components.py — same encoder
as V1.x family for downstream IC-comparison validity.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import TwoHotSymlog  # noqa: E402


def make_bucketer(num_bins: int = 255, min_val: float = -1.0, max_val: float = 1.0,
                  device: str = "cuda") -> TwoHotSymlog:
    """Construct the canonical TwoHot bucketer matching V1.x."""
    bucketer = TwoHotSymlog(num_bins=num_bins, min_val=min_val, max_val=max_val, device=device)
    return bucketer.to(device)


def horizon_loss(
    return_logits: Dict[str, torch.Tensor],
    target_returns: Dict[int, torch.Tensor],
    bucketer: TwoHotSymlog,
) -> torch.Tensor:
    """Multi-horizon TwoHot causal next-token loss.

    Args:
        return_logits:  {f"h{h}" : (B, NUM_BINS)}   from FoundationBackbone
        target_returns: {h        : (B,)}           raw returns at h-step ahead
        bucketer:       TwoHotSymlog instance

    Returns:
        scalar loss = mean across horizons of TwoHot CE.
    """
    losses = []
    for hk, logits in return_logits.items():
        h = int(hk.lstrip("h"))
        targets = target_returns[h]
        l = bucketer.compute_loss(logits, targets)
        losses.append(l)
    return torch.stack(losses).mean()


def contrastive_loss(
    emb_anchor: torch.Tensor,
    emb_pos: torch.Tensor,
    emb_neg: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE-style triplet contrastive (batch-softmax with explicit hard neg).

    Computes per-sample log-softmax over [pos, batch_other_pos, hard_neg]
    where batch_other_pos are the in-batch positives of other anchors
    (treated as easy negatives -- standard NT-Xent variant).

    Args:
        emb_anchor: (B, D)
        emb_pos:    (B, D)
        emb_neg:    (B, D)  hard negative (far-time same asset as pos)
        temperature: softmax temp; lower -> sharper

    Returns:
        scalar mean InfoNCE loss.
    """
    B = emb_anchor.size(0)
    a = F.normalize(emb_anchor, dim=-1)
    p = F.normalize(emb_pos,    dim=-1)
    n = F.normalize(emb_neg,    dim=-1)

    # Score matrix: anchors vs all positives in batch (B,B) plus per-anchor hard neg
    sim_ap = (a @ p.t()) / temperature           # (B, B): row=anchor, col=pos j
    sim_an = (a * n).sum(dim=-1, keepdim=True) / temperature  # (B, 1): hard neg per anchor

    logits = torch.cat([sim_ap, sim_an], dim=1)  # (B, B+1) -- pos at diagonal, hard neg last
    labels = torch.arange(B, device=emb_anchor.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


class FoundationLoss(nn.Module):
    """Composes horizon + contrastive losses with configurable weights."""

    def __init__(
        self,
        bucketer: TwoHotSymlog,
        w_horizon: float = 1.0,
        w_contrastive: float = 0.1,
        contrastive_temp: float = 0.1,
    ):
        super().__init__()
        self.bucketer = bucketer
        self.w_horizon = w_horizon
        self.w_contrastive = w_contrastive
        self.contrastive_temp = contrastive_temp

    def forward(
        self,
        return_logits: Dict[str, torch.Tensor],
        target_returns: Dict[int, torch.Tensor],
        emb_anchor: torch.Tensor = None,
        emb_pos: torch.Tensor = None,
        emb_neg: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        l_h = horizon_loss(return_logits, target_returns, self.bucketer)

        if emb_anchor is not None and emb_pos is not None and emb_neg is not None:
            l_c = contrastive_loss(emb_anchor, emb_pos, emb_neg,
                                    temperature=self.contrastive_temp)
        else:
            l_c = torch.zeros((), device=l_h.device, dtype=l_h.dtype)

        total = self.w_horizon * l_h + self.w_contrastive * l_c
        return {"total": total, "horizon": l_h, "contrastive": l_c}
