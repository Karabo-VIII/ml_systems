"""Distillation loss — hybrid KL + L1 + L2 per LITERATURE.md Hole 5.

Pure KL on TwoHot 255-bin teacher distributions transfers poorly when the
teacher is peaky (concentrated probability on 1-2 bins). The fix from
Phuong & Lampert 2019 (arxiv 1812.04106): hybrid loss combining KL on
softmaxed logits + L1 on the continuous expectation + L2 on variance.

L_distill = alpha * KL(student || teacher) + beta * L1(E_s - E_t) + gamma * L2(V_s - V_t)

Defaults: alpha=0.5, beta=0.4, gamma=0.1 (per Phuong & Lampert).

Per multi-horizon (h ∈ {1, 4, 16, 64}), each horizon has its own
ensemble-of-teacher logits. We compute the loss per-horizon and average.
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


def expected_return(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """E[r] = sum_i softmax(logits)_i * bin_center_i  ->  (B,)"""
    p = F.softmax(logits, dim=-1)
    return (p * bin_centers).sum(dim=-1)


def variance_return(logits: torch.Tensor, bin_centers: torch.Tensor) -> torch.Tensor:
    """V[r] = E[r^2] - E[r]^2  ->  (B,)"""
    p = F.softmax(logits, dim=-1)
    e_r = (p * bin_centers).sum(dim=-1)
    e_r2 = (p * bin_centers.pow(2)).sum(dim=-1)
    return (e_r2 - e_r.pow(2)).clamp(min=0.0)


def hybrid_distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    bin_centers: torch.Tensor,
    *,
    alpha: float = 0.5,
    beta: float = 0.4,
    gamma: float = 0.1,
    kl_temperature: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """Single-horizon hybrid distillation loss.

    Args:
        student_logits: (B, NUM_BINS)
        teacher_logits: (B, NUM_BINS)  -- ensemble-averaged
        bin_centers:    (NUM_BINS,)    -- TwoHot bin centers (same as teachers)
        alpha/beta/gamma: weight terms
        kl_temperature: softmax temperature applied to BOTH student and teacher
                        logits before KL (standard Hinton trick; T=2-4 typical)

    Returns dict with: total, kl, l1_expected, l2_var
    """
    T = kl_temperature
    # KL(teacher || student) per Hinton; the gradient of cross-entropy
    # of student wrt teacher distribution gives the right signal.
    p_t = F.softmax(teacher_logits / T, dim=-1)
    log_p_s = F.log_softmax(student_logits / T, dim=-1)
    # KL = sum(p_t * (log p_t - log p_s)) but the constant entropy of p_t
    # has zero gradient wrt student, so we use cross-entropy form:
    kl = -(p_t * log_p_s).sum(dim=-1).mean() * (T * T)

    # Expected return + variance match
    e_s = expected_return(student_logits, bin_centers)
    e_t = expected_return(teacher_logits, bin_centers)
    l1 = F.l1_loss(e_s, e_t)

    v_s = variance_return(student_logits, bin_centers)
    v_t = variance_return(teacher_logits, bin_centers)
    l2 = F.mse_loss(v_s, v_t)

    total = alpha * kl + beta * l1 + gamma * l2
    return {"total": total, "kl": kl, "l1_expected": l1, "l2_var": l2}


class HybridDistillLoss(nn.Module):
    """Multi-horizon wrapper around hybrid_distill_loss."""

    def __init__(
        self,
        bucketer: TwoHotSymlog,
        horizons=(1, 4, 16, 64),
        alpha: float = 0.5,
        beta: float = 0.4,
        gamma: float = 0.1,
        kl_temperature: float = 2.0,
    ):
        super().__init__()
        self.bucketer = bucketer
        self.horizons = tuple(horizons)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.kl_temperature = kl_temperature
        # Cache bin centers (NUM_BINS,) on bucketer device
        bin_edges = torch.linspace(bucketer.min_val, bucketer.max_val,
                                    bucketer.num_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        self.register_buffer("bin_centers", bin_centers)

    def forward(
        self,
        student_logits: Dict[str, torch.Tensor],
        teacher_logits: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        per_h = {}
        totals = []
        kls, l1s, l2s = [], [], []
        for h in self.horizons:
            hk = f"h{h}"
            d = hybrid_distill_loss(
                student_logits[hk],
                teacher_logits[hk].to(student_logits[hk].dtype),
                self.bin_centers.to(student_logits[hk].dtype),
                alpha=self.alpha, beta=self.beta, gamma=self.gamma,
                kl_temperature=self.kl_temperature,
            )
            per_h[hk] = d
            totals.append(d["total"])
            kls.append(d["kl"])
            l1s.append(d["l1_expected"])
            l2s.append(d["l2_var"])
        return {
            "total": torch.stack(totals).mean(),
            "kl": torch.stack(kls).mean(),
            "l1_expected": torch.stack(l1s).mean(),
            "l2_var": torch.stack(l2s).mean(),
            "per_horizon": per_h,
        }
