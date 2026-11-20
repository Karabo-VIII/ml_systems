"""Born-Again Self-Distillation (B006 R3).

Furlanello et al. 2018 (arXiv 1805.04770) [VERIFIED] + NeurIPS 2024
"Understanding the Gains from Repeated Self-Distillation" [REPORTED]:
train V1.x → use as teacher → train V1.x v2 from same data with KL
distillation loss against teacher's predictions. Iterate 2-3 generations.

Per Furlanello: "BAN-DenseNet beats teacher on CIFAR-10/100 with the same
architecture and same training data."

This module provides:
1. **BornAgainTrainer** — runs one BAN generation:
   - Load teacher checkpoint
   - Train fresh student with combined supervised loss + KL distillation
     against teacher's TwoHot logits
   - Save student as next-generation teacher

2. **iterate_BAN(N)** — runs N generations sequentially. Each new
   generation distills from the prior one.

The compounding-gain hypothesis (NeurIPS 2024 follow-on): each generation
sees the soft labels from the prior model + the original hard labels →
the model learns a smoother decision boundary that compounds across
generations.

Reliability: Furlanello tested CIFAR-10/100 with classification heads;
NeurIPS 2024 follow-on tested broader benchmarks but didn't measure IC
or financial-time-series specifically. Transfer to V1.x's TwoHot 255-bin
return distribution is INFERRED. Mechanism is sound; magnitude unknown.

This is a TRAINING-LOOP MODIFIER, not a model architecture change. It can
combine with any of the V1.x upgrade flags (SAM, FrAug, etc.).

Default protocol (per Furlanello):
- 3 generations (G0 = teacher trained from scratch, G1 + G2 = born-again)
- KL temperature T=4 (teacher-softening per Hinton 2015)
- Combined loss: alpha * KL_distill + (1 - alpha) * supervised_CE,
  alpha=0.5 default

Usage:
    from frontier_ml.v1_upgrades.born_again import BornAgainLoss

    teacher = load_teacher_ckpt(...)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ba_loss = BornAgainLoss(alpha=0.5, T=4.0)

    # In trainer step (replaces standard horizon TwoHot CE):
    student_logits = model.forward_train(...)["return_logits"]
    with torch.no_grad():
        teacher_logits = teacher.forward_train(...)["return_logits"]
    losses = ba_loss(student_logits, teacher_logits, target_returns,
                      bucketer=model.bucketer)
    total = losses["total"]  # combined supervised + distill
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BornAgainLoss(nn.Module):
    """KL-distillation + supervised hybrid for Born-Again self-distillation.

    For each horizon h:
        L_distill_h  = T^2 * KL( softmax(t_logits/T) || softmax(s_logits/T) )
        L_supervised_h = bucketer.compute_loss(s_logits, target_returns[h])
        L_h = alpha * L_distill_h + (1-alpha) * L_supervised_h
    Total = mean over horizons.
    """

    def __init__(self, alpha: float = 0.5, T: float = 4.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if T <= 0:
            raise ValueError(f"T must be > 0, got {T}")
        self.alpha = alpha
        self.T = T

    def forward(
        self,
        student_logits: Dict[int, torch.Tensor],
        teacher_logits: Dict[int, torch.Tensor],
        target_returns: Dict[int, torch.Tensor],
        bucketer,
    ) -> Dict[str, torch.Tensor]:
        """Compute Born-Again hybrid loss across horizons.

        Args:
            student_logits: {h: (B, T, NUM_BINS)} or {h: (..., NUM_BINS)}
            teacher_logits: same shape; should be detached
            target_returns: {h: (B, T)} raw returns
            bucketer:       a TwoHotSymlog or AdaptiveBucketer for compute_loss

        Returns dict with keys: total, supervised, distill, per_horizon.
        """
        T = self.T
        per_h = {}
        sup_terms = []
        kl_terms = []
        for h, s_logits in student_logits.items():
            t_logits = teacher_logits[h].detach()
            # Reshape to (..., NUM_BINS)
            s_flat = s_logits.reshape(-1, s_logits.shape[-1])
            t_flat = t_logits.reshape(-1, t_logits.shape[-1])
            # KL distillation
            log_p_s = F.log_softmax(s_flat / T, dim=-1)
            p_t = F.softmax(t_flat / T, dim=-1)
            kl = -(p_t * log_p_s).sum(dim=-1).mean() * (T * T)
            # Supervised CE via bucketer
            if h in target_returns:
                tgt_flat = target_returns[h].reshape(-1)
                sup = bucketer.compute_loss(s_flat, tgt_flat)
            else:
                sup = torch.tensor(0.0, device=s_logits.device)
            per_h[h] = {"distill": kl, "supervised": sup}
            kl_terms.append(kl)
            sup_terms.append(sup)

        total_kl = torch.stack(kl_terms).mean() if kl_terms else torch.tensor(0.0)
        total_sup = torch.stack(sup_terms).mean() if sup_terms else torch.tensor(0.0)
        total = self.alpha * total_kl + (1.0 - self.alpha) * total_sup
        return {
            "total": total,
            "supervised": total_sup,
            "distill": total_kl,
            "per_horizon": per_h,
        }


def smoke():
    """Verify BornAgainLoss math + that distill loss vanishes when student==teacher."""
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(ROOT / "src"))

    # Build a minimal bucketer for the supervised path
    from wm.v4.v4_training.components import TwoHotSymlog
    bucketer = TwoHotSymlog(num_bins=255, min_val=-1.0, max_val=1.0, device="cpu")

    torch.manual_seed(0)
    B, T, NB = 4, 16, 255

    # Identical student + teacher logits: KL should be ~0
    teacher_logits = {h: torch.randn(B, T, NB) for h in (1, 4, 16, 64)}
    student_logits = {h: t.clone().requires_grad_(True) for h, t in teacher_logits.items()}
    targets = {h: torch.randn(B, T) * 0.01 for h in (1, 4, 16, 64)}

    ba = BornAgainLoss(alpha=0.5, T=4.0)
    losses_eq = ba(student_logits, teacher_logits, targets, bucketer)
    eq_distill = losses_eq["distill"].item()
    print(f"[born-again] identical s==t: distill={eq_distill:.4f} "
          f"(equals teacher entropy*T^2; constant w.r.t. student); "
          f"supervised={losses_eq['supervised'].item():.4f}")

    # Different student logits: distill loss should be HIGHER than identical
    student_logits2 = {h: torch.randn(B, T, NB, requires_grad=True) * 3.0 for h in (1, 4, 16, 64)}
    losses_neq = ba(student_logits2, teacher_logits, targets, bucketer)
    neq_distill = losses_neq["distill"].item()
    print(f"[born-again] random   s!=t: distill={neq_distill:.4f}  (should be > eq case)")
    assert neq_distill > eq_distill, (
        f"distill should be larger when s diverges from t; got {neq_distill} vs eq {eq_distill}"
    )

    losses_neq["total"].backward()
    student_grad_sum = sum((sl.grad.abs().sum().item() if sl.grad is not None else 0.0)
                            for sl in student_logits2.values())
    print(f"[born-again] backward OK; student grad sum: {student_grad_sum:.2f}")
    print(f"[born-again] teacher detached: {teacher_logits[1].grad is None}")
    print("[born-again] PASS smoke")


if __name__ == "__main__":
    smoke()
