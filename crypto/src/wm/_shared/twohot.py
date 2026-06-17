"""Canonical TwoHotSymlog — single source of truth for the WM cohort.

This module is the SOURCE OF TRUTH for TwoHot return prediction across every
WM version. Every version's `components.py` SHOULD import from here:

    from src.wm._shared.twohot import TwoHotSymlog

Why the consolidation: prior to 2026-05-21, every version had an inline
TwoHotSymlog. When V1.0's `decode()` was fixed to be Jensen-correct, the
fix did not propagate. The Jensen-WRONG version `symexp(E[buckets])`
under-predicts fat-tail magnitudes (systematic bias toward zero on the
tails that crypto trades). V1.1/V1.4/V1.6/V4 carried the wrong formula
forward through 6+ months of training runs. Centralizing prevents this.

Math:
  encode: y_sym = symlog(target), then two-hot bucket via floor/ceil indices
  decode: probs = softmax(logits); pred = sum(probs * symexp(buckets))
          i.e. E[symexp(x)], NOT symexp(E[x]).

The latter underestimates large returns by Jensen's inequality on the
convex symexp function. The bucket grid in [-1, 1] symlog space maps to
[-1.7, +1.7] in return space at the extreme bins — a return of 4.4% on a
day is at the edge.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


__contract__ = {
    "kind": "wm_twohot_canonical",
    "owner": "wm/_shared",
    "outputs": [],
    "invariants": [
        "single source of truth for TwoHotSymlog across the WM cohort",
        "decode computes E[symexp(buckets)], not symexp(E[buckets])",
        "BIN_MIN=-1, BIN_MAX=1, NUM_BINS=255 are the canonical defaults",
        "encode clamps targets to [-1e6, 1e6] before symlog (overflow guard)",
        "no torch.compile decorators (compat with v1.x torch.compile path)",
    ],
}


class TwoHotSymlog:
    """Converts continuous scalar targets into soft categorical buckets.

    Uses symlog transform: sign(x) * log(|x| + 1).

    Default: 255 bins, range [-1, 1] in symlog space — corresponding to
    return magnitudes up to e^1 - 1 ≈ 1.72 in original space. Raw return
    targets (per CLAUDE.md cross-version invariant `target_prefix="target_return"`).
    """

    def __init__(self, num_bins: int = 255, min_val: float = -1.0,
                 max_val: float = 1.0, device: str = "cpu"):
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.device = device
        self.buckets = torch.linspace(min_val, max_val, num_bins).to(device)
        self.width = (max_val - min_val) / (num_bins - 1)

    def to(self, device):
        """Move buckets to specified device."""
        self.device = device
        self.buckets = self.buckets.to(device)
        return self

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

    def encode(self, targets: torch.Tensor) -> tuple:
        """Encode scalar targets into two-hot bucket indices and weights.

        Args:
            targets: [...] arbitrary shape of scalar values
        Returns:
            (idx_floor, idx_ceil, weight_floor, weight_ceil) — same shape as targets
        """
        targets = torch.clamp(targets, -1e6, 1e6)
        y = self.symlog(targets)
        y = torch.clamp(y, self.min_val, self.max_val)

        idx_continuous = (y - self.min_val) / self.width
        idx_floor = idx_continuous.floor().long().clamp(0, self.num_bins - 1)
        idx_ceil = (idx_floor + 1).clamp(0, self.num_bins - 1)

        w_ceil = idx_continuous - idx_continuous.floor()
        w_floor = 1.0 - w_ceil

        return idx_floor, idx_ceil, w_floor, w_ceil

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode bucket logits back to scalar predictions.

        Computes E[symexp(x)] = sum(probs * symexp(buckets)), NOT symexp(E[x]).
        The latter underestimates large returns due to Jensen's inequality.
        """
        probs = F.softmax(logits.float(), dim=-1)
        bucket_values = self.symexp(self.buckets)
        return torch.sum(probs * bucket_values, dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     label_smoothing: float = 0.0,
                     focal_gamma: float = 0.0) -> torch.Tensor:
        """Two-hot cross-entropy with optional label smoothing and focal weighting.

        Per CLAUDE.md cross-version invariant, focal_gamma = 0.0 is canonical
        (focal up-weights temporally-clustered tails and accelerates memorization).
        """
        idx_f, idx_c, w_f, w_c = self.encode(targets)
        loss_f = F.cross_entropy(logits, idx_f, reduction="none",
                                 label_smoothing=label_smoothing)
        loss_c = F.cross_entropy(logits, idx_c, reduction="none",
                                 label_smoothing=label_smoothing)
        per_sample = w_f * loss_f + w_c * loss_c
        if focal_gamma > 0.0:
            p_t = torch.exp(-per_sample.detach())
            focal_w = (1.0 - p_t) ** focal_gamma
            per_sample = focal_w * per_sample
        return per_sample.mean()

    def compute_crps_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CRPS — strictly proper scoring rule respecting ordinal bin structure.

        Penalizes predictions proportionally to distance from truth.
        Bin 126 when truth is 128 costs less than bin 200.

        Discrete CRPS = sum_k (F_k - I[y <= b_k])^2 * delta_b
        where F_k = predicted CDF, I[y <= b_k] = Heaviside step at truth.
        Two-hot encoding gives a smooth gradient at the boundary.
        """
        probs = F.softmax(logits.float(), dim=-1)
        cdf_pred = torch.cumsum(probs, dim=-1)

        idx_f, idx_c, w_f, w_c = self.encode(targets)
        bin_indices = torch.arange(self.num_bins, device=logits.device)
        idx_f_expanded = idx_f.unsqueeze(-1)
        idx_c_expanded = idx_c.unsqueeze(-1)
        w_c_expanded = w_c.unsqueeze(-1)

        cdf_true = torch.where(
            bin_indices < idx_f_expanded,
            torch.zeros_like(cdf_pred),
            torch.where(
                bin_indices < idx_c_expanded,
                w_c_expanded,
                torch.ones_like(cdf_pred),
            ),
        )

        crps_per_sample = (cdf_pred - cdf_true).pow(2).mean(dim=-1)
        return crps_per_sample.mean()
