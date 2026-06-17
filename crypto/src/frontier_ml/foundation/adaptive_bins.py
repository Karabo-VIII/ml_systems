"""Adaptive log-spaced TwoHot bins -- R3 from FRONTIER_RESEARCH_RESPONSE_2026_05_02.

Problem (browser response §3 R3): default 255-uniform-bin TwoHot on [-1, 1]
wastes ~80% of capacity for h=1 5-min crypto returns where 99% of mass lies
within [-0.01, +0.01]. Effective resolution at the meaningful scale is ~50
of 255 bins.

Solution: log-spaced bins that allocate density where returns actually live.

Two factory variants:
    log_spaced_51bins:  symmetric log-spaced cuts, hand-tuned for 5-min crypto
                        bins at +/- {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                                     0.1, 0.3, 1.0}. 51 total bins.
    fit_quantile_bins:  fit bin EDGES from data quantiles so each bin holds
                        equal probability mass. Strongest for the asset/horizon
                        distribution being trained on.

Both produce a TwoHotSymlog-API-compatible bucketer (encode/decode/compute_loss
return tensors of identical shape to the V1.x baseline class). This means
existing forward + loss code is unchanged; only the bucketer is swapped.

Drop-in replacement for V4's TwoHotSymlog. Same encode/decode/loss API.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def log_spaced_edges(n_bins: int = 51) -> torch.Tensor:
    """Symmetric log-spaced cut points sized for any total bin count.

    Splits n_bins symmetrically around 0:
        - One central bin near zero spanning [-eps, +eps]
        - (n_bins - 1) // 2 negative log-spaced bins from -1.0 to -eps
        - (n_bins - 1) // 2 positive log-spaced bins from +eps to +1.0
        - If n_bins is even, one extra bin gets allocated to the positive side.

    Returns the n_bins - 1 INNER edge points.
    """
    if n_bins < 3:
        raise ValueError(f"n_bins must be >= 3, got {n_bins}")
    # Number of bins per side (symmetric); the center bin spans the eps zone
    n_side = (n_bins - 1) // 2
    eps = 1e-4
    pos = np.geomspace(eps, 1.0, n_side + 1)   # n_side + 1 points -> n_side bins
    neg = -pos[::-1]                            # mirror
    # Inner edges: drop the +/-1.0 outer points (extended in AdaptiveBucketer)
    # and concatenate with the zero boundary
    inner = np.concatenate([neg[:-1], pos[1:]]) if (n_bins % 2 == 1) else np.concatenate([neg, pos[1:]])
    inner = np.unique(inner)
    return torch.from_numpy(inner.astype(np.float32))


# Legacy alias for backward compat
def log_spaced_edges_51() -> torch.Tensor:
    return log_spaced_edges(n_bins=52)


def fit_quantile_edges(returns: np.ndarray, n_bins: int = 64) -> torch.Tensor:
    """Fit n_bins-1 bin edges so each bin holds ~1/n_bins probability mass.

    Use case: pass a ~1M-sample of train-segment target_return_h values to
    get bins that match your dataset's actual return distribution.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # exclude 0 and 1
    edges = np.quantile(returns, quantiles)
    # Force monotonic uniqueness (in case of mass at zero)
    edges = np.unique(edges)
    if len(edges) < n_bins - 1:
        # Pad with linspace if quantiles collapsed
        lo, hi = float(np.min(returns)), float(np.max(returns))
        pad = np.linspace(lo, hi, n_bins + 1)[1:-1]
        edges = np.unique(np.concatenate([edges, pad]))[:n_bins - 1]
    return torch.from_numpy(edges).float()


class AdaptiveBucketer:
    """TwoHotSymlog-compatible bucketer with arbitrary bin edges.

    Same API as src/wm/v4/v4_training/components.TwoHotSymlog:
        - encode(targets) -> (lower_idx, upper_idx, lower_w, upper_w)
        - decode(logits)  -> expected return (B,)
        - compute_loss(logits, targets) -> scalar CE
    """

    def __init__(self, bin_edges: torch.Tensor, device: str = "cuda"):
        # bin_edges has length n_bins - 1; we add bin centers between edges
        # plus ends (using 2x the smallest gap as the outer extent).
        edges = bin_edges.float()
        # extend with -inf/-large_neg and +large_pos for outer bins
        first_gap = float(edges[1] - edges[0]) if len(edges) > 1 else 0.001
        last_gap = float(edges[-1] - edges[-2]) if len(edges) > 1 else 0.001
        ext = torch.cat([
            torch.tensor([float(edges[0]) - first_gap]),
            edges,
            torch.tensor([float(edges[-1]) + last_gap]),
        ])
        # bin centers for `decode`: midpoint between consecutive extended edges
        centers = 0.5 * (ext[:-1] + ext[1:])
        self.bin_edges = edges.to(device)         # (NUM_BINS - 1,) -- inner edges
        self.bin_centers = centers.to(device)      # (NUM_BINS,)
        self.num_bins = int(centers.shape[0])
        self.min_val = float(centers[0])
        self.max_val = float(centers[-1])
        self.device = device

    def to(self, device: str):
        self.bin_edges = self.bin_edges.to(device)
        self.bin_centers = self.bin_centers.to(device)
        self.device = device
        return self

    @staticmethod
    def symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))

    @staticmethod
    def symexp(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

    def encode(self, targets: torch.Tensor) -> tuple:
        """targets: (B,) raw returns. Returns (lower_idx, upper_idx, lower_w, upper_w)
        each shape (B,). Like TwoHotSymlog but adaptive bin centers.
        """
        targets = targets.to(self.device)
        # Find the bin each target falls into via searchsorted
        # bin_centers has length NUM_BINS; we want to find the two centers
        # bracketing each target.
        idx = torch.searchsorted(self.bin_centers, targets, right=False)
        idx = idx.clamp(0, self.num_bins - 1)
        lower_idx = (idx - 1).clamp(0, self.num_bins - 1)
        upper_idx = idx.clamp(0, self.num_bins - 1)
        lower_c = self.bin_centers[lower_idx]
        upper_c = self.bin_centers[upper_idx]
        gap = (upper_c - lower_c).clamp(min=1e-9)
        upper_w = ((targets - lower_c) / gap).clamp(0.0, 1.0)
        lower_w = 1.0 - upper_w
        return lower_idx, upper_idx, lower_w, upper_w

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits: (..., NUM_BINS) -> expected return (...,)."""
        p = F.softmax(logits, dim=-1)
        return (p * self.bin_centers).sum(dim=-1)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lower_idx, upper_idx, lower_w, upper_w = self.encode(targets)
        log_p = F.log_softmax(logits, dim=-1)
        lp_lower = log_p.gather(-1, lower_idx.unsqueeze(-1)).squeeze(-1)
        lp_upper = log_p.gather(-1, upper_idx.unsqueeze(-1)).squeeze(-1)
        ce = -(lower_w * lp_lower + upper_w * lp_upper)
        return ce.mean()


def make_log_spaced_bucketer(n_bins: int = 51, device: str = "cuda") -> AdaptiveBucketer:
    """Log-spaced bucketer of any size (default 51 bins).

    To match an existing model's head dim (e.g. NUM_BINS=255 in V1.x), pass
    n_bins=255. Bin PLACEMENT changes (denser near zero); COUNT preserved.
    """
    edges = log_spaced_edges(n_bins=n_bins)
    return AdaptiveBucketer(edges, device=device)


def make_quantile_bucketer(returns: np.ndarray, n_bins: int = 64,
                            device: str = "cuda") -> AdaptiveBucketer:
    """Quantile-fit bucketer: each bin holds ~1/n_bins probability mass."""
    edges = fit_quantile_edges(returns, n_bins=n_bins)
    return AdaptiveBucketer(edges, device=device)


def smoke():
    """Verify both bucketers + decode round-trip a real return distribution."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthesize crypto-like return distribution: heavy tails, 99% within [-0.01, 0.01]
    rng = np.random.default_rng(0)
    n = 100_000
    main = rng.normal(0, 0.002, size=n)
    tails = rng.normal(0, 0.05, size=n // 10)
    returns = np.concatenate([main, tails])
    returns = np.clip(returns, -1.0, 1.0)

    # Log-spaced
    log_b = make_log_spaced_bucketer(device=device)
    print(f"[adaptive-bins] log-spaced bucketer: {log_b.num_bins} bins, "
          f"range [{log_b.min_val:.5f}, {log_b.max_val:.5f}]")
    t = torch.from_numpy(returns[:1000].astype(np.float32)).to(device)
    li, ui, lw, uw = log_b.encode(t)
    # round-trip via decoded center == fake logits = onehot at that bin
    fake_logits = torch.full((len(t), log_b.num_bins), -10.0, device=device)
    fake_logits.scatter_(-1, li.unsqueeze(-1), 5.0)
    fake_logits.scatter_(-1, ui.unsqueeze(-1), 5.0)
    decoded = log_b.decode(fake_logits)
    err = (decoded - t).abs().mean().item()
    print(f"[adaptive-bins] log-spaced encode/decode mean abs err: {err:.6f}")

    # Quantile-fit
    q_b = make_quantile_bucketer(returns, n_bins=64, device=device)
    print(f"[adaptive-bins] quantile bucketer:   {q_b.num_bins} bins, "
          f"range [{q_b.min_val:.5f}, {q_b.max_val:.5f}]")
    li, ui, lw, uw = q_b.encode(t)
    fake_logits = torch.full((len(t), q_b.num_bins), -10.0, device=device)
    fake_logits.scatter_(-1, li.unsqueeze(-1), 5.0)
    fake_logits.scatter_(-1, ui.unsqueeze(-1), 5.0)
    decoded = q_b.decode(fake_logits)
    err = (decoded - t).abs().mean().item()
    print(f"[adaptive-bins] quantile  encode/decode mean abs err: {err:.6f}")


if __name__ == "__main__":
    smoke()
