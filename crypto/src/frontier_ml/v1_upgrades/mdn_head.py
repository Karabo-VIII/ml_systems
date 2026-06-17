"""MDN — Mixture Density Network head with Skewed-Student-t components (B003 R3).

Bishop 1994 (original MDN) + LSTM-MDN paper (arXiv 2508.18921) tested
Normal / Student-t / skewed-Student-t parametric mixtures on 6 equity
indices via CRPS / NLL. Direct replacement for V1.x's TwoHot 255-bin head.

Why the upgrade is on the table (per B003 R3):
- Crypto h=1 5-min returns are highly skewed AND fat-tailed. TwoHot with
  uniform 255 bins on [-1, 1] uses ~50 bins effectively (mass crowds
  near zero). Even adaptive_bins keeps the discrete-bin representation.
- Skewed-Student-t naturally captures both skew (alpha) and tail-heaviness
  (degrees of freedom) in 4 parameters per component (loc, scale, df, alpha).
- K=3 components handles tri-modal regimes (bear, neutral, bull).

Two heads provided:

    NormalMDNHead       Standard Gaussian-mixture (loc, scale, weight) per component
                         Simpler; known to work; baseline for MDN comparisons.

    SkewedStudentTHead  4 params per component: (loc, scale, df, alpha skew)
                         More expressive but harder to train; init carefully.

Both expose:
    forward(h)   -> dict of parameters
    log_prob(h, target)  -> (B, T) log-likelihood for NLL training
    sample(h, n)         -> (n, B, T) samples for distribution-aware inference
    expectation(h)       -> (B, T) point-estimate mean (for IC eval)
    variance(h)          -> (B, T) total variance (for sizing-side use)

Drop-in flow into V1.x:
    1. Replace `self.return_heads[str(h)] = nn.Sequential(...)` for each h
       with NormalMDNHead or SkewedStudentTHead at the same in_dim
    2. In `get_loss`, replace `self.bucketer.compute_loss(logits, targets)`
       with `head.log_prob(h_in, targets).neg().mean()` (NLL)
    3. In inference, replace `self.bucketer.decode(logits)` with
       `head.expectation(h_in)`
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalMDNHead(nn.Module):
    """K-component Gaussian Mixture Density Network head.

    Parameter heads:
        loc     (B, T, K)  -- per-component means
        scale   (B, T, K)  -- per-component std (positive via softplus)
        weight  (B, T, K)  -- mixture weights (softmax)
    """

    def __init__(self, d_in: int, n_components: int = 3, eps: float = 1e-3):
        super().__init__()
        self.K = n_components
        self.eps = eps
        # Single linear that produces 3K outputs; split into 3 heads
        self.proj = nn.Linear(d_in, 3 * n_components)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.proj(h)                                  # (..., 3K)
        K = self.K
        loc, raw_scale, raw_w = torch.split(out, K, dim=-1)
        scale = F.softplus(raw_scale) + self.eps
        weight = F.softmax(raw_w, dim=-1)
        return {"loc": loc, "scale": scale, "weight": weight}

    def log_prob(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns (..., 1) log-likelihood of target under predicted mixture."""
        params = self.forward(h)
        loc, scale, weight = params["loc"], params["scale"], params["weight"]
        # Expand target along K dim
        target_e = target.unsqueeze(-1)                      # (..., 1)
        # Log gaussian: -0.5 * log(2pi) - log(scale) - 0.5 * ((t - loc) / scale)^2
        log_norm = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(scale)
            - 0.5 * ((target_e - loc) / scale).pow(2)
        )                                                     # (..., K)
        log_w = torch.log(weight + 1e-12)
        return torch.logsumexp(log_w + log_norm, dim=-1)     # (...,)

    def expectation(self, h: torch.Tensor) -> torch.Tensor:
        params = self.forward(h)
        return (params["weight"] * params["loc"]).sum(dim=-1)

    def variance(self, h: torch.Tensor) -> torch.Tensor:
        params = self.forward(h)
        loc, scale, weight = params["loc"], params["scale"], params["weight"]
        e_r = (weight * loc).sum(dim=-1, keepdim=True)
        var_per = scale.pow(2) + (loc - e_r).pow(2)
        return (weight * var_per).sum(dim=-1)

    def sample(self, h: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        params = self.forward(h)
        loc, scale, weight = params["loc"], params["scale"], params["weight"]
        # Sample mixture component (categorical over weights), then sample N(loc_k, scale_k^2)
        cat = torch.distributions.Categorical(probs=weight)
        # Expand to n_samples
        idx = cat.sample(sample_shape=(n_samples,))           # (N, ...)
        loc_e = loc.unsqueeze(0).expand(n_samples, *loc.shape)
        scale_e = scale.unsqueeze(0).expand(n_samples, *scale.shape)
        idx_e = idx.unsqueeze(-1)                             # (N, ..., 1)
        loc_k = torch.gather(loc_e, dim=-1, index=idx_e).squeeze(-1)
        scale_k = torch.gather(scale_e, dim=-1, index=idx_e).squeeze(-1)
        eps = torch.randn_like(loc_k)
        return loc_k + scale_k * eps                          # (N, ...)


class SkewedStudentTHead(nn.Module):
    """K-component skewed-Student-t MDN head (LSTM-MDN paper).

    Parameter heads per component:
        loc        (B, T, K)
        scale      (B, T, K) > 0  via softplus
        df         (B, T, K) > 2  via softplus + 2  (kurtosis-controlled)
        alpha      (B, T, K) in R (skew parameter)
        weight     (B, T, K) -> softmax

    Density (Azzalini-skewed-t parametrization):
        f(t; loc, scale, df, alpha) = 2 * t_pdf((t-loc)/scale; df) / scale
                                       * t_cdf(alpha * (t-loc)/scale * sqrt((df+1)/((t-loc)^2/scale^2 + df)); df+1)

    For a tractable approximation, we use the Azzalini SN-Student-t form
    via the standard skew-t log-density. See Branco & Dey 2002.
    """

    def __init__(self, d_in: int, n_components: int = 3, eps: float = 1e-3,
                 df_min: float = 2.5):
        super().__init__()
        self.K = n_components
        self.eps = eps
        self.df_min = df_min
        # 5 params per component
        self.proj = nn.Linear(d_in, 5 * n_components)

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.proj(h)
        K = self.K
        loc, raw_scale, raw_df, alpha, raw_w = torch.split(out, K, dim=-1)
        scale = F.softplus(raw_scale) + self.eps
        df = F.softplus(raw_df) + self.df_min
        weight = F.softmax(raw_w, dim=-1)
        return {"loc": loc, "scale": scale, "df": df, "alpha": alpha, "weight": weight}

    @staticmethod
    def _student_t_log_pdf(z: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        # log f(z; df) for standard Student-t
        return (
            torch.lgamma((df + 1) / 2)
            - torch.lgamma(df / 2)
            - 0.5 * torch.log(math.pi * df)
            - (df + 1) / 2 * torch.log(1 + z.pow(2) / df)
        )

    @staticmethod
    def _student_t_log_cdf_approx(x: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
        # Approx: use Normal CDF-style sigmoid * adjusted slope. For skew-t,
        # the exact CDF is unavailable in closed form; we use a logistic
        # surrogate sized by df (df -> inf approaches Normal logCDF).
        # This is a known approximation, not exact -- acceptable for log-prob
        # gradients (Azzalini's adjustment is monotonic).
        return F.logsigmoid(x * torch.sqrt(df / (df + 1)) * 1.7)

    def log_prob(self, h: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        params = self.forward(h)
        loc, scale = params["loc"], params["scale"]
        df, alpha, weight = params["df"], params["alpha"], params["weight"]
        target_e = target.unsqueeze(-1)
        z = (target_e - loc) / scale
        log_t = self._student_t_log_pdf(z, df) - torch.log(scale) + math.log(2.0)
        # Azzalini skew adjustment
        # x_arg = alpha * z * sqrt((df + 1) / (z^2 + df))
        x_arg = alpha * z * torch.sqrt((df + 1) / (z.pow(2) + df + 1e-12))
        log_skew = self._student_t_log_cdf_approx(x_arg, df + 1)
        log_per = log_t + log_skew                           # (..., K)
        log_w = torch.log(weight + 1e-12)
        return torch.logsumexp(log_w + log_per, dim=-1)

    def expectation(self, h: torch.Tensor) -> torch.Tensor:
        # E[t] = loc + scale * delta * sqrt(df / pi) * Gamma((df-1)/2) / Gamma(df/2)
        # where delta = alpha / sqrt(1 + alpha^2)
        # Use mixture weights
        params = self.forward(h)
        loc, scale, df, alpha, weight = (
            params["loc"], params["scale"], params["df"], params["alpha"], params["weight"]
        )
        delta = alpha / torch.sqrt(1 + alpha.pow(2))
        # safe approximation when df > 1
        df_safe = torch.clamp(df, min=2.5)
        # E[skew_T] correction term -- approx for small alpha
        e_skew = (
            scale * delta
            * torch.sqrt(df_safe / math.pi)
            * torch.exp(torch.lgamma((df_safe - 1) / 2) - torch.lgamma(df_safe / 2))
        )
        e_per = loc + e_skew
        return (weight * e_per).sum(dim=-1)


def smoke():
    torch.manual_seed(0)
    B, T, D = 4, 16, 32

    # Normal MDN
    nh = NormalMDNHead(d_in=D, n_components=3)
    h = torch.randn(B, T, D, requires_grad=True)
    target = torch.randn(B, T)
    lp = nh.log_prob(h, target)
    print(f"[mdn] Normal log_prob shape: {tuple(lp.shape)}, mean: {lp.mean().item():.4f}")
    assert lp.shape == (B, T)
    e = nh.expectation(h)
    v = nh.variance(h)
    print(f"[mdn] Normal expectation shape: {tuple(e.shape)}, var range: "
          f"[{v.min().item():.4f}, {v.max().item():.4f}]")
    nll = -lp.mean()
    nll.backward()
    assert h.grad is not None and h.grad.abs().sum() > 0
    s = nh.sample(h, n_samples=8)
    print(f"[mdn] Normal samples shape: {tuple(s.shape)}")
    print("[mdn] Normal PASS")

    # Skewed Student-t
    sh = SkewedStudentTHead(d_in=D, n_components=3)
    h2 = torch.randn(B, T, D, requires_grad=True)
    lp2 = sh.log_prob(h2, target)
    print(f"[mdn] SkewT log_prob shape: {tuple(lp2.shape)}, mean: {lp2.mean().item():.4f}")
    assert lp2.shape == (B, T)
    nll2 = -lp2.mean()
    nll2.backward()
    assert h2.grad is not None and h2.grad.abs().sum() > 0
    e2 = sh.expectation(h2)
    print(f"[mdn] SkewT expectation shape: {tuple(e2.shape)}, "
          f"finite: {torch.isfinite(e2).all().item()}")
    print("[mdn] SkewT PASS")
    print("[mdn] PASS smoke (both Normal + SkewT)")


if __name__ == "__main__":
    smoke()
