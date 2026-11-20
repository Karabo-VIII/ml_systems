"""Shared frontier-protocol components (round-6/7).

Round-6 introduced V25 with five first-principles components. Round-7 promotes
the universally-applicable subset to a shared module so V4/V6/V8/V11/V13/V14/
V22/V23/V24 can adopt them without code duplication.

Components exported:
  - CryptoPeriodEmbedding   : hard-coded sinusoidal embeddings for known
                              crypto cycles (8h funding / 24h UTC / 7d weekly /
                              168 bars). Replaces FFT-based discovery.
  - tail_adaptive_huber     : asymmetric Huber that upweights |target| > tail_σ
                              for crypto's heavy-tailed returns (kurtosis 5-15).
  - RegimeGate              : per-bar / per-feature 3-way bull/sideways/bear
                              distribution from a hidden representation.
  - RateBudgetVIB           : information-theoretic VIB with auto-tuned β to
                              hit a bits-per-timestep target.
  - adversarial_regime_weight: per-batch worst-quintile regime upweighting.

Each component has independent first-principles justification for crypto regime
(see memory/feedback_unconstrained_default_synthesis.md for protocol).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Reuse RMSNorm from V1.x components (single canonical implementation)
_v1_path = Path(__file__).resolve().parent.parent / "v1" / "v1_0_training"
if str(_v1_path) not in sys.path:
    sys.path.insert(0, str(_v1_path))


def _get_RMSNorm():
    try:
        from components import RMSNorm
        return RMSNorm
    except Exception:
        # Fallback if import path differs
        class _RMSNorm(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps
            def forward(self, x):
                return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return _RMSNorm


# =============================================================================
# Component 1: Crypto Period Embedding
# =============================================================================

class CryptoPeriodEmbedding(nn.Module):
    """Hard-coded sinusoidal embedding for known crypto cycles.

    For each period p in `periods` (in BARS), generates sin(2πt/p) and
    cos(2πt/p). Each period has a learnable amplitude. Concatenated across
    periods, projected to d_model.

    Default periods cover:
      8 bars   ≈ funding cycle (Binance perp: 8h at ~1h cadence)
      24 bars  ≈ daily UTC cycle
      96 bars  ≈ weekly approximation at 4h cadence
      168 bars ≈ weekly at 1h cadence

    The cycles are EXOGENOUS and KNOWN. No FFT discovery needed.

    Usage:
      pe = CryptoPeriodEmbedding(d_model=256)
      period_signal = pe(T=96, device=...)  # [T, d_model]
      # Inject by adding to per-timestep input or per-feature embedding
    """

    DEFAULT_PERIODS = (8, 24, 96, 168)

    def __init__(self, d_model: int, periods: tuple = DEFAULT_PERIODS,
                 amp_init: float = 0.1):
        super().__init__()
        self.periods = tuple(periods)
        self.d_model = d_model
        self.amplitudes = nn.Parameter(torch.full((len(self.periods),), amp_init))
        # 2 channels per period (sin/cos), projected to d_model
        self.proj = nn.Linear(2 * len(self.periods), d_model)
        RMSNorm = _get_RMSNorm()
        self.norm = RMSNorm(d_model)

    def forward(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns [T, d_model] period embedding."""
        t = torch.arange(T, device=device).float()
        feats = []
        for i, p in enumerate(self.periods):
            phase = 2 * math.pi * t / p
            feats.append(torch.sin(phase) * self.amplitudes[i])
            feats.append(torch.cos(phase) * self.amplitudes[i])
        period_feat = torch.stack(feats, dim=-1)  # [T, 2*N_periods]
        return self.norm(self.proj(period_feat))   # [T, d_model]

    def scalar_signal(self, T: int, device: torch.device) -> torch.Tensor:
        """Returns [T] scalar mean-pooled period signal for additive injection."""
        return self.forward(T, device).mean(dim=-1)


# =============================================================================
# Component 2: Tail-Adaptive Huber Loss
# =============================================================================

def tail_adaptive_huber(decoded: torch.Tensor, target: torch.Tensor,
                        delta: float = 0.5, tail_sigma: float = 2.0,
                        tail_weight: float = 2.5) -> torch.Tensor:
    """Asymmetric Huber: standard Huber + multiplicative tail upweighting.

    Crypto returns have kurtosis 5-15. Standard Huber treats all magnitudes
    symmetrically; the tails get under-weighted when averaged. This applies
    `tail_weight` × extra weight to samples with |target| > tail_sigma * std.

    Args:
        decoded: [N] predicted values
        target: [N] actual values
        delta: Huber transition point
        tail_sigma: σ-multiplier for "tail" classification
        tail_weight: multiplicative weight on tail samples
    """
    err = decoded - target
    abs_err = err.abs()
    quad = 0.5 * err.pow(2)
    lin = delta * (abs_err - 0.5 * delta)
    huber = torch.where(abs_err < delta, quad, lin)
    target_std = target.std() + 1e-6
    tail_mask = (target.abs() > tail_sigma * target_std).float()
    weights = 1.0 + (tail_weight - 1.0) * tail_mask
    return (huber * weights).mean()


# =============================================================================
# Component 3: Regime Gate
# =============================================================================

class RegimeGate(nn.Module):
    """Per-bar 3-way regime distribution (bull/sideways/bear) from a hidden rep.

    Outputs softmax over `n_regimes`. Used to gate downstream modules
    (e.g., regime-conditioned FFN in V22/V25, regime-conditioned MoE).
    """

    def __init__(self, d_model: int, hidden: int = 64, n_regimes: int = 3):
        super().__init__()
        RMSNorm = _get_RMSNorm()
        self.gate = nn.Sequential(
            nn.Linear(d_model, hidden),
            RMSNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_regimes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(h), dim=-1)


# =============================================================================
# Component 4: Rate-Budget VIB
# =============================================================================

class RateBudgetVIB(nn.Module):
    """VIB with auto-tuned β toward an information-rate target.

    Information theory: I(X; Z) ≥ -E[log q(z|x)] is the channel capacity.
    Target a fixed nats/timestep budget; β auto-tunes via Lagrangian update.

    Forward returns (feat_expanded, mu, logvar, kl).
    Trainer should call update_beta(observed_kl) after each backward step.
    """

    def __init__(self, d_model: int, z_dim: int = 32,
                 target_rate_nats: float = 4.0,
                 beta_init: float = 0.05,
                 beta_lr: float = 1e-3,
                 beta_min: float = 1e-4,
                 beta_max: float = 1.0,
                 logvar_init: float = -1.0,
                 logvar_min: float = -6.0,
                 logvar_max: float = 2.0,
                 dropout: float = 0.1):
        super().__init__()
        RMSNorm = _get_RMSNorm()
        self.z_dim = z_dim
        self.target_rate_nats = target_rate_nats
        self.beta_lr = beta_lr
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.to_mu = nn.Linear(d_model, z_dim)
        self.to_logvar = nn.Linear(d_model, z_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, logvar_init)
        self.z_expand = nn.Sequential(
            nn.Linear(z_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.register_buffer("beta_log", torch.tensor(math.log(beta_init)))

    def forward(self, h: torch.Tensor, training: bool):
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(self.logvar_min, self.logvar_max)
        if training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(mu)
        else:
            z = mu
        feat = self.z_expand(z)
        kl = (-0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())).mean()
        return feat, mu, logvar, kl

    @torch.no_grad()
    def update_beta(self, kl_observed: torch.Tensor):
        error = kl_observed.detach() - self.target_rate_nats
        new_log = self.beta_log + self.beta_lr * error
        new_log.clamp_(math.log(self.beta_min), math.log(self.beta_max))
        self.beta_log.copy_(new_log)

    def get_beta(self) -> torch.Tensor:
        return torch.exp(self.beta_log)


# =============================================================================
# Component 5: Adversarial Regime Weighting
# =============================================================================

def adversarial_regime_weight(regime_dist: torch.Tensor,
                              adversarial_weight: float = 1.5,
                              return_idx: bool = False):
    """Compute per-bar loss weights that upweight the worst-quintile regime.

    Args:
        regime_dist: [B, T, n_regimes] softmax regime distribution
        adversarial_weight: multiplicative weight on worst-regime samples
        return_idx: if True, also return the worst-regime index
    Returns:
        weights: [B, T] per-bar loss weights (1.0 + extra on worst regime)
        (optionally) worst_idx: int
    """
    regime_freq = regime_dist.mean(dim=(0, 1))  # [n_regimes]
    worst_idx = int(torch.argmin(regime_freq).item())
    weights = 1.0 + (adversarial_weight - 1.0) * regime_dist[..., worst_idx]
    if return_idx:
        return weights, worst_idx
    return weights


__all__ = [
    "CryptoPeriodEmbedding",
    "tail_adaptive_huber",
    "RegimeGate",
    "RateBudgetVIB",
    "adversarial_regime_weight",
]
