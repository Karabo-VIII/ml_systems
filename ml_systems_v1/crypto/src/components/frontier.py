"""
Frontier Components -- SOTA Ceiling-Breaking Modules (v2, 2026-04-13)
======================================================================

Shared modules used by all --frontier model architectures.
Each addresses a specific limitation of the current system.

1. CurriculumWeighter: loss-conditioned sample difficulty (CRUCIAL-style)
2. ContrastiveRegimeLoss: Prototype-based regime contrast (ProtoNCE-style)
3. TradeWorthinessHead: cost-asymmetric 3-class prediction (UP/FLAT/DOWN)
4. VolatilityHead: QLIKE loss + forward vol prediction

SOTA upgrades (v2, based on research protocol 2026-04-13):
- CurriculumWeighter: |target| magnitude -> loss-conditioned difficulty (arXiv:2312.15853)
- ContrastiveRegimeLoss: SimCLR InfoNCE -> Prototype-based ProtoNCE (arXiv:2005.04966)
- TradeWorthinessHead: BCE -> asymmetric cost-sensitive CE (NIST 2024, c_FP=3*c_FN)
- VolatilityHead: MSE -> QLIKE loss (DeepVol, PMC:11473055), vectorized realized vol

Removed (dead code in v1):
- AdaptiveHorizonRouter (never wired into mixin)
- nuanced_shuffle (evaluation function, not training component)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


# =============================================================================
# 1. Curriculum Target Weighting (CRUCIAL-style, loss-conditioned)
# =============================================================================

class CurriculumWeighter:
    """Weight training examples by prediction difficulty, not label magnitude.

    v1 (naive): weight = 1 + 10 * |target|. Upweights large returns regardless
    of whether the model struggles to predict them.

    v2 (CRUCIAL-style, arXiv:2312.15853): weight = 1 + scale * per_sample_loss.
    Upweights samples the model currently gets WRONG. As the model improves on
    easy samples, they get downweighted and hard samples dominate.

    The caller computes per-sample loss externally and passes it here.
    """

    def __init__(self, scale: float = 5.0, min_weight: float = 1.0, max_weight: float = 10.0):
        self.scale = scale
        self.min_weight = min_weight
        self.max_weight = max_weight

    def __call__(self, per_sample_loss: torch.Tensor) -> torch.Tensor:
        """per_sample_loss: [B, T] or [N] -> [B, T] or [N] weights"""
        weights = self.min_weight + self.scale * per_sample_loss.detach()
        return weights.clamp(max=self.max_weight)


# =============================================================================
# 2. Contrastive Regime Loss (ProtoNCE-style, prototype-based)
# =============================================================================

class ContrastiveRegimeLoss(nn.Module):
    """Prototype-based contrastive loss for regime-aware representations.

    v1 (SimCLR): pairwise InfoNCE on batch samples. Problems: false negatives
    (same-regime samples treated as negatives), O(N^2) pair construction.

    v2 (ProtoNCE-style, arXiv:2005.04966): maintain per-regime prototypes
    (running mean of embeddings). Each sample is pushed toward its regime
    prototype and away from other prototypes. No false negatives possible
    because contrast is against prototypes, not batch members.

    3 regime prototypes: bear (0), neutral (1), bull (2).
    """

    def __init__(self, temperature: float = 0.1, momentum: float = 0.99):
        super().__init__()
        self.temperature = temperature
        self.momentum = momentum
        # Prototypes initialized lazily (need d_model from first forward)
        self.prototypes = None

    def _init_prototypes(self, d_model: int, device: torch.device):
        """Initialize 3 regime prototypes on first call."""
        self.prototypes = torch.randn(3, d_model, device=device)
        self.prototypes = F.normalize(self.prototypes, dim=-1)

    @torch.no_grad()
    def _update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """EMA update prototypes from batch embeddings."""
        emb_norm = F.normalize(embeddings, dim=-1)
        for r in range(3):
            mask = (labels == r)
            if mask.any():
                cluster_mean = emb_norm[mask].mean(dim=0)
                cluster_mean = F.normalize(cluster_mean, dim=0)
                self.prototypes[r] = (
                    self.momentum * self.prototypes[r]
                    + (1 - self.momentum) * cluster_mean
                )
                self.prototypes[r] = F.normalize(self.prototypes[r], dim=0)

    def forward(self, h_seq: torch.Tensor, regime_labels: torch.Tensor) -> torch.Tensor:
        """
        h_seq: [B, T, D], regime_labels: [B, T] in {0, 1, 2}
        Returns: scalar ProtoNCE loss
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        if self.prototypes is None:
            self._init_prototypes(D, device)
        elif self.prototypes.device != device:
            self.prototypes = self.prototypes.to(device)

        # Subsample for efficiency (every 4th bar)
        stride = 4
        h_sub = h_seq[:, ::stride, :].reshape(-1, D)        # [N, D]
        labels_sub = regime_labels[:, ::stride].reshape(-1)   # [N]

        if h_sub.shape[0] < 8:
            return torch.tensor(0.0, device=device)

        # L2 normalize embeddings
        emb_norm = F.normalize(h_sub, dim=-1)  # [N, D]

        # Cosine similarity to each prototype: [N, 3]
        proto_sim = emb_norm @ self.prototypes.T / self.temperature

        # ProtoNCE: cross-entropy where the correct class is the regime label
        loss = F.cross_entropy(proto_sim, labels_sub)

        # Update prototypes with current batch
        self._update_prototypes(h_sub, labels_sub)

        return loss


# =============================================================================
# 3. Trade Worthiness Head (cost-asymmetric classification)
# =============================================================================

class TradeWorthinessHead(nn.Module):
    """Three-class prediction: UP / FLAT / DOWN relative to cost threshold.

    v2 upgrade (NIST 2024): asymmetric cost-sensitive CE loss.
    False positives (trading FLAT bars) cost 0.24% round-trip = guaranteed loss.
    False negatives (missing UP/DOWN bars) = opportunity cost only.
    c_FP / c_FN ratio set to 3.0 (tunable).
    """

    def __init__(self, d_model: int, cost_threshold: float = 0.0012,
                 hidden: int = 128, dropout: float = 0.1, fp_fn_ratio: float = 3.0):
        super().__init__()
        self.cost_threshold = cost_threshold
        self.fp_fn_ratio = fp_fn_ratio
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),  # UP=2, FLAT=1, DOWN=0
        )

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: [B, T, D] -> [B, T, 3] logits for DOWN/FLAT/UP"""
        return self.head(h_seq)

    def compute_loss(self, logits: torch.Tensor, returns: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Asymmetric cost-sensitive cross-entropy.
        Class 1 (FLAT) misclassified as 0 or 2 = false positive = expensive.
        Class 0/2 misclassified as 1 = false negative = cheaper.
        """
        # Labels: 0=DOWN, 1=FLAT, 2=UP
        labels = torch.ones_like(returns, dtype=torch.long)
        labels[returns > self.cost_threshold] = 2
        labels[returns < -self.cost_threshold] = 0

        # Asymmetric class weights: FLAT class gets higher weight
        # because FP (predicting UP/DOWN when FLAT) is costlier than FN
        # Weight for FLAT = fp_fn_ratio, weight for UP/DOWN = 1.0
        class_weights = torch.tensor(
            [1.0, self.fp_fn_ratio, 1.0], device=logits.device
        )
        loss = F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1), weight=class_weights
        )

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = (pred == labels).float().mean()

        return loss, acc.item()


# =============================================================================
# 4. Volatility Head (QLIKE loss, vectorized)
# =============================================================================

class VolatilityHead(nn.Module):
    """Predicts forward realized volatility with QLIKE loss.

    v2 upgrades:
    - QLIKE loss (DeepVol, PMC:11473055): asymmetrically penalizes
      underestimation of vol (economically correct -- underestimating risk
      leads to oversized positions).
    - Vectorized realized vol computation (no Python for-loop).
    - Softplus output ensures positivity.

    QLIKE = sigma_hat/sigma_true - log(sigma_hat/sigma_true) - 1
    Minimum at sigma_hat = sigma_true. Steeper penalty for underestimation.
    """

    def __init__(self, d_model: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
            nn.Softplus(),  # Always positive
        )

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        """h_seq: [B, T, D] -> [B, T, 1] predicted forward volatility"""
        return self.head(h_seq)

    def compute_loss(self, pred_vol: torch.Tensor, returns: torch.Tensor,
                     window: int = 4) -> torch.Tensor:
        """
        QLIKE loss between predicted and realized forward volatility.

        pred_vol: [B, T, 1]
        returns: [B, T]
        window: bars for realized vol computation
        """
        B, T = returns.shape
        pred = pred_vol.squeeze(-1).clamp(min=1e-6)  # [B, T], prevent div by zero

        # Vectorized realized vol: rolling mean of |returns| over next `window` bars
        abs_ret = returns.abs()
        # Use unfold for vectorized rolling window (no Python loop)
        if T > window:
            # Pad end with last value for complete windows
            padded = F.pad(abs_ret, (0, window - 1), mode='replicate')  # [B, T + window - 1]
            realized = padded.unfold(1, window, 1).mean(dim=-1)  # [B, T]
        else:
            realized = abs_ret.mean(dim=1, keepdim=True).expand_as(abs_ret)

        realized = realized.clamp(min=1e-6)  # Prevent div by zero

        # QLIKE loss: sigma_hat/sigma_true - log(sigma_hat/sigma_true) - 1
        ratio = pred / realized
        loss = (ratio - torch.log(ratio) - 1.0).mean()

        return loss
