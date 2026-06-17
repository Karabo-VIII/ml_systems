"""Multi-task heads -- joint trade/flat/size objective.

Replaces single-objective return prediction with a multi-task setup:
  Head 1: Direction logits (long / flat / short, 3-class)
  Head 2: Position-size regression (in [0, 1], optimal Kelly fraction)
  Head 3: Hold-duration regression (bars, log-scale)
  Head 4: Regime classification (bear / neutral / bull, 3-class)

Total loss: sum of per-task losses with task-uncertainty weighting (Kendall 2018).
The model learns automatic per-task weighting via log-variance parameters,
avoiding manual loss balancing.

Usage:
    head = MultiTaskHead(feat_dim=512)
    out = head(features)        # dict of all 4 task predictions
    loss = head.compute_loss(out, targets)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """4-head multi-task with Kendall-style uncertainty weighting."""

    def __init__(self, feat_dim: int = 512,
                 n_directions: int = 3,    # long / flat / short
                 n_regimes: int = 3,        # bear / neutral / bull
                 hidden: int = 128):
        super().__init__()
        self.direction_head = TaskHead(feat_dim, n_directions, hidden=hidden)
        self.size_head = TaskHead(feat_dim, 1, hidden=hidden)
        self.duration_head = TaskHead(feat_dim, 1, hidden=hidden)
        self.regime_head = TaskHead(feat_dim, n_regimes, hidden=hidden)

        # Per-task log-variance (Kendall 2018). Used to weight losses.
        self.log_var_direction = nn.Parameter(torch.zeros(1))
        self.log_var_size = nn.Parameter(torch.zeros(1))
        self.log_var_duration = nn.Parameter(torch.zeros(1))
        self.log_var_regime = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> dict:
        return {
            "direction_logits": self.direction_head(features),
            "size_logit": self.size_head(features).squeeze(-1),
            "duration_logit": self.duration_head(features).squeeze(-1),
            "regime_logits": self.regime_head(features),
        }

    def compute_loss(self, predictions: dict, targets: dict) -> dict:
        """Per-task losses + uncertainty-weighted total.

        Args:
            predictions: dict from forward()
            targets: dict with keys
                'direction' (long [B], 0/1/2),
                'size_target' ([B], in [0, 1]),
                'duration_target' ([B], log-bars),
                'regime' (long [B], 0/1/2)
        """
        # Direction (cross-entropy)
        dir_loss = F.cross_entropy(predictions["direction_logits"], targets["direction"])
        # Size (MSE on sigmoid)
        size_pred = torch.sigmoid(predictions["size_logit"])
        size_loss = F.mse_loss(size_pred, targets["size_target"])
        # Duration (MSE on raw, target already in log space)
        dur_loss = F.mse_loss(predictions["duration_logit"], targets["duration_target"])
        # Regime (cross-entropy)
        reg_loss = F.cross_entropy(predictions["regime_logits"], targets["regime"])

        # Kendall 2018 uncertainty weighting:
        #   weighted_loss_i = exp(-log_var_i) * loss_i + log_var_i
        # This gives the model an automatic way to balance tasks.
        weighted_dir = torch.exp(-self.log_var_direction) * dir_loss + self.log_var_direction
        weighted_size = torch.exp(-self.log_var_size) * size_loss + self.log_var_size
        weighted_dur = torch.exp(-self.log_var_duration) * dur_loss + self.log_var_duration
        weighted_reg = torch.exp(-self.log_var_regime) * reg_loss + self.log_var_regime
        total = (weighted_dir + weighted_size + weighted_dur + weighted_reg).squeeze()

        return {
            "loss": total,
            "direction_loss": dir_loss,
            "size_loss": size_loss,
            "duration_loss": dur_loss,
            "regime_loss": reg_loss,
            "log_var_direction": self.log_var_direction.detach(),
            "log_var_size": self.log_var_size.detach(),
            "log_var_duration": self.log_var_duration.detach(),
            "log_var_regime": self.log_var_regime.detach(),
        }


def smoke_test():
    torch.manual_seed(0)
    B, feat_dim = 32, 512
    head = MultiTaskHead(feat_dim=feat_dim)
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[mt-heads] params: {n_params:,}")

    feats = torch.randn(B, feat_dim)
    pred = head(feats)
    print(f"[mt-heads] direction_logits {tuple(pred['direction_logits'].shape)}, "
          f"size_logit {tuple(pred['size_logit'].shape)}, "
          f"duration_logit {tuple(pred['duration_logit'].shape)}, "
          f"regime_logits {tuple(pred['regime_logits'].shape)}")

    targets = {
        "direction": torch.randint(0, 3, (B,)),
        "size_target": torch.rand(B),
        "duration_target": torch.randn(B),  # log-space
        "regime": torch.randint(0, 3, (B,)),
    }
    out = head.compute_loss(pred, targets)
    print(f"[mt-heads] losses: total={out['loss'].item():.4f}  "
          f"dir={out['direction_loss'].item():.4f}  "
          f"size={out['size_loss'].item():.4f}  "
          f"dur={out['duration_loss'].item():.4f}  "
          f"reg={out['regime_loss'].item():.4f}")
    out["loss"].backward()
    has_grad = sum(1 for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[mt-heads] backward OK; {has_grad}/{sum(1 for _ in head.parameters())} params have grad")
    print("[mt-heads] smoke test PASS")


if __name__ == "__main__":
    smoke_test()
