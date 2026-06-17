"""
Reversible Instance Normalization (RevIN)

Kim et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting
against Distribution Shift", ICLR 2022.

Per-sequence normalization that removes distribution shift at input, then
re-injects statistics at output. Fully decoupled from model architecture.

Usage in training loop:
    revin = RevIN(num_features=INPUT_DIM).to(device)
    # Before model:
    obs_norm = revin(obs, mode='norm')
    # Model forward:
    loss, loss_dict, outputs = model.get_loss(obs_norm, ...)
    # For return predictions (optional denorm):
    pred = revin.denorm_scalar(outputs['pred_return'], feature_idx=11)
"""
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for time-series distribution shift.

    Normalizes each (batch, sequence) slice by its own mean/std per feature,
    with learnable affine parameters that adapt during training.

    Args:
        num_features: Number of input features (e.g., 18)
        eps: Epsilon for numerical stability
        affine: If True, learn per-feature gamma/beta (recommended)
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

        # Stored during normalize, used during denormalize
        self.register_buffer('_mean', torch.zeros(1), persistent=False)
        self.register_buffer('_std', torch.ones(1), persistent=False)

    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: [B, T, C] input features
            mode: 'norm' to normalize, 'denorm' to reverse

        Returns:
            [B, T, C] normalized or denormalized features
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"RevIN mode must be 'norm' or 'denorm', got '{mode}'")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input and store statistics for later denormalization."""
        # Per-sample, per-feature statistics: [B, 1, C]
        self._mean = x.mean(dim=1, keepdim=True).detach()
        self._std = (x.std(dim=1, keepdim=True) + self.eps).detach()

        x_norm = (x - self._mean) / self._std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse normalization using stored statistics."""
        if self.affine:
            x = (x - self.beta) / (self.gamma + self.eps)

        return x * self._std + self._mean

    def denorm_scalar(self, scalar: torch.Tensor, feature_idx: int) -> torch.Tensor:
        """Denormalize a scalar prediction using a specific feature's statistics.

        Useful for return predictions that should inherit the scale of
        norm_return_1 (feature_idx=11).

        Args:
            scalar: [B, ...] scalar values to denormalize
            feature_idx: Which feature's mean/std to use

        Returns:
            [B, ...] denormalized scalar
        """
        mean = self._mean[:, 0, feature_idx]  # [B]
        std = self._std[:, 0, feature_idx]     # [B]

        # Reshape for broadcasting with arbitrary trailing dims
        shape = [scalar.shape[0]] + [1] * (scalar.dim() - 1)
        mean = mean.view(shape)
        std = std.view(shape)

        if self.affine:
            gamma = self.gamma[0, 0, feature_idx]
            beta = self.beta[0, 0, feature_idx]
            scalar = (scalar - beta) / (gamma + self.eps)

        return scalar * std + mean
