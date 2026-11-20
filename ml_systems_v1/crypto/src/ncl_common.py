"""
Shared NCL (Negative Correlation Learning) Components

DEPRECATED: V1.D uses its own inline ReturnHead (with RMSNorm matching V1 architecture)
and inline NCL computation in ncl_model.py. This file is currently unused.
Kept as a reference template for future V2-V9.D variants that may need it.
To use: replace LayerNorm with RMSNorm from the target version's components.py.

Reference: Liu & Yao (1999) "Ensemble Learning via Negative Correlation Learning"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReturnHead(nn.Module):
    """Single return prediction path: trunk + per-horizon heads.

    Architecture-agnostic: takes any feature tensor and predicts
    TwoHot bucket logits for each horizon.
    """

    def __init__(
        self,
        input_dim: int,
        head_dim: int = 384,
        num_bins: int = 255,
        dropout: float = 0.05,
        horizons: list = None,
    ):
        super().__init__()
        self.horizons = horizons or [1, 4, 16, 64]

        # Shared trunk per head
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, head_dim),
            nn.LayerNorm(head_dim),  # Use LayerNorm for simplicity (RMSNorm in components)
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Per-horizon projection
        self.heads = nn.ModuleDict({
            str(h): nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                nn.LayerNorm(head_dim // 2),
                nn.SiLU(),
                nn.Linear(head_dim // 2, num_bins),
            )
            for h in self.horizons
        })

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: [B, T, input_dim] backbone feature tensor

        Returns:
            dict of {horizon: [B, T, num_bins]}
        """
        trunk_out = self.trunk(feat)
        return {h: self.heads[str(h)](trunk_out) for h in self.horizons}


def compute_ncl_penalty(all_head_errors: dict, n_heads: int, horizons: list) -> torch.Tensor:
    """Compute NCL diversity penalty across K heads.

    NCL forces each head's prediction errors to be negatively correlated
    with other heads' errors, promoting prediction diversity.

    NCL = (1/K) * sum_k(e_k * sum_{j!=k}(e_j))

    Args:
        all_head_errors: dict {horizon: list of K error tensors [N]}
        n_heads: number of diversity heads K
        horizons: list of horizon values

    Returns:
        Scalar NCL penalty
    """
    device = next(iter(next(iter(all_head_errors.values())))
                  if all_head_errors else [torch.tensor(0.0)]).device
    l_ncl = torch.tensor(0.0, device=device)
    n_valid = 0

    for h in horizons:
        if h not in all_head_errors or len(all_head_errors[h]) < 2:
            continue
        n_valid += 1
        errors = torch.stack(all_head_errors[h])  # [K, N]
        total_error = errors.sum(dim=0)            # [N]
        for k in range(n_heads):
            others_error = (total_error - errors[k]).detach()  # stop grad on others
            l_ncl = l_ncl + (errors[k] * others_error).mean()

    if n_valid > 0:
        l_ncl = l_ncl / (n_heads * n_valid)

    return l_ncl


# Default NCL settings (can be overridden per version)
NCL_DEFAULTS = {
    "n_heads": 5,
    "ncl_lambda": 0.5,
    "head_dim": 384,
    "head_dropout": 0.05,
    "lr": 2e-4,
    "weight_decay": 5e-2,
    "total_epochs": 200,
    "steps_per_epoch": 2000,
}
