"""V17 — TD-MPC2 production backbone.

Hansen et al. 2024 (arXiv 2310.16828). Latent dynamics model + value
function + learned policy. Adapted for crypto WM: instead of inference-
time CEM planning, we use the latent dynamics + value head as a
PREDICTIVE backbone, suitable for our supervised-IC objective.

Design:
- Encoder: f34 -> z (latent state, d_z=128)
- Latent dynamics: z, (synthetic action embedding) -> z' via small MLP
- Value heads: predict K-step value targets at h={1,4,16,64} from z
- Reward heads (per horizon): reward = next-bar return at horizon h

Difference from V16 (DreamerV3):
- TD-MPC2 uses CONTINUOUS latent z (no categorical sampling); cleaner gradients
- No reconstruction; pure latent-prediction objective
- Smaller param count by design

This is V17 SOTA (per user 2026-05-02 directive). Forward + backward
verified; trainer integration follows V1.x pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reclassified 2026-06-11: was src/wm/v17/ (parents[3]=repo root); now
# src/agents/a1_wm_consuming/backbones/v17_tdmpc2/ (parents[5]=repo root).
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import TwoHotSymlog, RMSNorm  # noqa: E402


class V17TDMPC2WM(nn.Module):
    """TD-MPC2-style latent dynamics WM."""

    def __init__(
        self,
        n_features: int = 34,
        d_z: int = 128,
        d_hidden: int = 256,
        num_bins: int = 255,
        horizons: tuple = (1, 4, 16, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_z = d_z
        self.horizons = tuple(horizons)
        self.num_bins = num_bins

        # Encoder: input -> latent z
        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
        )
        self.z_norm = RMSNorm(d_z)

        # Latent dynamics: z_t -> z_{t+1}  (single-step; called per-bar across context)
        self.dynamics = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_z),
        )
        # Residual gate: z_{t+1} = z_t + alpha * delta(z_t)
        self.dyn_gate = nn.Parameter(torch.tensor(0.1))

        # Multi-horizon return heads operating on encoded z
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Sequential(
                nn.Linear(d_z, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, num_bins),
            )
            for h in self.horizons
        })

        # Value head (cumulative discounted return) -- for TD-MPC2 fidelity
        self.value_head = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, num_bins),
        )

        # Frontier-ML hooks
        self._use_mtp = False
        self.mtp_head = None
        self._use_mdn = False
        self.bucketer = TwoHotSymlog(num_bins, -1.0, 1.0, "cuda")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) -> z: (B, T, d_z)."""
        return self.z_norm(self.encoder(x))

    def step_dynamics(self, z: torch.Tensor) -> torch.Tensor:
        """One-step latent dynamics; residual update."""
        delta = self.dynamics(z)
        return z + self.dyn_gate * delta

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor = None) -> dict:
        B, T, F_in = obs_seq.shape

        # Per-timestep encoding (no temporal context; dynamics carries the chain)
        z_obs = self.encode(obs_seq)                         # (B, T, d_z)

        # Optional: roll dynamics forward over timesteps (causal TD-MPC2)
        # Initialize z from the first observation; subsequent z evolve via dynamics
        # while being CORRECTED by observation embeddings (filter-style).
        z_seq = []
        z = z_obs[:, 0, :]
        for t in range(T):
            if t == 0:
                z = z_obs[:, t, :]
            else:
                # Predict from prev dynamics + correct with observation
                z_pred = self.step_dynamics(z_seq[-1])
                z = 0.5 * (z_pred + z_obs[:, t, :])
            z_seq.append(z)
        z_t = torch.stack(z_seq, dim=1)                       # (B, T, d_z)

        # Multi-horizon return predictions from the latent at each timestep
        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(z_t)
            for h in self.horizons:
                return_logits[h] = mtp_out[f"h{h}"]
        else:
            for h in self.horizons:
                return_logits[h] = self.return_heads[f"h{h}"](z_t)

        # Value head (last timestep only; cumulative-future-return forecast)
        value_logits = self.value_head(z_t[:, -1, :])

        return {
            "return_logits": return_logits,
            "h_seq": z_t,
            "z_post": z_t,
            "ret_trunk": z_t,
            "value_logits": value_logits,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke():
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if DEV == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30)
    model = V17TDMPC2WM(n_features=34, d_z=128, d_hidden=256).to(DEV)
    print(f"[v17-tdmpc2] params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    B, T = 4, 32
    obs = torch.randn(B, T, 34, device=DEV)
    out = model.forward_train(obs)
    for k, v in out["return_logits"].items():
        assert v.shape == (B, T, model.num_bins)
    print(f"[v17-tdmpc2] return_logits OK; value: {tuple(out['value_logits'].shape)}")
    loss = sum(v.float().pow(2).mean() for v in out["return_logits"].values())
    loss = loss + out["value_logits"].float().pow(2).mean()
    loss.backward()
    # Verify dynamics gate has gradient (TD-MPC2 latent rollout learning)
    assert model.dyn_gate.grad is not None
    print(f"[v17-tdmpc2] dyn_gate grad: {model.dyn_gate.grad.item():.4f}  (rollout learning)")
    print("[v17-tdmpc2] PASS smoke")


if __name__ == "__main__":
    smoke()
