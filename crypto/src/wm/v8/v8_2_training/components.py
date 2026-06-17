"""
V8 Components -- Reusable building blocks for Neural ODE world model (SOTA 2025/26)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - TwoHotSymlog: Discretized regression target encoding (255 bins, [-1, 1])
  - SwiGLU: Gated linear unit activation
  - MLPHead: Standard MLP head with RMSNorm
  - ODEDynamics: Dynamics function f_theta(h, t, obs_t) for Neural ODE
  - RK4Solver: Fixed-step Runge-Kutta 4 integrator
  - EmissionNetwork: Decodes hidden state to observation space

SOTA components:
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==============================================================================
# NORMALIZATION
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Used in LLaMA, Mistral, Gemma. ~10-15% faster than LayerNorm.
    Omits mean-centering and learned bias.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ==============================================================================
# TWO-HOT SYMLOG ENCODING — canonical (Jensen-correct) lives in _shared
# ==============================================================================

import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from twohot import TwoHotSymlog  # noqa: E402, F401


# ==============================================================================
# ACTIVATION & MLP BLOCKS
# ==============================================================================

class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU activation."""
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int = None, dropout: float = 0.1):
        super().__init__()
        dim_out = dim_out or dim_in
        self.w_gate = nn.Linear(dim_in, dim_hidden)
        self.w_up = nn.Linear(dim_in, dim_hidden)
        self.w_down = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MLPHead(nn.Module):
    """Standard MLP head with RMSNorm."""
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            RMSNorm(dim_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


# ==============================================================================
# NEURAL ODE DYNAMICS FUNCTION
# ==============================================================================

class ODEDynamics(nn.Module):
    """
    Dynamics function f_theta(h, t, obs_t) for the Neural ODE.

    Learns dh/dt = f_theta(h, t, obs_t) where:
      - h: hidden state [B, D]
      - t: scalar time (encoded as sinusoidal features)
      - obs_t: observation at current timestep [B, INPUT_DIM] for conditioning

    Time encoding: [sin(t), cos(t), sin(t/10), cos(t/10)] = 4 dims
    Total input: h (d_model) + time_features (4) + obs_t (input_dim=13) = d_model + 17
    """
    def __init__(
        self,
        d_model: int = 256,
        hidden_layers: list = None,
        dropout: float = 0.15,
        input_dim: int = 18,
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [256, 512, 512, 256]

        self.d_model = d_model
        self.input_dim = input_dim
        self.time_dim = 4  # sin(t), cos(t), sin(t/10), cos(t/10)

        # Total input dimension: h + time_features + obs_t
        total_input_dim = d_model + self.time_dim + input_dim

        # Build MLP stack: input -> hidden_layers -> output (d_model)
        layers = []
        in_dim = total_input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        # Final layer + RMSNorm
        layers.append(RMSNorm(in_dim))
        layers.append(nn.Linear(in_dim, d_model))

        self.net = nn.Sequential(*layers)

    def _encode_time(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Encode scalar time t into sinusoidal features.

        Args:
            t: scalar tensor or float
            batch_size: B for broadcasting
            device: target device

        Returns: [B, 4] time features
        """
        t_val = t.float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32, device=device)
        t_val = t_val.to(device)

        # [sin(t), cos(t), sin(t/10), cos(t/10)]
        time_features = torch.stack([
            torch.sin(t_val),
            torch.cos(t_val),
            torch.sin(t_val / 10.0),
            torch.cos(t_val / 10.0),
        ])  # [4]

        # Expand to [B, 4]
        return time_features.unsqueeze(0).expand(batch_size, -1)

    def forward(self, h: torch.Tensor, t: torch.Tensor, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = f_theta(h, t, obs_t).

        Args:
            h: [B, D] hidden state
            t: scalar tensor -- current time
            obs_t: [B, INPUT_DIM] observation at current timestep

        Returns: [B, D] time derivative of hidden state
        """
        B = h.shape[0]
        time_feat = self._encode_time(t, B, h.device)  # [B, 4]
        x = torch.cat([h, time_feat, obs_t], dim=-1)    # [B, D + 4 + INPUT_DIM]
        return self.net(x)


# ==============================================================================
# RK4 SOLVER
# ==============================================================================

class RK4Solver(nn.Module):
    """
    Runge-Kutta 4 integrator for Neural ODE with configurable sub-stepping.

    Integrates dh/dt = f_theta(h, t, obs_t) over a sequence of time points,
    collecting the hidden state at each discrete timestep.

    For intermediate RK4 evaluation points (k2, k3 at t + dt/2), the observation
    from the current discrete step is reused (zero-order hold).

    Sub-stepping: When substeps > 1, each interval [t_i, t_{i+1}] is divided
    into `substeps` smaller RK4 steps. This improves accuracy for stiff dynamics
    without requiring a fully adaptive solver.
    NOTE: Higher substeps = more compute + VRAM. substeps=1 is the cheapest option.
    """
    def __init__(self, dynamics_fn: ODEDynamics, substeps: int = 1):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        self.substeps = max(1, substeps)

    def forward(
        self,
        h0: torch.Tensor,
        obs_seq: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate the ODE from t_span[0] to t_span[-1].

        Args:
            h0: [B, D] initial hidden state
            obs_seq: [B, T, F] observation sequence for conditioning
            t_span: [T] time points (e.g., 0, 1, 2, ..., 95)

        Returns: [B, T, D] trajectory of hidden states at each time point
        """
        B, T, F = obs_seq.shape
        D = h0.shape[1]

        # Allocate output trajectory
        trajectory = torch.empty(B, T, D, device=h0.device, dtype=h0.dtype)

        # First time point: initial state
        h = h0
        trajectory[:, 0, :] = h

        # Integrate step by step using RK4 with optional sub-stepping
        for i in range(1, T):
            t_start = t_span[i - 1]
            dt_total = t_span[i] - t_span[i - 1]
            sub_dt = dt_total / self.substeps

            # Observation at current discrete step (zero-order hold for intermediate points)
            obs_curr = obs_seq[:, i - 1, :]  # [B, F]

            # Sub-stepping loop
            t_sub = t_start
            for _ in range(self.substeps):
                # RK4 stages
                k1 = self.dynamics_fn(h, t_sub, obs_curr)
                k2 = self.dynamics_fn(h + sub_dt / 2.0 * k1, t_sub + sub_dt / 2.0, obs_curr)
                k3 = self.dynamics_fn(h + sub_dt / 2.0 * k2, t_sub + sub_dt / 2.0, obs_curr)
                k4 = self.dynamics_fn(h + sub_dt * k3, t_sub + sub_dt, obs_curr)

                # RK4 update
                h = h + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t_sub = t_sub + sub_dt

            trajectory[:, i, :] = h

        return trajectory


# ==============================================================================
# EULER SOLVER (V8.2)
# ==============================================================================

class EulerSolver(nn.Module):
    """
    Forward Euler integrator for Neural ODE (V8.2 ablation).

    Simpler than RK4 (1 evaluation per sub-step vs 4). Compensated with
    more sub-steps for stability. Cheaper per-step but may require more
    sub-steps for equivalent accuracy.
    """
    def __init__(self, dynamics_fn: ODEDynamics, substeps: int = 4):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        self.substeps = max(1, substeps)

    def forward(
        self,
        h0: torch.Tensor,
        obs_seq: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate the ODE from t_span[0] to t_span[-1] using Euler method.

        Args:
            h0: [B, D] initial hidden state
            obs_seq: [B, T, F] observation sequence for conditioning
            t_span: [T] time points

        Returns: [B, T, D] trajectory of hidden states at each time point
        """
        B, T, F = obs_seq.shape
        D = h0.shape[1]

        trajectory = torch.empty(B, T, D, device=h0.device, dtype=h0.dtype)
        h = h0
        trajectory[:, 0, :] = h

        for i in range(1, T):
            t_start = t_span[i - 1]
            dt_total = t_span[i] - t_span[i - 1]
            sub_dt = dt_total / self.substeps
            obs_curr = obs_seq[:, i - 1, :]

            t_sub = t_start
            for _ in range(self.substeps):
                dh = self.dynamics_fn(h, t_sub, obs_curr)
                h = h + sub_dt * dh
                t_sub = t_sub + sub_dt

            trajectory[:, i, :] = h

        return trajectory


# ==============================================================================
# EMISSION NETWORK
# ==============================================================================

class EmissionNetwork(nn.Module):
    """
    Decodes hidden state to observation space.

    Architecture: Linear -> SiLU -> Dropout -> Linear
    Maps from d_model dimensional hidden states back to input_dim observations.
    """
    def __init__(self, d_model: int, input_dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, T, D] or [B, D] hidden states

        Returns: [B, T, input_dim] or [B, input_dim] decoded observations
        """
        return self.net(h)
