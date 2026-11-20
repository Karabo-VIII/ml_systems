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

    @staticmethod
    def encode_time(t, device: torch.device = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode scalar time t into sinusoidal features.

        Static so RK4Solver can precompute the full time-feature table once
        per forward instead of 4× per RK4 stage.

        Args:
            t: scalar tensor or float
            device: target device (inferred from t if None)
            dtype: output dtype (default fp32; use bf16/fp16 when caller
                    is under that autocast regime to avoid a later downcast)

        Returns: [4] time features (broadcasts against [B, D] on cat)
        """
        if isinstance(t, torch.Tensor):
            t_val = t.to(device=device or t.device, dtype=torch.float32)
        else:
            t_val = torch.tensor(t, dtype=torch.float32, device=device)

        # [sin(t), cos(t), sin(t/10), cos(t/10)]
        time_features = torch.stack([
            torch.sin(t_val),
            torch.cos(t_val),
            torch.sin(t_val / 10.0),
            torch.cos(t_val / 10.0),
        ])  # [4]
        return time_features.to(dtype)

    # Backward-compat wrapper for any external code that still calls the
    # old signature. Not used by RK4Solver.
    def _encode_time(self, t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
        tf = self.encode_time(t, device=device, dtype=torch.float32)  # [4]
        return tf.unsqueeze(0).expand(batch_size, -1)

    def forward(self, h: torch.Tensor, time_or_feat, obs_t: torch.Tensor) -> torch.Tensor:
        """
        Compute dh/dt = f_theta(h, time, obs_t).

        Accepts either:
          - scalar t (0-d tensor or float) — encoded on the fly; slow path
            for callers like dynamics_regularization that pass raw t.
          - pre-encoded time features [4] or [B, 4] — fast path used by
            RK4Solver which precomputes the full time table.

        Args:
            h: [B, D] hidden state
            time_or_feat: scalar t OR pre-encoded time features
            obs_t: [B, INPUT_DIM] observation at current timestep

        Returns: [B, D] time derivative of hidden state
        """
        B = h.shape[0]
        # Detect whether input is raw scalar time or pre-encoded [4]/[B,4]
        is_encoded = isinstance(time_or_feat, torch.Tensor) and time_or_feat.dim() >= 1 and time_or_feat.shape[-1] == 4
        if is_encoded:
            time_feat = time_or_feat
            if time_feat.dim() == 1:
                time_feat = time_feat.unsqueeze(0).expand(B, -1)
        else:
            # Slow path: encode scalar on the fly (for backward compat)
            time_feat = self.encode_time(time_or_feat, device=h.device,
                                            dtype=h.dtype).unsqueeze(0).expand(B, -1)
        x = torch.cat([h, time_feat.to(h.dtype), obs_t], dim=-1)
        return self.net(x)


# ==============================================================================
# RK4 SOLVER
# ==============================================================================

class RK4Solver(nn.Module):
    """
    Configurable fixed-step Neural ODE integrator (RK4 / RK2 / Euler).

    Integrates dh/dt = f_theta(h, t, obs_t) over a sequence of time points,
    collecting the hidden state at each discrete timestep.

    For intermediate evaluation points the observation from the current
    discrete step is reused (zero-order hold).

    Method:
        rk4   (default, original): 4 dynamics evals per step, 4th-order accurate
        rk2   (midpoint):          2 dynamics evals per step, 2nd-order accurate
                                   (~2x faster than RK4, very mild stability loss
                                    for non-stiff dynamics like ours over T=96)
        euler:                     1 dynamics eval per step, 1st-order accurate
                                   (4x faster than RK4 but unstable for our scale;
                                    use only for debugging/profiling)

    Sub-stepping: When substeps > 1, each interval [t_i, t_{i+1}] is divided
    into `substeps` smaller steps. Improves accuracy for stiff dynamics.
    NOTE: Higher substeps = more compute + VRAM. substeps=1 is the cheapest option.
    """
    def __init__(self, dynamics_fn: ODEDynamics, substeps: int = 1, method: str = "rk4"):
        super().__init__()
        self.dynamics_fn = dynamics_fn
        self.substeps = max(1, substeps)
        if method not in ("rk4", "rk2", "euler"):
            raise ValueError(f"unknown method {method!r}; want rk4|rk2|euler")
        self.method = method

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

        # Precompute time features for ALL RK4 stages in ALL timesteps.
        # For substeps=1 this collapses to (T-1) * 3 unique time points
        # (t_start, t_mid, t_end). Avoids 1500+ sin/cos/stack ops per
        # forward and keeps the hot loop Python-light.
        dt_full = (t_span[1] - t_span[0]).item() if T > 1 else 1.0
        sub_dt = dt_full / self.substeps
        # Build a flat list of time scalars used in the integrator loop,
        # then batch-encode them once. RK4 needs 3 unique times per step
        # (start, mid, end). RK2 (midpoint) needs 2 (start, mid).
        # Euler needs only 1 (start). We always emit 3 per substep for a
        # uniform layout; unused slots are simply not read.
        time_points = []
        for i in range(1, T):
            t_base = t_span[i - 1].item() if isinstance(t_span[i - 1], torch.Tensor) else float(t_span[i - 1])
            for s in range(self.substeps):
                t_sub = t_base + s * sub_dt
                time_points.append(t_sub)                  # start
                time_points.append(t_sub + sub_dt / 2.0)   # mid
                time_points.append(t_sub + sub_dt)         # end
        # [n_points, 4] pre-encoded time features in the solver's dtype
        times_tensor = torch.tensor(time_points, dtype=torch.float32, device=h0.device)
        time_features = torch.stack([
            torch.sin(times_tensor),
            torch.cos(times_tensor),
            torch.sin(times_tensor / 10.0),
            torch.cos(times_tensor / 10.0),
        ], dim=-1).to(h0.dtype)  # [n_points, 4]

        # Integrate step by step using RK4 with optional sub-stepping
        idx = 0
        for i in range(1, T):
            # Observation at current discrete step (zero-order hold for intermediate points)
            obs_curr = obs_seq[:, i - 1, :]  # [B, F]

            for _ in range(self.substeps):
                tf_start = time_features[idx]
                tf_mid = time_features[idx + 1]
                tf_end = time_features[idx + 2]
                idx += 3

                if self.method == "rk4":
                    k1 = self.dynamics_fn(h, tf_start, obs_curr)
                    k2 = self.dynamics_fn(h + sub_dt / 2.0 * k1, tf_mid, obs_curr)
                    k3 = self.dynamics_fn(h + sub_dt / 2.0 * k2, tf_mid, obs_curr)
                    k4 = self.dynamics_fn(h + sub_dt * k3, tf_end, obs_curr)
                    h = h + (sub_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                elif self.method == "rk2":
                    # Midpoint (Heun-like): 2 evals/step, 2nd-order accurate
                    k1 = self.dynamics_fn(h, tf_start, obs_curr)
                    k2 = self.dynamics_fn(h + sub_dt / 2.0 * k1, tf_mid, obs_curr)
                    h = h + sub_dt * k2
                else:  # euler
                    k1 = self.dynamics_fn(h, tf_start, obs_curr)
                    h = h + sub_dt * k1

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
