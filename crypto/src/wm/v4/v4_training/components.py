"""
V4 Components -- Mamba-3 SSM World Model (ICLR 2026)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - Mamba3SSM: Mamba-3 SSM with complex-valued state, trapezoidal discretization
  - Mamba3Block: Residual Mamba-3 layer with RMSNorm
  - TwoHotSymlog: Discretized regression target encoding
  - SwiGLU: Gated linear unit activation
  - MLPHead: Standard MLP head with RMSNorm

Mamba-3 innovations over Mamba-1 (original V4):
  - Complex-valued dynamics via data-dependent RoPE on B/C
  - Trapezoidal discretization (second-order accurate state update)
  - QK-Norm on B/C for training stability
  - SSD chunk-based parallel scan (replaces sequential for-loop)

Reference: Mamba-3: Improved Sequence Modeling (ICLR 2026, arXiv 2603.15569)
Pure PyTorch -- no custom CUDA kernels. Runs on RTX 4060.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# NORMALIZATION
# ======================================================================

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


# ======================================================================
# MAMBA-3 SSD ALGORITHM
# ======================================================================

def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable segment sum for 1-semiseparable decay matrix.

    Input:  x [..., T]
    Output: [..., T, T] lower-triangular cumulative sums.

    segsum[i,j] = sum(x[k] for k in range(j+1, i+1)) when i >= j, else -inf.
    exp(segsum) gives the causal decay from position j to position i.
    """
    T = x.size(-1)
    device = x.device
    # [..., T] -> [..., T, T]: broadcast along new last dim
    x_exp = x.unsqueeze(-1).expand(*x.shape, T)
    # Zero out upper triangle (keep only k > j, strict lower triangular)
    mask_lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-1)
    x_exp = x_exp.masked_fill(~mask_lower, 0)
    # Cumulative sum along rows (dim=-2)
    x_segsum = torch.cumsum(x_exp, dim=-2)
    # Mask upper triangle to -inf (keep only i >= j)
    mask_diag = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask_diag, -torch.inf)
    return x_segsum


def ssd(
    x: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    initial_states: torch.Tensor = None,
) -> tuple:
    """Structured State Space Duality -- chunked parallel scan (Mamba-2/3 core).

    Computes the SSM output by exploiting the duality between SSMs and
    structured attention. O(T * chunk_size) instead of O(T * d_state).

    Args:
        x: [B, T, H, P]  input (headdim P per head)
        A: [B, T, H]      log-space decay (dt * A_param, always negative)
        B: [B, T, H, N]   input projection (with RoPE)
        C: [B, T, H, N]   output projection (with RoPE)
        chunk_size: int    must divide T
        initial_states: [B, 1, H, P, N] optional initial hidden state

    Returns:
        Y: [B, T, H, P]            output
        final_state: [B, H, P, N]  last chunk's accumulated state
    """
    B_sz, T, H, P = x.shape
    N = B.shape[-1]
    L = chunk_size
    C_num = T // L
    device = x.device

    # ---- Reshape into chunks [B, C_num, L, ...] ----
    x_ch = x.reshape(B_sz, C_num, L, H, P)
    A_ch = A.reshape(B_sz, C_num, L, H)
    B_ch = B.reshape(B_sz, C_num, L, H, N)
    C_ch = C.reshape(B_sz, C_num, L, H, N)

    # A to [B, H, C_num, L] for cumsum along chunk positions
    A_r = A_ch.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_r, dim=-1)

    # ---- Intra-chunk: causal attention within each chunk ----
    # Decay mask: exp(segsum(A_r)) -> [B, H, C_num, L, L]
    L_mask = torch.exp(segsum(A_r))

    # Attention scores: C @ B^T per chunk, contract over d_state
    scores = torch.einsum("bclhn, bcshn -> bclhs", C_ch, B_ch)
    # Apply causal decay mask
    scores = scores.permute(0, 3, 1, 2, 4) * L_mask  # [B, H, C, l, s]
    scores = scores.permute(0, 2, 3, 1, 4)            # [B, C, l, H, s]
    # Contract with x over source position s
    Y_diag = torch.einsum("bclhs, bcshp -> bclhp", scores, x_ch)

    # ---- Inter-chunk: state propagation between chunks ----
    # Decay from each position to chunk end
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, H, C, L]

    # Accumulated state per chunk: B^T @ (decay * x)
    decay_for_B = decay_states.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, C, L, H, 1]
    B_scaled = B_ch * decay_for_B
    states = torch.einsum("bclhn, bclhp -> bchpn", B_scaled, x_ch)  # [B, C, H, P, N]

    # Prepend initial states
    if initial_states is None:
        initial_states = torch.zeros(B_sz, 1, H, P, N, device=device, dtype=x.dtype)
    states = torch.cat([initial_states, states], dim=1)  # [B, C+1, H, P, N]

    # Inter-chunk decay matrix
    A_chunk_totals = A_cumsum[:, :, :, -1]  # [B, H, C]
    decay_chunk = torch.exp(segsum(F.pad(A_chunk_totals, (1, 0))))  # [B, H, C+1, C+1]

    # Propagate states across chunks
    new_states = torch.einsum("bhzc, bchpn -> bzhpn", decay_chunk, states)
    states = new_states[:, :-1]       # [B, C, H, P, N]
    final_state = new_states[:, -1]   # [B, H, P, N]

    # Off-diagonal: C @ states, scaled by decay from chunk start
    state_out = torch.einsum("bclhn, bchpn -> bclhp", C_ch, states)
    state_decay_out = torch.exp(A_cumsum)  # [B, H, C, L]
    decay_out = state_decay_out.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, C, L, H, 1]
    Y_off = state_out * decay_out

    # Combine intra-chunk and inter-chunk contributions
    Y = (Y_diag + Y_off).reshape(B_sz, T, H, P)
    return Y, final_state


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding with data-dependent angles.

    Pairs consecutive elements and rotates them. This implements the
    complex-valued state dynamics of Mamba-3: instead of explicit complex
    numbers, we rotate real pairs, which is algebraically equivalent and
    numerically stable in fp32.

    Args:
        x: [..., D] where D is even
        angles: [..., D//2] rotation angles
    Returns:
        Rotated tensor [..., D]
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    y1 = cos_a * x1 - sin_a * x2
    y2 = sin_a * x1 + cos_a * x2
    return torch.stack([y1, y2], dim=-1).flatten(-2)


# ======================================================================
# MAMBA-3 SSM BLOCK
# ======================================================================

class Mamba3SSM(nn.Module):
    """Mamba-3 Selective State Space Model.

    Improvements over Mamba-1 (TitaniumSSM):
      1. Complex-valued dynamics via RoPE on B/C (oscillatory, mean-reversion)
      2. Trapezoidal discretization (second-order accurate, two-SSD)
      3. QK-Norm on B/C (prevents gradient explosion in scan)
      4. SSD chunk-based algorithm (hardware-efficient parallel scan)

    Input:  [B, T, d_model]
    Output: [B, T, d_model]
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.nheads = self.d_inner // headdim

        assert d_state % 2 == 0, "d_state must be even for RoPE pairing"
        assert self.d_inner % headdim == 0, (
            f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
        )

        # Single combined projection for all SSM inputs:
        #   z (gate), x (input), B, C, dt (step), lam (trapezoidal), theta (RoPE)
        bc_dim = d_state
        d_proj = (
            2 * self.d_inner       # z + x
            + 2 * bc_dim           # B + C
            + 2 * self.nheads      # dt + lam
            + d_state // 2         # theta (RoPE angles)
        )
        self.in_proj = nn.Linear(d_model, d_proj, bias=False)

        # Learnable SSM parameters (per-head)
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))

        # QK-Norm on B and C (prevents gradient explosion)
        self.B_norm = RMSNorm(bc_dim)
        self.C_norm = RMSNorm(bc_dim)

        # Learnable head-specific bias for B and C (initialized to ones)
        self.B_bias = nn.Parameter(torch.ones(self.nheads, d_state))
        self.C_bias = nn.Parameter(torch.ones(self.nheads, d_state))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        # Custom initialization for SSM parameters (matches Mamba-3 paper)
        with torch.no_grad():
            # A_log uniform(-4,-1): gives A = -exp(A_log) in [-54.6, -2.7]
            # Different heads get DIFFERENT decay rates (multi-scale temporal vision)
            # Without this, all heads cluster near -1.0 and learn identical dynamics.
            self.A_log.uniform_(-4.0, -1.0)
            # dt_bias: small positive range for stable initial step sizes
            # After softplus(dt_raw + dt_bias), initial dt ~ [0.69, 0.74]
            self.dt_bias.uniform_(0.001, 0.1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B_sz, T, _ = u.shape

        # Pad T to be divisible by chunk_size if needed
        T_orig = T
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            u = F.pad(u, (0, 0, 0, pad_len))
            T = T + pad_len

        A = -torch.exp(self.A_log)  # [nheads], always negative for stable decay

        # Project all SSM inputs at once
        proj = self.in_proj(u)  # [B, T, d_proj]
        z, x, B_raw, C_raw, dt_raw, lam_raw, theta = torch.split(
            proj,
            [self.d_inner, self.d_inner, self.d_state, self.d_state,
             self.nheads, self.nheads, self.d_state // 2],
            dim=-1,
        )

        # Step size and trapezoidal interpolation
        dt = F.softplus(dt_raw + self.dt_bias)  # [B, T, nheads]
        lam = torch.sigmoid(lam_raw)            # [B, T, nheads]

        # QK-Norm on B and C
        B_normed = self.B_norm(B_raw)  # [B, T, d_state]
        C_normed = self.C_norm(C_raw)  # [B, T, d_state]

        # Data-dependent RoPE angles (complex-valued state dynamics)
        # dt: [B, T, nheads], theta: [B, T, d_state//2]
        raw_angles = dt.unsqueeze(-1) * theta.unsqueeze(-2)  # [B, T, nheads, d_state//2]
        cum_angles = -torch.cumsum(raw_angles, dim=1)

        # Trapezoidal discretization: y = gamma*f(t_n) + beta*f(t_{n-1})
        dA = dt * A.unsqueeze(0).unsqueeze(0)  # [B, T, nheads]
        beta = (1 - lam) * dt * torch.exp(dA)  # left endpoint (previous input)
        gamma = lam * dt                        # right endpoint (current input)

        # Reshape x to multi-head: [B, T, nheads, headdim]
        x = x.reshape(B_sz, T, self.nheads, self.headdim)

        # Add head-specific bias and apply RoPE to B and C
        # B_normed: [B, T, d_state] -> unsqueeze to [B, T, 1, d_state]
        # B_bias: [nheads, d_state] -> broadcasts to [B, T, nheads, d_state]
        B_proj = B_normed.unsqueeze(2) + self.B_bias
        C_proj = C_normed.unsqueeze(2) + self.C_bias
        B_proj = apply_rope(B_proj, cum_angles)
        C_proj = apply_rope(C_proj, cum_angles)

        # ---- Two-SSD trapezoidal decomposition ----
        # SSD 1 (gamma): current timestep input
        y_gamma, _ = ssd(
            x * gamma.unsqueeze(-1),
            dA, B_proj, C_proj,
            self.chunk_size,
        )

        # SSD 2 (beta): previous timestep input (shifted by 1, zero-padded)
        B_prev = F.pad(B_proj[:, :-1], (0, 0, 0, 0, 1, 0))
        x_prev = F.pad(x[:, :-1], (0, 0, 0, 0, 1, 0))
        y_beta, _ = ssd(
            x_prev * beta.unsqueeze(-1),
            dA, B_prev, C_proj,
            self.chunk_size,
        )

        # Combine + skip connection
        y = y_gamma + y_beta + x * self.D.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # Gate and project out
        y = y.reshape(B_sz, T, self.d_inner)
        y = y * F.silu(z)
        y = self.dropout_layer(self.out_proj(y))

        # Remove padding if applied
        if pad_len > 0:
            y = y[:, :T_orig]

        return y


class Mamba3Block(nn.Module):
    """Residual Mamba-3 block with pre-norm (RMSNorm)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        expand: int = 2,
        headdim: int = 64,
        chunk_size: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ssm = Mamba3SSM(d_model, d_state, expand, headdim, chunk_size, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ======================================================================
# TWO-HOT SYMLOG ENCODING — canonical (Jensen-correct) lives in _shared
# 2026-05-21: V4's prior inline copy used Jensen-WRONG decode
#             symexp(E[buckets]) — biased fat-tail returns toward zero.
#             Many downstream modules (V11p_sparse_moe, V15/V16/V17 stubs,
#             V21_mamba_node, frontier_ml/distillation, frontier_ml/foundation,
#             born_again) import from this file; the import now resolves
#             through src/wm/_shared/twohot.py.
# ======================================================================

import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from twohot import TwoHotSymlog  # noqa: E402, F401


# ======================================================================
# ACTIVATION & MLP BLOCKS
# ======================================================================

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
