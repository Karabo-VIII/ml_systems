"""
V4 Components — Reusable building blocks for world model and agent (SOTA 2025/26)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - TitaniumSSM: Mamba-style selective state space model (JIT-compiled scan)
  - MambaBlock: Residual Mamba layer with RMSNorm
  - TwoHotSymlog: Discretized regression target encoding
  - SwiGLU: Gated linear unit activation
  - MLPHead: Standard MLP head with RMSNorm

SOTA upgrades:
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
  - Consistent with V1's SOTA architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# MAMBA SELECTIVE STATE SPACE MODEL
# ══════════════════════════════════════════════════════════════════════════════

@torch.jit.script
def mamba_selective_scan(
    u: torch.Tensor,    # [B, T, d_inner]
    dt: torch.Tensor,   # [B, T, d_inner]
    A: torch.Tensor,    # [d_inner, d_state]
    B: torch.Tensor,    # [B, T, d_state]
    C: torch.Tensor,    # [B, T, d_state]
    D: torch.Tensor,    # [d_inner]
    h_prev: torch.Tensor  # [B, d_inner, d_state]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    JIT-compiled selective scan. Returns (output_seq, final_hidden_state).
    """
    batch_size, seq_len, d_inner = u.shape
    ys = torch.empty(batch_size, seq_len, d_inner, device=u.device, dtype=u.dtype)
    h = h_prev.clone()

    for t in range(seq_len):
        dt_t = dt[:, t, :, None]     # [B, d_inner, 1]
        u_t = u[:, t, :, None]       # [B, d_inner, 1]
        B_t = B[:, t, None, :]       # [B, 1, d_state]
        C_t = C[:, t, None, :]       # [B, 1, d_state]

        dA = torch.exp(dt_t * A)     # [B, d_inner, d_state]
        dB = dt_t * B_t              # [B, d_inner, d_state]

        h = h * dA + u_t * dB
        y_t = (h * C_t).sum(dim=-1)  # [B, d_inner]
        ys[:, t, :] = y_t

    ys = ys + u * D  # Skip connection
    return ys, h


class TitaniumSSM(nn.Module):
    """
    Mamba-style Selective State Space Model.
    
    Input:  [B, T, d_model]
    Output: [B, T, d_model]
    
    Optionally accepts and returns hidden state for streaming inference.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        # Input projection: x → (x_branch, z_gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=4, groups=self.d_inner, padding=3
        )

        # SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Learned SSM parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape

        # 1. Input projection and split
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # 2. Depthwise conv (local context mixing)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv = self.act(x_conv)

        # 3. SSM parameter computation
        x_dbl = self.x_proj(x_conv)
        dt_raw, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_raw))  # [B, T, d_inner]

        # 4. State space matrices
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]

        if h_prev is None:
            h_prev = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        # 5. Selective scan
        y, h_final = mamba_selective_scan(x_conv, dt, A, B_ssm, C_ssm, self.D, h_prev)

        # 6. Gate and project out
        output = self.dropout(self.out_proj(y * self.act(z_gate)))

        return output, h_final


class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection and RMSNorm.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ssm = TitaniumSSM(d_model, d_state, expand, dropout)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        y, h = self.ssm(self.norm(x), h_prev)
        return residual + y, h


# ══════════════════════════════════════════════════════════════════════════════
# TWO-HOT SYMLOG ENCODING — canonical (Jensen-correct) lives in _shared
# ══════════════════════════════════════════════════════════════════════════════

import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from twohot import TwoHotSymlog  # noqa: E402, F401


# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION & MLP BLOCKS
# ══════════════════════════════════════════════════════════════════════════════

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
