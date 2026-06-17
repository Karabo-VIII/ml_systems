"""
V3 Components — WaveNet-GRU Building Blocks (SOTA 2025/26)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - Chomp1d: Causal padding trimmer
  - WaveNetBlock: Gated dilated causal convolution (tanh * sigmoid)
  - WaveNetTCN: Stack of WaveNet blocks with skip connections
  - MultiScaleAggregator: Aggregates skip connections from all scales
  - CausalGRU: GRU with RMSNorm for sequential dynamics
  - TwoHotSymlog: Discretized regression encoding with symlog transform
  - SwiGLU: Gated Linear Unit with SiLU activation
  - MLPHead: Standard MLP head with RMSNorm

SOTA components:
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# NORMALIZATION
# =============================================================================

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


# =============================================================================
# CAUSAL PADDING
# =============================================================================

class Chomp1d(nn.Module):
    """Removes trailing padding to ensure strict causality in conv outputs."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size > 0:
            return x[:, :, :-self.chomp_size].contiguous()
        return x


# =============================================================================
# WAVENET-STYLE GATED DILATED CAUSAL CONVOLUTION
# =============================================================================

class WaveNetBlock(nn.Module):
    """
    WaveNet-style gated dilated causal convolution block.

    Uses gated activation: tanh(conv_filter(x)) * sigmoid(conv_gate(x))
    Produces both a residual output (for the next layer) and a skip output
    (for the multi-scale aggregator).

    Architecture:
      - Two parallel dilated causal convolutions (filter + gate)
      - Gated activation: tanh(filter) * sigmoid(gate)
      - Dropout for regularization
      - 1x1 residual projection + skip-connection projection
      - LayerNorm on residual output
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        # Filter and gate convolutions (parallel dilated causal convs)
        self.conv_filter = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv_gate = nn.Conv1d(
            channels, channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.chomp = Chomp1d(padding)

        # 1x1 projections
        self.residual = nn.Conv1d(channels, channels, 1)
        self.skip = nn.Conv1d(channels, channels, 1)

        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(channels)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, T]
        Returns:
            residual: [B, C, T] — input to next block
            skip:     [B, C, T] — collected for multi-scale aggregation
        """
        # Gated activation: tanh(filter) * sigmoid(gate)
        h_filter = self.chomp(self.conv_filter(x))
        h_gate = self.chomp(self.conv_gate(x))
        h = torch.tanh(h_filter.float()) * torch.sigmoid(h_gate.float())
        h = h.to(x.dtype)
        h = self.dropout(h)

        # Skip connection output
        skip_out = self.skip(h)

        # Residual connection with LayerNorm
        res_out = x + self.residual(h)
        # LayerNorm expects [B, C] or [B, T, C], so transpose
        res_out = self.norm(res_out.transpose(1, 2)).transpose(1, 2)

        return res_out, skip_out


# =============================================================================
# WAVENET TCN (STACK OF GATED BLOCKS)
# =============================================================================

class WaveNetTCN(nn.Module):
    """
    Stack of WaveNet blocks with increasing dilation.

    Accepts variable channel sizes per layer. The first layer includes
    a 1x1 input projection if input_dim != channels[0]. Subsequent layers
    use 1x1 projections between different channel sizes.

    Each layer produces skip connections that are collected for the
    MultiScaleAggregator.

    Receptive field with kernel=3, dilations=[1,2,4,8]:
      Layer 0: 3 steps
      Layer 1: 7 steps
      Layer 2: 15 steps
      Layer 3: 31 steps
    """

    def __init__(
        self,
        input_dim: int,
        channels: list,
        kernel_size: int = 3,
        dilations: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        if dilations is None:
            dilations = [2 ** i for i in range(len(channels))]

        assert len(channels) == len(dilations), (
            f"channels ({len(channels)}) and dilations ({len(dilations)}) must match"
        )

        self.out_channels = channels[-1]

        # Input projection to first channel size
        self.input_proj = nn.Conv1d(input_dim, channels[0], 1)

        # WaveNet blocks with channel transition projections
        self.blocks = nn.ModuleList()
        self.channel_projs = nn.ModuleList()  # 1x1 convs between channel sizes

        for i, (ch, dil) in enumerate(zip(channels, dilations)):
            prev_ch = channels[0] if i == 0 else channels[i - 1]

            # Channel transition if sizes differ between layers
            if prev_ch != ch:
                self.channel_projs.append(nn.Conv1d(prev_ch, ch, 1))
            else:
                self.channel_projs.append(nn.Identity())

            self.blocks.append(WaveNetBlock(ch, kernel_size, dil, dropout))

        # Final skip channel is the largest (last) channel size
        # We project all skip connections to the last channel size
        self.skip_projs = nn.ModuleList()
        for ch in channels:
            if ch != channels[-1]:
                self.skip_projs.append(nn.Conv1d(ch, channels[-1], 1))
            else:
                self.skip_projs.append(nn.Identity())

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C_in] — batch, time, features
        Returns:
            out: [B, T, C_out] — final layer output
            skips: list of [B, C_out, T] — skip connections from each layer
        """
        # TCN operates on [B, C, T]
        h = x.transpose(1, 2)  # [B, C_in, T]
        h = self.input_proj(h)  # [B, channels[0], T]

        skips = []
        for i, (block, ch_proj, sk_proj) in enumerate(
            zip(self.blocks, self.channel_projs, self.skip_projs)
        ):
            h = ch_proj(h)  # Channel transition
            h, skip = block(h)
            skips.append(sk_proj(skip))  # Project skip to uniform channel size

        # Return final hidden and skip connections
        out = h.transpose(1, 2)  # [B, T, C_out]
        return out, skips


# =============================================================================
# MULTI-SCALE AGGREGATOR
# =============================================================================

class MultiScaleAggregator(nn.Module):
    """
    Aggregates skip connections from all WaveNet layers.

    Sums all skip tensors (already projected to same channel size),
    then applies a small 1x1 convolution stack to produce the final
    multi-scale feature representation.
    """

    def __init__(self, channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )

    def forward(self, skips: list) -> torch.Tensor:
        """
        Args:
            skips: list of [B, C, T] tensors from each WaveNet layer
        Returns:
            out: [B, T, C_out]
        """
        aggregated = sum(skips)  # [B, C, T]
        out = self.proj(aggregated)  # [B, C_out, T]
        return out.transpose(1, 2)  # [B, T, C_out]


# =============================================================================
# CAUSAL GRU
# =============================================================================

class CausalGRU(nn.Module):
    """
    GRU for sequential dynamics modeling.

    Replaces LSTM with fewer parameters and comparable performance.
    Includes LayerNorm on the output for training stability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None,
    ):
        """
        Args:
            x: [B, T, input_dim]
            hidden: Optional [num_layers, B, hidden_dim]
        Returns:
            out: [B, T, hidden_dim]
            hidden: [num_layers, B, hidden_dim]
        """
        out, hidden = self.gru(x, hidden)
        return self.norm(out), hidden


# =============================================================================
# TWO-HOT SYMLOG — canonical (Jensen-correct) lives in _shared
# =============================================================================

import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from twohot import TwoHotSymlog  # noqa: E402, F401


# =============================================================================
# MLP HELPERS
# =============================================================================

class SwiGLU(nn.Module):
    """
    Gated Linear Unit with SiLU activation.

    Computes: dropout(W_down(SiLU(W_gate(x)) * W_up(x)))
    More expressive than standard MLP with comparable parameter count.
    """

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_out = dim_out or dim_in
        self.w_gate = nn.Linear(dim_in, dim_hidden)
        self.w_up = nn.Linear(dim_in, dim_hidden)
        self.w_down = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MLPHead(nn.Module):
    """Standard MLP head with RMSNorm + SiLU + Dropout."""

    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            RMSNorm(dim_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
