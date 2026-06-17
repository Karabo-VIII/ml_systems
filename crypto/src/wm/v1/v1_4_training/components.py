"""
V1 Components -- Transformer building blocks (SOTA 2025/26)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - PositionalEncoding: Sinusoidal position embeddings (legacy, kept for compat)
  - RotaryEmbedding: Rotary Position Embedding (RoPE, replaces sinusoidal PE)
  - CausalSelfAttention: Multi-head causal attention with FlashAttention + RoPE
  - FeedForward: Position-wise FFN with SwiGLU gating
  - CausalTransformerBlock: Pre-norm Transformer block (RMSNorm + attn + FFN)
  - TwoHotSymlog: Discretized regression target encoding with symlog transform
  - SwiGLU: Gated linear unit activation
  - MLPHead: Standard MLP head with RMSNorm

SOTA components:
  - FlashAttention via F.scaled_dot_product_attention (PyTorch 2.0+)
  - RoPE for relative position encoding (Su et al., 2021)
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
  - SwiGLU gated FFN (Shazeer, 2020)
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
# POSITIONAL ENCODING
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (legacy, kept for compatibility)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2] if d_model % 2 != 0 else div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input. x: [B, T, d_model]"""
        return x + self.pe[:, :x.size(1), :]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021).

    Encodes relative position directly into Q/K via complex rotation.
    Used in LLaMA, Mistral, GPT-NeoX. Replaces sinusoidal PE.

    Applied per-head: dim = d_model // n_heads.
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.cos_cached.size(2):
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to Q and K tensors."""
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


# ==============================================================================
# CAUSAL SELF-ATTENTION (FlashAttention + RoPE)
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with fused Q/K/V projection.

    SOTA features:
      - FlashAttention via F.scaled_dot_product_attention (PyTorch 2.0+)
      - Rotary Position Embedding (RoPE) applied to Q/K
    """

    _SDPA = hasattr(F, "scaled_dot_product_attention")

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.15):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout_p = dropout

        # Fused QKV projection for efficiency
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            rotary_emb: RotaryEmbedding instance (applies RoPE to Q/K)

        Returns:
            [B, T, d_model]
        """
        B, T, C = x.shape

        # Compute Q, K, V via fused projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.d_model, dim=-1)

        # Reshape for multi-head attention: [B, n_heads, T, d_head]
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K (relative position encoding)
        if rotary_emb is not None:
            cos, sin = rotary_emb(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self._SDPA:
            # FlashAttention path (PyTorch 2.0+): ~2-4x faster, O(T) memory
            dp = self.dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dp, is_causal=True,
            )
        else:
            # Manual attention fallback
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
            scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))


# ==============================================================================
# FEED-FORWARD NETWORK (SwiGLU)
# ==============================================================================

class FeedForward(nn.Module):
    """Position-wise FFN with SwiGLU gating (Shazeer, 2020)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.15):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)    # Gate branch
        self.w2 = nn.Linear(d_model, d_ff)    # Up branch
        self.w3 = nn.Linear(d_ff, d_model)    # Down projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ==============================================================================
# CAUSAL TRANSFORMER BLOCK (Pre-norm with RMSNorm)
# ==============================================================================

class CausalTransformerBlock(nn.Module):
    """Pre-norm Transformer block: RMSNorm -> Attention -> Residual -> RMSNorm -> FFN -> Residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.15):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, rotary_emb=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rotary_emb=rotary_emb)
        x = x + self.ffn(self.ln2(x))
        return x


# ==============================================================================
# TWO-HOT SYMLOG ENCODING — canonical (Jensen-correct) lives in _shared
# 2026-05-21: prior inline copy used Jensen-WRONG decode.
# ==============================================================================

import sys as _sys
from pathlib import Path as _Path
_shared_path = str(_Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in _sys.path:
    _sys.path.insert(0, _shared_path)
from twohot import TwoHotSymlog  # noqa: E402, F401


# ==============================================================================
# MLP HELPERS
# ==============================================================================

class SwiGLU(nn.Module):
    """Gated Linear Unit with SiLU activation."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int = None, dropout: float = 0.15):
        super().__init__()
        dim_out = dim_out or dim_in
        self.w_gate = nn.Linear(dim_in, dim_hidden)
        self.w_up = nn.Linear(dim_in, dim_hidden)
        self.w_down = nn.Linear(dim_hidden, dim_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MLPHead(nn.Module):
    """MLP head: Linear -> RMSNorm -> SiLU -> Dropout -> Linear."""

    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.15):
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
# FEATURE ATTENTION (iTransformer-style cross-feature interaction)
# ==============================================================================

class FeatureAttentionBlock(nn.Module):
    """
    iTransformer-style cross-feature attention (Liu et al., 2023).

    Attends over the feature dimension at each time step independently.
    Learns cross-feature correlations (e.g., when flow_imbalance AND vpin are
    both high, that's a stronger signal than either alone).

    Zero temporal mixing -- preserves causal structure of the downstream
    temporal Transformer.

    Input/output: [B, T, F] -- transparent to the rest of the model.
    """

    def __init__(self, n_features: int, d_feat: int = 32, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        assert d_feat % n_heads == 0, f"d_feat ({d_feat}) must be divisible by n_heads ({n_heads})"
        self.n_features = n_features

        # Lift each scalar feature to d_feat dimensions
        self.feat_proj_in = nn.Linear(1, d_feat)

        # Multi-head attention over F feature tokens
        self.attn_norm = RMSNorm(d_feat)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_feat,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN for feature token processing
        self.ffn_norm = RMSNorm(d_feat)
        self.ffn = nn.Sequential(
            nn.Linear(d_feat, d_feat * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_feat * 2, d_feat),
        )

        # Project back: d_feat -> scalar per feature
        self.feat_proj_out = nn.Linear(d_feat, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] -- raw feature observations

        Returns:
            [B, T, F] -- features enriched with cross-feature interactions
        """
        B, T, F = x.shape
        orig_dtype = x.dtype

        # Force fp32: head_dim=8 and 18 tokens are too small for fp16 precision.
        # RMSNorm eps=1e-6 underflows in fp16, causing NaN cascade at epoch 3+.
        # Cost: negligible (18 tokens x 32 dims).
        with torch.amp.autocast("cuda", enabled=False):
            x_fp32 = x.float()

            # Reshape: [B*T, F, 1] -- each feature is a token with dim=1
            x_flat = x_fp32.reshape(B * T, F, 1)

            # Lift: [B*T, F, 1] -> [B*T, F, d_feat]
            feat_tokens = self.feat_proj_in(x_flat)

            # Self-attention over F feature tokens (pre-norm)
            normed = self.attn_norm(feat_tokens)
            attn_out, _ = self.attn(normed, normed, normed)
            feat_tokens = feat_tokens + attn_out  # residual

            # FFN (pre-norm)
            feat_tokens = feat_tokens + self.ffn(self.ffn_norm(feat_tokens))

            # Project back: [B*T, F, d_feat] -> [B*T, F, 1] -> [B*T, F]
            out_flat = self.feat_proj_out(feat_tokens).squeeze(-1)

            # Reshape back: [B*T, F] -> [B, T, F]
            out = out_flat.reshape(B, T, F)

            # Final residual: preserve original features, add cross-feature signal
            result = x_fp32 + out

        return result.to(orig_dtype)
