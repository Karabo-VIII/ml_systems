"""MultiModalAdapter — cross-attention adapter on a frozen foundation backbone.

Architecture:
    Frozen FoundationBackbone -> h_seq (B, S, d_model)
    Multi-modal channels (per-bar) -> projected to (B, S, d_mm)
    Cross-attention: h_seq queries into channels' KV
    Residual fuse: h_seq + xattn_output  (B, S, d_model)
    New TwoHot heads on top of the fused representation.

Param budget (~2M total):
    channel projection: n_channels * d_mm                 ~ 1K
    xattn QKV proj:    3 * d_model * d_mm                ~ 600K
    output proj:       d_mm * d_model                    ~ 200K
    new TwoHot heads:  4 horizons * d_model * 255        ~ 800K
    misc norms + bias                                    ~ small

The frozen foundation is loaded once; adapter parameters require_grad.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from frontier_ml.foundation.backbone import FoundationBackbone, DEFAULT_CONFIG  # noqa: E402
from wm.v4.v4_training.components import RMSNorm  # noqa: E402


class ChannelCrossAttention(nn.Module):
    """Per-timestep cross-attention over channel embeddings.

    Channels arrive as (B, S, n_channels) scalars. We project each scalar
    to a d_mm vector using a learned per-channel embedding * value, giving
    (B, S, n_channels, d_mm) -- but to keep memory tight we simplify to:
        channel_emb: (n_channels, d_mm) learned bias
        ch_features: (B, S, n_channels) scalar values
        kv_in:       (B, S, n_channels, d_mm) = channel_emb broadcast * ch_features
    Then collapse to (B*S, n_channels, d_mm) and run multi-head attention
    against q from h_seq.
    """

    def __init__(self, d_model: int, n_channels: int, d_mm: int = 128,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_channels = n_channels
        self.d_mm = d_mm
        self.n_heads = n_heads
        assert d_mm % n_heads == 0
        self.head_dim = d_mm // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Learned per-channel embeddings (the "ID" of each channel)
        self.channel_id = nn.Parameter(torch.randn(n_channels, d_mm) * 0.02)
        # Linear that turns scalar value into a feature contribution
        self.value_gate = nn.Linear(1, d_mm, bias=True)

        self.q_proj = nn.Linear(d_model, d_mm, bias=False)
        self.k_proj = nn.Linear(d_mm, d_mm, bias=False)
        self.v_proj = nn.Linear(d_mm, d_mm, bias=False)
        self.out_proj = nn.Linear(d_mm, d_model, bias=False)

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_seq: torch.Tensor, channels: torch.Tensor) -> torch.Tensor:
        # h_seq:    (B, S, d_model)
        # channels: (B, S, n_channels)
        B, S, _ = h_seq.shape
        h_norm = self.norm(h_seq)
        # Build channel KV per (b, s):
        # channel_id (C, d_mm) + value_gate(channels[..., :, None])  -> (B, S, C, d_mm)
        ch = self.value_gate(channels.unsqueeze(-1))            # (B, S, C, d_mm)
        ch = ch + self.channel_id.unsqueeze(0).unsqueeze(0)     # broadcast bias

        # Q: (B, S, d_mm) -> (B, S, H, hd)
        q = self.q_proj(h_norm).view(B, S, self.n_heads, self.head_dim)
        # K, V over channels: (B, S, C, d_mm) -> (B, S, C, H, hd)
        k = self.k_proj(ch).view(B, S, self.n_channels, self.n_heads, self.head_dim)
        v = self.v_proj(ch).view(B, S, self.n_channels, self.n_heads, self.head_dim)

        # Scores: (B, S, H, C) = einsum("bshd,bschd->bshc")
        scores = torch.einsum("bshd,bschd->bshc", q, k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # Context: (B, S, H, hd) = einsum("bshc,bschd->bshd")
        ctx = torch.einsum("bshc,bschd->bshd", attn, v)
        ctx = ctx.reshape(B, S, self.d_mm)
        out = self.out_proj(ctx)
        return h_seq + self.dropout(out)


class MultiModalAdapter(nn.Module):
    """Frozen foundation + cross-attn adapter + new return heads.

    All foundation parameters are frozen (requires_grad=False). Only the
    adapter (~2M params) and new return heads train.
    """

    def __init__(
        self,
        foundation: FoundationBackbone,
        n_channels: int,
        d_mm: int = 128,
        n_heads: int = 4,
        n_layers_xattn: int = 2,
    ):
        super().__init__()
        self.foundation = foundation
        # Freeze foundation
        for p in self.foundation.parameters():
            p.requires_grad = False
        self.foundation.eval()

        d_model = foundation.cfg["d_model"]
        self.horizons = foundation.horizons
        self.num_bins = foundation.cfg["num_bins"]

        self.adapter_layers = nn.ModuleList([
            ChannelCrossAttention(d_model, n_channels, d_mm=d_mm,
                                    n_heads=n_heads, dropout=0.1)
            for _ in range(n_layers_xattn)
        ])
        self.adapter_norm = RMSNorm(d_model)

        # NEW return heads (don't reuse foundation's; they were trained on
        # bar-only data and may not transfer cleanly).
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, self.num_bins)
            for h in self.horizons
        })

    def adapter_params(self) -> list:
        """Return parameters that should train (foundation excluded)."""
        out = []
        for m in self.adapter_layers:
            out.extend(list(m.parameters()))
        out.extend(list(self.adapter_norm.parameters()))
        for m in self.return_heads.values():
            out.extend(list(m.parameters()))
        return out

    def num_adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapter_params() if p.requires_grad)

    def forward(
        self,
        x_seq: torch.Tensor,
        channels: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Frozen foundation forward (no_grad to save memory)
        with torch.no_grad():
            out = self.foundation(x_seq, asset_ids=asset_ids)
        h = out["h_seq"]                                   # (B, S, d_model) frozen
        # h is detached because we computed under no_grad; clone it so the
        # adapter's gradient flow doesn't try to backprop into the frozen base.
        h = h.detach()

        for layer in self.adapter_layers:
            h = layer(h, channels)
        h = self.adapter_norm(h)
        h_last = h[:, -1, :]
        return_logits = {f"h{h_}": self.return_heads[f"h{h_}"](h_last)
                          for h_ in self.horizons}
        return {"h_seq": h, "return_logits": return_logits}


def smoke():
    """Construct a small foundation + adapter, verify forward + param count."""
    import torch as _t
    foundation = FoundationBackbone(n_features=34, config=DEFAULT_CONFIG)
    adapter = MultiModalAdapter(foundation, n_channels=5, d_mm=128, n_layers_xattn=2)
    print(f"[mm-adapter] foundation params: {foundation.num_params():,} (frozen)")
    print(f"[mm-adapter] adapter params:    {adapter.num_adapter_params():,}")
    B, S = 2, 64
    x = _t.randn(B, S, 34)
    ch = _t.randn(B, S, 5)
    a_ids = _t.randint(0, 50, (B,))
    out = adapter(x, channels=ch, asset_ids=a_ids)
    print(f"[mm-adapter] h_seq: {tuple(out['h_seq'].shape)}")
    for hk, lg in out["return_logits"].items():
        print(f"[mm-adapter] return_logits[{hk}]: {tuple(lg.shape)}")


if __name__ == "__main__":
    smoke()
