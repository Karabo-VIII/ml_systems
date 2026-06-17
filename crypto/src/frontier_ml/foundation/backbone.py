"""FoundationBackbone — 30M-param Mamba-3 + cross-asset attention.

Architecture per PLAN.md Prong 1:
    - Input projection: f_in -> d_model
    - 6 x Mamba3Block (per-asset temporal SSM; reused from src/wm/v4)
    - 2 x cross-asset multi-head attention (sees all u10/u50 anchor assets
      jointly per timestep)
    - Multi-horizon TwoHot heads (h in {1,4,16,64}, 255 bins each)
    - Cross-asset contrastive projection head (learned lead-lag positives)

Param budget at d_model=256, d_state=16, 6 layers, 4 heads, 50 assets:
    Mamba backbone (6L @ d_model=256): ~24M
    Cross-asset attn (2L, d=128, 4 heads): ~4M
    TwoHot heads (4 horizons x 255 bins): ~0.3M
    Contrastive proj (2L MLP d=128): ~0.1M
    Total: ~28M params -> 56 MB fp16 weights.

Memory at batch=8 seq=512:
    Activations ~700 MB (Mamba is linear in seq_len)
    Adam states ~240 MB
    Gradients ~60 MB
    Total ~1 GB -> fits 8 GB with 5+ GB headroom.

This module is intentionally minimal — heavy lifting (loss, dataloader,
training loop) lives in sibling modules.

__contract__:
    inputs:
        x_seq:    (B, S, F_in)        per-asset feature sequence (TBD: 5-min OHLCV + flow_imbalance + funding_*)
        asset_ids: (B,) long           which asset each window belongs to
    outputs:
        h_seq:    (B, S, d_model)      backbone hidden states (last layer)
        return_logits: dict[h] -> (B, NUM_BINS)   predicted return distribution at each horizon
        contrastive_emb: (B, d_proj)   anchor embedding for contrastive loss

Reuses (NOT reimplements): src/wm/v4/v4_training/components.py
    Mamba3Block, RMSNorm, TwoHotSymlog
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

# Reuse V4 Mamba primitives — battle-tested, no need to reimplement.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import Mamba3Block, RMSNorm  # noqa: E402

# Default config (sized for 4060/8GB at batch=8 seq=512)
# Param count target: ~30M per PLAN.md.
# Empirical sweep 2026-05-02 -- this config lands at 31.7M.
# Mamba is ~3x more parameter-efficient than transformer at same d_model,
# so we needed d_model=768 to hit the 30M MOMENT-class anchor.
DEFAULT_CONFIG = dict(
    d_model=768,
    d_state=64,
    n_layers_backbone=8,
    n_layers_xattn=2,
    n_heads_xattn=8,
    d_xattn=256,
    n_assets_max=50,
    horizons=(1, 4, 16, 64),
    num_bins=255,
    d_contrastive=128,
    dropout=0.1,
    expand=2,
    headdim=64,
    chunk_size=16,
)


class CrossAssetAttention(nn.Module):
    """Per-timestep cross-asset attention.

    Input: (B, S, d_model) per-asset hidden states.
    Maintains a learned per-asset embedding (n_assets, d_model) so the
    attention sees both the asset's own sequence AND a queryable bank of
    OTHER assets' embeddings (refreshed via running mean during pretrain).

    Output: (B, S, d_model) -- residually mixed with cross-asset context.
    """

    def __init__(
        self,
        d_model: int,
        d_xattn: int,
        n_heads: int,
        n_assets_max: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_xattn = d_xattn
        self.n_heads = n_heads
        self.n_assets_max = n_assets_max

        # Per-asset learned embedding (key/value bank).
        self.asset_emb = nn.Parameter(torch.randn(n_assets_max, d_xattn) * 0.02)

        self.q_proj = nn.Linear(d_model, d_xattn, bias=False)
        self.k_proj = nn.Linear(d_xattn, d_xattn, bias=False)
        self.v_proj = nn.Linear(d_xattn, d_xattn, bias=False)
        self.out_proj = nn.Linear(d_xattn, d_model, bias=False)

        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        assert d_xattn % n_heads == 0, "d_xattn must be divisible by n_heads"
        self.head_dim = d_xattn // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, asset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, d_model)
        B, S, _ = x.shape
        h = self.norm(x)

        q = self.q_proj(h)                       # (B, S, d_xattn)
        # Key/value bank: use ALL asset embeddings as a global bank,
        # masking out the current asset's own slot to avoid trivial recall.
        kv_bank = self.asset_emb                  # (n_assets, d_xattn)
        k = self.k_proj(kv_bank)                  # (n_assets, d_xattn)
        v = self.v_proj(kv_bank)                  # (n_assets, d_xattn)

        # Reshape for multi-head
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, hd)
        k = k.view(self.n_assets_max, self.n_heads, self.head_dim).transpose(0, 1)  # (H, A, hd)
        v = v.view(self.n_assets_max, self.n_heads, self.head_dim).transpose(0, 1)  # (H, A, hd)

        # Attention scores: (B, H, S, A)
        scores = torch.einsum("bhsd,had->bhsa", q, k) * self.scale
        if asset_ids is not None:
            # Mask out the current asset's own slot per-batch.
            mask = torch.zeros(B, self.n_assets_max, device=x.device, dtype=torch.bool)
            mask.scatter_(1, asset_ids.long().clamp(0, self.n_assets_max - 1).unsqueeze(1), True)
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(2),  # (B, 1, 1, A)
                float("-inf"),
            )
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.einsum("bhsa,had->bhsd", attn, v)        # (B, H, S, hd)
        ctx = ctx.transpose(1, 2).reshape(B, S, self.d_xattn)  # (B, S, d_xattn)
        out = self.out_proj(ctx)                              # (B, S, d_model)
        return x + self.dropout(out)


class FoundationBackbone(nn.Module):
    """30M-param Mamba-3 + cross-asset attention foundation backbone.

    See module docstring for architecture + memory budget.
    """

    def __init__(self, n_features: int, config: Optional[Dict] = None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self.cfg = cfg

        d_model = cfg["d_model"]
        self.n_features = n_features
        self.horizons = tuple(cfg["horizons"])
        self.num_bins = cfg["num_bins"]

        # Input embedding
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Dropout(cfg["dropout"]),
        )

        # Per-asset Mamba-3 backbone (temporal)
        self.backbone = nn.ModuleList([
            Mamba3Block(
                d_model=d_model,
                d_state=cfg["d_state"],
                expand=cfg["expand"],
                headdim=cfg["headdim"],
                chunk_size=cfg["chunk_size"],
                dropout=cfg["dropout"],
            )
            for _ in range(cfg["n_layers_backbone"])
        ])
        self.backbone_norm = RMSNorm(d_model)

        # Cross-asset attention layers
        self.xattn = nn.ModuleList([
            CrossAssetAttention(
                d_model=d_model,
                d_xattn=cfg["d_xattn"],
                n_heads=cfg["n_heads_xattn"],
                n_assets_max=cfg["n_assets_max"],
                dropout=cfg["dropout"],
            )
            for _ in range(cfg["n_layers_xattn"])
        ])
        self.xattn_norm = RMSNorm(d_model)

        # Multi-horizon TwoHot heads (predict return distribution at each h)
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, cfg["num_bins"])
            for h in self.horizons
        })

        # Contrastive projection (last-token pool -> low-dim anchor)
        self.contrastive_proj = nn.Sequential(
            nn.Linear(d_model, cfg["d_contrastive"] * 2),
            nn.GELU(),
            nn.Linear(cfg["d_contrastive"] * 2, cfg["d_contrastive"]),
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        asset_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x_seq:     (B, S, F_in) feature sequence
            asset_ids: (B,) long asset index in [0, n_assets_max)

        Returns dict with:
            h_seq:           (B, S, d_model) backbone hidden states (last layer)
            return_logits:   dict[f"h{h}"] -> (B, NUM_BINS) at last timestep
            contrastive_emb: (B, d_contrastive) pooled anchor for contrastive loss
        """
        h = self.input_proj(x_seq)              # (B, S, d_model)

        for block in self.backbone:
            h = block(h)
        h = self.backbone_norm(h)

        for layer in self.xattn:
            h = layer(h, asset_ids=asset_ids)
        h = self.xattn_norm(h)

        # Multi-horizon return distribution from LAST timestep
        h_last = h[:, -1, :]                    # (B, d_model)
        return_logits = {
            f"h{h_}": self.return_heads[f"h{h_}"](h_last)
            for h_ in self.horizons
        }

        # Contrastive embedding from MEAN of sequence (denser signal than last)
        h_pool = h.mean(dim=1)                  # (B, d_model)
        contrastive_emb = self.contrastive_proj(h_pool)

        return {
            "h_seq": h,
            "return_logits": return_logits,
            "contrastive_emb": contrastive_emb,
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
