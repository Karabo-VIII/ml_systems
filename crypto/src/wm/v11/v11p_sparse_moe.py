"""V11' — Sparse Mixture of Experts upgrade for V11 (B005-aligned probe).

Per B005 — V11's current 3-dense-expert architecture is below the sparse-MoE
threshold. Honest upgrade attempt (per user 2026-05-02 directive: don't
archive prematurely): scale to 16 experts at smaller per-expert size with
top-2 routing + auxiliary load-balancing loss (Shazeer 2017 / Switch
Transformer / Llama 4 Maverick recipe).

Design:
- 16 experts at 256K params each = ~4.1M expert params total
- Top-2 routing (each token attends to 2 experts; outputs averaged)
- Aux load-balancing loss to prevent expert collapse
- ~8M total params (V11 baseline 2.9M; V11' = 8M)

This is a STANDALONE backbone module. Trainer integration follows the
V1.x apply_v1_upgrades pattern (this module exposes return_heads,
ret_trunk in outputs, and supports the same flag interface).

Decision rule per B005 plan: V11' must beat V11 baseline IC by >= +0.01
in an A/B at matched compute budget; else V11 stays frozen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import TwoHotSymlog, RMSNorm  # noqa: E402


class SparseMoE(nn.Module):
    """Top-2 routed MoE with load-balancing aux loss.

    Each expert is a 2-layer MLP at d_model -> d_expert -> d_model.
    """

    def __init__(self, d_model: int, n_experts: int = 16, top_k: int = 2,
                 d_expert: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_experts
        self.top_k = top_k

        # Router: linear projection to expert logits
        self.router = nn.Linear(d_model, n_experts, bias=False)

        # Experts (each a small MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_expert),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_expert, d_model),
            )
            for _ in range(n_experts)
        ])

        # Track for load-balancing loss
        self._last_load: torch.Tensor = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model). Returns (..., d_model) + load_balance_loss tracked."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)             # (N, d_model)
        N = x_flat.shape[0]

        # Router scores
        logits = self.router(x_flat)                      # (N, n_experts)
        # Top-k routing
        topk_logits, topk_idx = logits.topk(self.top_k, dim=-1)  # (N, top_k)
        topk_w = F.softmax(topk_logits, dim=-1)            # (N, top_k)

        # Compute output via gathered expert calls (efficient batching)
        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            # Mask: which tokens routed to expert i in their top-k
            mask = (topk_idx == i)
            if not mask.any():
                continue
            # For each token, find slot in top_k where this expert appears
            # Gather rows that route here, weighted by their corresponding topk_w
            tok_mask = mask.any(dim=-1)                   # (N,) tokens routing here
            tok_idx = tok_mask.nonzero(as_tuple=True)[0]
            # weight per token = sum of topk_w over slots where this expert appears
            slot_w = (mask.float() * topk_w).sum(dim=-1)[tok_idx]
            x_subset = x_flat[tok_idx]
            out[tok_idx] += slot_w.unsqueeze(-1) * expert(x_subset)

        # Load-balancing aux: encourage uniform token distribution across experts
        # f = fraction of tokens routed to each expert
        f = torch.zeros(self.n_experts, device=x.device)
        for i in range(self.n_experts):
            f[i] = (topk_idx == i).any(dim=-1).float().mean()
        # P = average router probability per expert
        P = F.softmax(logits, dim=-1).mean(dim=0)
        load_loss = (f * P).sum() * self.n_experts        # Switch Transformer aux
        self._last_load = load_loss.detach()

        return out.reshape(*orig_shape), load_loss


class V11pBackbone(nn.Module):
    """V11' — sparse-MoE backbone (16 experts, top-2 routing).

    Layer stack:
      Input embed (linear)
      4 x [LayerNorm + Self-Attention + Residual + LayerNorm + SparseMoE + Residual]
      Pool to (B, d_model)
      Multi-horizon TwoHot heads
    """

    def __init__(
        self,
        n_features: int = 34,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        n_experts: int = 16,
        d_expert: int = 256,
        top_k: int = 2,
        dropout: float = 0.1,
        num_bins: int = 255,
        horizons: tuple = (1, 4, 16, 64),
    ):
        super().__init__()
        self.d_model = d_model
        self.horizons = tuple(horizons)
        self.num_bins = num_bins
        self.n_experts = n_experts

        self.input_embed = nn.Linear(n_features, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "norm1": RMSNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True),
                "norm2": RMSNorm(d_model),
                "moe": SparseMoE(d_model, n_experts=n_experts, top_k=top_k,
                                  d_expert=d_expert, dropout=dropout),
            }))

        self.norm_out = RMSNorm(d_model)
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, num_bins) for h in self.horizons
        })

        # Frontier-ML hooks
        self._use_mtp = False
        self.mtp_head = None
        self._use_mdn = False

        self.bucketer = TwoHotSymlog(num_bins, -1.0, 1.0, "cuda")

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor = None) -> dict:
        h = self.input_embed(obs_seq)            # (B, T, d_model)
        load_losses = []
        for layer in self.layers:
            # Self-attention with residual
            h_norm = layer["norm1"](h)
            attn_out, _ = layer["attn"](h_norm, h_norm, h_norm, need_weights=False)
            h = h + attn_out
            # Sparse MoE with residual
            h_norm2 = layer["norm2"](h)
            moe_out, load_loss = layer["moe"](h_norm2)
            h = h + moe_out
            load_losses.append(load_loss)
        h = self.norm_out(h)
        h_pool = h[:, -1, :]                      # last-token pool

        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(h_pool)
            for hi in self.horizons:
                return_logits[hi] = mtp_out[f"h{hi}"]
        else:
            for hi in self.horizons:
                return_logits[hi] = self.return_heads[f"h{hi}"](h_pool)

        return {
            "return_logits": return_logits,
            "h_pool": h_pool,
            "ret_trunk": h_pool,
            "load_balance_loss": torch.stack(load_losses).mean(),
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke():
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model = V11pBackbone(n_features=34, d_model=256, n_layers=4, n_experts=16).to(DEV)
    print(f"[v11p] params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    B, T, Fc = 4, 96, 34
    obs = torch.randn(B, T, Fc, device=DEV)
    out = model.forward_train(obs)
    for k, v in out["return_logits"].items():
        assert v.shape == (B, model.num_bins)
    print(f"[v11p] return_logits OK; load_balance_loss = {out['load_balance_loss'].item():.4f}")

    loss = sum(v.float().pow(2).mean() for v in out["return_logits"].values())
    loss = loss + 0.01 * out["load_balance_loss"]
    loss.backward()
    print("[v11p] backward OK")
    print("[v11p] PASS smoke")


if __name__ == "__main__":
    smoke()
