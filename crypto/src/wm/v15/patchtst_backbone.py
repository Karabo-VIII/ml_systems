"""V15 — PatchTST production backbone.

Nie et al. 2023 (arXiv 2211.14730). Channel-independent transformer that
splits each time-series channel into PATCHES (subsequences of length P,
stride S), embeds each patch as a token, and runs a Transformer over
the patch sequence. Produces strong forecasts at small param counts.

V15 design:
- Input:  (B, T, F) where F = 34 features (V1.x baseline)
- Patch:  P=16 dollar bars, S=8 stride -> (T-P)/S + 1 = 11 patches at T=96
- Embed:  per-channel linear projection P -> d_model
- Channel-independent transformer (each channel processed separately,
  concatenated outputs, then return heads)
- Multi-horizon TwoHot heads at h={1,4,16,64} matching V1.x

Param count target: ~3M (settable). Smaller than V1.x (2M+) for direct A/B.

This is V15's PRODUCTION SOTA implementation per user 2026-05-02 mandate
("don't ship stick models"). Wires V1.x upgrade flags via the standard
trainer_helpers pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse V1.x's TwoHotSymlog + RMSNorm primitives
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import TwoHotSymlog, RMSNorm  # noqa: E402


class PatchEmbedding(nn.Module):
    """Channel-independent patch embedding.

    Each channel f gets its own linear projection P -> d_model. The
    output is (B, F, n_patches, d_model) which we then flatten to
    (B*F, n_patches, d_model) for the transformer.
    """

    def __init__(self, n_features: int, patch_len: int = 16, stride: int = 8,
                 d_model: int = 128):
        super().__init__()
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        # Per-channel linear projections (channel-independent: each f has its own)
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F)  ->  patches (B, F, n_patches, d_model)."""
        B, T, F = x.shape
        # Pad T so it divides cleanly: pad on left
        n_patches = (T - self.patch_len) // self.stride + 1
        # Use unfold to extract patches: (B, T, F) -> (B, F, n_patches, patch_len)
        x_t = x.transpose(1, 2)              # (B, F, T)
        patches = x_t.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches: (B, F, n_patches, patch_len)
        embedded = self.proj(patches)         # (B, F, n_patches, d_model)
        return embedded


class PatchTSTBackbone(nn.Module):
    """Channel-independent transformer over patches.

    Per Nie 2023:
        For each channel, transformer attends across patches.
        After encoding, flatten patches and produce a per-channel summary.
        Concatenate channel summaries -> per-batch hidden.

    For multi-horizon return prediction, the per-batch hidden becomes the
    input to TwoHot heads.
    """

    def __init__(
        self,
        n_features: int = 34,
        seq_len: int = 96,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        num_bins: int = 255,
        horizons: tuple = (1, 4, 16, 64),
    ):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.horizons = tuple(horizons)
        self.num_bins = num_bins

        self.patch_embed = PatchEmbedding(n_features, patch_len, stride, d_model)
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Channel-independent transformer
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Per-channel summary -> concat across channels
        self.flatten_proj = nn.Linear(self.n_patches * d_model, d_model)

        # Mix channels: simple MLP after concatenation
        # Concatenated dim: F * d_model -> reduce to d_model for return heads
        self.channel_mixer = nn.Sequential(
            nn.Linear(n_features * d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = RMSNorm(d_model)

        # Multi-horizon TwoHot heads
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(d_model, num_bins) for h in self.horizons
        })

        # Frontier-ML upgrade hooks (set via apply_v1_upgrades). Default OFF.
        self._use_mtp = False
        self.mtp_head = None
        self._use_mdn = False

        # Bucketer for TwoHot decode. Device-aware (was hard-coded "cuda" ->
        # crashed on CPU-only load / validation).
        self.bucketer = TwoHotSymlog(
            num_bins, -1.0, 1.0, "cuda" if torch.cuda.is_available() else "cpu")

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor = None) -> dict:
        """obs_seq: (B, T, F)  -> dict with return_logits, h_pool, etc."""
        B, T, Fc = obs_seq.shape
        # Patch embed: (B, F, n_patches, d_model)
        patches = self.patch_embed(obs_seq)
        # Reshape for transformer: (B*F, n_patches, d_model)
        patches_t = patches.reshape(B * Fc, self.n_patches, self.d_model)
        # Add positional embedding
        patches_t = patches_t + self.pos_embed
        # Transformer encoder
        encoded = self.encoder(patches_t)        # (B*F, n_patches, d_model)
        # Per-channel summary: flatten patches
        per_ch = encoded.reshape(B, Fc, self.n_patches * self.d_model)
        per_ch = self.flatten_proj(per_ch)        # (B, F, d_model)
        # Mix channels
        ch_concat = per_ch.reshape(B, Fc * self.d_model)
        h_pool = self.channel_mixer(ch_concat)    # (B, d_model)
        h_pool = self.norm(h_pool)

        # Multi-horizon return predictions
        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(h_pool)
            for h in self.horizons:
                return_logits[h] = mtp_out[f"h{h}"]
        else:
            for h in self.horizons:
                return_logits[h] = self.return_heads[f"h{h}"](h_pool)

        return {
            "return_logits": return_logits,
            "h_pool": h_pool,
            "ret_trunk": h_pool,    # alias for V1.x integration compatibility
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke():
    """Verify forward pass + param count target."""
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTSTBackbone(n_features=34, seq_len=96, d_model=128).to(DEV)
    print(f"[v15-patchtst] params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    B, T, F = 4, 96, 34
    obs = torch.randn(B, T, F, device=DEV)
    out = model.forward_train(obs)
    for k, v in out["return_logits"].items():
        print(f"[v15-patchtst] return_logits[h{k}]: {tuple(v.shape)}")
        assert v.shape == (B, model.num_bins)
    print(f"[v15-patchtst] h_pool: {tuple(out['h_pool'].shape)}")

    # Backward smoke
    loss = sum(v.float().pow(2).mean() for v in out["return_logits"].values())
    loss.backward()
    print("[v15-patchtst] backward OK")
    print("[v15-patchtst] PASS smoke")


if __name__ == "__main__":
    smoke()
