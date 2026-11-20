"""G10: PatchTST encoder (Nie et al., ICLR 2023).

Closes G10 from gap audit 2026-04-25.

Reference:
  Nie, Y., Nguyen, N., Sinthong, P., & Kalagnanam, J. (2023). A Time Series is
  Worth 64 Words: Long-term Forecasting with Transformers. ICLR.
  https://arxiv.org/abs/2211.14730

Design notes:
  - Channel-independent: each feature processed independently through the
    encoder, then aggregated. Avoids cross-channel attention overfitting on
    small datasets (the original paper's key win on financial data).
  - Patch-based: input sequence chopped into overlapping patches of length P
    with stride S. Reduces sequence length by P/S.
  - Drop-in encoder: returns [B, C, D_out] embeddings. Caller adds the
    prediction head matching its own loss/objective.

The project intentionally does NOT add a V15 training pipeline; this is provided
as a drop-in encoder block to be used inside V1-V14 frameworks if/when an
ablation is desired. This keeps the "signal-research-done" stance intact while
making the architecture available without re-deriving it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding -- vanilla Transformer."""

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]; add the first N positions
        return x + self.pe[:, : x.size(1), :]


class PatchTSTEncoder(nn.Module):
    """Channel-independent patch transformer encoder.

    Args:
      n_features: number of input channels C.
      seq_len: input sequence length L.
      patch_len: patch length P. Default 16.
      stride: stride S. Default 8 (50% overlap).
      d_model: embedding dim. Default 128.
      n_heads: attention heads. Default 4.
      n_layers: transformer encoder layers. Default 3.
      dropout: dropout rate. Default 0.1.
      output_mean_pool: if True, average-pool the patch dim and return
        [B, C, d_model]. If False, return [B, C, N, d_model] (N=num patches).
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        output_mean_pool: bool = True,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if seq_len < patch_len:
            raise ValueError(f"seq_len ({seq_len}) must be >= patch_len ({patch_len})")
        self.n_features = n_features
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.output_mean_pool = output_mean_pool
        self.num_patches = max(1, (seq_len - patch_len) // stride + 1)

        # Per-channel patch embedding: linear from patch_len -> d_model
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_enc = PositionalEncoding(d_model, max_len=max(self.num_patches, 64))
        # Transformer encoder shared across channels (channel-independence preserved
        # because each channel is processed in its own batch slice)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, L, C] input sequence.
        Returns:
          [B, C, d_model] if output_mean_pool, else [B, C, N, d_model].
        """
        B, L, C = x.shape
        if L != self.seq_len:
            # Allow shorter sequences via right-pad; longer via right-truncate.
            if L < self.seq_len:
                pad = self.seq_len - L
                x = nn.functional.pad(x, (0, 0, 0, pad))
            else:
                x = x[:, : self.seq_len, :]
        # [B, L, C] -> [B, C, L]
        x = x.transpose(1, 2)
        # Unfold into patches: [B, C, num_patches, patch_len]
        # nn.Tensor.unfold(dim, size, step)
        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        # [B, C, N, P]
        # Channel-independent processing: flatten (B, C) into batch
        N = x.size(2)
        x = x.reshape(B * C, N, self.patch_len)
        # Embed each patch
        x = self.patch_embed(x)  # [B*C, N, D]
        x = self.dropout(self.pos_enc(x))
        # Transformer
        x = self.encoder(x)
        x = self.norm(x)
        # [B*C, N, D]
        if self.output_mean_pool:
            x = x.mean(dim=1)  # [B*C, D]
            x = x.reshape(B, C, self.d_model)
        else:
            x = x.reshape(B, C, N, self.d_model)
        return x


class PatchTSTReturnHead(nn.Module):
    """Combine PatchTSTEncoder output -> 1-step return prediction.

    Aggregates per-channel embeddings via mean-pool then linear head.
    """

    def __init__(self, encoder: PatchTSTEncoder, n_outputs: int = 1, hidden: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        z = self.encoder(x)        # [B, C, D]
        z = z.mean(dim=1)          # [B, D] feature-mean pool
        return self.head(z)        # [B, n_outputs]


# -----------------------------------------------------------------------------
# Quick smoke test
# -----------------------------------------------------------------------------

def _smoke_test() -> None:
    """Run on CPU to verify shapes."""
    torch.manual_seed(42)
    B, L, C = 4, 96, 18  # batch, seq_len=96 bars, 18 features
    enc = PatchTSTEncoder(n_features=C, seq_len=L, patch_len=16, stride=8,
                          d_model=128, n_heads=4, n_layers=3)
    head = PatchTSTReturnHead(enc, n_outputs=1)
    x = torch.randn(B, L, C)
    out = head(x)
    assert out.shape == (B, 1), f"head shape {out.shape}"

    # backward pass
    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    n_params = sum(p.numel() for p in head.parameters())
    print(f"[g10] PatchTST encoder smoke test PASS: B={B} L={L} C={C} "
          f"-> out{tuple(out.shape)}, params={n_params:,}")
    print(f"[g10] num_patches={enc.num_patches}, patch_embed_in_dim={enc.patch_len}")


def main_cli():
    """STUB: V15 is a drop-in encoder, NOT a full trainer.

    `run_all_training.py --features 121 --model v15` will route here and
    exit with a clear "stub" status so the runner doesn't silently no-op.
    Use the encoder by importing PatchTSTEncoder / PatchTSTReturnHead from
    a parent training script (e.g. V1.6 retrofit experiment).
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="V15 PatchTST encoder (drop-in stub; not a trainer). "
                    "Run --smoke for the parameter-count smoke check."
    )
    parser.add_argument("--features", type=int, default=121,
                        help="Pseudo-feature count for run_all_training compat. "
                             "V15 ignores it; the encoder's input_dim is set by the parent.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the original smoke test from this file.")
    args = parser.parse_args()
    if args.smoke:
        _smoke_test()
        return
    print("[V15] STUB: PatchTST encoder is library-only. Import "
          "PatchTSTEncoder/PatchTSTReturnHead from a parent training script.")
    print("[V15] use --smoke to run the parameter-count smoke test.")


if __name__ == "__main__":
    main_cli()
