"""
V6 Components -- Causal JEPA with Adversarial Time Shuffling Building Blocks (SOTA 2025/26)

Contains:
  - RMSNorm: Root Mean Square normalization (LLaMA/Mistral style)
  - CausalGRUEncoder: Unidirectional GRU for causal context encoding (NOT BiGRU)
  - PredictorNetwork: Predict future embeddings from context
  - TimeDiscriminator: Adversarial classifier for temporal coherence detection
  - InfoNCELoss: MEMORY-SAFE per-timestep contrastive loss [B x B] not [B*T x B*T]
  - VICRegLoss: Variance-Invariance-Covariance regularization (prevents collapse)
  - TwoHotSymlog: Shared discretized regression encoding (255 bins, [-1, 1])
  - SwiGLU: Gated Linear Unit with SiLU
  - MLPHead: Standard MLP head

CRITICAL FIX vs V2:
  - CausalGRUEncoder replaces BiGRUEncoder (bidirectional=False, full hidden_dim)
  - TimeDiscriminator penalizes temporal dependence in latent space
  - InfoNCE unchanged: per-timestep [B x B] similarity matrices (memory-safe)

SOTA 2025/26:
  - RMSNorm replacing LayerNorm (Zhang & Sennrich, 2019)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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


# =============================================================================
# ENCODER NETWORKS
# =============================================================================

class CausalGRUEncoder(nn.Module):
    """
    Causal (unidirectional) GRU encoder for sequence context.

    CRITICAL FIX from V2's BiGRU:
      V2 used bidirectional GRU which caused catastrophic temporal overfitting --
      the model could "see the future" during training, leading to representations
      that depended on future information unavailable at inference time.

      V6 uses strictly causal (forward-only) GRU so representations at time t
      only depend on observations at times <= t.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3, dropout: float = 0.22):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,           # Full hidden_dim (NOT //2 like BiGRU)
            n_layers,
            batch_first=True,
            bidirectional=False,  # CAUSAL: forward-only
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = RMSNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            h: [B, T, hidden_dim]
        """
        h, _ = self.gru(x)
        return self.norm(h)


# =============================================================================
# PREDICTOR NETWORK
# =============================================================================

class PredictorNetwork(nn.Module):
    """Predicts future embeddings from context."""

    def __init__(self, context_dim: int, latent_dim: int, n_layers: int = 2, dropout: float = 0.15):
        super().__init__()

        layers = []
        in_dim = context_dim
        for i in range(n_layers):
            out_dim = latent_dim if i == n_layers - 1 else context_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(RMSNorm(out_dim))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(dropout))
            else:
                # Final layer: just Linear -> RMSNorm, no activation (output is a prediction)
                pass
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


# =============================================================================
# TIME DISCRIMINATOR -- NEW FOR V6
# =============================================================================

class TimeDiscriminator(nn.Module):
    """
    Adversarial discriminator that classifies whether a latent sequence is
    temporally coherent (real) or time-shuffled (fake).

    Architecture:
      1. Small CausalGRU processes the latent sequence [B, T, D] -> [B, T, H]
      2. Mean pooling over time: [B, T, H] -> [B, H]
      3. MLP head: [B, H] -> [B, 1] with sigmoid

    The discriminator is trained to output high probability for real (temporally
    coherent) sequences and low probability for shuffled sequences. The encoder
    is adversarially trained to fool the discriminator -- i.e., produce latents
    whose temporal structure is indistinguishable from shuffled sequences.

    This prevents the encoder from encoding spurious temporal patterns that
    cause overfitting to the training set's specific temporal ordering.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()

        # Small causal GRU to capture temporal patterns
        self.gru = nn.GRU(
            latent_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.norm = RMSNorm(hidden_dim)

        # MLP head: hidden_dim -> 1 scalar probability
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_seq: [B, T, D] - latent sequence (real or shuffled)
        Returns:
            prob: [B] - probability that the sequence is temporally coherent
        """
        h, _ = self.gru(latent_seq)            # [B, T, hidden_dim]
        h = self.norm(h)                        # [B, T, hidden_dim]
        h_pooled = h.mean(dim=1)                # [B, hidden_dim] -- mean pooling over time
        logit = self.head(h_pooled).squeeze(-1)  # [B]
        return torch.sigmoid(logit)


# =============================================================================
# CONTRASTIVE LOSS -- MEMORY-SAFE VERSION
# =============================================================================

class InfoNCELoss(nn.Module):
    """
    Memory-safe InfoNCE contrastive loss for JEPA.

    CRITICAL FIX: Computes per-timestep [B x B] similarity matrices instead of
    a single [B*T x B*T] matrix.

    Memory comparison (B=40, T=96):
      OLD: [3840 x 3840] = 14.7M floats = 59MB  -> OOM risk on 8GB GPU
      NEW: 96 x [40 x 40] = 154K floats = 0.6MB -> perfectly fine

    For each timestep t, the positive pair is (pred[b, t], target[b, t]) and
    negatives are all other batch elements at the same timestep:
    (pred[b, t], target[b', t]) for b' != b.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        """
        Args:
            pred:   [B, T, D] - predicted embeddings from online encoder + predictor
            target: [B, T, D] - target embeddings from EMA encoder (detached)
        Returns:
            (loss, accuracy) - scalar loss and top-1 contrastive accuracy
        """
        B, T, D = pred.shape

        # Normalize along feature dimension
        pred_norm = F.normalize(pred, dim=-1)      # [B, T, D]
        target_norm = F.normalize(target, dim=-1)   # [B, T, D]

        total_loss = 0.0
        total_correct = 0

        # Process each timestep independently -> [B x B] similarity per timestep
        # This is the key memory optimization: O(B^2 * T) instead of O(B^2 * T^2)
        for t in range(T):
            p = pred_norm[:, t, :]      # [B, D]
            tgt = target_norm[:, t, :]  # [B, D]

            # Similarity matrix: [B, B]
            logits = torch.mm(p, tgt.t()) / self.temperature

            # Diagonal elements are positives
            labels = torch.arange(B, device=pred.device)

            # Cross-entropy over batch dimension
            total_loss = total_loss + F.cross_entropy(logits, labels)

            # Top-1 accuracy
            total_correct = total_correct + (logits.argmax(dim=-1) == labels).sum()

        loss = total_loss / T
        accuracy = total_correct.float() / (B * T)

        return loss, accuracy


# =============================================================================
# VICREG LOSS -- COLLAPSE PREVENTION
# =============================================================================

class VICRegLoss(nn.Module):
    """
    Variance-Invariance-Covariance Regularization (Bardes et al., 2021).

    Prevents representation collapse in JEPA-style models by ensuring:
      1. Invariance: predicted and target embeddings should match (MSE)
      2. Variance: each embedding dimension should have std >= 1 across the batch
      3. Covariance: embedding dimensions should be decorrelated

    This is CRITICAL with small batches (B=40) where InfoNCE alone can collapse.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        sim_w: float = 25.0,
        var_w: float = 25.0,
        cov_w: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            z1: [N, D] - online encoder representations (flattened from [B, T, D])
            z2: [N, D] - target encoder representations (flattened from [B, T, D])
            sim_w: invariance loss weight
            var_w: variance loss weight
            cov_w: covariance loss weight
        Returns:
            Scalar VICReg loss
        """
        # -- Invariance: MSE between representations --
        sim_loss = F.mse_loss(z1, z2)

        # -- Variance: push std of each embedding dim above threshold 1.0 --
        z1_std = torch.sqrt(z1.var(dim=0) + 1e-4)
        z2_std = torch.sqrt(z2.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1.0 - z1_std)) + torch.mean(F.relu(1.0 - z2_std))

        # -- Covariance: decorrelate embedding dimensions --
        # Cast to fp32 before covariance matrix ops for numerical stability under AMP
        z1_c = (z1 - z1.mean(dim=0)).float()
        z2_c = (z2 - z2.mean(dim=0)).float()
        N = z1.shape[0]

        cov1 = (z1_c.T @ z1_c) / max(N - 1, 1)
        cov2 = (z2_c.T @ z2_c) / max(N - 1, 1)

        # Zero diagonal (we only penalize off-diagonal covariances)
        cov1 = cov1.fill_diagonal_(0)
        cov2 = cov2.fill_diagonal_(0)

        D = z1.shape[1]
        cov_loss = (cov1.pow(2).sum() + cov2.pow(2).sum()) / D

        return sim_w * sim_loss + var_w * var_loss + cov_w * cov_loss


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
