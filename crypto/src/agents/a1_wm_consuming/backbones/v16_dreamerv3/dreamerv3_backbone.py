"""V16 — DreamerV3 production backbone for crypto WM.

Hafner et al. 2025 (Nature). World-model RL across diverse domains.
Adapted to the crypto WM regime: instead of acting in an environment with
sparse rewards, we use REALIZED RETURN as a dense supervised signal —
which collapses the actor/critic step to a regression head while keeping
DreamerV3's RSSM dynamics + symlog targets + reward+value+continuation
heads.

Design (3-component world model):
- Encoder: f34 -> d_model (per-timestep)
- RSSM: deterministic GRU + stochastic discrete latent (32x32 categorical)
- Heads: reward (return), value (cumulative-return), continuation (always 1
  for non-episodic markets), and decoder (recon)

Training objective (DreamerV3 paper):
  L_pred = recon + reward + continuation + value(symlog) + KL(post||prior)
  All terms with symlog targets and free-bits KL.

For our crypto WM, "reward" = next-bar return, "value" = sum of K-step
discounted returns. The actor/critic loop is omitted -- this is the
WORLD-MODEL component only, used as a backbone for prediction-tier IC.

This is V16 SOTA (per user 2026-05-02: "not stick models"). Forward +
backward verified; trainer integration follows the V1.x apply_v1_upgrades
pattern.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reclassified 2026-06-11: was src/wm/v16/ (parents[3]=repo root); now
# src/agents/a1_wm_consuming/backbones/v16_dreamerv3/ (parents[5]=repo root).
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from wm.v4.v4_training.components import TwoHotSymlog, RMSNorm  # noqa: E402


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class DreamerRSSM(nn.Module):
    """Recurrent State-Space Model: GRU deterministic + categorical stochastic.

    state = (h, z)  where  h = GRU(prev_h, prev_z, action_or_input)
                            z = sample from categorical(prior_logits OR posterior_logits)
    """

    def __init__(self, d_input: int, d_hidden: int, n_categories: int = 32,
                 n_classes: int = 32):
        super().__init__()
        self.d_hidden = d_hidden
        self.n_categories = n_categories
        self.n_classes = n_classes
        flat_dim = n_categories * n_classes

        # Embed previous z + input into GRU hidden update
        self.gru_in_proj = nn.Linear(flat_dim + d_input, d_hidden)
        self.gru_cell = nn.GRUCell(d_hidden, d_hidden)
        # Prior: predict z from h alone
        self.prior_proj = nn.Linear(d_hidden, flat_dim)
        # Posterior: predict z from h + observation
        self.posterior_proj = nn.Linear(d_hidden + d_input, flat_dim)
        self.norm = RMSNorm(d_hidden)

    def step(self, h_prev: torch.Tensor, z_prev: torch.Tensor,
             obs: torch.Tensor) -> dict:
        """One RSSM step.

        Args:
            h_prev: (B, d_hidden)
            z_prev: (B, n_cat * n_class) flat prev stochastic state
            obs:    (B, d_input) current observation embedding

        Returns dict with: h, z_post, prior_logits, post_logits.
        """
        gru_in = self.gru_in_proj(torch.cat([z_prev, obs], dim=-1))
        h = self.gru_cell(gru_in, h_prev)
        h = self.norm(h)
        prior_logits = self.prior_proj(h).view(-1, self.n_categories, self.n_classes)
        post_logits = self.posterior_proj(torch.cat([h, obs], dim=-1)).view(
            -1, self.n_categories, self.n_classes)
        # Sample posterior via straight-through Gumbel
        post_probs = F.softmax(post_logits, dim=-1)
        if self.training:
            z_post = F.gumbel_softmax(post_logits, tau=1.0, hard=True, dim=-1)
        else:
            z_post = F.one_hot(post_probs.argmax(dim=-1),
                                num_classes=self.n_classes).float()
        return {
            "h": h,
            "z_post": z_post.view(-1, self.n_categories * self.n_classes),
            "prior_logits": prior_logits,
            "post_logits": post_logits,
        }


class V16DreamerWM(nn.Module):
    """DreamerV3 world-model backbone for crypto.

    Forward train returns return_logits at h={1,4,16,64} from the
    posterior latent at each timestep. Suitable for V1.x trainer
    integration via apply_v1_upgrades.
    """

    def __init__(
        self,
        n_features: int = 34,
        d_model: int = 256,
        d_hidden: int = 256,
        n_categories: int = 32,
        n_classes: int = 32,
        num_bins: int = 255,
        horizons: tuple = (1, 4, 16, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_categories = n_categories
        self.n_classes = n_classes
        self.horizons = tuple(horizons)
        self.num_bins = num_bins
        flat_dim = n_categories * n_classes

        self.encoder = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.rssm = DreamerRSSM(
            d_input=d_model, d_hidden=d_hidden,
            n_categories=n_categories, n_classes=n_classes,
        )

        feat_dim = d_hidden + flat_dim

        # Decoder for reconstruction (DreamerV3 standard auxiliary loss)
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_features),
        )
        # Continuation head (always ~1 for non-episodic markets but kept for
        # DreamerV3 fidelity; sigmoid output)
        self.continue_head = nn.Linear(feat_dim, 1)

        # Multi-horizon return heads (reward + value combined into per-h logits)
        self.return_heads = nn.ModuleDict({
            f"h{h}": nn.Linear(feat_dim, num_bins) for h in self.horizons
        })

        # Frontier-ML hooks
        self._use_mtp = False
        self.mtp_head = None
        self._use_mdn = False
        self.bucketer = TwoHotSymlog(num_bins, -1.0, 1.0,
                                     "cuda" if torch.cuda.is_available() else "cpu")

    def forward_train(self, obs_seq: torch.Tensor, asset_id: torch.Tensor = None) -> dict:
        B, T, F_in = obs_seq.shape
        DEV = obs_seq.device
        obs_emb = self.encoder(obs_seq)              # (B, T, d_model)

        h = torch.zeros(B, self.d_hidden, device=DEV, dtype=obs_emb.dtype)
        z = torch.zeros(B, self.n_categories * self.n_classes,
                        device=DEV, dtype=obs_emb.dtype)
        h_seq, z_seq = [], []
        prior_logits_seq, post_logits_seq = [], []
        for t in range(T):
            step = self.rssm.step(h, z, obs_emb[:, t, :])
            h = step["h"]
            z = step["z_post"]
            h_seq.append(h)
            z_seq.append(z)
            prior_logits_seq.append(step["prior_logits"])
            post_logits_seq.append(step["post_logits"])

        h_seq_t = torch.stack(h_seq, dim=1)            # (B, T, d_hidden)
        z_seq_t = torch.stack(z_seq, dim=1)            # (B, T, flat_dim)
        feat = torch.cat([h_seq_t, z_seq_t], dim=-1)    # (B, T, feat_dim)

        # Decoder + continuation
        recon = self.decoder(feat)
        cont = torch.sigmoid(self.continue_head(feat))

        # Multi-horizon return logits
        return_logits = {}
        if self._use_mtp and self.mtp_head is not None:
            mtp_out = self.mtp_head(feat)
            for hi in self.horizons:
                return_logits[hi] = mtp_out[f"h{hi}"]
        else:
            for hi in self.horizons:
                return_logits[hi] = self.return_heads[f"h{hi}"](feat)

        return {
            "return_logits": return_logits,
            "h_seq": h_seq_t,
            "z_post": z_seq_t,
            "ret_trunk": feat,                           # alias for V1.x compat
            "recon": recon,
            "continue": cont,
            "prior_logits": torch.stack(prior_logits_seq, dim=1),
            "post_logits": torch.stack(post_logits_seq, dim=1),
        }

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def smoke():
    torch.manual_seed(0)
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    if DEV == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.30)
    model = V16DreamerWM(n_features=34, d_model=256).to(DEV)
    print(f"[v16-dreamer] params: {model.num_params():,} ({model.num_params()/1e6:.2f}M)")

    B, T = 2, 32  # short sequence for smoke
    obs = torch.randn(B, T, 34, device=DEV)
    out = model.forward_train(obs)
    for k, v in out["return_logits"].items():
        assert v.shape == (B, T, model.num_bins), f"bad shape {k}: {v.shape}"
    print(f"[v16-dreamer] return_logits OK; recon: {tuple(out['recon'].shape)}")
    print(f"[v16-dreamer] continue: {tuple(out['continue'].shape)}")
    loss = sum(v.float().pow(2).mean() for v in out["return_logits"].values())
    loss.backward()
    print("[v16-dreamer] backward OK")
    print("[v16-dreamer] PASS smoke")


if __name__ == "__main__":
    smoke()
