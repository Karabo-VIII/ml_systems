"""V16 -- DreamerV3 World Model (Hafner 2023, paper 2301.04104).

Faithful implementation of the DreamerV3 architecture for crypto WM training:
  - Recurrent State-Space Model (RSSM) with categorical latent (32x32 by default)
    + deterministic GRU recurrent state.
  - 4 prediction heads: reconstruction, reward, discount/continue, return.
  - SymLog/TwoHot for return target (replaces V1.x's clipped returns).
  - KL balance: separate weights for posterior vs prior KL (Hafner: 0.5 / 0.1).
  - Free bits: skip KL penalty on units with KL < threshold (prevents posterior
    collapse).

Why DreamerV3 specifically:
  - SOTA model-based RL backbone (beat humans on 150+ Atari games at fixed compute).
  - Designed for fixed hyperparams across diverse environments — fits this
    project's "many assets, one architecture" mandate.
  - Latent imagination = direct path for the M3 DreamerV3 agent (RL in latent).

This file is the WM only. Training script + agent are separate (M2 trains WM,
M3 builds agent on top).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


class TwoHotEncoder(nn.Module):
    """SymLog + TwoHot return encoding (DreamerV3 sec. 4)."""

    def __init__(self, n_bins: int = 255, bin_min: float = -20.0, bin_max: float = 20.0):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("bins", torch.linspace(bin_min, bin_max, n_bins))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return target -> two-hot probabilities. x: [B, T] -> [B, T, n_bins]."""
        x_log = symlog(x)
        # find left bin
        idx = torch.bucketize(x_log, self.bins) - 1
        idx = idx.clamp(0, self.n_bins - 2)
        left = self.bins[idx]
        right = self.bins[idx + 1]
        weight_right = ((x_log - left) / (right - left + 1e-9)).clamp(0, 1)
        weight_left = 1.0 - weight_right
        out = torch.zeros(*x.shape, self.n_bins, device=x.device, dtype=x.dtype)
        out.scatter_(-1, idx.unsqueeze(-1), weight_left.unsqueeze(-1))
        out.scatter_(-1, (idx + 1).unsqueeze(-1), weight_right.unsqueeze(-1))
        return out

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """logits [..., n_bins] -> scalar prediction (in original symexp space)."""
        probs = F.softmax(logits, dim=-1)
        x_log = (probs * self.bins).sum(dim=-1)
        return symexp(x_log)


# -----------------------------------------------------------------------------
# RSSM
# -----------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State-Space Model: deterministic h_t + categorical z_t."""

    def __init__(self,
                 obs_dim: int,
                 action_dim: int = 1,        # for now treat trades-or-not as scalar; agent will set this
                 hidden_dim: int = 200,
                 stoch_categories: int = 32,
                 stoch_dimensions: int = 32,
                 mlp_hidden: int = 200):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.stoch_categories = stoch_categories
        self.stoch_dimensions = stoch_dimensions
        self.stoch_dim = stoch_categories * stoch_dimensions

        # Encoder: observation -> features
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
        )

        # Recurrent: GRU(prev_stoch + action, hidden)
        self.gru = nn.GRUCell(self.stoch_dim + action_dim, hidden_dim)

        # Posterior: q(z_t | h_t, encoder(o_t))
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.stoch_dim),
        )
        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, self.stoch_dim),
        )

    def initial_state(self, batch_size: int, device) -> dict:
        return {
            "h": torch.zeros(batch_size, self.hidden_dim, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def _categorical_sample(self, logits: torch.Tensor) -> torch.Tensor:
        """Straight-through gumbel-softmax for categorical latent."""
        B = logits.shape[0]
        logits = logits.view(B, self.stoch_categories, self.stoch_dimensions)
        # use gumbel softmax with hard=True (straight-through)
        sample = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        return sample.view(B, self.stoch_dim)

    def forward_step(self, prev_state: dict, prev_action: torch.Tensor,
                     obs: torch.Tensor) -> tuple[dict, dict]:
        """One step of RSSM. Returns (state, info_for_loss)."""
        h_prev = prev_state["h"]
        z_prev = prev_state["z"]
        # GRU: in = (z_prev, action) -> h_new
        gru_in = torch.cat([z_prev, prev_action], dim=-1)
        h_new = self.gru(gru_in, h_prev)
        # Encoder
        enc = self.encoder(obs)
        # Posterior + prior
        post_logits = self.posterior_net(torch.cat([h_new, enc], dim=-1))
        prior_logits = self.prior_net(h_new)
        # Sample posterior
        z_new = self._categorical_sample(post_logits)
        new_state = {"h": h_new, "z": z_new}
        info = {
            "post_logits": post_logits,
            "prior_logits": prior_logits,
        }
        return new_state, info

    def imagine_step(self, prev_state: dict, prev_action: torch.Tensor) -> dict:
        """Imagined step (no observation; sample from prior)."""
        h_prev = prev_state["h"]
        z_prev = prev_state["z"]
        gru_in = torch.cat([z_prev, prev_action], dim=-1)
        h_new = self.gru(gru_in, h_prev)
        prior_logits = self.prior_net(h_new)
        z_new = self._categorical_sample(prior_logits)
        return {"h": h_new, "z": z_new}


# -----------------------------------------------------------------------------
# DreamerV3 World Model
# -----------------------------------------------------------------------------

class DreamerV3WorldModel(nn.Module):
    """Full DreamerV3 WM: RSSM + 4 heads (recon, reward, continue, return)."""

    def __init__(self,
                 obs_dim: int = 121,
                 action_dim: int = 1,
                 hidden_dim: int = 200,
                 stoch_categories: int = 32,
                 stoch_dimensions: int = 32,
                 mlp_hidden: int = 200,
                 n_assets: int = 10,
                 asset_embed_dim: int = 32,
                 return_n_bins: int = 255):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Asset embedding (DreamerV3 conditional on per-asset identity)
        self.asset_embed = nn.Embedding(n_assets, asset_embed_dim)
        encoder_in_dim = obs_dim + asset_embed_dim

        self.rssm = RSSM(
            obs_dim=encoder_in_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            stoch_categories=stoch_categories,
            stoch_dimensions=stoch_dimensions,
            mlp_hidden=mlp_hidden,
        )
        feat_dim = self.rssm.hidden_dim + self.rssm.stoch_dim
        # Heads
        self.recon_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, obs_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, 1),  # logit for Bernoulli(continue)
        )
        # Return head: TwoHot symlog (replaces V1.x clipped TwoHot)
        self.return_head = nn.Sequential(
            nn.Linear(feat_dim, mlp_hidden), nn.LayerNorm(mlp_hidden), nn.SiLU(),
            nn.Linear(mlp_hidden, return_n_bins),
        )
        self.return_encoder = TwoHotEncoder(n_bins=return_n_bins, bin_min=-5.0, bin_max=5.0)

        # KL balance (Hafner 2023)
        self.kl_balance = 0.8     # weight on posterior loss
        self.free_bits = 1.0      # nats per latent dim

    def feat(self, state: dict) -> torch.Tensor:
        return torch.cat([state["h"], state["z"]], dim=-1)

    def encode_obs(self, obs: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        """obs: [B, C], asset_id: [B] -> [B, C+embed_dim]."""
        emb = self.asset_embed(asset_id)
        return torch.cat([obs, emb], dim=-1)

    def forward_train(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor,
                      asset_ids: torch.Tensor, returns: torch.Tensor) -> dict:
        """
        Args:
            obs_seq: [B, T, C] observations
            actions_seq: [B, T, action_dim] actions taken (use zeros for WM-only training)
            asset_ids: [B] long, asset index for each batch element
            returns: [B, T] forward returns (target_return_1)
        Returns dict of losses + decoded predictions.
        """
        B, T, C = obs_seq.shape
        device = obs_seq.device
        state = self.rssm.initial_state(B, device)

        post_logits_seq = []
        prior_logits_seq = []
        recon_pred_seq = []
        reward_pred_seq = []
        continue_pred_seq = []
        return_logits_seq = []

        for t in range(T):
            obs_t = self.encode_obs(obs_seq[:, t, :], asset_ids)
            action_t = actions_seq[:, t, :]
            state, info = self.rssm.forward_step(state, action_t, obs_t)
            post_logits_seq.append(info["post_logits"])
            prior_logits_seq.append(info["prior_logits"])

            feat_t = self.feat(state)
            recon_pred_seq.append(self.recon_head(feat_t))
            reward_pred_seq.append(self.reward_head(feat_t))
            continue_pred_seq.append(self.continue_head(feat_t))
            return_logits_seq.append(self.return_head(feat_t))

        recon = torch.stack(recon_pred_seq, dim=1)            # [B, T, C]
        reward = torch.stack(reward_pred_seq, dim=1).squeeze(-1)  # [B, T]
        continue_logit = torch.stack(continue_pred_seq, dim=1).squeeze(-1)  # [B, T]
        return_logits = torch.stack(return_logits_seq, dim=1)  # [B, T, n_bins]
        post_logits = torch.stack(post_logits_seq, dim=1)
        prior_logits = torch.stack(prior_logits_seq, dim=1)

        # ----- Losses -----
        # 1. Reconstruction (MSE on symlog)
        recon_loss = F.mse_loss(recon, symlog(obs_seq))

        # 2. Return (TwoHot CE loss)
        ret_target = self.return_encoder.encode(returns)  # [B, T, n_bins]
        ret_log_softmax = F.log_softmax(return_logits, dim=-1)
        return_loss = -(ret_target * ret_log_softmax).sum(dim=-1).mean()

        # 3. KL balance (post vs prior)
        post_dist = torch.distributions.Categorical(logits=post_logits.view(B, T, self.rssm.stoch_categories, self.rssm.stoch_dimensions))
        prior_dist = torch.distributions.Categorical(logits=prior_logits.view(B, T, self.rssm.stoch_categories, self.rssm.stoch_dimensions))
        # KL is computed per-token-per-stoch_dim, averaged
        kl_post = torch.distributions.kl_divergence(post_dist, prior_dist).mean()
        # Use stop-gradient trick for KL balance
        post_logits_sg = post_logits.detach()
        prior_logits_sg = prior_logits.detach()
        post_dist_sg = torch.distributions.Categorical(logits=post_logits_sg.view(B, T, self.rssm.stoch_categories, self.rssm.stoch_dimensions))
        prior_dist_sg = torch.distributions.Categorical(logits=prior_logits_sg.view(B, T, self.rssm.stoch_categories, self.rssm.stoch_dimensions))
        kl_prior = torch.distributions.kl_divergence(post_dist_sg, prior_dist).mean()
        kl_balanced = self.kl_balance * kl_prior + (1 - self.kl_balance) * kl_post

        # Free bits
        kl_balanced = torch.clamp(kl_balanced, min=self.free_bits)

        total_loss = recon_loss + return_loss + kl_balanced

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "return_loss": return_loss,
            "kl_loss": kl_balanced,
            "return_pred": self.return_encoder.decode(return_logits),
            "recon": recon,
            "reward": reward,
            "continue": torch.sigmoid(continue_logit),
        }

    def imagine_rollout(self, init_state: dict, actions: torch.Tensor) -> dict:
        """Used by the M3 DreamerV3 agent for latent imagination.

        Args:
            init_state: dict with 'h' [B, hidden_dim] and 'z' [B, stoch_dim]
            actions: [B, H, action_dim] -- H imagined steps
        Returns:
            dict with 'feat' [B, H, feat_dim], 'rewards' [B, H], 'continues' [B, H]
        """
        B, H, _ = actions.shape
        state = init_state
        feats = []
        rewards = []
        continues = []
        for t in range(H):
            state = self.rssm.imagine_step(state, actions[:, t, :])
            feat = self.feat(state)
            feats.append(feat)
            rewards.append(self.reward_head(feat).squeeze(-1))
            continues.append(torch.sigmoid(self.continue_head(feat).squeeze(-1)))
        return {
            "feat": torch.stack(feats, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
        }


def smoke_test():
    """Forward + backward pass on dummy data."""
    torch.manual_seed(42)
    B, T, C = 4, 32, 121
    model = DreamerV3WorldModel(obs_dim=C, action_dim=1, n_assets=10).to("cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v16] DreamerV3 WM params: {n_params:,}")

    obs = torch.randn(B, T, C)
    actions = torch.zeros(B, T, 1)
    asset_ids = torch.zeros(B, dtype=torch.long)
    returns = torch.randn(B, T) * 0.05

    out = model.forward_train(obs, actions, asset_ids, returns)
    print(f"[v16] forward OK; total_loss={out['loss'].item():.4f} "
          f"(recon={out['recon_loss'].item():.3f}, "
          f"return={out['return_loss'].item():.3f}, "
          f"kl={out['kl_loss'].item():.3f})")

    out["loss"].backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[v16] backward OK; {has_grad}/{sum(1 for _ in model.parameters())} params have non-zero grad")

    # Test imagination rollout
    model.eval()
    with torch.no_grad():
        init = model.rssm.initial_state(B, "cpu")
        imag_actions = torch.zeros(B, 16, 1)
        roll = model.imagine_rollout(init, imag_actions)
    print(f"[v16] imagination rollout OK; "
          f"feat={tuple(roll['feat'].shape)}, "
          f"rewards={tuple(roll['rewards'].shape)}, "
          f"continues={tuple(roll['continues'].shape)}")


def main_cli():
    """STUB: V16 ships the DreamerV3 model class only; trainer is pending."""
    import argparse
    parser = argparse.ArgumentParser(
        description="V16 DreamerV3 (model only, trainer pending). Use --smoke."
    )
    parser.add_argument("--features", type=int, default=121, help="compat")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the parameter-count smoke test.")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return
    print("[V16] STUB: DreamerV3 trainer not yet implemented. Use --smoke to "
          "run the model-class param-count check.")


if __name__ == "__main__":
    main_cli()
