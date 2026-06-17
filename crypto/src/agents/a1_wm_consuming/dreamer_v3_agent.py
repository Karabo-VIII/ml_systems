"""DreamerV3 agent -- actor-critic in latent imagination.

Hafner 2023 architecture: train actor + critic on imagined rollouts in the
frozen V16 WM's latent space. No reward calls to a real environment during
training; the critic learns from imagined value estimates.

Wires onto the V16 backbone at
src/agents/a1_wm_consuming/backbones/v16_dreamerv3/v16_training/dreamer_v3.py:
  - Frozen V16 WM provides initial state (h, z) from real observations
  - Actor: pi(a | feat=[h, z]) -- continuous action in [-1, 1] (long/short)
  - Critic: V(feat) -- TwoHot symlog return prediction
  - Lambda return: weighted average of n-step returns from imagined trajectories

Loss design (DreamerV3 sec. 5):
  - Actor: maximize E[lambda_return] - eta * entropy_regularization
  - Critic: TwoHot CE loss against detached lambda_return target
  - Trust region: clip ratio for actor, slow target net for critic
"""
from __future__ import annotations

__class_tag__ = "A1"  # WM-consuming agent: plans over a FROZEN forecaster (doc SS1.8)

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# V16 (DreamerV3) is an A1 backbone -- reclassified out of the forecaster zoo
# (src/wm/v16) into src/agents/a1_wm_consuming/backbones/v16_dreamerv3/ on
# 2026-06-11. Resolve the backbone trainer dir relative to THIS file.
_BACKBONE_DIR = Path(__file__).resolve().parent / "backbones" / "v16_dreamerv3" / "v16_training"
sys.path.insert(0, str(_BACKBONE_DIR))
from dreamer_v3 import DreamerV3WorldModel, TwoHotEncoder, symlog, symexp  # type: ignore  # noqa: E402


class Actor(nn.Module):
    """pi(a | feat) -- Gaussian policy over continuous action in [-1, 1]."""

    def __init__(self, feat_dim: int, action_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(feat)
        mu = torch.tanh(self.mu(h))
        log_std = self.log_std(h).clamp(-5, 0)
        return mu, log_std

    def sample(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.forward(feat)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()
        # squash to [-1, 1]
        action = torch.tanh(action)
        # log prob with tanh correction
        log_prob = normal.log_prob(action) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """V(feat) -- TwoHot symlog return."""

    def __init__(self, feat_dim: int, hidden: int = 256, n_bins: int = 255,
                 bin_min: float = -20.0, bin_max: float = 20.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, n_bins),
        )
        self.encoder = TwoHotEncoder(n_bins=n_bins, bin_min=bin_min, bin_max=bin_max)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.trunk(feat)

    def value(self, feat: torch.Tensor) -> torch.Tensor:
        return self.encoder.decode(self.forward(feat))


def lambda_return(rewards: torch.Tensor, values: torch.Tensor,
                  continues: torch.Tensor, lambda_: float = 0.95) -> torch.Tensor:
    """Compute lambda-return targets for an imagined rollout.

    Args:
        rewards: [B, H] imagined rewards
        values: [B, H+1] critic values along the trajectory + bootstrap at end
        continues: [B, H] discount/continue probabilities
        lambda_: lambda for n-step blending
    Returns:
        [B, H] lambda returns
    """
    H = rewards.shape[1]
    inputs = rewards + continues * values[:, 1:] * (1 - lambda_)
    last = values[:, -1]
    outputs = []
    for t in reversed(range(H)):
        last = inputs[:, t] + continues[:, t] * lambda_ * last
        outputs.append(last)
    return torch.stack(list(reversed(outputs)), dim=1)


class DreamerV3Agent(nn.Module):
    """Combines frozen V16 WM with actor + critic for imagination-based learning."""

    def __init__(self, wm: DreamerV3WorldModel, action_dim: int = 1):
        super().__init__()
        self.wm = wm
        # Freeze WM
        for p in self.wm.parameters():
            p.requires_grad = False

        feat_dim = self.wm.rssm.hidden_dim + self.wm.rssm.stoch_dim
        self.actor = Actor(feat_dim, action_dim=action_dim)
        self.critic = Critic(feat_dim)
        self.target_critic = Critic(feat_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.imagination_horizon = 15
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.entropy_coef = 3e-4
        self.target_critic_tau = 0.02

    @torch.no_grad()
    def encode_initial_state(self, obs_seq: torch.Tensor, asset_ids: torch.Tensor,
                              actions_seq: torch.Tensor | None = None) -> dict:
        """Run V16 WM forward to get the final RSSM state for imagination."""
        B, T, C = obs_seq.shape
        if actions_seq is None:
            actions_seq = torch.zeros(B, T, self.action_dim, device=obs_seq.device)
        state = self.wm.rssm.initial_state(B, obs_seq.device)
        for t in range(T):
            obs_t = self.wm.encode_obs(obs_seq[:, t, :], asset_ids)
            state, _ = self.wm.rssm.forward_step(state, actions_seq[:, t, :], obs_t)
        return state

    def imagine_with_actor(self, init_state: dict, horizon: int) -> dict:
        """Imagined rollout where actions come from the current actor."""
        state = {k: v.clone() for k, v in init_state.items()}
        feats, actions, log_probs = [], [], []
        rewards, continues = [], []
        for _ in range(horizon):
            feat = self.wm.feat(state)
            feats.append(feat)
            action, log_prob = self.actor.sample(feat)
            actions.append(action)
            log_probs.append(log_prob)
            state = self.wm.rssm.imagine_step(state, action)
            rewards.append(self.wm.reward_head(self.wm.feat(state)).squeeze(-1))
            continues.append(torch.sigmoid(self.wm.continue_head(self.wm.feat(state)).squeeze(-1)))

        return {
            "feats": torch.stack(feats, dim=1),
            "actions": torch.stack(actions, dim=1),
            "log_probs": torch.stack(log_probs, dim=1).squeeze(-1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
        }

    def actor_critic_loss(self, init_state: dict) -> dict:
        roll = self.imagine_with_actor(init_state, self.imagination_horizon)
        feats = roll["feats"]              # [B, H, feat_dim]
        rewards = roll["rewards"]          # [B, H]
        continues = roll["continues"]      # [B, H]
        log_probs = roll["log_probs"]      # [B, H]

        # Critic on the feats; bootstrap with target critic at end
        with torch.no_grad():
            B = feats.shape[0]
            # Append a bootstrap state (copy last)
            last_feat = feats[:, -1:].detach()
            full_feats = torch.cat([feats.detach(), last_feat], dim=1)
            target_values = self.target_critic.value(full_feats)
            lambda_ret = lambda_return(rewards, target_values, continues, lambda_=self.lambda_)

        # Critic loss: TwoHot CE against detached lambda return
        critic_logits = self.critic(feats.detach())  # [B, H, n_bins]
        critic_target = self.critic.encoder.encode(lambda_ret.detach())
        critic_loss = -(critic_target * F.log_softmax(critic_logits, dim=-1)).sum(dim=-1).mean()

        # Actor loss: maximize lambda return - entropy bonus
        # Reinforce-style estimator with critic as baseline
        with torch.no_grad():
            advantages = lambda_ret - self.target_critic.value(feats.detach())
        actor_loss = -(log_probs * advantages.detach()).mean()
        # Entropy bonus
        entropy = -log_probs.mean()
        actor_loss = actor_loss - self.entropy_coef * entropy

        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "lambda_return_mean": lambda_ret.mean().detach(),
            "advantages_mean": advantages.mean().detach(),
            "entropy": entropy.detach(),
        }

    def soft_update_target_critic(self):
        with torch.no_grad():
            for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.target_critic_tau).add_(p.data, alpha=self.target_critic_tau)


def smoke_test():
    torch.manual_seed(0)
    wm = DreamerV3WorldModel(obs_dim=121, action_dim=1, n_assets=10).eval()
    agent = DreamerV3Agent(wm, action_dim=1)
    n_params_total = sum(p.numel() for p in agent.parameters())
    n_params_trainable = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"[v16-agent] total params {n_params_total:,}, trainable (actor+critic) {n_params_trainable:,}")

    B, T, C = 4, 16, 121
    obs = torch.randn(B, T, C)
    asset_ids = torch.zeros(B, dtype=torch.long)
    init = agent.encode_initial_state(obs, asset_ids)
    print(f"[v16-agent] init state h={tuple(init['h'].shape)}, z={tuple(init['z'].shape)}")

    losses = agent.actor_critic_loss(init)
    print(f"[v16-agent] actor_loss={losses['actor_loss'].item():.4f}  "
          f"critic_loss={losses['critic_loss'].item():.4f}")

    # Backward only on actor + critic (WM frozen)
    (losses["actor_loss"] + losses["critic_loss"]).backward()
    actor_grad = sum(1 for p in agent.actor.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    critic_grad = sum(1 for p in agent.critic.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    wm_grad = sum(1 for p in agent.wm.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"[v16-agent] grads: actor={actor_grad}, critic={critic_grad}, wm={wm_grad} (should be 0)")
    assert wm_grad == 0, "WM should be frozen"

    agent.soft_update_target_critic()
    print("[v16-agent] target critic soft-updated -- OK")


if __name__ == "__main__":
    smoke_test()
