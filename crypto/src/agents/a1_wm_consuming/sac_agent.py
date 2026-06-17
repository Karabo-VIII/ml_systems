"""SAC -- Soft Actor-Critic drop-in replacement for PPO.

Off-policy continuous-control algorithm. 5-10x more sample-efficient than PPO
in standard benchmarks. Used as a baseline alongside the DreamerV3 agent.

Key differences vs PPO:
  - Off-policy: stores transitions in replay buffer, samples random batches.
  - Maximum-entropy objective: actor optimizes E[Q] + alpha * H(pi).
  - Twin Q-critics for clipped double-Q (Fujimoto 2018 trick).
  - Automatic temperature tuning of alpha.

Same agent interface as PPOTrainer in src/agents/a1_wm_consuming/ppo.py — can be substituted
without changing the environment or rewards code.

Reference: Haarnoja et al. 2018 (1801.01290).
"""
from __future__ import annotations

__class_tag__ = "A1"  # WM-consuming agent (SAC over frozen-WM features) (doc SS1.8)

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Networks
# -----------------------------------------------------------------------------

class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 1, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-5, 2)
        return mu, log_std

    def sample(self, obs: torch.Tensor):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=-1, keepdim=True)


class QCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

@dataclass
class Transition:
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        obs = torch.from_numpy(np.stack([t.obs for t in batch])).float()
        actions = torch.from_numpy(np.stack([t.action for t in batch])).float()
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)
        next_obs = torch.from_numpy(np.stack([t.next_obs for t in batch])).float()
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return len(self.buffer)


# -----------------------------------------------------------------------------
# SAC trainer
# -----------------------------------------------------------------------------

class SACAgent(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int = 1,
                 actor_lr: float = 3e-4, critic_lr: float = 3e-4,
                 alpha_lr: float = 3e-4, gamma: float = 0.99, tau: float = 5e-3,
                 target_entropy: float | None = None):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        self.actor = GaussianActor(obs_dim, action_dim)
        self.q1 = QCritic(obs_dim, action_dim)
        self.q2 = QCritic(obs_dim, action_dim)
        self.q1_target = QCritic(obs_dim, action_dim)
        self.q2_target = QCritic(obs_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Auto temperature
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha = nn.Parameter(torch.zeros(1))
        self.alpha_lr = alpha_lr

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            if deterministic:
                mu, _ = self.actor.forward(obs_t)
                a = torch.tanh(mu)
            else:
                a, _ = self.actor.sample(obs_t)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch) -> dict:
        obs, actions, rewards, next_obs, dones = batch

        # Critic update: y = r + gamma * (1-done) * (min Q_target(s', a') - alpha * log_pi)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q = torch.min(
                self.q1_target(next_obs, next_action),
                self.q2_target(next_obs, next_action),
            ) - self.alpha * next_log_prob.squeeze(-1)
            y = rewards + self.gamma * (1 - dones) * target_q

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)
        q1_loss = F.mse_loss(q1_pred, y)
        q2_loss = F.mse_loss(q2_pred, y)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Actor update
        new_action, log_prob = self.actor.sample(obs)
        log_prob = log_prob.squeeze(-1)
        q_min = torch.min(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (self.alpha.detach() * log_prob - q_min).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return {
            "q1_loss": float(q1_loss.item()),
            "q2_loss": float(q2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "mean_log_prob": float(log_prob.mean().item()),
        }


def smoke_test():
    torch.manual_seed(0)
    np.random.seed(0)
    obs_dim, action_dim = 121, 1

    agent = SACAgent(obs_dim, action_dim)
    buffer = ReplayBuffer(capacity=10000)

    # Push 200 random transitions
    for _ in range(200):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = agent.select_action(obs)
        reward = float(np.random.randn() * 0.01)
        next_obs = np.random.randn(obs_dim).astype(np.float32)
        done = bool(np.random.rand() > 0.95)
        buffer.push(Transition(obs, action, reward, next_obs, done))

    # Update
    print(f"[sac] buffer size {len(buffer)}, running 5 update steps...")
    for step in range(5):
        batch = buffer.sample(64)
        info = agent.update(batch)
        if step == 0 or step == 4:
            print(f"[sac]   step {step}: q1={info['q1_loss']:.4f} actor={info['actor_loss']:.4f} "
                  f"alpha={info['alpha']:.3f}")

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"[sac] params: {n_params:,}")
    print("[sac] smoke test PASS")


if __name__ == "__main__":
    smoke_test()
