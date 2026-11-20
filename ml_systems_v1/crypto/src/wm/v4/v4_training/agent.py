"""
V4 Trading Agent — Single ActorCritic

Replaces V303's Bull/Bear/Boss hierarchy with a clean unified agent.
The softmax over {Short, Neutral, Long} naturally handles allocation.

Two training phases:
  Phase A: Supervised pretraining (predict hindsight-optimal actions)
  Phase B: PPO fine-tuning with dreamed trajectories
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
from settings import *
from components import TwoHotSymlog


class ActorCritic(nn.Module):
    """
    Unified actor-critic agent.

    Input: world model state = concat(h_seq, z_posterior)
           Dimension: WM_D_MODEL + RSSM_FLAT_DIM

    Actor: state -> action logits [3]  {Short, Neutral, Long}
    Critic: state -> value [NUM_BINS] (TwoHot symlog)
    """
    def __init__(
        self,
        state_dim: int = WM_D_MODEL + RSSM_FLAT_DIM,
        hidden_dim: int = AGENT_HIDDEN,
        action_dim: int = ACTION_DIM,
        num_bins: int = NUM_BINS,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head (TwoHot value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

        # TwoHot decoder for value
        self.bucketer = TwoHotSymlog(num_bins, BIN_MIN, BIN_MAX, DEVICE)

        # Learnable entropy coefficient (log alpha)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    @property
    def alpha(self):
        return self.log_alpha.exp().clamp(min=1e-4, max=1.0)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: [B, state_dim] or [B, T, state_dim]
        Returns:
            action_logits: [B, 3] or [B, T, 3]
            value_logits:  [B, NUM_BINS] or [B, T, NUM_BINS]
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value_logits = self.critic(features)
        return action_logits, value_logits

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample or select action.
        Returns: action, log_prob, value_scalar, entropy
        """
        action_logits, value_logits = self.forward(state)
        dist = D.Categorical(logits=action_logits)

        if deterministic:
            action = action_logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.bucketer.decode(value_logits)

        return action, log_prob, value, entropy

    def evaluate_actions(self, state: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate given actions for PPO update.
        Returns: log_prob, value_scalar, entropy, value_logits
        """
        action_logits, value_logits = self.forward(state)
        dist = D.Categorical(logits=action_logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.bucketer.decode(value_logits)

        return log_prob, value, entropy, value_logits

    def get_value(self, state: torch.Tensor):
        """Get value estimate only (for GAE bootstrap)."""
        _, value_logits = self.forward(state)
        return self.bucketer.decode(value_logits)


def compute_reward(
    action: torch.Tensor,
    prev_action: torch.Tensor,
    market_return: torch.Tensor,
    base_fee: float = BASE_FEE,
) -> torch.Tensor:
    """
    Compute trading reward with transaction costs.

    Args:
        action:        [B]  {0=Short, 1=Neutral, 2=Long}
        prev_action:   [B]
        market_return:  [B]  actual or predicted return

    Returns:
        reward: [B]  log-utility PnL
    """
    position = (action - 1).float()  # -1, 0, +1
    prev_position = (prev_action - 1).float()
    gross_pnl = position * market_return

    # FIX(ugly): Transaction cost proportional to position change distance.
    # A flip (Long->Short or Short->Long) traverses distance=2, so it costs
    # 2 * base_fee (close + reopen), not just 1 * base_fee.
    # Old code: cost = (action != prev_action).float() * base_fee  # BUG: undercharges flips
    cost = torch.abs(position - prev_position) * base_fee

    net_return = gross_pnl - cost

    # Kelly-inspired log utility (risk-sensitive, bounded)
    reward = torch.log1p(torch.clamp(net_return, min=-0.5))

    return reward


def compute_gae(
    rewards: torch.Tensor,   # [T, B]
    values: torch.Tensor,    # [T, B]
    next_value: torch.Tensor,  # [B]
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Returns: advantages [T, B], returns [T, B]
    """
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_val = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae

    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    # Sanity check
    state_dim = WM_D_MODEL + RSSM_FLAT_DIM
    agent = ActorCritic().to(DEVICE)
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    state = torch.randn(8, state_dim).to(DEVICE)
    action, log_prob, value, entropy = agent.get_action(state)
    print(f"Action: {action}, Value: {value.mean():.4f}, Entropy: {entropy.mean():.4f}")
    print("[OK] Agent sanity check passed.")
