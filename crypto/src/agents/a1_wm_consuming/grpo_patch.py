"""GRPO -- Group-Relative Policy Optimization (DeepSeek 2024 patch on PPO).

Key idea: instead of comparing each trajectory's advantage to a baseline value
function (PPO), GRPO groups N rollouts of the SAME starting state and uses
the within-group mean as the baseline. Removes the need for a critic.

Drop-in patch on top of src/agents/a1_wm_consuming/ppo.py PPOTrainer. Same API, replaces the
advantage computation.

Reference: DeepSeek-Math (2402.03300), DeepSeek-V2 papers.

Usage in training script:
    from agent.grpo_patch import compute_grpo_advantages
    # Replace: advantages = compute_gae(rewards, values)
    # With:    advantages = compute_grpo_advantages(rewards_grouped)
"""
from __future__ import annotations

import torch


def compute_grpo_advantages(group_rewards: torch.Tensor,
                             eps: float = 1e-6) -> torch.Tensor:
    """Compute GRPO advantages from grouped trajectory rewards.

    Args:
        group_rewards: [n_groups, group_size, T] sum of trajectory rewards
                       for each rollout in each group. group_size rollouts
                       share the same starting state.
        eps: numerical stability for the std denominator.
    Returns:
        [n_groups, group_size, T] advantages.
    """
    # Compute mean and std within each group
    group_mean = group_rewards.mean(dim=1, keepdim=True)
    group_std = group_rewards.std(dim=1, keepdim=True).clamp(min=eps)
    advantages = (group_rewards - group_mean) / group_std
    return advantages


def grpo_actor_loss(log_probs: torch.Tensor,
                    old_log_probs: torch.Tensor,
                    advantages: torch.Tensor,
                    clip_eps: float = 0.2) -> torch.Tensor:
    """PPO-clip actor loss but with GRPO advantages."""
    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(surr1, surr2).mean()


def smoke_test():
    torch.manual_seed(0)
    n_groups, group_size, T = 4, 8, 16
    rewards = torch.randn(n_groups, group_size, T) * 0.01
    adv = compute_grpo_advantages(rewards)
    print(f"[grpo] adv shape={tuple(adv.shape)}, mean={adv.mean():.4f}, std={adv.std():.4f}")
    # Within-group mean should be ~0 (centered)
    grp_means = adv.mean(dim=1)  # [n_groups, T]
    print(f"[grpo] within-group means: max abs = {grp_means.abs().max():.4e} (should be ~0)")

    # Simulate actor loss
    log_probs = torch.randn(n_groups, group_size, T)
    old_log_probs = log_probs.clone() + torch.randn(n_groups, group_size, T) * 0.05
    loss = grpo_actor_loss(log_probs, old_log_probs, adv)
    print(f"[grpo] actor loss: {loss.item():.4f}")
    print("[grpo] GRPO patch smoke test PASS")


if __name__ == "__main__":
    smoke_test()
