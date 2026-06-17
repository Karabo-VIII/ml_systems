"""
PPO (Proximal Policy Optimization) for Trading Agent
=====================================================
Clean PPO implementation with TEMP_4-inspired enhancements:

1. Standard PPO with GAE (baseline)
2. Crucible training: difficulty-weighted episode sampling
3. Surprise auxiliary loss: predictive coding from DualStream policy
4. SAV: weight perturbation robustness test

References:
  Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
  CleanRL: https://github.com/vwxyzjn/cleanrl
"""

__class_tag__ = "A1"  # WM-consuming agent trainer (PPO over frozen-WM features) (doc SS1.8)

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm

from config import (
    DEVICE,
    PPO_LR, PPO_EPOCHS, PPO_BATCH_SIZE,
    PPO_GAMMA, PPO_GAE_LAMBDA,
    PPO_CLIP_EPS, PPO_CLIP_VALUE, PPO_MAX_GRAD_NORM,
    PPO_ENTROPY_COEFF, PPO_VALUE_COEFF,
    PPO_N_STEPS, AGENT_MODEL_DIR,
)
from policy import ActorCritic, DualStreamActorCritic


# ---------------------------------------------------------------------------
# Running Normalization (standard PPO practice -- CleanRL / SB3)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford's online algorithm for tracking running mean/variance.

    Used for observation and reward normalization, which are ESSENTIAL for PPO
    to work with features at different scales (return preds ~0.001 vs regime ~0.33).
    """

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # Avoid division by zero on first call

    def update(self, x: np.ndarray):
        """Update running stats with a batch. x: [batch, *shape] or [*shape]."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize: (x - mean) / sqrt(var + eps). Clips to [-10, 10]."""
        normed = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(normed, -10.0, 10.0).astype(np.float32)

    def state_dict(self) -> dict:
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, d: dict):
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores transitions from environment interaction."""

    observations: np.ndarray    # [N, obs_dim]
    actions: np.ndarray         # [N, action_dim]
    log_probs: np.ndarray       # [N]
    rewards: np.ndarray         # [N]
    values: np.ndarray          # [N]
    dones: np.ndarray           # [N]
    advantages: np.ndarray      # [N] (computed after rollout)
    returns: np.ndarray         # [N] (computed after rollout)

    @staticmethod
    def create(n_steps: int, obs_dim: int, action_dim: int) -> "RolloutBuffer":
        return RolloutBuffer(
            observations=np.zeros((n_steps, obs_dim), dtype=np.float32),
            actions=np.zeros((n_steps, action_dim), dtype=np.float32),
            log_probs=np.zeros(n_steps, dtype=np.float32),
            rewards=np.zeros(n_steps, dtype=np.float32),
            values=np.zeros(n_steps, dtype=np.float32),
            dones=np.zeros(n_steps, dtype=np.float32),
            advantages=np.zeros(n_steps, dtype=np.float32),
            returns=np.zeros(n_steps, dtype=np.float32),
        )

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation."""
        gae = 0.0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size: int):
        """Yield random minibatches for PPO update."""
        n = len(self.rewards)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_idx = indices[start:end]

            yield (
                torch.tensor(self.observations[batch_idx], device=DEVICE),
                torch.tensor(self.actions[batch_idx], device=DEVICE),
                torch.tensor(self.log_probs[batch_idx], device=DEVICE),
                torch.tensor(self.advantages[batch_idx], device=DEVICE),
                torch.tensor(self.returns[batch_idx], device=DEVICE),
                torch.tensor(self.values[batch_idx], device=DEVICE),
            )


# ---------------------------------------------------------------------------
# Crucible: Difficulty-Weighted Episode Sampling
# ---------------------------------------------------------------------------

class CrucibleTracker:
    """
    Tracks episode difficulty and biases sampling toward hard scenarios.

    Inspired by TEMP_4 V70 Dreamer's adversarial training:
    - Episodes where the agent performs poorly get higher weight
    - Weights decay over time so the agent isn't stuck on old failures
    - Produces a difficulty score per episode that can modulate learning rate

    Usage:
        crucible = CrucibleTracker(n_assets=10)
        crucible.update_difficulty(episode_reward, asset_idx, start_bar)
        weights = crucible.get_sampling_weights(n_assets, n_start_positions)
    """

    def __init__(self, n_segments: int = 50, decay: float = 0.95):
        self.n_segments = n_segments
        self.decay = decay
        # Difficulty weights per (asset, time_segment)
        # Higher = harder = sample more often
        self.difficulty = {}  # (asset_idx, segment) -> weight

    def update(self, asset_idx: int, bar_idx: int, n_bars: int, episode_reward: float):
        """Update difficulty based on episode outcome."""
        segment = min(int(bar_idx / max(n_bars, 1) * self.n_segments), self.n_segments - 1)
        key = (asset_idx, segment)

        # Pain signal: negative rewards increase difficulty
        pain = max(0.0, -episode_reward * 10.0)

        old = self.difficulty.get(key, 1.0)
        self.difficulty[key] = old * self.decay + pain + 0.1

        # Clamp
        self.difficulty[key] = min(self.difficulty[key], 50.0)

    def get_weight(self, asset_idx: int, bar_idx: int, n_bars: int) -> float:
        """Get sampling weight for a given (asset, position)."""
        segment = min(int(bar_idx / max(n_bars, 1) * self.n_segments), self.n_segments - 1)
        key = (asset_idx, segment)
        return self.difficulty.get(key, 1.0)


# ---------------------------------------------------------------------------
# SAV: Stability Adversarial Validation
# ---------------------------------------------------------------------------

def sav_test(policy, env_val, n_episodes: int = 5, noise_std: float = 0.02,
             obs_normalizer=None) -> dict:
    """
    Stability Adversarial Validation (from TEMP_4 V75 Singularity).

    Tests if the agent's performance is robust to small weight perturbations.
    If performance degrades significantly under noise, the agent has overfit
    to specific weight configurations (fragile).

    Args:
        policy: Trained policy network
        env_val: Validation environment
        n_episodes: Number of episodes to evaluate
        noise_std: Standard deviation of Gaussian noise added to weights
        obs_normalizer: Optional callable for observation normalization

    Returns:
        dict with clean_reward, noisy_reward, stability_ratio
    """
    from train_agent import evaluate_agent  # Lazy import to avoid circular

    # Clean evaluation
    clean_metrics = evaluate_agent(policy, env_val, n_episodes=n_episodes,
                                    obs_normalizer=obs_normalizer)
    clean_reward = clean_metrics["mean_return"]

    # Create noisy copy
    noisy_policy = copy.deepcopy(policy)
    with torch.no_grad():
        for param in noisy_policy.parameters():
            noise = torch.randn_like(param) * noise_std
            param.add_(noise)

    # Noisy evaluation
    noisy_metrics = evaluate_agent(noisy_policy, env_val, n_episodes=n_episodes,
                                    obs_normalizer=obs_normalizer)
    noisy_reward = noisy_metrics["mean_return"]

    # Stability ratio: how much performance survives noise
    # 1.0 = perfectly stable, 0.0 = completely fragile
    if abs(clean_reward) > 1e-8:
        stability = noisy_reward / clean_reward
    else:
        stability = 1.0 if abs(noisy_reward) < 1e-8 else 0.0

    return {
        "clean_reward": clean_reward,
        "noisy_reward": noisy_reward,
        "stability_ratio": stability,
        "noise_std": noise_std,
    }


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

def _format_eval(metrics: dict) -> str:
    """Format evaluation metrics as a clean table."""
    lines = []
    lines.append("    +---------------------+----------+")
    lines.append("    | Metric              |    Value |")
    lines.append("    +---------------------+----------+")
    row_order = [
        ("mean_return", "Mean Return"),
        ("sharpe", "Sharpe (ann.)"),
        ("max_drawdown", "Max Drawdown"),
        ("win_rate", "Win Rate"),
        ("mean_turnover", "Mean Turnover"),
        ("mean_final_value", "Final Value"),
        ("mean_cost", "Mean Cost"),
        ("mean_confidence", "Confidence"),
    ]
    for key, label in row_order:
        if key in metrics:
            val = metrics[key]
            if key == "win_rate":
                lines.append(f"    | {label:<19s} | {val:>7.1%} |")
            elif key == "max_drawdown":
                lines.append(f"    | {label:<19s} | {val:>7.2%} |")
            elif key == "mean_final_value":
                lines.append(f"    | {label:<19s} | {val:>8.0f} |")
            else:
                lines.append(f"    | {label:<19s} | {val:>+8.4f} |")
    lines.append("    +---------------------+----------+")
    return "\n".join(lines)


class PPOTrainer:
    """
    PPO training loop for the trading agent.

    Supports both ActorCritic (baseline) and DualStreamActorCritic (enhanced).
    Optional Crucible difficulty tracking and surprise auxiliary loss.

    Usage:
        trainer = PPOTrainer(env, policy_type="dual_stream")
        trainer.train(total_timesteps=2_000_000)
    """

    def __init__(
        self,
        env,
        policy=None,
        lr: float = PPO_LR,
        n_steps: int = PPO_N_STEPS,
        batch_size: int = PPO_BATCH_SIZE,
        n_epochs: int = PPO_EPOCHS,
        gamma: float = PPO_GAMMA,
        gae_lambda: float = PPO_GAE_LAMBDA,
        clip_eps: float = PPO_CLIP_EPS,
        entropy_coeff: float = PPO_ENTROPY_COEFF,
        value_coeff: float = PPO_VALUE_COEFF,
        max_grad_norm: float = PPO_MAX_GRAD_NORM,
        surprise_coeff: float = 0.1,
        use_crucible: bool = False,
    ):
        self.env = env
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.surprise_coeff = surprise_coeff

        # Policy
        if policy is None:
            self.policy = ActorCritic(
                obs_dim=env.observation_dim,
                action_dim=env.action_dim,
            ).to(DEVICE)
        else:
            self.policy = policy.to(DEVICE)

        self.is_dual_stream = isinstance(self.policy, DualStreamActorCritic)

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=lr, eps=1e-5
        )

        # Running normalization (standard PPO -- without these, the agent can't
        # learn because obs features are at wildly different scales and rewards
        # are uniformly negative during exploration)
        self.obs_rms = RunningMeanStd(shape=(env.observation_dim,))
        self.ret_rms = RunningMeanStd(shape=())

        # Crucible (difficulty-weighted training)
        self.crucible = CrucibleTracker() if use_crucible else None

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_metrics = []
        self.best_eval_reward = float("-inf")
        self.best_eval_sharpe = float("-inf")
        self.start_rollout = 0  # for resume support

    def collect_rollout(self) -> RolloutBuffer:
        """Collect a rollout of n_steps from the environment.

        Applies observation normalization (running mean/std) and reward
        normalization (running std, no centering) -- both standard PPO practice.
        """
        buffer = RolloutBuffer.create(
            self.n_steps, self.env.observation_dim, self.env.action_dim
        )

        obs = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        scenario_counts_this_rollout = {}

        # Track Hebbian fast weights across steps (DualStream only)
        fast_weights = None

        for step in range(self.n_steps):
            # Update running obs stats with RAW observation, then normalize
            self.obs_rms.update(obs)
            norm_obs = self.obs_rms.normalize(obs)

            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                if self.is_dual_stream:
                    action, log_prob, value, aux = self.policy.get_action(
                        obs_tensor, fast_weights=fast_weights
                    )
                    fast_weights = aux.get("fast_weights")
                else:
                    action, log_prob, value = self.policy.get_action(obs_tensor)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, done, info = self.env.step(action_np)

            # Track scenarios (augmentation)
            sc = info.get("scenario_type")
            if sc is not None and done:
                scenario_counts_this_rollout[sc] = scenario_counts_this_rollout.get(sc, 0) + 1

            # Store NORMALIZED obs in buffer (policy sees normalized during update too)
            buffer.observations[step] = norm_obs
            buffer.actions[step] = action_np
            buffer.log_probs[step] = log_prob.item()
            buffer.rewards[step] = reward
            buffer.values[step] = value.squeeze().item()
            buffer.dones[step] = float(done)

            episode_reward += reward
            episode_length += 1

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Crucible: track difficulty
                if self.crucible is not None:
                    for asset_idx, bar_idx in self.env.current_bar_indices.items():
                        n_bars = self.env.asset_data.get(asset_idx, {}).get("n_bars", 1)
                        self.crucible.update(asset_idx, bar_idx, n_bars, episode_reward)

                obs = self.env.reset()
                episode_reward = 0.0
                episode_length = 0
                fast_weights = None  # Reset plasticity on new episode
            else:
                obs = next_obs

        # Normalize rewards by running std (preserve sign, just scale)
        # This ensures the value function and advantages operate at a stable scale
        # even when all raw rewards are negative (cost-dominated exploration).
        self.ret_rms.update(buffer.rewards)
        reward_std = np.sqrt(self.ret_rms.var + 1e-8)
        buffer.rewards = buffer.rewards / reward_std

        # Compute last value for GAE (using normalized obs)
        self.obs_rms.update(obs)
        norm_obs = self.obs_rms.normalize(obs)
        with torch.no_grad():
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32, device=DEVICE)
            if self.is_dual_stream:
                _, _, last_value, _ = self.policy.get_action(obs_tensor, fast_weights=fast_weights)
            else:
                _, _, last_value = self.policy.get_action(obs_tensor)
            last_val = last_value.squeeze().item()

        buffer.compute_gae(last_val, self.gamma, self.gae_lambda)

        # Attach scenario counts for logging (avoids changing return signature)
        buffer._scenario_counts = scenario_counts_this_rollout

        return buffer

    def update(self, buffer: RolloutBuffer) -> dict:
        """Run PPO update on the collected rollout."""
        # Normalize advantages
        adv = buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantages = adv

        total_pg_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_surprise_loss = 0.0
        n_updates = 0

        for epoch in range(self.n_epochs):
            for obs_b, act_b, old_lp_b, adv_b, ret_b, old_val_b in buffer.get_batches(self.batch_size):
                # Evaluate current policy on old actions
                new_log_prob, new_value, entropy = self.policy.evaluate_actions(obs_b, act_b)
                new_value = new_value.squeeze(-1)

                # Policy loss (clipped surrogate)
                ratio = (new_log_prob - old_lp_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped to prevent large updates)
                value_clipped = old_val_b + torch.clamp(
                    new_value - old_val_b, -PPO_CLIP_VALUE, PPO_CLIP_VALUE
                )
                vf_loss1 = (new_value - ret_b).pow(2)
                vf_loss2 = (value_clipped - ret_b).pow(2)
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Surprise loss (DualStream only: predictive coding)
                surprise_loss = torch.tensor(0.0, device=DEVICE)
                if self.is_dual_stream and self.surprise_coeff > 0:
                    # Forward pass to get return predictions
                    _, _, aux = self.policy.forward(obs_b)
                    if "return_pred" in aux:
                        # Target: per-asset WM 1-bar return predictions from observation
                        # obs layout: per asset i, index i*10+0 = 1-bar return prediction
                        from config import PER_ASSET_OBS_DIM, NUM_ASSETS
                        per_asset_ret = obs_b[:, [i * PER_ASSET_OBS_DIM for i in range(NUM_ASSETS)]]
                        surprise_loss = 0.5 * (aux["return_pred"] - per_asset_ret.detach()).pow(2).mean()

                # Total loss
                loss = (
                    pg_loss
                    + self.value_coeff * value_loss
                    + self.entropy_coeff * entropy_loss
                    + self.surprise_coeff * surprise_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                total_surprise_loss += surprise_loss.item()
                n_updates += 1

        metrics = {
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "n_updates": n_updates,
        }
        if self.is_dual_stream:
            metrics["surprise_loss"] = total_surprise_loss / max(n_updates, 1)

        return metrics

    def save_checkpoint(self, path, total_steps: int, rollout_idx: int, extra: dict = None):
        """Save a full training checkpoint (policy + optimizer + normalization + metrics)."""
        ckpt = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "policy_type": "dual_stream" if self.is_dual_stream else "baseline",
            "total_steps": total_steps,
            "rollout_idx": rollout_idx,
            "best_eval_reward": self.best_eval_reward,
            "best_eval_sharpe": self.best_eval_sharpe,
            "episode_rewards": self.episode_rewards[-100:],  # keep last 100
            "episode_lengths": self.episode_lengths[-100:],
            "training_metrics": self.training_metrics[-50:],  # keep last 50
            "obs_rms": self.obs_rms.state_dict(),
            "ret_rms": self.ret_rms.state_dict(),
        }
        if extra:
            ckpt.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        """Load a training checkpoint and restore state. Returns (total_steps, rollout_idx)."""
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_eval_reward = ckpt.get("best_eval_reward", float("-inf"))
        self.best_eval_sharpe = ckpt.get("best_eval_sharpe", float("-inf"))
        self.episode_rewards = ckpt.get("episode_rewards", [])
        self.episode_lengths = ckpt.get("episode_lengths", [])
        self.training_metrics = ckpt.get("training_metrics", [])
        # Restore running normalization stats (critical for consistent obs/reward scale)
        if "obs_rms" in ckpt:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        if "ret_rms" in ckpt:
            self.ret_rms.load_state_dict(ckpt["ret_rms"])
        total_steps = ckpt.get("total_steps", 0)
        rollout_idx = ckpt.get("rollout_idx", 0)
        print(f"  [RESUME] Loaded checkpoint: step {total_steps:,}, rollout {rollout_idx}")
        print(f"  [RESUME] Best reward: {self.best_eval_reward:.4f}, "
              f"Best Sharpe: {self.best_eval_sharpe:.4f}")
        return total_steps, rollout_idx

    def train(self, total_timesteps: int, log_every: int = 5, eval_fn=None,
              agent_tag: str = "", resume_path=None, checkpoint_every: int = 50):
        """
        Main training loop with progress bar, ETA, formatted eval, and checkpoint resume.

        Args:
            total_timesteps: Total environment steps to train for
            log_every: Print metrics every N rollouts
            eval_fn: Optional function(policy) -> dict for evaluation
            agent_tag: Tag for checkpoint naming
            resume_path: Path to checkpoint to resume from (or None)
            checkpoint_every: Save periodic checkpoint every N rollouts
        """
        n_rollouts = total_timesteps // self.n_steps
        total_steps = 0
        start_rollout = 0

        # Resume from checkpoint if provided
        if resume_path is not None and resume_path.exists():
            total_steps, start_rollout = self.load_checkpoint(resume_path)
            start_rollout += 1  # continue from next rollout

        arch = "DualStream" if self.is_dual_stream else "Baseline"
        print(f"\n  PPO Training ({arch}): {total_timesteps:,} steps "
              f"({n_rollouts} rollouts x {self.n_steps} steps)")
        print(f"  Policy params: {sum(p.numel() for p in self.policy.parameters()):,}")
        if self.crucible is not None:
            print(f"  Crucible: ENABLED (adversarial difficulty tracking)")
        if self.is_dual_stream:
            print(f"  Surprise loss coeff: {self.surprise_coeff}")
        if start_rollout > 0:
            print(f"  Resuming from rollout {start_rollout}/{n_rollouts} "
                  f"(step {total_steps:,})")
        print()

        eval_every = log_every * 5  # evaluate every 25 rollouts by default
        rollout_times = []  # wall-clock per rollout for ETA
        train_start = time.time()

        pbar = tqdm(
            range(start_rollout, n_rollouts),
            initial=start_rollout,
            total=n_rollouts,
            desc="PPO Training",
            unit="rollout",
            ncols=120,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )

        for rollout_idx in pbar:
            rollout_start = time.time()

            # Collect experience
            buffer = self.collect_rollout()
            total_steps += self.n_steps

            # PPO update
            metrics = self.update(buffer)

            # Track wall-clock time
            rollout_elapsed = time.time() - rollout_start
            rollout_times.append(rollout_elapsed)

            # Update progress bar postfix
            recent_rewards = self.episode_rewards[-10:] if self.episode_rewards else [0]
            avg_reward = np.mean(recent_rewards)
            pbar.set_postfix({
                "R": f"{avg_reward:+.2f}",
                "PG": f"{metrics['pg_loss']:.3f}",
                "VF": f"{metrics['value_loss']:.3f}",
                "Ent": f"{metrics['entropy']:.3f}",
                "s/r": f"{rollout_elapsed:.1f}s",
            })

            # Detailed logging
            if (rollout_idx + 1) % log_every == 0:
                recent_lengths = self.episode_lengths[-10:] if self.episode_lengths else [0]
                avg_length = np.mean(recent_lengths)

                # ETA calculation
                avg_time = np.mean(rollout_times[-20:])
                remaining = n_rollouts - rollout_idx - 1
                eta_sec = avg_time * remaining
                eta_min = eta_sec / 60
                elapsed_min = (time.time() - train_start) / 60

                extra = ""
                if "surprise_loss" in metrics:
                    extra = f" | Surp: {metrics['surprise_loss']:>6.4f}"

                tqdm.write(
                    f"  R{rollout_idx+1:4d}/{n_rollouts} | "
                    f"Steps: {total_steps:>8,} | "
                    f"Reward: {avg_reward:>+7.2f} | "
                    f"EpLen: {avg_length:>5.0f} | "
                    f"PG: {metrics['pg_loss']:>7.4f} | "
                    f"VF: {metrics['value_loss']:>7.4f} | "
                    f"Ent: {metrics['entropy']:>5.3f}"
                    + extra
                    + f" | {elapsed_min:.0f}m/{elapsed_min+eta_min:.0f}m"
                )

                # Log augmentation scenarios
                if hasattr(buffer, "_scenario_counts") and buffer._scenario_counts:
                    sc_str = " | ".join(
                        f"{sc}:{cnt}" for sc, cnt in sorted(buffer._scenario_counts.items())
                    )
                    tqdm.write(f"    [AUG] {sc_str}")

                self.training_metrics.append({
                    "step": total_steps,
                    "avg_reward": float(avg_reward),
                    "avg_length": float(avg_length),
                    "wall_time": time.time() - train_start,
                    **metrics,
                })

            # Evaluation + best checkpoint
            if eval_fn is not None and (rollout_idx + 1) % eval_every == 0:
                eval_metrics = eval_fn(self.policy)
                tqdm.write(f"\n    [EVAL @ step {total_steps:,}]")
                tqdm.write(_format_eval(eval_metrics))

                # Per-scenario reward stats
                if hasattr(self.env, "get_scenario_stats"):
                    sc_stats = self.env.get_scenario_stats()
                    if sc_stats:
                        tqdm.write(f"    [AUG] Per-scenario rewards:")
                        for sc_name, stats in sorted(sc_stats.items()):
                            tqdm.write(f"      {sc_name}: {stats['count']} ep, "
                                       f"mean={stats['mean_reward']:.3f}")

                # Track best reward + best Sharpe
                eval_reward = eval_metrics.get("mean_return", 0)
                eval_sharpe = eval_metrics.get("sharpe", float("-inf"))

                improved = False
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    improved = True
                if eval_sharpe > self.best_eval_sharpe:
                    self.best_eval_sharpe = eval_sharpe
                    improved = True

                if improved and agent_tag:
                    best_path = AGENT_MODEL_DIR / f"agent_{agent_tag}_best.pt"
                    self.save_checkpoint(best_path, total_steps, rollout_idx, extra={
                        "eval_reward": eval_reward,
                        "eval_sharpe": eval_sharpe,
                        "eval_metrics": eval_metrics,
                    })
                    tqdm.write(
                        f"    [BEST] Saved: {best_path.name} "
                        f"(reward={eval_reward:+.4f}, sharpe={eval_sharpe:+.2f})"
                    )

                tqdm.write(
                    f"    [TRACK] Best reward: {self.best_eval_reward:+.4f} | "
                    f"Best Sharpe: {self.best_eval_sharpe:+.2f}"
                )
                tqdm.write("")

            # Periodic checkpoint (for resume)
            if agent_tag and (rollout_idx + 1) % checkpoint_every == 0:
                ckpt_path = AGENT_MODEL_DIR / f"agent_{agent_tag}_latest.pt"
                self.save_checkpoint(ckpt_path, total_steps, rollout_idx)
                tqdm.write(f"    [CKPT] Saved: {ckpt_path.name} (step {total_steps:,})")

        pbar.close()
        return self.policy, self.training_metrics
