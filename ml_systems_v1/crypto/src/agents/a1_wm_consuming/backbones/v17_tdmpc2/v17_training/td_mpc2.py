"""V17 -- TD-MPC2 (Hansen 2024, paper 2310.16828).

Decoupled World Model + MPPI planner. Different from DreamerV3 in that:
  - WM is value-equivalent (dynamics, reward, value) but NOT generative
    (no reconstruction).
  - At inference, run MPPI (Model Predictive Path Integral) sampling-based
    planning over imagined trajectories instead of amortized policy.
  - Trains on the same offline data as V16 / V19 — a frozen WM, not RL bootstrap.

Why include TD-MPC2 alongside DreamerV3:
  - Provides a planning-based agent (orthogonal paradigm to DreamerV3's
    amortized policy). MPPI excels in environments where re-planning each
    step is cheap relative to model error.
  - Hansen 2024 shows TD-MPC2 scales 80M -> 1B params with consistent gains;
    DreamerV3 doesn't scale as cleanly past 200M.

The full TD-MPC2 implementation is several thousand LOC. This is the core
WM component (encoder + dynamics + reward + value) + MPPI planner. Training
loop is a separate file (v17_train.py, in M5 cleanup).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2,
        dropout: float = 0.1) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class TDMPC2WorldModel(nn.Module):
    """Value-equivalent WM. NOT generative (no reconstruction loss).

    Components:
      Encoder:   obs -> latent z
      Dynamics:  (z_t, a_t) -> z_{t+1}
      Reward:    (z_t, a_t) -> r_t
      Value:     z_t -> V(z_t)
      Continue:  z_t -> Bernoulli(continue)

    Loss: latent-prediction consistency + reward MSE + TD value loss.
    """
    def __init__(self,
                 obs_dim: int = 121,
                 action_dim: int = 1,
                 latent_dim: int = 64,
                 hidden_dim: int = 256,
                 n_assets: int = 10,
                 asset_embed_dim: int = 16,
                 horizon: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.asset_embed = nn.Embedding(n_assets, asset_embed_dim)
        encoder_in_dim = obs_dim + asset_embed_dim
        self.encoder = mlp(encoder_in_dim, hidden_dim, latent_dim, n_hidden=2)
        self.dynamics = mlp(latent_dim + action_dim, hidden_dim, latent_dim, n_hidden=2)
        self.reward = mlp(latent_dim + action_dim, hidden_dim, 1, n_hidden=2)
        self.value = mlp(latent_dim, hidden_dim, 1, n_hidden=2)
        self.continue_head = mlp(latent_dim, hidden_dim, 1, n_hidden=2)

        # Target value (slow EMA)
        self.target_value = mlp(latent_dim, hidden_dim, 1, n_hidden=2)
        self.target_value.load_state_dict(self.value.state_dict())
        for p in self.target_value.parameters():
            p.requires_grad = False
        self.target_tau = 0.05

    def encode(self, obs: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        emb = self.asset_embed(asset_id)
        x = torch.cat([obs, emb], dim=-1)
        return self.encoder(x)

    def step(self, z: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        za = torch.cat([z, a], dim=-1)
        z_next = self.dynamics(za)
        r = self.reward(za).squeeze(-1)
        return z_next, r

    def forward_train(self, obs_seq: torch.Tensor, actions_seq: torch.Tensor,
                      rewards_seq: torch.Tensor, asset_ids: torch.Tensor) -> dict:
        """obs/actions/rewards: [B, T, *]. Returns aggregated losses."""
        B, T, _ = obs_seq.shape
        # Encode every step
        encoded = []
        for t in range(T):
            encoded.append(self.encode(obs_seq[:, t, :], asset_ids))
        z = torch.stack(encoded, dim=1)  # [B, T, latent]

        # Dynamics + reward consistency: predict z[t+1], r[t] from (z[t], a[t])
        z_pred_next = []
        r_pred = []
        for t in range(T - 1):
            zn, rn = self.step(z[:, t, :], actions_seq[:, t, :])
            z_pred_next.append(zn)
            r_pred.append(rn)
        z_pred_next = torch.stack(z_pred_next, dim=1)
        r_pred = torch.stack(r_pred, dim=1)

        # Latent consistency loss
        consistency_loss = F.mse_loss(z_pred_next, z[:, 1:, :].detach())
        reward_loss = F.mse_loss(r_pred, rewards_seq[:, :T-1])

        # TD value loss
        with torch.no_grad():
            target_v_next = self.target_value(z[:, 1:, :]).squeeze(-1)
            value_target = rewards_seq[:, :T-1] + 0.99 * target_v_next
        value_pred = self.value(z[:, :T-1, :]).squeeze(-1)
        value_loss = F.mse_loss(value_pred, value_target)

        total = consistency_loss + reward_loss + value_loss

        return {
            "loss": total,
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "value_loss": value_loss,
        }

    def soft_update_target(self):
        with torch.no_grad():
            for p, tp in zip(self.value.parameters(), self.target_value.parameters()):
                tp.data.mul_(1 - self.target_tau).add_(p.data, alpha=self.target_tau)


class MPPIPlanner:
    """Model Predictive Path Integral planner. Samples K trajectories of length H,
    weights by exponential of return, picks best action."""

    def __init__(self, wm: TDMPC2WorldModel, action_dim: int = 1,
                 n_samples: int = 64, horizon: int = 5,
                 temperature: float = 0.5, n_iters: int = 6):
        self.wm = wm
        self.action_dim = action_dim
        self.n_samples = n_samples
        self.horizon = horizon
        self.temperature = temperature
        self.n_iters = n_iters

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, asset_id: torch.Tensor) -> torch.Tensor:
        """Plan one action for a single state. Returns [action_dim] action."""
        B = obs.shape[0]
        z0 = self.wm.encode(obs, asset_id)  # [B, latent]
        # Initial action mean (zeros) + std (1.0)
        mean = torch.zeros(self.horizon, self.action_dim, device=z0.device)
        std = torch.ones(self.horizon, self.action_dim, device=z0.device)
        for _ in range(self.n_iters):
            # Sample K action sequences
            actions = (mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.n_samples, self.horizon, self.action_dim, device=z0.device)
            ).clamp(-1, 1)
            # Roll out for each sample
            z = z0.expand(self.n_samples, -1)
            total_r = torch.zeros(self.n_samples, device=z0.device)
            for t in range(self.horizon):
                z, r = self.wm.step(z, actions[:, t, :])
                total_r = total_r + (0.99 ** t) * r
            # Add bootstrap value at end
            total_r = total_r + self.wm.value(z).squeeze(-1) * (0.99 ** self.horizon)
            # Re-fit mean / std weighted by exp(total_r / temp)
            weights = F.softmax(total_r / self.temperature, dim=0)  # [K]
            mean = (weights.unsqueeze(-1).unsqueeze(-1) * actions).sum(dim=0)
            std = ((weights.unsqueeze(-1).unsqueeze(-1) *
                    (actions - mean.unsqueeze(0)).pow(2)).sum(dim=0) + 1e-6).sqrt()
        return mean[0]  # first-step action


def smoke_test():
    torch.manual_seed(0)
    wm = TDMPC2WorldModel(obs_dim=121, action_dim=1, n_assets=10)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"[v17] TDMPC2 WM params: {n_params:,}")

    B, T, C = 4, 16, 121
    obs = torch.randn(B, T, C)
    actions = torch.zeros(B, T, 1)
    rewards = torch.randn(B, T) * 0.01
    asset_ids = torch.zeros(B, dtype=torch.long)

    out = wm.forward_train(obs, actions, rewards, asset_ids)
    print(f"[v17] train losses: total={out['loss'].item():.4f}  "
          f"consistency={out['consistency_loss'].item():.4f}  "
          f"reward={out['reward_loss'].item():.4f}  "
          f"value={out['value_loss'].item():.4f}")
    out["loss"].backward()
    has_grad = sum(1 for p in wm.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_p = sum(1 for _ in wm.parameters())
    print(f"[v17] backward OK; {has_grad}/{total_p} params have non-zero grad")

    wm.soft_update_target()
    planner = MPPIPlanner(wm, action_dim=1, n_samples=32, horizon=5)
    test_obs = torch.randn(1, 121)
    test_aid = torch.zeros(1, dtype=torch.long)
    a = planner.plan(test_obs, test_aid)
    print(f"[v17] MPPI plan OK; action shape={tuple(a.shape)}, value={a.item():.4f}")


def main_cli():
    """STUB: V17 ships the TD-MPC2 model class only; trainer is pending."""
    import argparse
    parser = argparse.ArgumentParser(
        description="V17 TD-MPC2 (model only, trainer pending). Use --smoke."
    )
    parser.add_argument("--features", type=int, default=121, help="compat")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the parameter-count smoke test.")
    args = parser.parse_args()
    if args.smoke:
        smoke_test()
        return
    print("[V17] STUB: TD-MPC2 trainer not yet implemented. Use --smoke to "
          "run the model-class smoke check.")


if __name__ == "__main__":
    main_cli()
