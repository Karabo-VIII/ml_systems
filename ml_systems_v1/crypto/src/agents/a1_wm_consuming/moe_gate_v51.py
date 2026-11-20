"""M4 -- MoE gate over champion strategies, retried on v51 schema.

The 2026-04-22 MoE attempt CONCEDED on v50 schema (per
ml_framework_4plus_avenues_2026_04_22.md): variants of XGB d-neut rankers
were too correlated for the gate to add value.

This v51 retry differs in two ways:
  1. **Richer state input**: 154 v51 features vs 41 v50 features. The gate
     now sees Hawkes branching, ETF flow, S3 metrics, basis features etc.
     that may discriminate better between champion strategies.
  2. **Different champion mix**: includes paradigm-different routes:
       - xsec_ranker (XGB rank:ndcg, dollar-neutral momentum)
       - dib_flow_duo (BTC+ETH DIB flow imbalance)
       - prod_meta_combined (rule-based + meta-labeler)
       - asym_breakout (N=10 high + trail)
       - frontier_dib + stable + ETF overlays
     instead of 3 d-neut variants of the same paradigm.

The gate outputs softmax weights over K champions per day. Final allocation
is the weighted sum of champion daily returns.

Run:
  python src/agents/a1_wm_consuming/moe_gate_v51.py --train  (after Job 2 builds full v51)
  python src/agents/a1_wm_consuming/moe_gate_v51.py --smoke  (architecture sanity only)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGate(nn.Module):
    """Gating network: state -> softmax weights over K champions."""

    def __init__(self,
                 state_dim: int = 154,
                 n_champions: int = 5,
                 hidden: int = 128,
                 dropout: float = 0.2,
                 entropy_coef: float = 0.01):
        super().__init__()
        self.n_champions = n_champions
        self.entropy_coef = entropy_coef
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_champions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: [B, state_dim] -> weights [B, n_champions] (softmax)."""
        logits = self.trunk(state)
        return F.softmax(logits, dim=-1)

    def loss(self, weights: torch.Tensor, champion_returns: torch.Tensor,
             daily_costs: torch.Tensor | None = None) -> dict:
        """
        Args:
            weights: [B, K] gate weights
            champion_returns: [B, K] daily returns of each champion on that day
            daily_costs: [B, K] optional per-champion daily transaction cost
        Returns:
            dict with 'loss' (negative weighted return + entropy bonus).
        """
        if daily_costs is not None:
            net = champion_returns - daily_costs
        else:
            net = champion_returns
        weighted_ret = (weights * net).sum(dim=-1)  # [B]
        # Maximize return = minimize negative.
        primary = -weighted_ret.mean()
        # Entropy bonus prevents gate from collapsing onto one champion always.
        entropy = -(weights * (weights.clamp(min=1e-9)).log()).sum(dim=-1).mean()
        loss = primary - self.entropy_coef * entropy
        return {
            "loss": loss,
            "weighted_return": weighted_ret.mean().detach(),
            "entropy": entropy.detach(),
            "primary": primary.detach(),
        }


class MoEGateTrainer:
    """Pipeline-friendly wrapper. Reads v51 + champion daily-equity series, trains gate."""

    def __init__(self, state_features: list[str], champion_seeds: list[str],
                 hidden: int = 128, lr: float = 1e-3, weight_decay: float = 1e-4):
        self.state_features = state_features
        self.champion_seeds = champion_seeds
        self.gate = MoEGate(
            state_dim=len(state_features),
            n_champions=len(champion_seeds),
            hidden=hidden,
        )
        self.optimizer = torch.optim.AdamW(
            self.gate.parameters(), lr=lr, weight_decay=weight_decay,
        )

    def train_step(self, state_batch: torch.Tensor,
                   champion_returns_batch: torch.Tensor,
                   costs_batch: torch.Tensor | None = None) -> dict:
        weights = self.gate(state_batch)
        info = self.gate.loss(weights, champion_returns_batch, costs_batch)
        self.optimizer.zero_grad()
        info["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.gate.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {k: float(v) for k, v in info.items()}


def smoke_test():
    torch.manual_seed(0)
    state_dim = 154
    n_champ = 5
    B = 32

    gate = MoEGate(state_dim=state_dim, n_champions=n_champ)
    n_params = sum(p.numel() for p in gate.parameters())
    print(f"[moe-v51] params: {n_params:,}")

    state = torch.randn(B, state_dim)
    weights = gate(state)
    assert weights.shape == (B, n_champ)
    # softmax sanity
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B), atol=1e-4)

    # Mock champion returns: 3 positive-skew, 2 negative-skew
    champ_ret = torch.cat([
        torch.randn(B, 3) * 0.01 + 0.005,
        torch.randn(B, 2) * 0.01 - 0.005,
    ], dim=-1)
    info = gate.loss(weights, champ_ret)
    print(f"[moe-v51] loss={info['loss'].item():.4f}, "
          f"weighted_return={info['weighted_return'].item():+.4f}, "
          f"entropy={info['entropy'].item():.4f}")

    # Train 50 steps and watch entropy decline + return increase
    trainer = MoEGateTrainer(
        state_features=[f"f_{i}" for i in range(state_dim)],
        champion_seeds=[f"champ_{i}" for i in range(n_champ)],
    )
    print("[moe-v51] running 50 train steps on synthetic data...")
    history = []
    for step in range(50):
        batch_state = torch.randn(B, state_dim)
        # Simulate some champion-state correlation (champ 0 wins when state[0] > 0)
        batch_returns = torch.randn(B, n_champ) * 0.005
        batch_returns[:, 0] += 0.005 * batch_state[:, 0].clamp(0, 5) / 5
        info = trainer.train_step(batch_state, batch_returns)
        history.append(info)
    print(f"[moe-v51] step 0:  loss={history[0]['loss']:.4f}  "
          f"return={history[0]['weighted_return']:+.4f}  ent={history[0]['entropy']:.3f}")
    print(f"[moe-v51] step 49: loss={history[-1]['loss']:.4f}  "
          f"return={history[-1]['weighted_return']:+.4f}  ent={history[-1]['entropy']:.3f}")
    print("[moe-v51] smoke test PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.smoke or not args.train:
        smoke_test()
        return
    print("[moe-v51] real training requires Job 2 (full 53-asset v51 build) + champion daily-equity CSVs")
    print("[moe-v51] use logs/paper_trader_v2/seeds/<seed>/daily_snapshot.csv as champion returns")


if __name__ == "__main__":
    main()
