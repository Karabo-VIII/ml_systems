"""
Actor-Critic Policy Networks
==============================
Two architectures for the trading agent:

1. ActorCritic: Standard MLP actor-critic (baseline, 93K params)
2. DualStreamActorCritic: Enhanced with ideas from TEMP_4:
   - Dual-stream architecture (alpha + risk separation)
   - Hebbian plasticity (fast online adaptation within episodes)
   - Surprise-gated action (conservative when confused)
   - Return prediction head (predictive coding)

Both fit within 8GB VRAM alongside the frozen world model.
"""

__class_tag__ = "A1"  # WM-consuming agent policy nets (doc SS1.8)

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from config import (
    TOTAL_OBS_DIM, ACTION_DIM, NUM_ASSETS, PER_ASSET_OBS_DIM, OBS_RETURN_PREDS,
    POLICY_HIDDEN_DIM, POLICY_N_LAYERS,
    POLICY_LOG_STD_MIN, POLICY_LOG_STD_MAX,
    VALUE_HIDDEN_DIM, VALUE_N_LAYERS,
)


def build_mlp(input_dim: int, hidden_dim: int, output_dim: int,
              n_layers: int, activation: str = "silu") -> nn.Sequential:
    """Build a simple MLP with the specified architecture."""
    act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]

    layers = []
    prev_dim = input_dim
    for _ in range(n_layers):
        layers.extend([nn.Linear(prev_dim, hidden_dim), act_fn()])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))

    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Standard Actor-Critic (Baseline)
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """
    Standard MLP actor-critic for PPO (baseline).
    Separate actor and critic networks, ~93K params.
    """

    def __init__(
        self,
        obs_dim: int = TOTAL_OBS_DIM,
        action_dim: int = ACTION_DIM,
        actor_hidden: int = POLICY_HIDDEN_DIM,
        actor_layers: int = POLICY_N_LAYERS,
        critic_hidden: int = VALUE_HIDDEN_DIM,
        critic_layers: int = VALUE_N_LAYERS,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Actor: outputs mean of Gaussian policy
        self.actor_mean = build_mlp(obs_dim, actor_hidden, action_dim, actor_layers)

        # Learnable log_std (initialized at max for exploration, gradient can push down)
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), POLICY_LOG_STD_MAX)
        )

        # Critic: outputs scalar value
        self.critic = build_mlp(obs_dim, critic_hidden, 1, critic_layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)."""
        for module in [self.actor_mean, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.zeros_(layer.bias)

        # Last layer of actor: small init for near-zero initial actions
        last_actor = None
        for layer in reversed(list(self.actor_mean)):
            if isinstance(layer, nn.Linear):
                last_actor = layer
                break
        if last_actor is not None:
            nn.init.orthogonal_(last_actor.weight, gain=0.01)

        # Last layer of critic: small init for near-zero initial values
        last_critic = None
        for layer in reversed(list(self.critic)):
            if isinstance(layer, nn.Linear):
                last_critic = layer
                break
        if last_critic is not None:
            nn.init.orthogonal_(last_critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Actor
        mean = self.actor_mean(obs)
        log_std = self.actor_log_std.expand_as(mean)
        log_std = log_std.clamp(POLICY_LOG_STD_MIN, POLICY_LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)

        # Critic
        value = self.critic(obs)

        return dist, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist, value = self.forward(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        # No squashing here. Environment clips to [-MAX_POSITION_FRAC, MAX_POSITION_FRAC].
        # Raw Gaussian samples stored in buffer => evaluate_actions log_prob is consistent.

        return action, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist, value = self.forward(obs)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy


# ---------------------------------------------------------------------------
# Dual-Stream Actor-Critic (TEMP_4-Inspired)
# ---------------------------------------------------------------------------

# Per-asset observation layout:
#   [0:4] = return predictions (alpha signal)
#   [4:7] = regime probabilities (alpha signal)
#   [7]   = uncertainty (risk signal)
#   [8]   = position (portfolio state)
#   [9]   = unrealized PnL (portfolio state)
# Plus 1 global feature (cash fraction)

# Alpha features per asset: return preds (active horizons) + regime probs = 5
ALPHA_FEATURES_PER_ASSET = OBS_RETURN_PREDS + 3  # 2 + 3 = 5
# Risk features per asset: uncertainty = 1
RISK_FEATURES_PER_ASSET = 1
# Portfolio features per asset: position + pnl = 2
PORTFOLIO_FEATURES_PER_ASSET = 2


class HebbianLayer(nn.Module):
    """
    Hebbian plasticity layer for fast online adaptation.

    Maintains a "fast weight" matrix that adapts within episodes using
    Hebbian learning: neurons that fire together wire together.

    Inspired by V-Genesis (TEMP_4) Dual-Stream Hebbian RNN.

    The fast weights are NOT part of the trainable parameters --
    they are computed online during forward passes and reset per episode.
    """

    def __init__(self, hidden_dim: int, hebb_lr: float = 0.001, hebb_decay: float = 0.995):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Learnable plasticity parameters (evolved during training)
        self.hebb_lr = nn.Parameter(torch.tensor(hebb_lr))
        self.hebb_decay = nn.Parameter(torch.tensor(hebb_decay))
        # Linear projection for plastic component
        self.plastic_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.plastic_proj.weight)

    def forward(self, h: torch.Tensor, fast_weights: torch.Tensor | None = None,
                uncertainty: torch.Tensor | None = None):
        """
        Apply Hebbian plasticity to hidden state.

        Args:
            h: [B, hidden_dim] current hidden state
            fast_weights: [B, hidden_dim, hidden_dim] or None (creates zeros)
            uncertainty: [B] uncertainty signal in [0, 1]. Higher = learn faster.
                         Comes from posterior entropy (mean across assets).

        Returns:
            h_out: [B, hidden_dim] hidden state with plastic component
            fast_weights: [B, hidden_dim, hidden_dim] updated fast weights
        """
        B = h.shape[0]
        if fast_weights is None:
            fast_weights = torch.zeros(B, self.hidden_dim, self.hidden_dim,
                                       device=h.device, dtype=h.dtype)

        # Plastic contribution: h @ fast_weights
        plastic = torch.bmm(h.unsqueeze(1), fast_weights).squeeze(1)

        # Modulate with learned projection
        h_out = h + self.plastic_proj(plastic)

        # Base learning rate
        lr_base = torch.sigmoid(self.hebb_lr) * 0.01  # Clamp to [0, 0.01]

        # Adaptive modulation: higher uncertainty -> faster learning
        if uncertainty is not None:
            if not isinstance(uncertainty, torch.Tensor):
                uncertainty = torch.tensor(uncertainty, dtype=h.dtype, device=h.device)
            unc = uncertainty.view(-1, 1, 1).clamp(0.0, 1.0)
            lr = lr_base * (1.0 + unc)  # range [lr_base, 2 * lr_base]
        else:
            lr = lr_base

        decay = torch.sigmoid(self.hebb_decay)  # Clamp to [0, 1]

        # delta_A = lr * outer(h, h_out)
        delta = lr * torch.bmm(h.unsqueeze(2), h_out.unsqueeze(1))

        # Decay old + add new
        fast_weights = decay * fast_weights + delta

        # Clamp to prevent explosion (critical stability fix from V75)
        fast_weights = fast_weights.clamp(-0.5, 0.5)

        return h_out, fast_weights


class DualStreamActorCritic(nn.Module):
    """
    Enhanced actor-critic with ideas from TEMP_4 (V-Genesis, V70, V75).

    Architecture:
      - Alpha Stream: processes return predictions + regime (predictive signals)
      - Risk Stream: processes uncertainty + volatility (risk/physics signals)
      - Confidence Gate: risk stream gates alpha (suppress action when confused)
      - Hebbian Plasticity: fast weights for intra-episode adaptation
      - Return Prediction: auxiliary head for predictive coding (surprise signal)
      - Critic: separate network (standard PPO)

    The dual-stream separation means the alpha path learns WHAT to trade,
    while the risk path learns WHEN NOT to trade. The confidence gate is
    a sigmoid that suppresses the action mean when risk is high.

    ~120K params total (still fits in VRAM budget).
    """

    def __init__(
        self,
        obs_dim: int = TOTAL_OBS_DIM,
        action_dim: int = ACTION_DIM,
        num_assets: int = NUM_ASSETS,
        hidden_dim: int = POLICY_HIDDEN_DIM,
        n_layers: int = 2,
        critic_hidden: int = VALUE_HIDDEN_DIM,
        critic_layers: int = VALUE_N_LAYERS,
        use_hebbian: bool = True,
        use_surprise_gate: bool = True,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_assets = num_assets
        self.hidden_dim = hidden_dim
        self.use_hebbian = use_hebbian
        self.use_surprise_gate = use_surprise_gate

        # --- Observation decomposition ---
        # Alpha features: return predictions (2 active) + regime probs (3) per asset
        alpha_dim = num_assets * ALPHA_FEATURES_PER_ASSET  # 10 * 5 = 50
        # Risk features: uncertainty (1) per asset
        risk_dim = num_assets * RISK_FEATURES_PER_ASSET  # 10 * 1 = 10
        # Portfolio state: position (1) + pnl (1) per asset + cash (1)
        portfolio_dim = num_assets * PORTFOLIO_FEATURES_PER_ASSET + 1  # 10 * 2 + 1 = 21

        # --- Alpha Stream (learns WHAT to trade) ---
        self.alpha_encoder = nn.Sequential(
            nn.Linear(alpha_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Risk Stream (learns WHEN NOT to trade) ---
        self.risk_encoder = nn.Sequential(
            nn.Linear(risk_dim + portfolio_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # --- Confidence Gate (risk gates alpha) ---
        # Produces per-asset confidence in [0, 1]
        self.confidence_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid(),
        )

        # --- Hebbian Plasticity (fast online adaptation) ---
        if use_hebbian:
            self.hebbian = HebbianLayer(hidden_dim)

        # --- Actor Head (gated alpha -> action mean) ---
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Learnable log_std (initialized at max for exploration, gradient can push down)
        self.actor_log_std = nn.Parameter(
            torch.full((action_dim,), POLICY_LOG_STD_MAX)
        )

        # --- Return Prediction Head (predictive coding / surprise) ---
        if use_surprise_gate:
            # Predicts next-bar return per asset (for computing surprise)
            self.return_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, action_dim),
            )

        # --- Critic (separate, standard) ---
        self.critic = build_mlp(obs_dim, critic_hidden, 1, critic_layers)

        self._init_weights()

    def _init_weights(self):
        """Initialize with small weights for stable start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Small init for actor output (near-zero initial actions)
        last_actor = list(self.actor_head)[-1]
        if isinstance(last_actor, nn.Linear):
            nn.init.orthogonal_(last_actor.weight, gain=0.01)

        # Small init for confidence gate output (start near 0.5)
        gate_layers = list(self.confidence_gate)
        for layer in reversed(gate_layers):
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
                break

    def _decompose_obs(self, obs: torch.Tensor):
        """
        Split observation into alpha, risk, and portfolio components.

        Observation layout per asset (8 dims):
          [0:2]   return predictions -- active horizons only (alpha)
          [2:5]   regime probabilities (alpha)
          [5]     uncertainty (risk)
          [6]     position (portfolio)
          [7]     unrealized PnL (portfolio)
        Last dim: cash fraction (global portfolio)
        """
        B = obs.shape[0]

        alpha_parts = []
        risk_parts = []
        portfolio_parts = []

        unc_off = ALPHA_FEATURES_PER_ASSET      # uncertainty index within per-asset block
        port_off = ALPHA_FEATURES_PER_ASSET + 1  # portfolio start index

        for i in range(self.num_assets):
            offset = i * PER_ASSET_OBS_DIM
            # Alpha: return preds + regime probs
            alpha_parts.append(obs[:, offset:offset + ALPHA_FEATURES_PER_ASSET])
            # Risk: uncertainty
            risk_parts.append(obs[:, offset + unc_off:offset + unc_off + 1])
            # Portfolio: position + pnl
            portfolio_parts.append(obs[:, offset + port_off:offset + port_off + 2])

        # Global: cash fraction (last element)
        cash = obs[:, -1:]

        alpha = torch.cat(alpha_parts, dim=-1)    # [B, 50]
        risk = torch.cat(risk_parts, dim=-1)       # [B, 10]
        portfolio = torch.cat(portfolio_parts + [cash], dim=-1)  # [B, 21]

        return alpha, risk, portfolio

    def forward(
        self,
        obs: torch.Tensor,
        fast_weights: torch.Tensor | None = None,
    ) -> tuple[Normal, torch.Tensor, dict]:
        """
        Forward pass with dual-stream processing.

        Returns:
            dist: Normal distribution over actions
            value: Value estimate [B, 1]
            aux: Dict with auxiliary outputs (confidence, return_pred, fast_weights)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        alpha_obs, risk_obs, portfolio_obs = self._decompose_obs(obs)

        # Alpha stream
        h_alpha = self.alpha_encoder(alpha_obs)  # [B, hidden_dim]

        # Risk stream (includes portfolio state for context)
        risk_input = torch.cat([risk_obs, portfolio_obs], dim=-1)
        h_risk = self.risk_encoder(risk_input)  # [B, hidden_dim]

        # Hebbian plasticity on alpha stream (uncertainty-modulated LR)
        if self.use_hebbian:
            uncertainty_signal = risk_obs.mean(dim=-1)  # [B] mean entropy
            h_alpha, fast_weights = self.hebbian(
                h_alpha, fast_weights, uncertainty=uncertainty_signal
            )

        # Confidence gate: risk stream gates alpha
        confidence = self.confidence_gate(h_risk)  # [B, action_dim] in [0, 1]

        # Fuse streams for actor
        h_fused = torch.cat([h_alpha, h_risk], dim=-1)  # [B, 2 * hidden_dim]
        action_mean_raw = self.actor_head(h_fused)  # [B, action_dim]

        # Apply confidence gate: suppress mean when confused
        action_mean = action_mean_raw * confidence

        # Action distribution
        log_std = self.actor_log_std.expand_as(action_mean)
        log_std = log_std.clamp(POLICY_LOG_STD_MIN, POLICY_LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(action_mean, std)

        # Critic (uses full observation)
        value = self.critic(obs)

        # Auxiliary outputs
        aux = {
            "confidence": confidence,
            "fast_weights": fast_weights,
        }

        # Return prediction (for surprise/predictive coding)
        if self.use_surprise_gate:
            return_pred = self.return_predictor(h_alpha)  # [B, action_dim]
            aux["return_pred"] = return_pred

        return dist, value, aux

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        fast_weights: torch.Tensor | None = None,
    ):
        """
        Sample an action from the policy.

        Returns:
            action: [B, action_dim] clipped to [-1, 1]
            log_prob: [B]
            value: [B, 1]
            aux: Dict with confidence, return_pred, fast_weights
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        dist, value, aux = self.forward(obs, fast_weights)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        # No squashing. Environment clips. See ActorCritic.get_action comment.

        return action, log_prob, value, aux

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate previously taken actions (for PPO update)."""
        dist, value, _ = self.forward(obs)

        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, value, entropy


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import DEVICE

    print(f"Observation dim: {TOTAL_OBS_DIM}")
    print(f"Action dim: {ACTION_DIM}")

    # Test baseline
    print("\n--- Baseline ActorCritic ---")
    baseline = ActorCritic()
    print(f"  Parameters: {count_parameters(baseline):,}")
    obs = torch.randn(4, TOTAL_OBS_DIM)
    action, log_prob, value = baseline.get_action(obs)
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    # Test dual-stream
    print("\n--- DualStream ActorCritic ---")
    dual = DualStreamActorCritic()
    print(f"  Parameters: {count_parameters(dual):,}")
    obs = torch.randn(4, TOTAL_OBS_DIM)
    action, log_prob, value, aux = dual.get_action(obs)
    print(f"  Action shape: {action.shape}")
    print(f"  Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")
    print(f"  Confidence shape: {aux['confidence'].shape}")
    print(f"  Confidence range: [{aux['confidence'].min().item():.3f}, {aux['confidence'].max().item():.3f}]")
    if "return_pred" in aux:
        print(f"  Return pred shape: {aux['return_pred'].shape}")
    if aux["fast_weights"] is not None:
        print(f"  Fast weights shape: {aux['fast_weights'].shape}")

    # Test Hebbian continuity (simulate 3 steps)
    print("\n--- Hebbian Continuity Test ---")
    fw = None
    for step in range(3):
        obs_t = torch.randn(1, TOTAL_OBS_DIM)
        action, _, _, aux = dual.get_action(obs_t, fast_weights=fw)
        fw = aux["fast_weights"]
        fw_norm = fw.abs().mean().item()
        print(f"  Step {step}: action={action[0,:3].tolist()}, fw_norm={fw_norm:.6f}")
