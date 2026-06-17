"""
World Model Trading Environment
================================
Gym-like environment that uses a frozen world model as the market representation.

Two modes:
  REPLAY:  Steps through real historical data. World model provides features
           (latent states, return predictions, regime probs). Returns are REAL.
  DREAM:   World model imagines forward from a real starting point.
           Returns are PREDICTED (from the world model). For exploration.

For agent training, start with REPLAY (ground truth returns).
Add DREAM augmentation for exploration once replay-trained agent is stable.
"""

__class_tag__ = "A1"  # WM-consuming agent env: REPLAY (real returns) / DREAM (predicted) seam (doc SS1.8)

import sys
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from config import (
    DEVICE, NUM_ASSETS, REWARD_HORIZONS, ACTIVE_HORIZONS, DATA_DIR,
    INITIAL_CAPITAL, MAX_POSITION_FRAC, MAX_GROSS_EXPOSURE,
    TAKER_FEE_BPS, SLIPPAGE_BPS, FUNDING_RATE_HOURLY, BARS_PER_HOUR,
    SPOT_FEE_BPS, SPOT_SLIPPAGE_BPS,
    PERP_FEE_BPS, PERP_SLIPPAGE_BPS, PERP_FUNDING_RATE_HOURLY,
    DD_REDUCE_THRESHOLD, DD_KILL_THRESHOLD,
    EPISODE_LENGTH, WARMUP_BARS, MIN_EPISODE_BARS,
    PER_ASSET_OBS_DIM, GLOBAL_OBS_DIM, TOTAL_OBS_DIM, ACTION_DIM,
    ASSET_LIST, ASSET_TO_IDX,
    VAL_FRACTION, PURGE_GAP_BARS,
    PROJECT_ROOT,
    AUGMENT_PROB, AUGMENT_SCENARIOS,
    DECISION_INTERVAL,
)
from rewards import RewardCalculator


# ---------------------------------------------------------------------------
# Stress Augmentation: Synthetic Scenarios
# ---------------------------------------------------------------------------

class StressAugmentor:
    """
    Injects synthetic stress scenarios into episode returns only.

    Key invariant: ONLY target_return_* arrays are modified. World model
    feature arrays are NEVER touched. This creates "prediction failure"
    scenarios where the WM sees normal historical features but realized
    returns diverge from predictions.

    Scenarios:
        flash_crash  -- sudden large negative returns for N bars
        vol_spike    -- realized vol 2-4x expected for N bars
        squeeze      -- momentum followed by sharp reversal
        cost_shock   -- effective transaction costs 2-5x normal
    """

    def __init__(self, augment_prob: float = AUGMENT_PROB,
                 scenarios: dict | None = None):
        self.augment_prob = augment_prob
        self.scenarios = scenarios if scenarios is not None else AUGMENT_SCENARIOS

        # Build normalized sampling weights
        names = list(self.scenarios.keys())
        weights = np.array([self.scenarios[n]["prob"] for n in names], dtype=np.float64)
        weights /= weights.sum()
        self._scenario_names = names
        self._scenario_weights = weights

        # Episode state
        self.active_scenario: str | None = None
        self.cost_multiplier: float = 1.0

        # Saved originals: {asset_idx: {key: original_array}}
        self._originals: dict = {}

    def plan_episode(
        self,
        asset_data: dict,
        bar_starts: dict,
        episode_length: int,
        rng: np.random.Generator,
    ) -> None:
        """
        Decide whether to augment this episode and apply mutations.

        Called AFTER bar_starts are chosen in reset() but BEFORE
        _compute_world_model_features() so the WM sees clean features.
        """
        # Defensive: restore any leftover mutations
        self._restore_internal(asset_data)
        self.active_scenario = None
        self.cost_multiplier = 1.0
        self._originals = {}

        # Roll the dice
        if rng.random() >= self.augment_prob:
            return

        # Choose scenario
        scenario_name = rng.choice(self._scenario_names, p=self._scenario_weights)
        self.active_scenario = scenario_name
        cfg = self.scenarios[scenario_name]

        if scenario_name == "cost_shock":
            # cost_shock modifies costs, not return arrays
            self.cost_multiplier = float(
                rng.uniform(cfg["min_cost_mult"], cfg["max_cost_mult"])
            )
            return

        # Choose 1-3 affected assets
        affected = list(asset_data.keys())
        n_affected = int(rng.integers(1, min(4, len(affected) + 1)))
        rng.shuffle(affected)
        affected = affected[:n_affected]

        for asset_idx in affected:
            data = asset_data[asset_idx]
            start = bar_starts.get(asset_idx, 0)
            n_bars_available = data["n_bars"] - start

            if n_bars_available < 5:
                continue

            # Save originals before mutation
            if asset_idx not in self._originals:
                self._originals[asset_idx] = {
                    "target_return_1": data["target_return_1"].copy(),
                    "target_return_4": data["target_return_4"].copy(),
                    "target_return_16": data["target_return_16"].copy(),
                    "target_return_64": data["target_return_64"].copy(),
                }

            # Duration and offset within episode
            min_bars = max(1, cfg.get("min_bars", 5))
            max_bars = min(cfg.get("max_bars", 20), episode_length, n_bars_available - 1)
            if max_bars < min_bars:
                max_bars = min_bars
            duration = int(rng.integers(min_bars, max_bars + 1))

            max_offset = max(1, int(episode_length * 0.67) - duration)
            offset = int(rng.integers(0, max(1, max_offset)))
            shock_start = start + offset
            shock_end = min(shock_start + duration, data["n_bars"])

            self._apply_scenario(scenario_name, cfg, data, shock_start, shock_end, rng)

    def restore(self, asset_data: dict) -> None:
        """Restore original return arrays after episode ends."""
        self._restore_internal(asset_data)

    def get_info(self) -> dict:
        """Return scenario info for step() info dict."""
        return {
            "scenario_type": self.active_scenario,
            "cost_multiplier": self.cost_multiplier,
        }

    def _restore_internal(self, asset_data: dict) -> None:
        """Copy saved originals back into asset_data."""
        for asset_idx, saved in self._originals.items():
            if asset_idx in asset_data:
                for key, arr in saved.items():
                    asset_data[asset_idx][key] = arr
        self._originals = {}

    def _apply_scenario(
        self,
        name: str,
        cfg: dict,
        data: dict,
        start: int,
        end: int,
        rng: np.random.Generator,
    ) -> None:
        """Apply a scenario to a slice of return arrays."""
        r = data["target_return_1"]

        if name == "flash_crash":
            shock = rng.uniform(cfg["min_shock"], cfg["max_shock"])
            r[start:end] = shock

        elif name == "vol_spike":
            mult = rng.uniform(cfg["min_mult"], cfg["max_mult"])
            r[start:end] = r[start:end] * mult

        elif name == "squeeze":
            mom_bars = min(cfg.get("momentum_bars", 10), (end - start) // 2)
            rev_mult = cfg.get("reversal_mult", -2.0)
            if mom_bars < 1:
                return
            # Phase 1: positive momentum
            r[start:start + mom_bars] = np.abs(r[start:start + mom_bars]) + 0.003
            # Phase 2: sharp reversal
            rev_start = start + mom_bars
            if rev_start < end:
                mom_mean = np.abs(r[start:start + mom_bars]).mean()
                r[rev_start:end] = rev_mult * mom_mean


class WorldModelTradingEnv:
    """
    Trading environment driven by a frozen world model.

    The world model processes real observation sequences and produces:
      - Return predictions (4 horizons x 10 assets)
      - Regime probabilities (3 classes x 10 assets)
      - Posterior entropy (1 x 10 assets) as uncertainty measure

    The agent observes these + its portfolio state and outputs position targets.

    Args:
        world_model: Frozen TransformerWorldModel (eval mode, no grad)
        data_segments: List of asset data dicts from load_full_data()
        feature_list: List of feature column names
        mode: "train" or "val" (determines which portion of data is used)
        revin: Optional RevIN module (loaded from checkpoint)
        episode_length: Number of trading steps per episode
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        world_model,
        data_segments: list[dict],
        feature_list: list[str],
        mode: str = "train",
        revin=None,
        episode_length: int = EPISODE_LENGTH,
        seed: int = 42,
        enable_augmentation: bool = False,
        spot_mode: bool = True,
        long_only: bool = True,
        decision_interval: int = DECISION_INTERVAL,
    ):
        self.model = world_model
        self.model.eval()
        self.revin = revin
        if self.revin is not None:
            self.revin.eval()

        self.feature_list = feature_list
        self.n_features = len(feature_list)
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        # SPOT vs Perp cost model
        self.spot_mode = spot_mode
        self.long_only = long_only
        self.decision_interval = max(1, decision_interval)

        # Set cost parameters based on mode
        if spot_mode:
            self._fee_bps = SPOT_FEE_BPS
            self._slippage_bps = SPOT_SLIPPAGE_BPS
            self._funding_hourly = 0.0  # No funding in SPOT
        else:
            self._fee_bps = PERP_FEE_BPS
            self._slippage_bps = PERP_SLIPPAGE_BPS
            self._funding_hourly = PERP_FUNDING_RATE_HOURLY

        # Organize data by asset
        self._prepare_data(data_segments, mode)

        # Reward calculator
        self.reward_calc = RewardCalculator()

        # Stress augmentation (training only)
        self.augmentor = StressAugmentor() if enable_augmentation else None
        self.scenario_counts: dict = {}
        self.scenario_rewards: dict = {}

        # State
        self.positions = np.zeros(NUM_ASSETS, dtype=np.float32)
        self.cash = INITIAL_CAPITAL
        self.portfolio_value = INITIAL_CAPITAL
        self.peak_value = INITIAL_CAPITAL
        self.dd_reduced = False  # True after DD_REDUCE_THRESHOLD hit
        self.step_count = 0
        self._episode_reward = 0.0
        self.current_segments = {}  # per-asset current data slices
        self.current_wm_features = {}  # cached world model outputs per asset

    def _prepare_data(self, data_segments: list[dict], mode: str):
        """Split data into train/val and organize by asset."""
        self.asset_data = {}  # asset_idx -> {features, returns_1, returns_4, ...}

        for seg in data_segments:
            asset_idx = seg["asset_idx"]
            n_bars = len(seg["features"])

            # Train/val split
            split_point = int(n_bars * (1.0 - VAL_FRACTION))

            if mode == "train":
                start, end = 0, split_point
            elif mode == "val":
                start = split_point + PURGE_GAP_BARS
                end = n_bars
                if start >= end:
                    start = split_point  # Fallback if not enough data
            else:
                raise ValueError(f"mode must be 'train' or 'val', got {mode}")

            self.asset_data[asset_idx] = {
                "features": seg["features"][start:end],
                "target_return_1": seg["target_return_1"][start:end],
                "target_return_4": seg["target_return_4"][start:end],
                "target_return_16": seg["target_return_16"][start:end],
                "target_return_64": seg["target_return_64"][start:end],
                "n_bars": end - start,
            }

        # Valid assets: those with enough bars
        self.valid_assets = [
            idx for idx, data in self.asset_data.items()
            if data["n_bars"] >= MIN_EPISODE_BARS
        ]

        if not self.valid_assets:
            raise ValueError(
                f"No assets have enough bars for episodes "
                f"(need {MIN_EPISODE_BARS}, mode={mode})"
            )

    @torch.no_grad()
    def _precompute_episode_features(
        self,
        asset_idx: int,
        start_bar: int,
    ) -> dict:
        """
        Precompute world model features for the ENTIRE episode in one forward pass.

        Instead of calling the model 256 times (once per step), pass the full
        window [start_bar - WARMUP_BARS, start_bar + episode_length] through
        the model once and cache all timestep features.

        Returns dict with arrays indexed by position in the full window.
        Episode steps start at index 'warmup_offset'.
        """
        data = self.asset_data[asset_idx]

        # Full window: WARMUP context + all episode steps
        window_start = max(0, start_bar - WARMUP_BARS)
        window_end = min(start_bar + self.episode_length + 1, data["n_bars"])

        features = data["features"][window_start:window_end]  # [T_total, n_features]
        obs_seq = torch.tensor(features, dtype=torch.float32, device=DEVICE).unsqueeze(0)

        if self.revin is not None:
            obs_seq = self.revin(obs_seq, mode="norm")

        asset_id = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

        # ONE forward pass for the entire episode
        h_seq, z_post, return_preds = self.model.encode_sequence(obs_seq, asset_id)

        T = h_seq.shape[1]
        warmup_offset = start_bar - window_start  # episode bars start here

        # Return predictions: {horizon: [T] numpy}
        ret_preds_np = {}
        for h in REWARD_HORIZONS:
            ret_preds_np[h] = return_preds[h][0].cpu().numpy()  # [T]

        # Regime probabilities: [T, 3]
        # Ensemble caches full outputs in _last_outputs; single models use regime_head directly
        if hasattr(self.model, '_last_outputs') and self.model._last_outputs is not None:
            regime_logits = self.model._last_outputs["regime_logits"]  # [B, T, 3]
        else:
            feat = torch.cat([h_seq[0], z_post[0]], dim=-1)  # [T, d_model+flat_dim]
            regime_logits = self.model.regime_head(feat).unsqueeze(0)  # [1, T, 3]
        regime_probs = torch.softmax(regime_logits[0], dim=-1).cpu().numpy()  # [T, 3]

        # Uncertainty: posterior entropy for all timesteps
        uncertainties = np.full(T, 0.5, dtype=np.float32)
        if hasattr(self.model, "posterior_head") and hasattr(self.model, "latent_dim"):
            try:
                if hasattr(self.model, '_last_outputs') and self.model._last_outputs is not None:
                    post_logits = self.model._last_outputs["post_logits"][0]  # [T, flat_dim]
                else:
                    post_input = torch.cat([h_seq[0], obs_seq[0, :T]], dim=-1)
                    post_logits = self.model.posterior_head(post_input)  # [T, flat_dim]

                ld = self.model.latent_dim
                cl = self.model.classes
                post_probs = torch.softmax(post_logits.view(T, ld, cl), dim=-1)
                entropy = -(post_probs * (post_probs + 1e-8).log()).sum(dim=-1).mean(dim=-1)  # [T]
                max_entropy = np.log(cl)
                uncertainties = (entropy / max_entropy).cpu().numpy()
            except Exception:
                pass

        return {
            "ret_preds": ret_preds_np,       # {h: [T] array}
            "regime_probs": regime_probs,     # [T, 3]
            "uncertainties": uncertainties,   # [T]
            "warmup_offset": warmup_offset,   # episode starts at this index
        }

    def _get_cached_wm_features(self, asset_idx: int, step: int) -> dict:
        """Get precomputed WM features for a given episode step."""
        cache = self._episode_cache.get(asset_idx)
        if cache is None:
            # Fallback: zero features
            return {
                "return_preds": {h: 0.0 for h in REWARD_HORIZONS},
                "regime_probs": np.array([0.33, 0.34, 0.33], dtype=np.float32),
                "uncertainty": 0.5,
            }

        idx = cache["warmup_offset"] + step
        if idx < 0 or idx >= len(cache["uncertainties"]):
            idx = min(max(idx, 0), len(cache["uncertainties"]) - 1)

        return {
            "return_preds": {h: float(cache["ret_preds"][h][idx]) for h in REWARD_HORIZONS},
            "regime_probs": cache["regime_probs"][idx],
            "uncertainty": float(cache["uncertainties"][idx]),
        }

    def _build_observation(self) -> np.ndarray:
        """Build the full observation vector from cached world model features + portfolio state."""
        obs = np.zeros(TOTAL_OBS_DIM, dtype=np.float32)

        for i, asset_idx in enumerate(range(NUM_ASSETS)):
            offset = i * PER_ASSET_OBS_DIM

            if asset_idx in self.current_wm_features:
                wm = self.current_wm_features[asset_idx]

                # Return predictions (active horizons only)
                n_h = len(ACTIVE_HORIZONS)
                for j, h in enumerate(ACTIVE_HORIZONS):
                    obs[offset + j] = wm["return_preds"][h]

                # Regime probabilities (3 values)
                obs[offset + n_h:offset + n_h + 3] = wm["regime_probs"]

                # Uncertainty (1 value)
                obs[offset + n_h + 3] = wm["uncertainty"]

            # Portfolio state for this asset (fixed offsets from PER_ASSET_OBS_DIM layout)
            obs[offset + PER_ASSET_OBS_DIM - 2] = self.positions[asset_idx]
            obs[offset + PER_ASSET_OBS_DIM - 1] = self._unrealized_pnl(asset_idx)

        # Global: cash as fraction of initial capital
        obs[-1] = self.cash / INITIAL_CAPITAL

        return obs

    def _unrealized_pnl(self, asset_idx: int) -> float:
        """Unrealized PnL proxy for observation vector.

        Returns the LAST bar's realized PnL (already happened) as a proxy,
        NOT the current bar's return (which would leak future information).

        The position itself is already observed at obs[offset+6], and the
        world model return predictions provide forward-looking signal.
        This channel adds mark-to-market context without look-ahead.
        """
        if self.positions[asset_idx] == 0.0:
            return 0.0
        data = self.asset_data.get(asset_idx)
        if data is None:
            return 0.0
        bar_idx = self.current_bar_indices.get(asset_idx, 0)
        # Use PREVIOUS bar's return (already realized), not current bar's
        prev_idx = bar_idx - 1
        if prev_idx < 0 or prev_idx >= data["n_bars"]:
            return 0.0
        realized_ret = data["target_return_1"][prev_idx]
        return self.positions[asset_idx] * realized_ret * self.portfolio_value

    def _execute_trades(self, target_positions: np.ndarray) -> tuple[float, float]:
        """
        Execute trades to move from current positions to target positions.

        Returns (transaction_cost, funding_cost)
        """
        position_changes = target_positions - self.positions
        abs_changes = np.abs(position_changes)

        # Transaction cost: fee + slippage on traded notional
        total_fee_bps = self._fee_bps + self._slippage_bps

        # Apply cost multiplier if cost_shock scenario is active
        cost_mult = 1.0
        if self.augmentor is not None:
            cost_mult = self.augmentor.cost_multiplier

        notional_traded = abs_changes.sum() * self.portfolio_value
        transaction_cost = notional_traded * (total_fee_bps * cost_mult) / 10_000

        # Funding cost: proportional to absolute position size
        # SPOT mode: zero funding. Perp mode: charged per bar on open positions
        funding_cost = 0.0
        if self._funding_hourly > 0:
            abs_positions = np.abs(target_positions)
            funding_notional = abs_positions.sum() * self.portfolio_value
            funding_cost = funding_notional * self._funding_hourly / BARS_PER_HOUR

        self.positions = target_positions.copy()
        return transaction_cost, funding_cost

    def _compute_step_pnl(self, asset_idx: int) -> float:
        """Compute actual PnL from real returns for one asset."""
        data = self.asset_data.get(asset_idx)
        if data is None:
            return 0.0

        bar_idx = self.current_bar_indices.get(asset_idx, 0)
        if bar_idx >= data["n_bars"]:
            return 0.0

        # Use 1-bar actual return
        actual_return = data["target_return_1"][bar_idx]
        position_value = self.positions[asset_idx] * self.portfolio_value
        return position_value * actual_return

    def reset(self, seed: int | None = None) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Samples a random starting point for each asset (within the valid range)
        and computes initial world model features.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Restore any augmentation from previous episode FIRST
        if self.augmentor is not None:
            self.augmentor.restore(self.asset_data)

        # Reset portfolio
        self.positions = np.zeros(NUM_ASSETS, dtype=np.float32)
        self.cash = INITIAL_CAPITAL
        self.portfolio_value = INITIAL_CAPITAL
        self.peak_value = INITIAL_CAPITAL
        self.dd_reduced = False
        self.step_count = 0
        self._episode_reward = 0.0
        self.reward_calc.reset(INITIAL_CAPITAL)

        # Sample starting points per asset
        self.current_bar_indices = {}
        for asset_idx in range(NUM_ASSETS):
            if asset_idx not in self.asset_data:
                continue
            n_bars = self.asset_data[asset_idx]["n_bars"]
            max_start = n_bars - self.episode_length - 1
            if max_start <= WARMUP_BARS:
                start = WARMUP_BARS
            else:
                start = self.rng.integers(WARMUP_BARS, max_start)
            self.current_bar_indices[asset_idx] = start

        # Plan augmentation for this episode (mutates return arrays only)
        if self.augmentor is not None:
            self.augmentor.plan_episode(
                self.asset_data,
                self.current_bar_indices,
                self.episode_length,
                self.rng,
            )

        # Precompute world model features for ENTIRE episode (one forward pass per asset)
        self._episode_cache = {}
        for asset_idx in self.current_bar_indices:
            bar_idx = self.current_bar_indices[asset_idx]
            self._episode_cache[asset_idx] = self._precompute_episode_features(
                asset_idx, bar_idx
            )

        # Set initial WM features from cache (step 0)
        self.current_wm_features = {}
        for asset_idx in self.current_bar_indices:
            self.current_wm_features[asset_idx] = self._get_cached_wm_features(
                asset_idx, step=0
            )

        return self._build_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Take a trading action and advance one bar.

        Args:
            action: Target positions per asset, shape [NUM_ASSETS], values in [-1, +1]

        Returns:
            observation: Next observation vector
            reward: Scalar reward
            done: Whether the episode is over
            info: Dict with diagnostic information
        """
        # Clip and scale positions
        target_positions = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Long-only: positions in [0, MAX_POSITION_FRAC]
        if self.long_only:
            target_positions = np.clip(target_positions, 0.0, MAX_POSITION_FRAC)
        else:
            target_positions = np.clip(target_positions, -MAX_POSITION_FRAC, MAX_POSITION_FRAC)

        # Enforce portfolio-level gross exposure cap
        gross_exposure = np.abs(target_positions).sum()
        if gross_exposure > MAX_GROSS_EXPOSURE:
            target_positions *= MAX_GROSS_EXPOSURE / gross_exposure

        # Decision interval: only execute trades on decision steps
        # Agent observes every bar but holds positions between decisions
        if self.decision_interval > 1 and (self.step_count % self.decision_interval != 0):
            target_positions = self.positions.copy()  # Hold current positions

        # Execute trades
        txn_cost, funding_cost = self._execute_trades(target_positions)

        # Advance bars and compute PnL from REAL returns
        total_pnl = 0.0
        for asset_idx in self.current_bar_indices:
            pnl = self._compute_step_pnl(asset_idx)
            total_pnl += pnl
            self.current_bar_indices[asset_idx] += 1

        # Update portfolio value: costs reduce capital, PnL adjusts value
        # Positions are notional (fraction of portfolio), so cash tracks total equity
        self.portfolio_value += total_pnl - txn_cost - funding_cost
        self.cash = self.portfolio_value

        # Compute reward
        reward = self.reward_calc.compute(
            pnl=total_pnl,
            transaction_cost=txn_cost,
            funding_cost=funding_cost,
            portfolio_value=max(self.portfolio_value, 1.0),
        )

        self.step_count += 1

        # Track peak and drawdown
        self.peak_value = max(self.peak_value, self.portfolio_value)
        drawdown = 0.0
        if self.peak_value > 0:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value

        # Check if done
        done = False
        if self.step_count >= self.episode_length:
            done = True

        # Progressive drawdown circuit breaker
        if drawdown >= DD_KILL_THRESHOLD:
            done = True  # 25% drawdown kills episode
        elif drawdown >= DD_REDUCE_THRESHOLD and not self.dd_reduced:
            # 15% drawdown: halve all positions (one-time), with transaction costs
            half_positions = self.positions * 0.5
            abs_changes = np.abs(half_positions - self.positions)
            total_fee_bps = self._fee_bps + self._slippage_bps
            dd_notional = abs_changes.sum() * self.portfolio_value
            dd_txn_cost = dd_notional * total_fee_bps / 10_000
            self.portfolio_value -= dd_txn_cost
            self.cash = self.portfolio_value
            self.positions = half_positions
            self.dd_reduced = True

        # Update world model features from precomputed cache (zero model inference)
        if not done:
            for asset_idx in self.current_bar_indices:
                self.current_wm_features[asset_idx] = self._get_cached_wm_features(
                    asset_idx, step=self.step_count
                )

        obs = self._build_observation()

        info = {
            "pnl": total_pnl,
            "txn_cost": txn_cost,
            "funding_cost": funding_cost,
            "portfolio_value": self.portfolio_value,
            "step": self.step_count,
            "positions": self.positions.copy(),
            "scenario_type": None,
        }

        if self.augmentor is not None:
            info["scenario_type"] = self.augmentor.active_scenario
            info["cost_multiplier"] = self.augmentor.cost_multiplier

        # Track per-scenario rewards
        self._episode_reward += reward
        if done and self.augmentor is not None and self.augmentor.active_scenario is not None:
            sc = self.augmentor.active_scenario
            self.scenario_counts[sc] = self.scenario_counts.get(sc, 0) + 1
            if sc not in self.scenario_rewards:
                self.scenario_rewards[sc] = []
            self.scenario_rewards[sc].append(self._episode_reward)

        return obs, reward, done, info

    def get_scenario_stats(self) -> dict:
        """Return per-scenario episode counts and mean rewards for logging."""
        stats = {}
        for sc_name, count in self.scenario_counts.items():
            rewards = self.scenario_rewards.get(sc_name, [])
            stats[sc_name] = {
                "count": count,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            }
        return stats

    @property
    def observation_dim(self) -> int:
        return TOTAL_OBS_DIM

    @property
    def action_dim(self) -> int:
        return ACTION_DIM
