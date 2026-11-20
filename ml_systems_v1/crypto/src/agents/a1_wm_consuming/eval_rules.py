"""
Fixed-Rules Trading Baseline
=============================
Evaluates non-learning trading strategies on the same world model environment
used by the PPO agent. Establishes the performance floor ML must beat.

Usage:
    python src/agents/a1_wm_consuming/eval_rules.py --ensemble
    python src/agents/a1_wm_consuming/eval_rules.py --world-model v1_0
    python src/agents/a1_wm_consuming/eval_rules.py --ensemble --episodes 50
"""

import argparse
import json
import sys
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add agent dir to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DEVICE, NUM_ASSETS, REWARD_HORIZONS, ACTIVE_HORIZONS, INITIAL_CAPITAL,
    EPISODE_LENGTH, PER_ASSET_OBS_DIM, OBS_RETURN_PREDS, MAX_POSITION_FRAC,
    ASSET_LIST, BARS_PER_DAY, PURGE_GAP_BARS,
    AGENT_LOG_DIR,
)
from environment import WorldModelTradingEnv

# Re-use model loading from train_agent
from train_agent import (
    load_world_model, load_ensemble_model, load_data,
    DEFAULT_ENSEMBLE_MODELS, PROJECT_ROOT,
)


# ---------------------------------------------------------------------------
# Fixed-Rule Strategies
# ---------------------------------------------------------------------------

class DoNothing:
    """Hold cash. Absolute floor: zero costs, zero returns."""
    name = "DoNothing"

    def act(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(NUM_ASSETS, dtype=np.float32)


class ReturnProportional:
    """Position proportional to h=1 return prediction.

    The simplest signal-following rule: if the model predicts positive returns,
    go long proportionally. Scale factor converts raw predictions (~0.001-0.01)
    into meaningful position sizes (~0.05-0.20).
    """
    name = "ReturnProportional"

    def __init__(self, scale: float = 10.0, max_pos: float = MAX_POSITION_FRAC):
        self.scale = scale
        self.max_pos = max_pos

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(NUM_ASSETS, dtype=np.float32)
        for i in range(NUM_ASSETS):
            h1_pred = obs[i * PER_ASSET_OBS_DIM]  # offset+0 = h1 return pred
            actions[i] = np.clip(h1_pred * self.scale, -self.max_pos, self.max_pos)
        return actions


class MultiHorizonConsensus:
    """Trade only when all active horizons agree on direction.

    With ACTIVE_HORIZONS=[1,4], requires both h1 and h4 to agree.
    Size by mean absolute prediction magnitude across agreeing horizons.
    """
    name = "MultiHorizonConsensus"

    def __init__(self, scale: float = 10.0,
                 max_pos: float = MAX_POSITION_FRAC):
        self.scale = scale
        self.max_pos = max_pos

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(NUM_ASSETS, dtype=np.float32)
        n_h = OBS_RETURN_PREDS
        for i in range(NUM_ASSETS):
            offset = i * PER_ASSET_OBS_DIM
            preds = obs[offset:offset + n_h]  # active horizon predictions only

            signs = np.sign(preds)
            if np.all(signs > 0):
                magnitude = np.abs(preds).mean()
                actions[i] = np.clip(magnitude * self.scale, 0, self.max_pos)
            elif np.all(signs < 0):
                magnitude = np.abs(preds).mean()
                actions[i] = np.clip(-magnitude * self.scale, -self.max_pos, 0)
            # else: no consensus, stay flat

        return actions


class RegimeGated:
    """Regime-conditional trading.

    Long if bull_prob > threshold, short if bear_prob > threshold, else flat.
    Position sized by h=1 prediction magnitude within the regime direction.
    Tests whether regime probabilities add value beyond return predictions.
    """
    name = "RegimeGated"

    def __init__(self, threshold: float = 0.5, scale: float = 10.0,
                 max_pos: float = MAX_POSITION_FRAC):
        self.threshold = threshold
        self.scale = scale
        self.max_pos = max_pos

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(NUM_ASSETS, dtype=np.float32)
        regime_offset = OBS_RETURN_PREDS  # regime probs start after return predictions
        for i in range(NUM_ASSETS):
            offset = i * PER_ASSET_OBS_DIM
            h1_pred = obs[offset]
            bear_prob = obs[offset + regime_offset]
            bull_prob = obs[offset + regime_offset + 2]

            if bull_prob > self.threshold:
                # Bullish regime: go long, sized by prediction
                actions[i] = np.clip(abs(h1_pred) * self.scale, 0, self.max_pos)
            elif bear_prob > self.threshold:
                # Bearish regime: go short, sized by prediction
                actions[i] = np.clip(-abs(h1_pred) * self.scale, -self.max_pos, 0)
            # else: neutral regime, stay flat

        return actions


class UncertaintyGated:
    """Position proportional to h=1 prediction, but only when model is confident.

    Zeroes positions when posterior uncertainty exceeds threshold.
    Tests whether the uncertainty estimate is calibrated enough to be useful.
    """
    name = "UncertaintyGated"

    def __init__(self, uncertainty_threshold: float = 0.4, scale: float = 10.0,
                 max_pos: float = MAX_POSITION_FRAC):
        self.uncertainty_threshold = uncertainty_threshold
        self.scale = scale
        self.max_pos = max_pos

    def act(self, obs: np.ndarray) -> np.ndarray:
        actions = np.zeros(NUM_ASSETS, dtype=np.float32)
        uncertainty_offset = OBS_RETURN_PREDS + 3  # after return preds + 3 regime probs
        for i in range(NUM_ASSETS):
            offset = i * PER_ASSET_OBS_DIM
            h1_pred = obs[offset]
            uncertainty = obs[offset + uncertainty_offset]

            if uncertainty < self.uncertainty_threshold:
                actions[i] = np.clip(h1_pred * self.scale, -self.max_pos, self.max_pos)
            # else: too uncertain, stay flat

        return actions


class TopNMarketNeutral:
    """Long top-N assets, short bottom-N, flat rest.

    Ranks assets by h=1 return prediction. Equal-weight positions.
    Market-neutral by construction (dollar-neutral if N_long == N_short).
    Tests relative value signal (cross-sectional ranking).
    """
    name = "TopNMarketNeutral"

    def __init__(self, n_top: int = 3, pos_size: float = 0.10):
        self.n_top = n_top
        self.pos_size = pos_size

    def act(self, obs: np.ndarray) -> np.ndarray:
        # Extract h=1 predictions for all assets
        h1_preds = np.array([obs[i * PER_ASSET_OBS_DIM] for i in range(NUM_ASSETS)])
        ranked = np.argsort(h1_preds)

        actions = np.zeros(NUM_ASSETS, dtype=np.float32)
        # Short bottom-N
        for idx in ranked[:self.n_top]:
            actions[idx] = -self.pos_size
        # Long top-N
        for idx in ranked[-self.n_top:]:
            actions[idx] = self.pos_size

        return actions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(strategy, env, n_episodes: int, base_seed: int = 42) -> dict:
    """Run a strategy for N episodes and collect financial metrics.

    Returns dict with: sharpe, max_drawdown, win_rate, mean_turnover,
    mean_cost, mean_final_value, mean_return, std_return
    """
    episode_returns = []
    episode_costs = []
    episode_values = []
    episode_sharpes = []
    episode_max_dds = []
    episode_turnovers = []

    for ep in range(n_episodes):
        obs = env.reset(seed=base_seed + ep)
        done = False
        total_reward = 0.0
        total_cost = 0.0
        n_steps = 0

        step_returns = []
        portfolio_values = [env.portfolio_value]
        peak_value = env.portfolio_value
        max_dd = 0.0
        total_turnover = 0.0
        prev_positions = np.zeros(NUM_ASSETS, dtype=np.float32)

        while not done:
            action = strategy.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            total_cost += info.get("txn_cost", 0) + info.get("funding_cost", 0)
            n_steps += 1

            pv = info.get("portfolio_value", env.portfolio_value)
            portfolio_values.append(pv)
            if len(portfolio_values) >= 2 and portfolio_values[-2] > 0:
                step_returns.append(
                    (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                )
            peak_value = max(peak_value, pv)
            if peak_value > 0:
                dd = (peak_value - pv) / peak_value
                max_dd = max(max_dd, dd)

            positions = info.get("positions", prev_positions)
            total_turnover += np.abs(positions - prev_positions).sum()
            prev_positions = positions.copy()

        episode_returns.append(total_reward)
        episode_costs.append(total_cost)
        episode_values.append(info.get("portfolio_value", INITIAL_CAPITAL))
        episode_max_dds.append(max_dd)
        episode_turnovers.append(total_turnover / max(n_steps, 1))

        if len(step_returns) > 1:
            sr = np.array(step_returns)
            ep_sharpe = (sr.mean() / (sr.std() + 1e-8)) * np.sqrt(252 * BARS_PER_DAY)
            episode_sharpes.append(ep_sharpe)

    return {
        "sharpe": float(np.mean(episode_sharpes)) if episode_sharpes else 0.0,
        "max_drawdown": float(np.max(episode_max_dds)) if episode_max_dds else 0.0,
        "win_rate": float(np.mean([1.0 if v > INITIAL_CAPITAL else 0.0
                                    for v in episode_values])),
        "mean_turnover": float(np.mean(episode_turnovers)),
        "mean_cost": float(np.mean(episode_costs)),
        "mean_final_value": float(np.mean(episode_values)),
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
    }


def print_results_table(results: dict, header: str):
    """Print a formatted comparison table."""
    print(f"\n  {header}")
    print(f"  {'-' * 90}")
    print(f"  {'Strategy':<25s} {'Sharpe':>7s} {'MaxDD':>7s} {'WinRate':>8s} "
          f"{'Turnover':>9s} {'FinalVal':>10s} {'Costs':>8s}")
    print(f"  {'-' * 90}")

    for name, metrics in results.items():
        sharpe = metrics["sharpe"]
        maxdd = metrics["max_drawdown"] * 100
        winrate = metrics["win_rate"] * 100
        turnover = metrics["mean_turnover"]
        final_val = metrics["mean_final_value"]
        costs = metrics["mean_cost"]

        print(f"  {name:<25s} {sharpe:>7.3f} {maxdd:>6.1f}% {winrate:>6.1f}% "
              f"{turnover:>9.4f} {final_val:>10.1f} {costs:>8.2f}")

    print(f"  {'-' * 90}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-rule trading strategies on frozen world model"
    )
    parser.add_argument("--world-model", type=str, default="v1_0",
                        help="World model variant (v1_0, v1_1, ..., v9)")
    parser.add_argument("--features", type=int, choices=[13, 18, 19], default=13,
                        help="Number of features")
    parser.add_argument("--revin", action="store_true",
                        help="Enable RevIN (off by default)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Episodes per strategy per split (default: 20)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use V1.E cross-model ensemble")
    parser.add_argument("--ensemble-models", type=str, default=None,
                        help="Comma-separated model keys for ensemble "
                             f"(default: {','.join(DEFAULT_ENSEMBLE_MODELS)})")
    args = parser.parse_args()

    use_ensemble = args.ensemble
    variant = args.world_model
    model_label = "V1.E ensemble" if use_ensemble else variant

    print("=" * 70)
    print(f"  FIXED-RULES BASELINE")
    print("=" * 70)
    print(f"  World Model:  {model_label}")
    if not use_ensemble:
        print(f"  Features:     {args.features}")
    print(f"  Episodes:     {args.episodes} per strategy per split")
    print(f"  Device:       {DEVICE}")
    print(f"  Seed:         {args.seed}")
    print()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load world model
    if use_ensemble:
        ensemble_keys = None
        if args.ensemble_models:
            ensemble_keys = [k.strip() for k in args.ensemble_models.split(",")]
        model, revin, feature_list, load_data_fn = load_ensemble_model(ensemble_keys)
    else:
        model, revin, feature_list, load_data_fn = load_world_model(
            variant, args.features, args.revin
        )

    # Load data
    segments = load_data(load_data_fn, feature_list)

    # Create environments
    print("\n  Creating IS (train) environment...")
    env_train = WorldModelTradingEnv(
        world_model=model,
        data_segments=segments,
        feature_list=feature_list,
        mode="train",
        revin=revin,
        episode_length=EPISODE_LENGTH,
        seed=args.seed,
    )

    print("  Creating OOS (val) environment...")
    env_val = WorldModelTradingEnv(
        world_model=model,
        data_segments=segments,
        feature_list=feature_list,
        mode="val",
        revin=revin,
        episode_length=EPISODE_LENGTH,
        seed=args.seed + 1000,
    )

    print(f"  Valid IS assets: {len(env_train.valid_assets)}")
    print(f"  Valid OOS assets: {len(env_val.valid_assets)}")

    # Define strategies
    strategies = [
        DoNothing(),
        ReturnProportional(scale=10.0),
        MultiHorizonConsensus(scale=10.0),
        RegimeGated(threshold=0.5, scale=10.0),
        UncertaintyGated(uncertainty_threshold=0.4, scale=10.0),
        TopNMarketNeutral(n_top=3, pos_size=0.10),
    ]

    # Evaluate all strategies
    is_results = {}
    oos_results = {}

    start_time = time.time()

    for strategy in strategies:
        name = strategy.name
        print(f"\n  Evaluating: {name}...")

        print(f"    IS ({args.episodes} episodes)...", end="", flush=True)
        is_metrics = evaluate_strategy(
            strategy, env_train, n_episodes=args.episodes, base_seed=args.seed
        )
        is_results[name] = is_metrics
        print(f" Sharpe={is_metrics['sharpe']:.3f}")

        print(f"    OOS ({args.episodes} episodes)...", end="", flush=True)
        oos_metrics = evaluate_strategy(
            strategy, env_val, n_episodes=args.episodes, base_seed=args.seed + 5000
        )
        oos_results[name] = oos_metrics
        print(f" Sharpe={oos_metrics['sharpe']:.3f}")

    elapsed = time.time() - start_time

    # Print results
    print_results_table(is_results,
                        f"IN-SAMPLE (Train, 90% of data, {args.episodes} episodes)")
    print_results_table(oos_results,
                        f"OUT-OF-SAMPLE (Val, last 10%, purge={PURGE_GAP_BARS} bars, "
                        f"{args.episodes} episodes)")

    # OOS/IS Sharpe retention
    print(f"\n  OOS/IS Sharpe Retention:")
    for name in is_results:
        is_sharpe = is_results[name]["sharpe"]
        oos_sharpe = oos_results[name]["sharpe"]
        if abs(is_sharpe) > 0.01:
            ratio = oos_sharpe / is_sharpe
            tag = "[PASS]" if ratio > 0.5 else "[WARN]" if ratio > 0.0 else "[FAIL]"
            print(f"    {name:<25s} {ratio:>6.3f} {tag}")
        else:
            print(f"    {name:<25s}    N/A (IS Sharpe ~0)")

    # Save JSON
    if use_ensemble:
        tag = "v1e_ensemble"
    else:
        tag = f"{variant}_f{args.features}"

    results = {
        "model": model_label,
        "variant": variant if not use_ensemble else "ensemble",
        "features": args.features,
        "episodes_per_split": args.episodes,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "in_sample": is_results,
        "out_of_sample": oos_results,
    }

    results_path = AGENT_LOG_DIR / f"rules_baseline_{tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved: {results_path.name}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
