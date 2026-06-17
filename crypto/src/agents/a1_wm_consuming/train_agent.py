"""
Agent Training Script
=====================
Trains a PPO trading agent using a frozen world model as the market representation.

Usage:
    # Train agent on V1 ensemble (SPOT, long-only, default)
    python src/agents/a1_wm_consuming/train_agent.py --ensemble

    # SPOT with augmentation
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --augment --sav

    # Perpetual futures mode
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --perp

    # Allow short positions (margin mode)
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --margin

    # Multi-bar decision interval (trade every 64 bars)
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --decision-interval 64

    # Train on single world model
    python src/agents/a1_wm_consuming/train_agent.py --world-model v1 --features 13

    # Custom training steps
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --steps 5000000

    # Evaluate only (load existing agent)
    python src/agents/a1_wm_consuming/train_agent.py --ensemble --eval-only
"""

__class_tag__ = "A1"  # WM-consuming agent training entry point (doc SS1.8)

import argparse
import json
import re
import sys
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# Add project paths for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR

while not (PROJECT_ROOT / "data").exists():
    if PROJECT_ROOT.parent == PROJECT_ROOT:
        PROJECT_ROOT = SCRIPT_DIR.parent.parent
        break
    PROJECT_ROOT = PROJECT_ROOT.parent

# Add agent dir and world model dirs to path
sys.path.insert(0, str(SCRIPT_DIR))

from config import (
    DEVICE, DATA_DIR, AGENT_MODEL_DIR, AGENT_LOG_DIR,
    NUM_ASSETS, ASSET_LIST, ASSET_TO_IDX,
    REWARD_HORIZONS, TOTAL_TIMESTEPS, INITIAL_CAPITAL,
    EPISODE_LENGTH, AUGMENT_PROB, PURGE_GAP_BARS,
    BARS_PER_DAY,
)
from environment import WorldModelTradingEnv
from policy import ActorCritic, DualStreamActorCritic, count_parameters
from ppo import PPOTrainer, RunningMeanStd, sav_test, _format_eval

# World model class names per major version (V2-V9 have different class names)
MODEL_CLASS_MAP = {
    "v1": "TransformerWorldModel",
    "v2": "JEPAWorldModel",
    "v3": "WaveNetGRUWorldModel",
    "v4": "MambaWorldModel",
    "v5": "HybridMambaAttentionWorldModel",
    "v6": "CausalJEPAWorldModel",
    "v7": "ViTWorldModel",
    "v8": "NeuralODEWorldModel",
    "v9": "MoEWorldModel",
}

# Default models for V1.E cross-model ensemble (active models only)
# V1.2-V1.5 archived; V1.6 auto-discovered by cross_ensemble.py when trained
DEFAULT_ENSEMBLE_MODELS = ["v1_0", "v1_1_f13", "v1_6"]


def _resolve_major_version(variant: str) -> str:
    """Parse major version from variant string.

    v1 -> v1, v1_1 -> v1, v10 -> v10, v10_meta -> v10
    """
    m = re.match(r"v(\d+)", variant)
    if m:
        return f"v{m.group(1)}"
    raise ValueError(f"Cannot parse major version from: {variant}")


def setup_logging(log_dir: Path, prefix: str) -> Path:
    """Setup log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{prefix}_{timestamp}.log"
    return log_path


def load_world_model(variant: str, features: int, use_revin: bool):
    """
    Load a frozen world model and its data loading utilities.

    Supports all V1-V9 architectures with dynamic class discovery.

    Returns:
        model: Frozen world model (eval mode)
        revin: RevIN module (or None)
        feature_list: List of feature column names
        load_data_fn: Function to load data
    """
    # Determine source directory (grouped by major version)
    major_version = _resolve_major_version(variant)
    variant_dir = PROJECT_ROOT / "src" / major_version / f"{variant}_training"
    if not variant_dir.exists():
        raise FileNotFoundError(f"World model source not found: {variant_dir}")

    # Add to path for imports
    group_dir = variant_dir.parent
    sys.path.insert(0, str(group_dir))
    sys.path.insert(0, str(variant_dir))

    # Import settings and model from the variant (clear stale modules first)
    import importlib
    for mod_name in ["settings", "world_model", "components"]:
        sys.modules.pop(mod_name, None)
    settings = importlib.import_module("settings")
    world_model_mod = importlib.import_module("world_model")

    # Determine feature config
    if hasattr(settings, "get_feature_config"):
        feature_list, input_dim, base_dim = settings.get_feature_config(features)
    else:
        feature_list = settings.FEATURE_LIST
        input_dim = settings.INPUT_DIM
        base_dim = input_dim

    # Find checkpoint directory
    if hasattr(settings, "BASE_MODEL_DIR"):
        model_dir = Path(settings.BASE_MODEL_DIR)
    else:
        model_dir = Path(settings.MODEL_DIR)

    # Build checkpoint prefix
    if variant == "v1_0":
        ckpt_prefix = "v1_0_f13"
    elif variant == "v1":
        # Legacy alias
        ckpt_prefix = "v1_0_f13"
    else:
        feat_tag = f"f{features}"
        revin_tag = "_revin" if use_revin else ""
        ckpt_prefix = f"{variant}_{feat_tag}{revin_tag}"

    ckpt_path = model_dir / f"{ckpt_prefix}_wm_best_ema.pt"
    if not ckpt_path.exists():
        # Try MODEL_DIR without base/ subdir
        ckpt_path = Path(settings.MODEL_DIR) / f"{ckpt_prefix}_wm_best_ema.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"  Searched dirs: {model_dir}, {settings.MODEL_DIR}\n"
            f"  Prefix: {ckpt_prefix}"
        )

    print(f"  Loading world model: {ckpt_path.name}")
    print(f"  Features: {features} ({len(feature_list)} in list)")
    print(f"  RevIN: {'enabled' if use_revin else 'disabled'}")

    # Discover model class dynamically (V2-V9 have different class names)
    class_name = MODEL_CLASS_MAP.get(major_version)
    if class_name and hasattr(world_model_mod, class_name):
        ModelClass = getattr(world_model_mod, class_name)
    else:
        # Fallback: find first nn.Module subclass with encode_sequence
        import torch.nn as nn
        ModelClass = None
        for attr_name in dir(world_model_mod):
            obj = getattr(world_model_mod, attr_name)
            if (isinstance(obj, type) and issubclass(obj, nn.Module)
                    and obj is not nn.Module
                    and hasattr(obj, "encode_sequence")):
                ModelClass = obj
                break
        if ModelClass is None:
            raise RuntimeError(
                f"No world model class found in {variant_dir}/world_model.py\n"
                f"  Tried: {class_name}, then searched for nn.Module with encode_sequence"
            )
    print(f"  Model class: {ModelClass.__name__}")

    # Instantiate model
    if variant in ("v1", "v1_0"):
        model = ModelClass(input_dim=input_dim).to(DEVICE)
    else:
        model = ModelClass(input_dim=input_dim, base_dim=base_dim).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Load RevIN if applicable
    revin = None
    if use_revin:
        try:
            revin_path = PROJECT_ROOT / "src" / "revin.py"
            if revin_path.exists():
                sys.path.insert(0, str(revin_path.parent))
                from revin import RevIN
                revin = RevIN(num_features=input_dim).to(DEVICE)
                if isinstance(ckpt, dict) and "revin_state_dict" in ckpt:
                    revin.load_state_dict(ckpt["revin_state_dict"])
                    print(f"  RevIN loaded from checkpoint")
                revin.eval()
        except ImportError:
            print(f"  [WARN] RevIN requested but module not found, continuing without")

    # Data loader
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from anti_fragile import load_full_data

    print(f"  World model loaded: {count_parameters(model):,} params (frozen)")

    return model, revin, feature_list, load_full_data


def load_ensemble_model(model_keys: list = None):
    """
    Load V1.E CrossModelEnsemble for agent training.

    The ensemble provides the same interface as individual models:
      model.encode_sequence(obs_seq, asset_id) -> (h_seq, z_post, return_preds)
      model.regime_head(feat) -> regime_logits
      model.posterior_head(input) -> post_logits
      model.latent_dim, model.classes

    Returns:
        model: CrossModelEnsemble (frozen, eval mode)
        revin: None (ensemble models handle their own normalization)
        feature_list: List of feature column names (max across ensemble)
        load_data_fn: Function to load data
    """
    # Import CrossModelEnsemble
    ensemble_dir = PROJECT_ROOT / "src" / "wm" / "v1"
    sys.path.insert(0, str(ensemble_dir))

    from cross_ensemble import CrossModelEnsemble

    if model_keys is None:
        model_keys = DEFAULT_ENSEMBLE_MODELS

    print(f"  Loading V1.E ensemble with models: {model_keys}")
    model = CrossModelEnsemble(model_keys=model_keys, use_gating=False)

    # Use ENSEMBLE_FEATURE_LIST from cross_ensemble (single source of truth).
    # This ordering matches the feat_indices in _V1_FAMILY, which is critical
    # for correct feature routing to models with non-contiguous feature layouts.
    from cross_ensemble import ENSEMBLE_FEATURE_LIST
    max_features = max(e["n_features"] for e in model.model_entries)
    if max_features > 18:
        feature_list = ENSEMBLE_FEATURE_LIST  # Full 27 features (13 old + 5 XD + xd_ma + 4 new base + 4 baseline)
    elif max_features > 13:
        feature_list = ENSEMBLE_FEATURE_LIST[:18]  # 13 old base + 5 XD
    else:
        feature_list = ENSEMBLE_FEATURE_LIST[:13]  # 13 old base only
    print(f"  Max features across ensemble: {max_features}, loading {len(feature_list)} features")

    # Data loader
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from anti_fragile import load_full_data

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Ensemble loaded: {model.n_models} models, {total_params:,} total params (frozen)")

    return model, None, feature_list, load_full_data


def load_data(load_data_fn, feature_list):
    """Load all asset data."""
    print(f"\n  Loading data from {DATA_DIR}")
    segments = load_data_fn(
        data_dir=DATA_DIR,
        feature_list=feature_list,
        asset_to_idx=ASSET_TO_IDX,
        reward_horizons=REWARD_HORIZONS,
        target_prefix="target_return",  # Agent needs RAW returns for PnL, not voladj
    )
    print(f"  Loaded {len(segments)} asset segments")
    for seg in segments:
        asset_name = ASSET_LIST[seg["asset_idx"]]
        n_bars = len(seg["features"])
        print(f"    {asset_name}: {n_bars:,} bars")
    return segments


def evaluate_agent(policy, env_val, n_episodes: int = 5,
                    obs_normalizer=None) -> dict:
    """Evaluate agent on validation data with financial metrics.

    Args:
        policy: Trained policy network
        env_val: Validation environment
        n_episodes: Number of evaluation episodes
        obs_normalizer: Optional callable(obs) -> normalized_obs. Pass
            trainer.obs_rms.normalize during training so eval uses the
            same normalization as training.

    Returns dict with:
        mean_return, std_return: raw episode reward stats
        mean_final_value: mean portfolio value at episode end
        mean_cost: mean total costs (txn + funding)
        sharpe: annualized Sharpe ratio across episodes
        max_drawdown: worst peak-to-trough drawdown observed
        win_rate: fraction of episodes with portfolio value > initial capital
        mean_turnover: mean absolute position changes per step
        mean_confidence: (DualStream only) mean confidence gate
    """
    policy.eval()
    is_dual = isinstance(policy, DualStreamActorCritic)
    episode_returns = []
    episode_costs = []
    episode_values = []
    episode_confidences = []
    episode_sharpes = []
    episode_max_dds = []
    episode_turnovers = []

    for ep in range(n_episodes):
        obs = env_val.reset(seed=42 + ep)
        total_reward = 0.0
        total_cost = 0.0
        total_confidence = 0.0
        n_steps = 0
        done = False
        fast_weights = None

        # Track per-step returns and portfolio values for financial metrics
        step_returns = []
        portfolio_values = [env_val.portfolio_value]
        peak_value = env_val.portfolio_value
        max_dd = 0.0
        total_turnover = 0.0
        prev_positions = np.zeros(NUM_ASSETS, dtype=np.float32)

        while not done:
            # Apply same normalization as training (critical for consistency)
            norm_obs = obs_normalizer(obs) if obs_normalizer is not None else obs
            obs_t = torch.tensor(norm_obs, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                if is_dual:
                    action, _, _, aux = policy.get_action(
                        obs_t, deterministic=True, fast_weights=fast_weights
                    )
                    fast_weights = aux.get("fast_weights")
                    total_confidence += aux["confidence"].mean().item()
                else:
                    action, _, _ = policy.get_action(obs_t, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy()
            obs, reward, done, info = env_val.step(action_np)
            total_reward += reward
            total_cost += info.get("txn_cost", 0) + info.get("funding_cost", 0)
            n_steps += 1

            # Financial metrics tracking
            pv = info.get("portfolio_value", env_val.portfolio_value)
            portfolio_values.append(pv)
            if len(portfolio_values) >= 2 and portfolio_values[-2] > 0:
                step_returns.append(
                    (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                )
            peak_value = max(peak_value, pv)
            if peak_value > 0:
                dd = (peak_value - pv) / peak_value
                max_dd = max(max_dd, dd)

            # Turnover: sum of absolute position changes
            positions = info.get("positions", prev_positions)
            total_turnover += np.abs(positions - prev_positions).sum()
            prev_positions = positions.copy()

        episode_returns.append(total_reward)
        episode_costs.append(total_cost)
        episode_values.append(info.get("portfolio_value", 0))
        episode_max_dds.append(max_dd)
        episode_turnovers.append(total_turnover / max(n_steps, 1))

        # Per-episode Sharpe (annualized, assuming ~24 bars/day)
        if len(step_returns) > 1:
            sr = np.array(step_returns)
            ep_sharpe = (sr.mean() / (sr.std() + 1e-8)) * np.sqrt(252 * BARS_PER_DAY)
            episode_sharpes.append(ep_sharpe)

        if is_dual and n_steps > 0:
            episode_confidences.append(total_confidence / n_steps)

    policy.train()

    result = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_final_value": float(np.mean(episode_values)),
        "mean_cost": float(np.mean(episode_costs)),
        "sharpe": float(np.mean(episode_sharpes)) if episode_sharpes else 0.0,
        "max_drawdown": float(np.max(episode_max_dds)) if episode_max_dds else 0.0,
        "win_rate": float(np.mean([1.0 if v > INITIAL_CAPITAL else 0.0
                                    for v in episode_values])),
        "mean_turnover": float(np.mean(episode_turnovers)),
    }
    if episode_confidences:
        result["mean_confidence"] = float(np.mean(episode_confidences))

    return result


def main():
    parser = argparse.ArgumentParser(description="Train trading agent on frozen world model")
    parser.add_argument("--world-model", type=str, default="v1",
                        help="World model variant (v1, v1_1, ..., v1_5, v2, ..., v9, v9_3)")
    parser.add_argument("--features", type=int, choices=[13, 18, 19], default=13,
                        help="Number of features: 13 (V1 base), 18 (V1.1+/V2+), 19 (V1.5)")
    parser.add_argument("--revin", action="store_true",
                        help="Enable RevIN (off by default; causes memorization)")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help=f"Total training steps (default: {TOTAL_TIMESTEPS:,})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load existing agent and evaluate only")
    parser.add_argument("--dual-stream", action="store_true",
                        help="Use DualStream policy (TEMP_4 enhanced)")
    parser.add_argument("--crucible", action="store_true",
                        help="Enable Crucible adversarial training")
    parser.add_argument("--sav", action="store_true",
                        help="Run SAV robustness test after training")
    parser.add_argument("--augment", action="store_true",
                        help=f"Enable stress augmentation ({AUGMENT_PROB*100:.0f}%% "
                             f"of episodes get crash/vol-spike/squeeze/cost-shock)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use V1.E cross-model ensemble instead of single model")
    parser.add_argument("--ensemble-models", type=str, default=None,
                        help="Comma-separated model keys for ensemble "
                             f"(default: {','.join(DEFAULT_ENSEMBLE_MODELS)})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    # Cost model flags
    parser.add_argument("--spot", action="store_true", default=True,
                        help="SPOT cost model (default)")
    parser.add_argument("--perp", action="store_true",
                        help="Perpetual futures cost model (overrides --spot)")
    parser.add_argument("--long-only", action="store_true", default=True,
                        help="Long-only positions [0, MAX_POS] (default)")
    parser.add_argument("--margin", action="store_true",
                        help="Allow short positions [-MAX_POS, MAX_POS] (overrides --long-only)")
    parser.add_argument("--decision-interval", type=int, default=1,
                        help="Bars between trade decisions (default: 1 = every bar)")
    args = parser.parse_args()

    use_revin = args.revin
    use_ensemble = args.ensemble
    variant = args.world_model
    spot_mode = not args.perp
    long_only = not args.margin
    decision_interval = args.decision_interval

    policy_type = "dual_stream" if args.dual_stream else "baseline"
    cost_desc = "SPOT (0.10%/side, no funding)" if spot_mode else "Perp (0.05%/side + funding)"
    pos_desc = "Long-only [0, 0.20]" if long_only else "Long/Short [-0.20, 0.20]"

    model_label = "V1.E ensemble" if use_ensemble else variant
    print("=" * 70)
    print(f"  TRADING AGENT TRAINER (PPO)")
    print("=" * 70)
    print(f"  World Model:  {model_label}")
    if not use_ensemble:
        print(f"  Features:     {args.features}")
        print(f"  RevIN:        {'enabled' if use_revin else 'disabled'}")
    print(f"  Policy:       {policy_type}")
    print(f"  Cost Model:   {cost_desc}")
    print(f"  Positions:    {pos_desc}")
    if decision_interval > 1:
        print(f"  Decision Int: every {decision_interval} bars")
    print(f"  Crucible:     {'ON' if args.crucible else 'OFF'}")
    print(f"  Augmentation: {'ON (' + str(int(AUGMENT_PROB*100)) + '%)' if args.augment else 'OFF'}")
    print(f"  Device:       {DEVICE}")
    print(f"  Total Steps:  {args.steps:,}")
    print(f"  Seed:         {args.seed}")
    print()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load world model (single or ensemble)
    if use_ensemble:
        ensemble_keys = None
        if args.ensemble_models:
            ensemble_keys = [k.strip() for k in args.ensemble_models.split(",")]
        model, revin, feature_list, load_data_fn = load_ensemble_model(ensemble_keys)
    else:
        model, revin, feature_list, load_data_fn = load_world_model(
            variant, args.features, use_revin
        )

    # Load data
    segments = load_data(load_data_fn, feature_list)

    # Create environments
    print("\n  Creating training environment...")
    env_train = WorldModelTradingEnv(
        world_model=model,
        data_segments=segments,
        feature_list=feature_list,
        mode="train",
        revin=revin,
        episode_length=EPISODE_LENGTH,
        seed=args.seed,
        enable_augmentation=args.augment,
        spot_mode=spot_mode,
        long_only=long_only,
        decision_interval=decision_interval,
    )

    print("  Creating validation environment...")
    env_val = WorldModelTradingEnv(
        world_model=model,
        data_segments=segments,
        feature_list=feature_list,
        mode="val",
        revin=revin,
        episode_length=EPISODE_LENGTH,
        seed=args.seed + 1000,
        spot_mode=spot_mode,
        long_only=long_only,
        decision_interval=decision_interval,
    )

    print(f"  Obs dim: {env_train.observation_dim}, Action dim: {env_train.action_dim}")
    print(f"  Valid training assets: {len(env_train.valid_assets)}")
    print(f"  Valid validation assets: {len(env_val.valid_assets)}")

    # Build agent tag for checkpoints
    cost_tag = "_spot" if spot_mode else "_perp"
    pos_tag = "" if long_only else "_margin"
    if use_ensemble:
        augment_tag = "_augment" if args.augment else ""
        agent_tag = f"v1e_ensemble{cost_tag}{pos_tag}{augment_tag}"
    else:
        feat_tag = f"f{args.features}"
        revin_tag = "_revin" if use_revin else ""
        augment_tag = "_augment" if args.augment else ""
        agent_tag = f"{variant}_{feat_tag}{revin_tag}{cost_tag}{pos_tag}{augment_tag}"

    if args.eval_only:
        # Load and evaluate
        agent_path = AGENT_MODEL_DIR / f"agent_{agent_tag}_best.pt"
        if not agent_path.exists():
            print(f"  [ERROR] No agent checkpoint found at {agent_path}")
            sys.exit(1)

        ckpt = torch.load(agent_path, map_location=DEVICE, weights_only=False)
        saved_type = ckpt.get("policy_type", "baseline")

        if saved_type == "dual_stream":
            policy = DualStreamActorCritic(
                obs_dim=env_train.observation_dim,
                action_dim=env_train.action_dim,
            ).to(DEVICE)
        else:
            policy = ActorCritic(
                obs_dim=env_train.observation_dim,
                action_dim=env_train.action_dim,
            ).to(DEVICE)
        policy.load_state_dict(ckpt["policy_state_dict"])
        print(f"\n  Loaded: {agent_path.name}")

        # Restore observation normalizer from checkpoint (required for correct eval)
        obs_normalizer = None
        if "obs_rms" in ckpt:
            obs_rms = RunningMeanStd(shape=(env_train.observation_dim,))
            obs_rms.load_state_dict(ckpt["obs_rms"])
            obs_normalizer = obs_rms.normalize
            print(f"  Obs normalizer loaded (count={obs_rms.count:.0f})")
        else:
            print(f"  [WARN] No obs_rms in checkpoint -- using raw observations")

        # In-sample (train) evaluation
        print(f"\n  {'='*50}")
        print(f"  IN-SAMPLE (Train, 90% of data)")
        print(f"  {'='*50}")
        train_metrics = evaluate_agent(policy, env_train, n_episodes=10,
                                        obs_normalizer=obs_normalizer)
        print(_format_eval(train_metrics))

        # Out-of-sample (val) evaluation
        print(f"\n  {'='*50}")
        print(f"  OUT-OF-SAMPLE (Val, last 10%, purge={PURGE_GAP_BARS} bars)")
        print(f"  {'='*50}")
        val_metrics = evaluate_agent(policy, env_val, n_episodes=10,
                                      obs_normalizer=obs_normalizer)
        print(_format_eval(val_metrics))

        # Overfit diagnostic
        if train_metrics["sharpe"] != 0:
            oos_ratio = val_metrics["sharpe"] / (train_metrics["sharpe"] + 1e-8)
        else:
            oos_ratio = 0.0
        print(f"\n  OOS/IS Sharpe ratio: {oos_ratio:.3f}", end="")
        if oos_ratio > 0.5:
            print(f"  [PASS] (>{0.5:.1f})")
        elif oos_ratio > 0.0:
            print(f"  [WARN] Low OOS retention")
        else:
            print(f"  [FAIL] OOS performance collapsed")

        # SAV robustness test
        if args.sav:
            print(f"\n  {'='*50}")
            print(f"  SAV ROBUSTNESS TEST")
            print(f"  {'='*50}")
            sav_results = sav_test(policy, env_val, n_episodes=5,
                                    obs_normalizer=obs_normalizer)
            for k, v in sav_results.items():
                print(f"    {k}: {v:.4f}")
            stability = sav_results["stability_ratio"]
            if stability > 0.7:
                print(f"    [PASS] Agent is robust (stability={stability:.2f})")
            else:
                print(f"    [WARN] Agent may be fragile (stability={stability:.2f})")

        sys.exit(0)

    # Create policy
    if args.dual_stream:
        policy = DualStreamActorCritic(
            obs_dim=env_train.observation_dim,
            action_dim=env_train.action_dim,
        ).to(DEVICE)
        print(f"\n  DualStream Policy: {count_parameters(policy):,} params")
        print(f"    Hebbian plasticity: ON")
        print(f"    Surprise gating: ON")
        print(f"    Confidence gate: ON")
    else:
        policy = ActorCritic(
            obs_dim=env_train.observation_dim,
            action_dim=env_train.action_dim,
        ).to(DEVICE)
        print(f"\n  Baseline Policy: {count_parameters(policy):,} params")

    # Create trainer
    trainer = PPOTrainer(
        env=env_train,
        policy=policy,
        use_crucible=args.crucible,
    )

    def eval_fn(policy):
        return evaluate_agent(policy, env_val, n_episodes=3,
                              obs_normalizer=trainer.obs_rms.normalize)

    # Resume checkpoint path
    resume_path = None
    if args.resume:
        resume_path = AGENT_MODEL_DIR / f"agent_{agent_tag}_latest.pt"
        if not resume_path.exists():
            print(f"  [WARN] No checkpoint found at {resume_path}, starting fresh")
            resume_path = None
        else:
            print(f"  [RESUME] Found checkpoint: {resume_path.name}")

    # Train
    print(f"\n{'='*70}")
    print(f"  TRAINING START")
    print(f"{'='*70}\n")

    start_time = time.time()
    policy, metrics = trainer.train(
        total_timesteps=args.steps,
        log_every=5,
        eval_fn=eval_fn,
        agent_tag=agent_tag,
        resume_path=resume_path,
    )
    elapsed = time.time() - start_time

    # Save
    save_path = AGENT_MODEL_DIR / f"agent_{agent_tag}_final.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "policy_type": policy_type,
        "world_model_variant": variant,
        "features": args.features,
        "use_revin": use_revin,
        "spot_mode": spot_mode,
        "long_only": long_only,
        "decision_interval": decision_interval,
        "total_steps": args.steps,
        "seed": args.seed,
        "training_metrics": metrics,
        "obs_rms": trainer.obs_rms.state_dict(),
    }, save_path)
    print(f"\n  Agent saved to: {save_path}")

    # Final evaluation
    print(f"\n  Final Evaluation (10 episodes):")
    final_metrics = evaluate_agent(policy, env_val, n_episodes=10,
                                    obs_normalizer=trainer.obs_rms.normalize)
    for k, v in final_metrics.items():
        print(f"    {k}: {v:.4f}")

    # SAV robustness test
    sav_results = None
    if args.sav:
        print(f"\n  SAV Robustness Test (weight perturbation):")
        sav_results = sav_test(policy, env_val, n_episodes=5,
                                obs_normalizer=trainer.obs_rms.normalize)
        for k, v in sav_results.items():
            print(f"    {k}: {v:.4f}")
        stability = sav_results["stability_ratio"]
        if stability > 0.7:
            print(f"    [PASS] Agent is robust (stability={stability:.2f})")
        else:
            print(f"    [WARN] Agent may be fragile (stability={stability:.2f})")

    # Save results JSON
    # Collect scenario stats if augmentation was used
    scenario_stats = None
    if args.augment and hasattr(env_train, "get_scenario_stats"):
        scenario_stats = env_train.get_scenario_stats()
        if scenario_stats:
            print(f"\n  Scenario Stats:")
            for sc_name, stats in sorted(scenario_stats.items()):
                print(f"    {sc_name}: {stats['count']} episodes, "
                      f"mean_reward={stats['mean_reward']:.3f}")

    results = {
        "variant": variant,
        "features": args.features,
        "use_revin": use_revin,
        "spot_mode": spot_mode,
        "long_only": long_only,
        "decision_interval": decision_interval,
        "policy_type": policy_type,
        "augmentation": args.augment,
        "total_steps": args.steps,
        "training_time_sec": elapsed,
        "final_eval": final_metrics,
        "sav_results": sav_results,
        "scenario_stats": scenario_stats,
        "training_metrics": metrics[-10:] if metrics else [],
    }
    results_path = AGENT_LOG_DIR / f"agent_{agent_tag}_{datetime.now():%Y%m%d_%H%M%S}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE ({elapsed/60:.1f} min)")
    print(f"  Agent: {save_path.name}")
    print(f"  Results: {results_path.name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
