"""
V4 Agent Trainer — Two-Phase Training

Phase A: Supervised Pretraining
  - Feed real market data through frozen world model
  - Compute hindsight-optimal actions from actual returns
  - Train agent to predict these actions (cross-entropy)
  - Gives the agent basic market intuition before RL

Phase B: PPO Fine-Tuning
  - Dream short trajectories inside the world model
  - Optimize log-utility PnL with transaction costs
  - PPO clipping for stable policy updates
  - Uses GAE for advantage estimation

Prerequisites: World model must have passed validation gate.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import polars as pl
import sys
import copy
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from world_model import MambaWorldModel
from revin import RevIN
from agent import ActorCritic, compute_reward, compute_gae
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets


# ==============================================================================
# PHASE A: SUPERVISED PRETRAINING
# ==============================================================================

class SupervisedDataset(Dataset):
    """States from world model + hindsight-optimal actions."""

    def __init__(self, states: np.ndarray, targets: np.ndarray, returns: np.ndarray):
        self.states = states    # [N, state_dim]
        self.targets = targets  # [N] integer actions
        self.returns = returns  # [N] actual returns (for weighting)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.states[idx]).float(),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.returns[idx], dtype=torch.float32),
        )


def compute_oracle_actions(returns: np.ndarray, fee: float = BASE_FEE) -> np.ndarray:
    """
    Compute hindsight-optimal actions given actual returns.
    Simple threshold: long if return > fee, short if return < -fee, else neutral.
    """
    actions = np.ones(len(returns), dtype=np.int64)  # default neutral
    actions[returns > fee * 2] = 2   # long
    actions[returns < -fee * 2] = 0  # short
    return actions


@torch.no_grad()
def extract_world_model_states(wm, data_segments, revin=None):
    """
    Run observations through frozen world model and extract states.
    Returns: list of (states_array, oracle_actions_array, returns_array)
    """
    wm.eval()
    all_segments = []

    for feats, targs, asset_idx in data_segments:
        # Process in chunks to fit memory
        chunk_size = 512
        states_list = []

        for start in range(0, len(feats) - WM_SEQ_LEN, chunk_size):
            end = min(start + chunk_size, len(feats) - WM_SEQ_LEN)
            batch_obs = []
            batch_assets = []

            for i in range(start, end):
                obs_chunk = feats[i:i + WM_SEQ_LEN]
                batch_obs.append(obs_chunk)
                batch_assets.append(asset_idx)

            obs_tensor = torch.from_numpy(np.stack(batch_obs)).float().to(DEVICE)
            asset_tensor = torch.tensor(batch_assets, dtype=torch.long).to(DEVICE)
            if revin is not None:
                obs_tensor = revin(obs_tensor, mode='norm')

            h_seq, z_seq, _ = wm.encode_sequence(obs_tensor, asset_tensor)

            # Take the last timestep state from each sequence
            h_last = h_seq[:, -1, :].cpu().numpy()
            z_last = z_seq[:, -1, :].cpu().numpy()
            state = np.concatenate([h_last, z_last], axis=-1)
            states_list.append(state)

        if not states_list:
            continue

        states = np.concatenate(states_list, axis=0)

        # Align returns with states (each state corresponds to the last timestep)
        aligned_returns = targs[WM_SEQ_LEN - 1: WM_SEQ_LEN - 1 + len(states)]
        oracle_actions = compute_oracle_actions(aligned_returns)

        all_segments.append((states, oracle_actions, aligned_returns))
        print(f"    Asset {asset_idx}: {len(states):,} states extracted")

    return all_segments


def train_supervised(agent, wm, data_segments, revin=None, epochs=20):
    """Phase A: Supervised pretraining on oracle actions."""
    print("\n-- Phase A: Supervised Pretraining --")

    # Extract states from world model
    print("  Extracting world model states...")
    state_segments = extract_world_model_states(wm, data_segments, revin=revin)

    if not state_segments:
        print("  [FAIL] No states extracted. Aborting.")
        return

    # Combine into single dataset
    all_states = np.concatenate([s for s, _, _ in state_segments])
    all_actions = np.concatenate([a for _, a, _ in state_segments])
    all_returns = np.concatenate([r for _, _, r in state_segments])

    dataset = SupervisedDataset(all_states, all_actions, all_returns)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

    optimizer = optim.AdamW(agent.parameters(), lr=3e-4, weight_decay=1e-3)

    print(f"  Training on {len(dataset):,} samples for {epochs} epochs")

    for epoch in range(epochs):
        agent.train()
        losses, accs = [], []

        for states, targets, returns in loader:
            states = states.to(DEVICE)
            targets = targets.to(DEVICE)
            returns = returns.to(DEVICE)

            action_logits, _ = agent(states)
            loss = F.cross_entropy(action_logits, targets)

            # Weight by absolute return (focus on big moves)
            weights = 1.0 + returns.abs() * 10.0
            weighted_loss = (F.cross_entropy(action_logits, targets, reduction="none") * weights).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            acc = (action_logits.argmax(dim=-1) == targets).float().mean()
            losses.append(loss.item())
            accs.append(acc.item())

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)
        print(f"    Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Acc: {avg_acc*100:.1f}%")

    # Save supervised checkpoint
    torch.save(agent.state_dict(), MODEL_DIR / "v4_agent_supervised.pt")
    print(f"  [OK] Supervised pretraining complete. Saved to v4_agent_supervised.pt")


# ==============================================================================
# PHASE B: PPO FINE-TUNING (In World Model Dreams)
# ==============================================================================

def collect_rollouts(agent, wm, data_segments, revin=None, n_rollouts=64):
    """
    Collect rollout data by stepping through real observations
    and using the agent to select actions.

    Returns: dict with flattened trajectory data
    """
    agent.eval()
    wm.eval()

    storage = {
        "states": [], "actions": [], "log_probs": [],
        "values": [], "rewards": [], "dones": [],
    }

    # Randomly sample starting points from data
    for _ in range(n_rollouts):
        seg_idx = np.random.randint(len(data_segments))
        feats, targs, asset_idx = data_segments[seg_idx]

        # Random start with enough room for horizon
        max_start = len(feats) - WM_SEQ_LEN - DREAM_HORIZON - 1
        if max_start < 0:
            continue
        start = np.random.randint(0, max_start)

        # Encode initial context
        obs_context = torch.from_numpy(feats[start:start + WM_SEQ_LEN]).float().unsqueeze(0).to(DEVICE)
        asset_tensor = torch.tensor([asset_idx], dtype=torch.long).to(DEVICE)
        if revin is not None:
            obs_context = revin(obs_context, mode='norm')

        with torch.no_grad():
            h_seq, z_seq, _ = wm.encode_sequence(obs_context, asset_tensor)
            h = h_seq[:, -1, :]  # [1, d_model]
            z = z_seq[:, -1, :]  # [1, flat_dim]

        prev_action = torch.tensor([1], device=DEVICE)  # Start neutral

        # Roll out using real returns (not dreamed - more reliable)
        for t in range(DREAM_HORIZON):
            idx = start + WM_SEQ_LEN + t
            if idx >= len(targs):
                break

            state = torch.cat([h, z], dim=-1)  # [1, state_dim]

            with torch.no_grad():
                action, log_prob, value, _ = agent.get_action(state)

            # Real market return
            market_return = torch.tensor([targs[idx]], device=DEVICE)
            reward = compute_reward(action, prev_action, market_return)

            storage["states"].append(state.squeeze(0).cpu())
            storage["actions"].append(action.item())
            storage["log_probs"].append(log_prob.item())
            storage["values"].append(value.item())
            storage["rewards"].append(reward.item())

            prev_action = action

            # Step world model forward with next real observation
            if idx + 1 < len(feats):
                next_obs = torch.from_numpy(feats[idx:idx + 1]).float().unsqueeze(0).to(DEVICE)
                if revin is not None:
                    next_obs = revin(next_obs, mode='norm')
                with torch.no_grad():
                    h_s, z_s, _ = wm.encode_sequence(
                        torch.cat([obs_context[:, 1:, :], next_obs], dim=1),
                        asset_tensor,
                    )
                    h = h_s[:, -1, :]
                    z = z_s[:, -1, :]

    # Convert to tensors
    if not storage["states"]:
        return None

    return {
        "states": torch.stack(storage["states"]),
        "actions": torch.tensor(storage["actions"], dtype=torch.long),
        "log_probs": torch.tensor(storage["log_probs"]),
        "values": torch.tensor(storage["values"]),
        "rewards": torch.tensor(storage["rewards"]),
    }


def ppo_update(agent, rollout_data, optimizer):
    """Single PPO update from collected rollouts."""
    states = rollout_data["states"].to(DEVICE)
    actions = rollout_data["actions"].to(DEVICE)
    old_log_probs = rollout_data["log_probs"].to(DEVICE)
    values = rollout_data["values"].to(DEVICE)
    rewards = rollout_data["rewards"].to(DEVICE)

    # Compute advantages (simple, no next_value bootstrap for simplicity)
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(T)):
        running = rewards[t] + GAMMA * running
        advantages[t] = running - values[t]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values

    # PPO epochs
    total_loss_val = 0.0
    batch_size = min(256, T)

    for _ in range(PPO_EPOCHS_PER_UPDATE):
        indices = torch.randperm(T, device=DEVICE)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            idx = indices[start:end]

            log_prob, value_pred, entropy, value_logits = agent.evaluate_actions(
                states[idx], actions[idx]
            )

            # Policy loss (clipped)
            ratio = torch.exp(log_prob - old_log_probs[idx])
            surr1 = ratio * advantages[idx]
            surr2 = torch.clamp(ratio, 1.0 - PPO_CLIP, 1.0 + PPO_CLIP) * advantages[idx]
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (TwoHot)
            value_loss = agent.bucketer.compute_loss(value_logits, returns[idx])

            # Entropy bonus
            entropy_loss = -entropy.mean()

            loss = (
                policy_loss
                + VALUE_COEF * value_loss
                + agent.alpha.detach() * entropy_loss
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), AGENT_GRAD_CLIP)
            optimizer.step()

            total_loss_val += loss.item()

    return total_loss_val


def train_ppo(agent, wm, data_segments, revin=None, n_updates=200):
    """Phase B: PPO fine-tuning."""
    print("\n-- Phase B: PPO Fine-Tuning --")

    optimizer = optim.AdamW(agent.parameters(), lr=AGENT_LR, weight_decay=1e-3)
    best_avg_reward = -float("inf")

    for update in tqdm(range(n_updates), desc="PPO Updates"):
        # Collect rollouts
        rollout = collect_rollouts(agent, wm, data_segments, revin=revin, n_rollouts=32)
        if rollout is None:
            continue

        avg_reward = rollout["rewards"].mean().item()

        # PPO update
        agent.train()
        loss = ppo_update(agent, rollout, optimizer)

        if (update + 1) % 20 == 0:
            # Quick action distribution check
            action_dist = torch.bincount(rollout["actions"], minlength=3).float()
            action_dist = action_dist / action_dist.sum()

            print(
                f"  Update {update+1:4d} | "
                f"Reward: {avg_reward:.5f} | "
                f"Actions: [S:{action_dist[0]:.0%} N:{action_dist[1]:.0%} L:{action_dist[2]:.0%}]"
            )

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                torch.save(agent.state_dict(), MODEL_DIR / "v4_agent_ppo_best.pt")

    torch.save(agent.state_dict(), MODEL_DIR / "v4_agent_ppo_final.pt")
    print(f"  [OK] PPO fine-tuning complete. Best avg reward: {best_avg_reward:.5f}")


# ==============================================================================
# MAIN
# ==============================================================================

def load_data_segments():
    """Load processed data and split into train segments."""
    files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
    segments = []

    for f in files:
        asset_name = f.stem.split("_")[0].upper()
        if asset_name not in ASSET_TO_IDX:
            continue

        asset_idx = ASSET_TO_IDX[asset_name]
        try:
            df = pl.read_parquet(f)
            df = selective_drop_nulls(df, FEATURE_LIST, [1], asset_name)
            feats, tgt_dict = extract_features_targets(df, FEATURE_LIST, [1], asset_name)
            targs = tgt_dict[1]

            # Use 80% for training (reserve 20% for evaluation)
            split = int(len(feats) * 0.80)
            segments.append((feats[:split], targs[:split], asset_idx))
            print(f"  [OK] {asset_name}: {split:,} bars")

        except Exception as e:
            print(f"  [FAIL] {asset_name}: {e}")

    return segments


def train_agent():
    print("=" * 60)
    print("  V4 TRADING AGENT TRAINER")
    print("=" * 60)

    # -- Load World Model --------------------------------------------------
    wm_path = BASE_MODEL_DIR / "v4_wm_best_ema.pt"
    if not wm_path.exists():
        print("[FAIL] World model not found. Train world model first.")
        print(f"  Expected: {wm_path}")
        return

    wm = MambaWorldModel().to(DEVICE)
    ckpt = torch.load(wm_path, map_location=DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        wm.load_state_dict(ckpt["model_state_dict"])
    else:
        wm.load_state_dict(ckpt)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Load RevIN for distribution-consistent inference
    revin = RevIN(num_features=INPUT_DIM).to(DEVICE)
    if isinstance(ckpt, dict) and "revin_state_dict" in ckpt:
        revin.load_state_dict(ckpt["revin_state_dict"])
        print(f"  [OK] World model + RevIN loaded from {wm_path.name}")
    else:
        print(f"  [OK] World model loaded from {wm_path.name} (no RevIN state, using identity)")
    revin.eval()
    for p in revin.parameters():
        p.requires_grad = False

    # -- Load Data ---------------------------------------------------------
    segments = load_data_segments()
    if not segments:
        print("[FAIL] No data found.")
        return

    # -- Initialize Agent --------------------------------------------------
    agent = ActorCritic().to(DEVICE)
    print(f"  Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # -- Phase A: Supervised -----------------------------------------------
    train_supervised(agent, wm, segments, revin=revin, epochs=20)

    # -- Phase B: PPO ------------------------------------------------------
    train_ppo(agent, wm, segments, revin=revin, n_updates=200)

    print("\n" + "=" * 60)
    print("  AGENT TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    train_agent()
