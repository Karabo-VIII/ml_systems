"""
chess_zero.az.dqn_minatar -- a REAL model-free DQN agent for the MinAtar scaled-Atari benchmark.

WHY DQN (not MuZero): MinAtar (Young & Tian 2019, arXiv:1903.03176) is the academically-standard
miniaturized-Atari suite -- a (10,10,C) multi-channel pixel grid, single agent, real Atari dynamics
(Breakout/Asterix/Freeway/Seaquest/SpaceInvaders). It is SPECIFICALLY designed to be learnable by a
fast RL method in minutes-to-an-hour on modest hardware. A prior MuZero (search-based) attempt was too
slow (batch-1 latent search ~4 min/iter). DQN is the RELIABLE model-free path: a small conv Q-network
over the grid + experience replay + epsilon-greedy + a target network. This is exactly the method the
MinAtar paper benchmarks with, and it learns Breakout/Asterix clearly above random fast.

This module exposes:
  - MinAtarQNet : the conv Q-network. ctor kwargs (in_channels, num_actions, n_filters, hidden) are
                  saved verbatim into the checkpoint["arch"] so a fresh net can be reconstructed bit-
                  exactly on reload.
  - ReplayBuffer: a fixed-capacity uint8/int circular experience-replay buffer (memory-frugal so we
                  share the 8GB 4060 politely with the chess + WM processes already running).
  - DQNAgent    : ties the online/target nets + optimizer + epsilon schedule together; exposes
                  act() (epsilon-greedy), act_greedy() (eval), push(), learn() (Double-DQN step),
                  sync_target(), and save()/load() helpers around the canonical checkpoint dict.

The network architecture is the standard MinAtar DQN (one conv layer 3x3 stride 1 over the 10x10 grid
-> flatten -> one hidden ReLU layer -> Q-values per action). Tiny by design: ~100k params, so it both
fits trivially in spare VRAM and learns in a sub-hour budget.

GPU: mixed precision (torch.amp) on the learn step; a SEPARATE CUDA context is fine (we share the GPU
politely via a small batch). No emoji (Windows cp1252). ADDITIVE: this file is new; it touches nothing
outside projects/chess_zero/.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__contract__ = {
    "kind": "rl-agent",
    "inputs": ["MinAtar (H,W,C) float grid observations", "scalar per-step rewards"],
    "outputs": [
        "act(obs, eps) -> action int (epsilon-greedy)",
        "act_greedy(obs) -> action int (argmax Q, eval)",
        "learn() -> td_loss float (Double-DQN minibatch step)",
        "save(path, game, meta) / load(path) -> canonical checkpoint dict",
    ],
    "invariants": [
        "model-free Q-learning (NO search) -- fast + reliable on MinAtar",
        "obs are CHW-permuted from MinAtar's HWC before the conv (channels first)",
        "checkpoint stores arch ctor kwargs so a fresh net reconstructs bit-exactly on reload",
        "no emoji in any print (Windows cp1252)",
    ],
}


# --------------------------------------------------------------------------- #
# Q-network -- the standard MinAtar conv DQN.
# --------------------------------------------------------------------------- #
class MinAtarQNet(nn.Module):
    """Conv Q-network over a MinAtar (C,10,10) grid.

    Architecture (Young & Tian 2019 DQN baseline, lightly modernized):
        Conv2d(in_channels -> n_filters, 3x3, stride 1, pad 1) -> ReLU
        flatten (n_filters * H * W) -> Linear(-> hidden) -> ReLU -> Linear(-> num_actions)

    The ctor kwargs are the ONLY thing needed to rebuild this net on reload; they are saved verbatim
    into the checkpoint["arch"] block."""

    def __init__(self, in_channels: int, num_actions: int, h: int = 10, w: int = 10,
                 n_filters: int = 32, hidden: int = 128, dueling: bool = False):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_actions = int(num_actions)
        self.h = int(h)
        self.w = int(w)
        self.n_filters = int(n_filters)
        self.hidden = int(hidden)
        self.dueling = bool(dueling)   # Rainbow: split Q into V(s) + A(s,a) streams
        self.conv = nn.Conv2d(self.in_channels, self.n_filters, kernel_size=3, stride=1, padding=1)
        flat = self.n_filters * self.h * self.w
        if self.dueling:
            self.val_fc = nn.Linear(flat, self.hidden)
            self.val_out = nn.Linear(self.hidden, 1)
            self.adv_fc = nn.Linear(flat, self.hidden)
            self.adv_out = nn.Linear(self.hidden, self.num_actions)
        else:
            self.fc1 = nn.Linear(flat, self.hidden)
            self.fc2 = nn.Linear(self.hidden, self.num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) float
        x = F.relu(self.conv(x))
        x = x.flatten(start_dim=1)
        if self.dueling:
            v = self.val_out(F.relu(self.val_fc(x)))                    # (B,1)
            a = self.adv_out(F.relu(self.adv_fc(x)))                    # (B,A)
            return v + (a - a.mean(dim=1, keepdim=True))               # identifiable dueling combine
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def arch_kwargs(self) -> dict:
        """The exact ctor kwargs needed to rebuild this net on reload. `dueling` is included so a
        dueling checkpoint reconstructs correctly; old checkpoints lack it -> defaults False (the
        original conv->fc1->fc2 net), so they keep loading bit-exactly."""
        return {
            "in_channels": self.in_channels,
            "num_actions": self.num_actions,
            "h": self.h,
            "w": self.w,
            "n_filters": self.n_filters,
            "hidden": self.hidden,
            "dueling": self.dueling,
        }


# --------------------------------------------------------------------------- #
# Replay buffer -- fixed-capacity circular, memory-frugal.
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Circular experience replay over (obs, action, reward, next_obs, done).

    Stores observations as float32 grids (MinAtar grids are small binary planes; float32 keeps the
    learn step branch-free). Capacity default 100k transitions of (10,10,~6) ~= 240 MB host RAM --
    fine; the GPU only ever sees one minibatch at a time, so VRAM use is tiny (we share the 4060)."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]):
        self.capacity = int(capacity)
        self.obs_shape = obs_shape  # (H, W, C)
        c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
        self._obs = np.zeros((self.capacity, c, h, w), dtype=np.float32)
        self._next = np.zeros((self.capacity, c, h, w), dtype=np.float32)
        self._act = np.zeros((self.capacity,), dtype=np.int64)
        self._rew = np.zeros((self.capacity,), dtype=np.float32)
        self._done = np.zeros((self.capacity,), dtype=np.float32)
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self.capacity if self._full else self._pos

    @staticmethod
    def _to_chw(obs: np.ndarray) -> np.ndarray:
        # MinAtar gives HWC; conv wants CHW.
        return np.ascontiguousarray(np.transpose(obs, (2, 0, 1)), dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done) -> None:
        i = self._pos
        self._obs[i] = self._to_chw(obs)
        self._next[i] = self._to_chw(next_obs)
        self._act[i] = int(action)
        self._rew[i] = float(reward)
        self._done[i] = 1.0 if done else 0.0
        self._pos += 1
        if self._pos >= self.capacity:
            self._pos = 0
            self._full = True

    def sample(self, batch_size: int, device: torch.device):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        obs = torch.from_numpy(self._obs[idx]).to(device, non_blocking=True)
        nxt = torch.from_numpy(self._next[idx]).to(device, non_blocking=True)
        act = torch.from_numpy(self._act[idx]).to(device, non_blocking=True)
        rew = torch.from_numpy(self._rew[idx]).to(device, non_blocking=True)
        done = torch.from_numpy(self._done[idx]).to(device, non_blocking=True)
        return obs, act, rew, nxt, done


# --------------------------------------------------------------------------- #
# DQN agent -- online/target nets + epsilon schedule + Double-DQN learn step.
# --------------------------------------------------------------------------- #
class DQNAgent:
    """A Double-DQN agent. Holds an online net + a target net + the optimizer + the epsilon schedule.

    Double-DQN: the online net selects the next action (argmax), the target net evaluates it -- this
    de-biases the standard DQN max-overestimation and is a near-free stability win. Huber (smooth-L1)
    TD loss. AMP on the learn step. Gradient clipping for stability when sharing the GPU."""

    def __init__(self, obs_shape: Tuple[int, int, int], num_actions: int,
                 device: Optional[torch.device] = None, lr: float = 2.5e-4, gamma: float = 0.99,
                 n_filters: int = 32, hidden: int = 128, grad_clip: float = 10.0,
                 use_amp: bool = True, dueling: bool = False, n_step: int = 1):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = float(gamma)
        self.n_step = max(1, int(n_step))   # Rainbow n-step: buffer holds n-step returns; target uses gamma^n
        self.grad_clip = float(grad_clip)
        in_channels = obs_shape[2]
        self.online = MinAtarQNet(in_channels, num_actions, h=obs_shape[0], w=obs_shape[1],
                                  n_filters=n_filters, hidden=hidden, dueling=dueling).to(self.device)
        self.target = MinAtarQNet(in_channels, num_actions, h=obs_shape[0], w=obs_shape[1],
                                  n_filters=n_filters, hidden=hidden, dueling=dueling).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.num_actions = int(num_actions)
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # --- action selection --------------------------------------------------- #
    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        chw = np.ascontiguousarray(np.transpose(obs, (2, 0, 1)), dtype=np.float32)
        return torch.from_numpy(chw).unsqueeze(0).to(self.device)

    def act(self, obs: np.ndarray, eps: float) -> int:
        """Epsilon-greedy action for training."""
        if random.random() < eps:
            return random.randrange(self.num_actions)
        return self.act_greedy(obs)

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray) -> int:
        """Greedy (argmax Q) action for evaluation."""
        self.online.eval()
        q = self.online(self._obs_to_tensor(obs))
        a = int(torch.argmax(q, dim=1).item())
        self.online.train()
        return a

    # --- learning ----------------------------------------------------------- #
    def learn(self, buffer: ReplayBuffer, batch_size: int) -> float:
        """One Double-DQN minibatch update. Returns the TD (Huber) loss as a float."""
        obs, act, rew, nxt, done = buffer.sample(batch_size, self.device)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            q = self.online(obs).gather(1, act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                # Double-DQN: online selects, target evaluates.
                next_a = torch.argmax(self.online(nxt), dim=1, keepdim=True)
                next_q = self.target(nxt).gather(1, next_a).squeeze(1)
                tgt = rew + (self.gamma ** self.n_step) * next_q * (1.0 - done)
            loss = F.smooth_l1_loss(q, tgt)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        return float(loss.detach().item())

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    # --- checkpoint helpers ------------------------------------------------- #
    def save(self, path: str, game: str, meta: dict) -> dict:
        """Write the canonical checkpoint dict and return it."""
        ckpt = {
            "state_dict": self.online.state_dict(),
            "arch": self.online.arch_kwargs(),
            "game": game,
            "meta": dict(meta),
        }
        torch.save(ckpt, path)
        return ckpt


# --------------------------------------------------------------------------- #
# Reload helper -- reconstruct a fresh net from a checkpoint on disk.
# --------------------------------------------------------------------------- #
def load_qnet(path: str, device: Optional[torch.device] = None) -> Tuple[MinAtarQNet, dict]:
    """Reconstruct a FRESH MinAtarQNet from the saved checkpoint and load its weights.

    This is the reload contract used by the VERIFY step: a brand-new net built ONLY from the saved
    arch kwargs, then loaded from the saved state_dict, must reproduce the trained eval return."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    net = MinAtarQNet(**ckpt["arch"]).to(dev)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, ckpt
