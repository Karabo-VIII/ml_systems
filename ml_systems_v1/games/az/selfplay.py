"""
chess_zero.az.selfplay -- the AlphaZero self-play -> data -> train loop.

This is the REAL structure of the AlphaZero training cycle (arXiv:1712.01815),
with the loss exactly as the paper specifies:

    L = (z - v)^2  -  pi^T log p  +  c * ||theta||^2
        \_______/    \__________/    \__________/
        value MSE     policy CE       L2 weight decay

where:
    pi = MCTS visit-count distribution at each visited state (the "improved policy"),
    z  = final game outcome in {+1, 0, -1} from that state's player's perspective,
    p  = net policy, v = net value.

STATUS (2026-06-08): the primitives here are LOAD-BEARING in the real pipeline;
only the in-file demo curriculum stays guarded:
    * Sample / generate_selfplay_game() / train_step()  -- LOAD-BEARING. The
      hardened loop (train_robust.py) imports Sample + train_step and drives the
      REAL self-play -> train -> eval run (with the champion gate + curriculum +
      OOM guards). train_step() is the exact paper-loss optimisation step it calls.
    * train_loop()  -- a DEMO-ONLY curriculum kept in this file for reference,
      still GUARDED behind `if RUN_TRAINING` (default False) so importing/compiling
      this module never launches a run. The PRODUCTION loop lives in train_robust.py,
      NOT here -- this guard is about THIS file's __main__, not the project.

To run the production training, use train_robust.py (or `play.py learn` /
`play.py learn-watch`). The demo train_loop() below scales net (channels/blocks),
MCTS sims (~800), games/iter on a GPU; see ../README.md "Path to AlphaZero".
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import chess

from .encoding import board_to_planes, move_to_index, N_POLICY, N_INPUT_PLANES
from .net import AlphaZeroNet
from .mcts import MCTS

# Hard guard: importing this module must NEVER start training.
RUN_TRAINING = False


@dataclass
class Sample:
    planes: np.ndarray      # (N_INPUT_PLANES, 8, 8)
    pi: np.ndarray          # (N_POLICY,) MCTS visit distribution
    player: bool            # whose move it was (for z sign assignment)
    z: float = 0.0          # filled in after the game ends


# --------------------------------------------------------------------------- #
# Self-play (data generation)
# --------------------------------------------------------------------------- #
def _visits_to_pi(board: chess.Board, visits: dict) -> np.ndarray:
    """Turn MCTS root visit counts into a full (N_POLICY,) target distribution."""
    pi = np.zeros(N_POLICY, dtype=np.float32)
    total = sum(visits.values())
    if total == 0:
        return pi
    for mv, n in visits.items():
        idx = move_to_index(board, mv)
        if idx is not None:
            pi[idx] = n / total
    return pi


def generate_selfplay_game(net, n_simulations: int = 100,
                           temp_moves: int = 30, max_moves: int = 200,
                           device=None) -> List[Sample]:
    """Play ONE self-play game; return the list of training Samples with z filled.

    temp_moves: first N plies sample with temperature=1 (exploration); after that
    play greedily (temperature->0). This is the AlphaZero self-play schedule.
    """
    mcts = MCTS(net, n_simulations=n_simulations, device=device)
    board = chess.Board()
    samples: List[Sample] = []
    ply = 0

    while not board.is_game_over(claim_draw=True) and ply < max_moves:
        temperature = 1.0 if ply < temp_moves else 0.0
        visits = mcts.run(board, add_noise=(temperature > 0))
        pi = _visits_to_pi(board, visits)
        samples.append(Sample(planes=board_to_planes(board), pi=pi,
                              player=board.turn))
        move = mcts.best_move(board, temperature=temperature)
        board.push(move)
        ply += 1

    # Assign z from each sample's player's perspective.
    result = board.result(claim_draw=True)
    if result == "1-0":
        winner = chess.WHITE
    elif result == "0-1":
        winner = chess.BLACK
    else:
        winner = None
    for s in samples:
        if winner is None:
            s.z = 0.0
        else:
            s.z = 1.0 if s.player == winner else -1.0
    return samples


# --------------------------------------------------------------------------- #
# Training step (the AlphaZero loss)
# --------------------------------------------------------------------------- #
def train_step(net, optimizer, batch: List[Sample], device, l2: float = 1e-4,
               value_loss_weight: float = 1.0, grad_clip: float = 5.0):
    """One optimisation step. Returns (total_loss, policy_loss, value_loss, grad_norm) floats.

    Implements L = CE(p, pi) + value_loss_weight*MSE(v, z) + L2 (L2 is folded into the
    optimizer's weight_decay; we report the data-loss components here).

    NUMERICAL SAFETY (2026-06-09 pre-restart audit): a SINGLE non-finite loss, if backpropped,
    silently corrupts the net to a uniform-policy state and poisons every subsequent self-play
    sample. So before backward we GUARD: if the loss is not finite, we SKIP the step (no
    backward/step) and return total_loss=NaN so the caller can count + skip it. We also CLIP the
    global gradient norm (one outlier batch can otherwise corrupt weights for many steps) and
    return the pre-clip grad_norm for monitoring (a 10x spike predicts collapse). value_loss_weight
    (<=1) keeps the large early value-MSE from drowning the policy head (E3 value-overfit)."""
    import torch
    import torch.nn.functional as F

    net.train()
    planes = torch.as_tensor(np.stack([s.planes for s in batch]),
                             dtype=torch.float32, device=device)
    target_pi = torch.as_tensor(np.stack([s.pi for s in batch]),
                                dtype=torch.float32, device=device)
    target_z = torch.as_tensor(np.array([s.z for s in batch], dtype=np.float32),
                               device=device).unsqueeze(1)

    logits, value = net(planes)
    log_p = F.log_softmax(logits, dim=1)
    # Cross-entropy against the (soft) MCTS target distribution.
    policy_loss = -(target_pi * log_p).sum(dim=1).mean()
    value_loss = F.mse_loss(value, target_z)
    loss = policy_loss + value_loss_weight * value_loss

    # GUARD: never backprop a non-finite loss (silent weight corruption). Skip the step.
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return float("nan"), float("nan"), float("nan"), float("nan")

    optimizer.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip))
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item()), grad_norm


# --------------------------------------------------------------------------- #
# The outer curriculum (guarded -- does NOT run on import)
# --------------------------------------------------------------------------- #
def train_loop(iterations: int = 1, games_per_iter: int = 10,
               n_simulations: int = 100, batch_size: int = 64,
               train_steps_per_iter: int = 50, buffer_size: int = 20000,
               channels: int = 64, n_blocks: int = 6, lr: float = 1e-3,
               ckpt_dir: str = "az_checkpoints"):
    """The full self-play <-> train curriculum. GUARDED by RUN_TRAINING.

    Pseudocode realised here:
        net = AlphaZeroNet(...)
        for it in iterations:
            for g in games_per_iter:   buffer += generate_selfplay_game(net)
            for s in train_steps:      train_step(net, sampled batch from buffer)
            save checkpoint
    """
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    buffer: deque = deque(maxlen=buffer_size)
    os.makedirs(ckpt_dir, exist_ok=True)

    for it in range(iterations):
        # --- self-play ---
        for _ in range(games_per_iter):
            buffer.extend(generate_selfplay_game(net, n_simulations=n_simulations,
                                                 device=device))
        # --- train ---
        if len(buffer) >= batch_size:
            for _ in range(train_steps_per_iter):
                idx = np.random.randint(0, len(buffer), size=batch_size)
                batch = [buffer[i] for i in idx]
                loss, pl, vl, _gn = train_step(net, optimizer, batch, device)
            print(f"[iter {it}] buffer={len(buffer)} "
                  f"loss={loss:.4f} (policy={pl:.4f} value={vl:.4f})")
        # --- checkpoint ---
        torch.save(net.state_dict(), os.path.join(ckpt_dir, f"net_iter{it}.pt"))
    return net


if __name__ == "__main__":
    if RUN_TRAINING:
        # Intentionally not the default. A real run needs a GPU + hours.
        train_loop(iterations=2, games_per_iter=4, n_simulations=50,
                   train_steps_per_iter=20)
    else:
        # Default: prove the pieces import + a single train_step runs on tiny data.
        import torch
        print("RUN_TRAINING=False -> NOT launching the curriculum (by design).")
        print("Smoke: one self-play game (untrained, 8 sims) + one train_step...")
        net = AlphaZeroNet(channels=16, n_blocks=2)
        samples = generate_selfplay_game(net, n_simulations=8, max_moves=12)
        print(f"  self-play produced {len(samples)} training samples")
        if len(samples) >= 4:
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            loss, pl, vl, gn = train_step(net, opt, samples[:4], torch.device("cpu"))
            print(f"  train_step: loss={loss:.4f} policy={pl:.4f} value={vl:.4f} grad_norm={gn:.3f}")
