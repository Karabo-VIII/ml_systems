"""
chess_zero.az.train_demo -- a bounded, time-boxed AlphaZero self-play -> train
loop that DEMONSTRATES LEARNING (loss trends down; the net moves off random),
NOT strength. A strong AZ needs days-weeks on this GPU; this proves the PIPELINE
learns within ~30-40 minutes on an RTX 4060.

Each iteration:
    (a) SELF-PLAY a few games with MCTS + the current net  -> (planes, pi, z) data
    (b) TRAIN the net on the replay buffer for a few steps  (AlphaZero loss =
        CE(policy, MCTS-visit-pi) + MSE(value, game-outcome z)), logging the
        policy_loss + value_loss components
    (c) EVAL the net (low-sim MCTS, greedy) vs a RANDOM mover for a handful of
        games -> win/draw/loss + win-rate

Outputs:
    train_metrics.json  -- per-iter loss + win-rate + config + provenance
    az_demo_checkpoints/net_iter{it}.pt  -- the net checkpoint each iteration

The loss going DOWN across iterations is the primary learning signal. The
win-rate-vs-random is secondary and may be noisy with a tiny net + few iters --
we report it honestly either way.

Run:
    python -m chess_zero.az.train_demo
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import chess
import torch

from .encoding import N_INPUT_PLANES, N_POLICY
from .net import AlphaZeroNet, count_params
from .mcts import MCTS
from .selfplay import Sample, generate_selfplay_game, train_step


# --------------------------------------------------------------------------- #
# Config (DELIBERATELY SMALL + TIME-BOXED for the ~30-40 min demo)
# --------------------------------------------------------------------------- #
@dataclass
class DemoConfig:
    # net (scaled down from the paper's 19-20 blocks x 256 filters)
    channels: int = 32
    n_blocks: int = 4
    # self-play
    iterations: int = 4
    games_per_iter: int = 12
    selfplay_sims: int = 24          # MCTS sims/move during self-play
    temp_moves: int = 15             # plies sampled w/ temperature=1 (exploration)
    max_plies: int = 80              # short games (cap) to keep the run bounded
    # training
    train_steps_per_iter: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    l2: float = 1e-4
    buffer_size: int = 20000
    # evaluation vs a random mover
    eval_games: int = 14
    eval_sims: int = 16              # low-sim MCTS for the net at eval time
    # io
    ckpt_dir: str = "az_demo_checkpoints"
    metrics_path: str = "train_metrics.json"
    seed: int = 0


# --------------------------------------------------------------------------- #
# Evaluation: net (low-sim MCTS, greedy) vs a RANDOM-legal-move opponent
# --------------------------------------------------------------------------- #
def _random_move(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    moves = list(board.legal_moves)
    return moves[int(rng.integers(len(moves)))]


def play_vs_random(net, n_games: int, eval_sims: int, max_plies: int,
                   device, rng: np.random.Generator) -> Dict[str, int]:
    """Play `n_games` of (net) vs (random mover). The net plays greedily (argmax
    of MCTS visit counts, no exploration noise). Net alternates colour each game
    so the result is not confounded by the first-move advantage.

    Returns {"win", "draw", "loss", "games"} from the NET's perspective.
    """
    mcts = MCTS(net, n_simulations=eval_sims, device=device)
    wins = draws = losses = 0
    for g in range(n_games):
        net_is_white = (g % 2 == 0)
        board = chess.Board()
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            net_to_move = (board.turn == chess.WHITE) == net_is_white
            if net_to_move:
                move = mcts.best_move(board, temperature=0.0)  # greedy
            else:
                move = _random_move(board, rng)
            board.push(move)
            ply += 1

        result = board.result(claim_draw=True)
        if result == "1-0":
            winner = chess.WHITE
        elif result == "0-1":
            winner = chess.BLACK
        else:
            winner = None
        if winner is None:
            draws += 1
        elif (winner == chess.WHITE) == net_is_white:
            wins += 1
        else:
            losses += 1
    return {"win": wins, "draw": draws, "loss": losses, "games": n_games}


# --------------------------------------------------------------------------- #
# The bounded demo loop
# --------------------------------------------------------------------------- #
def run_demo(cfg: DemoConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[device] {device} ({dev_name})")

    here = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(here, cfg.ckpt_dir)
    metrics_path = os.path.join(here, cfg.metrics_path)
    os.makedirs(ckpt_dir, exist_ok=True)

    net = AlphaZeroNet(channels=cfg.channels, n_blocks=cfg.n_blocks).to(device)
    n_params = count_params(net)
    print(f"[net] channels={cfg.channels} blocks={cfg.n_blocks} "
          f"params={n_params:,}  input_planes={N_INPUT_PLANES} n_policy={N_POLICY}")
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.l2)

    buffer: List[Sample] = []

    # ---- iteration 0 baseline: eval the UNTRAINED net vs random ---- #
    print("\n[iter -1] baseline eval (UNTRAINED net) vs random ...")
    t0 = time.time()
    base_eval = play_vs_random(net, cfg.eval_games, cfg.eval_sims,
                               cfg.max_plies, device, rng)
    base_wr = base_eval["win"] / base_eval["games"]
    print(f"           untrained vs random: {base_eval}  win_rate={base_wr:.3f}  "
          f"({time.time()-t0:.1f}s)")

    metrics: dict = {
        "config": asdict(cfg),
        "device": str(device),
        "device_name": dev_name,
        "n_params": n_params,
        "n_input_planes": N_INPUT_PLANES,
        "n_policy": N_POLICY,
        "baseline_untrained": {**base_eval, "win_rate": base_wr},
        "iters": [],
    }

    wall_start = time.time()
    total_examples = 0
    first_loss = None
    last_loss = None

    for it in range(cfg.iterations):
        it_t0 = time.time()

        # ---- (a) SELF-PLAY ---- #
        sp_t0 = time.time()
        new_examples = 0
        net.eval()
        for g in range(cfg.games_per_iter):
            samples = generate_selfplay_game(
                net, n_simulations=cfg.selfplay_sims,
                temp_moves=cfg.temp_moves, max_moves=cfg.max_plies,
                device=device)
            buffer.extend(samples)
            new_examples += len(samples)
        # cap the buffer
        if len(buffer) > cfg.buffer_size:
            buffer = buffer[-cfg.buffer_size:]
        total_examples += new_examples
        sp_dt = time.time() - sp_t0

        # ---- (b) TRAIN ---- #
        tr_t0 = time.time()
        step_losses: List[Tuple[float, float, float]] = []
        if len(buffer) >= cfg.batch_size:
            net.train()
            for _ in range(cfg.train_steps_per_iter):
                idx = rng.integers(0, len(buffer), size=cfg.batch_size)
                batch = [buffer[i] for i in idx]
                loss, pl, vl, _gn = train_step(net, optimizer, batch, device, l2=cfg.l2)
                step_losses.append((loss, pl, vl))
        tr_dt = time.time() - tr_t0

        if step_losses:
            # report first-step and mean-of-last-10-steps to show the in-iter trend
            first_step = step_losses[0]
            tail = step_losses[-min(10, len(step_losses)):]
            mean_tail = tuple(float(np.mean([s[k] for s in tail])) for k in range(3))
            iter_loss = mean_tail[0]
            iter_pl = mean_tail[1]
            iter_vl = mean_tail[2]
        else:
            first_step = (float("nan"),) * 3
            iter_loss = iter_pl = iter_vl = float("nan")

        if first_loss is None and step_losses:
            first_loss = first_step[0]
        if step_losses:
            last_loss = iter_loss

        # ---- (c) EVAL vs random ---- #
        ev_t0 = time.time()
        net.eval()
        ev = play_vs_random(net, cfg.eval_games, cfg.eval_sims,
                            cfg.max_plies, device, rng)
        wr = ev["win"] / ev["games"]
        ev_dt = time.time() - ev_t0

        # ---- checkpoint ---- #
        ckpt_path = os.path.join(ckpt_dir, f"net_iter{it}.pt")
        torch.save({"state_dict": net.state_dict(),
                    "channels": cfg.channels, "n_blocks": cfg.n_blocks,
                    "iter": it}, ckpt_path)

        it_dt = time.time() - it_t0
        rec = {
            "iter": it,
            "buffer_size": len(buffer),
            "new_examples": new_examples,
            "total_examples": total_examples,
            "train_steps": len(step_losses),
            "first_step_loss": {"total": first_step[0], "policy": first_step[1],
                                "value": first_step[2]},
            "iter_loss_mean_last10": {"total": iter_loss, "policy": iter_pl,
                                      "value": iter_vl},
            "eval_vs_random": {**ev, "win_rate": wr},
            "timing_s": {"selfplay": round(sp_dt, 1), "train": round(tr_dt, 1),
                         "eval": round(ev_dt, 1), "iter_total": round(it_dt, 1)},
            "checkpoint": os.path.relpath(ckpt_path, here),
        }
        metrics["iters"].append(rec)

        print(f"\n[iter {it}] buffer={len(buffer)} new={new_examples} "
              f"steps={len(step_losses)}")
        print(f"           loss first-step total={first_step[0]:.4f} "
              f"(policy={first_step[1]:.4f} value={first_step[2]:.4f})")
        print(f"           loss last-10-mean total={iter_loss:.4f} "
              f"(policy={iter_pl:.4f} value={iter_vl:.4f})")
        print(f"           eval vs random: W{ev['win']} D{ev['draw']} L{ev['loss']} "
              f"win_rate={wr:.3f}")
        print(f"           timing: selfplay={sp_dt:.1f}s train={tr_dt:.1f}s "
              f"eval={ev_dt:.1f}s iter={it_dt:.1f}s")

        # incremental write so partial progress is never lost
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    wall = time.time() - wall_start
    metrics["summary"] = {
        "total_wall_s": round(wall, 1),
        "total_wall_min": round(wall / 60, 2),
        "total_examples": total_examples,
        "first_iter_loss": first_loss,
        "last_iter_loss": last_loss,
        "loss_decreased": (first_loss is not None and last_loss is not None
                           and last_loss < first_loss),
        "baseline_win_rate": base_wr,
        "final_win_rate": (metrics["iters"][-1]["eval_vs_random"]["win_rate"]
                           if metrics["iters"] else None),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 64)
    print(f"[done] wall={wall/60:.1f} min  examples={total_examples}  device={device}")
    print(f"[done] loss: first-iter {first_loss:.4f} -> last-iter {last_loss:.4f}  "
          f"(decreased={metrics['summary']['loss_decreased']})")
    print(f"[done] win-rate vs random: baseline {base_wr:.3f} -> "
          f"final {metrics['summary']['final_win_rate']:.3f}")
    print(f"[done] metrics -> {os.path.relpath(metrics_path, here)}")
    print("=" * 64)
    return metrics


if __name__ == "__main__":
    run_demo(DemoConfig())
