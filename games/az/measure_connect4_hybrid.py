"""
Measure the net+solver HYBRID vs the bare trained net at Connect-4 (the integrity step before
shipping the hybrid as the engine). Paired, colour-swapped games from short random openings so the
result is not opening-luck. Reports W/D/L from the HYBRID's POV + how often each hybrid layer fired.

  python -m az.measure_connect4_hybrid --pairs 15 --budget 0.6 --sims 128

No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import os
import random
import time

import torch

from az.connect4 import Connect4
from az.connect4_hybrid import HybridConnect4Player
from az.mcts import NeuralMCTS
from az.net import Connect4Net

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(_HERE, "checkpoints", "connect4.pt")


def load_net(device: str):
    net = Connect4Net(channels=64, n_blocks=5, n_input_planes=3, n_policy=7, rows=6, cols=7).to(device)
    tag = "UNTRAINED"
    if os.path.exists(CKPT):
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        net.load_state_dict(ck["state_dict"], strict=False)
        tag = f"trained ({ck.get('meta', {})})"
    net.eval()
    return net, tag


def random_opening(game, k: int, rng: random.Random):
    """k random legal plies from the start; abort if it ends the game."""
    state = game.initial_state()
    for _ in range(k):
        if game.is_terminal(state):
            break
        a = rng.choice(game.legal_actions(state))
        state = game.apply(state, a)
    return state


def net_action(game, net, device, sims, state):
    a = NeuralMCTS(game, net, n_simulations=sims, device=device).best_action_batched(
        state, temperature=0.0, batch_size=16)
    if a not in game.legal_actions(state):
        a = game.legal_actions(state)[0]
    return a


def play(game, start_state, hybrid, net, device, sims, hybrid_is_p0: bool, max_plies: int = 60):
    state = start_state
    plies = 0
    while not game.is_terminal(state) and plies < max_plies:
        p = game.current_player(state)
        hybrid_to_move = (p == 0) == hybrid_is_p0
        a = hybrid.action(state) if hybrid_to_move else net_action(game, net, device, sims, state)
        if a not in game.legal_actions(state):
            a = game.legal_actions(state)[0]
        state = game.apply(state, a)
        plies += 1
    z = game.returns(state)  # +1 p0, -1 p1, 0 draw
    if z == 0:
        return "draw"
    p0_won = z > 0
    return "hybrid" if (p0_won == hybrid_is_p0) else "net"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=15, help="colour-swapped opening pairs (2 games each)")
    ap.add_argument("--budget", type=float, default=0.6, help="solver per-move budget (s)")
    ap.add_argument("--sims", type=int, default=128, help="net MCTS sims/move")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = Connect4()
    net, tag = load_net(device)
    print(f"[measure] device={device} net={tag}")
    print(f"[measure] HYBRID(net+solver, budget={args.budget}s) vs bare net (sims={args.sims}) | "
          f"{args.pairs} colour-swapped pairs = {2 * args.pairs} games", flush=True)

    rng = random.Random(args.seed)
    hw = hd = hl = 0
    counts = {"win": 0, "block": 0, "proven": 0, "net": 0}
    t0 = time.perf_counter()
    for i in range(args.pairs):
        opening = random_opening(game, args.opening_plies, rng)
        for hybrid_is_p0 in (True, False):
            hybrid = HybridConnect4Player(game, net, device=device, mcts_sims=args.sims,
                                          budget_s=args.budget)
            r = play(game, opening, hybrid, net, device, args.sims, hybrid_is_p0)
            for k in counts:
                counts[k] += hybrid.counts[k]
            if r == "draw":
                hd += 1
            elif r == "hybrid":
                hw += 1
            else:
                hl += 1
            print(f"  pair {i + 1}/{args.pairs} hybrid={'p0' if hybrid_is_p0 else 'p1'} -> {r}  "
                  f"(hybrid W{hw} D{hd} L{hl})", flush=True)

    n = 2 * args.pairs
    score = (hw + 0.5 * hd) / n if n else 0.0
    print(f"\n[measure] RESULT hybrid vs net: W{hw} D{hd} L{hl} over {n}  score={score:.3f}", flush=True)
    print(f"[measure] hybrid layer firings: {counts}  "
          f"(proven+win+block = perfect/forced moves; net = learned-opening moves)", flush=True)
    print(f"[measure] {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
