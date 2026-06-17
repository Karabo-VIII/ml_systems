"""
chess_zero.az.eval_bootstrap -- HONEST strength evaluation of a trained AZ net.

Plays REAL games (python-chess decides every result; draws reported honestly) and
reports win/draw/loss from the NET's perspective, in FOUR conditions:

    net-only   vs random     (argmax of the masked policy head -- NO search)
    net+MCTS   vs random     (greedy MCTS with --mcts-sims simulations)
    net-only   vs classical  (Engine(depth=--classical-depth))
    net+MCTS   vs classical

This is the deliverable PROOF. The honest bar: substantially BEAT RANDOM (the
from-scratch self-play run never reliably did -- it stalled at 0.00). Any win vs
the classical depth-1 engine is real progress over that 0.00 baseline.

HONEST CEILING (binding): the net imitates the classical engine, so it APPROACHES
but should not EXCEED its teacher. No master/superhuman claims. Distinguish
net-only from net+MCTS results explicitly.

Run:
    .venv\\Scripts\\python.exe -m az.eval_bootstrap \\
        --ckpt projects/chess_zero/az/bootstrap_checkpoints/net_bootstrap.pt \\
        --games 200 --mcts-sims 32 --classical-depth 1
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import chess
import torch

from .encoding import board_to_planes, legal_policy_mask
from .net import AlphaZeroNet, count_params
from .mcts import MCTS
from chess_engine.engine import Engine


HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# Adjudication (so a capped game still has a real signed result; never hangs)
# --------------------------------------------------------------------------- #
def _material_balance(board: chess.Board) -> int:
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}
    bal = 0
    for _, piece in board.piece_map().items():
        v = vals[piece.piece_type]
        bal += v if piece.color == chess.WHITE else -v
    return bal


def _adjudicate(board: chess.Board) -> Optional[bool]:
    bal = _material_balance(board)
    if bal >= 150:
        return chess.WHITE
    if bal <= -150:
        return chess.BLACK
    return None


# --------------------------------------------------------------------------- #
# Net move selection: net-only (argmax masked policy) vs net+MCTS
# --------------------------------------------------------------------------- #
def net_only_move(net, board: chess.Board, device) -> chess.Move:
    """Greedy argmax over the LEGAL-masked policy head. No search at all."""
    planes = board_to_planes(board)
    mask, idx_to_move = legal_policy_mask(board)
    probs, _ = net.predict(planes, legal_mask=mask, device=device)
    # restrict to legal indices and take argmax
    best_idx, best_p = None, -1.0
    for idx in idx_to_move:
        if probs[idx] > best_p:
            best_p, best_idx = probs[idx], idx
    if best_idx is None:  # no encodable legal move (shouldn't happen) -> any legal
        return next(iter(board.legal_moves))
    return idx_to_move[best_idx]


def _random_move(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    moves = list(board.legal_moves)
    return moves[int(rng.integers(len(moves)))]


def play_match(net, device, opponent: str, n_games: int, net_mode: str,
               mcts_sims: int, classical_depth: int, max_plies: int,
               game_wall_s: float, rng: np.random.Generator) -> Dict[str, float]:
    """Play n_games (net vs opponent). net_mode in {'net_only','mcts'}; opponent in
    {'random','classical'}. Net alternates colour. Returns W/D/L from net POV."""
    mcts = MCTS(net, n_simulations=mcts_sims, device=device) if net_mode == "mcts" else None
    engine = Engine(depth=classical_depth) if opponent == "classical" else None
    wins = draws = losses = 0
    adjudicated = 0  # games whose result came from the +/-150cp material heuristic, NOT
                     # a real python-chess terminal result (honesty: weak nets rarely
                     # reach checkmate/stalemate, so a high adjudicated_fraction means the
                     # win-rate leans on the heuristic rather than on actual won games).
    net.eval()
    for g in range(n_games):
        net_is_white = (g % 2 == 0)
        board = chess.Board()
        ply = 0
        t0 = time.time()
        aborted = False
        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            if time.time() - t0 > game_wall_s:
                aborted = True
                break
            net_to_move = (board.turn == chess.WHITE) == net_is_white
            if net_to_move:
                if net_mode == "mcts":
                    move = mcts.best_move(board, temperature=0.0)
                else:
                    move = net_only_move(net, board, device)
            elif opponent == "random":
                move = _random_move(board, rng)
            else:
                res = engine.search(board)
                move = res.move if res.move is not None else _random_move(board, rng)
            board.push(move)
            ply += 1

        if board.is_game_over(claim_draw=True) and not aborted:
            result = board.result(claim_draw=True)
            winner = (chess.WHITE if result == "1-0"
                      else chess.BLACK if result == "0-1" else None)
        else:
            winner = _adjudicate(board)  # ply/wall-clock cap hit -> material heuristic
            adjudicated += 1
        if winner is None:
            draws += 1
        elif (winner == chess.WHITE) == net_is_white:
            wins += 1
        else:
            losses += 1
    score = wins + 0.5 * draws
    return {"win": wins, "draw": draws, "loss": losses, "games": n_games,
            "win_rate": wins / n_games if n_games else 0.0,
            "score_pct": 100.0 * score / n_games if n_games else 0.0,
            "adjudicated": adjudicated,
            "adjudicated_fraction": adjudicated / n_games if n_games else 0.0}


# --------------------------------------------------------------------------- #
# Checkpoint loading (schema-tolerant; reads channels/n_blocks from the payload)
# --------------------------------------------------------------------------- #
def load_net(ckpt_path: str, device) -> AlphaZeroNet:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    channels = ck.get("channels", 80)
    n_blocks = ck.get("n_blocks", 8)
    net = AlphaZeroNet(channels=channels, n_blocks=n_blocks).to(device)
    net.load_state_dict(ck["state_dict"], strict=False)
    net.eval()
    print(f"[load] {os.path.basename(ckpt_path)}: C={channels}/B={n_blocks} "
          f"params={count_params(net):,} source={ck.get('source','?')}", flush=True)
    return net


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Honest strength eval of a trained AZ net.")
    ap.add_argument("--ckpt", type=str, required=True, help="path to a checkpoint .pt")
    ap.add_argument("--games", type=int, default=200,
                    help="games per condition (split across colours)")
    ap.add_argument("--mcts-sims", type=int, default=32)
    ap.add_argument("--classical-depth", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--game-wall-s", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--skip-mcts", action="store_true",
                    help="net-only conditions only (faster)")
    ap.add_argument("--out", type=str, default=None, help="optional json results path")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} "
          f"({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})",
          flush=True)
    net = load_net(args.ckpt, device)
    rng = np.random.default_rng(args.seed)

    results = {}
    conditions = [("net_only", "random"), ("net_only", "classical")]
    if not args.skip_mcts:
        conditions += [("mcts", "random"), ("mcts", "classical")]

    for net_mode, opp in conditions:
        t0 = time.time()
        r = play_match(net, device, opp, args.games, net_mode, args.mcts_sims,
                       args.classical_depth, args.max_plies, args.game_wall_s, rng)
        r["wall_s"] = round(time.time() - t0, 1)
        key = f"{net_mode}_vs_{opp}"
        results[key] = r
        label = (f"{net_mode:<8} vs {opp:<9}"
                 + (f"(d={args.classical_depth})" if opp == "classical" else ""))
        print(f"[eval] {label}: W{r['win']} D{r['draw']} L{r['loss']} "
              f"/{r['games']}  win_rate={r['win_rate']:.3f}  "
              f"score={r['score_pct']:.1f}%  adj={r['adjudicated']}/{r['games']} "
              f"({r['adjudicated_fraction']*100:.0f}%)  ({r['wall_s']:.0f}s)", flush=True)

    print("\n" + "=" * 64)
    print("[SUMMARY] (net perspective; win_rate = wins/games; score = (W+0.5D)/games)")
    print("          adj_frac = fraction decided by the +/-150cp material heuristic")
    print("          (NOT a real checkmate/stalemate) -- high = win-rate is heuristic-leaning")
    for k, r in results.items():
        print(f"  {k:<24} win_rate={r['win_rate']:.3f}  score={r['score_pct']:.1f}%  "
              f"adj_frac={r['adjudicated_fraction']:.2f}  "
              f"(W{r['win']}/D{r['draw']}/L{r['loss']})")
    print("=" * 64)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ckpt)),
                                   "eval_results.json")
    payload = {"ckpt": args.ckpt, "games_per_condition": args.games,
               "mcts_sims": args.mcts_sims, "classical_depth": args.classical_depth,
               "results": results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] results -> {out}", flush=True)


if __name__ == "__main__":
    main()
