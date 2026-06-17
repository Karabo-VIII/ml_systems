"""
chess_zero.az.train_connect4_gpu -- HARD GPU self-play training for Connect-4, to GENUINE tactical
competence (beats the 1-ply win/block heuristic by a clear margin), and a turnkey CHAMPION checkpoint.

This is the headline GPU driver (NOT the <240s CI gate in _test_connect4.py). It uses the
GPU-exploiting parallel self-play (connect4.selfplay_games_parallel: N games in lockstep, leaf evals
pooled across games into one net.predict_many) so the RTX 4060 is actually saturated -- measured ~9x
faster per game than CPU at 256 parallel games. A bigger net (C=64/B=5) + high sim count are what buy
real win/block tactics, not just beat-random.

CHAMPION-GATE training: every iteration we periodically eval vs random AND vs the 1-ply heuristic; the
BEST-vs-heuristic net seen is kept and saved (so a late noisy iter cannot regress the saved artifact).

Saved checkpoint format (projects/chess_zero/az/checkpoints/connect4.pt):
    torch.save({"state_dict": net.state_dict(),
                "arch": {<Connect4Net ctor kwargs>},
                "meta": {"vs_random": "W..D..L..", "vs_heuristic": "W..D..L..", "iters": N, ...}}, path)
Loader contract:  net = Connect4Net(**ckpt["arch"]); net.load_state_dict(ckpt["state_dict"]).

Run:  .venv/Scripts/python.exe -m az.train_connect4_gpu [--iters N ...]
No emoji (Windows cp1252). ADDITIVE (new file; touches no existing module).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .connect4 import train_connect4, eval_wdl_vs_random, eval_wdl_vs_heuristic
from .net import Connect4Net, count_params

_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPT_DIR = os.path.join(_HERE, "checkpoints")
_CKPT_PATH = os.path.join(_CKPT_DIR, "connect4.pt")
_LOG_PATH = os.path.join(_HERE, "connect4_gpu_train.log")


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _wdl_str(w: int, d: int, l: int) -> str:
    return f"W{w}D{d}L{l}"


def main() -> int:
    ap = argparse.ArgumentParser(description="HARD GPU Connect-4 self-play -> champion checkpoint.")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--games-per-iter", type=int, default=256)
    ap.add_argument("--parallel-games", type=int, default=256)
    ap.add_argument("--sims", type=int, default=96)
    ap.add_argument("--eval-sims", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-epochs", type=int, default=3)
    ap.add_argument("--train-minibatch", type=int, default=256)
    ap.add_argument("--buffer-iters", type=int, default=5)
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=3,
                    help="train this many iters per train_connect4 call (champion-gate cadence).")
    ap.add_argument("--target-nonloss-heuristic", type=float, default=0.60)
    args = ap.parse_args()

    os.makedirs(_CKPT_DIR, exist_ok=True)
    open(_LOG_PATH, "w", encoding="utf-8").close()  # fresh log

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = dict(channels=args.channels, n_blocks=args.blocks,
                n_input_planes=3, n_policy=7, rows=6, cols=7)
    net = Connect4Net(**arch).to(device)
    _log(f"device={device}  net=Connect4Net(C={args.channels},B={args.blocks}) "
         f"params={count_params(net):,}")
    _log(f"plan: {args.iters} iters x {args.games_per_iter} games x {args.sims} sims "
         f"(parallel={args.parallel_games}), eval every {args.eval_every} train-iters")

    best_heur_nonloss = -1.0
    best_meta = None
    curve = []  # (iter_done, vs_random WDL, vs_heuristic WDL, nonloss_heur)
    t_start = time.perf_counter()

    iters_done = 0
    while iters_done < args.iters:
        chunk = min(args.chunk, args.iters - iters_done)
        # train this chunk (no in-train eval; we eval explicitly below for both opponents)
        net, _m = train_connect4(
            net, n_iters=chunk, games_per_iter=args.games_per_iter, sims=args.sims,
            lr=args.lr, seed=args.seed + iters_done, eval_games=0,
            opening_plies=args.opening_plies, train_epochs=args.train_epochs,
            train_minibatch=args.train_minibatch, buffer_iters=args.buffer_iters,
            verbose=False, device=device, parallel_games=args.parallel_games,
        )
        iters_done += chunk

        # --- champion-gate eval: vs random AND vs heuristic ---
        rw, rd, rl = eval_wdl_vs_random(net, n_games=args.eval_games, sims=args.eval_sims,
                                        seed=7000 + iters_done, device=device)
        hw, hd, hl = eval_wdl_vs_heuristic(net, n_games=args.eval_games, sims=args.eval_sims,
                                           seed=9000 + iters_done, device=device)
        rand_nl = (rw + rd) / args.eval_games
        heur_nl = (hw + hd) / args.eval_games
        elapsed = time.perf_counter() - t_start
        _log(f"iter {iters_done:2d}/{args.iters}  vs_random {_wdl_str(rw, rd, rl)} "
             f"(non-loss {rand_nl:.3f})  vs_heuristic {_wdl_str(hw, hd, hl)} "
             f"(non-loss {heur_nl:.3f})  [{elapsed:.0f}s]")
        curve.append({"iter": iters_done, "vs_random": _wdl_str(rw, rd, rl),
                      "vs_heuristic": _wdl_str(hw, hd, hl),
                      "rand_nonloss": round(rand_nl, 3), "heur_nonloss": round(heur_nl, 3)})

        # keep the BEST-vs-heuristic net (tie-break by vs-random non-loss)
        score = heur_nl + 0.01 * rand_nl
        if score > best_heur_nonloss:
            best_heur_nonloss = score
            best_meta = {
                "vs_random": _wdl_str(rw, rd, rl), "vs_heuristic": _wdl_str(hw, hd, hl),
                "rand_nonloss": round(rand_nl, 3), "heur_nonloss": round(heur_nl, 3),
                "iters": iters_done, "eval_games": args.eval_games, "eval_sims": args.eval_sims,
                "train_sims": args.sims, "games_per_iter": args.games_per_iter,
                "parallel_games": args.parallel_games,
                "arch": arch, "device": str(device),
            }
            torch.save({"state_dict": net.state_dict(), "arch": arch, "meta": best_meta}, _CKPT_PATH)
            _log(f"  -> NEW CHAMPION saved (heur non-loss {heur_nl:.3f}): {_CKPT_PATH}")

        # early-stop once we clearly clear the target (and crush random)
        if heur_nl >= args.target_nonloss_heuristic and rand_nl >= 0.95 and iters_done >= 8:
            _log(f"  TARGET MET: vs_heuristic non-loss {heur_nl:.3f} >= "
                 f"{args.target_nonloss_heuristic} AND vs_random non-loss {rand_nl:.3f} >= 0.95.")
            # keep training a couple more chunks for headroom, then stop
            if heur_nl >= args.target_nonloss_heuristic + 0.10:
                break

    total = time.perf_counter() - t_start
    _log("=" * 60)
    _log(f"DONE in {total:.0f}s. CHAMPION: {best_meta['vs_heuristic']} vs heuristic, "
         f"{best_meta['vs_random']} vs random at iter {best_meta['iters']}.")
    _log(f"checkpoint: {_CKPT_PATH}")
    with open(os.path.join(_HERE, "connect4_gpu_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"curve": curve, "champion": best_meta}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
