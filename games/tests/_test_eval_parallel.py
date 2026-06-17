"""RWYB probe for PARALLEL EVALUATION (2026-06-08): _play_match(n_workers>1) splits the games across
CPU worker processes (each runs the verified _play_match sequentially on its chunk) and aggregates.

Asserts:
  1. play_match_parallel returns a valid aggregated dict over exactly n_games (win+draw+loss==games,
     rates in [0,1]) -- i.e. no game is lost or double-counted across the split.
  2. _play_match(n_workers=N) delegates to the pool and returns the SAME dict shape as the sequential
     path (same keys; counts sum to n_games).
  3. The pool path no-ops below the n_games>=4 threshold (spawn overhead not worth it) -> identical to
     sequential for tiny matches.

Run from repo root:  python -m az._test_eval_parallel
CPU, tiny net, spawn workers (~15s). No emoji (Windows cp1252).
"""
from __future__ import annotations

import numpy as np
import torch

from az.train_robust import _play_match, AlphaZeroNet
from az.selfplay_pool import play_match_parallel

KEYS = {"win", "draw", "loss", "games", "win_rate", "score"}


def _valid(d, n):
    assert set(d) == KEYS, f"keys {set(d)} != {KEYS}"
    assert d["games"] == n, f"games {d['games']} != {n}"
    assert d["win"] + d["draw"] + d["loss"] == n, f"W+D+L {d['win']+d['draw']+d['loss']} != {n}"
    assert 0.0 <= d["win_rate"] <= 1.0 and 0.0 <= d["score"] <= 1.0, "rate out of [0,1]"


def main() -> int:
    torch.manual_seed(0)
    net = AlphaZeroNet(channels=8, n_blocks=1)
    net.eval()
    common = dict(eval_sims=4, max_plies=20, game_wall_s=10.0, classical_depth=1)

    # (1) direct pool call over 6 games / 3 workers
    r = play_match_parallel(net, "random", 6, n_workers=3, channels=8, n_blocks=1, **common)
    _valid(r, 6)
    print(f"[1] play_match_parallel(6 games, 3 workers) -> {r['win']}W/{r['draw']}D/{r['loss']}L OK")

    # (2) _play_match delegates when n_workers>1 and n_games>=4
    rng = np.random.default_rng(0)
    d = _play_match(net, "random", 6, 4, 20, 10.0, 1, torch.device("cpu"), rng, n_workers=3)
    _valid(d, 6)
    print(f"[2] _play_match(n_workers=3) delegated -> {d['win']}W/{d['draw']}D/{d['loss']}L OK")

    # (3) below threshold (n_games<4) -> sequential, identical structure
    rng2 = np.random.default_rng(0)
    s = _play_match(net, "random", 2, 4, 20, 10.0, 1, torch.device("cpu"), rng2, n_workers=8)
    _valid(s, 2)
    print(f"[3] _play_match(2 games, n_workers=8) no-op'd pool -> sequential {s['win']}W/{s['draw']}D/{s['loss']}L OK")

    print("\nALL EVAL-PARALLEL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
