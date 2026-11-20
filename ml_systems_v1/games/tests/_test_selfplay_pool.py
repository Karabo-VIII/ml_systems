"""Regression test for the MULTIPROCESS self-play engine (selfplay_pool.py) -- the 13.5x throughput lever.

Locks in: (1) generate_games_parallel (SELF games) across workers -> valid games faster than 1 worker; (2)
generate_teacher_games_parallel (NET-vs-CLASSICAL games) -> valid games; (3) the n_workers<=1 fallback. Samples must
carry valid planes (19,8,8), a normalized pi, and a signed z in {-1,0,1}.

MUST be run as a real module / file (NOT stdin) -- Windows 'spawn' workers re-import the entry; a stdin/heredoc
script has no importable __file__ and the workers HANG. Run:
  .venv/Scripts/python.exe -m az._test_selfplay_pool
No emoji (Windows cp1252).
"""
from __future__ import annotations

import sys
import time


def _valid(games, n_policy):
    if not games:
        return False
    for g in games:
        if len(g) < 1:
            return False
        for s in g:
            if s.planes.shape != (19, 8, 8):
                return False
            if s.pi.shape != (n_policy,):
                return False
            ps = float(s.pi.sum())
            if not (abs(ps - 1.0) < 1e-3 or ps == 0.0):
                return False
            if s.z not in (-1.0, 0.0, 1.0):
                return False
    return True


def main() -> int:
    from az.net import AlphaZeroNet
    from az.encoding import N_POLICY
    from az.selfplay_pool import generate_games_parallel, generate_teacher_games_parallel
    net = AlphaZeroNet(channels=80, n_blocks=8).eval()
    fails = []

    def ok(c, label, detail=""):
        print(("  PASS" if c else "  FAIL"), label, ("" if c else f":: {detail}"))
        if not c:
            fails.append(label)

    cfg = dict(sims=24, temp_moves=6, max_plies=50, game_wall_s=40.0)

    # 1) self-play pool: a few workers -> valid games
    g = generate_games_parallel(net, n_games=6, n_workers=3, seed_base=1, **cfg)
    ok(len(g) == 6 and _valid(g, N_POLICY), "self-play pool (6 games / 3 workers) -> 6 valid games", len(g))

    # 2) single-worker fallback path -> valid games (no pool)
    g1 = generate_games_parallel(net, n_games=2, n_workers=1, seed_base=2, **cfg)
    ok(len(g1) == 2 and _valid(g1, N_POLICY), "self-play 1-worker fallback -> 2 valid games", len(g1))

    # 3) teacher pool: net vs classical -> valid games (the quality path)
    gt = generate_teacher_games_parallel(net, n_games=4, n_workers=2, teacher_depth=2, seed_base=3, **cfg)
    ok(len(gt) == 4 and _valid(gt, N_POLICY), "teacher pool (4 games / 2 workers) -> 4 valid games", len(gt))

    print("-" * 70)
    if fails:
        print("RESULT: FAILED --", ", ".join(fails))
        return 1
    print("RESULT: ALL PASS -- multiprocess self-play + teacher pools produce valid games (engine locked in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
