#!/usr/bin/env python
"""Run the ATARI engine on its own, rendered. Real MinAtar games (DQN) by default.

    python play_atari.py                       # real MinAtar Breakout (default)
    python play_atari.py --game all            # all 3 trained MinAtar games in sequence
    python play_atari.py --game space_invaders # or breakout / asterix
    python play_atari.py --mode catch          # the MuZero variant on CatchEnv (learned-model planning)
    python play_atari.py --fast                # quick look

The Atari agent genuinely learned 3 real MinAtar games (the standard scaled-Atari benchmark):
Breakout ~300x random, Space Invaders ~12x, Asterix ~2.9x. The --mode catch variant is a MuZero
agent planning entirely over a LEARNED model (0 env steps in the search). No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_engines import (  # noqa: E402
    play_atari, play_atari_minatar, _available_minatar_games, _resolve_device, make_web_viz,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Atari engine (real MinAtar via DQN, or MuZero/CatchEnv).")
    ap.add_argument("--games", type=int, default=1, help="episodes to play (default 1)")
    ap.add_argument("--delay", type=float, default=0.3, help="per-step sleep for watchability (default 0.3)")
    ap.add_argument("--fast", action="store_true", help="delay=0 + short episode (quick look)")
    ap.add_argument("--no-render", action="store_true", help="result only (no animated grid)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--mode", choices=["minatar", "catch"], default="minatar",
                    help="minatar = real MinAtar games via DQN (default); catch = MuZero on CatchEnv")
    ap.add_argument("--game", choices=["breakout", "space_invaders", "asterix", "all"], default="breakout",
                    help="which real MinAtar game (default breakout), or 'all' for every trained one")
    ap.add_argument("--web", action="store_true",
                    help="ALSO open a live browser visualizer (real graphics; self-contained, no server)")
    args = ap.parse_args()

    dev = _resolve_device(args.device)
    delay = 0.0 if args.fast else args.delay
    render = not args.no_render
    viz = make_web_viz("ATARI -- trained agent playing live") if args.web else None

    if args.mode == "catch":
        play_atari(games=max(1, args.games), delay=delay, render=render, device=dev,
                   mcts_sims=8 if args.fast else 24, viz=viz)
        return 0

    avail = _available_minatar_games() or ["breakout"]
    selected = avail if args.game == "all" else [args.game if args.game in avail else avail[0]]
    steps = 40 if args.fast else 300
    for g in selected:
        play_atari_minatar(games=max(1, args.games), delay=delay, render=render, device=dev,
                           max_steps=steps, game=g, viz=viz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
