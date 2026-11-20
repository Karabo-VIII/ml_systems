#!/usr/bin/env python
"""Run the CHESS engine on its own, rendered move-by-move.

    python play_chess.py --strong         # the STRONG ~1600 classical engine vs Stockfish (the best)
    python play_chess.py                   # champion SELF-PLAY (the net plays itself; default)
    python play_chess.py --vs-classical   # champion vs the in-repo classical engine
    python play_chess.py --games 2 --delay 0.6
    python play_chess.py --fast           # no delay (quick look)

HONEST: the strongest chess here is the CLASSICAL engine (negamax + evaluate_v2 -- MEASURED ~1600
Elo); use --strong to watch it compete against Stockfish. The AlphaZero champion crushes random but
is a weak-but-real learner vs a classical minimax (chess mastery is compute-bound). No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_engines import play_chess, play_chess_strong, _resolve_device, make_web_viz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the chess engine (strong classical vs Stockfish, "
                                             "or the AlphaZero net self-play / vs classical).")
    ap.add_argument("--games", type=int, default=1, help="games to play (default 1)")
    ap.add_argument("--delay", type=float, default=0.4, help="per-move sleep for watchability (default 0.4)")
    ap.add_argument("--fast", action="store_true", help="delay=0 + low sims (quick look)")
    ap.add_argument("--no-render", action="store_true", help="result only (no animated board)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--strong", action="store_true",
                    help="showcase the STRONG ~1600 classical engine vs Stockfish (the best); "
                         "falls back to strong self-play if no Stockfish binary is present")
    ap.add_argument("--sf-elo", type=int, default=1500, help="Stockfish Elo cap for --strong (1320-3190)")
    ap.add_argument("--movetime", type=float, default=0.5, help="seconds/move for --strong engines")
    ap.add_argument("--vs-classical", action="store_true",
                    help="play vs the classical negamax engine instead of self-play")
    ap.add_argument("--depth", type=int, default=2, help="classical engine depth (with --vs-classical)")
    ap.add_argument("--web", action="store_true",
                    help="ALSO open a live browser visualizer (real piece graphics; self-contained, no server)")
    args = ap.parse_args()

    if args.strong:
        play_chess_strong(
            games=max(1, args.games),
            delay=0.0 if args.fast else args.delay,
            render=not args.no_render,
            movetime=args.movetime,
            sf_elo=args.sf_elo,
            viz=make_web_viz("CHESS -- strong classical engine vs Stockfish") if args.web else None,
        )
        return 0

    play_chess(
        games=max(1, args.games),
        delay=0.0 if args.fast else args.delay,
        render=not args.no_render,
        device=_resolve_device(args.device),
        mcts_sims=8 if args.fast else 64,
        mode="vs-classical" if args.vs_classical else "selfplay",
        chess_depth=args.depth,
        viz=make_web_viz("CHESS -- AlphaZero champion") if args.web else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
