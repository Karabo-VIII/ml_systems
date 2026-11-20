#!/usr/bin/env python
"""Run the CONNECT-4 engine on its own (AlphaZero net vs a 1-ply win/block heuristic), rendered.

    python play_connect4.py               # one game, animated
    python play_connect4.py --games 3 --delay 0.6
    python play_connect4.py --fast        # no delay (quick look)

The trained net (PUCT search) is genuinely competent: ~W40/L0 vs random and ~W33/D2/L5 vs the
1-ply win/block heuristic (it started out losing 0:24 -- GPU training taught it real tactics).
No emoji (cp1252).
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_engines import play_connect4, play_connect4_human, _resolve_device, make_web_viz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the AlphaZero Connect-4 engine vs a 1-ply heuristic.")
    ap.add_argument("--games", type=int, default=1, help="games to play (default 1)")
    ap.add_argument("--delay", type=float, default=0.4, help="per-move sleep for watchability (default 0.4)")
    ap.add_argument("--fast", action="store_true", help="delay=0 + low sims (quick look)")
    ap.add_argument("--no-render", action="store_true", help="result only (no animated board)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--web", action="store_true",
                    help="ALSO open a live browser visualizer (real graphics; self-contained, no server)")
    ap.add_argument("--human", action="store_true",
                    help="PLAY vs the trained net yourself (interactive; type a column 0-6)")
    ap.add_argument("--ai-first", action="store_true",
                    help="the AI moves first, you play second (use with --human)")
    args = ap.parse_args()

    if args.human:
        play_connect4_human(
            human_first=not args.ai_first,
            device=_resolve_device(args.device),
            mcts_sims=24 if args.fast else 160,
        )
        return 0

    play_connect4(
        games=max(1, args.games),
        delay=0.0 if args.fast else args.delay,
        render=not args.no_render,
        device=_resolve_device(args.device),
        mcts_sims=24 if args.fast else 128,
        viz=make_web_viz("CONNECT-4 -- AlphaZero net") if args.web else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
