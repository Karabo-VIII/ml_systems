"""
Measure the opening BOOK head-to-head: Engine(book) vs Engine(no book), same eval, same time.
Games START from a sample of EARLY book positions (so there is book depth ahead for the booked
engine to exploit), played from both colours. Reports W/D/L + relative Elo from the book's POV.
This is the integrity step before shipping the book (like the eval A/B).

  python -m chess_engine.measure_book --starts 15 --movetime 0.2

No emoji (Windows cp1252).
"""
from __future__ import annotations

import argparse
import os
import random
import time

import chess

from chess_engine.bench import _adjudicate, elo_from_score
from chess_engine.book import OpeningBook
from chess_engine.engine import Engine

BOOK_PATH = os.path.join(os.path.dirname(__file__), "opening_book.json")


def play_from(board0: chess.Board, white: Engine, black: Engine, max_plies: int = 200) -> str:
    board = board0.copy()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = white if board.turn == chess.WHITE else black
        mv = eng.search(board).move
        if mv is None or mv not in board.legal_moves:
            mv = next(iter(board.legal_moves))
        board.push(mv)
    return (board.result(claim_draw=True)
            if board.is_game_over(claim_draw=True) else _adjudicate(board))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=15, help="distinct early book positions to start from")
    ap.add_argument("--movetime", type=float, default=0.2)
    ap.add_argument("--max-fullmove", type=int, default=4, help="only start from positions this early")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    book = OpeningBook.load(BOOK_PATH)
    if len(book) == 0:
        print(f"[measure_book] no book at {BOOK_PATH} -- generate it first (python -m chess_engine.book --generate)")
        return 1
    print(f"[measure_book] book has {len(book)} positions", flush=True)

    rng = random.Random(args.seed)
    keys = list(book.moves.keys())
    rng.shuffle(keys)
    starts = []
    seen = set()
    for k in keys:
        try:
            b = chess.Board(k + " 0 1")
        except ValueError:
            continue
        if b.is_game_over() or b.fullmove_number > args.max_fullmove:
            continue
        if k in seen:
            continue
        seen.add(k)
        starts.append(b)
        if len(starts) >= args.starts:
            break
    print(f"[measure_book] starting from {len(starts)} early book positions, both colours "
          f"= {2 * len(starts)} games @ {args.movetime}s/move", flush=True)

    def bk():
        return Engine(depth=64, time_limit=args.movetime, book=book)

    def pl():
        return Engine(depth=64, time_limit=args.movetime)

    w = d = l = 0
    t0 = time.perf_counter()
    for i, b0 in enumerate(starts):
        for book_white in (True, False):
            white = bk() if book_white else pl()
            black = pl() if book_white else bk()
            r = play_from(b0, white, black)
            if r == "1/2-1/2":
                d += 1
            else:
                book_won = (r == "1-0") == book_white
                w += 1 if book_won else 0
                l += 0 if book_won else 1
        print(f"  start {i + 1}/{len(starts)}: book W{w} D{d} L{l}", flush=True)

    n = 2 * len(starts)
    score = (w + 0.5 * d) / n if n else 0.0
    print(f"\n[measure_book] RESULT book vs plain: W{w} D{d} L{l} over {n}  score={score:.3f}  "
          f"-> book is {elo_from_score(score):+.0f} Elo  ({time.perf_counter() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
