"""
chess_engine.bench -- a real strength yardstick for the classical chess engine.

Plays an engine config A vs config B in a head-to-head match from varied openings (colours
alternated), adjudicates long games by material, and reports W/D/L + a RELATIVE Elo estimate.
This is how we MEASURE strength honestly -- "config X beats config Y by E Elo over N games" --
instead of asserting it. (Absolute Elo needs a calibrated reference like Stockfish; this gives
the rigorous relative gain between two of our own configs, which is what proves an upgrade.)

  python -m chess_engine.bench                       # quick self-test: strong vs baseline
  from chess_engine.bench import match
  r = match(make_a, make_b, n=40, movetime=0.3)      # make_* are zero-arg Engine factories

No emoji (Windows cp1252).
"""
from __future__ import annotations

import math
import random
import time

import chess

from chess_engine.engine import Engine, PIECE_VALUE


def _material_diff(board: chess.Board) -> int:
    """White-minus-black material in centipawns (pawns..queens; kings excluded)."""
    d = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        v = PIECE_VALUE[pt]
        d += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return d


def _adjudicate(board: chess.Board) -> str:
    """Result for a game that hit the ply cap: material leader wins (>=300cp), else draw."""
    md = _material_diff(board)
    if md >= 300:
        return "1-0"
    if md <= -300:
        return "0-1"
    return "1/2-1/2"


def random_opening(n_plies: int, rng: random.Random):
    """A short random-but-legal opening so games differ (these plies are not scored)."""
    board = chess.Board()
    moves = []
    for _ in range(n_plies):
        legal = list(board.legal_moves)
        if not legal:
            break
        m = rng.choice(legal)
        moves.append(m)
        board.push(m)
    return moves


def play_game(white_eng: Engine, black_eng: Engine, opening, max_plies: int = 240) -> str:
    """Play one game from `opening` plies. Returns '1-0' / '0-1' / '1/2-1/2'."""
    board = chess.Board()
    for mv in opening:
        board.push(mv)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        eng = white_eng if board.turn == chess.WHITE else black_eng
        res = eng.search(board)
        mv = res.move
        if mv is None or mv not in board.legal_moves:
            mv = next(iter(board.legal_moves))
        board.push(mv)
    if board.is_game_over(claim_draw=True):
        return board.result(claim_draw=True)
    return _adjudicate(board)


def elo_from_score(score: float) -> float:
    if score <= 0.0:
        return -9999.0
    if score >= 1.0:
        return 9999.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def match(make_a, make_b, n: int = 20, movetime: float = 0.3,
          opening_plies: int = 4, seed: int = 0, max_plies: int = 240, verbose: bool = True):
    """Play `n` games of A vs B (colours alternated, A starts White on even games). make_a/make_b
    are zero-arg factories returning a fresh Engine. Returns a dict from A's perspective."""
    rng = random.Random(seed)
    w = d = l = 0
    t0 = time.perf_counter()
    for i in range(n):
        opening = random_opening(opening_plies, rng)
        a_white = (i % 2 == 0)
        white = make_a() if a_white else make_b()
        black = make_b() if a_white else make_a()
        r = play_game(white, black, opening, max_plies=max_plies)
        if r == "1/2-1/2":
            d += 1
            tag = "draw"
        else:
            a_won = (r == "1-0") == a_white
            if a_won:
                w += 1
            else:
                l += 1
            tag = "A-win" if a_won else "B-win"
        if verbose:
            print(f"  game {i + 1}/{n}: A={'W' if a_white else 'B'} -> {r} [{tag}]  "
                  f"(running A: W{w} D{d} L{l})", flush=True)
    score = (w + 0.5 * d) / n if n else 0.0
    out = {"w": w, "d": d, "l": l, "n": n, "score": score,
           "elo": round(elo_from_score(score), 1), "wall_s": round(time.perf_counter() - t0, 1)}
    if verbose:
        print(f"RESULT A vs B: W{w} D{d} L{l} over {n}  score={score:.3f}  "
              f"A is {out['elo']:+.0f} Elo  ({out['wall_s']}s)", flush=True)
    return out


def match_vs_uci(make_ours, sf_path: str, n: int = 10, movetime: float = 0.3,
                 sf_elo: int = 1320, opening_plies: int = 2, seed: int = 0,
                 max_plies: int = 240, verbose: bool = True):
    """Play our engine vs a UCI engine (e.g. Stockfish) capped at `sf_elo` strength. Returns our
    W/D/L + an ABSOLUTE Elo estimate (our_elo ~= sf_elo + 400*log10(score/(1-score))). This is the
    real integrity yardstick -- where our engine actually stands against a calibrated reference."""
    import chess.engine
    rng = random.Random(seed)
    sf = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        sf.configure({"UCI_LimitStrength": True, "UCI_Elo": int(sf_elo)})
    except Exception as exc:
        if verbose:
            print(f"  (could not cap Stockfish at {sf_elo} Elo: {exc}; using default strength)")
    w = d = l = 0
    t0 = time.perf_counter()
    try:
        for i in range(n):
            opening = random_opening(opening_plies, rng)
            ours_white = (i % 2 == 0)
            board = chess.Board()
            for mv in opening:
                board.push(mv)
            ours = make_ours()
            while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
                our_turn = (board.turn == chess.WHITE) == ours_white
                if our_turn:
                    mv = ours.search(board).move
                else:
                    mv = sf.play(board, chess.engine.Limit(time=movetime)).move
                if mv is None or mv not in board.legal_moves:
                    mv = next(iter(board.legal_moves))
                board.push(mv)
            r = (board.result(claim_draw=True)
                 if board.is_game_over(claim_draw=True) else _adjudicate(board))
            if r == "1/2-1/2":
                d += 1
            else:
                ours_won = (r == "1-0") == ours_white
                w += 1 if ours_won else 0
                l += 0 if ours_won else 1
            if verbose:
                print(f"  game {i + 1}/{n}: ours={'W' if ours_white else 'B'} -> {r}  "
                      f"(ours W{w} D{d} L{l})", flush=True)
    finally:
        sf.quit()
    score = (w + 0.5 * d) / n if n else 0.0
    our_elo = sf_elo + elo_from_score(score)
    out = {"w": w, "d": d, "l": l, "n": n, "score": round(score, 3), "sf_elo": sf_elo,
           "our_elo_est": round(our_elo), "wall_s": round(time.perf_counter() - t0, 1)}
    if verbose:
        print(f"RESULT ours vs Stockfish@{sf_elo}: W{w} D{d} L{l} score={score:.3f}  "
              f"-> our engine ~= {out['our_elo_est']} Elo  ({out['wall_s']}s)", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Head-to-head chess engine benchmark.")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--movetime", type=float, default=0.3)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline", action="store_true",
                    help="A and B are BOTH the default engine (sanity: expect ~50%%)")
    args = ap.parse_args()

    def strong():
        return Engine(depth=64, time_limit=args.movetime, strong=True)

    def base():
        return Engine(depth=64, time_limit=args.movetime, strong=False)

    make_a = strong
    make_b = strong if args.baseline else base
    print(f"A = {'default' if True else ''} engine; B = {'default' if args.baseline else 'baseline (no TT/null/LMR)'} "
          f"| n={args.n} movetime={args.movetime}s")
    match(make_a, make_b, n=args.n, movetime=args.movetime,
          opening_plies=args.opening_plies, seed=args.seed)
