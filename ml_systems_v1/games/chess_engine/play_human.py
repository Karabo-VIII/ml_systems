"""
chess_zero.play_human -- interactive human-vs-engine CLI.

The LIVE-INTERFACE layer for a person at a terminal: you type moves (UCI like
`e2e4` or SAN like `Nf3`), the classical `engine.Engine` replies, and the board
is re-rendered every ply. Illegal / unparsable input is rejected with a clean
message (your turn is NOT consumed). The game ends only on a real python-chess
termination (checkmate / stalemate / insufficient material / 75-move / fivefold
/ claimable draw) -- we never invent a result.

Usage:
    python play_human.py [--engine-white] [--depth N] [--movetime SECONDS]

    --engine-white   engine plays White (moves first); default: human is White.
    --depth N        engine search depth in plies (default 4).
    --movetime S     per-move wall-clock budget in seconds (depth stays the cap).
    --ascii          force ASCII board (default tries unicode, falls back).

Type `quit` / `resign` to stop, `board` to re-print, `fen` to show the FEN,
`moves` to list legal moves, `help` for this list.

__contract__:
    kind: human-vs-engine-cli
    inputs: stdin move strings (UCI or SAN) + flags
    outputs: rendered board + engine replies on stdout
    invariants:
        - only legal moves are applied (illegal input never consumes a turn)
        - game ends on a real python-chess termination, never a fabricated one
        - engine never plays an illegal move (python-chess generated)
"""
from __future__ import annotations

import argparse
import sys

import chess

from chess_engine.engine import Engine, MATE_THRESHOLD


def _stdout_handles_unicode() -> bool:
    """True iff the current stdout encoding can actually encode the unicode chess figurines.
    On Windows cp1252 it cannot -- and the crash is in the caller's print() (encoding the
    string), NOT in board.unicode(), so we must check the ENCODING here, not catch later."""
    import sys
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "♔♟".encode(enc)  # white king + black pawn
        return True
    except Exception:
        return False


def render_board(board: chess.Board, use_unicode: bool) -> str:
    """Return a board rendering from White's perspective. Falls back to ASCII when the terminal
    can't encode unicode figurines (e.g. Windows cp1252) -- printing the unicode board there
    raises UnicodeEncodeError in the caller's print(), so we degrade BEFORE returning it."""
    if use_unicode and _stdout_handles_unicode():
        try:
            # board.unicode() draws figurines; invert colours so White is at bottom.
            return board.unicode(borders=False, invert_color=True)
        except Exception:
            pass  # fall through to ASCII
    return str(board)


def parse_human_move(board: chess.Board, text: str):
    """Parse a human move string as UCI first, then SAN. Returns a legal
    chess.Move, or None if it cannot be parsed / is illegal."""
    text = text.strip()
    # Try UCI (e2e4, e7e8q).
    try:
        mv = chess.Move.from_uci(text)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass
    # Try SAN (Nf3, exd5, O-O, e8=Q+).
    try:
        mv = board.parse_san(text)
        if mv in board.legal_moves:
            return mv
    except ValueError:
        pass
    return None


def termination_reason(board: chess.Board) -> str:
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75-move rule"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.can_claim_fifty_moves():
        return "50-move rule (claimable)"
    if board.can_claim_threefold_repetition():
        return "threefold repetition (claimable)"
    return "game over"


def announce_result(board: chess.Board) -> None:
    result = board.result(claim_draw=True)
    reason = termination_reason(board)
    print(f"\n=== GAME OVER: {result} ({reason}) ===")


def engine_move(board: chess.Board, engine: Engine) -> chess.Move:
    res = engine.search(board)
    mv = res.move
    if mv is None:  # no legal move -> caller will detect game over
        return None
    san = board.san(mv)
    extra = ""
    if abs(res.score) >= MATE_THRESHOLD:
        extra = " (forced mate)"
    print(f"\nengine plays: {san}  ({mv.uci()})  "
          f"eval={res.score}cp{extra}  depth={res.depth}  "
          f"nodes={res.nodes:,}  time={res.time_s:.2f}s")
    return mv


HELP_TEXT = (
    "commands: <move> (UCI e2e4 or SAN Nf3) | board | fen | moves | "
    "resign | quit | help"
)


def play(engine_white: bool, depth: int, movetime, use_unicode: bool,
         in_stream=None, engine=None) -> None:
    board = chess.Board()
    if engine is None:
        engine = Engine(depth=depth, time_limit=movetime)
    human_color = chess.BLACK if engine_white else chess.WHITE
    src = in_stream if in_stream is not None else sys.stdin

    opp = getattr(engine, "name", None) or f"classical engine (depth {depth})"
    print("=== chess_zero -- human vs engine ===")
    print(f"you are {'White' if human_color == chess.WHITE else 'Black'}; opponent: {opp}"
          + (f", movetime={movetime}s" if movetime else ""))
    print(HELP_TEXT)
    print()
    print(render_board(board, use_unicode))

    while not board.is_game_over(claim_draw=True):
        if board.turn == human_color:
            sys.stdout.write(f"\nyour move ({'White' if board.turn else 'Black'}): ")
            sys.stdout.flush()
            line = src.readline()
            if not line:  # EOF
                print("\n(eof -- ending session)")
                return
            text = line.strip()
            if text == "":
                continue
            low = text.lower()
            if low in ("quit", "exit", "resign"):
                print(f"\nyou {low}. {'engine' if low == 'resign' else 'session'} "
                      f"{'wins' if low == 'resign' else 'ended'}.")
                return
            if low == "help":
                print(HELP_TEXT); continue
            if low == "board":
                print(render_board(board, use_unicode)); continue
            if low == "fen":
                print(board.fen()); continue
            if low == "moves":
                print(" ".join(m.uci() for m in board.legal_moves)); continue

            mv = parse_human_move(board, text)
            if mv is None:
                print(f"  illegal/unparsable move: '{text}'. "
                      f"Try UCI (e2e4) or SAN (Nf3); 'moves' lists legal moves.")
                continue
            board.push(mv)
            print(render_board(board, use_unicode))
        else:
            mv = engine_move(board, engine)
            if mv is None:
                break
            board.push(mv)
            print(render_board(board, use_unicode))

    announce_result(board)


def main() -> None:
    ap = argparse.ArgumentParser(description="chess_zero human-vs-engine CLI")
    ap.add_argument("--engine-white", action="store_true",
                    help="engine plays White (moves first); default human=White")
    ap.add_argument("--depth", type=int, default=4,
                    help="engine search depth in plies (default 4)")
    ap.add_argument("--movetime", type=float, default=None,
                    help="per-move time budget in seconds (depth stays the cap)")
    ap.add_argument("--ascii", action="store_true",
                    help="force ASCII board rendering")
    ap.add_argument("--stockfish", action="store_true",
                    help="play vs STOCKFISH (one of the BEST) instead of the classical engine")
    ap.add_argument("--sf-elo", type=int, default=1600,
                    help="cap Stockfish to this Elo for a fair game (default 1600; range 1320-3190)")
    args = ap.parse_args()

    engine = None
    if args.stockfish:
        from chess_engine.sf_engine import StockfishEngine, find_stockfish
        path = find_stockfish()
        if not path:
            print("Stockfish not found (set STOCKFISH_PATH or place it under engines/). "
                  "Falling back to the classical engine.")
        else:
            engine = StockfishEngine(path, movetime=args.movetime or 0.3, elo=args.sf_elo)
    try:
        play(engine_white=args.engine_white, depth=args.depth,
             movetime=args.movetime, use_unicode=not args.ascii, engine=engine)
    finally:
        if engine is not None and hasattr(engine, "close"):
            engine.close()


if __name__ == "__main__":
    main()
