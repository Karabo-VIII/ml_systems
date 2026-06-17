"""
chess_zero.uci -- a UCI-protocol wrapper around the classical engine.

This is the LIVE-INTERFACE layer (not a new engine): it speaks the Universal
Chess Interface on stdin/stdout so `engine.Engine` runs inside ANY UCI GUI
(Arena, CuteChess, Banksia), `cutechess-cli`, or a lichess-bot adapter.

Supported commands (UCI subset sufficient for GUIs and match runners):
    uci                  -> id name / id author / uciok
    isready              -> readyok
    ucinewgame           -> reset internal state
    position [startpos | fen <FEN>] [moves <m1> <m2> ...]
    go [depth N | movetime MS | wtime/btime/winc/binc/movestogo ...]
                         -> info ... / bestmove <uci>
    stop                 -> (best-effort) print bestmove for current position
    quit                 -> exit the loop

Design notes:
    - Reads stdin LINE BY LINE so partial / streamed input never blocks or
      crashes the loop. Unknown tokens are ignored (UCI requires tolerance).
    - Malformed `position` / `go` lines degrade gracefully: an illegal move in
      a `moves` list is skipped (with an `info string`), never raised.
    - The move is emitted via `chess.Move.uci()` (engine returns python-chess
      moves), e.g. `bestmove e2e4` or `bestmove e7e8q` (promotion).
    - When the position is terminal (no legal moves) we emit `bestmove (none)`
      per the UCI convention, rather than crashing.

__contract__:
    kind: uci-protocol-adapter
    inputs: UCI command lines on stdin (str)
    outputs: UCI response lines on stdout (id/uciok/readyok/info/bestmove)
    invariants:
        - never raises on malformed input (loop is crash-proof)
        - bestmove is always a legal move (or '(none)' when no legal move)
        - does not mutate any caller state; owns its own chess.Board
"""
from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

# --- repo-root bootstrap: make chess_engine/ importable as a top-level package
# whether run as 'python chess_engine/uci.py', '-m chess_engine.uci', or from a GUI.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # games-engine root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import chess

from chess_engine.engine import Engine, MATE_THRESHOLD

ENGINE_NAME = "chess_zero"
ENGINE_AUTHOR = "chess_zero (classical alpha-beta, AlphaZero-lineage project)"

# Defaults when `go` carries no depth/time directive.
DEFAULT_DEPTH = 4
# Fraction of remaining clock spent on one move when only wtime/btime are given.
CLOCK_FRACTION = 30.0  # ~ assume 30 moves left
MIN_MOVETIME_S = 0.05
MAX_MOVETIME_S = 30.0


class UCIEngine:
    """Holds the live board + search settings across a UCI session."""

    def __init__(self, out: Optional[TextIO] = None):
        self.board = chess.Board()
        self.out = out if out is not None else sys.stdout

    # --- output helper -------------------------------------------------------
    def _send(self, line: str) -> None:
        self.out.write(line + "\n")
        self.out.flush()

    # --- command: position ---------------------------------------------------
    def cmd_position(self, tokens: list) -> None:
        """position [startpos | fen <6 FEN fields>] [moves <m1> <m2> ...]."""
        if not tokens:
            return
        idx = 0
        if tokens[0] == "startpos":
            self.board = chess.Board()
            idx = 1
        elif tokens[0] == "fen":
            # FEN is up to 6 space-separated fields after the 'fen' keyword.
            fen_fields = []
            idx = 1
            while idx < len(tokens) and tokens[idx] != "moves" and len(fen_fields) < 6:
                fen_fields.append(tokens[idx])
                idx += 1
            fen = " ".join(fen_fields)
            try:
                self.board = chess.Board(fen)
            except (ValueError, IndexError) as exc:
                self._send(f"info string invalid fen, keeping previous board: {exc}")
                return
        else:
            # Unknown subcommand -- ignore but stay alive.
            self._send(f"info string ignoring unknown position arg: {tokens[0]}")
            return

        # Optional trailing `moves m1 m2 ...`.
        if idx < len(tokens) and tokens[idx] == "moves":
            for mv_str in tokens[idx + 1:]:
                try:
                    move = chess.Move.from_uci(mv_str)
                except ValueError:
                    self._send(f"info string skipping unparsable move: {mv_str}")
                    continue
                if move in self.board.legal_moves:
                    self.board.push(move)
                else:
                    self._send(f"info string skipping illegal move: {mv_str}")

    # --- command: go ---------------------------------------------------------
    def _parse_go(self, tokens: list):
        """Return (depth, time_limit_s) from a `go` token list.

        Precedence: explicit `depth` / `movetime` win; otherwise derive a
        per-move time budget from the side-to-move's clock (wtime/btime + inc).
        """
        depth = None
        movetime_ms = None
        wtime = btime = winc = binc = None
        movestogo = None
        infinite = False

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            try:
                if tok == "depth" and i + 1 < len(tokens):
                    depth = int(tokens[i + 1]); i += 2; continue
                if tok == "movetime" and i + 1 < len(tokens):
                    movetime_ms = int(tokens[i + 1]); i += 2; continue
                if tok == "wtime" and i + 1 < len(tokens):
                    wtime = int(tokens[i + 1]); i += 2; continue
                if tok == "btime" and i + 1 < len(tokens):
                    btime = int(tokens[i + 1]); i += 2; continue
                if tok == "winc" and i + 1 < len(tokens):
                    winc = int(tokens[i + 1]); i += 2; continue
                if tok == "binc" and i + 1 < len(tokens):
                    binc = int(tokens[i + 1]); i += 2; continue
                if tok == "movestogo" and i + 1 < len(tokens):
                    movestogo = int(tokens[i + 1]); i += 2; continue
                if tok == "infinite":
                    infinite = True; i += 1; continue
            except (ValueError, TypeError):
                # Non-integer argument -> ignore this token, keep parsing.
                i += 2; continue
            i += 1

        # Explicit movetime wins.
        if movetime_ms is not None:
            tl = max(MIN_MOVETIME_S, min(movetime_ms / 1000.0, MAX_MOVETIME_S))
            # Use a generous depth cap so the clock, not depth, is the limit.
            return (depth if depth is not None else 64), tl

        # Explicit depth (and no movetime) -> depth-bounded, no clock.
        if depth is not None:
            return depth, None

        # Clock-based control: budget = remaining/CLOCK_FRACTION + ~0.8*inc.
        my_time = wtime if self.board.turn == chess.WHITE else btime
        my_inc = winc if self.board.turn == chess.WHITE else binc
        if my_time is not None:
            divisor = movestogo if (movestogo and movestogo > 0) else CLOCK_FRACTION
            budget_s = (my_time / 1000.0) / divisor
            if my_inc:
                budget_s += 0.8 * (my_inc / 1000.0)
            tl = max(MIN_MOVETIME_S, min(budget_s, MAX_MOVETIME_S))
            return 64, tl

        # `go infinite` or bare `go`: fall back to the default fixed depth.
        return DEFAULT_DEPTH, None

    def cmd_go(self, tokens: list) -> None:
        depth, time_limit = self._parse_go(tokens)

        legal = list(self.board.legal_moves)
        if not legal:
            self._send("bestmove (none)")
            return

        eng = Engine(depth=depth, time_limit=time_limit)
        res = eng.search(self.board)
        move = res.move if res.move is not None else legal[0]

        # Emit an info line (depth / score / nodes / nps / pv) for GUIs.
        nps = int(res.nodes / res.time_s) if res.time_s > 0 else 0
        if abs(res.score) >= MATE_THRESHOLD:
            mate_in = (MATE_THRESHOLD + 1000 - abs(res.score) + 1) // 2
            sign = 1 if res.score > 0 else -1
            score_str = f"mate {sign * mate_in}"
        else:
            score_str = f"cp {res.score}"
        pv_uci = " ".join(m.uci() for m in res.pv) if res.pv else move.uci()
        self._send(
            f"info depth {res.depth} score {score_str} nodes {res.nodes} "
            f"nps {nps} time {int(res.time_s * 1000)} pv {pv_uci}"
        )
        self._send(f"bestmove {move.uci()}")

    # --- main loop -----------------------------------------------------------
    def handle(self, line: str) -> bool:
        """Process one UCI line. Returns False when the loop should exit."""
        line = line.strip()
        if not line:
            return True
        parts = line.split()
        cmd, args = parts[0], parts[1:]

        if cmd == "uci":
            self._send(f"id name {ENGINE_NAME}")
            self._send(f"id author {ENGINE_AUTHOR}")
            self._send("uciok")
        elif cmd == "isready":
            self._send("readyok")
        elif cmd == "ucinewgame":
            self.board = chess.Board()
        elif cmd == "position":
            self.cmd_position(args)
        elif cmd == "go":
            self.cmd_go(args)
        elif cmd == "stop":
            # Not multithreaded: search is synchronous, so by the time `stop`
            # arrives `go` has already replied. Emit a best move for safety.
            legal = list(self.board.legal_moves)
            if legal:
                eng = Engine(depth=1)
                res = eng.search(self.board)
                mv = res.move if res.move is not None else legal[0]
                self._send(f"bestmove {mv.uci()}")
            else:
                self._send("bestmove (none)")
        elif cmd in ("quit", "exit"):
            return False
        elif cmd in ("setoption", "debug", "register", "ponderhit", "ucinewgame"):
            # Recognised-but-noop commands: stay silent, stay alive.
            pass
        else:
            # Unknown command: UCI requires us to ignore it, not crash.
            self._send(f"info string ignoring unknown command: {cmd}")
        return True


def main() -> None:
    eng = UCIEngine()
    # Line-by-line so streamed / partial input is handled gracefully.
    for line in sys.stdin:
        try:
            keep_going = eng.handle(line)
        except Exception as exc:  # noqa: BLE001 -- loop must never die on input
            eng._send(f"info string internal error handling line: {exc}")
            keep_going = True
        if not keep_going:
            break


if __name__ == "__main__":
    main()
