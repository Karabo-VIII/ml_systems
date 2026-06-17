"""
chess_zero.az.uci_engine -- a PLUGGABLE strong-opponent interface for self-play.

The robust training loop (train_robust.py) can refine the net against a TEACHER
opponent (see --selfplay-opponent {teacher,mix}). By default the teacher is our
own in-repo classical engine (engine.Engine), which needs
ZERO setup and always works. This module lets the user OPTIONALLY upgrade that
teacher to a REAL-WORLD SOTA UCI engine (e.g. Stockfish) by passing a binary
path -- WITHOUT changing any call site: both paths share one interface.

Factory:
    make_opponent(engine_path="", classical_depth=1, uci_movetime_ms=50) -> opponent

The returned opponent exposes:
    .select_move(board: chess.Board) -> chess.Move    # legal move for side-to-move
    .close()                                          # release engine resources

Design rules (load-bearing):
  - engine_path empty/missing  -> wrap the classical Engine (always available).
  - engine_path given          -> launch it over UCI (python-chess SimpleEngine).
  - ANY UCI launch/play error  -> print a clear WARN and FALL BACK to classical;
                                  training NEVER crashes because of the opponent.
  - chess.engine is imported LAZILY inside the UCI branch, so a missing/odd
    python-chess engine install can never break the classical default path.
  - Every .select_move is guarded to return a LEGAL move (or None only when the
    side to move has no legal moves, i.e. the game is already over).

HONEST NOTE (engine strength vs learning signal): a strong UCI engine is a
strong OPPONENT, but the net will LOSE most games against it, so the RL value
target is mostly z=-1 -- a weak positive gradient. The real way to CLIMB from a
strong engine is IMITATION of its moves (what the supervised bootstrap did)
and/or a CURRICULUM of increasing engine strength, NOT pure RL on losses. This
module is the PLUMBING (pluggable engine, classical-by-default, optional-UCI);
it does not by itself make the agent strong.

__contract__:
    kind: pluggable-chess-opponent
    inputs: engine_path (str), classical_depth (int), uci_movetime_ms (int)
    outputs: opponent object with .select_move(board)->chess.Move and .close()
    invariants:
        - classical path needs no external binary and never imports chess.engine
        - .select_move returns a legal move for the side to move (or None iff none)
        - a bad/failing UCI path WARNs and falls back to classical (never raises)
        - never mutates the caller's board
"""
from __future__ import annotations

from typing import Optional

import chess

from chess_engine.engine import Engine


def _legal_fallback(board: chess.Board) -> Optional[chess.Move]:
    """A deterministic legal move (first generated) for the side to move, or None
    if there are no legal moves (game already over)."""
    for mv in board.legal_moves:
        return mv
    return None


class ClassicalOpponent:
    """Wraps our in-repo classical engine (negamax + alpha-beta) as an opponent.

    Zero external dependencies beyond python-chess; this is the always-works
    default the training loop falls back to."""

    def __init__(self, depth: int = 1):
        self.depth = int(depth)
        self.engine = Engine(depth=self.depth)
        self.kind = "classical"

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        try:
            res = self.engine.search(board)
            mv = res.move
        except Exception as e:
            print(f"[uci_engine] WARN classical search failed ({e}); legal fallback")
            mv = None
        # Guard: the classical search may legitimately return None on a terminal
        # board; pick a legal move when one exists.
        if mv is None or mv not in board.legal_moves:
            mv = _legal_fallback(board)
        return mv

    def close(self) -> None:  # no external process to release
        return None


class UCIOpponent:
    """Wraps a real-world UCI engine (e.g. Stockfish) as an opponent.

    Launch failures and per-move failures degrade gracefully: any error makes
    .select_move return a classical-engine move so training never crashes."""

    def __init__(self, engine_path: str, movetime_ms: int = 50,
                 classical_depth: int = 1):
        # Lazy import so a missing/odd chess.engine install cannot break the
        # classical path -- only code that actually requests UCI touches it.
        import chess.engine as _ce  # noqa: F401  (imported for popen_uci + Limit)

        self._ce = _ce
        self.engine_path = engine_path
        self.movetime_s = max(0.001, float(movetime_ms) / 1000.0)
        self.kind = "uci"
        # A private classical engine for the per-move fallback (so a transient
        # UCI failure mid-game still yields a strong-ish legal move).
        self._fallback = ClassicalOpponent(depth=classical_depth)
        # popen_uci may raise (bad path, not a UCI engine, perms) -- let it
        # propagate so the factory can WARN + fall back to classical entirely.
        self.engine = _ce.SimpleEngine.popen_uci(engine_path)

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        try:
            result = self.engine.play(board, self._ce.Limit(time=self.movetime_s))
            mv = result.move
        except Exception as e:
            print(f"[uci_engine] WARN UCI play failed ({e}); classical fallback move")
            return self._fallback.select_move(board)
        # Guard a None / illegal UCI move -> legal fallback.
        if mv is None or mv not in board.legal_moves:
            return self._fallback.select_move(board)
        return mv

    def close(self) -> None:
        try:
            self.engine.quit()
        except Exception:
            pass  # best-effort release; never raise on teardown


def make_opponent(engine_path: str = "", classical_depth: int = 1,
                  uci_movetime_ms: int = 50):
    """Build the self-play TEACHER opponent.

    engine_path empty/missing -> ClassicalOpponent(depth=classical_depth) (our
    in-repo engine; zero setup, always works). engine_path given -> UCIOpponent
    over that binary; if the launch fails for ANY reason, print a clear WARN and
    fall back to the classical opponent so training never crashes.
    """
    path = (engine_path or "").strip()
    if not path:
        return ClassicalOpponent(depth=classical_depth)
    try:
        opp = UCIOpponent(path, movetime_ms=uci_movetime_ms,
                          classical_depth=classical_depth)
        print(f"[uci_engine] using UCI engine: {path} "
              f"(movetime={uci_movetime_ms}ms)")
        return opp
    except Exception as e:
        print(f"[uci_engine] WARN could not launch UCI engine '{path}' ({e}); "
              f"falling back to classical Engine(depth={classical_depth})")
        return ClassicalOpponent(depth=classical_depth)
