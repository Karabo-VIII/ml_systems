"""
chess_engine.sf_engine -- a drop-in Stockfish-backed engine ("compete against the best").

Our from-scratch python engine is honestly capped (pure-python move-gen is too slow to search
deep -- see docs/STRENGTH_ROADMAP.md). When the goal is to actually PLAY or MEASURE against the
best, this wraps Stockfish behind the SAME `.search(board) -> SearchResult` interface as
`chess_engine.engine.Engine`, so it drops into any caller (net-play, the demo, the benchmark).

It is honest about WHICH engine it is (`.name`), and you can cap its strength to a target Elo
(`elo=`) so it is a fair, calibrated opponent rather than an unbeatable wall.

    from chess_engine.sf_engine import StockfishEngine, find_stockfish
    eng = StockfishEngine(find_stockfish(), movetime=0.3, elo=1600)
    res = eng.search(board)          # res.move, res.score (cp, side-to-move)
    eng.close()

No emoji (Windows cp1252).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import chess
import chess.engine

from chess_engine.engine import SearchResult, MATE_SCORE

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root


def find_stockfish() -> Optional[str]:
    """Locate a Stockfish binary: env STOCKFISH_PATH, the repo's engines/ dir, or PATH."""
    env = os.environ.get("STOCKFISH_PATH")
    if env and os.path.exists(env):
        return env
    for pat in ("engines/**/stockfish*.exe", "engines/**/stockfish*", "engines/stockfish*"):
        hits = glob.glob(os.path.join(_HERE, pat), recursive=True)
        hits = [h for h in hits if os.path.isfile(h) and not h.endswith(".zip")]
        if hits:
            return hits[0]
    from shutil import which
    return which("stockfish")


class StockfishEngine:
    """Stockfish behind the Engine.search() interface. Optionally Elo-capped for fair play."""

    def __init__(self, path: str, movetime: float = 0.3, elo: Optional[int] = None,
                 depth: Optional[int] = None, threads: int = 1, hash_mb: int = 64):
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Stockfish binary not found: {path!r}")
        self._eng = chess.engine.SimpleEngine.popen_uci(path)
        opts = {"Threads": threads, "Hash": hash_mb}
        if elo is not None:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = max(1320, min(3190, int(elo)))
        try:
            self._eng.configure(opts)
        except Exception:
            pass
        self.movetime = movetime
        self.depth = depth
        self.elo = elo
        self.name = f"Stockfish 16.1{f' @{elo} Elo' if elo else ''}"

    def search(self, board: chess.Board) -> SearchResult:
        limit = (chess.engine.Limit(depth=self.depth) if self.depth
                 else chess.engine.Limit(time=self.movetime))
        result = self._eng.play(board, limit, info=chess.engine.INFO_SCORE)
        mv = result.move
        score = 0
        info = result.info or {}
        povs = info.get("score")
        if povs is not None:
            try:
                rel = povs.pov(board.turn)
                score = rel.score(mate_score=MATE_SCORE) if rel.is_mate() else rel.score()
                score = int(score) if score is not None else 0
            except Exception:
                score = 0
        return SearchResult(move=mv, score=score, depth=info.get("depth", 0) or 0,
                            nodes=info.get("nodes", 0) or 0, time_s=self.movetime,
                            pv=[mv] if mv else [])

    def close(self):
        try:
            self._eng.quit()
        except Exception:
            pass

    def __del__(self):
        self.close()


if __name__ == "__main__":
    sf = find_stockfish()
    print("stockfish:", sf)
    if sf:
        e = StockfishEngine(sf, movetime=0.2, elo=1600)
        b = chess.Board()
        r = e.search(b)
        print(f"{e.name} plays {b.san(r.move)}  (score {r.score}cp, depth {r.depth})")
        e.close()
