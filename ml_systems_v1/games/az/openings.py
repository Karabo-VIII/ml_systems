"""
chess_zero.az.openings -- OPENING DIVERSITY for self-play (the "vary the starting
conditions" lever).

WHY THIS EXISTS (2026-06-09 user diagnosis, confirmed):
    Every self-play game used to start from the IDENTICAL standard position
    (chess.Board()). The only variation was Dirichlet root noise + a short
    temperature window. On a sharply-peaked bootstrap-imitation net at low sims
    that washes out, so games funnel through the SAME opening line, the net
    reinforces one rote repertoire ("plays the same way" / bad learned habits),
    and the VALUE head rarely sees diverse, decisive positions -- self-play
    cannot teach what it never visits. This is the textbook AlphaZero/Leela
    failure mode; the textbook fix is to START EACH GAME FROM A DIVERSE, SOUND
    OPENING (Leela uses randomized opening books for exactly this).

WHAT THIS DOES:
    sample_opening_board(rng, mode, ...) returns a chess.Board advanced to a
    varied starting position. Modes:
      - "startpos" : plain chess.Board() (back-compat; what EVAL uses).
      - "book"     : a random line from a curated book of sound, balanced
                     openings (broad ECO spread) -> sound diversity, no risk of
                     starting from a blundered/lost position.
      - "random"   : `random_plies` random legal plies, REJECTED+resampled if the
                     resulting material imbalance exceeds `max_material_cp` or the
                     position is already over -> diversity without teaching the net
                     to play from a hung-piece position.
      - "mixed"    : a book line THEN up to `random_plies` guarded random plies
                     (within-line jitter on top of book diversity). DEFAULT for
                     self-play.

IMPORTANT CONTRACT: the opening plies are NOT training samples. Callers apply the
returned board as the GAME START and only record (planes, pi, z) from the net's
own searched moves onward -- the book/random plies are just a varied initial
condition, never a policy target. EVAL never uses this (it stays on startpos) so
the strength curve remains a stable, comparable yardstick.

No emoji (Windows cp1252). Depends only on python-chess + numpy.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import chess


# --------------------------------------------------------------------------- #
# Curated opening book: sound, roughly-balanced lines across a broad ECO spread.
# Each entry is a space-separated SAN sequence. Lines are kept short-to-medium
# (2-9 plies) so the NET still makes most opening decisions; the book only sets a
# varied, non-losing starting CONTEXT. Any line that fails to parse (a SAN typo)
# is silently skipped at sample time and flagged by the self-test -- a bad line
# degrades diversity slightly, it never crashes training.
# --------------------------------------------------------------------------- #
OPENING_BOOK: List[str] = [
    # --- 1.e4 e5 (open games) ---
    "e4 e5",
    "e4 e5 Nf3 Nc6",
    "e4 e5 Nf3 Nc6 Bb5",            # Ruy Lopez
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6", # Ruy, closed
    "e4 e5 Nf3 Nc6 Bc4",            # Italian
    "e4 e5 Nf3 Nc6 Bc4 Bc5",        # Giuoco Piano
    "e4 e5 Nf3 Nc6 Bc4 Nf6",        # Two Knights
    "e4 e5 Nf3 Nc6 d4",             # Scotch
    "e4 e5 Nf3 Nf6",               # Petrov
    "e4 e5 Nc3",                   # Vienna
    "e4 e5 Bc4",                   # Bishop's Opening
    "e4 e5 f4",                    # King's Gambit
    # --- Sicilian ---
    "e4 c5",
    "e4 c5 Nf3",
    "e4 c5 Nf3 d6",
    "e4 c5 Nf3 Nc6",
    "e4 c5 Nf3 e6",
    "e4 c5 Nc3",                   # Closed Sicilian
    "e4 c5 c3",                    # Alapin
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3",  # Open Sicilian main tabiya
    "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6",      # Accelerated Dragon-ish
    # --- French ---
    "e4 e6",
    "e4 e6 d4 d5",
    "e4 e6 d4 d5 Nc3 Nf6",         # French, Classical
    "e4 e6 d4 d5 e5",              # Advance
    "e4 e6 d4 d5 exd5 exd5",       # Exchange
    # --- Caro-Kann ---
    "e4 c6",
    "e4 c6 d4 d5",
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4",   # Classical
    "e4 c6 d4 d5 e5",             # Advance
    # --- other 1.e4 ---
    "e4 d5 exd5 Qxd5 Nc3 Qa5",     # Scandinavian
    "e4 d6 d4 Nf6 Nc3 g6",         # Pirc
    "e4 g6 d4 Bg7",               # Modern
    "e4 Nf6",                     # Alekhine
    # --- 1.d4 d5 ---
    "d4 d5",
    "d4 d5 c4",                    # Queen's Gambit
    "d4 d5 c4 e6",                # QGD
    "d4 d5 c4 e6 Nc3 Nf6",         # QGD main
    "d4 d5 c4 c6",                # Slav
    "d4 d5 c4 dxc4",              # QGA
    "d4 d5 Nf3 Nf6 c4 e6",         # QGD via Nf3
    # --- 1.d4 Nf6 (Indian) ---
    "d4 Nf6",
    "d4 Nf6 c4",
    "d4 Nf6 c4 e6",
    "d4 Nf6 c4 e6 Nc3 Bb4",        # Nimzo-Indian
    "d4 Nf6 c4 e6 Nf3 b6",         # Queen's Indian
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6",  # King's Indian
    "d4 Nf6 c4 g6 Nc3 d5",         # Gruenfeld
    "d4 Nf6 c4 c5 d5 e6",          # Benoni
    "d4 Nf6 Nf3 e6 g3",            # Catalan setup
    "d4 f5",                      # Dutch
    # --- English / Reti / flank ---
    "c4",
    "c4 e5",                       # English, reversed Sicilian
    "c4 c5",                       # Symmetrical English
    "c4 Nf6",
    "c4 e6",
    "c4 g6",
    "Nf3",
    "Nf3 d5",
    "Nf3 Nf6 c4 g6 g3",            # Reti / KIA-ish
    "Nf3 d5 g3",                   # Reti
    "g3 d5 Bg2",                   # King's Indian Attack setup
    "b3",                          # Larsen / Nimzo-Larsen
    "f4 d5",                       # Bird
]


# --------------------------------------------------------------------------- #
# Material balance guard (self-contained: do NOT import train_robust -> no cycle).
# --------------------------------------------------------------------------- #
_PIECE_CP = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}


def material_balance_cp(board: chess.Board) -> int:
    """Centipawn material balance from White's perspective (cheap; for the
    random-opening 'not already lost' guard)."""
    bal = 0
    for _sq, piece in board.piece_map().items():
        v = _PIECE_CP[piece.piece_type]
        bal += v if piece.color == chess.WHITE else -v
    return bal


def _line_to_board(line: str) -> Optional[chess.Board]:
    """Apply a SAN line to a fresh board. Returns the board, or None if any SAN
    token is illegal (a typo in the book) or the line ends in a game-over."""
    board = chess.Board()
    try:
        for san in line.split():
            board.push_san(san)
    except Exception:
        return None
    if board.is_game_over(claim_draw=True):
        return None
    return board


def _random_plies(board: chess.Board, n: int, rng: np.random.Generator,
                  max_material_cp: int) -> bool:
    """Apply up to n random legal plies IN PLACE, stopping early if the position
    would go game-over. Returns True if the final position is within the material
    guard (|balance| <= max_material_cp) and not over, else False (caller retries)."""
    for _ in range(n):
        if board.is_game_over(claim_draw=True):
            return False
        moves = list(board.legal_moves)
        if not moves:
            return False
        board.push(moves[int(rng.integers(len(moves)))])
    if board.is_game_over(claim_draw=True):
        return False
    return abs(material_balance_cp(board)) <= max_material_cp


def sample_opening_board(rng: np.random.Generator, mode: str = "book",
                         random_plies: int = 4, max_material_cp: int = 120,
                         max_tries: int = 8) -> chess.Board:
    """Return a chess.Board advanced to a diverse, SOUND starting position.

    mode:
      "startpos" -> plain chess.Board() (no diversity; what EVAL uses).
      "book"     -> a random curated opening line (balanced by construction).
      "random"   -> `random_plies` random plies, guarded so we never start from an
                    already-lost/over position (|material| <= max_material_cp).
      "mixed"    -> a book line, then up to `random_plies` GUARDED random plies for
                    within-line jitter (DEFAULT for self-play). Falls back to the
                    pure book line if every jittered attempt busts the guard.

    rng is an np.random.Generator (seeded by the caller) so a seeded run stays
    reproducible. Always returns a LEGAL, non-terminal board (worst case: startpos
    or a clean book line)."""
    if mode == "startpos":
        return chess.Board()

    if mode == "random":
        for _ in range(max_tries):
            board = chess.Board()
            if _random_plies(board, random_plies, rng, max_material_cp):
                return board
        return chess.Board()  # guard never satisfied -> safe fallback

    # "book" or "mixed": pick a parseable book line.
    book = chess.Board()
    for _ in range(max_tries):
        line = OPENING_BOOK[int(rng.integers(len(OPENING_BOOK)))]
        b = _line_to_board(line)
        if b is not None:
            book = b
            break

    if mode == "book" or random_plies <= 0:
        return book

    # "mixed": add guarded random jitter on top of the book line.
    for _ in range(max_tries):
        b = book.copy()
        if _random_plies(b, random_plies, rng, max_material_cp):
            return b
    return book  # jitter kept busting the guard -> the clean book line is fine


# --------------------------------------------------------------------------- #
# Self-test: validate the book + show the diversity the sampler produces.
#   .venv/Scripts/python.exe -m az.openings
# --------------------------------------------------------------------------- #
def _selftest() -> None:
    # 1) every book line must parse to a legal, non-terminal position.
    bad = [ln for ln in OPENING_BOOK if _line_to_board(ln) is None]
    n_ok = len(OPENING_BOOK) - len(bad)
    print(f"[openings] book lines: {len(OPENING_BOOK)} total, {n_ok} valid, "
          f"{len(bad)} bad")
    for ln in bad:
        print(f"   BAD (skipped at sample time): {ln!r}")
    assert n_ok >= 40, f"book too small after validation ({n_ok} valid)"

    # 2) the sampler must produce DIVERSE, legal starts.
    rng = np.random.default_rng(0)
    for mode in ("startpos", "book", "random", "mixed"):
        fens = set()
        for _ in range(200):
            b = sample_opening_board(rng, mode=mode, random_plies=4)
            assert b.is_valid(), f"{mode}: produced an invalid board"
            assert not b.is_game_over(claim_draw=True), f"{mode}: started game-over"
            assert abs(material_balance_cp(b)) <= 900, f"{mode}: wildly unbalanced"
            fens.add(b.board_fen())
        print(f"[openings] mode={mode:8s} -> {len(fens):3d} distinct start positions "
              f"out of 200 samples")
        if mode == "startpos":
            assert len(fens) == 1, "startpos must be unique"
        else:
            assert len(fens) >= 10, f"{mode}: too few distinct openings ({len(fens)})"
    print("[openings] SELF-TEST PASS")


if __name__ == "__main__":
    _selftest()
