"""
chess_zero.az.encoding -- board <-> tensor planes, and move <-> policy index.

Follows AlphaZero (arXiv:1712.01815, "Methods"):

INPUT PLANES (per position, current-player perspective):
    For each colour (us, them): 6 piece-type planes  -> 12 planes
    Repetition planes (we use 1 simple "twofold so far" plane)  -> 1
    Colour-to-move plane (all 1 if white to move else 0)  -> 1
    Total move count / no-progress (50-move) plane (scaled)  -> 1
    Castling rights (us K, us Q, them K, them Q)  -> 4
    => 19 planes of 8x8.  (The paper stacks 8 history steps; we use 1 step here
       for the scaffold; HISTORY_STEPS makes that configurable.)

POLICY ENCODING (the 8x8x73 = 4672 move planes of AlphaZero):
    73 = 56 "queen moves" (8 directions x 7 distances)
       + 8 knight moves
       + 9 underpromotions (3 pieces x 3 directions: forward/capture-left/right)
    Queen-move promotions to QUEEN are encoded in the queen-move planes
    (a pawn reaching the last rank along a queen direction).
    Index = from_square * 73 + move_plane.

This module is pure (numpy) and import-clean; net.py turns planes into a tensor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import chess

HISTORY_STEPS = 1            # paper uses 8; scaffold uses 1
PIECE_PLANES = 12            # 6 piece types x 2 colours
AUX_PLANES = 7              # rep(1) + colour(1) + move-count(1) + castling(4)
N_INPUT_PLANES = HISTORY_STEPS * (PIECE_PLANES + 1) + (AUX_PLANES - 1)
# Note: we keep N_INPUT_PLANES simple/explicit below rather than derive it.
N_INPUT_PLANES = 19

POLICY_PLANES = 73
N_POLICY = 64 * POLICY_PLANES   # 4672

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

# Queen-move directions (dfile, drank) in the order used for plane indexing:
# N, NE, E, SE, S, SW, W, NW
_QUEEN_DIRS = [(0, 1), (1, 1), (1, 0), (1, -1),
               (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# Knight deltas in a fixed order (8 of them).
_KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2),
                  (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Underpromotion pieces (knight, bishop, rook) x 3 directions (left, straight, right).
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """Return a (N_INPUT_PLANES, 8, 8) float32 tensor from the side-to-move POV.

    When black is to move we flip the board so 'us' is always at the bottom --
    this is the AlphaZero canonical-orientation trick.
    """
    planes = np.zeros((N_INPUT_PLANES, 8, 8), dtype=np.float32)
    us = board.turn
    them = not us

    def sq_to_rc(sq: int):
        # From the side-to-move perspective: flip ranks for black.
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        if us == chess.BLACK:
            rank = 7 - rank
            file = 7 - file
        return rank, file

    # 12 piece planes: first 6 = our pieces, next 6 = their pieces.
    for i, ptype in enumerate(PIECE_ORDER):
        for sq in board.pieces(ptype, us):
            r, c = sq_to_rc(sq)
            planes[i, r, c] = 1.0
        for sq in board.pieces(ptype, them):
            r, c = sq_to_rc(sq)
            planes[6 + i, r, c] = 1.0

    # Plane 12: repetition (twofold) indicator.
    if board.is_repetition(2):
        planes[12, :, :] = 1.0
    # Plane 13: colour to move (1 if white to move).
    if board.turn == chess.WHITE:
        planes[13, :, :] = 1.0
    # Plane 14: halfmove clock scaled by 100 (no-progress counter).
    planes[14, :, :] = board.halfmove_clock / 100.0
    # Planes 15-18: castling rights (us-K, us-Q, them-K, them-Q).
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(us) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(us) else 0.0
    planes[17, :, :] = 1.0 if board.has_kingside_castling_rights(them) else 0.0
    planes[18, :, :] = 1.0 if board.has_queenside_castling_rights(them) else 0.0
    return planes


def _orient_square(sq: int, white_to_move: bool) -> int:
    """Map a real square to the canonical (side-to-move-at-bottom) square."""
    if white_to_move:
        return sq
    return sq ^ 63  # mirror both rank and file


def move_to_index(board: chess.Board, move: chess.Move) -> Optional[int]:
    """Map a legal move to its [0, 4672) policy index, in canonical orientation.

    Returns None if the move cannot be encoded (should not happen for legal
    chess moves under this scheme).
    """
    white_to_move = board.turn == chess.WHITE
    from_sq = _orient_square(move.from_square, white_to_move)
    to_sq = _orient_square(move.to_square, white_to_move)

    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    tf, tr = chess.square_file(to_sq), chess.square_rank(to_sq)
    df, dr = tf - ff, tr - fr

    promo = move.promotion

    # --- underpromotions (to N/B/R): 9 planes at offset 64 ---
    if promo is not None and promo != chess.QUEEN:
        # direction: capture-left(-1), straight(0), capture-right(+1) by file
        try:
            dir_idx = {-1: 0, 0: 1, 1: 2}[df]
            piece_idx = _UNDERPROMO_PIECES.index(promo)
        except (KeyError, ValueError):
            return None
        plane = 64 + piece_idx * 3 + dir_idx
        return from_sq * POLICY_PLANES + plane

    # --- knight moves: 8 planes at offset 56 ---
    if (abs(df), abs(dr)) in [(1, 2), (2, 1)]:
        try:
            k_idx = _KNIGHT_DELTAS.index((df, dr))
        except ValueError:
            return None
        plane = 56 + k_idx
        return from_sq * POLICY_PLANES + plane

    # --- queen moves (incl. queen-promotions): 56 planes at offset 0 ---
    if df == 0 and dr == 0:
        return None
    # Determine the unit direction.
    step_f = (df > 0) - (df < 0)
    step_r = (dr > 0) - (dr < 0)
    # Must be a straight/diagonal line.
    if not (df == 0 or dr == 0 or abs(df) == abs(dr)):
        return None
    try:
        dir_idx = _QUEEN_DIRS.index((step_f, step_r))
    except ValueError:
        return None
    distance = max(abs(df), abs(dr))
    if not (1 <= distance <= 7):
        return None
    plane = dir_idx * 7 + (distance - 1)
    return from_sq * POLICY_PLANES + plane


def legal_policy_mask(board: chess.Board):
    """Return (mask, idx_to_move): a (N_POLICY,) {0,1} mask over legal moves and
    a dict policy_index -> chess.Move for decoding the net's output."""
    mask = np.zeros(N_POLICY, dtype=np.float32)
    idx_to_move = {}
    for mv in board.legal_moves:
        idx = move_to_index(board, mv)
        if idx is not None:
            mask[idx] = 1.0
            idx_to_move[idx] = mv
    return mask, idx_to_move
