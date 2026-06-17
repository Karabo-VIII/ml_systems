"""
chess_zero.engine -- a real classical chess engine (NOT an LLM, NOT Claude).

An AlphaZero-LINEAGE project: this file is the classical baseline (the "from
scratch" strength reference). The neural AlphaZero frontier lives in ./az/.

Search:    NEGAMAX + ALPHA-BETA + ITERATIVE DEEPENING (time-bounded)
Eval:      evaluate_v2 (DEFAULT) = material + tapered piece-square tables + mobility +
           pawn structure (doubled/isolated/passed) + king pawn-shield + rook-on-open-file,
           SIDE-TO-MOVE relative. MEASURED +108 Elo over the simpler `evaluate` (30-game match).
Ordering:  MVV-LVA captures first, then quiet moves (so alpha-beta prunes).
Quiesce:   capture-only quiescence search (mitigates the horizon effect).

Legality/castling/en-passant/check/checkmate are delegated to python-chess
(do NOT reinvent them). This module only adds search + evaluation.

Public API:
    best_move(board, depth=4, time_limit=None) -> (chess.Move, info_dict)
    Engine(...).search(board) -> SearchResult

__contract__:
    kind: chess-search-engine
    inputs: chess.Board
    outputs: chess.Move (best), eval cp, nodes, depth, pv
    invariants:
        - returns only legal moves (python-chess generated)
        - eval is side-to-move relative (negamax)
        - never mutates the caller's board (push/pop balanced)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import chess
import chess.polyglot  # Zobrist hashing for the transposition table

# Transposition-table entry flags + size cap.
_TT_EXACT, _TT_LOWER, _TT_UPPER = 0, 1, 2
_TT_MAX = 1_500_000

# --------------------------------------------------------------------------- #
# Material values (centipawns). King has no material value for search (its loss
# is encoded as a mate score, not a material swing).
# --------------------------------------------------------------------------- #
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

MATE_SCORE = 1_000_000
MATE_THRESHOLD = MATE_SCORE - 1000  # scores above this are "mate in n"

# --------------------------------------------------------------------------- #
# Piece-square tables (from White's perspective, a1=index 0 .. h8=index 63).
# Values in centipawns, added to material. For Black we mirror vertically.
# Classic tables (Tomasz Michniewski's "simplified evaluation function").
# --------------------------------------------------------------------------- #
PAWN_PST = [
      0,   0,   0,   0,   0,   0,   0,   0,
      5,  10,  10, -20, -20,  10,  10,   5,
      5,  -5, -10,   0,   0, -10,  -5,   5,
      0,   0,   0,  20,  20,   0,   0,   0,
      5,   5,  10,  25,  25,  10,   5,   5,
     10,  10,  20,  30,  30,  20,  10,  10,
     50,  50,  50,  50,  50,  50,  50,  50,
      0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20,   0,   5,   5,   0, -20, -40,
    -30,   5,  10,  15,  15,  10,   5, -30,
    -30,   0,  15,  20,  20,  15,   0, -30,
    -30,   5,  15,  20,  20,  15,   5, -30,
    -30,   0,  10,  15,  15,  10,   0, -30,
    -40, -20,   0,   0,   0,   0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10,   5,   0,   0,   0,   0,   5, -10,
    -10,  10,  10,  10,  10,  10,  10, -10,
    -10,   0,  10,  10,  10,  10,   0, -10,
    -10,   5,   5,  10,  10,   5,   5, -10,
    -10,   0,   5,  10,  10,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_PST = [
      0,   0,   0,   5,   5,   0,   0,   0,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
     -5,   0,   0,   0,   0,   0,   0,  -5,
      5,  10,  10,  10,  10,  10,  10,   5,
      0,   0,   0,   0,   0,   0,   0,   0,
]
QUEEN_PST = [
    -20, -10, -10,  -5,  -5, -10, -10, -20,
    -10,   0,   5,   0,   0,   0,   0, -10,
    -10,   5,   5,   5,   5,   5,   0, -10,
      0,   0,   5,   5,   5,   5,   0,  -5,
     -5,   0,   5,   5,   5,   5,   0,  -5,
    -10,   0,   5,   5,   5,   5,   0, -10,
    -10,   0,   0,   0,   0,   0,   0, -10,
    -20, -10, -10,  -5,  -5, -10, -10, -20,
]
# King: middlegame table (favour castled safety, penalise centre).
KING_PST_MID = [
     20,  30,  10,   0,   0,  10,  30,  20,
     20,  20,   0,   0,   0,   0,  20,  20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
]
# King: endgame table (centralise the king when few pieces remain).
KING_PST_END = [
    -50, -30, -30, -30, -30, -30, -30, -50,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -50, -40, -30, -20, -20, -30, -40, -50,
]

PST = {
    chess.PAWN: PAWN_PST,
    chess.KNIGHT: KNIGHT_PST,
    chess.BISHOP: BISHOP_PST,
    chess.ROOK: ROOK_PST,
    chess.QUEEN: QUEEN_PST,
}

# Endgame detection: total non-pawn material (per side) at/below this -> endgame.
ENDGAME_MATERIAL_THRESHOLD = 1300  # ~ a queen + a minor, or two rooks


class TimeUp(Exception):
    """Raised internally to abort iterative deepening when the clock runs out."""


@dataclass
class SearchResult:
    move: Optional[chess.Move]
    score: int          # centipawns, side-to-move relative at the root
    depth: int          # last fully-completed depth
    nodes: int
    time_s: float
    pv: list = field(default_factory=list)  # principal variation (chess.Move list)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _is_endgame(board: chess.Board) -> bool:
    """Endgame if BOTH sides' non-pawn material is low (queens often traded)."""
    npm = {chess.WHITE: 0, chess.BLACK: 0}
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        v = PIECE_VALUE[piece_type]
        npm[chess.WHITE] += len(board.pieces(piece_type, chess.WHITE)) * v
        npm[chess.BLACK] += len(board.pieces(piece_type, chess.BLACK)) * v
    return (npm[chess.WHITE] <= ENDGAME_MATERIAL_THRESHOLD and
            npm[chess.BLACK] <= ENDGAME_MATERIAL_THRESHOLD)


def evaluate(board: chess.Board) -> int:
    """
    Static evaluation in centipawns, SIDE-TO-MOVE relative (negamax convention):
    positive => the side to move is better.

    Components: material + PST + mobility + a small centre/king-safety nudge.
    Terminal positions are handled by the search (mate/stalemate), but we guard
    here too so direct calls are safe.
    """
    if board.is_checkmate():
        # Side to move has been mated -> worst possible.
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    endgame = _is_endgame(board)
    king_pst = KING_PST_END if endgame else KING_PST_MID

    score = 0  # from White's perspective; flipped at the end.

    for square, piece in board.piece_map().items():
        val = PIECE_VALUE[piece.piece_type]
        if piece.piece_type == chess.KING:
            pst = king_pst
        else:
            pst = PST[piece.piece_type]
        # For White read the table directly; for Black mirror the rank (xor 56).
        if piece.color == chess.WHITE:
            pos = pst[square]
            score += val + pos
        else:
            pos = pst[square ^ 56]
            score -= val + pos

    # Mobility: legal-move count differential (cheap, one make/unmake of turn).
    # Count this side's legal moves, then the opponent's via a null-ish flip.
    mob_stm = board.legal_moves.count()
    board.push(chess.Move.null())
    mob_opp = board.legal_moves.count()
    board.pop()
    mobility = (mob_stm - mob_opp)  # from side-to-move perspective
    # Convert mobility to White-perspective so it composes with `score`.
    if board.turn == chess.WHITE:
        score += 4 * mobility
    else:
        score -= 4 * mobility

    # Small bishop-pair bonus.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30

    # King-safety nudge: being in check is uncomfortable (side-to-move relative
    # already captured by mobility, but add an explicit small term).
    if board.is_check():
        # side to move is in check -> bad for them (White-perspective adjust)
        if board.turn == chess.WHITE:
            score -= 20
        else:
            score += 20

    # Flip to side-to-move relative.
    return score if board.turn == chess.WHITE else -score


# --------------------------------------------------------------------------- #
# Upgraded evaluation (evaluate_v2): everything in evaluate() PLUS the terms a
# shallow search cannot discover for itself -- pawn structure (doubled / isolated
# / passed), a real king pawn-shield, rook-on-open-file, and a TAPERED (smoothly
# interpolated) king PST instead of a hard mid/endgame switch. These are classic,
# proven eval terms; they are the highest Elo-per-ply win within the python speed
# cap. Selectable via Engine(eval_fn=evaluate_v2) and MEASURED head-to-head.
# --------------------------------------------------------------------------- #
# Precomputed bitboard masks (built once at import; cheap bitwise structure tests).
_FILE_BB = list(chess.BB_FILES)                       # a..h, each an int bitboard
_ADJ_FILES_BB = []                                    # union of file f and its neighbours
for _f in range(8):
    _m = _FILE_BB[_f]
    if _f > 0:
        _m |= _FILE_BB[_f - 1]
    if _f < 7:
        _m |= _FILE_BB[_f + 1]
    _ADJ_FILES_BB.append(_m)
_NEIGHBOUR_FILES_BB = [_ADJ_FILES_BB[_f] ^ _FILE_BB[_f] for _f in range(8)]  # neighbours only
_WHITE_AHEAD_BB = [0] * 8                              # squares on ranks strictly above r
_BLACK_AHEAD_BB = [0] * 8                              # squares on ranks strictly below r
for _r in range(8):
    _wm = 0
    _bm = 0
    for _rr in range(_r + 1, 8):
        _wm |= chess.BB_RANKS[_rr]
    for _rr in range(0, _r):
        _bm |= chess.BB_RANKS[_rr]
    _WHITE_AHEAD_BB[_r] = _wm
    _BLACK_AHEAD_BB[_r] = _bm

# Passed-pawn bonus by rank (from White's view: rank index 0=back .. 7=promote).
_PASSED_BONUS = [0, 8, 16, 28, 48, 80, 130, 0]
_DOUBLED_PENALTY = 12
_ISOLATED_PENALTY = 14
_ROOK_OPEN_FILE = 24
_ROOK_SEMIOPEN_FILE = 12
_SHIELD_MISSING_PENALTY = 12

# Opening non-pawn material per side (2N+2B+2R+Q) -> phase=1.0 (full middlegame).
_OPENING_NPM = 2 * PIECE_VALUE[chess.KNIGHT] + 2 * PIECE_VALUE[chess.BISHOP] + \
    2 * PIECE_VALUE[chess.ROOK] + PIECE_VALUE[chess.QUEEN]


def _pawn_structure(white_pawns: int, black_pawns: int) -> int:
    """White-perspective centipawn score for pawn structure (doubled/isolated/passed)."""
    s = 0
    for pawns, ahead, enemy, sign in ((white_pawns, _WHITE_AHEAD_BB, black_pawns, 1),
                                      (black_pawns, _BLACK_AHEAD_BB, white_pawns, -1)):
        bb = pawns
        while bb:
            sq = (bb & -bb).bit_length() - 1     # least-significant set square
            bb &= bb - 1
            f = sq & 7
            r = sq >> 3
            # Isolated: no friendly pawn on a neighbouring file.
            if not (pawns & _NEIGHBOUR_FILES_BB[f]):
                s -= sign * _ISOLATED_PENALTY
            # Passed: no enemy pawn on this/adjacent files anywhere ahead.
            if not (enemy & _ADJ_FILES_BB[f] & ahead[r]):
                bonus = _PASSED_BONUS[r] if sign == 1 else _PASSED_BONUS[7 - r]
                s += sign * bonus
        # Doubled: each extra pawn on a file beyond the first.
        for f in range(8):
            cnt = bin(pawns & _FILE_BB[f]).count("1")
            if cnt > 1:
                s -= sign * _DOUBLED_PENALTY * (cnt - 1)
    return s


def _king_shield(board: chess.Board, color: bool, friendly_pawns: int) -> int:
    """Penalty (>=0) for missing pawn-shield squares in front of `color`'s king."""
    ks = board.king(color)
    if ks is None:
        return 0
    kf = ks & 7
    kr = ks >> 3
    missing = 0
    step = 1 if color == chess.WHITE else -1
    for dr in (1, 2):
        r = kr + step * dr
        if r < 0 or r > 7:
            continue
        for f in (kf - 1, kf, kf + 1):
            if 0 <= f <= 7:
                sq = r * 8 + f
                if not (friendly_pawns & (1 << sq)):
                    missing += 1
    return missing * _SHIELD_MISSING_PENALTY


def evaluate_v2(board: chess.Board) -> int:
    """Upgraded static eval (centipawns, side-to-move relative). Superset of evaluate():
    material + tapered PST + mobility + bishop pair + check, PLUS pawn structure, a king
    pawn-shield, and rook-on-(semi)open-file. See the section header for rationale."""
    if board.is_checkmate():
        return -MATE_SCORE
    if board.is_stalemate() or board.is_insufficient_material() or \
            board.is_seventyfive_moves() or board.is_fivefold_repetition():
        return 0

    white_pawns = board.pieces_mask(chess.PAWN, chess.WHITE)
    black_pawns = board.pieces_mask(chess.PAWN, chess.BLACK)

    # Game phase from non-pawn material (1.0 = full middlegame, 0.0 = bare endgame).
    npm = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        v = PIECE_VALUE[pt]
        npm += v * (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK)))
    phase = max(0.0, min(1.0, npm / (2.0 * _OPENING_NPM)))

    score = 0  # White perspective.
    for square, piece in board.piece_map().items():
        val = PIECE_VALUE[piece.piece_type]
        if piece.color == chess.WHITE:
            idx = square
        else:
            idx = square ^ 56
        if piece.piece_type == chess.KING:
            # Tapered king PST: interpolate mid<->end by phase (smooth, no hard switch).
            pos = KING_PST_MID[idx] * phase + KING_PST_END[idx] * (1.0 - phase)
            pos = int(pos)
        else:
            pos = PST[piece.piece_type][idx]
        if piece.color == chess.WHITE:
            score += val + pos
        else:
            score -= val + pos

    # Mobility (same as evaluate()).
    mob_stm = board.legal_moves.count()
    board.push(chess.Move.null())
    mob_opp = board.legal_moves.count()
    board.pop()
    mobility = mob_stm - mob_opp
    score += (4 * mobility) if board.turn == chess.WHITE else (-4 * mobility)

    # Bishop pair.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30

    # Pawn structure (doubled / isolated / passed).
    score += _pawn_structure(white_pawns, black_pawns)

    # Rook on open / semi-open file.
    for color, sign, own_pawns in ((chess.WHITE, 1, white_pawns), (chess.BLACK, -1, black_pawns)):
        for rsq in board.pieces(chess.ROOK, color):
            fbb = _FILE_BB[rsq & 7]
            if not (own_pawns & fbb):
                score += sign * (_ROOK_OPEN_FILE if not ((white_pawns | black_pawns) & fbb)
                                 else _ROOK_SEMIOPEN_FILE)

    # King pawn-shield (scaled by phase -- only matters with pieces on the board).
    if phase > 0.3:
        w_missing = _king_shield(board, chess.WHITE, white_pawns)
        b_missing = _king_shield(board, chess.BLACK, black_pawns)
        score -= int(w_missing * phase)
        score += int(b_missing * phase)

    # Check nudge.
    if board.is_check():
        score += -20 if board.turn == chess.WHITE else 20

    return score if board.turn == chess.WHITE else -score


# --------------------------------------------------------------------------- #
# Move ordering (MVV-LVA: most-valuable-victim / least-valuable-attacker)
# --------------------------------------------------------------------------- #
def _mvv_lva_key(board: chess.Board, move: chess.Move) -> int:
    """Higher = search earlier. Captures (esp. big victim / small attacker) first,
    then promotions, then quiet moves."""
    score = 0
    if board.is_capture(move):
        victim_sq = move.to_square
        if board.is_en_passant(move):
            victim_val = PIECE_VALUE[chess.PAWN]
        else:
            victim_piece = board.piece_at(victim_sq)
            victim_val = PIECE_VALUE[victim_piece.piece_type] if victim_piece else 0
        attacker_piece = board.piece_at(move.from_square)
        attacker_val = PIECE_VALUE[attacker_piece.piece_type] if attacker_piece else 0
        # MVV-LVA: big victim, small attacker => high priority.
        score += 10_000 + victim_val * 10 - attacker_val
    if move.promotion:
        score += 9_000 + PIECE_VALUE.get(move.promotion, 0)
    if board.gives_check(move):
        score += 500
    return score


def _ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: _mvv_lva_key(board, m), reverse=True)
    return moves


def _ordered_captures(board: chess.Board):
    caps = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    caps.sort(key=lambda m: _mvv_lva_key(board, m), reverse=True)
    return caps


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
class Engine:
    def __init__(self, depth: int = 4, time_limit: Optional[float] = None,
                 quiescence: bool = True, q_max_depth: int = 6, strong: bool = False,
                 eval_fn=None, book=None):
        """
        depth:      maximum iterative-deepening depth (plies).
        time_limit: wall-clock seconds per move; None => depth-bounded only.
        quiescence: run capture-only quiescence at the leaves.
        q_max_depth: cap on quiescence recursion (safety against capture loops).
        eval_fn:    static evaluation function (board -> centipawns, side-to-move relative).
                    Defaults to `evaluate_v2` (richer: tapered PST + pawn structure + king
                    pawn-shield + rook-on-open-file). MEASURED 2026-06-13 to beat the older
                    `evaluate` by +108 Elo (W14-D11-L5 over 30 games @ 0.2s/move) -- so it is the
                    default per the integrity rule (an eval change must be MEASURED to win first).
                    Pass `eval_fn=evaluate` to fall back to the simpler eval for comparison.
        strong:     enable the strong-search machinery -- transposition table, null-move pruning,
                    late-move reductions, killer/history + TT-move ordering. MEASURED 2026-06-13:
                    at the depth pure-python-chess reaches in interactive time (3-5 ply) this is
                    ~strength-NEUTRAL (the +1 ply it buys is offset by the pruning's accuracy
                    loss; it pays off at depth 8+, which needs a faster core -- see
                    docs/STRENGTH_ROADMAP.md). So it DEFAULTS OFF (no regression, no overhead);
                    `strong=True` opts in (and will be the win once a Cython/bitboard core exists).
        """
        self.max_depth = depth
        self.time_limit = time_limit
        self.quiescence = quiescence
        self.q_max_depth = q_max_depth
        self.strong = strong
        self._eval = eval_fn if eval_fn is not None else evaluate_v2
        self.book = book                                # optional OpeningBook (instant strong opening moves)
        self.nodes = 0
        self._start = 0.0
        self._deadline = None
        self._tt = {}                                  # zobrist_key -> (depth, flag, score, move)
        self._killers = [[None, None] for _ in range(128)]
        self._history = {}                             # (piece_type, to_square) -> score

    # --- time control --------------------------------------------------------
    def _check_time(self):
        if self._deadline is not None and time.perf_counter() >= self._deadline:
            raise TimeUp()

    # --- quiescence ----------------------------------------------------------
    def _quiesce(self, board: chess.Board, alpha: int, beta: int, qdepth: int) -> int:
        """Search only 'noisy' moves (captures/promotions) until quiet, so the
        static eval is applied to a stable position (mitigates horizon effect)."""
        self.nodes += 1
        self._check_time()

        stand_pat = self._eval(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        if qdepth <= 0:
            return alpha

        for move in _ordered_captures(board):
            board.push(move)
            try:
                score = -self._quiesce(board, -beta, -alpha, qdepth - 1)
            finally:
                board.pop()  # restore the board even if TimeUp is raised
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    # --- strong move ordering: TT move, MVV-LVA captures, killers, history ---
    def _order_moves(self, board: chess.Board, ply: int, tt_move):
        killers = self._killers[ply] if ply < len(self._killers) else (None, None)
        hist = self._history

        def key(m):
            if m == tt_move:
                return 2_000_000
            if board.is_capture(m) or m.promotion:
                return 1_000_000 + _mvv_lva_key(board, m)
            if m == killers[0]:
                return 900_000
            if m == killers[1]:
                return 890_000
            pc = board.piece_at(m.from_square)
            return hist.get((pc.piece_type, m.to_square), 0) if pc else 0

        moves = list(board.legal_moves)
        moves.sort(key=key, reverse=True)
        return moves

    def _store_killer(self, ply: int, move: chess.Move):
        if ply < len(self._killers):
            k = self._killers[ply]
            if k[0] != move:
                k[1] = k[0]
                k[0] = move

    @staticmethod
    def _has_non_pawn_material(board: chess.Board) -> bool:
        stm = board.turn
        return bool(board.pieces(chess.KNIGHT, stm) or board.pieces(chess.BISHOP, stm)
                    or board.pieces(chess.ROOK, stm) or board.pieces(chess.QUEEN, stm))

    # --- negamax + alpha-beta (TT + null-move + LMR when strong) -------------
    def _negamax(self, board: chess.Board, depth: int, alpha: int, beta: int,
                 ply: int) -> int:
        self.nodes += 1
        self._check_time()
        alpha_orig = alpha

        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_stalemate() or board.is_insufficient_material() or \
                board.is_seventyfive_moves() or board.is_fivefold_repetition() or \
                board.can_claim_threefold_repetition():
            return 0

        tt_move = None
        key = None
        if self.strong:
            key = chess.polyglot.zobrist_hash(board)
            ent = self._tt.get(key)
            if ent is not None:
                e_depth, e_flag, e_score, e_move = ent
                tt_move = e_move
                if e_depth >= depth:
                    if e_flag == _TT_EXACT:
                        return e_score
                    if e_flag == _TT_LOWER:
                        if e_score > alpha:
                            alpha = e_score
                    elif e_flag == _TT_UPPER:
                        if e_score < beta:
                            beta = e_score
                    if alpha >= beta:
                        return e_score

        if depth <= 0:
            if self.quiescence:
                return self._quiesce(board, alpha, beta, self.q_max_depth)
            return self._eval(board)

        in_check = board.is_check()

        # Null-move pruning: give the opponent a free move; if we are STILL >= beta the
        # position is so strong we can prune. Skip in check / near-mate / pawn-only (zugzwang).
        if (self.strong and not in_check and depth >= 3 and beta < MATE_THRESHOLD
                and self._has_non_pawn_material(board)):
            R = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            try:
                score = -self._negamax(board, depth - 1 - R, -beta, -beta + 1, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta

        moves = self._order_moves(board, ply, tt_move) if self.strong else _ordered_moves(board)

        best = -MATE_SCORE - 1
        best_move = None
        move_idx = 0
        for move in moves:
            is_quiet = not (board.is_capture(move) or move.promotion)
            board.push(move)
            try:
                gives_check = board.is_check()
                if (self.strong and depth >= 3 and move_idx >= 4 and is_quiet
                        and not in_check and not gives_check):
                    red = 2 if move_idx >= 8 else 1            # late-move reduction
                    score = -self._negamax(board, depth - 1 - red, -alpha - 1, -alpha, ply + 1)
                    if score > alpha:                         # promising -> full re-search
                        score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
                else:
                    score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()  # restore the board even if TimeUp is raised
            if score > best:
                best = score
                best_move = move
            if best > alpha:
                alpha = best
            if alpha >= beta:
                if self.strong and is_quiet:                  # record cutoff heuristics
                    self._store_killer(ply, move)
                    pc = board.piece_at(move.from_square)
                    if pc:
                        hk = (pc.piece_type, move.to_square)
                        self._history[hk] = self._history.get(hk, 0) + depth * depth
                break
            move_idx += 1

        if self.strong:
            if len(self._tt) >= _TT_MAX:
                self._tt.clear()
            flag = _TT_UPPER if best <= alpha_orig else (_TT_LOWER if best >= beta else _TT_EXACT)
            self._tt[key] = (depth, flag, best, best_move)
        return best

    # --- root search with PV extraction --------------------------------------
    def _search_root(self, board: chess.Board, depth: int):
        alpha, beta = -MATE_SCORE - 1, MATE_SCORE + 1
        best_move = None
        best_score = -MATE_SCORE - 1
        if self.strong:
            ent = self._tt.get(chess.polyglot.zobrist_hash(board))
            moves = self._order_moves(board, 0, ent[3] if ent else None)
        else:
            moves = _ordered_moves(board)
        for move in moves:
            board.push(move)
            try:
                score = -self._negamax(board, depth - 1, -beta, -alpha, 1)
            finally:
                board.pop()  # restore the board even if TimeUp is raised
            if score > best_score:
                best_score = score
                best_move = move
            if best_score > alpha:
                alpha = best_score
        if self.strong and best_move is not None:
            self._tt[chess.polyglot.zobrist_hash(board)] = (depth, _TT_EXACT, best_score, best_move)
        return best_move, best_score

    def _extract_pv(self, board: chess.Board, max_len: int = 12) -> list:
        """Re-derive a short principal variation by greedily picking the best
        move at shallow depth from the resulting position (cheap, display-only)."""
        pv = []
        b = board.copy()
        for _ in range(max_len):
            if b.is_game_over():
                break
            mv, _ = self._search_root(b, 1)
            if mv is None:
                break
            pv.append(mv)
            b.push(mv)
        return pv

    def search(self, board: chess.Board) -> SearchResult:
        """Iterative deepening: search depth 1, 2, ..., max_depth, keeping the
        best move from the last fully-completed depth if time runs out."""
        self.nodes = 0
        self._killers = [[None, None] for _ in range(128)]   # fresh killers per move
        self._start = time.perf_counter()
        self._deadline = (self._start + self.time_limit
                          if self.time_limit is not None else None)

        best_move = None
        best_score = 0
        completed_depth = 0

        # Guarantee a legal move even if depth 1 is interrupted.
        legal = list(board.legal_moves)
        if not legal:
            return SearchResult(None, 0, 0, self.nodes,
                                time.perf_counter() - self._start, [])
        best_move = legal[0]

        # Opening book: a strong (Stockfish-labelled) move played INSTANTLY, no search.
        if self.book is not None:
            bm = self.book.get(board)
            if bm is not None:
                return SearchResult(move=bm, score=0, depth=0, nodes=0,
                                    time_s=time.perf_counter() - self._start, pv=[bm])

        for d in range(1, self.max_depth + 1):
            try:
                mv, sc = self._search_root(board, d)
                if mv is not None:
                    best_move, best_score = mv, sc
                    completed_depth = d
                # Early exit on a found forced mate.
                if abs(best_score) >= MATE_THRESHOLD:
                    break
            except TimeUp:
                break

        # PV extraction is display-only and cheap (depth-1 greedy). It must NOT
        # be subject to the move deadline -- clear it first, else a TimeUp raised
        # inside _search_root would escape search() and crash the game loop.
        self._deadline = None
        try:
            pv = self._extract_pv(board)
        except TimeUp:
            pv = [best_move] if best_move is not None else []
        return SearchResult(
            move=best_move,
            score=best_score,
            depth=completed_depth,
            nodes=self.nodes,
            time_s=time.perf_counter() - self._start,
            pv=pv,
        )


# --------------------------------------------------------------------------- #
# Convenience API
# --------------------------------------------------------------------------- #
def best_move(board: chess.Board, depth: int = 4,
              time_limit: Optional[float] = None):
    """Return (best_move, info_dict). info has score_cp, depth, nodes, time_s, pv (SAN)."""
    eng = Engine(depth=depth, time_limit=time_limit)
    res = eng.search(board)
    # Build SAN PV against a scratch board.
    san_pv = []
    b = board.copy()
    for m in res.pv:
        try:
            san_pv.append(b.san(m))
            b.push(m)
        except Exception:
            break
    info = {
        "score_cp": res.score,
        "depth": res.depth,
        "nodes": res.nodes,
        "time_s": res.time_s,
        "nps": int(res.nodes / res.time_s) if res.time_s > 0 else 0,
        "pv_san": san_pv,
    }
    return res.move, info


if __name__ == "__main__":
    # Tiny self-check: search the opening position at depth 4.
    board = chess.Board()
    mv, info = best_move(board, depth=4)
    print(f"Best opening move (depth 4): {board.san(mv)}  {info}")
