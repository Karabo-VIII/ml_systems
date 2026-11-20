"""
chess_zero.az.batched_selfplay -- GPU-batched, GAME-PARALLEL self-play (the throughput knob).

WHY: a single 4060 is wildly under-used at batch=1, which is what plain MCTS self-play does --
one net forward per simulation per game. This module runs G self-play games IN LOCKSTEP and
batches their per-simulation leaf evaluations into ONE forward (net.predict_many), so each GPU
launch does up to G leaf-evals instead of 1. That is the "control the rate of parallel
instances" lever: --parallel-games G trades a little host bookkeeping for G-fold fewer GPU
round-trips, i.e. more self-play games per unit time.

GAME parallelism, NOT tree parallelism: within ONE game the simulations are still SEQUENTIAL
(select -> expand -> backup, one leaf at a time), exactly like the sequential MCTS -- we only
batch the EVAL across DIFFERENT games at the same simulation index. So no virtual loss is needed
and each game's search semantics are unchanged from mcts.MCTS. With G=1 this reduces to the
ordinary single-game search (one game, batch=1).

It reuses the canonical primitives (mcts.Node + PUCT, encoding, train_robust's _move_from_visits
/ _adjudicate) so there is ONE source of truth for the search math and the adjudication -- this
module only adds the batching orchestration. SELF-play only (net plays both sides); the 'teacher'
opponent path stays on the sequential generator (an external engine is not batchable the same way).

Output: List[List[Sample]] (one inner list per game), Samples identical in schema to
selfplay.generate_selfplay_game (planes, pi, player, z) -- a drop-in batch producer. No emoji.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional

import numpy as np
import chess

from .mcts import Node
from .encoding import board_to_planes, legal_policy_mask, move_to_index, N_POLICY
from .selfplay import Sample
from .openings import sample_opening_board


def _select_child(node: Node, c_puct: float):
    """PUCT child selection -- identical formula to mcts.MCTS._select_child."""
    sqrt_total = math.sqrt(max(1, node.visit_count))
    best_score, best_move, best_child = -float("inf"), None, None
    for move, child in node.children.items():
        q = -child.value()
        u = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score, best_move, best_child = score, move, child
    return best_move, best_child


def _terminal_value(board: chess.Board) -> Optional[float]:
    """Side-to-move terminal value, identical to mcts.MCTS._terminal_value."""
    if board.is_checkmate():
        return -1.0
    if board.is_game_over(claim_draw=True):
        return 0.0
    return None


def _priors_from_probs(probs: np.ndarray, idx_to_move: dict) -> dict:
    """Build {move: prior} over legal moves from a masked-softmax probability vector,
    renormalising over legals (matches mcts.MCTS._evaluate)."""
    priors = {mv: float(probs[idx]) for idx, mv in idx_to_move.items()}
    total = sum(priors.values())
    if total > 0:
        priors = {m: p / total for m, p in priors.items()}
    else:
        n = max(1, len(idx_to_move))
        priors = {m: 1.0 / n for m in idx_to_move}
    return priors


def _expand(node: Node, priors: dict, board_turn: bool) -> None:
    for move, p in priors.items():
        node.children[move] = Node(prior=p, to_play=not board_turn)
    node.is_expanded = True


def _backup(path: List[Node], value: float) -> None:
    for n in reversed(path):
        n.visit_count += 1
        n.value_sum += value
        value = -value


def _add_dirichlet_noise(root: Node, alpha: float, eps: float, rng: np.random.Generator) -> None:
    if not root.children:
        return
    moves = list(root.children.keys())
    noise = rng.dirichlet([alpha] * len(moves))
    for mv, nz in zip(moves, noise):
        ch = root.children[mv]
        ch.prior = (1 - eps) * ch.prior + eps * nz


def _move_from_visits_rng(visits: dict, temperature: float, rng: np.random.Generator):
    """Same rule as train_robust._move_from_visits but with an injected RNG (reproducible across
    the batch). argmax for temperature<=1e-6, else sample proportional to N^(1/temperature)."""
    moves = list(visits.keys())
    if not moves:
        return None
    counts = np.array([visits[m] for m in moves], dtype=np.float64)
    if counts.sum() == 0:
        return moves[0]
    if temperature <= 1e-6:
        return moves[int(counts.argmax())]
    probs = counts ** (1.0 / temperature)
    probs = probs / probs.sum()
    return moves[int(rng.choice(len(moves), p=probs))]


class _Game:
    """Per-game lockstep state. `start_board` is the (possibly diverse) opening
    position this game begins from; ply/temperature count the NET's own moves from
    that opening (the opening plies are never recorded as training samples)."""
    __slots__ = ("board", "samples", "ply", "t0", "done", "root")

    def __init__(self, start_board: Optional[chess.Board] = None):
        self.board = start_board.copy() if start_board is not None else chess.Board()
        self.samples: List[Sample] = []
        self.ply = 0
        self.t0 = time.time()
        self.done = False
        self.root: Optional[Node] = None


def generate_selfplay_games_batched(
        net, n_games: int, n_simulations: int, temp_moves: int, max_plies: int,
        game_wall_s: float, device, c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
        seed: int = 0, opening_mode: str = "startpos",
        opening_plies: int = 4) -> List[List[Sample]]:
    """Play `n_games` SELF-play games in lockstep, batching leaf evals across games.

    Returns a list (length n_games) of per-game Sample lists, z filled (real result, or material
    adjudication when a game hits the ply/wall-clock cap). NEVER hangs: per-game ply cap +
    wall-clock guard, identical to generate_selfplay_game_guarded. SELF-play only.

    OPENING DIVERSITY (opening_mode != "startpos"): each game starts from a distinct, sound
    opening sampled from openings.sample_opening_board (book/random/mixed) so the games -- and
    the value targets -- are not all the same line (the "vary the starting conditions" fix).
    The opening plies are NOT recorded as samples; only the net's searched moves from the
    opening onward are. opening_mode="startpos" (default here) reproduces the old behaviour.
    """
    rng = np.random.default_rng(seed)
    games = [_Game(sample_opening_board(rng, mode=opening_mode, random_plies=opening_plies))
             for _ in range(n_games)]

    def _batched_eval(items):
        """items: list of (game_idx, board). Returns {game_idx: (priors, value)} via ONE forward."""
        if not items:
            return {}
        planes, masks, metas = [], [], []
        for gi, board in items:
            mask, idx_to_move = legal_policy_mask(board)
            planes.append(board_to_planes(board))
            masks.append(mask)
            metas.append((gi, board.turn, idx_to_move))
        out = net.predict_many(planes, masks, device=device)
        res = {}
        for (gi, turn, idx_to_move), (probs, value) in zip(metas, out):
            res[gi] = (_priors_from_probs(probs, idx_to_move), value, turn)
        return res

    while not all(g.done for g in games):
        active = [i for i, g in enumerate(games) if not g.done]

        # wall-clock / ply guard checked per move-step
        for i in active:
            g = games[i]
            if (time.time() - g.t0 > game_wall_s) or g.ply >= max_plies:
                g.done = True
        active = [i for i in active if not games[i].done]
        if not active:
            break

        # 1) fresh root per active game; EXPAND all roots in ONE batch
        for i in active:
            games[i].root = Node(prior=1.0, to_play=games[i].board.turn)
        root_eval = _batched_eval([(i, games[i].board) for i in active])
        for i in active:
            priors, value, turn = root_eval[i]
            _expand(games[i].root, priors, games[i].board.turn)
            temperature = 1.0 if games[i].ply < temp_moves else 0.0
            if temperature > 0:
                _add_dirichlet_noise(games[i].root, dirichlet_alpha, dirichlet_eps, rng)

        # 2) run n_simulations; at each sim, SELECT a leaf per game, batch-EVAL, EXPAND+BACKUP
        for _ in range(n_simulations):
            leaves = []   # (game_idx, leaf_node, scratch_board, path, terminal_value_or_None)
            to_eval = []  # (game_idx, scratch_board) for non-terminal leaves
            for i in active:
                node = games[i].root
                scratch = games[i].board.copy()
                path = [node]
                while node.is_expanded and node.children:
                    mv, child = _select_child(node, c_puct)
                    if mv is None:
                        break
                    scratch.push(mv)
                    node = child
                    path.append(node)
                term = _terminal_value(scratch)
                leaves.append([i, node, scratch, path, term])
                if term is None:
                    to_eval.append((i, scratch))
            ev = _batched_eval(to_eval)
            # NOTE: a game can only contribute ONE leaf per sim, so game_idx -> one eval is unique.
            for entry in leaves:
                i, node, scratch, path, term = entry
                if term is not None:
                    _backup(path, term)
                else:
                    priors, value, turn = ev[i]
                    _expand(node, priors, scratch.turn)
                    _backup(path, value)

        # 3) each active game: record the Sample at the CURRENT position, play a move, advance
        for i in active:
            g = games[i]
            visits = {mv: ch.visit_count for mv, ch in g.root.children.items()}
            pi = np.zeros(N_POLICY, dtype=np.float32)
            total = sum(visits.values())
            if total > 0:
                for mv, n in visits.items():
                    idx = move_to_index(g.board, mv)
                    if idx is not None:
                        pi[idx] = n / total
            g.samples.append(Sample(planes=board_to_planes(g.board), pi=pi, player=g.board.turn))
            temperature = 1.0 if g.ply < temp_moves else 0.0
            move = _move_from_visits_rng(visits, temperature, rng)
            if move is None or move not in g.board.legal_moves:
                move = next(iter(g.board.legal_moves))
            g.board.push(move)
            g.ply += 1
            if g.board.is_game_over(claim_draw=True) or g.ply >= max_plies:
                g.done = True

    # assign z per game: real result if finished cleanly, else material adjudication
    from .train_robust import _adjudicate  # lazy import to avoid an import cycle
    for g in games:
        if g.board.is_game_over(claim_draw=True):
            result = g.board.result(claim_draw=True)
            winner = (chess.WHITE if result == "1-0"
                      else chess.BLACK if result == "0-1" else None)
        else:
            winner = _adjudicate(g.board)  # ply/wall-clock cap hit
        for s in g.samples:
            s.z = 0.0 if winner is None else (1.0 if s.player == winner else -1.0)

    return [g.samples for g in games]
