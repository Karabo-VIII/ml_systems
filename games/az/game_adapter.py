"""
chess_zero.az.game_adapter -- the ENGINE-AGNOSTIC contract + a working non-chess proof.

THE IDEA: the AlphaZero-lineage pipeline (search + net + self-play + train + gate + the whole
safety/eval/infra stack in train_robust.py) is the SAME for ANY perfect-information game. Only
three things are engine-specific: the RULES, the STATE/observation encoding, and the ACTION space.
So "give me a DIFFERENT engine to solve" reduces to "implement ~7 methods" -- the GameAdapter below.
The generic search/training pipeline then consumes the adapter unchanged.

The contract is distilled from OpenSpiel's Game/State API + the alpha-zero-general `Game` interface
(see docs/ENGINE_AGNOSTIC_FRAMEWORK.md for the mapping + the architecture decision tree that picks
AlphaZero vs MuZero vs Gumbel/Sampled/Stochastic for a new engine).

This file is SELF-CONTAINED and PROVES generality without a GPU/net: it includes a generic UCT
MCTS that runs over ANY GameAdapter, a TicTacToe adapter, and a __main__ smoke showing the generic
search SOLVES a non-chess engine (never loses to a random opponent). The neural pipeline swaps the
random rollout for the net's (policy prior + value) = PUCT -- the adapter interface is identical;
that's the whole point. No emoji (Windows cp1252).
"""
from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional


# --------------------------------------------------------------------------- #
# THE CONTRACT: implement these for a new engine; the pipeline consumes them.
# `state` is any object YOU choose (a tuple, a board, a FEN string) -- the
# pipeline treats it opaquely and only calls these methods. apply() MUST be pure
# (return a NEW state; never mutate the input) so the search can branch freely.
# --------------------------------------------------------------------------- #
class GameAdapter(ABC):
    """The minimal interface a perfect-information game implements to be solvable by the generic
    AlphaZero-lineage pipeline. (Imperfect-info / stochastic / continuous-action engines add a few
    methods -- see ENGINE_AGNOSTIC_FRAMEWORK.md; this base covers the AlphaZero case.)"""

    name: str = "game"

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Size of the FLAT action space -> the policy head width (e.g. chess 4672, TicTacToe 9)."""

    @abstractmethod
    def initial_state(self) -> Any:
        """The start state. (For opening diversity, a pipeline may instead sample from a set of
        states -- the openings.py idea generalized.)"""

    @abstractmethod
    def current_player(self, state) -> int:
        """Whose turn: 0 or 1 (2-player zero-sum). Drives the per-ply value-sign convention."""

    @abstractmethod
    def legal_actions(self, state) -> List[int]:
        """The legal action indices in [0, num_actions). This is the legal-move MASK."""

    @abstractmethod
    def apply(self, state, action: int) -> Any:
        """Return the NEXT state after `action`. PURE -- must not mutate `state`."""

    @abstractmethod
    def is_terminal(self, state) -> bool:
        ...

    @abstractmethod
    def returns(self, state) -> float:
        """Terminal value from PLAYER 0's perspective in [-1, 1] (+1 p0 wins, -1 p1 wins, 0 draw).
        The pipeline negates per ply for the side-to-move convention (the I1 lesson, generalized)."""

    # ---- optional: only the NEURAL pipeline needs these; the UCT proof below does not ----
    # The NEURAL PUCT search (mcts.NeuralMCTS) needs exactly two things a random rollout does not:
    #   (a) encode(state)              -> the net's input tensor/planes, and
    #   (b) an action<->policy-index map + a legal mask over the policy vector.
    # By the contract's design, legal_actions(state) ALREADY returns indices in [0, num_actions),
    # i.e. policy-vector indices -- so the action<->index map is the IDENTITY by default (chess and
    # TicTacToe both satisfy this: chess legal_actions are encoding.py policy indices; TicTacToe's 9
    # cells map 1:1 to a 9-logit head). A game whose flat action space differs from its policy head
    # width overrides action_to_index / index_to_action below.
    def encode(self, state):
        """state -> np.ndarray input planes/features for the net. Required for the neural (PUCT)
        pipeline (NeuralMCTS). The UCT proof above does NOT need it."""
        raise NotImplementedError(f"{self.name}.encode() needed for the neural pipeline")

    def action_to_index(self, action: int) -> int:
        """Map a legal action (as returned by legal_actions) to its index in the policy vector.
        Default: identity -- legal_actions already yields policy indices (the contract convention)."""
        return action

    def index_to_action(self, index: int) -> int:
        """Inverse of action_to_index: a policy-vector index -> the action the pipeline applies().
        Default: identity. Override together with action_to_index if the two spaces differ."""
        return index

    def legal_policy_mask(self, state):
        """A (num_actions,) {0.,1.} mask over the policy vector marking the legal actions at `state`,
        plus an {index -> action} decode dict. Built generically from legal_actions + action_to_index,
        so a NEW engine gets this for free (no per-engine encoding code). Returns (mask, idx_to_action).
        Lazy-imports numpy so the random-rollout path of this file stays numpy-free."""
        import numpy as np
        mask = np.zeros(self.num_actions, dtype=np.float32)
        idx_to_action: dict = {}
        for a in self.legal_actions(state):
            idx = self.action_to_index(a)
            mask[idx] = 1.0
            idx_to_action[idx] = a
        return mask, idx_to_action

    def symmetries(self, state, policy):
        """Optional data-augmentation: list of (state, policy) under board symmetries (Go=8-fold;
        chess=NONE -- castling/en-passant break symmetry, the C11 lesson). Default: identity only."""
        return [(state, policy)]

    def render(self, state) -> str:
        return str(state)


# --------------------------------------------------------------------------- #
# THE UNBOUNDED-PROBLEM SUPERSET: DecisionProblemAdapter.
# GameAdapter above is the 2-player, terminal, zero-sum, perfect-info, EXACT-SIMULATOR case
# (chess, Go, TicTacToe). Real-world UNBOUNDED problems -- TIME-SERIES and CRYPTO TRADING -- are
# a superset: SINGLE agent vs a stochastic environment, PARTIAL observability (you see features,
# not the true state), CONTINUOUS or large action space, PER-STEP reward (not just terminal),
# (near-)INFINITE horizon, and -- critically -- NO exact simulator. You cannot `apply()` the market.
#
# That last point is the whole game: with no simulator you need a LEARNED dynamics model to plan/
# imagine over. THAT learned model is exactly our WM (src/wm/*). So this is the MuZero / DreamerV3
# branch of the decision tree, and the WM is the dynamics function it requires. This adapter is the
# CONTRACT for that regime; the pipeline that consumes it (plan-over-WM or imagine-train-actor) is
# the next build, NOT claimed here. See docs/ENGINE_AGNOSTIC_FRAMEWORK.md S6.
# --------------------------------------------------------------------------- #
class DecisionProblemAdapter(ABC):
    """The RL/POMDP superset for UNBOUNDED problems (crypto, time-series, control). Worked mapping
    for CRYPTO in the method docstrings. Honest status: this is the interface; no pipeline runs it
    yet (the chess GameAdapter above is the proven instance). It exists so the framework ROUTES an
    unbounded problem instead of pretending it's a board game."""

    name: str = "decision_problem"
    has_exact_simulator: bool = False   # games=True; crypto/real-world=False -> use a LEARNED model (WM)

    @abstractmethod
    def observation(self, history) -> Any:
        """The agent's OBSERVATION (NOT the true state -- partial observability). CRYPTO: the chimera
        feature vector for the current bar/window (past-only; the look-ahead lesson is load-bearing)."""

    @abstractmethod
    def action_spec(self):
        """The action space. CRYPTO: continuous position in [-1, 1] under LO+spot+lev=1 (or a discrete
        set {flat, long, ...}). Returns a spec the policy head + (sampled) search consume."""

    @abstractmethod
    def reward(self, history, action, next_history) -> float:
        """PER-STEP reward. CRYPTO: realized step P&L net of cost (the objective is robust held-out
        COMPOUND return -- per-bar IC is BANNED as the objective; reward must ladder up to compound)."""

    def discount(self) -> float:
        """Horizon discount (1.0 = undiscounted episodic compound)."""
        return 1.0

    def dynamics(self):
        """The LEARNED dynamics model used for planning/imagination when has_exact_simulator is False.
        CRYPTO: the WM (src/wm/*) -- representation + transition over latent market state. THE central
        risk: a policy that PLANS over an imperfect WM learns to exploit the WM's ERRORS (great in
        imagination, fails live) -- so the framework's robustness/eval-trust stack (held-out compound,
        block-bootstrap, seed-robustness, the forgetting/CI gates) is MANDATORY here, not optional."""
        raise NotImplementedError("provide the learned dynamics model (the WM) for the no-simulator case")


# --------------------------------------------------------------------------- #
# GENERIC search over ANY adapter (UCT with random rollouts -- no net needed).
# This is the engine-AGNOSTIC search: the AlphaZero pipeline replaces the random
# rollout with the net's value and the uniform expansion with the net's policy
# prior (PUCT), but the tree + the adapter calls are IDENTICAL. Proving UCT works
# over the adapter proves the interface is sufficient.
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("state", "player", "children", "untried", "N", "W")

    def __init__(self, game: GameAdapter, state):
        self.state = state
        self.player = game.current_player(state)
        self.children: dict = {}                 # action -> _Node
        self.untried: List[int] = list(game.legal_actions(state))
        self.N = 0
        self.W = 0.0                              # sum of PLAYER-0-perspective outcomes through here


def _rollout(game: GameAdapter, state, rng: random.Random) -> float:
    """Random playout to a terminal; return the outcome from PLAYER 0's perspective."""
    while not game.is_terminal(state):
        state = game.apply(state, rng.choice(game.legal_actions(state)))
    return game.returns(state)


def uct_search(game: GameAdapter, root_state, n_sims: int = 200,
               c: float = 1.4, rng: Optional[random.Random] = None) -> int:
    """Generic UCT over a GameAdapter. Returns the chosen action index. Engine-agnostic: it only
    ever calls the adapter's methods. Value perspective: W accumulates player-0-perspective
    outcomes; a node whose player is 0 maximizes W/N, a node whose player is 1 maximizes -W/N
    (negamax sign -- the generalized value-sign invariant)."""
    rng = rng or random.Random()
    if game.is_terminal(root_state):
        raise ValueError("uct_search called on a terminal state")
    root = _Node(game, root_state)

    for _ in range(n_sims):
        node = root
        path = [node]
        # SELECT down to an expandable/terminal node
        while not node.untried and node.children and not game.is_terminal(node.state):
            p = node.player
            logN = math.log(node.N + 1)
            best, best_score = None, -float("inf")
            for a, ch in node.children.items():
                q = ch.W / ch.N if ch.N else 0.0
                q = q if p == 0 else -q                      # value from the mover's perspective
                score = q + c * math.sqrt(logN / (ch.N + 1e-9))
                if score > best_score:
                    best_score, best = score, ch
            node = best
            path.append(node)
        # EXPAND one untried action
        if node.untried and not game.is_terminal(node.state):
            a = node.untried.pop(rng.randrange(len(node.untried)))
            child = _Node(game, game.apply(node.state, a))
            node.children[a] = child
            node = child
            path.append(node)
        # SIMULATE
        z = game.returns(node.state) if game.is_terminal(node.state) else _rollout(game, node.state, rng)
        # BACKUP (player-0 perspective, consistent at every node)
        for nd in path:
            nd.N += 1
            nd.W += z
    # choose the most-visited root action (robust child)
    return max(root.children.items(), key=lambda kv: kv[1].N)[0]


# --------------------------------------------------------------------------- #
# A NON-CHESS engine implementing the contract (the generality proof).
# --------------------------------------------------------------------------- #
class TicTacToe(GameAdapter):
    """3x3 Tic-Tac-Toe. State = (board: tuple of 9 in {0=empty,1=X(p0),2=O(p1)}, player: 0|1)."""

    name = "tictactoe"
    _LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
              (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

    @property
    def num_actions(self) -> int:
        return 9

    def initial_state(self):
        return ((0,) * 9, 0)

    def current_player(self, state) -> int:
        return state[1]

    def legal_actions(self, state) -> List[int]:
        board, _ = state
        return [i for i in range(9) if board[i] == 0]

    def apply(self, state, action: int):
        board, player = state
        if board[action] != 0:
            raise ValueError(f"illegal action {action} on {board}")
        nb = list(board)
        nb[action] = player + 1                              # 1 for p0 (X), 2 for p1 (O)
        return (tuple(nb), 1 - player)

    def _winner(self, board):
        for a, b, cc in self._LINES:
            if board[a] != 0 and board[a] == board[b] == board[cc]:
                return board[a]                              # 1 or 2
        return 0

    def is_terminal(self, state) -> bool:
        board, _ = state
        return self._winner(board) != 0 or all(v != 0 for v in board)

    def returns(self, state) -> float:
        w = self._winner(state[0])
        return 0.0 if w == 0 else (1.0 if w == 1 else -1.0)  # player-0 (X) perspective

    def encode(self, state):
        """state -> (3, 3, 3) float32 planes for the tiny TicTacToe net, from the SIDE-TO-MOVE POV
        (the AlphaZero canonical-orientation convention, so the net is symmetric in colour):
          plane 0 = 'my' pieces, plane 1 = 'their' pieces, plane 2 = side-to-move flag (1 if p0/X).
        Returns a numpy array; lazy-imports numpy so the rollout path stays numpy-free."""
        import numpy as np
        board, player = state
        me = player + 1                       # 1 for p0 (X), 2 for p1 (O)
        them = 2 - player                     # the other mark
        planes = np.zeros((3, 3, 3), dtype=np.float32)
        for i in range(9):
            r, c = divmod(i, 3)
            if board[i] == me:
                planes[0, r, c] = 1.0
            elif board[i] == them:
                planes[1, r, c] = 1.0
        planes[2, :, :] = 1.0 if player == 0 else 0.0
        return planes

    def render(self, state) -> str:
        board, _ = state
        sym = {0: ".", 1: "X", 2: "O"}
        return "\n".join(" ".join(sym[board[r * 3 + c]] for c in range(3)) for r in range(3))


# --------------------------------------------------------------------------- #
# PROOF: the generic UCT solves TicTacToe (a non-chess engine) -> never loses to
# a random opponent. Optimal TicTacToe never loses; UCT at a few hundred sims is
# near-optimal. This validates that the GameAdapter contract is sufficient for
# the generic pipeline -- i.e. a NEW engine is plug-in.
# --------------------------------------------------------------------------- #
def _play(game: GameAdapter, uct_sims: int, uct_is_p0: bool, rng: random.Random) -> float:
    """One game: UCT (one side) vs random (other). Returns player-0-perspective outcome."""
    s = game.initial_state()
    while not game.is_terminal(s):
        p = game.current_player(s)
        uct_to_move = (p == 0) == uct_is_p0
        a = uct_search(game, s, uct_sims, rng=rng) if uct_to_move else rng.choice(game.legal_actions(s))
        s = game.apply(s, a)
    return game.returns(s)


def _proof(n_games: int = 40, uct_sims: int = 300) -> int:
    """UCT vs random over n_games (UCT alternates colour). Assert UCT NEVER LOSES (optimal
    TicTacToe play). Returns 0 on success (for the test gate)."""
    game = TicTacToe()
    rng = random.Random(0)
    uct_w = uct_d = uct_l = 0
    for g in range(n_games):
        uct_is_p0 = (g % 2 == 0)
        z = _play(game, uct_sims, uct_is_p0, rng)          # player-0 perspective
        uct_z = z if uct_is_p0 else -z                      # UCT's perspective
        if uct_z > 0:
            uct_w += 1
        elif uct_z == 0:
            uct_d += 1
        else:
            uct_l += 1
    print(f"[game_adapter] GENERIC UCT over the '{game.name}' adapter vs random, {n_games} games "
          f"({uct_sims} sims): UCT W{uct_w} D{uct_d} L{uct_l}")
    print(f"[game_adapter] -> a NON-chess engine solved through the SAME adapter-driven search; "
          f"the contract is sufficient. (Neural pipeline swaps random-rollout for net PUCT.)")
    assert uct_l == 0, f"UCT LOST {uct_l} games at TicTacToe -- search/adapter is broken"
    print("[game_adapter] PROOF PASS: generic search never loses -> the engine-agnostic pipeline works")
    return 0


if __name__ == "__main__":
    raise SystemExit(_proof())
