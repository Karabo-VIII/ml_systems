"""
chess_zero.az.mcts -- PUCT-guided Monte-Carlo Tree Search (AlphaZero style).

No random rollouts: leaves are evaluated by the network's VALUE head, and the
network's POLICY head supplies the prior P(s,a) that biases exploration via the
PUCT formula (arXiv:1712.01815, "Search"):

    a* = argmax_a [ Q(s,a) + U(s,a) ],
    U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))

Each simulation: SELECT (PUCT to a leaf) -> EXPAND+EVALUATE (net) -> BACKUP
(propagate the value up, negating per ply because it is side-to-move relative).

Root Dirichlet noise (added in selfplay, optional here) injects exploration.

STATUS: LOAD-BEARING. This is the real search used throughout the pipeline:
self-play game generation (selfplay.py / train_robust.py), the per-move policy
target, and the greedy best_move() at eval/watch time. It runs against TRAINED
nets (bootstrap + self-play), not just the uniform-prior smoke case.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import chess

from .encoding import board_to_planes, legal_policy_mask, move_to_index


class Node:
    __slots__ = ("prior", "to_play", "children", "visit_count",
                 "value_sum", "is_expanded")

    def __init__(self, prior: float, to_play: bool):
        self.prior = prior
        self.to_play = to_play           # chess.WHITE / chess.BLACK at this node
        self.children: Dict[chess.Move, "Node"] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, c_puct: float = 1.5, n_simulations: int = 200,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25,
                 device=None):
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    # --- net evaluation of a single board ---------------------------------- #
    def _evaluate(self, board: chess.Board):
        """Run the net on `board`. Returns (priors_by_move dict, value float).
        value is from the side-to-move perspective in [-1, 1]."""
        planes = board_to_planes(board)
        mask, idx_to_move = legal_policy_mask(board)
        probs, value = self.net.predict(planes, legal_mask=mask, device=self.device)
        priors = {}
        for idx, mv in idx_to_move.items():
            priors[mv] = float(probs[idx])
        # Renormalise over legal moves (mask+softmax already did, but be safe).
        total = sum(priors.values())
        if total > 0:
            priors = {m: p / total for m, p in priors.items()}
        else:
            n = max(1, len(idx_to_move))
            priors = {m: 1.0 / n for m in idx_to_move}
        return priors, value

    # --- terminal value ---------------------------------------------------- #
    @staticmethod
    def _terminal_value(board: chess.Board) -> Optional[float]:
        """Value from the side-to-move perspective if `board` is terminal, else None."""
        if board.is_checkmate():
            return -1.0  # side to move has been mated
        if board.is_game_over(claim_draw=True):
            return 0.0   # any non-mate game-over = draw
        return None

    # --- PUCT child selection ---------------------------------------------- #
    def _select_child(self, node: Node):
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score, best_move, best_child = -float("inf"), None, None
        for move, child in node.children.items():
            q = -child.value()  # child value is from child's POV -> negate
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_move, best_child = score, move, child
        return best_move, best_child

    def _expand(self, node: Node, board: chess.Board):
        priors, value = self._evaluate(board)
        for move, p in priors.items():
            node.children[move] = Node(prior=p, to_play=not board.turn)
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, root: Node):
        if not root.children:
            return
        moves = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for mv, n in zip(moves, noise):
            child = root.children[mv]
            child.prior = (1 - self.dirichlet_eps) * child.prior + self.dirichlet_eps * n

    # --- public: run search at a position ---------------------------------- #
    def run(self, board: chess.Board, add_noise: bool = False) -> Dict[chess.Move, int]:
        """Run n_simulations from `board`. Returns visit counts per root move
        (the improved policy AlphaZero trains the policy head toward)."""
        root = Node(prior=1.0, to_play=board.turn)
        self._expand(root, board)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node = root
            scratch = board.copy()
            path = [node]

            # SELECT down to a leaf (an unexpanded node or terminal).
            while node.is_expanded and node.children:
                move, child = self._select_child(node)
                if move is None:
                    break
                scratch.push(move)
                node = child
                path.append(node)

            # EVALUATE the leaf.
            term = self._terminal_value(scratch)
            if term is not None:
                value = term
            else:
                value = self._expand(node, scratch)

            # BACKUP, negating per ply (value is side-to-move relative).
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return {mv: child.visit_count for mv, child in root.children.items()}

    def best_move(self, board: chess.Board, temperature: float = 0.0) -> chess.Move:
        """Pick a move from the visit counts. temperature=0 -> argmax (deterministic,
        for evaluation/play); >0 -> sample proportional to N^(1/temp) (for self-play
        exploration)."""
        visits = self.run(board, add_noise=(temperature > 0))
        moves = list(visits.keys())
        counts = np.array([visits[m] for m in moves], dtype=np.float64)
        if counts.sum() == 0:
            return moves[0]
        if temperature <= 1e-6:
            return moves[int(counts.argmax())]
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        return moves[int(np.random.choice(len(moves), p=probs))]


# =========================================================================== #
# GENERIC NEURAL PUCT over the GameAdapter contract (NeuralMCTS).
#
# The class above (MCTS) is the chess FAST-PATH: it talks to chess.Board + the
# chess encoding directly, and is the load-bearing search the whole chess
# pipeline (selfplay / train_robust / eval / play) uses -- UNCHANGED.
#
# NeuralMCTS below is the SAME PUCT search (identical Q + c*P*sqrt(sumN)/(1+N)
# formula, identical per-ply value negation, identical Dirichlet-root option),
# but it consumes ONLY the engine-agnostic GameAdapter contract:
#     adapter.encode(state)              -> net input planes
#     adapter.legal_policy_mask(state)   -> (num_actions,) mask + {idx -> action}
#     net.predict(planes, legal_mask, device) -> (probs over num_actions, value)
#     adapter.apply / is_terminal / returns / current_player
# So ANY GameAdapter (TicTacToe, chess, a future engine) plugs into the NEURAL
# pipeline, not just the random-rollout UCT. The action<->policy-index map is the
# adapter's (identity by default, overridable). The net just needs a predict()
# returning per-policy-index probabilities + a side-to-move value in [-1, 1].
#
# VALUE CONVENTION: net.value is from the SIDE-TO-MOVE perspective (the same
# convention AlphaZeroNet trains under). We negate per ply on backup, exactly
# like the chess MCTS. We DO NOT call adapter.returns() at non-terminal leaves --
# the net supplies the leaf value; returns() (player-0 absolute) is used only at
# TRUE terminals and converted to side-to-move there.
# =========================================================================== #
class _GNode:
    # `vloss` (default 0) is the IN-FLIGHT VIRTUAL-LOSS counter used ONLY by the batched
    # search (run_batched). The sequential run()/best_action() path never touches it, so it
    # stays 0 there and those methods are bit-for-bit unchanged. ADDITIVE slot.
    __slots__ = ("prior", "to_play", "action", "children", "visit_count", "value_sum",
                 "is_expanded", "vloss")

    def __init__(self, prior: float, to_play: int, action=None):
        self.prior = prior
        self.to_play = to_play          # current_player at THIS node's state (0 or 1)
        self.action = action            # the action that led here (None at root)
        self.children: Dict[int, "_GNode"] = {}   # action -> child
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
        self.vloss = 0                  # in-flight virtual-loss count (batched search only)

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class NeuralMCTS:
    """AlphaZero PUCT search over ANY GameAdapter, evaluated by a neural net (no random rollouts).

    net contract: net.predict(planes, legal_mask=<(num_actions,) {0,1}>, device=...) ->
                  (probs: np.ndarray (num_actions,), value: float in [-1,1] side-to-move POV).
    AlphaZeroNet satisfies this already; a tiny TicTacToe net (net.TicTacToeNet) satisfies it too."""

    def __init__(self, game, net, c_puct: float = 1.5, n_simulations: int = 200,
                 dirichlet_alpha: float = 0.3, dirichlet_eps: float = 0.25, device=None):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = device

    # --- net evaluation of one adapter state -> (priors {action: p}, side-to-move value) ---
    def _evaluate(self, state):
        planes = self.game.encode(state)
        mask, idx_to_action = self.game.legal_policy_mask(state)
        probs, value = self.net.predict(planes, legal_mask=mask, device=self.device)
        priors = {idx_to_action[idx]: float(probs[idx]) for idx in idx_to_action}
        total = sum(priors.values())
        if total > 0:
            priors = {a: p / total for a, p in priors.items()}
        else:
            n = max(1, len(idx_to_action))
            priors = {a: 1.0 / n for a in idx_to_action}
        return priors, value

    def _terminal_value_stm(self, state) -> Optional[float]:
        """Side-to-move terminal value, or None if not terminal. Converts the adapter's
        PLAYER-0-absolute returns() into the side-to-move convention NeuralMCTS backs up."""
        if not self.game.is_terminal(state):
            return None
        z_p0 = self.game.returns(state)             # +1 player0 wins, -1 player1 wins, 0 draw
        stm = self.game.current_player(state)        # side to move at the terminal node
        return z_p0 if stm == 0 else -z_p0           # flip to the mover's perspective

    def _select_child(self, node: "_GNode"):
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score, best_action, best_child = -float("inf"), None, None
        for action, child in node.children.items():
            q = -child.value()                       # child value is child's POV -> negate
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _expand(self, node: "_GNode", state) -> float:
        priors, value = self._evaluate(state)
        child_to_play = 1 - node.to_play             # 2-player alternation
        for action, p in priors.items():
            node.children[action] = _GNode(prior=p, to_play=child_to_play, action=action)
        node.is_expanded = True
        return value

    def _add_dirichlet_noise(self, root: "_GNode"):
        if not root.children:
            return
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        for a, n in zip(actions, noise):
            ch = root.children[a]
            ch.prior = (1 - self.dirichlet_eps) * ch.prior + self.dirichlet_eps * n

    def run(self, root_state, add_noise: bool = False) -> Dict[int, int]:
        """Run n_simulations from root_state. Returns {action: visit_count} over root actions."""
        if self.game.is_terminal(root_state):
            raise ValueError("NeuralMCTS.run called on a terminal state")
        root = _GNode(prior=1.0, to_play=self.game.current_player(root_state))
        self._expand(root, root_state)
        if add_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.n_simulations):
            node = root
            state = root_state
            path = [node]
            # SELECT down to a leaf (unexpanded or terminal).
            while node.is_expanded and node.children:
                action, child = self._select_child(node)
                if action is None:
                    break
                state = self.game.apply(state, action)
                node = child
                path.append(node)
            # EVALUATE the leaf.
            term = self._terminal_value_stm(state)
            value = term if term is not None else self._expand(node, state)
            # BACKUP, negating per ply (value is side-to-move relative).
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value

        return {a: ch.visit_count for a, ch in root.children.items()}

    def best_action(self, root_state, temperature: float = 0.0) -> int:
        """Pick a root action from visit counts. temperature=0 -> argmax (play/eval);
        >0 -> sample proportional to N^(1/temp) (self-play exploration)."""
        visits = self.run(root_state, add_noise=(temperature > 0))
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if counts.sum() == 0:
            return actions[0]
        if temperature <= 1e-6:
            return actions[int(counts.argmax())]
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))]

    # ===================================================================== #
    # TREE-LEVEL PARALLELISM: virtual loss + in-tree leaf batching.
    #
    # The sequential run() above does ONE net.predict per simulation. SOTA
    # AlphaZero (arXiv:1712.01815 S"Search", Leela/KataGo "virtual loss")
    # instead collects B leaves from the SAME tree per iteration and evaluates
    # them in ONE net forward (net.predict_many) -- cutting per-game search
    # latency and letting a single game push more sims/sec, especially on GPU.
    #
    # To stop all B descents from collapsing onto the same path, a VIRTUAL LOSS
    # is applied at every edge a descent traverses: the edge is temporarily
    # made to look like it LOST `vloss` extra games from the PARENT's
    # perspective, so PUCT steers the next descent elsewhere. After the batch
    # is evaluated and backed up, the virtual loss is REMOVED on every in-flight
    # path (a leaked vloss would permanently corrupt the tree -- this is the
    # critical correctness step, locked by _test_mcts_parallel.py).
    #
    # Public, ADDITIVE: run() / best_action() are unchanged and never touch
    # `vloss`. run_batched() is a drop-in alternative that returns the SAME
    # {action: visit_count} contract and converges to the same policy (virtual
    # loss changes the ORDER leaves are gathered, not the converged statistics).
    # ===================================================================== #
    def _select_child_vl(self, node: "_GNode"):
        """PUCT child selection under VIRTUAL LOSS. Identical formula to _select_child,
        but every count is inflated by the in-flight `vloss` and each virtual visit is
        scored as a LOSS for the parent (so a child already chosen by an in-flight
        descent is penalized and later descents diversify)."""
        parent_n = node.visit_count + node.vloss
        sqrt_total = math.sqrt(max(1, parent_n))
        best_score, best_action, best_child = -float("inf"), None, None
        for action, child in node.children.items():
            eff_n = child.visit_count + child.vloss
            if eff_n > 0:
                # child.value_sum is from the CHILD's POV; parent sees -value_sum.
                # Each virtual visit contributes a parent-POV value of -1 (a loss).
                q = ((-child.value_sum) - child.vloss) / eff_n
            else:
                q = 0.0
            u = self.c_puct * child.prior * sqrt_total / (1 + eff_n)
            score = q + u
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        return best_action, best_child

    def _descend_with_vloss(self, root: "_GNode", root_state):
        """One PUCT descent from root to a leaf, APPLYING virtual loss (+1 vloss) on every
        node it touches (root included). Returns (path, leaf_state, terminal_value_or_None).
        The leaf is either an unexpanded node or a terminal state."""
        node = root
        state = root_state
        path = [node]
        node.vloss += 1
        while node.is_expanded and node.children:
            action, child = self._select_child_vl(node)
            if action is None:
                break
            state = self.game.apply(state, action)
            node = child
            path.append(node)
            node.vloss += 1
        term = self._terminal_value_stm(state)
        return path, state, term

    @staticmethod
    def _remove_vloss(path):
        """Strip the virtual loss this path added (one per node). Called for EVERY collected
        path before backup so the tree is left with zero in-flight virtual loss."""
        for n in path:
            n.vloss -= 1

    @staticmethod
    def _backup(path, value):
        """Real backup: +1 visit and accumulate value, negating per ply (side-to-move POV)."""
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    def run_batched(self, root_state, add_noise: bool = False, batch_size: int = 16) -> Dict[int, int]:
        """Tree-parallel PUCT: gather up to `batch_size` leaves per iteration via virtual-loss
        descents, evaluate ALL distinct non-terminal leaves in ONE net.predict_many call, then
        back them up and remove the virtual loss. Returns {action: visit_count} -- SAME contract
        as run(). Total simulations == self.n_simulations (the last batch is clamped so the count
        is exact). batch_size <= 1 degrades to a correct sequential search."""
        if self.game.is_terminal(root_state):
            raise ValueError("NeuralMCTS.run_batched called on a terminal state")
        if batch_size < 1:
            batch_size = 1
        root = _GNode(prior=1.0, to_play=self.game.current_player(root_state))
        self._expand(root, root_state)
        if add_noise:
            self._add_dirichlet_noise(root)

        done = 0
        while done < self.n_simulations:
            b = min(batch_size, self.n_simulations - done)
            collected = []          # list of (path, leaf_state, term_or_None)
            # Distinct UNEXPANDED leaf nodes needing a net eval, in first-seen order.
            unique_leaves = []      # list of (leaf_node, leaf_state)
            leaf_index = {}         # id(leaf_node) -> position in unique_leaves
            for _ in range(b):
                path, leaf_state, term = self._descend_with_vloss(root, root_state)
                collected.append((path, leaf_state, term))
                if term is None:
                    leaf_node = path[-1]
                    key = id(leaf_node)
                    if key not in leaf_index:
                        leaf_index[key] = len(unique_leaves)
                        unique_leaves.append((leaf_node, leaf_state))

            # ONE batched net forward over all distinct non-terminal leaves (the whole point).
            leaf_values = {}        # id(leaf_node) -> side-to-move value
            if unique_leaves:
                planes_list, masks_list, decode_maps = [], [], []
                for _node, st in unique_leaves:
                    planes = self.game.encode(st)
                    mask, idx_to_action = self.game.legal_policy_mask(st)
                    planes_list.append(planes)
                    masks_list.append(mask)
                    decode_maps.append(idx_to_action)
                results = self.net.predict_many(planes_list, masks_list, device=self.device)
                for (leaf_node, _st), (probs, value), idx_to_action in zip(
                        unique_leaves, results, decode_maps):
                    self._expand_from_probs(leaf_node, probs, idx_to_action)
                    leaf_values[id(leaf_node)] = float(value)

            # Remove virtual loss on EVERY path, then real-backup each collected descent.
            for path, _leaf_state, term in collected:
                self._remove_vloss(path)
                value = term if term is not None else leaf_values[id(path[-1])]
                self._backup(path, value)

            done += b

        # Expose the root for white-box inspection (e.g. the leaked-virtual-loss CI check).
        # Diagnostic only -- not part of the public return contract.
        self._last_root = root
        return {a: ch.visit_count for a, ch in root.children.items()}

    def _expand_from_probs(self, node: "_GNode", probs, idx_to_action):
        """Expand `node` from an ALREADY-COMPUTED batched policy (no extra net call). Mirrors
        _expand's prior normalization + child wiring. Idempotent-safe: only expands once
        (a node reached by two in-flight descents is in unique_leaves exactly once)."""
        if node.is_expanded:
            return
        priors = {idx_to_action[idx]: float(probs[idx]) for idx in idx_to_action}
        total = sum(priors.values())
        if total > 0:
            priors = {a: p / total for a, p in priors.items()}
        else:
            n = max(1, len(idx_to_action))
            priors = {a: 1.0 / n for a in idx_to_action}
        child_to_play = 1 - node.to_play
        for action, p in priors.items():
            node.children[action] = _GNode(prior=p, to_play=child_to_play, action=action)
        node.is_expanded = True

    def best_action_batched(self, root_state, temperature: float = 0.0,
                            batch_size: int = 16) -> int:
        """best_action twin that uses run_batched. SAME selection rule as best_action."""
        visits = self.run_batched(root_state, add_noise=(temperature > 0), batch_size=batch_size)
        actions = list(visits.keys())
        counts = np.array([visits[a] for a in actions], dtype=np.float64)
        if counts.sum() == 0:
            return actions[0]
        if temperature <= 1e-6:
            return actions[int(counts.argmax())]
        probs = counts ** (1.0 / temperature)
        probs = probs / probs.sum()
        return actions[int(np.random.choice(len(actions), p=probs))]


if __name__ == "__main__":
    # Smoke test with an UNTRAINED net: search runs, returns a legal move.
    from .net import AlphaZeroNet
    net = AlphaZeroNet(channels=32, n_blocks=2)
    mcts = MCTS(net, n_simulations=20)
    b = chess.Board()
    visits = mcts.run(b, add_noise=True)
    mv = mcts.best_move(b)
    print(f"MCTS (untrained net, 20 sims): {len(visits)} root moves explored, "
          f"picked {b.san(mv)} (legal={mv in b.legal_moves})")
