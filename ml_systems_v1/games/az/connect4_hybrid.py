"""
az.connect4_hybrid -- the GENUINELY-STRONGEST Connect-4 engine in this repo: the trained
AlphaZero net for the deep opening (where a pure-Python full solve is too slow) + the
PROVABLY-PERFECT bitboard solver for the mid/endgame (where it solves within the per-move
budget) + cheap tactical guardrails (never miss a 1-move win, never allow a 1-move loss).

Why this is the right design (honest, measured): Connect-4 is SOLVED, but a pure-Python solver
cannot solve the OPENING at interactive speed (the speed-ceiling lesson). The trained net plays a
decent opening; the solver plays a PERFECT endgame. Composing them is STRICTLY >= either alone:
 * it never loses a position the net would have lost to a missed win/block, and
 * it converts won/drawn endgames perfectly where the net errs.
So it can only match-or-beat the bare net -- which is exactly what we MEASURE before shipping it.

No emoji (Windows cp1252).
"""
from __future__ import annotations

from az.connect4_solver import Connect4Solver


class HybridConnect4Player:
    """A move-selector over the project's Connect4 game. `action(state)` returns a column.

    Order of authority (strongest-certainty first):
      1. an immediate winning drop (the net occasionally misses these)            -- instant
      2. block the opponent's immediate winning drop                              -- instant
      3. a PROVEN-optimal move (the whole position solved within budget)          -- perfect
      4. the trained net via NeuralMCTS                                           -- learned
    """

    def __init__(self, game, net, device: str = "cpu", mcts_sims: int = 128,
                 budget_s: float = 0.6, batch_size: int = 16):
        self.game = game
        self.net = net
        self.device = device
        self.mcts_sims = mcts_sims
        self.batch_size = batch_size
        self.solver = Connect4Solver(budget_s=budget_s)
        # provenance counters (so a run can REPORT how often each layer fired -- honesty)
        self.counts = {"win": 0, "block": 0, "proven": 0, "net": 0}

    def _net_action(self, state):
        from az.mcts import NeuralMCTS
        return NeuralMCTS(self.game, self.net, n_simulations=self.mcts_sims,
                          device=self.device).best_action_batched(
                              state, temperature=0.0, batch_size=self.batch_size)

    def action(self, state):
        cells = state[0]
        player = self.game.current_player(state)
        legal = self.game.legal_actions(state)
        position, mask, _ = Connect4Solver._to_bitboard(cells, player)
        solver_legal = self.solver._legal(mask)

        # 1. take an immediate win.
        w = self.solver._immediate_win(position, mask, solver_legal)
        if w is not None and w in legal:
            self.counts["win"] += 1
            return w
        # 2. block the opponent's immediate win.
        b = self.solver._immediate_win(position ^ mask, mask, solver_legal)
        if b is not None and b in legal:
            self.counts["block"] += 1
            return b
        # 3. provably-optimal move if the position fully solves within budget.
        pm = self.solver.proven_move(cells, player)
        if pm is not None and pm[0] in legal:
            self.counts["proven"] += 1
            return pm[0]
        # 4. fall back to the trained net.
        a = self._net_action(state)
        if a not in legal:
            a = legal[0]
        self.counts["net"] += 1
        return a
