"""
CI wrapper for the ENGINE-AGNOSTIC generality proof (game_adapter.py).

Runs the generic UCT search over the TicTacToe GameAdapter and asserts it never loses to a random
opponent -- locking "a NEW engine plugs into the SAME adapter-driven pipeline" into the test gate,
so the generalization can't silently rot. Fast, CPU, no net. Exit 0 = the contract holds.

Run:  .venv/Scripts/python.exe -m az._test_game_adapter
"""
from __future__ import annotations

from az.game_adapter import _proof, TicTacToe, GameAdapter, uct_search


def main() -> int:
    # 1) the contract is a real ABC with the required methods
    assert issubclass(TicTacToe, GameAdapter)
    g = TicTacToe()
    s = g.initial_state()
    assert g.num_actions == 9 and len(g.legal_actions(s)) == 9 and not g.is_terminal(s)
    # 2) the generic search solves the non-chess engine (never loses to random)
    rc = _proof()
    assert rc == 0, "engine-agnostic generality proof failed"
    print("[ok] engine-agnostic generality proof holds (generic search solves a non-chess engine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
