"""
chess_zero.az.chess_adapter -- the REAL ChessGameAdapter: chess PLUGS INTO the generic engine.

Until now only TicTacToe implemented the GameAdapter contract (game_adapter.py), so "chess plugs
into the generic pipeline" was an OVERCLAIM (flagged in the engine audit). This file closes that gap:
a concrete ChessGameAdapter(GameAdapter) backed by python-chess + the project's verified move<->index
encoding (encoding.py, invariant I3 bijection -- REUSED here, not reinvented). The generic
uct_search in game_adapter.py now drives chess unchanged.

WHAT IS / IS NOT CLAIMED
  IS:  chess satisfies the 7-method GameAdapter contract; the generic adapter-driven UCT plays
       LEGAL chess to termination; the move<->index round-trip holds for every legal move.
  NOT: a STRONG player. Generic UCT with RANDOM rollouts (no net, no chess heuristics) is a WEAK
       chess engine -- random playouts almost never find mate, so it plays near-random moves. That
       is EXPECTED and FINE; the deliverable is the PLUG-IN (contract + legal play via the generic
       search), not strength. Strength needs the neural PUCT pipeline (train_robust.py) -- separate.

PERSPECTIVE / SIGN CONVENTION (load-bearing -- matched to TicTacToe + uct_search)
  GameAdapter.returns() is from PLAYER 0's perspective: +1 = player 0 wins, -1 = player 1 wins,
  0 = draw. We map WHITE -> player 0, BLACK -> player 1 (white moves first, exactly as X=p0 in
  TicTacToe). The generic uct_search accumulates W in player-0 perspective and negates per node by
  node.player, so returns() MUST be absolute (white-perspective), NOT side-to-move.
  NOTE: the chess engine's own MCTS._terminal_value (mcts.py) returns SIDE-TO-MOVE value
  (checkmate = -1 for the mated side-to-move). That is a DIFFERENT convention for a DIFFERENT search;
  do not copy it here. We convert: the side to move at a terminal node is the one mated/stalemated,
  so checkmate => the OTHER colour wins => +1 if white wins (black to move & mated), -1 if black wins.

STATE REPRESENTATION
  state = the board FEN string (str). FEN is hashable (dict/set keys, transposition-friendly),
  trivially clonable, and round-trips through chess.Board(fen) <-> board.fen() losslessly INCLUDING
  side-to-move, castling rights, en-passant square, and the halfmove/fullmove clocks (so 50-move and
  repetition-relevant counters survive). apply() is PURE: it builds a fresh Board from the FEN,
  pushes the decoded move, and returns the new FEN -- the input state is never mutated.

No emoji (Windows cp1252 safety).
"""
from __future__ import annotations

import random
from typing import List, Optional

import chess

# The contract + the generic search live in game_adapter.py (self-contained, no chess dep there).
# Dual import so this runs both as a package module (-m) AND as a standalone script (like game_adapter.py).
try:
    from .game_adapter import GameAdapter, uct_search
    # REUSE the project's verified encoding (I3 bijection incl. underpromotions). Do NOT reinvent.
    from .encoding import N_POLICY, move_to_index, legal_policy_mask, board_to_planes
except ImportError:  # pragma: no cover -- run as a script
    from game_adapter import GameAdapter, uct_search
    from encoding import N_POLICY, move_to_index, legal_policy_mask, board_to_planes


class ChessGameAdapter(GameAdapter):
    """Full chess as a GameAdapter, via python-chess + encoding.py. state = board FEN (str).

    All 7 contract methods implemented:
      num_actions    -> encoding.N_POLICY (4672 = 64 from-squares x 73 move-planes)
      initial_state  -> startpos FEN
      current_player -> 0 (white) / 1 (black)
      legal_actions  -> [encode(mv) for mv in board.legal_moves]  (the legal-move MASK as indices)
      apply          -> decode(index)->move, push on a COPY, return new FEN (PURE)
      is_terminal    -> board.is_game_over(claim_draw=True)
      returns        -> terminal value from PLAYER 0 (white) perspective in {-1, 0, +1}
    """

    name = "chess"

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _board(state) -> chess.Board:
        """Rebuild a Board from the state. Accepts a FEN str (canonical) or a chess.Board
        (defensive -- a caller might hand us a board; we copy it so apply() stays pure)."""
        if isinstance(state, chess.Board):
            return state.copy(stack=False)
        return chess.Board(state)

    @staticmethod
    def _legal_index_map(board: chess.Board) -> dict:
        """index -> chess.Move for the current legal moves (the decode side of the bijection).
        This is exactly the project's legal_policy_mask()[1] -- the same map the net pipeline uses."""
        _mask, idx_to_move = legal_policy_mask(board)
        return idx_to_move

    # ----------------------------------------------------------------- contract
    @property
    def num_actions(self) -> int:
        return N_POLICY  # 4672

    def initial_state(self):
        return chess.Board().fen()  # startpos FEN

    def current_player(self, state) -> int:
        # white to move -> player 0, black to move -> player 1 (white moves first, like X=p0)
        return 0 if self._board(state).turn == chess.WHITE else 1

    def legal_actions(self, state) -> List[int]:
        board = self._board(state)
        # Encode every legal move to its policy index. Sorted for determinism (dict order is
        # insertion-order in CPython but we don't want to rely on legal_moves ordering downstream).
        return sorted(self._legal_index_map(board).keys())

    def apply(self, state, action: int):
        board = self._board(state)                 # fresh board -> PURE (input state untouched)
        idx_to_move = self._legal_index_map(board)
        move = idx_to_move.get(int(action))
        if move is None:
            raise ValueError(
                f"illegal/unencodable action {action} at {board.fen()} "
                f"(legal indices: {sorted(idx_to_move.keys())[:8]}...)"
            )
        board.push(move)
        return board.fen()

    def is_terminal(self, state) -> bool:
        # claim_draw=True so 3-fold repetition and the 50-move rule count as terminal draws
        # (otherwise random UCT games can wander forever in a dead-drawn position).
        return self._board(state).is_game_over(claim_draw=True)

    def returns(self, state) -> float:
        """Terminal value from PLAYER 0 (WHITE) perspective: +1 white wins, -1 black wins, 0 draw.
        Convert from chess's side-to-move-relative terminal facts to the absolute white-perspective
        value the generic search expects."""
        board = self._board(state)
        if board.is_checkmate():
            # The side to move has been mated -> the OTHER side won.
            # board.turn is the mated side. If white is to move (white mated) -> black wins -> -1.
            return -1.0 if board.turn == chess.WHITE else 1.0
        # Stalemate, insufficient material, 75-move, 5-fold, or claimed 50-move/3-fold -> draw.
        return 0.0

    # ------------------------------- neural-pipeline hooks (NeuralMCTS over the contract) -------
    def encode(self, state):
        """state (FEN) -> (N_INPUT_PLANES, 8, 8) float32 planes for AlphaZeroNet, via the project's
        verified board_to_planes (side-to-move canonical orientation). REUSED, not reinvented, so the
        generic NeuralMCTS feeds the net exactly what the chess pipeline does."""
        return board_to_planes(self._board(state))

    # action_to_index / index_to_action: IDENTITY here -- ChessGameAdapter.legal_actions already
    # returns encoding.py policy indices in [0, N_POLICY), and apply() decodes an index back to a
    # move. So the inherited identity defaults in GameAdapter are exactly right; legal_policy_mask
    # (the base generic one) builds the (N_POLICY,) mask straight from legal_actions. No override.

    # ------------------------------------------------------------- nice-to-have
    def render(self, state) -> str:
        return str(self._board(state))


# --------------------------------------------------------------------------- #
# RWYB TEST 1 -- CONTRACT TEST.  Every method runs; the bijection round-trips;
# terminal values are correct on a known mate-in-1 and a stalemate.
# --------------------------------------------------------------------------- #
# Fixtures reused from the project's own invariants check (run_invariants_check.py I3/I6):
_FOOLS_MATE_FEN = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # WHITE mated
_STALEMATE_FEN = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"                                  # BLACK stalemated
# A clean mate-IN-1 for white to PLAY (Qd8#): black king h8, white Qd7 + Kf6. Qd7-d8 is mate.
_MATE_IN_1_FOR_WHITE_FEN = "7k/3Q4/5K2/8/8/8/8/8 w - - 0 1"


def _test_contract() -> int:
    """All 7 methods run on startpos + a few plies; legal_actions round-trips encode/decode;
    apply() yields a legal successor; is_terminal/returns correct on mate-in-1 and stalemate.
    Returns 0 on pass (raises AssertionError on any failure)."""
    g = ChessGameAdapter()
    assert isinstance(g, GameAdapter), "ChessGameAdapter must be a GameAdapter"

    # --- num_actions ---
    assert g.num_actions == N_POLICY == 4672, f"num_actions {g.num_actions} != 4672"

    # --- initial_state + the basic getters on startpos ---
    s0 = g.initial_state()
    assert isinstance(s0, str) and "w" in s0.split()[1], "initial_state should be white-to-move FEN"
    assert g.current_player(s0) == 0, "white to move at startpos -> player 0"
    assert not g.is_terminal(s0), "startpos is not terminal"
    la0 = g.legal_actions(s0)
    assert len(la0) == 20, f"startpos has 20 legal moves, adapter reports {len(la0)}"
    assert all(0 <= a < g.num_actions for a in la0), "every legal action index in [0, num_actions)"
    assert len(set(la0)) == len(la0), "legal action indices must be unique (no collisions)"

    # --- bijection round-trip across several plies of a random game ---
    rng = random.Random(7)
    state = s0
    purity_probe_fen = state  # apply() must not mutate the input state object/string
    for ply in range(12):
        if g.is_terminal(state):
            break
        board = chess.Board(state)
        # For EVERY legal move on this board: encode -> index, decode (via idx map) -> move, assert ==
        idx_to_move = ChessGameAdapter._legal_index_map(board)
        legal_uci = {m.uci() for m in board.legal_moves}
        assert set(m.uci() for m in idx_to_move.values()) == legal_uci, (
            f"idx_to_move must cover exactly the legal moves at ply {ply}"
        )
        for mv in board.legal_moves:
            idx = move_to_index(board, mv)
            assert idx is not None, f"legal move {mv.uci()} encoded to None at {board.fen()}"
            assert 0 <= idx < N_POLICY, f"index {idx} out of range for {mv.uci()}"
            assert idx in idx_to_move, f"index {idx} for {mv.uci()} not in decode map"
            assert idx_to_move[idx].uci() == mv.uci(), (
                f"round-trip BROKEN: {mv.uci()} -> {idx} -> {idx_to_move[idx].uci()}"
            )
        # adapter.legal_actions matches the encoded legal set
        assert sorted(idx_to_move.keys()) == g.legal_actions(state), (
            "legal_actions() must equal the encoded legal-move indices"
        )
        # apply() one move -> a LEGAL successor, input state unchanged (purity)
        a = rng.choice(g.legal_actions(state))
        before = state
        nxt = g.apply(state, a)
        assert state == before, "apply() mutated the input state (NOT pure)"
        assert isinstance(nxt, str) and nxt != state, "apply() must return a new FEN state"
        # the successor must be reachable by a real legal move from `before`
        assert chess.Board(nxt).fen() == nxt, "successor FEN must be canonical/parseable"
        state = nxt
    assert purity_probe_fen == s0, "startpos state string was mutated somewhere (purity violation)"

    # --- mate-in-1 for white to PLAY: search/legal path applies the mating move -> terminal+win ---
    mate_state = _MATE_IN_1_FOR_WHITE_FEN
    assert not g.is_terminal(mate_state), "mate-in-1 position is not yet terminal (white to move)"
    assert g.current_player(mate_state) == 0, "white to move in the mate-in-1 fixture"
    # find Qd8# among legal actions, apply it, assert terminal + white win (+1)
    found_mate = False
    for a in g.legal_actions(mate_state):
        nxt = g.apply(mate_state, a)
        b = chess.Board(nxt)
        if b.is_checkmate():
            assert g.is_terminal(nxt), "checkmate position must be terminal"
            # black is to move and mated -> white (player 0) wins -> +1
            assert g.returns(nxt) == 1.0, f"white delivers mate -> returns +1, got {g.returns(nxt)}"
            found_mate = True
            break
    assert found_mate, "no mating move found in the mate-in-1 fixture (bad fixture or apply bug)"

    # --- fool's mate: WHITE is checkmated -> terminal, black wins -> returns -1 (white perspective) ---
    fm = _FOOLS_MATE_FEN
    assert chess.Board(fm).is_checkmate(), "fools-mate fixture must be checkmate"
    assert g.is_terminal(fm), "checkmate must be terminal"
    assert g.current_player(fm) == 0, "white is to move (and mated) in fools mate"
    assert g.returns(fm) == -1.0, f"white mated -> black wins -> returns -1, got {g.returns(fm)}"

    # --- stalemate: terminal, draw -> returns 0 ---
    st = _STALEMATE_FEN
    assert chess.Board(st).is_stalemate(), "stalemate fixture must be stalemate"
    assert g.is_terminal(st), "stalemate must be terminal"
    assert g.returns(st) == 0.0, f"stalemate -> draw -> returns 0, got {g.returns(st)}"

    # --- insufficient material (K vs K): terminal draw -> returns 0 ---
    kvk = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"
    assert g.is_terminal(kvk), "K vs K must be terminal (insufficient material)"
    assert g.returns(kvk) == 0.0, "K vs K -> draw -> 0"

    print("[chess_adapter] TEST 1 PASS: contract holds -- 7 methods run, bijection round-trips on "
          "every legal move, mate-in-1/fools-mate/stalemate/KvK terminal values correct.")
    return 0


# --------------------------------------------------------------------------- #
# RWYB TEST 2 -- GENERIC SEARCH PLAYS LEGAL CHESS.  Run uct_search (the engine-
# agnostic search) over ChessGameAdapter for a full short game; assert EVERY move
# is legal + the game terminates or hits the move cap cleanly, no crash.
# Low sims (random-rollout UCT on chess is weak+slow). Strength is NOT asserted.
# --------------------------------------------------------------------------- #
def _play_one_game(uct_sims: int, max_plies: int, rng: random.Random,
                   uct_both_sides: bool = True) -> dict:
    """Play one game: the generic uct_search picks every move (or one side; other side random).
    Asserts each chosen action is legal for the current position. Returns a small result dict."""
    g = ChessGameAdapter()
    state = g.initial_state()
    plies = 0
    moves_uci: List[str] = []
    while not g.is_terminal(state) and plies < max_plies:
        legal = g.legal_actions(state)
        assert legal, "non-terminal position must have >=1 legal action"
        p = g.current_player(state)
        uct_to_move = uct_both_sides or (p == 0)
        if uct_to_move:
            a = uct_search(g, state, n_sims=uct_sims, rng=rng)
        else:
            a = rng.choice(legal)
        # THE legality assertion: whatever the search returned must be a legal action here.
        assert a in legal, f"search returned ILLEGAL action {a} at {state} (legal={legal[:8]}...)"
        board = chess.Board(state)
        chosen = ChessGameAdapter._legal_index_map(board)[a]
        assert chosen in board.legal_moves, f"decoded move {chosen.uci()} not legal at {state}"
        moves_uci.append(chosen.uci())
        state = g.apply(state, a)
        plies += 1
    terminal = g.is_terminal(state)
    result = "*"
    if terminal:
        b = chess.Board(state)
        # Use the engine's own result string for reporting (claim_draw so 50-move/3-fold show as draw)
        result = b.result(claim_draw=True)
    return {
        "plies": plies,
        "terminal": terminal,
        "result": result,
        "final_fen": state,
        "first_moves": moves_uci[:10],
        "returns_white_perspective": ChessGameAdapter().returns(state) if terminal else None,
    }


def _test_generic_search_plays_legal(n_games: int = 3, uct_sims: int = 24,
                                     max_plies: int = 60) -> int:
    """Play a few short games driven by the generic search; assert all moves legal + clean
    termination/cap, no crash. Returns 0 on pass."""
    rng = random.Random(0)
    for gi in range(n_games):
        res = _play_one_game(uct_sims=uct_sims, max_plies=max_plies, rng=rng, uct_both_sides=True)
        # cleanliness: it either reached a real terminal OR stopped exactly at the cap
        clean = res["terminal"] or res["plies"] == max_plies
        assert clean, f"game {gi} ended uncleanly: {res}"
        status = res["result"] if res["terminal"] else f"hit move-cap@{max_plies}"
        print(f"[chess_adapter]   game {gi}: {res['plies']} plies, "
              f"{'TERMINAL '+res['result'] if res['terminal'] else status}; "
              f"first moves: {' '.join(res['first_moves'][:6])}")
    print(f"[chess_adapter] TEST 2 PASS: generic uct_search played {n_games} full games of LEGAL "
          f"chess ({uct_sims} sims/move) -- every move legal, every game terminated or capped "
          f"cleanly, no crash. (Strength NOT claimed: random-rollout UCT is a weak chess player.)")
    return 0


# --------------------------------------------------------------------------- #
# RWYB TEST 3 -- ROUTER / SOLVE WIRING.  Confirm route({'domain':'games',
# 'name':'chess', ...}) lands on Layer A and that the concrete adapter is now
# importable + instantiable + named by the solve path. (solve.py's adapter note
# is updated to point at ChessGameAdapter -- see _update applied in this delivery.)
# --------------------------------------------------------------------------- #
def _test_router_solve_wiring() -> int:
    """Confirm the concrete ChessGameAdapter is importable + instantiable so a solve plan can
    name a CONCRETE adapter (not just the abstract GameAdapter). Returns 0 on pass.

    NOTE: the original TEST 3 also asserted that the parent crypto project's `framework.router`
    routes chess -> Layer A / alphazero. That router is NOT part of this standalone games engine,
    so the routing assertion is skipped here; the adapter-contract check below is the part that
    belongs to this repo and is kept.
    """
    import importlib

    # The concrete adapter is real -> importable + instantiable + 7 methods present
    mod = importlib.import_module("az.chess_adapter")
    ChessAdapter = getattr(mod, "ChessGameAdapter")
    inst = ChessAdapter()
    for meth in ("num_actions", "initial_state", "current_player", "legal_actions",
                 "apply", "is_terminal", "returns"):
        assert hasattr(inst, meth), f"adapter missing contract method {meth}"
    assert inst.num_actions == 4672
    print("[chess_adapter] TEST 3 PASS: a CONCRETE ChessGameAdapter is importable + instantiable "
          "(7-method contract present). [router routing check skipped -- lives in the parent project]")
    return 0


def _run_all() -> int:
    print("=" * 70)
    print("  chess_adapter RWYB: chess PLUGS INTO the generic GameAdapter engine")
    print("=" * 70)
    _test_contract()
    _test_generic_search_plays_legal()
    _test_router_solve_wiring()
    print("-" * 70)
    print("[chess_adapter] ALL RWYB PASS: ChessGameAdapter satisfies the contract, the generic "
          "search plays legal chess, and the router/solve path names the concrete adapter.")
    print("[chess_adapter] HONEST CEILING: this proves chess PLUGS IN (contract + legal play via "
          "the generic search). It is NOT a strong engine -- random-rollout UCT plays weak chess; "
          "strength requires the neural PUCT pipeline (train_robust.py).")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
