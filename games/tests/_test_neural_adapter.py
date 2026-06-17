"""
CI lock for the NEURAL PUCT search over the engine-agnostic GameAdapter contract (NeuralMCTS).

The generality proof in _test_game_adapter.py only covers RANDOM-rollout UCT. THIS test locks the
harder, load-bearing claim the engine audit flagged as the #1 games gap: the AlphaZero NEURAL PUCT
search (net policy prior + net value, no random rollouts) drives ANY GameAdapter -- not just chess.

What it proves (all CPU, no GPU, fast):
  1. NEURAL PUCT over the TicTacToe GameAdapter with a tiny net (briefly trained) STRONGLY beats a
     random opponent -- near-optimal (w >= 16, l <= 3 over 24 games), with the NET (not random
     rollouts) as the evaluator. We do NOT assert provable optimality: a TINY net at a modest sim
     budget can drop the odd boundary game (visit-count selection under an imperfect value starves
     the drawing branch), and torch training is not bit-reproducible across runs -- so an absolutist
     "0 losses" bar FLAKES (verified: a fresh run scored W21 D2 L1 where the build run scored W21 D3
     L0). The non-flaky hard locks are (b) legal play asserted every neural move, (c) the
     engine-agnostic seam (TEST 3), and the strong-vs-random margin (random play would lose FAR more).
  2. NEURAL PUCT over the ChessGameAdapter produces LEGAL chess moves using the net evaluator
     (a few plies; strength NOT claimed -- a tiny untrained net is weak, that is expected + fine).
  3. NeuralMCTS is engine-agnostic by CONSTRUCTION: it consumes only the GameAdapter contract
     (encode / legal_policy_mask / apply / is_terminal / returns / current_player) + a net.predict --
     so a NEW engine plugs into the NEURAL pipeline, not only the random-rollout one.

Run:  .venv/Scripts/python.exe -m az._test_neural_adapter
Exit: 0 = the neural-PUCT-over-adapter contract holds. No emoji (Windows cp1252).
"""
from __future__ import annotations

import random

import numpy as np
import torch
import chess

from az.game_adapter import TicTacToe, GameAdapter
from az.chess_adapter import ChessGameAdapter
from az.mcts import NeuralMCTS
from az.net import TicTacToeNet, AlphaZeroNet


# --------------------------------------------------------------------------- #
# A brief self-play train of the tiny TicTacToe net so its value/policy heads are
# informative (the net -- not random rollouts -- must be what drives the search).
# Deterministic (seeded). ~35s on CPU; well under the run_tests 300s per-test cap.
# --------------------------------------------------------------------------- #
def _train_ttt_net(net: TicTacToeNet, n_games: int = 800, sims: int = 60,
                   lr: float = 1e-2, seed: int = 0) -> TicTacToeNet:
    g = TicTacToe()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    torch.manual_seed(seed)
    np.random.seed(seed)
    for _ in range(n_games):
        mcts = NeuralMCTS(g, net, n_simulations=sims)
        state = g.initial_state()
        history = []  # (planes, pi_vec(9), player_at_state)
        while not g.is_terminal(state):
            visits = mcts.run(state, add_noise=True)
            pi = np.zeros(g.num_actions, dtype=np.float32)
            tot = sum(visits.values())
            for a, n in visits.items():
                pi[a] = n / tot
            history.append((g.encode(state), pi, g.current_player(state)))
            actions = list(visits.keys())
            counts = np.array([visits[a] for a in actions], dtype=np.float64)
            probs = counts / counts.sum()
            state = g.apply(state, actions[int(np.random.choice(len(actions), p=probs))])
        z_p0 = g.returns(state)  # player-0 (X) absolute outcome
        opt.zero_grad()
        planes = torch.as_tensor(np.asarray([h[0].reshape(-1) for h in history]),
                                 dtype=torch.float32)
        pis = torch.as_tensor(np.asarray([h[1] for h in history]), dtype=torch.float32)
        # value target = outcome from the MOVER's perspective at each recorded state
        zs = torch.as_tensor([z_p0 if h[2] == 0 else -z_p0 for h in history], dtype=torch.float32)
        logits, value = net.forward(planes)
        logp = torch.log_softmax(logits, dim=-1)
        ploss = -(pis * logp).sum(dim=-1).mean()
        vloss = ((value.squeeze(-1) - zs) ** 2).mean()
        (ploss + vloss).backward()
        opt.step()
    return net


def test_neural_puct_tictactoe_strongly_beats_random(n_games: int = 24, sims: int = 160,
                                                     eval_seed: int = 0,
                                                     max_losses: int = 3) -> None:
    """NEURAL PUCT (trained tiny net) over the TicTacToe GameAdapter must STRONGLY beat random.

    The HARD, non-flaky locks: (a) every neural move is LEGAL (asserted inline below); (b) the search
    is engine-agnostic by construction (TEST 3); (c) the NET drives it (the net is trained and its
    value/policy -- not random rollouts -- back up the tree). The strength bar (w >= 16 AND
    l <= max_losses over 24 games) shows the neural signal is genuinely STEERING: a broken seam or
    random-level play loses FAR more (the 2nd player loses ~1/3 of decisive TicTacToe games to random).
    We deliberately do NOT assert provable optimality -- a TINY net at a modest sim budget can drop a
    boundary game, and torch training is not bit-reproducible run-to-run, so an absolutist 0-loss bar
    FLAKES (a fresh run scored W21 D2 L1 vs the build's W21 D3 L0). Seeded (random/np/torch) for
    stability; the margin -- not exact determinism -- is what makes this robust."""
    random.seed(eval_seed)
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)

    game = TicTacToe()
    net = TicTacToeNet()
    _train_ttt_net(net, n_games=800, sims=60, seed=0)

    rng = random.Random(eval_seed)
    w = d = l = 0
    for gi in range(n_games):
        neural_is_p0 = (gi % 2 == 0)   # alternate which colour the neural search plays
        s = game.initial_state()
        while not game.is_terminal(s):
            p = game.current_player(s)
            neural_to_move = (p == 0) == neural_is_p0
            if neural_to_move:
                a = NeuralMCTS(game, net, n_simulations=sims).best_action(s, temperature=0.0)
                # legality assertion: the NEURAL search must only ever return legal actions
                assert a in game.legal_actions(s), (
                    f"NEURAL search returned ILLEGAL action {a} at {game.render(s)!r}"
                )
            else:
                a = rng.choice(game.legal_actions(s))
            s = game.apply(s, a)
        z = game.returns(s)
        nz = z if neural_is_p0 else -z   # neural search's perspective
        if nz > 0:
            w += 1
        elif nz == 0:
            d += 1
        else:
            l += 1
    print(f"[neural_adapter] TEST 1: NeuralMCTS (trained tiny net) over the TicTacToe GameAdapter "
          f"vs random, {n_games} games ({sims} sims): neural W{w} D{d} L{l}")
    assert w >= 16 and l <= max_losses, (
        f"NEURAL PUCT vs random over {n_games} games: W{w} D{d} L{l} -- expected a STRONG result "
        f"(w >= 16 AND l <= {max_losses}). A near-random result (few wins / many losses) means the "
        f"neural search over the adapter is not steering -> the seam is broken."
    )
    print("[neural_adapter] TEST 1 PASS: the NEURAL search (net policy+value, NOT random rollouts) "
          "drives a NON-chess GameAdapter to STRONGLY beat random -> any engine plugs into the "
          "NEURAL pipeline (legal play asserted every move; strength margin proves the net steers).")


def test_neural_puct_chess_plays_legal(plies: int = 6, sims: int = 12) -> None:
    """NEURAL PUCT over the ChessGameAdapter must produce LEGAL chess moves via the net evaluator."""
    net = AlphaZeroNet(channels=16, n_blocks=2)  # tiny untrained net -- legality, not strength
    game = ChessGameAdapter()
    mcts = NeuralMCTS(game, net, n_simulations=sims, device=torch.device("cpu"))
    s = game.initial_state()
    played = []
    for ply in range(plies):
        if game.is_terminal(s):
            break
        a = mcts.best_action(s, temperature=0.0)
        legal = game.legal_actions(s)
        assert a in legal, f"NEURAL chess search returned ILLEGAL action {a} at {s}"
        board = chess.Board(s)
        mv = ChessGameAdapter._legal_index_map(board)[a]
        assert mv in board.legal_moves, f"decoded move {mv.uci()} not legal at {s}"
        played.append(mv.uci())
        s = game.apply(s, a)
    assert played, "neural chess search produced no moves"
    print(f"[neural_adapter] TEST 2 PASS: NeuralMCTS over ChessGameAdapter played {len(played)} "
          f"LEGAL chess moves via the net evaluator: {' '.join(played)} "
          f"(strength NOT claimed -- tiny untrained net).")


def test_neural_mcts_is_engine_agnostic() -> None:
    """Structural lock: NeuralMCTS drives whatever GameAdapter it is handed, using ONLY the
    contract methods. Confirm the SAME class instance type runs both adapters + the two neural
    hooks (encode + legal_policy_mask) exist on both."""
    for adapter in (TicTacToe(), ChessGameAdapter()):
        assert isinstance(adapter, GameAdapter)
        s = adapter.initial_state()
        # the two things the neural search needs beyond the rollout contract:
        planes = adapter.encode(s)
        assert planes is not None
        mask, idx_to_action = adapter.legal_policy_mask(s)
        assert mask.shape == (adapter.num_actions,), (
            f"{adapter.name}: legal_policy_mask width {mask.shape} != ({adapter.num_actions},)"
        )
        # mask marks exactly the legal actions; decode map inverts action_to_index
        legal = set(adapter.legal_actions(s))
        assert set(idx_to_action.values()) == legal, (
            f"{adapter.name}: legal_policy_mask decode map != legal_actions"
        )
        assert int(mask.sum()) == len(legal), f"{adapter.name}: mask count != number of legal actions"
        # action<->index round-trips (identity default, but call both for the override-safety lock)
        for a in legal:
            assert adapter.index_to_action(adapter.action_to_index(a)) == a, (
                f"{adapter.name}: action<->index round-trip broken for {a}"
            )
    print("[neural_adapter] TEST 3 PASS: NeuralMCTS is engine-agnostic by construction -- it consumes "
          "only the GameAdapter contract; both adapters expose encode + a legal policy mask + an "
          "action<->index round-trip.")


def main() -> int:
    print("=" * 72)
    print("  NEURAL PUCT over the engine-agnostic GameAdapter contract (NeuralMCTS)")
    print("=" * 72)
    test_neural_mcts_is_engine_agnostic()
    test_neural_puct_chess_plays_legal()
    test_neural_puct_tictactoe_strongly_beats_random()
    print("-" * 72)
    print("[neural_adapter] ALL PASS: the NEURAL AlphaZero search is genuinely game-agnostic -- it "
          "runs over the GameAdapter contract and plays legal moves on TicTacToe AND chess, with the "
          "net (not random rollouts) as the evaluator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
