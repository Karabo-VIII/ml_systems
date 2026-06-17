"""
CI lock for TREE-LEVEL MCTS PARALLELISM in the generic neural search (NeuralMCTS.run_batched).

This locks the engine audit's remaining search-SOTA item: SOTA AlphaZero (arXiv:1712.01815;
Leela/KataGo "virtual loss") batches MANY leaves from ONE tree per net forward, instead of one
leaf per forward. run_batched() descends B times per iteration, applies a VIRTUAL LOSS on every
in-flight edge so PUCT diversifies the descents, evaluates all distinct leaves in ONE
net.predict_many call, then backs them up AND removes the virtual loss.

What it proves (all CPU, fast, < 240s):
  1. LEGAL PLAY: run_batched returns only legal root actions on chess AND TicTacToe.
  2. SEQUENTIAL-vs-BATCHED AGREEMENT: given the same total sims + seed, the batched policy is
     CLOSE to the sequential one (virtual loss changes the ORDER leaves are gathered, not the
     converged statistics). We lock:
       - chess opening (clear net signal): rank-correlation ~1.0 AND top-move match.
       - TicTacToe root: high rank-correlation AND top-move match.
       - a near-tie TicTacToe interior position: at a HIGHER sim budget both methods converge to
         the SAME top-2 action SET (a single-top-move lock there FLAKES because the net genuinely
         near-ties two moves and the thin visit margin reorders under either method -- the honest
         convergence claim is the top-K SET, not a brittle single argmax at low sims).
  3. ZERO LEAKED VIRTUAL LOSS: after a batched search, EVERY node in the tree has vloss == 0.
     A leaked virtual loss permanently corrupts the tree, so this is the critical correctness lock.

The sequential public methods (run / best_action) are UNCHANGED and locked by _test_neural_adapter.py.

Run:  .venv/Scripts/python.exe -m az._test_mcts_parallel
Exit: 0 = tree-parallel search is correct (legal, agrees with sequential, no vloss leak).
No emoji (Windows cp1252).
"""
from __future__ import annotations

import numpy as np
import torch
import chess

from az.game_adapter import TicTacToe
from az.chess_adapter import ChessGameAdapter
from az.mcts import NeuralMCTS
from az.net import TicTacToeNet, AlphaZeroNet


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rank_corr(seq_visits: dict, bat_visits: dict) -> float:
    """Spearman rank correlation between two {action: visit_count} dicts over their union of
    actions. numpy-only (no scipy dependency). Returns 1.0 for a length-1 vector (trivially
    perfectly ranked)."""
    actions = sorted(set(seq_visits) | set(bat_visits))
    if len(actions) < 2:
        return 1.0
    s = np.array([seq_visits.get(a, 0) for a in actions], dtype=float)
    b = np.array([bat_visits.get(a, 0) for a in actions], dtype=float)

    def _ranks(x):
        order = x.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        # average ties so equal counts do not spuriously disagree
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        # mean rank per unique value
        sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
        mean = sums / counts
        return mean[inv]

    rs, rb = _ranks(s), _ranks(b)
    if rs.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(rs, rb)[0, 1])


def _top_move(visits: dict):
    return max(visits, key=visits.get)


def _topk_set(visits: dict, k: int) -> set:
    return set(sorted(visits, key=lambda a: -visits[a])[:k])


def _max_abs_vloss(root) -> tuple:
    """Walk the whole tree from `root`; return (max|vloss|, n_nodes)."""
    seen, stack, worst = set(), [root], 0
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        worst = max(worst, abs(n.vloss))
        stack.extend(n.children.values())
    return worst, len(seen)


def _train_ttt_net(net: TicTacToeNet, n_games: int = 300, sims: int = 40, seed: int = 0):
    """Brief deterministic self-play train so the net value/policy heads steer the search
    (so the agreement claim is about a MEANINGFUL search, not a uniform-prior one). ~15s CPU."""
    g = TicTacToe()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    torch.manual_seed(seed); np.random.seed(seed)
    for _ in range(n_games):
        mcts = NeuralMCTS(g, net, n_simulations=sims)
        state = g.initial_state(); history = []
        while not g.is_terminal(state):
            visits = mcts.run(state, add_noise=True)
            pi = np.zeros(g.num_actions, dtype=np.float32); tot = sum(visits.values())
            for a, n in visits.items():
                pi[a] = n / tot
            history.append((g.encode(state), pi, g.current_player(state)))
            acts = list(visits.keys()); counts = np.array([visits[a] for a in acts], dtype=np.float64)
            state = g.apply(state, acts[int(np.random.choice(len(acts), p=counts / counts.sum()))])
        z = g.returns(state)
        opt.zero_grad()
        planes = torch.as_tensor(np.asarray([h[0].reshape(-1) for h in history]), dtype=torch.float32)
        pis = torch.as_tensor(np.asarray([h[1] for h in history]), dtype=torch.float32)
        zs = torch.as_tensor([z if h[2] == 0 else -z for h in history], dtype=torch.float32)
        logits, value = net.forward(planes)
        logp = torch.log_softmax(logits, dim=-1)
        loss = -(pis * logp).sum(dim=-1).mean() + ((value.squeeze(-1) - zs) ** 2).mean()
        loss.backward(); opt.step()
    return net


# --------------------------------------------------------------------------- #
# TEST 1 -- chess: legal play + (near-)exact agreement with sequential.
# --------------------------------------------------------------------------- #
def test_batched_chess_legal_and_agrees(sims: int = 200) -> None:
    torch.manual_seed(1); np.random.seed(0)
    net = AlphaZeroNet(channels=16, n_blocks=2)   # tiny untrained net: legality + agreement, not strength
    game = ChessGameAdapter()
    state = game.initial_state()
    legal = set(game.legal_actions(state))

    m_seq = NeuralMCTS(game, net, n_simulations=sims, device=torch.device("cpu"))
    np.random.seed(3); seq = m_seq.run(state, add_noise=False)

    for B in (8, 16):
        m_bat = NeuralMCTS(game, net, n_simulations=sims, device=torch.device("cpu"))
        np.random.seed(3); bat = m_bat.run_batched(state, add_noise=False, batch_size=B)

        assert set(bat.keys()) <= legal, f"batched chess returned ILLEGAL root actions: {set(bat.keys()) - legal}"
        for a in bat:
            board = chess.Board(state)
            mv = ChessGameAdapter._legal_index_map(board)[a]
            assert mv in board.legal_moves, f"decoded batched move {mv.uci()} not legal"
        assert sum(bat.values()) == sims, f"batched total visits {sum(bat.values())} != {sims} sims"

        rho = _rank_corr(seq, bat)
        top_ok = _top_move(seq) == _top_move(bat)
        print(f"[mcts_parallel] TEST 1 chess B={B:2d}: spearman={rho:.4f} top_match={top_ok} "
              f"all_legal=True seq_total={sum(seq.values())} bat_total={sum(bat.values())}")
        assert rho >= 0.90, f"chess seq-vs-batched rank-corr {rho:.4f} < 0.90 (B={B})"
        assert top_ok, f"chess top move differs seq={_top_move(seq)} bat={_top_move(bat)} (B={B})"
    print("[mcts_parallel] TEST 1 PASS: batched chess search plays LEGAL moves and matches the "
          "sequential policy (rank-corr >= 0.90 + same top move) at the same sims/seed.")


# --------------------------------------------------------------------------- #
# TEST 2 -- TicTacToe (trained net): legal play, root agreement, interior convergence.
# --------------------------------------------------------------------------- #
def test_batched_ttt_legal_and_converges() -> None:
    g = TicTacToe()
    net = TicTacToeNet()
    _train_ttt_net(net, n_games=300, sims=40, seed=0)

    # (a) ROOT: clear-ish signal -> high rank-corr + same top move at a modest budget.
    root = g.initial_state()
    legal_root = set(g.legal_actions(root))
    m_seq = NeuralMCTS(g, net, n_simulations=240)
    np.random.seed(7); seq = m_seq.run(root, add_noise=False)
    for B in (8, 16):
        m_bat = NeuralMCTS(g, net, n_simulations=240)
        np.random.seed(7); bat = m_bat.run_batched(root, add_noise=False, batch_size=B)
        assert set(bat.keys()) <= legal_root, "batched TTT root returned ILLEGAL actions"
        assert sum(bat.values()) == 240
        rho = _rank_corr(seq, bat)
        top_ok = _top_move(seq) == _top_move(bat)
        print(f"[mcts_parallel] TEST 2a TTT-root B={B:2d}: spearman={rho:.4f} top_match={top_ok}")
        assert rho >= 0.85, f"TTT root rank-corr {rho:.4f} < 0.85 (B={B})"
        assert top_ok, f"TTT root top move differs seq={_top_move(seq)} bat={_top_move(bat)}"

    # (b) NEAR-TIE INTERIOR position: at a HIGHER budget the two methods converge to the SAME
    #     top-2 SET. (A single-argmax lock here flakes -- the net genuinely near-ties two moves;
    #     see the probe in the deliverable: at 1200 sims seq={1:409,3:417} bat={1:403,3:399}.)
    s = g.apply(g.initial_state(), 4); s = g.apply(s, 0)   # X center, O corner
    legal_s = set(g.legal_actions(s))
    HI = 1000
    m_seq2 = NeuralMCTS(g, net, n_simulations=HI)
    np.random.seed(7); seq2 = m_seq2.run(s, add_noise=False)
    for B in (8, 16):
        m_bat2 = NeuralMCTS(g, net, n_simulations=HI)
        np.random.seed(7); bat2 = m_bat2.run_batched(s, add_noise=False, batch_size=B)
        assert set(bat2.keys()) <= legal_s, "batched TTT interior returned ILLEGAL actions"
        top2_seq, top2_bat = _topk_set(seq2, 2), _topk_set(bat2, 2)
        rho = _rank_corr(seq2, bat2)
        print(f"[mcts_parallel] TEST 2b TTT-interior B={B:2d} ({HI} sims): top2_seq={sorted(top2_seq)} "
              f"top2_bat={sorted(top2_bat)} spearman={rho:.4f}")
        assert top2_seq == top2_bat, (
            f"TTT interior top-2 SET differs at {HI} sims: seq={sorted(top2_seq)} bat={sorted(top2_bat)} "
            f"-- batched search did NOT converge to the same dominant moves (B={B})"
        )
    print("[mcts_parallel] TEST 2 PASS: batched TTT search is legal, matches the sequential ROOT "
          "policy, and CONVERGES to the same dominant moves on a near-tie interior position.")


# --------------------------------------------------------------------------- #
# TEST 3 -- ZERO LEAKED VIRTUAL LOSS (the critical correctness lock).
# --------------------------------------------------------------------------- #
def test_no_leaked_virtual_loss() -> None:
    # chess
    torch.manual_seed(1); np.random.seed(0)
    cnet = AlphaZeroNet(channels=16, n_blocks=2)
    cg = ChessGameAdapter()
    for B in (8, 16):
        m = NeuralMCTS(cg, cnet, n_simulations=200, device=torch.device("cpu"))
        np.random.seed(3); m.run_batched(cg.initial_state(), add_noise=False, batch_size=B)
        worst, n_nodes = _max_abs_vloss(m._last_root)
        print(f"[mcts_parallel] TEST 3 chess B={B:2d}: max|vloss| over {n_nodes} nodes = {worst}")
        assert worst == 0, f"LEAKED virtual loss in chess tree (max|vloss|={worst}, B={B}) -- tree corrupted"
    # TicTacToe (trained)
    g = TicTacToe(); net = TicTacToeNet(); _train_ttt_net(net, n_games=200, sims=40, seed=0)
    for B in (8, 16):
        m = NeuralMCTS(g, net, n_simulations=240)
        np.random.seed(7); m.run_batched(g.initial_state(), add_noise=False, batch_size=B)
        worst, n_nodes = _max_abs_vloss(m._last_root)
        print(f"[mcts_parallel] TEST 3 TTT   B={B:2d}: max|vloss| over {n_nodes} nodes = {worst}")
        assert worst == 0, f"LEAKED virtual loss in TTT tree (max|vloss|={worst}, B={B}) -- tree corrupted"
    print("[mcts_parallel] TEST 3 PASS: ZERO leaked virtual loss -- every node's in-flight counter "
          "is 0 after the batched search (the tree is left in a clean, uncorrupted state).")


def main() -> int:
    print("=" * 72)
    print("  TREE-LEVEL MCTS PARALLELISM (virtual loss + leaf batching) -- NeuralMCTS.run_batched")
    print("=" * 72)
    test_batched_chess_legal_and_agrees()
    test_batched_ttt_legal_and_converges()
    test_no_leaked_virtual_loss()
    print("-" * 72)
    print("[mcts_parallel] ALL PASS: the batched/parallel neural search plays legal moves, agrees "
          "with the sequential search (order, not converged policy, changes), and leaves NO leaked "
          "virtual loss -- the SOTA tree-parallel capability is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
