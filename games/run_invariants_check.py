"""
chess_zero / run_invariants_check.py -- the PRE-TRAINING INVARIANT GATE (audit item I13).

Runs the catastrophic-silent-failure correctness invariants in <60s and EXITS 2 on any
failure, so a future refactor of move-encoding / MCTS backup / target construction can NEVER
silently regress and burn a multi-hour restart. This is the mechanical pre-flight the
2026-06-09 pre-restart audit (docs/PRE_RESTART_AUDIT_2026_06_09.md, S4) called the meta-control:
it turns every correctness invariant in docs/SELFPLAY_SOTA_FEATURES.md (I-series) into an
assert that runs before training starts.

Checks (all against the REAL production modules, CPU, tiny net):
  I1  value sign negated per ply in MCTS backup     (mate-in-1 -> search picks the mate)
  I2  training z from each position's mover's view   (decisive game -> W/B samples opposite z)
  I3  policy<->move-index bijection, no collisions   (round-trip over random positions)
  I4  illegal-move mask applied (logits, not post)   (illegal probability mass ~ 0)
  I6  terminal value correctness                     (mate=-1 STM, stalemate/insufficient=0)
  I8  search-vs-played target integrity + opening    (full pi stored; opening plies not samples)
  I12 game-length cap / never-hang / adjudication    (guarded game terminates with valid z)
  S1  NaN/Inf loss guard in train_step               (non-finite loss -> SKIP, weights unchanged)

Run:  .venv/Scripts/python.exe projects/chess_zero/run_invariants_check.py
Exit: 0 = all invariants hold; 2 = a CATASTROPHIC invariant is broken (HALT -- do not train).
No emoji (Windows cp1252).
"""
from __future__ import annotations

import os
import sys
import traceback

# Allow running as a plain script (python run_invariants_check.py) by putting the
# games-engine root (this file's own dir) on sys.path, as well as via -m.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import chess
import torch

from az.net import AlphaZeroNet
from az.mcts import MCTS
from az.encoding import (board_to_planes, move_to_index,
                                             legal_policy_mask, N_POLICY)
from az.selfplay import Sample, train_step
from az.batched_selfplay import generate_selfplay_games_batched
from az.train_robust import generate_selfplay_game_guarded


def _fail(name, msg):
    raise AssertionError(f"{name}: {msg}")


def check_I1_value_sign(net):
    """Mate-in-1: MCTS must back the terminal -1 (mated side) up to a +1 root Q for the
    mating move and PLAY it. A missing per-ply negation makes Q collapse to ~0 and the
    search misses forced mate."""
    board = chess.Board("k7/7R/1K6/8/8/8/8/8 w - - 0 1")
    mating = []
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            mating.append(m)
        board.pop()
    if not mating:
        _fail("I1", "test position has no mate-in-1 (bad fixture)")
    mcts = MCTS(net, n_simulations=80, device=torch.device("cpu"))
    played = mcts.best_move(board, temperature=0.0)
    if played not in mating:
        _fail("I1", f"MCTS did not find forced mate (played {board.san(played)}, "
                    f"mate is {[board.san(x) for x in mating]}) -> value-sign backup suspect")


def check_I2_z_perspective(net):
    """In a DECISIVE game, samples whose mover was White must carry the opposite z of
    samples whose mover was Black. The catastrophic bug (z stored from White's global
    perspective) would give them the SAME sign."""
    dev = torch.device("cpu")
    for seed in range(6):
        games = generate_selfplay_games_batched(net, n_games=1, n_simulations=8,
                                                temp_moves=2, max_plies=30, game_wall_s=20.0,
                                                device=dev, seed=seed, opening_mode="startpos")
        g = games[0]
        zw = {s.z for s in g if s.player == chess.WHITE}
        zb = {s.z for s in g if s.player == chess.BLACK}
        # decisive game: each colour's samples share one z, and they are opposite
        if zw and zb and (0.0 not in zw) and (0.0 not in zb):
            if not (len(zw) == 1 and len(zb) == 1 and next(iter(zw)) == -next(iter(zb))):
                _fail("I2", f"decisive game has non-opposite z (white={zw}, black={zb}) "
                            f"-> z-perspective broken")
            return  # validated on the first decisive game
    # no decisive game in the sample -> not a failure, just note it
    print("  [I2] note: no decisive game in 6 tries (untrained net draws); "
          "logic-consistency still checked on all samples")
    for s in g:
        if s.z not in (-1.0, 0.0, 1.0):
            _fail("I2", f"z out of range: {s.z}")


def check_I3_bijection():
    """Round-trip move<->index over many random positions: every legal move maps to a
    unique index, no None, no collisions (esp. underpromotions)."""
    rng = np.random.default_rng(0)
    n_underpromo = 0
    for _ in range(300):
        board = chess.Board()
        for _ply in range(int(rng.integers(0, 40))):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(moves[int(rng.integers(len(moves)))])
        seen = {}
        for m in board.legal_moves:
            idx = move_to_index(board, m)
            if idx is None:
                _fail("I3", f"legal move {m.uci()} -> None index at {board.fen()}")
            if idx in seen:
                _fail("I3", f"index collision {idx}: {seen[idx].uci()} vs {m.uci()} "
                            f"at {board.fen()}")
            seen[idx] = m
            if m.promotion and m.promotion != chess.QUEEN:
                n_underpromo += 1
    if n_underpromo == 0:
        print("  [I3] note: no underpromotions sampled (still 0 collisions on all legals)")


def check_I4_masking(net):
    """net.predict with a legal mask must put ~zero probability on illegal moves."""
    board = chess.Board()
    mask, _idx_to_move = legal_policy_mask(board)
    probs, _v = net.predict(board_to_planes(board), legal_mask=mask, device=torch.device("cpu"))
    illegal_mass = float(probs[~mask.astype(bool)].sum())
    legal_mass = float(probs[mask.astype(bool)].sum())
    if illegal_mass > 1e-4:
        _fail("I4", f"illegal move mass {illegal_mass:.2e} (should be ~0) -> mask applied "
                    f"post-softmax?")
    if abs(legal_mass - 1.0) > 1e-3:
        _fail("I4", f"legal mass {legal_mass:.4f} != 1.0 -> renormalization broken")


def check_I6_terminals():
    """Terminal value: mate = -1 (side-to-move), stalemate/insufficient = 0."""
    tv = MCTS._terminal_value
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")  # fool's mate
    if not mate.is_checkmate():
        _fail("I6", "fool's-mate fixture is not checkmate (bad fixture)")
    if tv(mate) != -1.0:
        _fail("I6", f"checkmate terminal value {tv(mate)} != -1.0")
    stale = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # black stalemated
    if not stale.is_stalemate():
        _fail("I6", "stalemate fixture is not stalemate (bad fixture)")
    if tv(stale) != 0.0:
        _fail("I6", f"stalemate terminal value {tv(stale)} != 0.0")
    kvk = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")
    if tv(kvk) != 0.0:
        _fail("I6", f"insufficient-material terminal value {tv(kvk)} != 0.0")


def check_I8_target_and_opening(net):
    """Stored policy targets are FULL visit distributions (sum~1, shape N_POLICY, not
    one-hot for multi-legal positions); opening-diversity plies are NOT recorded as samples."""
    dev = torch.device("cpu")
    games = generate_selfplay_games_batched(net, n_games=4, n_simulations=8, temp_moves=4,
                                            max_plies=24, game_wall_s=20.0, device=dev,
                                            seed=1, opening_mode="mixed", opening_plies=4)
    starts = set()
    saw_soft = False
    for g in games:
        if not g:
            _fail("I8", "a game produced 0 samples")
        first = g[0]
        if first.pi.shape != (N_POLICY,):
            _fail("I8", f"pi shape {first.pi.shape} != ({N_POLICY},)")
        if abs(float(first.pi.sum()) - 1.0) > 1e-3 and float(first.pi.sum()) != 0.0:
            _fail("I8", f"pi does not sum to 1 ({float(first.pi.sum()):.4f})")
        # opening plies must not be recorded: first sample's board (reconstructed via planes
        # hash) varies across games because each started from a distinct opening
        starts.add(hash(first.planes.tobytes()))
        for s in g:
            if 0.0 < float(s.pi.max()) < 1.0 - 1e-6:
                saw_soft = True
    if len(starts) < 2:
        _fail("I8", f"opening diversity not reaching the recorded samples "
                    f"({len(starts)} distinct starts of {len(games)})")
    if not saw_soft:
        print("  [I8] note: no soft (non-one-hot) pi seen at 8 sims (low-sim trees are peaked); "
              "shape/sum still verified")


def check_I12_never_hang(net):
    """A guarded game with a tiny ply cap must terminate and emit valid signed-z samples."""
    samples = generate_selfplay_game_guarded(net, n_simulations=6, temp_moves=2, max_plies=10,
                                              game_wall_s=15.0, device=torch.device("cpu"),
                                              opponent="self", opening_mode="startpos")
    if not samples:
        _fail("I12", "guarded game produced 0 samples")
    for s in samples:
        if s.z not in (-1.0, 0.0, 1.0):
            _fail("I12", f"z out of range after cap: {s.z}")


def check_S1_nan_guard(net):
    """train_step must SKIP a non-finite-loss batch (return NaN) WITHOUT mutating weights --
    the silent-weight-corruption guard. We force a non-finite loss via an inf value target."""
    dev = torch.device("cpu")
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in net.parameters()]
    planes = board_to_planes(chess.Board())
    pi = np.zeros(N_POLICY, dtype=np.float32); pi[0] = 1.0
    bad = [Sample(planes=planes, pi=pi, player=True, z=float("inf")) for _ in range(4)]
    loss, pl, vl, gn = train_step(net, opt, bad, dev)
    if loss == loss:  # not NaN -> guard failed to flag the non-finite loss
        _fail("S1", f"non-finite-loss batch returned finite loss {loss} -> NaN guard missing")
    after = list(net.parameters())
    for b, a in zip(before, after):
        if not torch.equal(b, a):
            _fail("S1", "weights changed on a non-finite-loss batch -> guard did not skip the step")


CHECKS = [
    ("I1  value-sign backup", check_I1_value_sign, True),
    ("I2  z perspective", check_I2_z_perspective, True),
    ("I3  policy<->index bijection", check_I3_bijection, False),
    ("I4  illegal-move masking", check_I4_masking, True),
    ("I6  terminal values", check_I6_terminals, False),
    ("I8  target integrity + opening", check_I8_target_and_opening, True),
    ("I12 never-hang / adjudication", check_I12_never_hang, True),
    ("S1  NaN/Inf loss guard", check_S1_nan_guard, True),
]


def main() -> int:
    torch.manual_seed(0); np.random.seed(0)
    net = AlphaZeroNet(channels=16, n_blocks=2).eval()
    print("=" * 68)
    print("PRE-TRAINING INVARIANT GATE (I13) -- chess_zero")
    print("=" * 68)
    failures = []
    for name, fn, needs_net in CHECKS:
        try:
            fn(net) if needs_net else fn()
            print(f"  [PASS] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
            failures.append(name)
    print("=" * 68)
    if failures:
        print(f"HALT: {len(failures)} CATASTROPHIC invariant(s) BROKEN -- DO NOT TRAIN: {failures}")
        return 2
    print("ALL INVARIANTS HOLD -- safe to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
