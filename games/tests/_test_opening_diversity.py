"""
Regression test: OPENING DIVERSITY for self-play (the "vary the starting conditions" fix).

Locks in the thing the user caught by eye ("it plays the same way") so it can never silently
regress again:
  (1) openings.sample_opening_board produces many distinct, legal, non-terminal starts for
      book/random/mixed, and exactly ONE for startpos.
  (2) the PRODUCTION batched self-play generator actually USES it: opening_mode="mixed" makes
      games start from distinct positions; opening_mode="startpos" reproduces the OLD single-start
      behaviour (the control that proves the diversity gauge discriminates).

In-process (no spawn workers) + tiny net + few sims, so this is a FAST test. Exit 0 = pass.
No emoji (Windows cp1252).

Run:  .venv/Scripts/python.exe -m az._test_opening_diversity
"""
from __future__ import annotations

import numpy as np

from az.openings import sample_opening_board, OPENING_BOOK, _line_to_board
from az.net import AlphaZeroNet
from az.batched_selfplay import generate_selfplay_games_batched


def _distinct_starts(games) -> int:
    """Count distinct opening positions = distinct hash of each game's FIRST sample planes
    (the exact metric the trainer's diversity gauge uses)."""
    return len({hash(g[0].planes.tobytes()) for g in games if g})


def test_book_all_valid():
    bad = [ln for ln in OPENING_BOOK if _line_to_board(ln) is None]
    assert not bad, f"opening book has unparseable lines: {bad}"
    print(f"[ok] opening book: {len(OPENING_BOOK)} lines, all parse to legal positions")


def test_sampler_diversity():
    rng = np.random.default_rng(0)
    # startpos -> exactly one position
    starts = {sample_opening_board(rng, mode="startpos").board_fen() for _ in range(50)}
    assert len(starts) == 1, f"startpos must be unique, got {len(starts)}"
    # book/random/mixed -> many distinct, legal, non-terminal positions
    for mode in ("book", "random", "mixed"):
        fens = set()
        for _ in range(120):
            b = sample_opening_board(rng, mode=mode, random_plies=4)
            assert b.is_valid() and not b.is_game_over(claim_draw=True), \
                f"{mode}: produced an invalid/terminal start"
            fens.add(b.board_fen())
        assert len(fens) >= 10, f"{mode}: too few distinct starts ({len(fens)})"
        print(f"[ok] sampler mode={mode:8s}: {len(fens)} distinct starts / 120 samples")


def test_generator_uses_openings():
    net = AlphaZeroNet(channels=16, n_blocks=2).eval()
    import torch
    dev = torch.device("cpu")
    common = dict(n_simulations=6, temp_moves=4, max_plies=24, game_wall_s=30.0, device=dev)

    # CONTROL: startpos -> all 6 games share ONE opening (the OLD behaviour the user saw).
    g_start = generate_selfplay_games_batched(net, n_games=6, seed=1,
                                              opening_mode="startpos", **common)
    d_start = _distinct_starts(g_start)
    assert d_start == 1, f"startpos control should give 1 distinct start, got {d_start}"
    print(f"[ok] generator opening_mode=startpos -> {d_start} distinct start (old behaviour)")

    # FIX: mixed -> games start from distinct positions.
    g_mixed = generate_selfplay_games_batched(net, n_games=6, seed=1,
                                              opening_mode="mixed", opening_plies=4, **common)
    d_mixed = _distinct_starts(g_mixed)
    assert d_mixed >= 4, f"mixed openings should give >=4 distinct starts of 6, got {d_mixed}"
    # every game must still be a valid, z-labelled training game
    assert all(len(g) >= 1 and all(s.z in (-1.0, 0.0, 1.0) for s in g) for g in g_mixed), \
        "mixed games must still produce valid signed-z samples"
    print(f"[ok] generator opening_mode=mixed    -> {d_mixed} distinct starts of 6 (DIVERSE)")


def main() -> int:
    test_book_all_valid()
    test_sampler_diversity()
    test_generator_uses_openings()
    print("ALL OPENING-DIVERSITY TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
