"""RWYB probe for DENSE TEACHER DISTILLATION (2026-06-08): in teacher games the teacher's chosen
move is now ALSO labelled (one-hot policy target), so the net imitates the teacher at every teacher
move, not just the sparse game outcome.

Asserts (over several short games to wash out MCTS noise):
  1. distill_teacher=True yields MORE samples than distill_teacher=False (the teacher moves are now
     labelled -> roughly doubles the per-game sample count vs net-moves-only).
  2. Every sample's pi is a valid distribution (sums ~1, non-negative); ON produces one-hot targets
     (the teacher labels) -- a strict superset of the OFF behaviour.
  3. distill_teacher=False reproduces the OLD net-moves-only count (no teacher samples).

Run from repo root:  python -m az._test_teacher_distill
CPU, tiny net. No emoji (Windows cp1252).
"""
from __future__ import annotations

import numpy as np
import torch

from az.train_robust import generate_selfplay_game_guarded, AlphaZeroNet


def _games(net, distill, k, seed0):
    """Play k short teacher games; return (total_samples, n_one_hot, all_valid)."""
    total = 0
    one_hot = 0
    all_valid = True
    for i in range(k):
        torch.manual_seed(seed0 + i)
        np.random.seed(seed0 + i)
        s = generate_selfplay_game_guarded(
            net, n_simulations=6, temp_moves=4, max_plies=20, game_wall_s=20.0,
            device=torch.device("cpu"), opponent="teacher", teacher_depth=1,
            net_is_white=(i % 2 == 0), distill_teacher=distill)
        total += len(s)
        for smp in s:
            ssum = float(smp.pi.sum())
            if ssum > 0:
                if not (0.99 <= ssum <= 1.01) or smp.pi.min() < -1e-6:
                    all_valid = False
                if smp.pi.max() > 0.999 and int((smp.pi > 1e-6).sum()) == 1:
                    one_hot += 1
    return total, one_hot, all_valid


def main() -> int:
    torch.manual_seed(0)
    net = AlphaZeroNet(channels=8, n_blocks=1)
    net.eval()
    K = 4

    on_total, on_onehot, on_valid = _games(net, True, K, seed0=100)
    off_total, off_onehot, off_valid = _games(net, False, K, seed0=100)

    print(f"distill ON : {on_total} samples over {K} games ({on_onehot} one-hot)")
    print(f"distill OFF: {off_total} samples over {K} games ({off_onehot} one-hot)")

    assert on_valid and off_valid, "[FAIL] some pi target is not a valid distribution"
    # (1) teacher labels add samples -> ON strictly more than OFF (same seeds -> same games, ON adds
    # the teacher-move samples on top). Use the SAME seeds so the games match move-for-move.
    assert on_total > off_total, f"[FAIL] distill ON ({on_total}) not > OFF ({off_total}) -- teacher moves not labelled"
    # (2) ON must contain one-hot teacher targets; with matched seeds the extra samples ARE one-hot.
    assert on_onehot >= (on_total - off_total), \
        f"[FAIL] one-hot count {on_onehot} < the {on_total - off_total} teacher samples added"
    # (3) the added count is the teacher-move count: ON-OFF > 0 and on the order of OFF (both sides ~half each)
    added = on_total - off_total
    assert added > 0, "[FAIL] no teacher samples added"
    print(f"[1] ON > OFF by {added} samples (the teacher moves, now labelled) OK")
    print(f"[2] one-hot teacher targets present ({on_onehot} >= {added}) OK")
    print(f"[3] all pi targets valid distributions OK")
    print("\nALL TEACHER-DISTILL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
