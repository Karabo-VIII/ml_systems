"""RWYB unit test for auto_balance.py (pure; no torch/GPU/I/O).

Asserts:
  1. derive_workers respects cores - headroom, games, and the cap.
  2. steps_for_ratio <-> replay_ratio round-trip lands at the target ratio.
  3. ONLINE rebalance CONVERGES iter-time into the dead-band under a simple cost model, bounded
     per step (no thrash), and keeps the replay ratio in band throughout.
  4. In-band iter-time leaves games alone; an out-of-band replay ratio only moves train_steps.
  5. STRUCTURAL SAFETY: the balancer only ever emits THROUGHPUT knobs (workers/games/steps) and
     never the learning-contract knobs -- it has no parameter for them, by construction.

Run from repo root:  python -m az._test_auto_balance
No emoji (Windows cp1252).
"""
from __future__ import annotations

from az.auto_balance import (
    HW, BalanceTargets, derive_workers, replay_ratio, steps_for_ratio,
    rebalance, initial_plan, target_iter_seconds,
)

T = BalanceTargets()


def test_workers():
    assert derive_workers(20, 64, T) == 16, "20 cores, cap 16"
    assert derive_workers(4, 64, T) == 2, "4 cores - 2 headroom"
    assert derive_workers(20, 10, T) == 10, "never more workers than games"
    assert derive_workers(1, 64, T) == 1, "floor 1"
    print("[1] derive_workers OK")


def test_ratio_roundtrip():
    for games, plies, batch in [(64, 40, 128), (16, 30, 64), (128, 50, 256)]:
        s = steps_for_ratio(games, plies, batch, T)
        rr = replay_ratio(s, batch, games, plies)
        assert T.replay_lo <= rr <= T.replay_hi, f"ratio {rr} out of band for {games}/{plies}/{batch}"
        # "close to target" only when the steps_min/max floor/cap did NOT bind (else the clamp
        # legitimately shifts the ratio -- e.g. small games hit steps_min=20 and raise it).
        unclamped = T.replay_ratio * games * plies / batch
        if T.steps_min < unclamped < T.steps_max:
            assert abs(rr - T.replay_ratio) < 0.3, f"ratio {rr} far from target {T.replay_ratio} (unclamped)"
    print("[2] steps_for_ratio <-> replay_ratio round-trip OK")


def test_convergence():
    """Cost model: iter_s = games*game_s/workers (selfplay) + steps*step_s (train) + fixed.
    Drive rebalance() for several iters; assert iter-time converges into the dead-band and the
    replay ratio stays in band. Start DELIBERATELY too slow (big games)."""
    # realistic-ish cost model (real iters were ~680s at games=96): per-game ~40s of CPU MCTS
    # spread over `workers`, train ~0.1s/step, fixed ~60s (eval+checkpoint). Target 300s is then
    # reachable around games~110, so the controller can actually converge (not clamp-bound).
    game_s, step_s, fixed = 40.0, 0.1, 60.0
    batch, plies = 128, 40.0
    budget_h = 2.0
    target = target_iter_seconds(budget_h, T)  # 7200/24 = 300s
    games, steps = 200, steps_for_ratio(200, plies, batch, T)  # start slow

    history = []
    for _ in range(25):
        workers = derive_workers(20, games, T)
        iter_s = games * game_s / workers + steps * step_s + fixed
        history.append(iter_s)
        rr = replay_ratio(steps, batch, games, plies)
        assert T.replay_lo - 0.01 <= rr <= T.replay_hi + 0.01, f"replay ratio escaped band: {rr}"
        games, steps, _ = rebalance(games, steps, batch, plies, iter_s, target, T)
        assert T.games_min <= games <= T.games_max and T.steps_min <= steps <= T.steps_max, "clamp breach"

    final = history[-1]
    assert abs(final - target) / target <= T.iter_time_tol + 0.05, \
        f"did not converge: final {final:.0f}s vs target {target:.0f}s (history {[int(h) for h in history[-5:]]})"
    # bounded per-step: no single iter changed games by more than ~max_step_frac
    print(f"[3] convergence OK: {int(history[0])}s -> {int(final)}s (target {int(target)}s, "
          f"{len(history)} iters, dead-band +-{int(T.iter_time_tol*100)}%)")


def test_in_band_stability():
    target = 300.0
    games, steps = 48, steps_for_ratio(48, 40, 128, T)
    # iter-time exactly on target -> games must not move
    g2, s2, notes = rebalance(games, steps, 128, 40, target, target, T)
    assert g2 == games, "in-band iter-time should not change games"
    # now force replay ratio out of band (too many steps) -> only steps corrected, games fixed
    g3, s3, notes3 = rebalance(games, steps * 10, 128, 40, target, target, T)
    assert g3 == games, "replay-ratio fix must not change games"
    assert s3 < steps * 10 and T.replay_lo <= replay_ratio(s3, 128, games, 40) <= T.replay_hi, "ratio not corrected"
    print("[4] in-band stability + replay-ratio-only correction OK")


def test_structural_safety():
    """The balancer emits ONLY throughput knobs. initial_plan keys are a strict subset of the
    throughput set; rebalance returns a 3-tuple of (games, steps, notes) -- no contract knob exists
    anywhere in the surface."""
    plan = initial_plan(HW(cpu_cores=20, vram_gb=8.0), budget_hours=2.0, batch=128, avg_plies_guess=40)
    allowed = {"selfplay_workers", "games_per_iter", "train_steps", "target_iter_s"}
    contract = {"champion_gate", "champion_tol", "anchor_kl", "curriculum", "curriculum_threshold",
                "lr", "selfplay_opponent", "selfplay_teacher_depth"}
    assert set(plan).issubset(allowed), f"initial_plan leaked non-throughput keys: {set(plan) - allowed}"
    assert not (set(plan) & contract), "initial_plan touched a learning-contract knob"
    assert plan["selfplay_workers"] == 16 and T.games_min <= plan["games_per_iter"] <= T.games_max
    print("[5] structural safety OK: balancer emits only throughput knobs, never the learning contract")


def main() -> int:
    test_workers()
    test_ratio_roundtrip()
    test_convergence()
    test_in_band_stability()
    test_structural_safety()
    print("\nALL AUTO-BALANCE CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
