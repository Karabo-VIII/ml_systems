"""AUTO-BALANCE: let the system self-determine the THROUGHPUT knobs (workers / games-per-iter /
train-steps) from hardware + wall-clock budget, balanced per unit time, instead of the human
hand-tuning them -- WITHOUT touching the LEARNING-CONTRACT knobs (champion gate, anchor-kl,
curriculum thresholds, LR, opponent), which encode what counts as progress and must stay
principled. This separation is the whole point: auto-tuning throughput is safe; auto-tuning the
learning contract = optimizing a proxy that degrades the real objective (the chess lesson).

Two principled targets the balancer holds:
  1. REPLAY RATIO = (train_steps * batch) / (games_per_iter * avg_plies)  -- gradient-samples per
     new sample. Held in a sane band (~1-4): too high overfits the buffer, too low wastes data.
     This is what COUPLES train_steps to games_per_iter (so you never set both by hand).
  2. ITERS-IN-BUDGET -- enough champion-gate promotion ATTEMPTS in the window. Drives the target
     per-iter wall-time = budget / iters_in_budget; the online controller nudges games (and steps,
     to keep the replay ratio) toward that target each iteration.

GPU blow-up is already handled elsewhere (the floor-OOM backoff shrinks sims/batch under VRAM
pressure). This module handles CPU/iter-time bloat: it never lets an iteration run away.

PURE + deterministic + unit-testable (no torch, no I/O). By construction it returns ONLY throughput
knobs -- it has no access to and never emits a learning-contract knob, so it cannot weaken the gate.
No emoji (Windows cp1252).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HW:
    cpu_cores: int
    vram_gb: float = 8.0


@dataclass
class BalanceTargets:
    iters_in_budget: int = 24          # >= this many champion-gate attempts across the window
    replay_ratio: float = 2.0          # gradient-samples / new-samples (sane band below)
    replay_lo: float = 1.0
    replay_hi: float = 4.0
    iter_time_tol: float = 0.30        # dead-band: +-30% around target iter-time before adjusting
    max_step_frac: float = 0.25        # bound a single iter's games change to +-25% (no thrashing)
    worker_headroom: int = 2           # leave cores for the GPU feeder + OS
    worker_cap: int = 16               # diminishing returns + pool overhead beyond this
    games_min: int = 8
    games_max: int = 256
    steps_min: int = 20
    steps_max: int = 2000


def derive_workers(cores: int, games_per_iter: int, targets: BalanceTargets) -> int:
    """Hardware-derived: one CPU worker per core (minus headroom), never more than there are games
    to play, capped. Self-play is CPU/Python-bound so this is the dominant throughput lever."""
    return max(1, min(cores - targets.worker_headroom, games_per_iter, targets.worker_cap))


def target_iter_seconds(budget_hours: float, targets: BalanceTargets) -> float:
    return (budget_hours * 3600.0) / max(1, targets.iters_in_budget)


def replay_ratio(train_steps: int, batch: int, games: int, avg_plies: float) -> float:
    new_samples = max(1.0, games * max(1.0, avg_plies))
    return (train_steps * batch) / new_samples


def steps_for_ratio(games: int, avg_plies: float, batch: int, targets: BalanceTargets) -> int:
    """train_steps that hits the TARGET replay ratio for a given games_per_iter."""
    new_samples = games * max(1.0, avg_plies)
    s = int(round(targets.replay_ratio * new_samples / max(1, batch)))
    return max(targets.steps_min, min(targets.steps_max, s))


def _clamp_games(g: int, targets: BalanceTargets) -> int:
    return max(targets.games_min, min(targets.games_max, int(g)))


def rebalance(games: int, train_steps: int, batch: int, avg_plies: float,
              measured_iter_s: float, target_iter_s: float, targets: BalanceTargets):
    """ONLINE controller: nudge games_per_iter toward the target iter-time, then set train_steps to
    keep the replay ratio at target. Bounded per step (no thrash). Returns (games, steps, notes).

    Only throughput knobs in, only throughput knobs out -- the learning contract is structurally
    out of reach."""
    notes = []
    if target_iter_s <= 0 or measured_iter_s <= 0:
        return games, train_steps, notes
    err = (measured_iter_s - target_iter_s) / target_iter_s   # >0 = too slow

    if abs(err) <= targets.iter_time_tol:
        # iter-time in band -> leave games; only correct the replay ratio if it drifted out of band
        rr = replay_ratio(train_steps, batch, games, avg_plies)
        if rr < targets.replay_lo or rr > targets.replay_hi:
            new_steps = steps_for_ratio(games, avg_plies, batch, targets)
            if new_steps != train_steps:
                notes.append(f"[auto-balance] replay-ratio {rr:.2f} out of [{targets.replay_lo},"
                             f"{targets.replay_hi}] -> train_steps {train_steps}->{new_steps}")
                return games, new_steps, notes
        return games, train_steps, notes

    # out of band -> scale games toward target, bounded; iter too slow (err>0) shrinks games
    factor = 1.0 - max(-targets.max_step_frac, min(targets.max_step_frac, err))
    new_games = _clamp_games(round(games * factor), targets)
    new_steps = steps_for_ratio(new_games, avg_plies, batch, targets)
    if new_games != games or new_steps != train_steps:
        notes.append(f"[auto-balance] iter {measured_iter_s:.0f}s vs target {target_iter_s:.0f}s "
                     f"(err {err:+.0%}) -> games {games}->{new_games}, train_steps "
                     f"{train_steps}->{new_steps} (replay~{targets.replay_ratio:.1f})")
    return new_games, new_steps, notes


def initial_plan(hw: HW, budget_hours: float, batch: int, avg_plies_guess: float,
                 targets: BalanceTargets | None = None) -> dict:
    """Startup point before any live timing exists: pick games_per_iter from iters-in-budget, set
    train_steps for the target replay ratio, derive workers from cores. The online rebalance() then
    refines from the FIRST measured iter onward (iter 0 is the live calibration)."""
    t = targets or BalanceTargets()
    games = _clamp_games(t.iters_in_budget, t)  # neutral seed; refined online once timing is known
    workers = derive_workers(hw.cpu_cores, games, t)
    steps = steps_for_ratio(games, avg_plies_guess, batch, t)
    return {"selfplay_workers": workers, "games_per_iter": games, "train_steps": steps,
            "target_iter_s": target_iter_seconds(budget_hours, t)}
