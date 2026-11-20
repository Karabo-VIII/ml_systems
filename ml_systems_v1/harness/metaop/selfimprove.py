"""harness/metaop/selfimprove.py -- the CONTINUOUS self-evolution daemon for the standalone harness.

Hands-off: you start it once and it keeps improving the PLANNER against the HONEST solve-rate fitness, round after
round, with no further input. Each round seeds from the best-so-far champion, evolves, and (monotonically) installs
any improvement -- so the planner that the next `launch`/`resume` uses gets better over time ON ITS OWN. Bounded by
`--rounds` (or `--max-minutes`); resumable -- the champion persists on disk between rounds AND between process runs,
so stopping and restarting the daemon continues where it left off.

This is the standalone equivalent of the project's evolution_loop, shipped INSIDE the harness so "self-evolving out
of the box" actually means it runs itself. The planner-evolution signal needs NO external objectives (it scores
against the built-in benchmark); skill-growth from real work is the separate `--harvest` lever on `launch`.

  python -m metaop.manager improve --backend cli --rounds 20        # 20 hands-off rounds, then stop
  python -m metaop.manager improve --backend cli --max-minutes 60   # run continuously for an hour
  python -m metaop.manager improve --backend cli --rounds 0         # until --max-minutes (or Ctrl-C); resumes on restart

No third-party deps. No emoji (Windows cp1252).
"""
from __future__ import annotations

import time

from .brain import make_brain
from . import evolve as _evo
from .champion import write_champion, read_champion


def self_improve(rounds: int = 10, backend: str = "cli", generations: int = 2, pop_size: int = 3,
                 eval_limit: int = 4, max_minutes: float = 0.0, domain: str | None = None, log=print) -> list:
    """Run the continuous self-evolution daemon. Returns the per-round history. `rounds<=0` means 'until max_minutes'
    (and max_minutes<=0 with rounds<=0 means a single round -- never an unbounded spin). Monotonic + resumable: the
    champion on disk is the durable state; a worse round can never regress it (write_champion refuses)."""
    if rounds <= 0 and max_minutes <= 0:
        rounds = 1  # never spin unbounded
    t0 = time.time()
    history: list = []
    r = 0
    start_rate = float((read_champion() or {}).get("best_solve_rate", 0.0))
    while True:
        if rounds > 0 and r >= rounds:
            break
        if max_minutes > 0 and (time.time() - t0) >= max_minutes * 60:
            break
        brain = make_brain(backend, domain=domain or "a software engine-builder project")
        cur = read_champion()                       # seed each round from the best-so-far -> monotonic chaining
        seed = cur.get("prompt") if cur else None
        base = float((cur or {}).get("best_solve_rate", 0.0))
        try:
            res = _evo.evolve_planner(brain, generations=generations, pop_size=pop_size,
                                      eval_limit=eval_limit, seed_prompt=seed)
        except Exception as e:
            log(f"  round {r + 1}: evolve FAILED ({type(e).__name__}: {str(e)[:120]}) -- stopping")
            break
        seed_f, best_f = float(res.get("seed_fitness", 0.0)), float(res.get("best_fitness", 0.0))
        installed = False
        if res.get("best") is not None:
            p = write_champion(str(res["best"]), best_f, max(seed_f, base), source="selfimprove")
            installed = bool(p) and best_f > base   # write_champion itself refuses to regress an incumbent
        champ_now = max(best_f, base)
        history.append({"round": r + 1, "seed_fitness": seed_f, "best_fitness": best_f,
                        "champion_solve_rate": champ_now, "installed_new": installed})
        log(f"  round {r + 1}: solve_rate {seed_f:.3f} -> {best_f:.3f}  | champion now {champ_now:.3f}"
            + ("  (NEW champion installed)" if installed else ""))
        r += 1
    final = float((read_champion() or {}).get("best_solve_rate", 0.0))
    log(f"=== self-improve done: {r} round(s), champion solve_rate {start_rate:.3f} -> {final:.3f} "
        f"({'improved' if final > start_rate else 'no gain this session'}) ===")
    return history
