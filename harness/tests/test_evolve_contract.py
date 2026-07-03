"""EVOLUTION CONTRACT: the generic core's best-so-far is monotonic non-decreasing and never below the seed
(elitism floor); evolve_planner reports the HELD-OUT metrics and never regresses train. With the credential-free
MockBrain (which ignores the planner prompt) there is nothing to optimize, so `improved` is HONESTLY False --
the very behavior that, mislabeled, produced the audit's '114/114 champion==seed' finding."""
from __future__ import annotations

from metaop.evolve import evolve, evolve_planner


def test_generic_core_monotonic_and_floored():
    target = "evolve"

    def fit(s):
        return sum(1 for a, b in zip(s, target) if a == b) / len(target)

    def mut(parent, ctx):
        g, i = ctx.get("generation", 0), ctx.get("index", 0)
        pos = (g * 7 + i * 3) % len(target)
        chars = list(parent.ljust(len(target), "a")[:len(target)])
        chars[pos] = target[pos]
        return "".join(chars)

    res = evolve("aaaaaa", fit, mut, generations=3, pop_size=4, elite_k=1)
    bests = [h["best_fitness"] for h in res["history"]]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:])), "best-so-far must be non-decreasing"
    assert res["best_fitness"] >= res["seed_fitness"], "champion floor: never below the seed"


def test_planner_reports_holdout_and_floors_train():
    r = evolve_planner(generations=1, pop_size=2, eval_limit=2, budget=2)
    assert r["best_solve_rate"] >= r["seed_solve_rate"]                 # train elitism floor
    assert r["has_holdout"] is True                                     # a held-out split exists
    for k in ("test_solve_rate", "seed_test_solve_rate", "improved", "generalizes", "n_train", "n_test"):
        assert k in r, f"evolve_planner must report {k}"
    # MockBrain ignores the planner prompt -> solve_rate invariant -> no honest improvement possible
    assert r["improved"] is False
    assert r["generalizes"] is False
