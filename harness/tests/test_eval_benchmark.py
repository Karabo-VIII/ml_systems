"""BENCHMARK INTEGRITY (RWYB): every benchmark task's known-correct oracle artifact must satisfy its OWN mechanical
verify_cmd, on BOTH the train and held-out(test) splits, in BOTH pre-seeded and planner modes. A task whose oracle
fails its own check is a BROKEN task -- this catches it, and proves the train/test split filter is wired."""
from __future__ import annotations

import pytest

from metaop.eval_harness import run_eval


@pytest.mark.parametrize("planner_mode", [False, True], ids=["pre_seeded", "planner"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_oracle_plumbing_solves_every_task(planner_mode, split):
    card = run_eval(None, oracle_mock=True, planner_mode=planner_mode, split=split)
    assert card["n"] > 0, f"split {split!r} is empty -- the held-out set must be non-degenerate"
    assert card["solve_rate"] == 1.0, (
        f"{'planner' if planner_mode else 'pre-seeded'} split={split}: "
        f"{card['n_passed']}/{card['n']} -- a task's oracle failed its own verifier")
