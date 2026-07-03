"""CHAMPION HELD-OUT GATE: a champion is installed ONLY when it improves the GATE METRIC -- the HELD-OUT test
solve_rate when present (generalization), else the train solve_rate. An overfit champion (train up, held-out flat)
is REFUSED; a worse-held-out later champion cannot clobber a better one (monotonic floor). The default path lives
under the configurable workspace, never a host repo. Pins the 2026-07-03 held-out gate + path-decouple fixes."""
from __future__ import annotations

from metaop import champion as C


def test_train_gate_when_no_holdout():
    assert C.is_improvement({"best_solve_rate": 0.8, "baseline_solve_rate": 0.5}) is True
    assert C.is_improvement({"best_solve_rate": 0.5, "baseline_solve_rate": 0.5}) is False


def test_overfit_rejected():
    # train jumped 0.5 -> 0.9 but held-out is FLAT -> NOT an improvement (the whole point of the held-out gate)
    assert C.is_improvement({"best_solve_rate": 0.9, "baseline_solve_rate": 0.5,
                             "test_solve_rate": 0.5, "baseline_test_solve_rate": 0.5}) is False


def test_generalization_accepted():
    assert C.is_improvement({"best_solve_rate": 0.9, "baseline_solve_rate": 0.5,
                             "test_solve_rate": 0.75, "baseline_test_solve_rate": 0.5}) is True


def test_gate_metric_prefers_heldout():
    _cand, _base, which = C._gate_metric({"best_solve_rate": 0.9, "baseline_solve_rate": 0.5,
                                          "test_solve_rate": 0.7, "baseline_test_solve_rate": 0.5})
    assert which == "test"


def test_write_monotonic_on_heldout(tmp_path):
    p = str(tmp_path / "champion.json")
    assert C.write_champion("GEN1", 0.7, 0.5, "t", path=p, test_solve_rate=0.7, baseline_test_solve_rate=0.5)
    assert C.read_champion(p)["prompt"] == "GEN1"
    # a better held-out champion replaces it
    C.write_champion("GEN2", 0.8, 0.5, "t", path=p, test_solve_rate=0.85, baseline_test_solve_rate=0.5)
    assert C.read_champion(p)["prompt"] == "GEN2"
    # higher TRAIN but LOWER held-out must NOT clobber -> the floor is on generalization, not in-sample fit
    C.write_champion("WORSE", 0.99, 0.5, "t", path=p, test_solve_rate=0.6, baseline_test_solve_rate=0.5)
    assert C.read_champion(p)["prompt"] == "GEN2"


def test_apply_champion_installs_only_on_gain(tmp_path):
    p = str(tmp_path / "champion.json")

    class _Brain:
        def __init__(self):
            self.plan_instruction = None

        def set_plan_instruction(self, s):
            self.plan_instruction = s

    C.write_champion("PROMPT_GEN", 0.9, 0.5, "t", path=p, test_solve_rate=0.8, baseline_test_solve_rate=0.5)
    b = _Brain()
    res = C.apply_champion(b, path=p)
    assert res["applied"] is True and b.plan_instruction == "PROMPT_GEN"


def test_default_path_not_host_repo():
    dp = str(C.champion_path()).replace("\\", "/")
    assert "runs/autonomy" not in dp  # the old crypto-coupled default is gone
