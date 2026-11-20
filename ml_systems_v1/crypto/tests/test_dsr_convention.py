"""DSR kurtosis-convention guard (R32+++ validator-CRIT fix, 2026-05-15).

Verifies all 5 DSR implementations honour the explicit `kurtosis_convention`
keyword and yield IDENTICAL output for Pearson/Fisher inputs of the same
underlying distribution.

Pre-fix bug: 3 discovery scripts passed `scipy.stats.kurtosis(arr, fisher=True)`
(excess; normal=0) into DSR functions whose `(kurt-1)/4` term assumed Pearson
(normal=3). Variance was UNDERSTATED 1.5-3x => DSR INFLATED. Worst measured
case: SR=2.4 / 30 trials / kurt_excess=8 inflates DSR by +3.3pp, flipping
"MARGINAL 0.90" cells to "PASS 0.93" false-positive deploys.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy.deflated_sharpe import (  # noqa: E402
    deflated_sharpe_ratio,
    kurt_to_pearson,
    KURT_PEARSON,
    KURT_FISHER,
)
from analysis.honest_validation import deflated_sharpe as dsr_honest  # noqa: E402
from analysis.eval_v2 import compute_dsr  # noqa: E402
from analysis.model_bakeoff import deflated_sharpe as dsr_bakeoff  # noqa: E402
from strategy.ta_sml.cpcv_validate import deflated_sharpe as dsr_cpcv  # noqa: E402


# Use a single fat-tail return stream for reproducibility
@pytest.fixture(scope="module")
def returns():
    rng = np.random.default_rng(42)
    return rng.standard_t(df=5, size=500) * 0.02 + 0.001


@pytest.fixture(scope="module")
def moments(returns):
    return {
        "skew": float(stats.skew(returns)),
        "kurt_fisher": float(stats.kurtosis(returns, fisher=True)),
        "kurt_pearson": float(stats.kurtosis(returns, fisher=True)) + 3.0,
    }


def test_helper_round_trip():
    assert kurt_to_pearson(4.0, KURT_PEARSON) == 4.0
    assert kurt_to_pearson(1.0, KURT_FISHER) == 4.0


def test_helper_rejects_unknown():
    with pytest.raises(ValueError):
        kurt_to_pearson(0.0, "skewness")


def test_deflated_sharpe_ratio_convention_equivalence(moments):
    sh, n, n_trials = 2.4, 252, 30  # gradient-zone params
    r_p = deflated_sharpe_ratio(sh, n_trials, sr_variance=0.5, n_returns=n,
                                 skewness=moments["skew"],
                                 kurtosis=moments["kurt_pearson"],
                                 kurtosis_convention="pearson")
    r_f = deflated_sharpe_ratio(sh, n_trials, sr_variance=0.5, n_returns=n,
                                 skewness=moments["skew"],
                                 kurtosis=moments["kurt_fisher"],
                                 kurtosis_convention="fisher")
    assert abs(r_p["dsr"] - r_f["dsr"]) < 1e-9


def test_honest_validation_convention_equivalence(moments):
    sh, n, n_trials = 2.4, 252, 30
    p = dsr_honest(sh, n, moments["skew"], moments["kurt_pearson"], n_trials,
                    kurtosis_convention="pearson")
    f = dsr_honest(sh, n, moments["skew"], moments["kurt_fisher"], n_trials,
                    kurtosis_convention="fisher")
    assert abs(p[0] - f[0]) < 1e-9


def test_eval_v2_compute_dsr_convention_equivalence(moments):
    sh, n, n_trials = 2.4, 252, 30
    p = compute_dsr(sh, n, n_trials, moments["skew"], moments["kurt_pearson"],
                     kurtosis_convention="pearson")
    f = compute_dsr(sh, n, n_trials, moments["skew"], moments["kurt_fisher"],
                     kurtosis_convention="fisher")
    assert abs(p - f) < 1e-9


def test_model_bakeoff_convention_equivalence(moments):
    sh, n, n_trials = 2.4, 252, 30
    p = dsr_bakeoff(sh, n, moments["skew"], moments["kurt_pearson"], n_trials,
                     kurtosis_convention="pearson")
    f = dsr_bakeoff(sh, n, moments["skew"], moments["kurt_fisher"], n_trials,
                     kurtosis_convention="fisher")
    assert abs(p - f) < 1e-9


def test_cpcv_convention_equivalence(moments):
    sh, n, n_trials, sh_std = 2.4, 252, 30, 0.5
    f = dsr_cpcv(sh, sh_std, n_trials, n, moments["skew"], moments["kurt_fisher"],
                  kurtosis_convention="fisher")
    p = dsr_cpcv(sh, sh_std, n_trials, n, moments["skew"], moments["kurt_pearson"],
                  kurtosis_convention="pearson")
    assert abs(p[0] - f[0]) < 1e-9


def test_all_dsr_fns_reject_bad_convention():
    sh, n, n_trials = 2.4, 252, 30
    with pytest.raises(ValueError):
        dsr_honest(sh, n, 0.0, 3.0, n_trials, kurtosis_convention="nonsense")
    with pytest.raises(ValueError):
        compute_dsr(sh, n, n_trials, 0.0, 3.0, kurtosis_convention="nonsense")
    with pytest.raises(ValueError):
        dsr_bakeoff(sh, n, 0.0, 3.0, n_trials, kurtosis_convention="nonsense")
    with pytest.raises(ValueError):
        dsr_cpcv(sh, 0.5, n_trials, n, 0.0, 0.0, kurtosis_convention="nonsense")
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(sh, n_trials, 0.5, n, kurtosis_convention="nonsense")


def test_pre_fix_bug_inflated_dsr():
    """Reproduces the exact false-positive-deploy bug pattern."""
    sh, n, n_trials = 2.4, 252, 30
    skew, kurt_fisher = -0.3, 8.0
    # The bug path: pass Fisher kurt but get Pearson treatment
    bug_dsr, _ = dsr_honest(sh, n, skew, kurt_fisher, n_trials)  # default pearson
    correct_dsr, _ = dsr_honest(sh, n, skew, kurt_fisher, n_trials,
                                  kurtosis_convention="fisher")
    # Bug inflates DSR (lower variance => higher z => higher cdf)
    assert bug_dsr > correct_dsr, (
        f"Bug should inflate DSR but bug={bug_dsr:.4f} <= correct={correct_dsr:.4f}"
    )
    # Inflation should be material (3pp+ in this regime)
    inflation_pp = (bug_dsr - correct_dsr) * 100
    assert inflation_pp >= 2.0, f"Expected >=2pp inflation, got {inflation_pp:.2f}pp"
