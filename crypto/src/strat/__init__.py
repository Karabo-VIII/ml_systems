"""src/strat/ -- the V4 trading-strategy APPARATUS (the validation gate every candidate flows through).

Rebuilt 2026-06-05 on the kept CanonicalHarness (the archived src/strat/ was wiped in the 2026-06-04
reset; the discover/trader skills reference this path). This is the FOUNDATION toolkit -- it does NOT
contain or endorse any strategy; it is the trustworthy measurement layer for the solving phase.

Public API
----------
Discovery:        discriminate(sym, cadence, H)  ->  candidate gates
                  scan(assets, sma_configs, gates)  ->  per-cell SHIP/NOT-SHIP verdicts
Integrated gate:  evaluate_candidate(harness, family_n)  ->  consolidated verdict (the one callable)
Robustness:       evaluate(...) / evaluate_setup_chaser(...)  ->  Lens A/B/C
Firewall:         random_entry_null(harness)  ->  beats cost-matched random entries?
Cost realism:     apply_fill_model(harness, mode)  ->  taker / maker_pessimistic / ideal_ref
Leak probe:       (use wealth_bot.leak_probe.relative_leak_test -- wired inside evaluate_candidate)

The gate chain (docs/AVENUE_SPECS_2026_06_05.md, every candidate, in order):
    cost-honest backtest (taker 0.24%)  ->  relative leak probe  ->  cost-matched random-entry null
    ->  robustness battery (block-bootstrap p05>0, jk, n_eff>=8, maxDD<20%)  ->  benchmark-excess incl.
    bear (strat.benchmark.benchmark_excess)  ->  DSR/Holm @ true family-N (src/audit/check_dsr_holm.py).
    (block-bootstrap p05 is the seed-equivalent for DETERMINISTIC rule strategies; STOCHASTIC/ML
    candidates must add a 10-seed outer loop to claim the CLAUDE.md "10/10 seeds" bar.)

All modules are numpy/pandas only and import the canonical harness from wealth_bot.harness. Hardened
against the 2026-06-05 apparatus red-audit -- see docs/APPARATUS_AUDIT_2026_06_05.md for the verified
findings (F2/F5/F6/F11/F12 fixed here; F8/F3/F9/F13 are canonical-harness items documented there).
"""
# Self-improvement fix (2026-06-05): make `import src.strat` work from the repo root too, not only when a
# caller has already put src/ on sys.path. The submodules + _default_windows import `wealth_bot.harness`
# (absolute), which needs src/ on the path. Without this, `import src.strat` raised ModuleNotFoundError:
# No module named 'wealth_bot' at load time (the selftest masked it by adding src/ itself).
import os as _os
import sys as _sys
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # src/strat -> src
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

from .battery import (
    evaluate, evaluate_setup_chaser, compound, jackknife, herfindahl_neff,
    block_bootstrap_p05_p95, expectancy, win_rate, profit_factor, monthly,
)
from .firewall import random_entry_null
from .fill_model import apply_fill_model, MODES
from .benchmark import benchmark_excess
from .candidate_gate import evaluate_candidate, build_clean_reference, TAKER_COST_RT
from .discover import discriminate, scan, GATE_FEATS


def _default_windows():
    """The canonical TRAIN/VAL/OOS/UNSEEN split (G-7: one authoritative source; previously hardcoded in
    discover/candidate_gate/positive_control). UNSEEN = [oos_end, unseen_end] = [2025-12-31, 2026-05-22]."""
    from wealth_bot.harness import WindowSpec
    return WindowSpec(train_end="2024-05-15", val_end="2025-03-15",
                      oos_end="2025-12-31", unseen_end="2026-05-22")


DEFAULT_WINDOWS = _default_windows()

__all__ = [
    "evaluate", "evaluate_setup_chaser", "compound", "jackknife", "herfindahl_neff",
    "block_bootstrap_p05_p95", "expectancy", "win_rate", "profit_factor", "monthly",
    "random_entry_null", "apply_fill_model", "MODES", "benchmark_excess",
    "evaluate_candidate", "build_clean_reference", "TAKER_COST_RT",
    "discriminate", "scan", "GATE_FEATS", "DEFAULT_WINDOWS",
]
