"""Operator DSPY_PLANNER -- crypto-consumer SHIM over the canonical harness.metaop.dspy_planner (U6 reachability).

The LITERAL dspy compiled-planner (compile_planner against the eval_harness-grounded metric + install_compiled_planner)
lives ONCE in harness/metaop/dspy_planner.py. dspy itself is imported LAZILY there, so importing this shim does NOT
require dspy to be installed -- only running compile_planner does. This shim re-exports the symbols UNCHANGED so the
scripts metaop package exposes the SAME surface the harness does (copy-parity firewall) and so install_compiled_planner
is reachable from the live crypto loop. No emoji (Windows cp1252).
"""
from __future__ import annotations

from harness.metaop import dspy_planner as _h  # noqa: F401  the canonical engine (dspy is lazy-imported inside)
from harness.metaop.dspy_planner import (  # noqa: F401
    compile_planner, install_compiled_planner, save_compiled, planner_solve_rate,
)
