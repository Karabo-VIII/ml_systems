"""Operator EVOLVE -- crypto-consumer SHIM over the canonical harness.metaop.evolve (U6 copy-parity reachability).

The dependency-free EVOLUTIONARY optimizer (evolve / evolve_planner: optimize the planner prompt against the HONEST
mechanical solve_rate) lives ONCE in harness/metaop/evolve.py (project-agnostic). This shim re-exports it UNCHANGED so
the scripts metaop package exposes the SAME symbols the harness does (the copy-parity firewall requires it) and the
live crypto loop can drive an evolve pass through `metaop.evolve`. No emoji (Windows cp1252).
"""
from __future__ import annotations

from harness.metaop import evolve as _h  # noqa: F401  the canonical engine
from harness.metaop.evolve import evolve, evolve_planner  # noqa: F401
