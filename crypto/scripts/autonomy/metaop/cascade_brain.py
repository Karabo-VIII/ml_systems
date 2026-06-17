"""Operator CASCADE_BRAIN -- crypto-consumer SHIM over the canonical harness.metaop.cascade_brain (U3/U4/U6).

The cheap->strong escalation router (CascadeBrain + make_cascade) lives ONCE in harness/metaop/cascade_brain.py
(project-agnostic; a drop-in Brain). This shim re-exports it UNCHANGED so the scripts metaop package exposes the SAME
symbols (the copy-parity firewall requires it) and the live crypto loop can select the 'cascade' backend / the graph
dispatch can call set_node_context on it. make_brain('cascade') in the crypto brain shim already routes here through
the canonical engine. No emoji (Windows cp1252).
"""
from __future__ import annotations

from harness.metaop import cascade_brain as _h  # noqa: F401  the canonical engine
from harness.metaop.cascade_brain import CascadeBrain, make_cascade  # noqa: F401
