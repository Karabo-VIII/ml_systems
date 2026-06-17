#!/usr/bin/env python3
"""SessionStart hook (H2) -- re-surface the load-bearing core every session (and on resume).

Counters the "buried in a 5,600-word constitution -> decays / ignored" failure by presenting
the small set of always-true rules fresh at session start, where salience is highest.

Output on stdout (exit 0) is injected as session context. PROPOSED / STAGED. No emoji (cp1252).
"""
import sys

print(
    "[BRAIN CORE -- always true] "
    "(1) Real capital: WEALTH (robust held-out compound) not Sharpe, under LO+spot+lev=1. "
    "(2) RWYB: run code on real data before commit; document command+result. "
    "(3) No inflation: tag claims VERIFIED/REPORTED/INFERRED; check look-ahead, beta-confound, "
    "multi-test; report DD/p05 with returns. "
    "(4) Self-audit before delivering anything (commit OR analysis). "
    "(5) Brain edits go sandbox->review->push (mechanically enforced by brain_guard). "
    "(6) Run the 5-step INTENT preflight every task; spirit >= letter. "
    "(7) Route by task-shape (single-agent for coupled coding; fan-out for breadth; "
    "panel for load-bearing reversals -- see AF_BL_RF). "
    "(8) Honest-stop > busywork; ground every elapsed claim with a real `date`. "
    "Detail is on-demand in the skills -- do not expect the full constitution to be held in context."
)
sys.exit(0)
