"""Audit subsystem — Contract-Driven Audit Protocol (CDAP).

Per docs/DOUBLE_AUDIT_PROTOCOL.md.

Core entry point: src/audit/check_invariants.py — runs the global invariants
registry (config/_invariants.yaml) against the current tree. Exit codes:
    0 = clean
    1 = warnings only (non-blocking)
    2 = critical drift (block commit)

Companion: src/audit/contract_loader.py — loads per-file `__contract__` dicts
for input/output/invariant declarations (CDAP Axis 1).
"""
