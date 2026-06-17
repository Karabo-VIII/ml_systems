"""Operator CHAMPION -- crypto-consumer SHIM over the canonical harness.metaop.champion (U1 install seam).

The champion install seam (persist the evolved/dspy planner prompt + GATED apply onto the live brain) lives ONCE in
harness/metaop/champion.py. This shim re-exports it UNCHANGED so the live crypto loop (manager.make_brain/launch) and
the copy-parity firewall can reach the SAME symbols. The canonical champion path already defaults to the crypto live
loop's runs/autonomy/evolve/champion.json (resolved off the repo root), so no workspace injection is needed here.
No emoji (Windows cp1252).
"""
from __future__ import annotations

from harness.metaop import champion as _h  # noqa: F401  the canonical engine
from harness.metaop.champion import (  # noqa: F401
    apply_champion, write_champion, read_champion, champion_path, is_improvement, REQUIRED_FIELDS,
)
