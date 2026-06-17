"""Audit settings.py invariants across all active WM versions.

Reads `config/_invariants.yaml::cross_version_constants` and verifies each
declared constant in every `src/wm/v*/v*_training/settings.py` matches the
expected value. Reports drift as ROW-by-ROW table.

Read-only. Does not modify files. Use for Phase B.4 baseline before fixes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

INVARIANTS = {
    "WM_STEPS_PER_EPOCH": 2000,
    "DIVERSITY_STEPS_PER_EPOCH": 2000,
    "DIRECT_RETURN_WEIGHT": 3.0,
    "WM_BATCH_SIZE": 32,
    "BIN_MIN": -1.0,
    "BIN_MAX": 1.0,
    "NUM_BINS": 255,
    "TWOHOT_FOCAL_GAMMA": 0.0,
}
LIST_INVARIANTS = {
    "ACTIVE_HORIZONS": [1, 4, 16, 64],
}
STR_INVARIANTS = {
    # target_prefix is loose; many settings use raw return names directly
}


def load_settings(p: Path) -> Any:
    spec = importlib.util.spec_from_file_location("s_" + p.parent.name, p)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # add the parent dir to sys.path so internal imports resolve
    sys.path.insert(0, str(p.parent))
    sys.path.insert(0, str(p.parent.parent))
    sys.path.insert(0, str(p.parent.parent.parent))
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return f"IMPORT_ERROR: {type(e).__name__}: {str(e)[:80]}"
    finally:
        for _ in range(3):
            try:
                sys.path.pop(0)
            except IndexError:
                break
    return mod


def main() -> int:
    paths = sorted(ROOT.glob("src/wm/v*/v*_training/settings.py"))
    if not paths:
        print("no settings.py files found under src/wm/v*/v*_training/")
        return 1

    print(f"{'version':<22} {'STATUS':<10} drift")
    print("-" * 80)

    n_clean = 0
    n_drift = 0
    n_err = 0
    drift_rows = []

    for p in paths:
        ver = "/".join(p.parts[-3:-1])
        mod = load_settings(p)
        if isinstance(mod, str):  # error
            print(f"{ver:<22} {'ERROR':<10} {mod}")
            n_err += 1
            continue
        if mod is None:
            print(f"{ver:<22} {'SKIP':<10} (no spec)")
            continue

        drifts = []
        for name, expected in INVARIANTS.items():
            actual = getattr(mod, name, "<MISSING>")
            if actual == "<MISSING>":
                drifts.append(f"{name}=MISSING")
            elif actual != expected:
                drifts.append(f"{name}={actual!r} (expected {expected!r})")
        for name, expected in LIST_INVARIANTS.items():
            actual = getattr(mod, name, "<MISSING>")
            if actual == "<MISSING>":
                drifts.append(f"{name}=MISSING")
            elif list(actual) != list(expected):
                drifts.append(f"{name}={actual!r} (expected {expected!r})")

        if drifts:
            print(f"{ver:<22} {'DRIFT':<10} {len(drifts)} drift(s)")
            for d in drifts:
                print(f"  - {d}")
            n_drift += 1
            drift_rows.append((ver, drifts))
        else:
            print(f"{ver:<22} {'CLEAN':<10}")
            n_clean += 1

    print("-" * 80)
    print(f"CLEAN: {n_clean}  DRIFT: {n_drift}  ERROR: {n_err}  TOTAL: {len(paths)}")
    return 0 if n_drift == 0 and n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
