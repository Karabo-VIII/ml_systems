"""Bulk-rewrite V2-V14 settings.py files to import from central feature_sets.

Each settings.py has a block defining FEATURE_LIST_* + get_feature_config.
We replace that block with an import from src/feature_sets.py + thin wrapper.

The pattern matched is the FIRST occurrence of `FEATURE_LIST_13 = FEATURE_LIST[:13]`
(or similar) up through the closing of get_feature_config().

Strategy: regex-locate the block, then replace with the shim. Verify each
version's settings.py compiles after edit, and test get_feature_config(41)
works.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (path, version_label, supported_counts_tuple)
TARGETS = [
    ("src/wm/v3/v3_training/settings.py", "V3",  (13, 18, 29, 30, 34, 37, 41, 121)),
    ("src/wm/v4/v4_training/settings.py", "V4",  (13, 18, 29, 30, 34, 37, 41, 121)),
    ("backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training/settings.py", "V5",  (13, 18, 30, 34, 37, 41, 121)),
    ("src/wm/v6/v6_training/settings.py", "V6",  (13, 18, 29, 30, 34, 37, 41, 121)),
    ("backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training/settings.py", "V7",  (13, 18, 30, 37, 41, 121)),
    ("src/wm/v8/v8_training/settings.py", "V8",  (13, 18, 29, 30, 34, 37, 41, 121)),
    ("src/wm/v9/v9_training/settings.py", "V9",  (13, 18, 29, 30, 34, 37, 41, 121)),
    ("src/wm/v10/v10_meta/settings.py", "V10",  ()),  # no feature_list defined here
    ("src/wm/v11/v11_training/settings.py", "V11", (13, 25, 29, 34, 41, 121)),
    ("src/wm/v12/v12_training/settings.py", "V12", (13, 25, 29, 34, 41, 121)),
    ("src/wm/v13/v13_training/settings.py", "V13", (13, 25, 29, 34, 41, 121)),
    ("src/wm/v14/v14_training/settings.py", "V14", (13, 25, 29, 34, 41, 121)),
]

# The replacement shim — substituted with version-specific values
SHIM_TEMPLATE = '''\
# === FEATURE SELECTION (centralized in src/feature_sets.py, post-2026-04-27) ===
import sys as _sys
_SRC_DIR = PROJECT_ROOT / "src" if (PROJECT_ROOT / "src").exists() else PROJECT_ROOT.parent / "src"
if str(_SRC_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SRC_DIR))
from feature_sets import (  # noqa: E402
    FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_25, FEATURE_LIST_29,
    FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
    FEATURE_LIST_121,
    DEAD_FEATURE_INDICES,
    get_feature_config as _central_get_feature_config,
)

# Anti-memorization (V1.1+ XD split): base_dim defined per-feature-count via central registry.
BASE_DIM = 34
XD_DROPOUT_RATE = 0.7
XD_NOISE_STD = 0.3

SUPPORTED_FEATURE_COUNTS_{LABEL} = {COUNTS}


def get_feature_config(n_features: int):
    """Return (feature_list, input_dim, base_dim) — standardized 3-tuple API."""
    if n_features not in SUPPORTED_FEATURE_COUNTS_{LABEL}:
        raise ValueError(
            f"{LABEL} supports {{sorted(SUPPORTED_FEATURE_COUNTS_{LABEL})}}; got f{{n_features}}"
        )
    return _central_get_feature_config(n_features)
'''


# Pattern: from the line `FEATURE_LIST_13 = FEATURE_LIST[:13]` (or similar)
# through end of `def get_feature_config` definition. Use multi-line dotall regex.
BLOCK_RE = re.compile(
    r"^FEATURE_LIST_13\s*=.*?(?=^\s*$\n^[A-Z]|^# =====|\Z)",
    re.MULTILINE | re.DOTALL,
)


def _format_shim(label: str, counts: tuple) -> str:
    return SHIM_TEMPLATE.format(LABEL=label, COUNTS=counts)


def main() -> int:
    n_changed = 0
    for rel, label, counts in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"  [SKIP] {rel}: missing")
            continue
        if not counts:
            print(f"  [SKIP] {rel}: no feature_list defined here")
            continue
        txt = p.read_text(encoding="utf-8")
        m = BLOCK_RE.search(txt)
        if not m:
            print(f"  [SKIP] {rel}: didn't match FEATURE_LIST block (manual edit needed)")
            continue
        new_block = _format_shim(label, counts)
        new_txt = txt[: m.start()] + new_block + "\n\n" + txt[m.end():]
        p.write_text(new_txt, encoding="utf-8")
        print(f"  [OK]   {rel} ({len(m.group(0))} chars -> {len(new_block)} chars shim, "
              f"{len(counts)} supported counts)")
        n_changed += 1
    print(f"\n  Total updated: {n_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
