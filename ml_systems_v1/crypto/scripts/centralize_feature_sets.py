"""One-shot helper to wire each version's settings.py to import from
src/feature_sets.py instead of defining its own FEATURE_LIST_* constants.

This is a SCAN tool that prints what each settings.py currently defines vs
what's centrally available. Manual edits follow — we don't auto-rewrite
because each version's settings.py has version-specific constants we
shouldn't blow away.

Run: python scripts/centralize_feature_sets.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERSIONS = [
    "src/wm/v0/v0_baseline/settings.py",
    "src/wm/v1/v1_0_training/settings.py",
    "src/wm/v1/v1_1_training/settings.py",
    "src/wm/v1/v1_4_training/settings.py",
    "src/wm/v1/v1_6_training/settings.py",
    "backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training/settings.py",
    "src/wm/v3/v3_training/settings.py",
    "src/wm/v4/v4_training/settings.py",
    "backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training/settings.py",
    "src/wm/v6/v6_training/settings.py",
    "backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training/settings.py",
    "src/wm/v8/v8_training/settings.py",
    "src/wm/v9/v9_training/settings.py",
    "src/wm/v10/v10_meta/settings.py",
    "src/wm/v11/v11_training/settings.py",
    "src/wm/v12/v12_training/settings.py",
    "src/wm/v13/v13_training/settings.py",
    "src/wm/v14/v14_training/settings.py",
    "src/wm/v19/v19_training/settings.py",
]

FEATURE_LIST_PATTERN = re.compile(r"^FEATURE_LIST_(\d+)\s*=", re.MULTILINE)


def main():
    print(f"{'Version':<45} {'Defines':<30} Imports central?")
    print("-" * 95)
    for rel in VERSIONS:
        p = ROOT / rel
        if not p.exists():
            print(f"  {rel:<43} (missing)")
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        defines = sorted({int(m.group(1)) for m in FEATURE_LIST_PATTERN.finditer(txt)})
        imports = "YES" if "from feature_sets import" in txt else "no"
        print(f"  {rel:<43} f{defines}  {imports}")


if __name__ == "__main__":
    main()
