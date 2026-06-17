"""Fix TWOHOT_FOCAL_GAMMA drift in V3-V9 variant settings.py.

Audit (`scripts/audit_wm_settings.py`) flags 15 variant settings.py files
that have the constant commented out (with `# REMOVED:` rationale) but no
defined value. Cross-version invariant requires `TWOHOT_FOCAL_GAMMA = 0.0`.

Fix: insert an uncommented assignment immediately after the commented line,
preserving the "REMOVED" rationale as context.

Idempotent: skips files that already have an uncommented assignment.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    "src/wm/v3/v3_1_training/settings.py",
    "src/wm/v3/v3_2_training/settings.py",
    "src/wm/v3/v3_3_training/settings.py",
    "src/wm/v4/v4_1_training/settings.py",
    "src/wm/v4/v4_2_training/settings.py",
    "src/wm/v4/v4_3_training/settings.py",
    "src/wm/v6/v6_1_training/settings.py",
    "src/wm/v6/v6_2_training/settings.py",
    "src/wm/v6/v6_3_training/settings.py",
    "src/wm/v8/v8_1_training/settings.py",
    "src/wm/v8/v8_2_training/settings.py",
    "src/wm/v8/v8_3_training/settings.py",
    "src/wm/v9/v9_1_training/settings.py",
    "src/wm/v9/v9_2_training/settings.py",
    "src/wm/v9/v9_3_training/settings.py",
]

INSERT = (
    "\n# Cross-version invariant: TwoHot focal gamma is DISABLED (focal upweights\n"
    "# temporally-clustered tail returns and accelerates memorization). All\n"
    "# variants inherit base model's TwoHot head; this constant exists for\n"
    "# `from settings import *` callers.\n"
    "TWOHOT_FOCAL_GAMMA = 0.0\n"
)


def main() -> int:
    n_changed = 0
    n_skip = 0
    for rel in TARGETS:
        p = ROOT / rel
        if not p.exists():
            print(f"  SKIP missing: {rel}")
            continue
        text = p.read_text(encoding="utf-8")
        # Idempotency: if uncommented assignment already present, skip
        for line in text.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("TWOHOT_FOCAL_GAMMA"):
                print(f"  already-set: {rel}")
                n_skip += 1
                break
        else:
            # Anchor: append after the existing commented line if present,
            # otherwise append to the end with the rationale block.
            commented = "# TWOHOT_FOCAL_GAMMA"
            if commented in text:
                idx = text.find(commented)
                eol = text.find("\n", idx)
                new = text[:eol + 1] + INSERT + text[eol + 1:]
            else:
                new = text + INSERT
            p.write_text(new, encoding="utf-8")
            print(f"  fixed: {rel}")
            n_changed += 1
    print(f"\nfixed: {n_changed}  already_set: {n_skip}  total: {len(TARGETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
