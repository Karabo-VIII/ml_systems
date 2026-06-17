"""One-shot scan: every V*/settings.py vs the cross-version invariants.

Intended as a diagnostic — gives the full drift picture at a glance, including
which versions are clean and which have intentional architectural deviations.
"""
import glob
import re
from pathlib import Path

INVARIANTS = {
    "WM_BATCH_SIZE":              32,
    "BIN_MIN":                    -1.0,
    "BIN_MAX":                    1.0,
    "NUM_BINS":                   255,
    "TWOHOT_FOCAL_GAMMA":         0.0,
    "WM_STEPS_PER_EPOCH":         2000,
    "DIVERSITY_STEPS_PER_EPOCH":  2000,
    "DIRECT_RETURN_WEIGHT":       3.0,
}


def _parse_value(raw: str):
    raw = raw.strip().rstrip(",")
    try:
        return eval(raw, {"__builtins__": {}}, {})
    except Exception:
        return raw


def main():
    files = sorted(glob.glob("src/wm/v*/v*_training/settings.py"))
    print(f"Found {len(files)} V settings files")
    print()

    rows = []
    n_check = 0
    n_drift = 0
    for fp in files:
        text = Path(fp).read_text(encoding="utf-8")
        short = fp.replace("\\", "/").replace("src/", "").replace("/settings.py", "")
        for inv, exp in INVARIANTS.items():
            m = re.search(r"^\s*" + inv + r"\s*[:=]\s*([^\s#]+)", text, re.M)
            if not m:
                continue
            n_check += 1
            v = _parse_value(m.group(1))
            if v != exp:
                n_drift += 1
                rows.append((short, inv, v, exp))

    if not rows:
        print("CLEAN: no drift across any V model")
        return

    print(f"{'file':<28} {'INVARIANT':<28} {'value':<10} {'expected':<10}")
    print("-" * 80)
    for short, inv, v, exp in rows:
        print(f"{short:<28} {inv:<28} {str(v):<10} {str(exp):<10}")
    print()
    print(f"TOTAL: {n_check} invariant-checks across {len(files)} files; "
          f"{n_drift} drift findings")


if __name__ == "__main__":
    main()
