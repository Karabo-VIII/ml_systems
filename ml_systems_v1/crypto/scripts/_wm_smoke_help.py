"""Smoke-test every WM trainer / stub by running `--help` and capturing
import errors. Used by Phase B+D follow-up review."""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    # Tier 1 active (full)
    ("V1.0",  "src/wm/v1/v1_0_training/train_world_model.py"),
    ("V1.1",  "src/wm/v1/v1_1_training/train_world_model.py"),
    ("V1.4",  "src/wm/v1/v1_4_training/train_world_model.py"),
    ("V1.6",  "src/wm/v1/v1_6_training/train_world_model.py"),
    ("V3",    "src/wm/v3/v3_training/train_world_model.py"),
    ("V4",    "src/wm/v4/v4_training/train_world_model.py"),
    ("V6",    "src/wm/v6/v6_training/train_world_model.py"),
    ("V8",    "src/wm/v8/v8_training/train_world_model.py"),
    ("V9",    "src/wm/v9/v9_training/train_world_model.py"),
    # Tier 3 frozen
    ("V11",   "src/wm/v11/v11_training/train_world_model.py"),
    ("V12",   "src/wm/v12/v12_training/train_world_model.py"),
    ("V13",   "src/wm/v13/v13_training/train_world_model.py"),
    ("V14",   "src/wm/v14/v14_training/train_world_model.py"),
    # Tier 4 stubs
    ("V15",   "src/wm/v15/patchtst_encoder.py"),
    # V16/V17 reclassified 2026-06-11 to A1 backbones (src/agents/...).
    ("V16",   "src/agents/a1_wm_consuming/backbones/v16_dreamerv3/v16_training/dreamer_v3.py"),
    ("V17",   "src/agents/a1_wm_consuming/backbones/v17_tdmpc2/v17_training/td_mpc2.py"),
    ("V18",   "src/wm/v18/v18_training/finetune_chronos.py"),
    ("V19",   "src/wm/v19/v19_training/smoke_test.py"),
    # V10 meta
    ("V10",   "src/wm/v10/v10_meta/train_meta.py"),
]


def smoke(label: str, rel: str) -> tuple[str, str]:
    """Returns (status, detail). status in {OK, IMPORT_ERR, NO_HELP, NO_FILE, OTHER}."""
    p = ROOT / rel
    if not p.exists():
        return ("NO_FILE", f"missing: {rel}")
    try:
        r = subprocess.run(
            [sys.executable, str(p), "--help"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return ("TIMEOUT", "30s no response")
    except Exception as e:
        return ("OTHER", f"{type(e).__name__}: {e}")
    out = (r.stdout or "") + (r.stderr or "")
    if "usage:" in out.lower() or "options:" in out.lower():
        # Distill the flags advertised
        flags = sorted(set(t.split("=")[0].split()[0] for t in out.split() if t.startswith("--") and len(t) > 4))
        return ("OK", " ".join(flags[:12]))
    if "ImportError" in out or "ModuleNotFoundError" in out or "AttributeError" in out:
        first_err_line = next((ln for ln in out.splitlines() if "Error" in ln), "")
        return ("IMPORT_ERR", first_err_line[:160])
    if r.returncode != 0:
        return ("OTHER", out.splitlines()[-1][:160] if out.strip() else f"rc={r.returncode}")
    return ("NO_HELP", "ran clean but no usage output")


def main() -> int:
    print(f"{'version':<6} {'status':<12} {'detail':<60}")
    print("-" * 90)
    n_ok = 0
    n_fail = 0
    for label, rel in TARGETS:
        status, detail = smoke(label, rel)
        print(f"{label:<6} {status:<12} {detail[:60]}")
        if status == "OK":
            n_ok += 1
        else:
            n_fail += 1
    print("-" * 90)
    print(f"OK: {n_ok}  FAIL: {n_fail}  TOTAL: {len(TARGETS)}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
