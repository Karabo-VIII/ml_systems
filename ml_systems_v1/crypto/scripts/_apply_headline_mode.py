"""Apply CC-H4 (HEADLINE_MODE) anti-mem upgrades to V1.x settings.

When env var V1_HEADLINE_MODE=1 is set:
  WM_FREE_NATS    1.0 -> 1.5    (KL latent forced to throw away more info)
  XD_DROPOUT_RATE 0.7 -> 0.85   (drop ~12 of 34 features per batch)

Default off; legacy training paths unchanged.

Per WM_HEADLINE_UPGRADE_PLAN_2026_04_30 §0 CC-H4. Expected effect:
ShIC +0.005-0.012 at IC -0.003-0.008. Ratio (ShIC/IC) lift from
~0.49 to ~0.65+. The point is the RATIO improvement, not raw IC.

Idempotent.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    "src/wm/v1/v1_1_training/settings.py",
    "src/wm/v1/v1_4_training/settings.py",
    "src/wm/v1/v1_6_training/settings.py",
]

PATCH_BLOCK = """
# ─── CC-H4 HEADLINE_MODE anti-mem upgrades (added 2026-04-30) ──────────────
# Per WM_HEADLINE_UPGRADE_PLAN §0 CC-H4. When env var V1_HEADLINE_MODE=1 is
# set BEFORE python invocation, anti-mem knobs tighten to push ShIC into the
# Headline tier (>= 0.045). Default off; legacy training paths unchanged.
#
# Acceptance band per upgrade plan:
#   ShIC delta:  +0.005 to +0.012 (V1.1 base 0.033 -> target 0.040-0.045)
#   IC delta:   -0.003 to -0.008 (V1.1 base 0.067 -> tolerated 0.060)
#   Ratio:       0.49 -> 0.65+ (the load-bearing improvement)
#
# Activation: V1_HEADLINE_MODE=1 python train_world_model.py --features 34
import os as _os
HEADLINE_MODE = bool(int(_os.environ.get("V1_HEADLINE_MODE", "0")))
if HEADLINE_MODE:
    WM_FREE_NATS = 1.5         # was 1.0; raise the KL-throw-away floor
    XD_DROPOUT_RATE = 0.85     # was 0.7; drop ~12 of 34 per batch
    print(f"[V1.x HEADLINE_MODE] WM_FREE_NATS=1.5 XD_DROPOUT_RATE=0.85")
"""


def patch_file(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return f"SKIP missing: {rel}"
    text = p.read_text(encoding="utf-8")

    if "V1_HEADLINE_MODE" in text or "HEADLINE_MODE" in text:
        return f"already-patched: {rel}"

    # Append the patch block at end of file
    new_text = text.rstrip() + "\n" + PATCH_BLOCK + "\n"
    p.write_text(new_text, encoding="utf-8")
    return f"patched: {rel}"


def main() -> int:
    n_patched = 0
    n_skipped = 0
    for rel in TARGETS:
        result = patch_file(rel)
        print(f"  {result}")
        if result.startswith("patched"):
            n_patched += 1
        else:
            n_skipped += 1
    print(f"\npatched: {n_patched}  skipped: {n_skipped}  total: {len(TARGETS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
