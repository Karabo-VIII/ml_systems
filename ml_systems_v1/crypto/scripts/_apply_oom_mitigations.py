"""Apply 2026-04-30 OOM mitigations to all WM trainer files.

V1.0 OOM'd at ep 99 due to memory fragmentation (9-epoch gap between
empty_cache calls insufficient at long runs). Three patches per trainer:
  1. PYTORCH_CUDA_ALLOC_CONF env var at module top (max_split_size_mb=128)
  2. empty_cache cadence 10 -> 5 epochs
  3. empty_cache after ShIC computation (memory peak)

Idempotent: skips files that already have the patch.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TARGETS = [
    "src/wm/v1/v1_1_training/train_world_model.py",
    "src/wm/v1/v1_4_training/train_world_model.py",
    "src/wm/v1/v1_6_training/train_world_model.py",
    "src/wm/v3/v3_training/train_world_model.py",
    "src/wm/v4/v4_training/train_world_model.py",
    "src/wm/v6/v6_training/train_world_model.py",
    "src/wm/v8/v8_training/train_world_model.py",
]

ALLOC_HINT = (
    'import os\n'
    '# OOM mitigation (2026-04-30): cap CUDA caching allocator splits at 128MB\n'
    '# to prevent long-run fragmentation (V1.0 ep99 OOM).\n'
    'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")\n\n'
)


def patch_file(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return f"SKIP missing: {rel}"
    text = p.read_text(encoding="utf-8")

    # Idempotency check
    if "PYTORCH_CUDA_ALLOC_CONF" in text:
        return f"already-patched: {rel}"

    # 1. Insert ALLOC hint before first `import torch`
    if "import torch" not in text:
        return f"NO-MATCH (no torch import): {rel}"
    idx = text.index("import torch")
    new_text = text[:idx] + ALLOC_HINT + text[idx:]

    # 2. Replace empty_cache cadence 10 -> 5
    # Match: `if (epoch + 1) % 10 == 0:` followed by gc.collect / empty_cache
    if "if (epoch + 1) % 10 == 0:" in new_text:
        new_text = new_text.replace(
            "if (epoch + 1) % 10 == 0:",
            "if (epoch + 1) % 5 == 0:  # OOM fix 2026-04-30 (was 10; ep99 V1.0 OOM)",
            1,
        )

    # 3. Add empty_cache post-ShIC if not present
    # Match: `compute_shuffled_ic(...)` followed by record/etc; insert after the record call
    # Heuristic: insert after `ic_tracker.record(epoch, contiguous_ic, shuffled_ic)` if found
    needle = "ic_tracker.record(epoch, contiguous_ic, shuffled_ic)"
    insert_after = (
        "ic_tracker.record(epoch, contiguous_ic, shuffled_ic)\n"
        "                # OOM mitigation (2026-04-30): ShIC compute is the memory peak\n"
        "                if torch.cuda.is_available():\n"
        "                    torch.cuda.empty_cache()"
    )
    if needle in new_text and "ShIC compute is the memory peak" not in new_text:
        new_text = new_text.replace(needle, insert_after, 1)

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
