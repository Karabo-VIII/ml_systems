"""DistilledStudent — 5-10M-param compact model that gets distilled from
the V1.x ensemble + foundation backbone.

Two student configs registered:
  small  -- 5M params (d_model=384, d_state=32, 4 Mamba layers, 1 xattn)
  med    -- 10M params (d_model=512, d_state=32, 6 Mamba layers, 1 xattn)

Same forward signature as FoundationBackbone so the distillation training
loop is architecture-agnostic. Reuses V4 Mamba primitives.

Inference budget (target):
    small  ~3 ms / window  on 4060
    med    ~5 ms / window
vs ensemble-of-9 (~25-30 ms / window). Meets the 1/4-latency requirement
in LITERATURE.md Hole 7.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Reuse the foundation backbone shape; swap config for smaller dims.
from frontier_ml.foundation.backbone import FoundationBackbone  # noqa: E402

STUDENT_CONFIGS = {
    "small": dict(
        d_model=384,
        d_state=32,
        n_layers_backbone=4,
        n_layers_xattn=1,
        n_heads_xattn=4,
        d_xattn=128,
        n_assets_max=50,
        horizons=(1, 4, 16, 64),
        num_bins=255,
        d_contrastive=64,
        dropout=0.1,
        expand=2,
        headdim=64,
        chunk_size=16,
    ),
    "med": dict(
        d_model=512,
        d_state=32,
        n_layers_backbone=6,
        n_layers_xattn=1,
        n_heads_xattn=4,
        d_xattn=128,
        n_assets_max=50,
        horizons=(1, 4, 16, 64),
        num_bins=255,
        d_contrastive=64,
        dropout=0.1,
        expand=2,
        headdim=64,
        chunk_size=16,
    ),
}


def make_student(size: str = "small", n_features: int = 34) -> FoundationBackbone:
    """Construct a distilled student. `size` in {'small', 'med'}."""
    if size not in STUDENT_CONFIGS:
        raise ValueError(f"unknown student size {size!r}; expected one of {list(STUDENT_CONFIGS)}")
    cfg = STUDENT_CONFIGS[size]
    student = FoundationBackbone(n_features=n_features, config=cfg)
    return student


def smoke():
    """Construct both student sizes and print param counts."""
    for size in ("small", "med"):
        s = make_student(size=size, n_features=34)
        print(f"[student] {size:5s}  params={s.num_params():,} ({s.num_params()/1e6:.1f}M)")
    return True


if __name__ == "__main__":
    smoke()
