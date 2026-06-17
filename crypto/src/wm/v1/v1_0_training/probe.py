"""200-step empirical probe for V1.0 — per CLAUDE.md §12.

Run with:  python src/wm/v1/v1_0_training/probe.py
"""
import sys
from pathlib import Path

# Path setup mirrors train_world_model.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))  # src/

from settings import *  # noqa: F403,E402
from world_model import TransformerWorldModel  # noqa: E402

_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from probe_runner import run_probe  # noqa: E402


if __name__ == "__main__":
    result = run_probe(
        model_factory=lambda: TransformerWorldModel(input_dim=INPUT_DIM),
        data_dir=DATA_DIR,
        feature_list=FEATURE_LIST,
        asset_to_idx=ASSET_TO_IDX,
        reward_horizons=REWARD_HORIZONS,
        seq_len=WM_SEQ_LEN,
        batch_size=WM_BATCH_SIZE,
        device=DEVICE,
        n_steps=200,
        label="V1.0 f13",
        lr=WM_LR,
        grad_clip=WM_GRAD_CLIP,
        mask_ratio=WM_MASK_RATIO_START,
    )
    sys.exit(0 if result["pass"] else 1)
