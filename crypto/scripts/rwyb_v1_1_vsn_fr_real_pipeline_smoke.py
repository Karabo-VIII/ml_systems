"""RWYB: V1.1 real-data-pipeline smoke for the user's vsn_fr command.

Runs the ACTUAL V1.1 trainer (real chimera dollar-bar data -> AntifragileDataset ->
collate_fn -> _targets_to_device -> label-noise -> get_loss) with the user's flags:

    V1_VSN=1 V1_FORWARD_REGIME=1

for a FEW steps (WM_STEPS_PER_EPOCH + WM_TOTAL_EPOCHS monkeypatched tiny), then stops.
Goal: confirm it reaches a stepping batch without crashing on the real pipeline.

Usage:  V1_VSN=1 V1_FORWARD_REGIME=1 python scripts/rwyb_v1_1_vsn_fr_real_pipeline_smoke.py
"""
import os
import sys
from pathlib import Path

os.environ["V1_VSN"] = "1"
os.environ["V1_FORWARD_REGIME"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = Path(__file__).resolve().parents[1]
TDIR = REPO / "src" / "wm" / "v1" / "v1_1_training"
sys.path.insert(0, str(TDIR))
sys.path.insert(0, str(REPO / "src" / "wm" / "_shared"))
sys.path.insert(0, str(REPO / "src"))

import importlib

twm = importlib.import_module("train_world_model")
settings = importlib.import_module("settings")

# --- cap the run to a few real steps, then let it finish + stop -------------
twm.WM_STEPS_PER_EPOCH = 5
twm.WM_TOTAL_EPOCHS = 1
# also patch the names the loop reads from settings.* (train_world_model does
# `from settings import *`, so the module-level globals above are what the loop uses)
print("[smoke] V1.1 vsn_fr REAL-pipeline smoke: "
      f"V1_VSN={os.environ['V1_VSN']} V1_FORWARD_REGIME={os.environ['V1_FORWARD_REGIME']} "
      f"steps/epoch={twm.WM_STEPS_PER_EPOCH} epochs={twm.WM_TOTAL_EPOCHS}")

try:
    # VSN is env-driven (V1_VSN=1 set above); no use_vsn kwarg in train_world_model.
    ok = twm.train_world_model(use_revin=False, n_features=41)
    print(f"[smoke] train_world_model returned: {ok}")
    print("[smoke] V1.1 vsn_fr REAL-pipeline smoke: PASS (reached stepping batch, no crash)")
    sys.exit(0)
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[smoke] V1.1 vsn_fr REAL-pipeline smoke: FAIL ({type(e).__name__}: {e})")
    sys.exit(1)
