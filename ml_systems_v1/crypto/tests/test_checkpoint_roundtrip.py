"""
Checkpoint Round-Trip Test — verifies save/load for V11-V14.

Uses dummy nn.Linear models (save/load don't depend on architecture).
Tests BOTH checkpoint formats: ema_only and full state.
"""
import sys
import tempfile
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "wm" / "v1" / "v1_0_training"))

import torch
import torch.nn as nn
import numpy as np
import importlib.util


def _load_mod(name, filepath):
    for m in list(sys.modules.keys()):
        if m in (name, "settings", "world_model", "components", "train_world_model"):
            del sys.modules[m]
    spec = importlib.util.spec_from_file_location(name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.input_dim = 13


def test_version(version: str, has_disc: bool):
    ver_dir = PROJECT_ROOT / "src" / version / f"{version}_training"
    # Load settings first
    settings = _load_mod("settings", ver_dir / "settings.py")
    if hasattr(settings, "get_feature_config"):
        fl, dim = settings.get_feature_config(13)
        settings.FEATURE_LIST = fl
        settings.INPUT_DIM = dim
    train = _load_mod("train_world_model", ver_dir / "train_world_model.py")

    model = DummyModel()
    ema = DummyModel()
    # Make EMA distinct
    with torch.no_grad():
        ema.fc.weight.fill_(2.0)
        ema.fc.bias.fill_(0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    disc_opt = torch.optim.Adam(model.parameters(), lr=1e-4) if has_disc else None

    errors = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # ── Test 1: ema_only (best_ema.pt) ──
        ema_path = td / f"{version}_f13_wm_best_ema.pt"
        try:
            if has_disc:
                train.save_checkpoint(model, ema, opt, disc_opt,
                                       1, 0.5, 0, "PENDING", 0.0, 0,
                                       ema_path, ema_only=True)
            else:
                train.save_checkpoint(model, ema, opt, 1, 0.5,
                                       0, "PENDING", 0.0, 0, 13,
                                       ema_path, ema_only=True)
        except Exception as e:
            errors.append(f"ema_only save: {e}")

        if ema_path.exists():
            ckpt = torch.load(ema_path, weights_only=False)
            if "model_state_dict" not in ckpt:
                errors.append("ema_only: missing model_state_dict")
            if "ema_state_dict" in ckpt:
                errors.append("ema_only: unexpected ema_state_dict key")
            # Verify saved weights are EMA (value=2.0), not model
            w = ckpt["model_state_dict"]["fc.weight"]
            if not torch.allclose(w, torch.full_like(w, 2.0)):
                errors.append("ema_only: saved weights are NOT EMA (expected 2.0)")

            # Load back
            m2 = DummyModel(); e2 = DummyModel()
            o2 = torch.optim.Adam(m2.parameters())
            try:
                if has_disc:
                    d2 = torch.optim.Adam(m2.parameters())
                    train.load_latest(m2, e2, o2, d2, td, 13)
                else:
                    train.load_latest(m2, e2, o2, td, 13)
                # After load: both m2 and e2 should have EMA weights (2.0)
                if not torch.allclose(m2.fc.weight.data, torch.full((10, 10), 2.0)):
                    errors.append("ema_only load: model weights wrong")
                if not torch.allclose(e2.fc.weight.data, torch.full((10, 10), 2.0)):
                    errors.append("ema_only load: ema weights wrong")
            except Exception as e:
                errors.append(f"ema_only load: {e}")
                traceback.print_exc()

            # wm_ensemble compat
            m3 = DummyModel()
            try:
                m3.load_state_dict(ckpt["model_state_dict"], strict=False)
            except Exception as e:
                errors.append(f"ensemble compat: {e}")

        # ── Test 2: full state (latest.pt) ──
        latest_path = td / f"{version}_f13_wm_latest.pt"
        try:
            if has_disc:
                train.save_checkpoint(model, ema, opt, disc_opt,
                                       5, 0.3, 2, "PENDING", 0.02, 1,
                                       latest_path, ema_only=False)
            else:
                train.save_checkpoint(model, ema, opt, 5, 0.3,
                                       2, "PENDING", 0.02, 1, 13,
                                       latest_path, ema_only=False)
        except Exception as e:
            errors.append(f"full save: {e}")

        if latest_path.exists():
            ckpt2 = torch.load(latest_path, weights_only=False)
            for key in ["model_state_dict", "ema_state_dict", "epoch", "val_loss"]:
                if key not in ckpt2:
                    errors.append(f"full save: missing key '{key}'")
            if ckpt2.get("epoch") != 5:
                errors.append(f"full save: epoch={ckpt2.get('epoch')}, expected 5")

            # Load prefers latest.pt over best_ema.pt
            m4 = DummyModel(); e4 = DummyModel()
            o4 = torch.optim.Adam(m4.parameters())
            try:
                if has_disc:
                    d4 = torch.optim.Adam(m4.parameters())
                    result = train.load_latest(m4, e4, o4, d4, td, 13)
                else:
                    result = train.load_latest(m4, e4, o4, td, 13)
                if result[0] != 5:
                    errors.append(f"load_latest: epoch={result[0]}, expected 5 (should prefer latest.pt)")
                # model should have model weights, ema should have ema weights
                if not torch.allclose(e4.fc.weight.data, torch.full((10, 10), 2.0)):
                    errors.append("full load: ema weights wrong (expected 2.0)")
            except Exception as e:
                errors.append(f"full load: {e}")
                traceback.print_exc()

    if errors:
        print(f"  {version}: FAIL ({len(errors)} errors)")
        for e in errors:
            print(f"    - {e}")
        return False
    print(f"  {version}: PASS")
    return True


def main():
    print("Checkpoint Round-Trip Test")
    print("=" * 60)
    results = {}
    for ver, disc in [("v11", True), ("v12", False), ("v13", False), ("v14", False)]:
        try:
            results[ver] = test_version(ver, disc)
        except Exception as e:
            print(f"  {ver}: ERROR ({e})")
            traceback.print_exc()
            results[ver] = False

    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    print(f"Results: {passed}/{len(results)} passed")
    if passed < len(results):
        print(f"Failed: {[k for k, v in results.items() if not v]}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
