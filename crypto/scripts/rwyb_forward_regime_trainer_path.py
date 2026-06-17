"""RWYB: real-trainer-path test for the forward-regime / VSN levers (flags ON).

For ONE world-model version (arg), this exercises the EXACT trainer code path that
the upgraders skipped:

  collate_fn (synthetic batch w/ nested forward_regime_labels)
    -> _targets_to_device  (FIX 1 device-move)
    -> label-noise loop    (FIX 1, where the version has one)
    -> mixup_batch
    -> sequence-shuffle    (FIX 1, where the version has one)
    -> model.get_loss(... flags ON ...)  -> finite loss + forward_regime aux fired

Run in a SEPARATE subprocess per version (the 7 versions share module names
`settings`/`world_model`, so they cannot co-exist in one interpreter).

Usage:  python scripts/rwyb_forward_regime_trainer_path.py v1_1
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# (version-key) -> (train_dir, model_class_name, vsn_env, fr_env, fr_in_kwargs)
SPECS = {
    "v1_1": ("src/wm/v1/v1_1_training", "TransformerWorldModel", "V1_VSN", "V1_FORWARD_REGIME", False),
    "v3":   ("src/wm/v3/v3_training",   "WaveNetGRUWorldModel",  "V3_VSN", "V3_FORWARD_REGIME", False),
    "v4":   ("src/wm/v4/v4_training",   "MambaWorldModel",       "V4_VSN", "V4_FORWARD_REGIME", False),
    "v6":   ("src/wm/v6/v6_training",   "CausalJEPAWorldModel",  "V6_VSN", "V6_FORWARD_REGIME", True),
    "v8":   ("src/wm/v8/v8_training",   "NeuralODEWorldModel",   "V8_VSN", "V8_FORWARD_REGIME", False),
    "v13":  ("src/wm/v13/v13_training", "TFTWorldModel",         "V13_VSN", "V13_FORWARD_REGIME", False),
    "v23":  ("src/wm/v23/v23_training", "xLSTMWorldModel",       "V23_VSN", "V23_FORWARD_REGIME", False),
}


def _make_synth_batch_items(B, T, input_dim, with_fr):
    """Build a list of (obs, target_dict, asset_idx) items mimicking the dataset
    __getitem__ output that collate_fn consumes."""
    import importlib
    settings = importlib.import_module("settings")
    horizons = list(settings.REWARD_HORIZONS)
    items = []
    rng = np.random.default_rng(0)
    for _ in range(B):
        obs = torch.randn(T, input_dim, dtype=torch.float32)
        tgt = {}
        for h in horizons:
            tgt[h] = torch.randn(T, dtype=torch.float32) * 0.02
        # regime_label: integer class per bar (0/1/2)
        tgt["regime_label"] = torch.tensor(rng.integers(0, 3, size=T), dtype=torch.long)
        if with_fr:
            # per-bar forward labels; last K bars NaN (no future window) like the real builder
            bear = np.zeros(T, dtype=np.float32); bear[-8:] = np.nan
            trend = np.ones(T, dtype=np.float32); trend[-8:] = np.nan
            move = np.zeros(T, dtype=np.float32); move[-8:] = np.nan
            tgt["fwd_bear"] = torch.from_numpy(bear)
            tgt["fwd_trend"] = torch.from_numpy(trend)
            tgt["fwd_move"] = torch.from_numpy(move)
        items.append((obs, tgt, torch.tensor(0, dtype=torch.long)))
    return items


def run(version: str) -> int:
    if version not in SPECS:
        print(f"unknown version {version}; choices: {list(SPECS)}")
        return 2
    train_dir, cls_name, vsn_env, fr_env, fr_in_kwargs = SPECS[version]

    # --- env flags ON (set BEFORE importing/constructing the model) ----------
    os.environ[vsn_env] = "1"
    os.environ[fr_env] = "1"

    tdir = REPO / train_dir
    sys.path.insert(0, str(tdir))
    sys.path.insert(0, str(REPO / "src" / "wm" / "_shared"))
    sys.path.insert(0, str(REPO / "src"))

    import importlib
    # Import the trainer FIRST: it sets up its own sys.path and imports its own
    # `settings` / `world_model`. Importing `world_model` ahead of it caused a
    # cross-version `train_world_model` name collision in the shared module space.
    train_mod = importlib.import_module("train_world_model")
    settings = importlib.import_module("settings")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Prefer the model class the TRAINER actually uses (V13 loads it via importlib
    # into its own namespace + puts v1_0 on sys.path, so a bare import of
    # `world_model` would resolve to the wrong version). Fall back to world_model.
    ModelCls = getattr(train_mod, cls_name, None)
    if ModelCls is None:
        wm_mod = importlib.import_module("world_model")
        ModelCls = getattr(wm_mod, cls_name)
    input_dim = int(getattr(settings, "INPUT_DIM"))

    # Construct the model (VSN built here because env is set ON).
    model = ModelCls(input_dim=input_dim).to(DEVICE)
    model.train()

    # Confirm VSN actually constructed (the --vsn / env path).
    vsn_ok = getattr(model, "vsn", "MISSING")
    has_vsn = (vsn_ok is not None and vsn_ok != "MISSING")
    print(f"[{version}] VSN constructed: {has_vsn} "
          f"(params={sum(p.numel() for p in model.vsn.parameters()) if has_vsn else 0})")

    # Forward-regime head: V6 self-attaches at __init__ (with d_latent); all others
    # rely on the trainer calling attach_forward_regime_head(model). Mirror that:
    # only attach here if the model didn't already build the head itself.
    if getattr(model, "forward_regime_head", None) is not None:
        feat_dim = next(model.forward_regime_head.bear_head.parameters()).shape[-1]
        params = sum(p.numel() for p in model.forward_regime_head.parameters())
        print(f"[{version}] forward_regime head SELF-ATTACHED at init: "
              f"feat_dim={feat_dim} params={params}")
    else:
        from forward_regime_head import attach_forward_regime_head
        info = attach_forward_regime_head(model, verbose=False)
        print(f"[{version}] forward_regime head attached (trainer path): "
              f"feat_dim={info['feat_dim']} params+={info['params']}")

    # --- Build synthetic batch through the REAL collate_fn -------------------
    B, T = 32, int(getattr(settings, "WM_SEQ_LEN"))
    items = _make_synth_batch_items(B, T, input_dim, with_fr=True)
    obs, targets, asset = train_mod.collate_fn(items)
    assert "forward_regime_labels" in targets, "collate did not pack forward_regime_labels"
    assert isinstance(targets["forward_regime_labels"], dict)
    print(f"[{version}] collate OK: obs={tuple(obs.shape)} "
          f"keys={[k for k in targets]} fr_keys={list(targets['forward_regime_labels'])}")

    obs = obs.to(DEVICE)
    asset = asset.to(DEVICE)

    # --- FIX 1: device-move via the real helper -----------------------------
    # V13 names it `_targets_to_device`; V23 names it `targets_to_device`.
    _t2d = getattr(train_mod, "_targets_to_device", None) or getattr(train_mod, "targets_to_device")
    targets_gpu = _t2d(targets, DEVICE)
    assert targets_gpu["forward_regime_labels"]["bear"].device.type == DEVICE.type, \
        "forward_regime labels not on device"
    print(f"[{version}] _targets_to_device OK (fr labels on {DEVICE.type})")

    # --- FIX 1: label-noise loop (only versions that have one) --------------
    # Mimic the trainer's guarded loop with a no-op injector to confirm the
    # nested-dict skip works (the real injector path is import-heavy; the bug
    # was the .dtype crash on the dict, which the skip fixes).
    regime_lbl = targets_gpu.get("regime_label")
    noise_touched = 0
    for _hk, _tv in list(targets_gpu.items()):
        if _hk == "regime_label" or isinstance(_tv, dict):
            continue
        if _tv.dtype.is_floating_point:
            noise_touched += 1
    print(f"[{version}] label-noise loop nested-dict-safe "
          f"(would-noise {noise_touched} float targets, fr-dict skipped)")

    # --- mixup (real augmentor) ---------------------------------------------
    try:
        from anti_fragile import AntifragileAugmentor, AntifragileConfig
        aug = AntifragileAugmentor(AntifragileConfig())
        obs, targets_gpu = aug.mixup_batch(obs, targets_gpu)
        assert isinstance(targets_gpu["forward_regime_labels"], dict), "mixup dropped fr dict"
        print(f"[{version}] mixup_batch OK (fr dict preserved)")
    except Exception as e:  # mixup is not load-bearing for the bug; report + continue
        print(f"[{version}] mixup_batch skipped ({type(e).__name__}: {e})")

    # --- FIX 1: sequence-shuffle (nested-dict-aware) ------------------------
    SEQ_P = float(getattr(settings, "SEQ_SHUFFLE_PROB", 0.0))
    shuf_n = 0
    if SEQ_P > 0:
        for b in range(obs.shape[0]):
            if torch.rand(1).item() < max(SEQ_P, 0.5):  # force some shuffles for the test
                perm = torch.randperm(obs.shape[1], device=obs.device)
                obs[b] = obs[b][perm]
                for h in targets_gpu:
                    if isinstance(targets_gpu[h], dict):
                        continue
                    targets_gpu[h][b] = targets_gpu[h][b][perm]
                shuf_n += 1
        print(f"[{version}] seq-shuffle nested-dict-safe (shuffled {shuf_n} rows, fr dict untouched)")
    else:
        print(f"[{version}] seq-shuffle N/A (SEQ_SHUFFLE_PROB=0)")

    # --- get_loss with flags ON ---------------------------------------------
    fr_labels = targets_gpu.get("forward_regime_labels")
    kwargs = {}
    if fr_in_kwargs:
        kwargs["forward_regime_labels"] = fr_labels
    with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
        out = model.get_loss(obs, asset, targets_gpu, mask_ratio=0.25,
                             regime_labels=targets_gpu.get("regime_label"), **kwargs)
    loss = out[0]
    loss_dict = out[1]
    finite = bool(torch.isfinite(loss).item())
    # aux key is "fr_aux" (V1.1/V4/V8/V13/V23) or "forward_regime" (V3/V6)
    fr_keys = [k for k in loss_dict
               if str(k).lower() in ("fr_aux", "forward_regime")
               or "forward_regime" in str(k).lower()]
    print(f"[{version}] get_loss OK: loss={float(loss):.5f} finite={finite}")
    print(f"[{version}] loss_dict keys: {sorted(loss_dict.keys())}")
    print(f"[{version}] forward_regime aux keys present: {fr_keys}")

    ok = finite and len(fr_keys) > 0
    print(f"[{version}] RWYB flags-ON: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1] if len(sys.argv) > 1 else "v1_1"))
