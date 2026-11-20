"""Stress-test every model version: instantiate + B=32 forward pass.

Reports per-version: import OK?, instantiate OK?, forward OK?, n_params,
output shape, all-finite check. Writes a single status line per version.

This is the post-2026-04-27 sweep to catch any regression introduced by
the feature_sets centralization or invariant fixes.
"""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]

# (version_name, training_dir, model_class_name_substring_OR_explicit)
# (version_name, training_dir, (module_name, class_name[, init_kwargs[, input_dim_override]]) or None)
VERSIONS = [
    ("V1.0", "src/wm/v1/v1_0_training", ("world_model", "TransformerWorldModel")),
    ("V1.1", "src/wm/v1/v1_1_training", ("world_model", "TransformerWorldModel")),
    ("V1.4", "src/wm/v1/v1_4_training", ("world_model", "TransformerWorldModel")),
    ("V1.6", "src/wm/v1/v1_6_training", ("world_model", "TransformerWorldModel")),
    ("V2",   "backups/BKP_20260429_MODEL_HARMONIZATION/v2/v2_training",   ("world_model", "JEPAWorldModel")),
    ("V3",   "src/wm/v3/v3_training",   ("world_model", "WaveNetGRUWorldModel")),
    ("V4",   "src/wm/v4/v4_training",   ("world_model", "MambaWorldModel")),
    ("V5",   "backups/BKP_20260429_MODEL_HARMONIZATION/v5/v5_training",   ("world_model", "HybridMambaAttentionWorldModel")),
    ("V6",   "src/wm/v6/v6_training",   ("world_model", "CausalJEPAWorldModel")),
    ("V7",   "backups/BKP_20260429_MODEL_HARMONIZATION/v7/v7_training",   ("world_model", "ViTWorldModel")),
    ("V8",   "src/wm/v8/v8_training",   ("world_model", "NeuralODEWorldModel")),
    ("V9",   "src/wm/v9/v9_training",   ("world_model", "MoEWorldModel")),
    ("V11",  "src/wm/v11/v11_training", ("world_model", "MicrostructureWorldModel")),
    ("V12",  "src/wm/v12/v12_training", ("world_model", "CrossAssetWorldModel")),
    ("V15",  "src/wm/v15",              ("patchtst_encoder", "PatchTSTEncoder", {"n_features": 41, "seq_len": 96})),
    # V16/V17/V18 default to f121 (V19/SOTA-aligned). Override input_dim accordingly.
    # V16/V17 reclassified 2026-06-11 to A1 backbones (src/agents/...).
    ("V16",  "src/agents/a1_wm_consuming/backbones/v16_dreamerv3/v16_training", ("dreamer_v3", "DreamerV3WorldModel", {}, 121)),
    ("V17",  "src/agents/a1_wm_consuming/backbones/v17_tdmpc2/v17_training", ("td_mpc2", "TDMPC2WorldModel", {}, 121)),
    # V18 ChronosFineTuneHead takes pooled[B, 768] not a sequence; we test it
    # via the (pooled,) signature path.
    ("V18",  "src/wm/v18/v18_training", ("finetune_chronos", "ChronosFineTuneHead", {"d_model": 768}, 768)),
]


def find_model_class(wm_module):
    """Pick the most likely top-level model class."""
    candidates = []
    for cname in dir(wm_module):
        obj = getattr(wm_module, cname)
        if not isinstance(obj, type):
            continue
        if not issubclass(obj, nn.Module):
            continue
        if obj is nn.Module:
            continue
        # Ignore obvious sub-components by name
        excl = ("Block", "Layer", "Norm", "Head", "Attention", "MLP", "Encoder",
                "Decoder", "Conv", "Solver", "Dynamics", "Bucketer", "Embedding",
                "Discriminator", "TwoHot", "FFN", "Cell", "GRU", "LSTM", "Mixer")
        if any(e in cname for e in excl):
            continue
        # Prefer names containing World/Model
        score = 0
        if "World" in cname:
            score += 10
        if "Model" in cname:
            score += 5
        if cname.endswith("Model"):
            score += 3
        if cname.startswith("V"):
            score += 2
        candidates.append((score, cname, obj))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0]


def stress_one(name: str, dir_rel: str, hint: tuple | None) -> dict:
    out = {"name": name, "dir": dir_rel, "import": "FAIL",
           "instantiate": "FAIL", "forward": "FAIL", "params": 0,
           "err": "", "model_class": None}
    full_dir = ROOT / dir_rel
    if not full_dir.exists():
        out["err"] = f"dir missing: {dir_rel}"
        return out

    sys.path.insert(0, str(full_dir))
    sys.path.insert(0, str(ROOT / "src"))

    wm_mod = None
    init_kwargs = {}
    input_dim_override = None
    if hint:
        mod_name = hint[0]
        if len(hint) >= 3 and isinstance(hint[2], dict):
            init_kwargs = hint[2]
        if len(hint) >= 4:
            input_dim_override = hint[3]
        try:
            wm_mod = importlib.import_module(mod_name)
            out["import"] = f"OK ({mod_name})"
        except Exception as e:
            out["err"] = f"import {mod_name}: {type(e).__name__}: {str(e)[:80]}"
    else:
        for mod_name in ["world_model", "model"]:
            try:
                wm_mod = importlib.import_module(mod_name)
                out["import"] = f"OK ({mod_name})"
                break
            except Exception:
                continue
    if wm_mod is None:
        _cleanup(full_dir)
        return out

    # Resolve the class
    if hint and hint[1]:
        klass = getattr(wm_mod, hint[1], None)
        if klass is None:
            out["err"] = f"class {hint[1]!r} not in {hint[0]}"
            _cleanup(full_dir)
            return out
        cname = hint[1]
    else:
        mc = find_model_class(wm_mod)
        if mc is None:
            out["err"] = "no model class in module"
            _cleanup(full_dir)
            return out
        score, cname, klass = mc
    out["model_class"] = cname

    # Instantiate
    try:
        m = klass(**init_kwargs)
        out["instantiate"] = "OK"
        out["params"] = sum(p.numel() for p in m.parameters())
    except Exception as e:
        out["err"] = f"instantiate: {type(e).__name__}: {str(e)[:120]}"
        _cleanup(full_dir)
        return out

    # Forward pass at B=32
    try:
        try:
            st = importlib.import_module("settings")
            input_dim = getattr(st, "INPUT_DIM", 41)
            num_assets = getattr(st, "NUM_ASSETS", 10)
            seq_len = getattr(st, "SEQ_LEN", 96)
        except ImportError:
            input_dim = 41
            num_assets = 10
            seq_len = 96
        if input_dim_override is not None:
            input_dim = input_dim_override

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = m.to(device)
        m.eval()

        B, T = 32, seq_len
        torch.manual_seed(42)
        obs = torch.randn(B, T, input_dim, device=device)
        asset_id = torch.randint(0, num_assets, (B,), device=device)

        # Optional action + reward sequence (for V16 DreamerV3 / V17 TDMPC2)
        action_dim = getattr(st, "ACTION_DIM", 1)
        actions = torch.randn(B, T, action_dim, device=device)
        rewards = torch.randn(B, T, device=device)

        # Try forward signatures, longest first
        success = False
        # V15 PatchTSTEncoder takes [B, C, L] (channel-independent), so transpose obs.
        # V18 ChronosFineTuneHead takes [B, D_pooled] not a sequence.
        obs_chan_first = obs.transpose(1, 2)  # [B, C=input_dim, L=T]
        pooled = torch.randn(B, 768, device=device)  # ChronosFineTuneHead default d_model
        signatures = [
            ("forward_train", (obs, actions, rewards, asset_id)),  # V17 TDMPC2
            ("forward_train", (obs, actions, asset_id, rewards)),  # V16 DreamerV3 (returns last)
            ("forward_train", (obs, actions, asset_id)),           # action-cond WM
            ("forward_train", (obs, asset_id)),                    # standard WM
            ("forward",       (obs, asset_id)),
            ("forward",       (obs_chan_first,)),                  # V15 encoder [B,C,L]
            ("forward",       (obs,)),                              # generic [B,T,F]
            ("forward",       (pooled,)),                           # V18 head [B,D]
        ]
        for fm, args in signatures:
            if not hasattr(m, fm):
                continue
            method = getattr(m, fm)
            try:
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            result = method(*args)
                    else:
                        result = method(*args)
                if result is None:
                    out["err"] = f"{fm}{tuple(type(a).__name__ for a in args)} returned None"
                    continue
                all_finite = True
                if isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                            all_finite = False
                            break
                        if isinstance(v, dict):
                            for h, t in v.items():
                                if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                                    all_finite = False
                                    break
                elif isinstance(result, torch.Tensor):
                    all_finite = torch.isfinite(result).all().item()
                out["forward"] = f"OK ({fm}/{len(args)}, finite={all_finite})"
                success = True
                break
            except (TypeError, NotImplementedError) as e:
                # signature mismatch — try next
                out["err"] = f"{fm}/{len(args)}: {type(e).__name__}: {str(e)[:80]}"
                continue
            except Exception as e:
                tb_short = traceback.format_exc().splitlines()[-1][:120]
                out["err"] = f"{fm}: {tb_short}"
                continue
        if not success and out["forward"] == "FAIL":
            pass  # err already set
    except Exception as e:
        out["err"] = f"setup: {type(e).__name__}: {str(e)[:120]}"

    _cleanup(full_dir)
    return out


def _cleanup(full_dir):
    sys.path[:] = [p for p in sys.path if p != str(full_dir) and p != str(ROOT / "src")]
    for k in list(sys.modules.keys()):
        if k in ("settings", "world_model", "components", "train_world_model",
                 "validate_world", "model", "feature_sets"):
            del sys.modules[k]


def main():
    print(f"{'Ver':5s} {'class':28s} {'params':>8s}  {'import':30s} {'inst':6s} {'forward':30s}  err")
    print("-" * 130)
    for name, dir_, hint in VERSIONS:
        r = stress_one(name, dir_, hint)
        params_s = f"{r['params']/1e6:.1f}M" if r["params"] > 0 else "-"
        print(f"{r['name']:5s} {str(r['model_class'] or '-'):28s} {params_s:>8s}  "
              f"{r['import']:30s} {r['instantiate']:6s} {r['forward']:30s}  {r['err'][:40]}")


if __name__ == "__main__":
    main()
