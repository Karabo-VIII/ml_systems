#!/usr/bin/env python
"""V23 Training -- xLSTM (Beck et al., NeurIPS 2024).

Stacked alternating sLSTM/mLSTM blocks with a VIB bottleneck on h_seq. Recurrent
SOTA alternative to the GRU-JEPA family; linear-in-T compute. Adapted from the
V22 trainer (this file was an un-adapted V22 copy until 2026-06-10).

World-class levers (all flag-gated, default OFF -> base path byte-for-byte unchanged):
  - V23_VSN=1            : per-timestep causal feature gate before the obs_encoder
  - V23_FORWARD_REGIME=1 : forward bear/trend/move-onset aux head (weight 0.10)
  - RECON_WEIGHT>0       : masked recon anchor off the VIB latent (settings.py)

Usage:
    python src/wm/v23/v23_training/train_world_model.py --features 29
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

# Local world_model import (avoids package vs script ambiguity)
import importlib.util
_wm_spec = importlib.util.spec_from_file_location(
    "v23_world_model", str(Path(__file__).resolve().parent / "world_model.py")
)
_wm_mod = importlib.util.module_from_spec(_wm_spec)
_wm_spec.loader.exec_module(_wm_mod)
xLSTMWorldModel = _wm_mod.xLSTMWorldModel
count_parameters = _wm_mod.count_parameters

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anti_fragile import (  # noqa: E402
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, AntifragileDataset,
    compute_regime_weights, make_predict_fn,
)
from data_api import load_full_data_for_training as load_full_data  # noqa: E402

# Round-9: shared meta-learner runtime (opt-in via --meta; no-op when empty)
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from meta_runtime import MetaRuntime, add_meta_args  # noqa: E402


def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def collate_fn(batch):
    obs_list, tgt_list, asset_list = [], [], []
    for obs, tgt, asset_idx in batch:
        obs_list.append(obs)
        tgt_list.append(tgt)
        asset_list.append(asset_idx)
    obs = torch.stack(obs_list)
    asset = torch.tensor(asset_list, dtype=torch.long)
    # Forward-regime label keys (V23_FORWARD_REGIME) are packed into a nested
    # "forward_regime_labels" dict (matching the V1.1 template) instead of being
    # stacked as flat keys. When the flag is OFF, the dataset emits no fwd_* keys
    # and this is a no-op -> targets are byte-for-byte the generic-stacked set.
    _FR_KEYS = ("fwd_bear", "fwd_trend", "fwd_move")
    targets = {}
    for key in tgt_list[0]:
        if key in _FR_KEYS:
            continue
        targets[key] = torch.stack([t[key] for t in tgt_list])
    if "fwd_bear" in tgt_list[0]:
        targets["forward_regime_labels"] = {
            "bear":  torch.stack([t["fwd_bear"]  for t in tgt_list]),
            "trend": torch.stack([t["fwd_trend"] for t in tgt_list]),
            "move":  torch.stack([t["fwd_move"]  for t in tgt_list]),
        }
    return obs, targets, asset


def targets_to_device(targets, device):
    """Move a targets dict to `device`, handling the nested forward_regime_labels dict
    (V23_FORWARD_REGIME). When the flag is OFF there is no nested dict -> identical to
    the prior flat {k: t.to(device)} move."""
    out = {}
    for k, v in targets.items():
        if isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def save_checkpoint(model, ema_model, optimizer, epoch, val_loss, patience,
                    gate, best_shic, shic_decline, n_features, path,
                    ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        torch.save({"model_state_dict": ema_model.state_dict(),
                    "n_features": n_features, "version": "v23"}, path)
    else:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "patience_counter": patience,
            "gate_status": gate, "best_shic": best_shic,
            "shic_decline_count": shic_decline, "n_features": n_features,
            "version": "v23",
        }, path)


def load_latest(model, ema_model, optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    latest_path = ckpt_dir / f"v23_f{n_features}{run_tag_str}_wm_latest.pt"
    ema_path = ckpt_dir / f"v23_f{n_features}{run_tag_str}_wm_best_ema.pt"
    ckpt_path = latest_path if latest_path.exists() else ema_path
    if not ckpt_path.exists():
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    print(f"  Loading from {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # load_latest_collision guard (per CLAUDE.md Code Change Verification #11)
    if ckpt.get("n_features", n_features) != n_features:
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        ema_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception:
        pass
    return (ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf")),
            ckpt.get("patience_counter", 0), ckpt.get("gate_status", "PENDING"),
            ckpt.get("best_shic", 0.0), ckpt.get("shic_decline_count", 0))


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    metrics = {"total": [], "direct_ret": [], "regime": [], "regime_acc": []}
    for h in REWARD_HORIZONS:
        metrics[f"ret_{h}"] = []
    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = targets_to_device(targets, DEVICE)
        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(
                obs, asset, targets_gpu, mask_ratio=0.0,
                regime_labels=targets_gpu.get("regime_label"),
            )
        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])
        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(
                outputs["return_logits"][h].reshape(-1, NUM_BINS)
            )
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    result = {k: np.mean(v) for k, v in metrics.items() if v}
    for h in REWARD_HORIZONS:
        if ic_data[h]["preds"]:
            preds = np.concatenate(ic_data[h]["preds"])
            reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(preds) & np.isfinite(reals)
            if mask.sum() > 100:
                result[f"ic_{h}"] = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
    result["ic_mean"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])
    return result


def run_training(n_features=None, resume=True, args=None):
    if n_features is None:
        n_features = INPUT_DIM
    feature_list, input_dim, _ = get_feature_config(n_features)

    run_tag_str = ""

    print("=" * 70)
    print("  V23 TRAINING -- xLSTM (Beck et al., NeurIPS 2024)")
    print(f"  Features: {input_dim} | d_model: {WM_D_MODEL} | n_layers: {WM_N_LAYERS} | Device: {DEVICE}")
    print(f"  Stacked sLSTM/mLSTM ({BLOCK_PATTERN}) + VIB bottleneck on h_seq")
    print("=" * 70)

    af_config = AntifragileConfig()
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        print("  [ERROR] No valid data. Exiting.")
        return False

    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way(all_segments)
    regime_weights = compute_regime_weights(train_seg)

    # -- Forward-regime LABELS (V23_FORWARD_REGIME lever) ----------------------
    # Pre-compute per-segment forward bear/trend/move labels and attach them onto the
    # train+val segments BEFORE the datasets are built, so AntifragileDataset.__getitem__
    # picks them up (keys fwd_bear/fwd_trend/fwd_move) and collate_fn packs them into
    # targets["forward_regime_labels"]. When OFF (default): no label keys are written ->
    # datasets emit no fwd_* -> targets are byte-for-byte the base set. Mirrors V1.1.
    _use_forward_regime = os.environ.get("V23_FORWARD_REGIME", "0") == "1"
    if _use_forward_regime:
        from regime_targets import (
            forward_bear_label as _fwd_bear_lbl,
            forward_trend_label as _fwd_trend_lbl,
            move_onset_label as _fwd_move_lbl,
        )
        print("  [V23_FORWARD_REGIME] Pre-computing forward labels on train+val segments ...")
        for _seg in (list(train_seg) + list(val_seg)):
            _ret1 = _seg.get("target_return_1")
            _n = len(_seg["features"])
            if _ret1 is None:
                _seg["fwd_bear"]  = np.full(_n, np.nan, dtype=np.float32)
                _seg["fwd_trend"] = np.full(_n, np.nan, dtype=np.float32)
                _seg["fwd_move"]  = np.full(_n, np.nan, dtype=np.float32)
                continue
            # Close proxy: cumprod(1 + target_return_1). Forward labels depend only on
            # RELATIVE price ratios, so a normalized close (close[0]=1) is sufficient.
            _cret = np.clip(np.asarray(_ret1, dtype=np.float64), -0.99, 10.0)
            _close_ext = np.empty(len(_cret) + 1, dtype=np.float64)
            _close_ext[0] = 1.0
            np.cumprod(1.0 + _cret, out=_close_ext[1:])
            _close_ext = np.maximum(_close_ext, 1e-8)
            _seg["fwd_bear"]  = _fwd_bear_lbl(_close_ext[:-1],  K=64, dd_thresh=0.05)
            _seg["fwd_trend"] = _fwd_trend_lbl(_close_ext[:-1], K=64)
            _seg["fwd_move"]  = _fwd_move_lbl(_close_ext[:-1],  a=1,  b=64)
        print(f"  [V23_FORWARD_REGIME] Labels ready (K=64, bear/trend/move, NaN tail masked)")

    train_ds = AntifragileDataset(train_seg, seq_len=WM_SEQ_LEN,
                                  reward_horizons=REWARD_HORIZONS,
                                  augment=True, config=af_config,
                                  sample_weights=regime_weights)
    val_ds = AntifragileDataset(val_seg, seq_len=WM_SEQ_LEN,
                                reward_horizons=REWARD_HORIZONS,
                                augment=False, config=af_config)

    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE,
                              sampler=train_ds.get_sampler(),
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, collate_fn=collate_fn)

    model = xLSTMWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = xLSTMWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    print(f"  Parameters: {count_parameters(model):,}")

    # -- Forward-regime HEAD attach (V23_FORWARD_REGIME) -----------------------
    # Attach to BOTH model and ema_model so the EMA copy has the head params for
    # validation. Default OFF -> not attached -> guarded forward/loss blocks no-op.
    if _use_forward_regime:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "_shared"))
        from forward_regime_head import attach_forward_regime_head as _attach_frh
        _attach_frh(model)
        _attach_frh(ema_model, verbose=False)
        ema_model.load_state_dict(model.state_dict())   # re-sync after both attach
        print(f"  [V23_FORWARD_REGIME] head attached to model + ema (aux weight 0.10)")

    # -- VSN flag report (V23_VSN; model wired the gate at __init__ from the env) --
    _use_vsn = os.environ.get("V23_VSN", "0") == "1"
    if _use_vsn:
        vsn_params = sum(p.numel() for p in model.vsn.parameters()) if model.vsn is not None else 0
        print(f"  [V23_VSN] Variable Selection Network ENABLED ({vsn_params} params, "
              f"causal per-timestep sigmoid gate over [B,T,{input_dim}])")
    else:
        print(f"  [V23_VSN] VSN disabled (default). Set V23_VSN=1 to enable.")

    # AutopsyMode: opt-in internals diagnostic (--autopsy)
    autopsy = None
    if getattr(args, "autopsy", False):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from diagnostics.autopsy_mode import AutopsyMode
            autopsy = AutopsyMode(
                model,
                log_dir=Path(__file__).resolve().parents[4] / "logs" / "v23",
                run_tag=f"v23_f{input_dim}{run_tag_str}",
                sample_every=getattr(args, "autopsy_every", 50),
                loss_window=getattr(args, "autopsy_loss_window", 200),
                explosion_z_threshold=getattr(args, "autopsy_z", 3.0),
            )
            print(f"  [AUTOPSY] enabled; jsonl={autopsy.jsonl_path.name}")
        except Exception as e:
            print(f"  [AUTOPSY] init failed ({type(e).__name__}: {e}); continuing without")

    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                    betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, patience, gate, best_shic, shic_decline = (
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str)
        if resume else (0, float("inf"), 0, "PENDING", 0.0, 0)
    )

    # Round-9: meta-learner runtime (no-op when --meta empty; default)
    meta_rt = MetaRuntime.from_args(args, model, MODEL_DIR,
                                     trunk_dim=RETURN_HEAD_DIM,
                                     device=DEVICE, version="v23")

    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(start_epoch, WM_EPOCHS):
        model.train()
        train_iter = iter(train_loader)
        epoch_nan_count = 0
        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc=f"Epoch {epoch + 1:3d}", leave=False)

        for step in pbar:
            try:
                obs, targets, asset = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                obs, targets, asset = next(train_iter)

            obs = obs.to(DEVICE, non_blocking=True)
            asset = asset.to(DEVICE, non_blocking=True)
            targets_gpu = targets_to_device(targets, DEVICE)
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            kl_anneal = min((epoch + 1) / max(VIB_KL_ANNEAL_EPOCHS, 1), 1.0)
            with torch.amp.autocast("cuda"):
                loss, loss_dict, base_outputs = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                    regime_labels=targets_gpu.get("regime_label"),
                    kl_anneal=kl_anneal,
                )

            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= 100:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf in epoch {epoch + 1}")
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            update_ema(model, ema_model)
            scheduler.step()

            if autopsy is not None:
                try:
                    autopsy.step(step, loss_components={
                        "total": float(loss_dict.get("total", 0.0)),
                        "ret_1": float(loss_dict.get("ret_1", 0.0)),
                        "recon": float(loss_dict.get("recon", 0.0)),
                    })
                except Exception:
                    pass

            # Round-9: residual meta-learner step (no-op when --meta empty)
            if meta_rt.flags:
                meta_rt.train_step_residual(model, base_outputs, targets_gpu, step)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(L=f"{loss_dict['total']:.3f}",
                                 r1=f"{loss_dict.get('ret_1', 0):.3f}")

        val_metrics = validate(ema_model, val_loader)
        val_loss = val_metrics.get("total", float("inf"))
        ic_str = " ".join([f"ic{h}={val_metrics.get(f'ic_{h}', 0):.4f}" for h in ACTIVE_HORIZONS])
        print(f"  Ep {epoch + 1:3d} | val={val_loss:.3f} | {ic_str}")

        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_seg, predict_fn, horizon=1)
            print(f"    ShIC={shic:.4f} | best={best_shic:.4f}")
            if shic > best_shic:
                best_shic = shic
                shic_decline = 0
            else:
                shic_decline += 1
            if shic_decline >= SHUFFLED_IC_PATIENCE:
                print("    [STOP] ShIC patience exhausted.")
                break

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            ckpt = MODEL_DIR / f"v23_f{input_dim}{run_tag_str}_wm_best_ema.pt"
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ckpt,
                            ema_only=True)
            print(f"    [SAVE] {ckpt.name}")

        latest = MODEL_DIR / f"v23_f{input_dim}{run_tag_str}_wm_latest.pt"
        save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                        patience, gate, best_shic, shic_decline, input_dim, latest)
        if (epoch + 1) % 5 == 0:
            ep = MODEL_DIR / f"v23_f{input_dim}{run_tag_str}_wm_epoch_{epoch + 1}.pt"
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ep)

    # Round-9: save meta variant checkpoints + summary
    meta_rt.save_and_summarize(version="v23", n_features=input_dim)

    if autopsy is not None:
        try:
            try:
                vb = next(iter(val_loader))
                autopsy.memorization_probe(ema_model, vb, n_features=input_dim)
            except Exception:
                pass
            autopsy.close()
        except Exception:
            pass

    print("\n  TRAINING COMPLETE (Anti-Fragile) -- V23")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V23 xLSTM Training")
    parser.add_argument("--features", type=int, default=29,
                        choices=sorted(SUPPORTED_FEATURE_COUNTS_V23))
    parser.add_argument("--resume", action="store_true")
    add_meta_args(parser)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from diagnostics.autopsy_mode import AutopsyMode as _AP
    _AP.add_argparse(parser)
    args = parser.parse_args()
    run_training(n_features=args.features, resume=args.resume, args=args)
