#!/usr/bin/env python
"""
V13 Training -- Temporal Fusion Transformer Model
=============================================

Trains in single-asset mode (V1-compatible) by default.
Each asset is processed independently through the shared WaveNet encoder,
then cross-asset attention is applied during multi-asset evaluation.

Usage:
    python src/wm/v13/v13_training/train_world_model.py --features 25
    python src/wm/v13/v13_training/train_world_model.py --features 13
"""
import argparse
import gc
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

# 2026-05-10 fix: add_meta_args was missing (regression from commit
# e5c17ef "--meta wired across all 14 trainers" sweep — V13 missed
# alongside V3/V4/V8 which were fixed in 5ff95f4).
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
try:
    from meta_runtime import MetaRuntime, add_meta_args  # noqa: E402
except Exception:
    MetaRuntime = None
    def add_meta_args(parser):
        parser.add_argument("--meta", type=str, default="",
                             help="NO-OP for V13 (only V25 implements meta-learners)")
        parser.add_argument("--meta-distill-alpha", type=float, default=0.0,
                             help="NO-OP for V13 (paired with --meta)")

# Import world model using importlib to avoid path conflicts
import importlib.util
_wm_spec = importlib.util.spec_from_file_location(
    "v13_world_model", str(Path(__file__).resolve().parent / "world_model.py"))
_wm_mod = importlib.util.module_from_spec(_wm_spec)
_wm_spec.loader.exec_module(_wm_mod)
TFTWorldModel = _wm_mod.TFTWorldModel
count_parameters = _wm_mod.count_parameters

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, AntifragileDataset,
compute_regime_weights,
    make_predict_fn, print_antifragile_header
)
# Read-side contract: load_full_data goes through data_api so future
# pipeline changes only touch one module. See src/data_api/__init__.py.
from data_api import load_full_data_for_training as load_full_data


def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def _targets_to_device(targets, device):
    """Move a targets dict to `device`, recursing into nested label dicts.

    Plain tensors -> .to(device). A nested dict (e.g. the V13_FORWARD_REGIME
    `forward_regime_labels` = {"bear","trend","move"}) is recursed one level so its
    inner tensors land on GPU (forward_regime_aux_loss runs F.cross_entropy on GPU
    logits and needs the labels there too). When V13_FORWARD_REGIME is OFF there is
    NO nested dict, so this behaves identically to the old flat comprehension.
    """
    out = {}
    for k, v in targets.items():
        if isinstance(v, dict):
            out[k] = {ik: iv.to(device, non_blocking=True) for ik, iv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def collate_fn(batch):
    obs_list, tgt_list, asset_list = [], [], []
    for obs, tgt, asset_idx in batch:
        obs_list.append(obs)
        tgt_list.append(tgt)
        asset_list.append(asset_idx)
    obs = torch.stack(obs_list)
    asset = torch.tensor(asset_list, dtype=torch.long)
    targets = {}
    # Forward-regime labels (V13_FORWARD_REGIME flag): the shared AntifragileDataset
    # emits flat fwd_bear/fwd_trend/fwd_move keys; pack them into the nested dict the
    # guarded aux-loss block in get_loss expects. When the flag is OFF these keys are
    # absent -> the nesting is skipped -> targets are identical to base.
    _fr_keys = ("fwd_bear", "fwd_trend", "fwd_move")
    for key in tgt_list[0]:
        if key in _fr_keys:
            continue
        targets[key] = torch.stack([t[key] for t in tgt_list])
    if "fwd_bear" in tgt_list[0]:
        targets["forward_regime_labels"] = {
            "bear":  torch.stack([t["fwd_bear"]  for t in tgt_list]),
            "trend": torch.stack([t["fwd_trend"] for t in tgt_list]),
            "move":  torch.stack([t["fwd_move"]  for t in tgt_list]),
        }
    return obs, targets, asset


def save_checkpoint(model, ema_model, optimizer, epoch, val_loss,
                    patience, gate, best_shic, shic_decline, n_features, path,
                    ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        torch.save({"model_state_dict": ema_model.state_dict(),
                     "n_features": n_features, "version": "v13"}, path)
    else:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "patience_counter": patience,
            "gate_status": gate, "best_shic": best_shic,
            "shic_decline_count": shic_decline, "n_features": n_features,
            "version": "v13",
        }, path)


def load_latest(model, ema_model, optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    latest_path = ckpt_dir / ("v13_f%d%s_wm_latest.pt" % (n_features, run_tag_str))
    ema_path = ckpt_dir / ("v13_f%d%s_wm_best_ema.pt" % (n_features, run_tag_str))
    ckpt_path = latest_path if latest_path.exists() else ema_path
    if not ckpt_path.exists():
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    print("  Loading from %s" % ckpt_path.name)
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
        metrics["ret_%d" % h] = []
    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = _targets_to_device(targets, DEVICE)
        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(
                obs, asset, targets_gpu, mask_ratio=0.0,
                regime_labels=targets_gpu.get("regime_label"))
        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])
        for h in REWARD_HORIZONS:
            # 2026-05-10: quantile-aware decode. When USE_QUANTILE_LOSS=True the
            # return_heads output shape is [B, T, n_q] instead of [B, T, NUM_BINS].
            # Decode IC from median quantile (q=0.5) directly.
            if getattr(model, "_use_quantile_loss", False):
                q = model._quantiles
                median_idx = q.index(0.5) if 0.5 in q else len(q) // 2
                pred_ret = outputs["return_logits"][h][..., median_idx]
            else:
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
                result["ic_%d" % h] = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
    result["ic_mean"] = float(np.mean([result.get("ic_%d" % h, 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)
    return result


def run_training(n_features=None, resume=True, args=None):
    if n_features is None:
        n_features = INPUT_DIM
    # 2026-05-10 fix: standardized 3-tuple API returns (feature_list, input_dim, base_dim)
    _cfg = get_feature_config(n_features)
    feature_list, input_dim = _cfg[0], _cfg[1]

    # V1-standard run-tag isolation (2026-05-03 cohort harmonization).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""

    print("=" * 70)
    print("  V13 TRAINING -- Temporal Fusion Transformer")
    print("  Features: %d | d_model: %d | Device: %s" % (input_dim, WM_D_MODEL, DEVICE))
    print("  Per-timestep variable selection (VSN).")
    print("=" * 70)

    af_config = AntifragileConfig()
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        return False

    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way(all_segments)
    regime_weights = compute_regime_weights(train_seg)

    # -- SHARED LEVER: forward-regime LABELS (V13_FORWARD_REGIME flag) ----------
    # Build labels on the train+val segments BEFORE dataset construction so
    # AntifragileDataset.__getitem__ picks them up by key (fwd_bear/fwd_trend/
    # fwd_move) just like regime_label. When OFF (default): no labels built, keys
    # absent, dataset + collate behave exactly as base. The labels use FUTURE bars
    # only at TARGET-CONSTRUCTION time (never as model inputs); NaN tail rows are
    # masked in forward_regime_aux_loss. (Attach of the head itself happens after
    # the model is built, below.)
    _use_forward_regime = os.environ.get("V13_FORWARD_REGIME", "0") == "1"
    if _use_forward_regime:
        from regime_targets import (
            forward_bear_label as _fwd_bear_lbl,
            forward_trend_label as _fwd_trend_lbl,
            move_onset_label as _fwd_move_lbl,
        )
        print("  [V13_FORWARD_REGIME] Pre-computing forward labels on train+val segments ...")
        _label_segs = list(train_seg) + list(val_seg)
        for _seg in _label_segs:
            _ret1 = _seg.get("target_return_1")
            if _ret1 is None:
                _n = len(_seg["features"])
                _seg["fwd_bear"]  = np.full(_n, np.nan, dtype=np.float32)
                _seg["fwd_trend"] = np.full(_n, np.nan, dtype=np.float32)
                _seg["fwd_move"]  = np.full(_n, np.nan, dtype=np.float32)
                continue
            # Strictly-positive close proxy from h=1 returns: close[t]=prod_{j<t}(1+ret[j]).
            # Labels depend only on RELATIVE moves (ratios), so the proxy is sufficient.
            _cret = np.clip(np.asarray(_ret1, dtype=np.float64), -0.99, 10.0)
            _close_ext = np.empty(len(_cret) + 1, dtype=np.float64)
            _close_ext[0] = 1.0
            np.cumprod(1.0 + _cret, out=_close_ext[1:])
            _close_ext = np.maximum(_close_ext, 1e-8)
            _seg["fwd_bear"]  = _fwd_bear_lbl(_close_ext[:-1],  K=64, dd_thresh=0.05)
            _seg["fwd_trend"] = _fwd_trend_lbl(_close_ext[:-1], K=64)
            _seg["fwd_move"]  = _fwd_move_lbl(_close_ext[:-1],  a=1,  b=64)
        print(f"  [V13_FORWARD_REGIME] Labels ready on {len(_label_segs)} segments "
              f"(K=64, bear/trend/move, NaN tail masked in loss)")
        del _label_segs

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

    model = TFTWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = TFTWorldModel(input_dim=input_dim).to(DEVICE)

    # -- SHARED LEVER: attach forward-regime head (V13_FORWARD_REGIME) ----------
    # Attach to BOTH model and ema_model BEFORE the state-dict sync below so the
    # EMA copy has the matching head structure (validate() runs on ema_model). When
    # OFF (default): no attachment -> _use_forward_regime stays False, head is None,
    # the guarded forward/loss blocks are no-ops -> base model byte-for-byte unchanged.
    if _use_forward_regime:
        from forward_regime_head import attach_forward_regime_head as _attach_frh
        _attach_frh(model)
        _attach_frh(ema_model, verbose=False)

    ema_model.load_state_dict(model.state_dict())
    print("  Parameters: %d" % count_parameters(model))

    # -- VSN flag report (V13_VSN; the model was built with the shared VSN wired
    # in at __init__ time -- env var read there). Self-describing training log.
    _use_vsn = os.environ.get("V13_VSN", "0") == "1"
    if _use_vsn:
        _vsn_params = sum(p.numel() for p in model.shared_vsn.parameters()) if model.shared_vsn is not None else 0
        print(f"  [V13_VSN] Shared causal feature-gate ENABLED ({_vsn_params} params, "
              f"pre-gate before native TFT VSN). NOTE: V13 already has a native TFT VSN; "
              f"this shared input-gate is the SECONDARY/redundant lever (default OFF).")
    else:
        print("  [V13_VSN] Shared VSN disabled (default). V13's native TFT VSN is always on.")
    if _use_forward_regime:
        print("  [V13_FORWARD_REGIME] Forward bear/trend/move head ATTACHED (aux weight 0.10).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                    betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, patience, gate, best_shic, shic_decline = \
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str) if resume else \
        (0, float("inf"), 0, "PENDING", 0.0, 0)

    # Round-9: meta-learner runtime (no-op when --meta empty; default)
    meta_rt = MetaRuntime.from_args(args, model, MODEL_DIR,
                                     trunk_dim=RETURN_HEAD_DIM,
                                     device=DEVICE, version="v13")

    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    # LR scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 2026-05-10: --max-epochs CLI override for short validation runs
    _max_epochs = WM_EPOCHS
    if args is not None and getattr(args, "max_epochs", None) is not None:
        _max_epochs = args.max_epochs
    for epoch in range(start_epoch, _max_epochs):
        model.train()
        train_iter = iter(train_loader)
        epoch_nan_count = 0
        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc="Epoch %3d" % (epoch + 1), leave=False)

        for step in pbar:
            try:
                obs, targets, asset = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                obs, targets, asset = next(train_iter)

            obs = obs.to(DEVICE, non_blocking=True)
            asset = asset.to(DEVICE, non_blocking=True)
            targets_gpu = _targets_to_device(targets, DEVICE)
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            kl_anneal = min((epoch + 1) / max(VIB_KL_ANNEAL_EPOCHS, 1), 1.0)
            with torch.amp.autocast("cuda"):
                loss, loss_dict, base_outputs = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                    regime_labels=targets_gpu.get("regime_label"),
                    kl_anneal=kl_anneal)

            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= 100:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf in epoch {epoch+1}")
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

            # Round-9: residual meta-learner step (no-op when --meta empty)
            if meta_rt.flags:
                meta_rt.train_step_residual(model, base_outputs, targets_gpu, step)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(L="%.3f" % loss_dict["total"],
                                 r1="%.3f" % loss_dict.get("ret_1", 0))

        val_metrics = validate(ema_model, val_loader)
        val_loss = val_metrics.get("total", float("inf"))
        ic_str = " ".join(["ic%d=%.4f" % (h, val_metrics.get("ic_%d" % h, 0)) for h in ACTIVE_HORIZONS])
        print("  Ep %3d | val=%.3f | %s" % (epoch + 1, val_loss, ic_str))

        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_seg, predict_fn, horizon=1)
            print("    ShIC=%.4f | best=%.4f" % (shic, best_shic))
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
            ckpt = MODEL_DIR / ("v13_f%d%s_wm_best_ema.pt" % (input_dim, run_tag_str))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ckpt,
                            ema_only=True)
            print("    [SAVE] %s" % ckpt.name)

        # Latest (full state, every epoch)
        latest = MODEL_DIR / ("v13_f%d%s_wm_latest.pt" % (input_dim, run_tag_str))
        save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                        patience, gate, best_shic, shic_decline, input_dim, latest)
        if (epoch + 1) % 5 == 0:
            ep = MODEL_DIR / ("v13_f%d%s_wm_epoch_%d.pt" % (input_dim, run_tag_str, epoch + 1))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ep)

    # Round-9: save meta variant checkpoints + summary
    meta_rt.save_and_summarize(version="v13", n_features=input_dim)

    print("\n  TRAINING COMPLETE (Anti-Fragile)  -- V13")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V13 Training")
    parser.add_argument("--features", type=int, default=25, choices=sorted(SUPPORTED_FEATURE_COUNTS_V13))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override WM_EPOCHS for short validation runs")
    add_meta_args(parser)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args
    add_upgrade_args(parser)  # V1-standard upgrade flags + --run-tag
    args = parser.parse_args()
    run_training(n_features=args.features, resume=args.resume, args=args)
