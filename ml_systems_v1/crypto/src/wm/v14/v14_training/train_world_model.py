#!/usr/bin/env python
"""
V14 Training -- Diffusion Return Distribution Model
=============================================

Trains in single-asset mode (V1-compatible) by default.
Each asset is processed independently through the shared WaveNet encoder,
then cross-asset attention is applied during multi-asset evaluation.

Usage:
    python src/wm/v14/v14_training/train_world_model.py --features 25
    python src/wm/v14/v14_training/train_world_model.py --features 13
"""
import argparse
import gc
import math
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

# Import world model using importlib to avoid path conflicts
import importlib.util
_wm_spec = importlib.util.spec_from_file_location(
    "v14_world_model", str(Path(__file__).resolve().parent / "world_model.py"))
_wm_mod = importlib.util.module_from_spec(_wm_spec)
_wm_spec.loader.exec_module(_wm_mod)
DiffusionWorldModel = _wm_mod.DiffusionWorldModel
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


def collate_fn(batch):
    obs_list, tgt_list, asset_list = [], [], []
    for obs, tgt, asset_idx in batch:
        obs_list.append(obs)
        tgt_list.append(tgt)
        asset_list.append(asset_idx)
    obs = torch.stack(obs_list)
    asset = torch.tensor(asset_list, dtype=torch.long)
    targets = {}
    for key in tgt_list[0]:
        targets[key] = torch.stack([t[key] for t in tgt_list])
    return obs, targets, asset


def save_checkpoint(model, ema_model, optimizer, epoch, val_loss,
                    patience, gate, best_shic, shic_decline, n_features, path,
                    ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        torch.save({"model_state_dict": ema_model.state_dict(),
                     "n_features": n_features, "version": "v14"}, path)
    else:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "patience_counter": patience,
            "gate_status": gate, "best_shic": best_shic,
            "shic_decline_count": shic_decline, "n_features": n_features,
            "version": "v14",
        }, path)


def load_latest(model, ema_model, optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    latest_path = ckpt_dir / ("v14_f%d%s_wm_latest.pt" % (n_features, run_tag_str))
    ema_path = ckpt_dir / ("v14_f%d%s_wm_best_ema.pt" % (n_features, run_tag_str))
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
        targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}
        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(
                obs, asset, targets_gpu, mask_ratio=0.0,
                regime_labels=targets_gpu.get("regime_label"))
        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])
        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(outputs["return_logits"][h].reshape(-1, NUM_BINS))
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

    # V1-standard run-tag isolation (2026-05-03 cohort harmonization).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""
    _cfg = get_feature_config(n_features)
    feature_list, input_dim = _cfg[0], _cfg[1]

    print("=" * 70)
    print("  V14 TRAINING -- Diffusion Return Distribution")
    print("  Features: %d | d_model: %d | Device: %s" % (input_dim, WM_D_MODEL, DEVICE))
    print("  Full return distribution via iterative denoising.")
    print("=" * 70)

    af_config = AntifragileConfig()
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        return False

    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way(all_segments)
    regime_weights = compute_regime_weights(train_seg)

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

    model = DiffusionWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = DiffusionWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    print("  Parameters: %d" % count_parameters(model))

    autopsy = None
    if getattr(args, "autopsy", False):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from diagnostics.autopsy_mode import AutopsyMode
            autopsy = AutopsyMode(
                model,
                log_dir=Path(__file__).resolve().parents[4] / "logs" / "v14",
                run_tag=f"v14_f{input_dim}{run_tag_str}",
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
    start_epoch, best_val, patience, gate, best_shic, shic_decline = \
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str) if resume else \
        (0, float("inf"), 0, "PENDING", 0.0, 0)

    # Round-9: meta-learner runtime (no-op when --meta empty; default)
    meta_rt = MetaRuntime.from_args(args, model, MODEL_DIR,
                                     trunk_dim=RETURN_HEAD_DIM,
                                     device=DEVICE, version="v14")

    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    # LR scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 2026-05-10: --max-epochs CLI override
    _max_epochs = WM_EPOCHS
    if args is not None and getattr(args, "max_epochs", None) is not None:
        _max_epochs = args.max_epochs
    for epoch in range(start_epoch, _max_epochs):
        model.train()
        # VIB KL annealing: ramp from 0 to 1 over VIB_KL_ANNEAL_EPOCHS
        kl_anneal = min(1.0, (epoch + 1) / VIB_KL_ANNEAL_EPOCHS) if VIB_KL_ANNEAL_EPOCHS > 0 else 1.0
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
            targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

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
            ckpt = MODEL_DIR / ("v14_f%d%s_wm_best_ema.pt" % (input_dim, run_tag_str))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ckpt,
                            ema_only=True)
            print("    [SAVE] %s" % ckpt.name)

        # Latest (full state, every epoch)
        latest = MODEL_DIR / ("v14_f%d%s_wm_latest.pt" % (input_dim, run_tag_str))
        save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                        patience, gate, best_shic, shic_decline, input_dim, latest)
        if (epoch + 1) % 5 == 0:
            ep = MODEL_DIR / ("v14_f%d%s_wm_epoch_%d.pt" % (input_dim, run_tag_str, epoch + 1))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ep)

    # Round-9: save meta variant checkpoints + summary
    meta_rt.save_and_summarize(version="v14", n_features=input_dim)

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

    print("\n  TRAINING COMPLETE (Anti-Fragile)  -- V14")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V14 Training")
    parser.add_argument("--features", type=int, default=25, choices=sorted(SUPPORTED_FEATURE_COUNTS_V14))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override WM_EPOCHS for short validation runs")
    add_meta_args(parser)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args
    add_upgrade_args(parser)  # V1-standard upgrade flags + --run-tag
    from diagnostics.autopsy_mode import AutopsyMode as _AP
    _AP.add_argparse(parser)
    args = parser.parse_args()
    run_training(n_features=args.features, resume=args.resume, args=args)
