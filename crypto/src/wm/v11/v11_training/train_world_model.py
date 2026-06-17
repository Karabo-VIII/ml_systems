#!/usr/bin/env python
"""
V11 Training Script -- Microstructure Feature Extractor
=========================================================

Trains V11 world model with dual optimizers:
  - Main optimizer: encoder + return heads + regime head
  - Discriminator optimizer: time-shuffle discriminator

Usage:
    python src/wm/v11/v11_training/train_world_model.py
    python src/wm/v11/v11_training/train_world_model.py --features 25
    python src/wm/v11/v11_training/train_world_model.py --features 13 --resume
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *
from world_model import MicrostructureWorldModel, count_parameters

# Import shared anti-fragile infrastructure from V1
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

# Round-9: shared meta-learner runtime (opt-in via --meta; no-op when empty)
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
try:
    from meta_runtime import MetaRuntime, add_meta_args  # noqa: E402
except Exception:
    MetaRuntime = None
    def add_meta_args(parser):
        # Phase 14.7 cleanup: --meta is a NO-OP here. Only V25 wires it.
        # Declared for orchestrator-compat (run_all_training uniformly passes --meta).
        parser.add_argument("--meta", type=str, default="",
                             help="NO-OP for this version (only V25 implements meta-learners)")
        parser.add_argument("--meta-distill-alpha", type=float, default=0.0,
                             help="NO-OP for this version (paired with --meta)")



def update_ema(model, ema_model, decay=EMA_DECAY):
    """Exponential moving average of model parameters."""
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def save_checkpoint(model, ema_model, optimizer, disc_optimizer,
                    epoch, val_loss, patience, gate_status,
                    best_shic, shic_decline_count, path,
                    ema_only: bool = False):
    """Save checkpoint. ema_only=True saves just EMA weights (for best_ema.pt
    compatibility with wm_ensemble.py which loads ckpt['model_state_dict'])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        # Match V1.0 convention: best_ema.pt = {"model_state_dict": EMA weights}
        torch.save({"model_state_dict": ema_model.state_dict(),
                     "n_features": model.input_dim, "version": "v11"}, path)
    else:
        # Full state for latest.pt / resume
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            "val_loss": val_loss,
            "patience_counter": patience,
            "gate_status": gate_status,
            "best_shic": best_shic,
            "shic_decline_count": shic_decline_count,
            "n_features": model.input_dim,
            "version": "v11",
        }, path)


def load_latest(model, ema_model, optimizer, disc_optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    """Load latest checkpoint if compatible. Prefers latest.pt over best_ema.pt."""
    latest_path = ckpt_dir / ("v11_f%d%s_wm_latest.pt" % (n_features, run_tag_str))
    ema_path = ckpt_dir / ("v11_f%d%s_wm_best_ema.pt" % (n_features, run_tag_str))
    ckpt_path = latest_path if latest_path.exists() else ema_path
    if not ckpt_path.exists():
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    print("  Loading from %s" % ckpt_path.name)

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    # load_latest_collision guard (per CLAUDE.md Code Change Verification #11)
    saved_nf = ckpt.get("n_features", n_features)
    if saved_nf != n_features:
        print("  [WARN] Checkpoint has n_features=%d, expected %d. Skipping." % (saved_nf, n_features))
        return 0, float("inf"), 0, "PENDING", 0.0, 0

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # ema_state_dict absent in ema_only checkpoints (best_ema.pt)
    if "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        ema_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception:
        pass
    try:
        disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
    except Exception:
        pass

    epoch = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    patience = ckpt.get("patience_counter", 0)
    gate = ckpt.get("gate_status", "PENDING")
    best_shic = ckpt.get("best_shic", 0.0)
    shic_decline = ckpt.get("shic_decline_count", 0)

    print("  Resumed from epoch %d (val_loss=%.4f, gate=%s, shic=%.4f)" % (
        epoch, val_loss, gate, best_shic))
    return epoch, val_loss, patience, gate, best_shic, shic_decline


@torch.no_grad()
def validate(model, val_loader):
    """Validate model. Returns metrics dict."""
    model.eval()
    metrics = {"total": [], "direct_ret": [], "regime": [], "regime_acc": []}
    for h in REWARD_HORIZONS:
        metrics["ret_%d" % h] = []
        metrics["dir_acc_%d" % h] = []

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
            pred_ret = model.bucketer.decode(outputs["return_logits"][h])
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    result = {k: np.mean(v) for k, v in metrics.items() if v}

    for h in REWARD_HORIZONS:
        if ic_data[h]["preds"]:
            all_preds = np.concatenate(ic_data[h]["preds"])
            all_reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(all_preds) & np.isfinite(all_reals)
            if mask.sum() > 100:
                result["ic_%d" % h] = float(np.corrcoef(all_preds[mask], all_reals[mask])[0, 1])
                nonzero = np.abs(all_reals[mask]) > 1e-6
                if nonzero.sum() > 50:
                    correct = np.sign(all_preds[mask][nonzero]) == np.sign(all_reals[mask][nonzero])
                    result["dir_acc_%d" % h] = float(correct.mean())

    result["ic_mean"] = float(np.mean([result.get("ic_%d" % h, 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)
    return result


def collate_fn(batch):
    """Custom collate for AntifragileDataset."""
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


def run_training(n_features=None, resume=True, args=None):
    """Main training loop."""
    if n_features is None:
        n_features = INPUT_DIM

    feature_list, input_dim, _ = get_feature_config(n_features)  # 3-tuple; 2-unpack crashed V11 at startup

    # V1-standard run-tag isolation (2026-05-03 cohort harmonization).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""
    print("=" * 70)
    print("  V11 TRAINING -- Microstructure Feature Extractor")
    print("  Features: %d | d_model: %d | Device: %s" % (input_dim, WM_D_MODEL, DEVICE))
    print("  No RSSM | No reconstruction | No dream | WaveNet + MoE + Discriminator")
    print("=" * 70)

    # Anti-fragile config
    af_config = AntifragileConfig(
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        oos_ratio=OOS_RATIO, unseen_ratio=UNSEEN_RATIO,
    )

    # Load data
    print("\n  Loading data...")
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        print("[ERROR] No valid data.")
        return False

    splitter = WalkForwardSplitter(af_config)
    train_segments, val_segments, oos_segments, unseen_segments = splitter.split_four_way(all_segments)
    print("  Train: %d | Val: %d | OOS: %d | Unseen: %d segments" % (
        len(train_segments), len(val_segments), len(oos_segments), len(unseen_segments)))

    regime_weights = compute_regime_weights(train_segments)

    train_ds = AntifragileDataset(train_segments, seq_len=WM_SEQ_LEN,
                                   reward_horizons=REWARD_HORIZONS,
                                   augment=True, config=af_config,
                                   sample_weights=regime_weights)
    val_ds = AntifragileDataset(val_segments, seq_len=WM_SEQ_LEN,
                                 reward_horizons=REWARD_HORIZONS,
                                 augment=False, config=af_config)

    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE,
                               sampler=train_ds.get_sampler(),
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, collate_fn=collate_fn)

    # Build model
    model = MicrostructureWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = MicrostructureWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())

    total_params, enc_params, disc_params = count_parameters(model)
    print("  Parameters: %d total (%d encoder, %d discriminator)" % (
        total_params, enc_params, disc_params))

    autopsy = None
    if getattr(args, "autopsy", False):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from diagnostics.autopsy_mode import AutopsyMode
            autopsy = AutopsyMode(
                model,
                log_dir=Path(__file__).resolve().parents[4] / "logs" / "v11",
                run_tag=f"v11_f{input_dim}",
                sample_every=getattr(args, "autopsy_every", 50),
                loss_window=getattr(args, "autopsy_loss_window", 200),
                explosion_z_threshold=getattr(args, "autopsy_z", 3.0),
            )
            print(f"  [AUTOPSY] enabled; jsonl={autopsy.jsonl_path.name}")
        except Exception as e:
            print(f"  [AUTOPSY] init failed ({type(e).__name__}: {e}); continuing without")

    # Dual optimizers
    enc_params_list = [p for n, p in model.named_parameters()
                       if "discriminator" not in n and p.requires_grad]
    disc_params_list = list(model.discriminator.parameters())

    optimizer = torch.optim.AdamW(enc_params_list, lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                    betas=(0.9, 0.95))
    disc_optimizer = torch.optim.AdamW(disc_params_list, lr=DISC_LR, weight_decay=1e-4,
                                         betas=(0.9, 0.95))

    scaler = torch.amp.GradScaler("cuda")

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 0
    gate_status = "PENDING"
    best_shic = 0.0
    shic_decline_count = 0

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if resume:
        start_epoch, best_val_loss, patience, gate_status, best_shic, shic_decline_count = \
            load_latest(model, ema_model, optimizer, disc_optimizer, MODEL_DIR, input_dim, run_tag_str)

    # ShIC tracker
    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    # LR scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    # 2026-05-10: --max-epochs CLI override
    _max_epochs = WM_EPOCHS
    if args is not None and getattr(args, "max_epochs", None) is not None:
        _max_epochs = args.max_epochs
    for epoch in range(start_epoch, _max_epochs):
        model.train()
        epoch_stats = {k: [] for k in [
            "total", "direct_ret", "regime", "regime_acc",
            "disc", "adv", "dir_acc_1", "dir_acc_4",
        ]}
        for h in REWARD_HORIZONS:
            epoch_stats["ret_%d" % h] = []

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

            # Mixup augmentation
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # VIB KL annealing (2026-05-29 fix): was never passed, so get_loss used
            # its default and the VIB bottleneck ran at full strength from epoch 0,
            # dominating early training. Ramp 0->1 over VIB_KL_ANNEAL_EPOCHS.
            _kl_anneal = min(1.0, (epoch + 1) / max(1, VIB_KL_ANNEAL_EPOCHS))
            # Forward + loss
            with torch.amp.autocast("cuda"):
                loss, loss_dict, outputs = model.get_loss(
                    obs, asset, targets_gpu,
                    mask_ratio=WM_MASK_RATIO,
                    regime_labels=targets_gpu.get("regime_label"),
                    kl_anneal=_kl_anneal,
                )

            # NaN guard with counting (catches nan, +inf, -inf)
            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                disc_optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= 100:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf in epoch {epoch+1}")
                    break
                continue

            # ── Main optimizer step ──────────────────────────────────────
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(enc_params_list, WM_GRAD_CLIP)
            if math.isfinite(grad_norm.item()):
                scaler.step(optimizer)
            scaler.update()

            # ── Discriminator optimizer step ─────────────────────────────
            disc_loss = outputs.get("_disc_loss")
            # 2026-05-09 V11 upgrade: discriminator warmup + freeze cadence
            # (settings flags DISC_WARMUP_STEPS, DISC_FREEZE_AFTER_STEPS).
            global_step = epoch * WM_STEPS_PER_EPOCH + step
            disc_active = (
                global_step >= DISC_WARMUP_STEPS
                and (DISC_FREEZE_AFTER_STEPS == 0 or global_step < DISC_FREEZE_AFTER_STEPS)
            )
            if disc_loss is not None and step % DISC_UPDATE_FREQ == 0 and disc_active:
                if math.isfinite(disc_loss.item()):
                    disc_optimizer.zero_grad(set_to_none=True)
                    disc_loss.backward()
                    clip_grad_norm_(disc_params_list, WM_GRAD_CLIP)
                    disc_optimizer.step()

            if autopsy is not None:
                try:
                    autopsy.step(step, loss_components={
                        "total": float(loss.item()) if hasattr(loss, "item") else float(loss),
                        "disc": float(disc_loss.item()) if disc_loss is not None and hasattr(disc_loss, "item") else 0.0,
                    })
                except Exception:
                    pass

            # EMA update
            update_ema(model, ema_model)
            scheduler.step()

            # Track metrics
            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L="%.3f" % loss_dict["total"],
                    r1="%.3f" % loss_dict.get("ret_1", 0),
                    d="%.3f" % loss_dict.get("disc", 0),
                    a="%.3f" % loss_dict.get("adv", 0),
                )

        # ── Epoch-level validation ───────────────────────────────────────
        val_metrics = validate(ema_model, val_loader)
        val_loss = val_metrics.get("total", float("inf"))

        # Print epoch summary
        train_avg = {k: np.mean(v) for k, v in epoch_stats.items() if v}
        ic_str = " ".join(["ic%d=%.4f" % (h, val_metrics.get("ic_%d" % h, 0)) for h in ACTIVE_HORIZONS])
        print("  Ep %3d | train=%.3f val=%.3f | %s | disc=%.3f adv=%.3f | lr=%.2e" % (
            epoch + 1, train_avg.get("total", 0), val_loss,
            ic_str,
            train_avg.get("disc", 0), train_avg.get("adv", 0),
            optimizer.param_groups[0]["lr"],
        ))

        # ── ShIC check ──────────────────────────────────────────────────
        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_segments, predict_fn, horizon=1)
            cont_ic = val_metrics.get("ic_1", 0)
            ratio = shic / max(cont_ic, 1e-6)
            print("    ShIC=%.4f | IC=%.4f | ratio=%.2f" % (shic, cont_ic, ratio))

            if shic > best_shic:
                best_shic = shic
                shic_decline_count = 0
            else:
                shic_decline_count += 1
                print("    [WARN] ShIC declined (%d/%d)" % (shic_decline_count, SHUFFLED_IC_PATIENCE))

            if shic_decline_count >= SHUFFLED_IC_PATIENCE:
                print("    [STOP] ShIC patience exhausted. Stopping.")
                gate_status = "SHIC_STOP"
                break

        # ── Save best (EMA-only for ensemble compat) ────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            ckpt_path = MODEL_DIR / ("v11_f%d%s_wm_best_ema.pt" % (input_dim, run_tag_str))
            save_checkpoint(model, ema_model, optimizer, disc_optimizer,
                            epoch + 1, val_loss, patience, gate_status,
                            best_shic, shic_decline_count, ckpt_path,
                            ema_only=True)
            print("    [SAVE] %s" % ckpt_path.name)
        else:
            patience += 1

        # ── Latest (full state, every epoch — enables crash resume) ──────
        latest_path = MODEL_DIR / ("v11_f%d%s_wm_latest.pt" % (input_dim, run_tag_str))
        save_checkpoint(model, ema_model, optimizer, disc_optimizer,
                        epoch + 1, val_loss, patience, gate_status,
                        best_shic, shic_decline_count, latest_path)
        # Periodic epoch snapshot every 5 epochs
        if (epoch + 1) % 5 == 0:
            ep_path = MODEL_DIR / ("v11_f%d%s_wm_epoch_%d.pt" % (input_dim, run_tag_str, epoch + 1))
            save_checkpoint(model, ema_model, optimizer, disc_optimizer,
                            epoch + 1, val_loss, patience, gate_status,
                            best_shic, shic_decline_count, ep_path)

    # ── Final gate check ─────────────────────────────────────────────────
    predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
    final_shic = shic_tracker.compute_shuffled_ic(ema_model, val_segments, predict_fn, horizon=1)
    final_ic = val_metrics.get("ic_1", 0)
    final_ratio = final_shic / max(final_ic, 1e-6)

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

    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE (Anti-Fragile)  -- V11")
    print("  Final IC(h=1)=%.4f | ShIC=%.4f | ratio=%.2f" % (final_ic, final_shic, final_ratio))
    print("  Gate: %s" % ("PASS" if final_ratio > 0.3 and final_ic > 0.015 else "FAIL"))
    print("=" * 70)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V11 Training")
    parser.add_argument("--features", type=int, default=34, choices=sorted(SUPPORTED_FEATURE_COUNTS_V11))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override WM_EPOCHS for short validation runs")
    # V1-standard frontier-ML upgrade flags (argparse-only on V11; loss-path
    # wiring is V1-only as of 2026-05-03 cohort harmonization).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args
    add_upgrade_args(parser)
    from diagnostics.autopsy_mode import AutopsyMode as _AP
    _AP.add_argparse(parser)
    args = parser.parse_args()

    run_training(n_features=args.features, resume=args.resume and not args.no_resume,
                  args=args)
