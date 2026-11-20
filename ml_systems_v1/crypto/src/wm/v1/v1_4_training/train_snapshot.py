"""
V1.E Snapshot Ensemble Trainer -- Cyclical Cosine Annealing with Warm Restarts

Same as V1 trainer but uses cyclical LR to produce diverse snapshots.
Each LR minimum represents a different local minimum in loss landscape.
Ensemble of snapshots gives free diversity.

Reference: Huang et al. (2017) "Snapshot Ensembles: Train 1, Get M for Free"

Key differences from train_world_model.py:
  1. LR schedule: cyclical cosine (get_snapshot_lr_for_epoch) instead of single cosine
  2. Snapshot saving at end of each cycle (when LR hits minimum)
  3. Cycle index tracking and logging
  4. Snapshots saved to models/wm/v1/snapshots/

Everything else (data loading, augmentation, validation, anti-fragile) is identical.
"""
import argparse
import torch
import torch.optim as optim
import numpy as np
import sys
import copy
import gc
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from world_model import TransformerWorldModel, count_parameters
from train_world_model import collate_fn, validate, check_gate, get_mask_ratio, update_ema
from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, OverfitMonitor, AntifragileDataset,
    load_full_data, compute_regime_weights,
    make_predict_fn, print_antifragile_header,
)
from log_utils import setup_logging

SNAPSHOT_DIR = MODEL_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def train_snapshot_ensemble():
    """
    Snapshot ensemble training with cyclical cosine annealing.
    Saves EMA model at each LR minimum for later ensemble inference.

    The cyclical schedule divides WM_TOTAL_EPOCHS into SNAPSHOT_N_CYCLES cycles.
    Each cycle does cosine decay from SNAPSHOT_LR_MAX to SNAPSHOT_LR_MIN.
    At the end of each cycle (LR minimum), the EMA model is saved as a snapshot.
    """
    log_path = setup_logging(LOG_DIR, "v1_4e_snapshot_train")
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V1.E SNAPSHOT ENSEMBLE TRAINER (Cyclical Cosine Annealing)")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:     {INPUT_DIM}")
    print(f"  Horizons:     {REWARD_HORIZONS}")
    print(f"  Architecture: d_model={WM_D_MODEL}, layers={WM_N_LAYERS}, "
          f"heads={WM_N_HEADS}, d_ff={WM_D_FF}")
    print(f"  RSSM:         {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  Regularization: dropout={WM_DROPOUT}, weight_decay={WM_WEIGHT_DECAY}")
    print(f"  Augmentation: noise={AUG_NOISE_STD}, feat_drop={AUG_FEAT_DROP}")
    print(f"  LR Schedule:  CYCLICAL COSINE | {SNAPSHOT_N_CYCLES} cycles x "
          f"{SNAPSHOT_EPOCHS_PER_CYCLE} epochs")
    print(f"  LR Range:     [{SNAPSHOT_LR_MIN:.1e}, {SNAPSHOT_LR_MAX:.1e}]")
    print(f"  Warmup:       {SNAPSHOT_WARMUP_EPOCHS} epochs (first cycle only)")
    print(f"  Snapshot Dir: {SNAPSHOT_DIR}")
    print(f"  Top-K:        {SNAPSHOT_TOP_K} snapshots for ensemble")
    print_antifragile_header("V1.E", af_config)

    # -- Load Full Data (no pre-split) -----------------------------------------
    print(f"\n  Loading full data from {DATA_DIR}")
    print("-" * 60)
    all_segments = load_full_data(
        DATA_DIR, FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if all_segments is None:
        print("[ERROR] No valid data. Exiting.")
        return False

    # -- 4-Way Split: 50/20/20/10 (train/val/oos/unseen) ------------------------
    splitter = WalkForwardSplitter(af_config)
    train_segments, val_segments, oos_segments, unseen_segments = \
        splitter.split_four_way_dated(all_segments)

    if not train_segments or not val_segments:
        print("[ERROR] Data split produced empty train/val sets. Exiting.")
        return False

    n_train = sum(len(s["features"]) for s in train_segments)
    n_val = sum(len(s["features"]) for s in val_segments)
    n_oos = sum(len(s["features"]) for s in oos_segments)
    n_unseen = sum(len(s["features"]) for s in unseen_segments)
    n_total = n_train + n_val + n_oos + n_unseen
    print(f"\n  Data Split (purge gap={af_config.purge_gap_bars} bars):")
    print(f"    Train:  {n_train:>12,} bars ({n_train/n_total*100:.1f}%)")
    print(f"    Val:    {n_val:>12,} bars ({n_val/n_total*100:.1f}%)")
    print(f"    OOS:    {n_oos:>12,} bars ({n_oos/n_total*100:.1f}%)")
    print(f"    Unseen: {n_unseen:>12,} bars ({n_unseen/n_total*100:.1f}%) [held out]")

    # -- Regime-Balanced Sampling Weights --------------------------------------
    regime_weights = compute_regime_weights(train_segments)

    # -- Anti-Fragile Datasets -------------------------------------------------
    train_ds = AntifragileDataset(
        train_segments,
        seq_len=WM_SEQ_LEN,
        reward_horizons=REWARD_HORIZONS,
        augment=True,
        config=af_config,
        sample_weights=regime_weights,
    )
    val_ds = AntifragileDataset(
        val_segments,
        seq_len=WM_SEQ_LEN,
        reward_horizons=REWARD_HORIZONS,
        augment=False,
        config=af_config,
    )

    print(f"  Train samples: {len(train_ds):,} | Val samples: {len(val_ds):,}")

    # -- DataLoaders with Regime-Balanced Sampling -----------------------------
    sampler = train_ds.get_sampler()
    persistent = NUM_WORKERS > 0 and not IS_WINDOWS

    train_loader = DataLoader(
        train_ds,
        batch_size=WM_BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=WM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=persistent,
    )

    # -- Initialize Model ------------------------------------------------------
    print(f"\n  Initializing Transformer World Model...")
    model = TransformerWorldModel().to(DEVICE)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    print(f"  Parameters: {count_parameters(model):,}")

    # -- Optimizer -------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=WM_LR,
        weight_decay=WM_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # -- Anti-Fragile Components -----------------------------------------------
    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="rssm")
    best_shuffled_ic = -float("inf")
    latest_shuffled_ic = None

    # -- Snapshot tracking -----------------------------------------------------
    snapshot_paths = []      # List of (cycle_idx, path, shuffled_ic) tuples
    snapshot_ics = {}        # cycle_idx -> shuffled_ic at snapshot time

    # -- Checkpoint: load if exists --------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    gate_passed = False

    ckpt_path = MODEL_DIR / "v1_4e_snapshot_latest.pt"
    if ckpt_path.exists():
        print(f"\n  [RESUME] Loading from {ckpt_path.name}")
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            gate_passed = ckpt.get("gate_passed", False)
            best_shuffled_ic = ckpt.get("best_shuffled_ic", -float("inf"))

            if "ema_state_dict" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema_model.load_state_dict(model.state_dict())

            # Restore snapshot tracking
            snapshot_ics = ckpt.get("snapshot_ics", {})

            print(f"    Resumed at epoch {start_epoch}, "
                  f"best_val_loss={best_val_loss:.4f}, "
                  f"best_shuffled_ic={best_shuffled_ic:.4f}, "
                  f"patience={patience_counter}, "
                  f"gate={'PASSED' if gate_passed else 'pending'}")

        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            start_epoch = 0
            best_val_loss = float("inf")
            patience_counter = 0
            gate_passed = False

    ckpt_history = []

    print(f"\n  Starting from epoch {start_epoch}")
    print("-" * 70)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        model.train()

        # -- Determine cycle info ----------------------------------------------
        cycle_idx = epoch // SNAPSHOT_EPOCHS_PER_CYCLE
        cycle_idx = min(cycle_idx, SNAPSHOT_N_CYCLES - 1)
        epoch_in_cycle = epoch - cycle_idx * SNAPSHOT_EPOCHS_PER_CYCLE
        is_cycle_end = (epoch + 1) % SNAPSHOT_EPOCHS_PER_CYCLE == 0

        # -- Set LR for this epoch (CYCLICAL cosine) ---------------------------
        current_lr = get_snapshot_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        mask_ratio = get_mask_ratio(epoch)

        epoch_keys = ["total", "rec", "kl", "regime"] + [f"ret_{h}" for h in REWARD_HORIZONS]
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0
        train_iter = iter(train_loader)

        pbar = tqdm(
            range(WM_STEPS_PER_EPOCH),
            desc=f"Ep {epoch+1:3d} C{cycle_idx+1}/{SNAPSHOT_N_CYCLES}",
            leave=False,
        )

        for step in pbar:
            # -- Get batch (cycle if exhausted) --------------------------------
            try:
                obs, targets, asset = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                obs, targets, asset = next(train_iter)

            obs = obs.to(DEVICE, non_blocking=True)
            asset = asset.to(DEVICE, non_blocking=True)
            targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}

            # -- Mixup augmentation (batch-level) ------------------------------
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # -- Forward pass with AMP -----------------------------------------
            with torch.amp.autocast("cuda"):
                loss, loss_dict, _ = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True
                )

            # -- NaN detection per loss component ------------------------------
            has_nan = False
            for comp_name, comp_val in loss_dict.items():
                if math.isnan(comp_val) or math.isinf(comp_val):
                    has_nan = True
                    break

            if has_nan or loss.item() > 500:
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            # -- Backward pass with gradient scaling ---------------------------
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
            if math.isfinite(grad_norm.item()):
                grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            # -- Update EMA model ----------------------------------------------
            update_ema(model, ema_model)

            # -- Track metrics -------------------------------------------------
            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L=f"{loss_dict['total']:.3f}",
                    R=f"{loss_dict['rec']:.3f}",
                    KL=f"{loss_dict['kl']:.2f}",
                    r1=f"{loss_dict.get('ret_1', 0):.3f}",
                    gn=f"{grad_norm.item():.2f}",
                )

        # -- Epoch Summary -----------------------------------------------------
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items() if v}
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        ret_str = " | ".join([
            f"r{h}:{avg_stats.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS
        ])
        nan_str = f" | NaN:{nan_count}" if nan_count > 0 else ""
        cycle_str = f"C{cycle_idx+1}/{SNAPSHOT_N_CYCLES} e{epoch_in_cycle+1}/{SNAPSHOT_EPOCHS_PER_CYCLE}"
        print(
            f"  Ep {epoch+1:3d} [{cycle_str}] | "
            f"Loss: {avg_stats.get('total', 0):.4f} | "
            f"Rec: {avg_stats.get('rec', 0):.4f} | "
            f"KL: {avg_stats.get('kl', 0):.2f} | "
            f"{ret_str} | "
            f"GN: {avg_grad_norm:.2f} | "
            f"Mask: {mask_ratio:.2f} | LR: {current_lr:.1e}{nan_str}"
        )

        # -- Snapshot saving deferred to after validation/ShIC -----------------
        # (moved from here to inside validation block below, so ShIC gate works)
        pending_snapshot = is_cycle_end

        # -- Memory cleanup every 10 epochs -----------------------------------
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -- Validation --------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_metrics = validate(ema_model, val_loader)
            val_loss = val_metrics.get("total", 999)
            contiguous_ic = val_metrics.get("ic_1", 0)  # h1 to match ShIC horizon

            # -- Shuffled IC computation (every N epochs) ----------------------
            shuffled_ic = None
            if (epoch + 1) % af_config.shuffled_ic_every == 0:
                shuffled_ic = ic_tracker.compute_shuffled_ic(
                    ema_model, all_segments, predict_fn, horizon=1,
                )
                latest_shuffled_ic = shuffled_ic
                ic_tracker.record(epoch, contiguous_ic, shuffled_ic)

                # Overfitting detection
                should_stop, reason = overfit_monitor.check_overfit(
                    contiguous_ic, shuffled_ic, epoch,
                )
                if should_stop:
                    print(f"\n  [OVERFIT STOP] {reason}")
                    break

                # Track best shuffled IC (primary model selection metric)
                if shuffled_ic > best_shuffled_ic:
                    best_shuffled_ic = shuffled_ic
                    torch.save(ema_model.state_dict(), MODEL_DIR / "v1_4e_best_ema.pt")
                    print(f"  [NEW BEST SHUFFLED IC] {shuffled_ic:.4f}")

                # Update snapshot IC tracking for the current cycle
                # (overwrite with latest shuffled IC if we're still in same cycle)
                if cycle_idx in snapshot_ics:
                    snapshot_ics[cycle_idx] = shuffled_ic

            # -- Deferred snapshot saving (after ShIC is known) ----------------
            if pending_snapshot:
                pending_snapshot = False
                snap_shic = latest_shuffled_ic if latest_shuffled_ic is not None else 0.0
                save_snapshot = True
                skip_reason = ""

                if SNAPSHOT_SHIC_GATE and snap_shic < GATE_IC_MIN:
                    save_snapshot = False
                    skip_reason = (f"ShIC={snap_shic:.4f} < {GATE_IC_MIN} "
                                   f"(gate fail, snapshot skipped)")

                if save_snapshot:
                    snap_path = SNAPSHOT_DIR / f"v1_4_snapshot_{cycle_idx}.pt"
                    torch.save(ema_model.state_dict(), snap_path)
                    snapshot_ics[cycle_idx] = snap_shic
                    print(f"  [SNAPSHOT] Saved cycle {cycle_idx} -> {snap_path.name} "
                          f"(LR={current_lr:.1e}, ShIC={snap_shic:.4f})")
                else:
                    print(f"  [SNAPSHOT SKIP] Cycle {cycle_idx}: {skip_reason}")

            # -- Gate check (with shuffled IC + train/val ratio) ----------------
            passed, reason = check_gate(
                val_metrics, shuffled_ic=latest_shuffled_ic,
                train_loss=avg_stats.get("total"),
            )
            gate_status = "[GATE PASS]" if passed else "[gate fail]"
            if passed:
                gate_passed = True

            save_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_marker = " *BEST_LOSS*"
            else:
                patience_counter += WM_VAL_EVERY

            # -- Save checkpoint with full state -------------------------------
            state = {
                "epoch": epoch + 1,  # next epoch to train (so resume skips completed epoch)
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_shuffled_ic": best_shuffled_ic,
                "patience_counter": patience_counter,
                "gate_passed": gate_passed,
                "snapshot_ics": snapshot_ics,
                "version": "v1_4e_snapshot_ensemble",
            }
            torch.save(state, MODEL_DIR / "v1_4e_snapshot_latest.pt")

            ep_path = MODEL_DIR / f"v1_4e_snapshot_epoch_{epoch+1}.pt"
            torch.save(state, ep_path)
            ckpt_history.append((val_loss, epoch + 1, ep_path))
            ckpt_history.sort(key=lambda x: x[0])
            while len(ckpt_history) > 3:
                _, _, old_path = ckpt_history.pop()
                if old_path.exists() and old_path != ep_path:
                    try:
                        old_path.unlink(missing_ok=True)
                    except Exception:
                        pass

            # -- Print validation results --------------------------------------
            ic_str = " | ".join([
                f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
            ])
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL | "
                f"Loss: {val_loss:.4f} | "
                f"Rec: {val_metrics.get('rec', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"KL: {val_metrics.get('kl', 0):.2f} | "
                f"{gate_status}{save_marker}"
            )

            if not passed:
                print(f"       Reason: {reason}")

            # Early stopping
            if patience_counter >= WM_PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} "
                      f"(patience={WM_PATIENCE} exhausted)")
                break

    # -- Final Report ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  SNAPSHOT ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Best Shuffled IC:    {best_shuffled_ic:.4f}")
    print(f"  Gate Status:         {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  Best EMA weights:    {MODEL_DIR / 'v1_4e_best_ema.pt'}")
    print(f"  Latest checkpoint:   {MODEL_DIR / 'v1_4e_snapshot_latest.pt'}")

    # -- Snapshot summary ------------------------------------------------------
    existing_snapshots = sorted(SNAPSHOT_DIR.glob("v1_4_snapshot_*.pt"))
    print(f"\n  Snapshots saved: {len(existing_snapshots)}")
    for sp in existing_snapshots:
        # Extract cycle index from filename
        try:
            cidx = int(sp.stem.split("_")[-1])
            sic = snapshot_ics.get(cidx, 0.0)
            print(f"    {sp.name}  (Shuffled IC at save: {sic:.4f})")
        except (ValueError, IndexError):
            print(f"    {sp.name}")

    if not gate_passed:
        print("\n  [WARN] World model did not pass validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")
    else:
        print("\n  [OK] Snapshot ensemble ready for inference.")
        print(f"  Use snapshot_ensemble.py to load top-{SNAPSHOT_TOP_K} snapshots.")

    # -- Print shuffled IC history ---------------------------------------------
    if ic_tracker.history["epoch"]:
        print("\n  Shuffled IC History:")
        for i, ep in enumerate(ic_tracker.history["epoch"]):
            c_ic = ic_tracker.history["contiguous_ic"][i]
            s_ic = ic_tracker.history["shuffled_ic"][i]
            gap = ic_tracker.history["ic_gap"][i]
            # Determine which cycle this epoch belongs to
            cyc = min(ep // SNAPSHOT_EPOCHS_PER_CYCLE, SNAPSHOT_N_CYCLES - 1)
            print(f"    Epoch {ep+1:3d} (C{cyc+1}): Contiguous={c_ic:.4f} "
                  f"Shuffled={s_ic:.4f} Gap={gap:.4f}")

    return gate_passed


if __name__ == "__main__":
    # Accept --features so run_all_training.py can pass it; echo it back for is_complete() detection.
    # Snapshot inherits actual dims from the checkpoint, so n_features is used only for logging.
    _ap = argparse.ArgumentParser(description="V1.E Snapshot Ensemble Trainer (V1.4)", add_help=True)
    _ap.add_argument("--features", type=int, default=None,
                     help="Feature count passed by run_all_training.py (used for completion detection)")
    _args, _ = _ap.parse_known_args()
    _nf = _args.features if _args.features is not None else INPUT_DIM
    print("Features:     %d" % _nf)

    success = train_snapshot_ensemble()
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
