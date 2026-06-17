"""
V8.E Multi-Seed Ensemble Trainer -- K independent full training runs.

Each seed produces a model in a different basin of the loss landscape.
Low inter-model correlation (rho ~ 0.3-0.5) gives strong ensemble IC boost:
  IC_ensemble = IC_single * sqrt(K / (1 + (K-1) * rho))

Replaces the old cyclical cosine snapshot approach (Huang et al. 2017)
which failed ShIC due to aggressive LR resets destroying cross-sectional patterns.

Key design:
  - K=ENSEMBLE_N_SEEDS independent full training runs (back-to-back)
  - Each seed uses vanilla LR schedule (warmup + cosine decay)
  - Master checkpoint tracks which seeds completed (robust resume)
  - Even if a seed early-stops, the orchestrator continues to the next

V8-specific:
  - NeuralODEWorldModel (Neural ODE with RK4 solver, RSSM latent space)
  - KL annealing and Gumbel tau annealing
  - dynamics_reg epoch key (ODE regularization)
  - model_type="rssm" for predict_fn
"""
import torch
import torch.optim as optim
import numpy as np
import random
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
from settings import get_feature_config
from world_model import NeuralODEWorldModel, count_parameters
from train_world_model import collate_fn, validate, check_gate, get_mask_ratio, update_ema
from revin import RevIN
from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, OverfitMonitor, AntifragileDataset,
    load_full_data, compute_regime_weights,
    make_predict_fn, print_antifragile_header,
)
from log_utils import setup_logging
from diagnostics.feature_autopsy import FeatureAutopsy


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_master_checkpoint(seeds, completed, seed_metrics):
    """Save master checkpoint tracking ensemble training progress."""
    state = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "completed": completed,
        "seed_metrics": seed_metrics,
        "version": "v8e_multi_seed_ensemble",
    }
    torch.save(state, ENSEMBLE_MODEL_DIR / "v8e_master.pt")


def train_single_seed(
    seed_idx: int,
    seed: int,
    all_segments: list,
    train_segments: list,
    val_segments: list,
    train_loader: DataLoader,
    val_loader: DataLoader,
    af_config,
    use_revin: bool = False,
    n_features: int = None,
    feature_list: list = None,
    input_dim: int = None,
    base_dim: int = None,
) -> dict:
    """
    Train one full NeuralODE model with a specific seed. Returns metrics dict.

    This is essentially the vanilla training loop from train_world_model.py
    but with explicit seeding and seed-specific checkpoint naming.

    Args:
        seed_idx:        Index of this seed (0..K-1)
        seed:            Random seed value
        all_segments:    All data segments (for ShIC computation)
        train_segments:  Training segments
        val_segments:    Validation segments
        train_loader:    Training DataLoader (reused across seeds)
        val_loader:      Validation DataLoader (reused across seeds)
        af_config:       AntifragileConfig instance
        use_revin:       Enable RevIN normalization (disabled by default)

    Returns:
        Dict with: best_ic, best_shic, final_epoch, gate_passed, early_stopped
    """
    set_seed(seed)
    feature_list = feature_list or FEATURE_LIST
    input_dim = input_dim or INPUT_DIM
    base_dim = base_dim or INPUT_DIM
    n_features = n_features or INPUT_DIM

    print(f"\n{'=' * 70}")
    print(f"  SEED {seed_idx}/{ENSEMBLE_N_SEEDS - 1} (seed={seed})")
    print(f"{'=' * 70}")
    print(f"  Architecture: NeuralODE (d_model={WM_D_MODEL})")
    print(f"  ODE:          method={ODE_METHOD}, hidden={ODE_HIDDEN_LAYERS}")
    print(f"  RSSM:         {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  LR Schedule:  warmup={WM_WARMUP_EPOCHS}ep, peak={WM_LR}, min={WM_MIN_LR}")
    print(f"  Epochs:       {WM_TOTAL_EPOCHS} (patience={WM_PATIENCE})")

    # -- Initialize fresh model ---------------------------------------------------
    model = NeuralODEWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    # -- RevIN (disabled by default; causes temporal memorization) ---------------
    revin = RevIN(num_features=INPUT_DIM).to(DEVICE) if use_revin else None
    print(f"  Parameters: {count_parameters(model):,}")

    # -- Optimizer ----------------------------------------------------------------
    all_params = list(model.parameters())
    if revin is not None:
        all_params += list(revin.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=WM_LR,
        weight_decay=WM_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # -- Anti-Fragile Components --------------------------------------------------
    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="rssm", revin=revin)
    best_shuffled_ic = -float("inf")
    latest_shuffled_ic = None
    shic_decline_count = 0
    best_val_loss = float("inf")
    patience_counter = 0
    gate_passed = False
    best_ic = 0.0

    # -- Per-seed checkpoint: resume if exists ------------------------------------
    start_epoch = 0
    ckpt_path = ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_latest.pt"
    if ckpt_path.exists():
        print(f"  [RESUME] Loading from {ckpt_path.name}")
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "ema_state_dict" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state_dict"])
            else:
                ema_model.load_state_dict(model.state_dict())
            if revin is not None and "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])

            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            best_shuffled_ic = ckpt.get("best_shuffled_ic", -float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            gate_passed = ckpt.get("gate_passed", False)
            best_ic = ckpt.get("best_ic", 0.0)


            # -- Checkpoint collision guard --
            ckpt_n_feat = ckpt.get("n_features")
            if ckpt_n_feat is not None and ckpt_n_feat != n_features:
                raise RuntimeError(
                    f"Checkpoint was trained with {ckpt_n_feat} features but "
                    f"current settings use INPUT_DIM={INPUT_DIM}. "
                    f"Delete {ckpt_path.name} or use matching feature config."
                )

            print(f"    Resumed at epoch {start_epoch}, "
                  f"best_val_loss={best_val_loss:.4f}, "
                  f"best_shic={best_shuffled_ic:.4f}, "
                  f"patience={patience_counter}")
        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            start_epoch = 0

    ckpt_history = []
    early_stopped = False
    nan_recovery_count = 0

    # -- Feature Autopsy (non-console diagnostics) ----------------------------
    autopsy_path = ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_autopsy.jsonl"
    autopsy = FeatureAutopsy(
        feature_list=feature_list, base_dim=base_dim,
        log_path=autopsy_path, horizons=REWARD_HORIZONS, device=DEVICE,
    )

    print(f"\n  Starting from epoch {start_epoch}")
    print("-" * 70)

    # ==========================================================================
    # TRAINING LOOP (per seed)
    # ==========================================================================
    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        model.train()

        # -- LR schedule (vanilla cosine, NOT cyclical) ----------------------------
        current_lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        mask_ratio = get_mask_ratio(epoch)

        # KL annealing
        kl_anneal = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) if KL_ANNEAL_EPOCHS > 0 else 1.0

        # Gumbel tau annealing
        gumbel_tau = GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * min(
            1.0, (epoch + 1) / GUMBEL_TAU_ANNEAL_EPOCHS
        )

        epoch_keys = ["total", "rec", "kl", "kl_raw", "regime", "regime_acc", "dynamics_reg"] + [
            f"ret_{h}" for h in REWARD_HORIZONS
        ]
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0
        train_iter = iter(train_loader)

        pbar = tqdm(
            range(WM_STEPS_PER_EPOCH),
            desc=f"S{seed_idx} Ep {epoch+1:3d}/{WM_TOTAL_EPOCHS}",
            leave=False,
        )

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

            # -- Sequence shuffling (anti-temporal-memorization) --
            if SEQ_SHUFFLE_PROB > 0:
                B_sz = obs.shape[0]
                for b in range(B_sz):
                    if torch.rand(1).item() < SEQ_SHUFFLE_PROB:
                        perm = torch.randperm(obs.shape[1], device=obs.device)
                        obs[b] = obs[b][perm]
                        for h in targets_gpu:
                            targets_gpu[h][b] = targets_gpu[h][b][perm]

            # RevIN normalization (disabled by default)
            if revin is not None:
                obs = revin(obs, mode='norm')

            # Forward + loss
            with torch.amp.autocast("cuda"):
                loss, loss_dict, _ = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                    kl_anneal=kl_anneal, gumbel_tau=gumbel_tau,
                    temporal_ctx_drop=TEMPORAL_CTX_DROP,
                )

            # NaN detection
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 500:
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                if nan_count > 20:
                    print(f"\n  [ERROR] Too many NaN batches ({nan_count}). Aborting epoch.")
                    break
                continue

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_params = list(model.parameters())
            if revin is not None:
                grad_params += list(revin.parameters())
            grad_norm = clip_grad_norm_(grad_params, WM_GRAD_CLIP)
            if math.isfinite(grad_norm.item()):
                grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            # EMA update
            update_ema(model, ema_model)
            ema_model._gumbel_tau = gumbel_tau

            # Track metrics
            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L=f"{loss_dict['total']:.3f}",
                    R=f"{loss_dict['rec']:.3f}",
                    KL=f"{loss_dict['kl']:.2f}",
                    Dyn=f"{loss_dict.get('dynamics_reg', 0):.3f}",
                    r1=f"{loss_dict.get('ret_1', 0):.3f}",
                    gn=f"{grad_norm.item():.2f}",
                )

        # -- Epoch Summary ---------------------------------------------------------
        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items() if v}
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        ret_str = " | ".join([
            f"r{h}:{avg_stats.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS
        ])
        nan_str = f" | NaN:{nan_count}" if nan_count > 0 else ""

        # Kendall weights
        with torch.no_grad():
            _s = model.log_vars.clamp(-6.0, 6.0)
            _s_rec = _s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN).item()
            _s_r1 = _s[2].clamp(max=RETURN_LOG_VAR_CLAMP_MAX).item()
            _w_rec = math.exp(-_s_rec)
            _w_r1 = math.exp(-_s_r1)
            _w_kl = math.exp(-_s[1].item())
            _regime_idx = 2 + len(REWARD_HORIZONS)
            _s_reg = _s[_regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX).item()
            _w_reg = math.exp(-_s_reg)

        print(
            f"  S{seed_idx} Ep {epoch+1:3d} | "
            f"Loss: {avg_stats.get('total', 0):.4f} | "
            f"Rec: {avg_stats.get('rec', 0):.4f} | "
            f"KL: {avg_stats.get('kl', 0):.2f} | "
            f"DynReg: {avg_stats.get('dynamics_reg', 0):.4f} | "
            f"RegAcc: {avg_stats.get('regime_acc', 0):.3f} | "
            f"{ret_str} | "
            f"GN: {avg_grad_norm:.2f} | "
            f"Mask: {mask_ratio:.2f} | LR: {current_lr:.1e} | "
            f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} w_kl:{_w_kl:.2f} w_reg:{_w_reg:.2f}"
            f"{nan_str}"
        )

        # -- Memory cleanup --------------------------------------------------------
        # -- NaN collapse recovery ------------------------------------------------
        nan_frac = nan_count / WM_STEPS_PER_EPOCH if WM_STEPS_PER_EPOCH > 0 else 0
        if nan_frac > 0.5:
            nan_recovery_count += 1
            if nan_recovery_count > 3:
                print(f"  [FATAL] NaN collapse after {nan_recovery_count} recovery attempts. Aborting.")
                break
            print(f"  [NaN RECOVERY] {nan_frac:.0%} NaN batches at epoch {epoch+1}. "
                  f"Reinitializing (attempt {nan_recovery_count}/3)")
            for m in model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            if hasattr(model, 'log_vars'):
                model.log_vars.data.zero_()
            optimizer = torch.optim.AdamW(
                list(model.parameters()), lr=current_lr,
                weight_decay=WM_WEIGHT_DECAY,
            )
            scaler = torch.amp.GradScaler("cuda")
            ema_model = copy.deepcopy(model)
            best_val_loss = float("inf")
            best_shuffled_ic = -float("inf")
            patience_counter = 0
            print(f"  [NaN RECOVERY] Model reinitialized. Continuing from epoch {epoch+2}.")
            continue

        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -- Validation ------------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_metrics = validate(ema_model, val_loader, revin=revin)
            val_loss = val_metrics.get("total", 999)
            contiguous_ic = val_metrics.get("ic_1", 0)  # h1 to match ShIC horizon

            if contiguous_ic > best_ic:
                best_ic = contiguous_ic

            # Shuffled IC computation
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
                    print(f"\n  [OVERFIT STOP] Seed {seed_idx}: {reason}")
                    early_stopped = True
                    break

                # Track best shuffled IC
                if shuffled_ic > best_shuffled_ic:
                    best_shuffled_ic = shuffled_ic
                    # Save best EMA for this seed
                    best_ema_state = {"model_state_dict": ema_model.state_dict()}
                    if revin is not None:
                        best_ema_state["revin_state_dict"] = revin.state_dict()
                    torch.save(best_ema_state, ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_best.pt")
                    print(f"  [NEW BEST ShIC] Seed {seed_idx}: {shuffled_ic:.4f}")
                    shic_decline_count = 0
                else:
                    shic_drop = best_shuffled_ic - shuffled_ic
                    if shic_drop > SHUFFLED_IC_MIN_DECLINE:
                        shic_decline_count += 1
                        print(f"  [ShIC decline #{shic_decline_count}] "
                              f"{shuffled_ic:.4f} < best {best_shuffled_ic:.4f}")
                    if shic_decline_count >= SHUFFLED_IC_PATIENCE:
                        print(f"\n  [SHIC STOP] Seed {seed_idx}: declining for "
                              f"{shic_decline_count} consecutive checks")
                        early_stopped = True
                        break

            # Gate check
            passed, reason = check_gate(
                val_metrics, shuffled_ic=latest_shuffled_ic,
                train_loss=avg_stats.get("total"),
            )
            gate_status = "[GATE PASS]" if passed else "[gate fail]"
            if latest_shuffled_ic is not None:
                gate_passed = passed  # Only meaningful after ShIC is measured

            save_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_marker = " *BEST_LOSS*"
            else:
                patience_counter += WM_VAL_EVERY

            # Save per-seed checkpoint
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_loss": best_val_loss,
                "best_shuffled_ic": best_shuffled_ic,
                "patience_counter": patience_counter,
                "gate_passed": gate_passed,
                "best_ic": best_ic,
                "seed": seed,
                "seed_idx": seed_idx,
                "version": "v8e_seed_training",
                "n_features": n_features,
            }
            if revin is not None:
                state["revin_state_dict"] = revin.state_dict()
            torch.save(state, ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_latest.pt")

            # Keep top-3 epoch checkpoints
            ep_path = ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_epoch_{epoch+1}.pt"
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

            # Print validation results
            ic_str = " | ".join([
                f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
            ])
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL S{seed_idx} | "
                f"Loss: {val_loss:.4f} | "
                f"Rec: {val_metrics.get('rec', 0):.4f} | "
                f"DynReg: {val_metrics.get('dynamics_reg', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"KL: {val_metrics.get('kl', 0):.2f} | "
                f"{gate_status}{save_marker}"
            )

            if not passed:
                print(f"       Reason: {reason}")

            # -- Feature Autopsy (non-console diagnostics) --------------------
            try:
                do_ablation = (epoch + 1) % 10 == 0
                do_raw_ic = (epoch + 1 == WM_VAL_EVERY)
                autopsy.run(
                    ema_model, val_loader, epoch + 1, revin=revin,
                    do_ablation=do_ablation, do_raw_ic=do_raw_ic,
                )
            except Exception:
                pass  # autopsy must never crash training

            # Early stopping
            if patience_counter >= WM_PATIENCE:
                print(f"\n  [STOP] Seed {seed_idx}: early stopping at epoch {epoch+1} "
                      f"(patience={WM_PATIENCE} exhausted)")
                early_stopped = True
                break

    # -- Save final seed model (EMA weights + RevIN) -------------------------------
    final_path = ENSEMBLE_MODEL_DIR / f"v8_seed_{seed_idx}.pt"
    final_state = {"model_state_dict": ema_model.state_dict()}
    if revin is not None:
        final_state["revin_state_dict"] = revin.state_dict()
    torch.save(final_state, final_path)
    print(f"\n  [OK] Seed {seed_idx} final model saved -> {final_path.name}")

    # Cleanup per-seed training checkpoint
    latest_ckpt = ENSEMBLE_MODEL_DIR / f"v8e_seed_{seed_idx}_latest.pt"
    if latest_ckpt.exists():
        try:
            latest_ckpt.unlink(missing_ok=True)
        except Exception:
            pass
    for ep_ckpt in ENSEMBLE_MODEL_DIR.glob(f"v8e_seed_{seed_idx}_epoch_*.pt"):
        try:
            ep_ckpt.unlink(missing_ok=True)
        except Exception:
            pass

    final_epoch = epoch + 1 if epoch < WM_TOTAL_EPOCHS else WM_TOTAL_EPOCHS

    # Print ShIC history for this seed
    if ic_tracker.history["epoch"]:
        print(f"\n  Seed {seed_idx} ShIC History:")
        for i, ep in enumerate(ic_tracker.history["epoch"]):
            c_ic = ic_tracker.history["contiguous_ic"][i]
            s_ic = ic_tracker.history["shuffled_ic"][i]
            gap = ic_tracker.history["ic_gap"][i]
            print(f"    Epoch {ep+1:3d}: Contiguous={c_ic:.4f} "
                  f"Shuffled={s_ic:.4f} Gap={gap:.4f}")

    return {
        "best_ic": best_ic,
        "best_shic": best_shuffled_ic,
        "final_epoch": final_epoch,
        "gate_passed": gate_passed,
        "early_stopped": early_stopped,
    }


def train_ensemble(use_revin: bool = False, n_features: int = None):
    """
    Multi-seed ensemble orchestrator for V8 NeuralODE. Loads data once,
    then trains K independent models back-to-back with different seeds.
    """
    log_path = setup_logging(LOG_DIR, "v8e_ensemble_train")
    torch.set_float32_matmul_precision("medium")

    af_config = AntifragileConfig()

    # -- Feature config (runtime override via --features) ---------------------
    if n_features is not None:
        feature_list, input_dim, base_dim = get_feature_config(n_features)
        feat_tag = f"f{n_features}"
    else:
        feature_list, input_dim, base_dim = FEATURE_LIST, INPUT_DIM, INPUT_DIM
        n_features = INPUT_DIM
        feat_tag = f"f{INPUT_DIM}"


    seeds = ENSEMBLE_SEEDS[:ENSEMBLE_N_SEEDS]
    n_seeds = len(seeds)

    print("=" * 70)
    print("  V8.E MULTI-SEED ENSEMBLE TRAINER (Neural ODE)")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:     {n_features} ({feat_tag})")
    print(f"  Seeds (K={n_seeds}): {seeds}")
    print(f"  Top-K:        {ENSEMBLE_TOP_K} seeds for ensemble inference")
    print(f"  Epochs/seed:  {WM_TOTAL_EPOCHS} (patience={WM_PATIENCE})")
    print(f"  Output dir:   {ENSEMBLE_MODEL_DIR}")
    print_antifragile_header("V8.E", af_config)

    # -- Load Full Data (ONCE, shared across all seeds) ----------------------------
    print(f"\n  Loading full data from {DATA_DIR}")
    print("-" * 60)
    all_segments = load_full_data(
        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if all_segments is None:
        print("[ERROR] No valid data. Exiting.")
        return False

    # -- 4-Way Split: 50/20/20/10 (train/val/oos/unseen) ------------------------
    splitter = WalkForwardSplitter(af_config)
    train_segments, val_segments, oos_segments, unseen_segments = \
        splitter.split_four_way(all_segments)

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

    # -- Regime-Balanced Sampling Weights -----------------------------------------
    regime_weights = compute_regime_weights(train_segments)

    # -- Datasets & DataLoaders (shared across seeds) -----------------------------
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

    # -- Load master checkpoint (resume support) ----------------------------------
    completed = []
    seed_metrics = {}

    master_path = ENSEMBLE_MODEL_DIR / "v8e_master.pt"
    if master_path.exists():
        print(f"\n  [RESUME] Loading master checkpoint")
        try:
            master = torch.load(master_path, map_location="cpu", weights_only=False)
            completed = master.get("completed", [])
            seed_metrics = master.get("seed_metrics", {})
            print(f"    Completed seeds: {completed}")
            for idx in completed:
                m = seed_metrics.get(idx, {})
                print(f"    Seed {idx}: IC={m.get('best_ic', 0):.4f}, "
                      f"ShIC={m.get('best_shic', 0):.4f}, "
                      f"epochs={m.get('final_epoch', '?')}, "
                      f"gate={'PASS' if m.get('gate_passed') else 'fail'}")
        except Exception as e:
            print(f"  [WARN] Master checkpoint load failed: {e}. Starting fresh.")

    # -- Train each seed back-to-back ---------------------------------------------
    any_gate_passed = False

    for seed_idx in range(n_seeds):
        if seed_idx in completed:
            print(f"\n  [SKIP] Seed {seed_idx} already completed")
            if seed_metrics.get(seed_idx, {}).get("gate_passed", False):
                any_gate_passed = True
            continue

        seed = seeds[seed_idx]

        print(f"\n{'#' * 70}")
        print(f"  STARTING SEED {seed_idx}/{n_seeds - 1} (seed={seed})")
        print(f"  Remaining: {n_seeds - seed_idx} seeds")
        print(f"{'#' * 70}")

        metrics = train_single_seed(
            seed_idx=seed_idx,
            seed=seed,
            all_segments=all_segments,
            train_segments=train_segments,
            val_segments=val_segments,
            train_loader=train_loader,
            val_loader=val_loader,
            af_config=af_config,
            use_revin=use_revin,
            n_features=n_features,
            feature_list=feature_list,
            input_dim=input_dim,
            base_dim=base_dim,
        )

        # Record completion
        completed.append(seed_idx)
        seed_metrics[seed_idx] = metrics
        save_master_checkpoint(seeds, completed, seed_metrics)

        if metrics["gate_passed"]:
            any_gate_passed = True

        print(f"\n  [DONE] Seed {seed_idx}: "
              f"IC={metrics['best_ic']:.4f}, "
              f"ShIC={metrics['best_shic']:.4f}, "
              f"epochs={metrics['final_epoch']}, "
              f"gate={'PASS' if metrics['gate_passed'] else 'fail'}, "
              f"early_stopped={metrics['early_stopped']}")

        # VRAM cleanup between seeds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- Final Report --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  MULTI-SEED ENSEMBLE TRAINING COMPLETE (V8 Neural ODE)")
    print("=" * 70)

    existing_seeds = sorted(ENSEMBLE_MODEL_DIR.glob("v8_seed_*.pt"))
    print(f"\n  Seed models saved: {len(existing_seeds)}/{n_seeds}")

    for seed_idx in range(n_seeds):
        m = seed_metrics.get(seed_idx, {})
        status = "[PASS]" if m.get("gate_passed") else "[fail]"
        stop_reason = " (early)" if m.get("early_stopped") else ""
        print(f"    Seed {seed_idx} (seed={seeds[seed_idx]}): "
              f"IC={m.get('best_ic', 0):.4f} | "
              f"ShIC={m.get('best_shic', 0):.4f} | "
              f"Ep={m.get('final_epoch', '?')}{stop_reason} | "
              f"{status}")

    n_passed = sum(1 for m in seed_metrics.values() if m.get("gate_passed"))
    print(f"\n  Gate passed: {n_passed}/{n_seeds} seeds")

    if not any_gate_passed:
        print("\n  [WARN] No seed passed the validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")
    else:
        print(f"\n  [OK] Ensemble ready. Use snapshot_ensemble.py to load top-{ENSEMBLE_TOP_K}.")

    return any_gate_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V8.E Multi-Seed Ensemble Trainer")
    parser.add_argument("--features", type=int, choices=[13, 18, 30, 34, 37, 41], default=None,
                        help="Feature count override: 13/18/30/37 (default: settings.INPUT_DIM)")
    parser.add_argument("--revin", action="store_true",
                        help="Enable RevIN normalization (disabled by default)")
    args = parser.parse_args()
    success = train_ensemble(use_revin=args.revin, n_features=args.features)
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
