"""
V8.3.D Diversity World Model Trainer -- Multi-Head NCL (Anti-Fragile Edition)

Architecture: V8.3 backbone (Neural ODE-RSSM) + K parallel return heads with NCL
Version: v8_1d_diversity_ncl_antifragile

Key differences from train_world_model.py:
  - Uses DiversityWorldModel instead of NeuralODEWorldModel
  - Logs NCL penalty and dynamics regularization in epoch summary
  - Can optionally load V8.3 backbone weights (--load-backbone)
  - Saves checkpoints as v8_1d_wm_*.pt
  - Uses same LR schedule, augmentation, and anti-fragile framework as V8

Anti-fragile training features (inherited from V8.3):
  - Walk-forward cross-validation with purge gap
  - Shuffled IC as primary model selection metric
  - Rich augmentation: noise, feature dropout, temporal jitter, mixup, block swap
  - Regime-balanced sampling
  - Overfitting monitor (IC gap detection with auto-stop)
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import copy
import gc
import math
import argparse
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from ncl_model import DiversityWorldModel, count_parameters
from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, OverfitMonitor, AntifragileDataset,
    load_full_data, compute_regime_weights,
    make_predict_fn, print_antifragile_header,
)
from log_utils import setup_logging, teardown_logging
from diagnostics.feature_autopsy import FeatureAutopsy
from revin import RevIN


# ==============================================================================
# COLLATE
# ==============================================================================

def collate_fn(batch):
    """Custom collate for dict targets."""
    obs = torch.stack([b[0] for b in batch])
    asset = torch.stack([b[2] for b in batch])
    targets = {}
    for h in REWARD_HORIZONS:
        targets[h] = torch.stack([b[1][h] for b in batch])
    return obs, targets, asset


# ==============================================================================
# TRAINING UTILITIES
# ==============================================================================

def get_mask_ratio(epoch: int) -> float:
    """Progressive masking schedule: ramp from start to end over ramp_epochs."""
    if epoch >= WM_MASK_RAMP_EPOCHS:
        return WM_MASK_RATIO_END
    progress = epoch / WM_MASK_RAMP_EPOCHS
    return WM_MASK_RATIO_START + progress * (WM_MASK_RATIO_END - WM_MASK_RATIO_START)


def update_ema(model, ema_model, decay=EMA_DECAY):
    """Update EMA model weights: ema = decay * ema + (1 - decay) * model."""
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# ==============================================================================
# VALIDATION
# ==============================================================================

@torch.no_grad()
def validate(model, val_loader, revin=None):
    """Validate model on the FULL holdout set with per-horizon IC."""
    model.eval()
    metrics = {
        "rec": [], "kl": [], "kl_raw": [], "regime": [], "regime_acc": [],
        "total": [], "ncl": [], "dynamics_reg": [],
    }
    for h in REWARD_HORIZONS:
        metrics[f"ret_{h}"] = []

    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}

        if revin is not None:
            obs = revin(obs, mode='norm')

        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(obs, asset, targets_gpu, mask_ratio=0.0)

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
                result[f"ic_{h}"] = float(np.corrcoef(all_preds[mask], all_reals[mask])[0, 1])
            else:
                result[f"ic_{h}"] = 0.0
        else:
            result[f"ic_{h}"] = 0.0

    result["ic"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    return result


def check_gate(val_metrics: dict, shuffled_ic: float = None,
               train_loss: float = None) -> tuple:
    """Validate model quality against gate criteria (including anti-fragile)."""
    reasons = []
    rec = val_metrics.get("rec", 999)
    ic = val_metrics.get("ic", 0)
    kl = val_metrics.get("kl", 0)
    val_loss = val_metrics.get("total", 999)

    if rec > GATE_REC_MSE_MAX:
        reasons.append(f"Rec MSE={rec:.4f} > {GATE_REC_MSE_MAX}")
    if ic < GATE_IC_MIN:
        reasons.append(f"IC={ic:.4f} < {GATE_IC_MIN}")
    if kl < GATE_KL_MIN:
        reasons.append(f"KL={kl:.4f} too low (collapse)")
    if kl > GATE_KL_MAX:
        reasons.append(f"KL={kl:.4f} too high")

    if shuffled_ic is not None and shuffled_ic < GATE_IC_MIN:
        reasons.append(f"Shuffled IC={shuffled_ic:.4f} < {GATE_IC_MIN} (memorizing)")

    if shuffled_ic is not None and ic > 0:
        ratio = shuffled_ic / ic
        if ratio < GATE_SHUFFLED_IC_RATIO_MIN:
            reasons.append(
                f"ShIC/IC ratio={ratio:.3f} < {GATE_SHUFFLED_IC_RATIO_MIN} "
                f"(temporal memorization)"
            )

    if train_loss is not None and train_loss > 1e-8 and val_loss < 900:
        loss_ratio = val_loss / train_loss
        if loss_ratio > GATE_LOSS_RATIO_MAX:
            reasons.append(
                f"Val/Train loss ratio={loss_ratio:.2f} > {GATE_LOSS_RATIO_MAX} "
                f"(overfitting)"
            )

    if reasons:
        return False, " | ".join(reasons)
    return True, "All criteria met"


# ==============================================================================
# MAIN TRAINING LOOP (Anti-Fragile)
# ==============================================================================

def train_diversity_model(load_backbone: str = None, freeze_backbone: bool = False,
                          n_features: int = 18):
    """
    Anti-fragile training loop for V8.3.D Diversity World Model.

    Same framework as V8 trainer but with:
    - DiversityWorldModel (K parallel return heads + NCL)
    - NCL penalty and dynamics regularization logging
    - Optional V8.3 backbone loading
    - v8_1d_* checkpoint naming

    Args:
        load_backbone: path to V8.3 checkpoint to load backbone from (optional)
        freeze_backbone: if True, freeze backbone and only train diversity heads
        n_features: number of input features (14 or 19)
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    feat_tag = f"f{n_features}"

    log_path = setup_logging(LOG_DIR, f"v8_1d_{feat_tag}_train")
    torch.set_float32_matmul_precision("medium")

    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V8.D DIVERSITY WORLD MODEL TRAINER (Multi-Head NCL | Anti-Fragile)")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:     {input_dim} ({feat_tag})")
    print(f"  Horizons:     {REWARD_HORIZONS}")
    print(f"  Architecture: d_model={WM_D_MODEL}, ODE hidden={ODE_HIDDEN_LAYERS}")
    print(f"  RSSM:         {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  Diversity:    K={DIVERSITY_N_HEADS} heads, NCL_lambda={DIVERSITY_NCL_LAMBDA}")
    print(f"  Head dim:     {DIVERSITY_HEAD_DIM}, dropout={DIVERSITY_HEAD_DROPOUT}")
    print(f"  Regularization: dropout={WM_DROPOUT}, weight_decay={DIVERSITY_WEIGHT_DECAY}")
    print(f"  Augmentation: noise={AUG_NOISE_STD}, feat_drop={AUG_FEAT_DROP}")
    print(f"  LR Schedule:  warmup={WM_WARMUP_EPOCHS}ep, peak={DIVERSITY_LR}, min={WM_MIN_LR}")
    print(f"  Dynamics reg: lambda={LAMBDA_DYNAMICS}")
    if load_backbone:
        print(f"  Backbone:     Loading from {load_backbone}")
        print(f"  Freeze:       {'Yes' if freeze_backbone else 'No'}")
    print_antifragile_header("V8.D", af_config)

    print(f"\n  Loading full data from {DATA_DIR}")
    print("-" * 60)
    all_segments = load_full_data(
        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if all_segments is None:
        print("[ERROR] No valid data. Exiting.")
        return False

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

    regime_weights = compute_regime_weights(train_segments)

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

    print(f"\n  Initializing V8.3.D Diversity World Model...")
    model = DiversityWorldModel(input_dim=input_dim).to(DEVICE)

    if load_backbone:
        backbone_path = Path(load_backbone)
        if backbone_path.exists():
            print(f"  Loading V8.3 backbone from {backbone_path}")
            ckpt = torch.load(backbone_path, map_location=DEVICE, weights_only=False)
            if "model_state_dict" in ckpt:
                v8_sd = ckpt["model_state_dict"]
            elif "ema_state_dict" in ckpt:
                v8_sd = ckpt["ema_state_dict"]
            else:
                v8_sd = ckpt
            model.load_backbone_from_v8(v8_sd, freeze=freeze_backbone)
        else:
            print(f"  [WARN] Backbone path not found: {backbone_path}. Training from scratch.")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    revin = RevIN(num_features=input_dim).to(DEVICE)

    total_params = count_parameters(model)
    head_params = sum(p.numel() for p in model.diversity_heads.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Diversity head params: {head_params:,} ({head_params // DIVERSITY_N_HEADS:,} per head)")

    all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(revin.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=DIVERSITY_LR,
        weight_decay=DIVERSITY_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="rssm", revin=revin)
    best_shuffled_ic = -float("inf")
    latest_shuffled_ic = None
    shic_decline_count = 0

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    gate_passed = False

    ckpt_path = NCL_MODEL_DIR / f"v8_1d_{feat_tag}_wm_latest.pt"
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

            if "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])


            # -- Checkpoint collision guard --
            ckpt_n_feat = ckpt.get("n_features")
            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:
                raise RuntimeError(
                    f"Checkpoint was trained with {ckpt_n_feat} features but "
                    f"current settings use INPUT_DIM={INPUT_DIM}. "
                    f"Delete {ckpt_path.name} or use matching feature config."
                )

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
    nan_recovery_count = 0

    # -- Feature Autopsy (non-console diagnostics) ----------------------------
    autopsy_path = NCL_MODEL_DIR / f"v8_3d_autopsy.jsonl"
    autopsy = FeatureAutopsy(
        feature_list=FEATURE_LIST, base_dim=INPUT_DIM,
        log_path=autopsy_path, horizons=REWARD_HORIZONS, device=DEVICE,
    )

    print(f"\n  Starting from epoch {start_epoch}")
    print("-" * 70)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    for epoch in range(start_epoch, DIVERSITY_TOTAL_EPOCHS):
        model.train()

        current_lr = get_lr_for_epoch(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        mask_ratio = get_mask_ratio(epoch)

        kl_anneal = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) if KL_ANNEAL_EPOCHS > 0 else 1.0

        gumbel_tau = GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * min(1.0, (epoch + 1) / GUMBEL_TAU_ANNEAL_EPOCHS)

        epoch_keys = (
            ["total", "rec", "kl", "kl_raw", "regime", "regime_acc", "ncl", "dynamics_reg"]
            + [f"ret_{h}" for h in REWARD_HORIZONS]
        )
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0
        train_iter = iter(train_loader)

        pbar = tqdm(
            range(DIVERSITY_STEPS_PER_EPOCH),
            desc=f"Epoch {epoch+1:3d}",
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

            obs = revin(obs, mode='norm')

            with torch.amp.autocast("cuda"):
                loss, loss_dict, _ = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                    kl_anneal=kl_anneal, gumbel_tau=gumbel_tau,
                    temporal_ctx_drop=TEMPORAL_CTX_DROP,
                )

            has_nan = False
            for comp_name, comp_val in loss_dict.items():
                if math.isnan(comp_val) or math.isinf(comp_val):
                    has_nan = True
                    break

            if has_nan or loss.item() > 500:
                nan_count += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = clip_grad_norm_(
                list(model.parameters()) + list(revin.parameters()), WM_GRAD_CLIP
            )
            if math.isfinite(grad_norm.item()):
                grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            update_ema(model, ema_model)
            ema_model._gumbel_tau = gumbel_tau

            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L=f"{loss_dict['total']:.3f}",
                    R=f"{loss_dict['rec']:.3f}",
                    KL=f"{loss_dict['kl']:.2f}",
                    r1=f"{loss_dict.get('ret_1', 0):.3f}",
                    ncl=f"{loss_dict.get('ncl', 0):.4f}",
                    dyn=f"{loss_dict.get('dynamics_reg', 0):.4f}",
                    gn=f"{grad_norm.item():.2f}",
                )

        avg_stats = {k: np.mean(v) for k, v in epoch_stats.items() if v}
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0

        ret_str = " | ".join([
            f"r{h}:{avg_stats.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS
        ])
        nan_str = f" | NaN:{nan_count}" if nan_count > 0 else ""
        ncl_str = f" | NCL:{avg_stats.get('ncl', 0):.4f}"
        dyn_str = f" | Dyn:{avg_stats.get('dynamics_reg', 0):.4f}"
        print(
            f"  Ep {epoch+1:3d} | "
            f"Loss: {avg_stats.get('total', 0):.4f} | "
            f"Rec: {avg_stats.get('rec', 0):.4f} | "
            f"KL: {avg_stats.get('kl', 0):.2f} | "
            f"{ret_str}{ncl_str}{dyn_str} | "
            f"GN: {avg_grad_norm:.2f} | "
            f"Mask: {mask_ratio:.2f} | LR: {current_lr:.1e}{nan_str}"
        )

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
            all_params = list(model.parameters())
            if revin is not None:
                all_params += list(revin.parameters())
            optimizer = torch.optim.AdamW(
                all_params, lr=current_lr,
                weight_decay=DIVERSITY_WEIGHT_DECAY,
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

        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == DIVERSITY_TOTAL_EPOCHS - 1:
            val_metrics = validate(ema_model, val_loader, revin=revin)
            val_loss = val_metrics.get("total", 999)
            contiguous_ic = val_metrics.get("ic_1", 0)  # h1 to match ShIC horizon

            shuffled_ic = None
            if (epoch + 1) % af_config.shuffled_ic_every == 0:
                shuffled_ic = ic_tracker.compute_shuffled_ic(
                    ema_model, all_segments, predict_fn, horizon=1,
                )
                latest_shuffled_ic = shuffled_ic
                ic_tracker.record(epoch, contiguous_ic, shuffled_ic)

                should_stop, reason = overfit_monitor.check_overfit(
                    contiguous_ic, shuffled_ic, epoch,
                )
                if should_stop:
                    print(f"\n  [OVERFIT STOP] {reason}")
                    break

                if shuffled_ic > best_shuffled_ic:
                    best_shuffled_ic = shuffled_ic
                    torch.save({
                        "model_state_dict": ema_model.state_dict(),
                        "revin_state_dict": revin.state_dict(),
                    }, NCL_MODEL_DIR / f"v8_1d_{feat_tag}_wm_best_ema.pt")
                    print(f"  [NEW BEST SHUFFLED IC] {shuffled_ic:.4f}")
                    shic_decline_count = 0
                else:
                    shic_drop = best_shuffled_ic - shuffled_ic
                    if shic_drop > SHUFFLED_IC_MIN_DECLINE:
                        shic_decline_count += 1
                        print(f"  [ShIC decline #{shic_decline_count}] "
                              f"{shuffled_ic:.4f} < best {best_shuffled_ic:.4f} "
                              f"(drop={shic_drop:.4f} > {SHUFFLED_IC_MIN_DECLINE})")
                    else:
                        print(f"  [ShIC flat] {shuffled_ic:.4f} ~ best {best_shuffled_ic:.4f} "
                              f"(drop={shic_drop:.4f} < {SHUFFLED_IC_MIN_DECLINE}, not counted)")
                    if shic_decline_count >= SHUFFLED_IC_PATIENCE:
                        print(f"\n  [SHIC STOP] ShIC declining for "
                              f"{shic_decline_count} consecutive checks "
                              f"(best={best_shuffled_ic:.4f}, "
                              f"current={shuffled_ic:.4f})")
                        break

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

            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "revin_state_dict": revin.state_dict(),
                "best_val_loss": best_val_loss,
                "best_shuffled_ic": best_shuffled_ic,
                "patience_counter": patience_counter,
                "gate_passed": gate_passed,
                "version": "v8_1d_diversity_ncl_antifragile",
                "n_features": INPUT_DIM,
                "n_diversity_heads": DIVERSITY_N_HEADS,
                "ncl_lambda": DIVERSITY_NCL_LAMBDA,
            }
            torch.save(state, NCL_MODEL_DIR / f"v8_1d_{feat_tag}_wm_latest.pt")
            torch.save(model.state_dict(), NCL_MODEL_DIR / f"v8_1d_{feat_tag}_wm_weights.pt")

            ep_path = NCL_MODEL_DIR / f"v8_1d_{feat_tag}_wm_epoch_{epoch+1}.pt"
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

            ic_str = " | ".join([
                f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
            ])
            ncl_val_str = f" | NCL:{val_metrics.get('ncl', 0):.4f}"
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL | "
                f"Loss: {val_loss:.4f} | "
                f"Rec: {val_metrics.get('rec', 0):.4f} | "
                f"{ic_str}{ncl_val_str}{shuffled_str} | "
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

            if patience_counter >= WM_PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} "
                      f"(patience={WM_PATIENCE} exhausted)")
                break

    print("\n" + "=" * 70)
    print("  V8.D DIVERSITY TRAINING COMPLETE (Anti-Fragile)")
    print("=" * 70)
    print(f"  Diversity heads:     {DIVERSITY_N_HEADS}")
    print(f"  NCL lambda:          {DIVERSITY_NCL_LAMBDA}")
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Best Shuffled IC:    {best_shuffled_ic:.4f}")
    print(f"  Gate Status:         {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  EMA weights:         {NCL_MODEL_DIR / f'v8_1d_{feat_tag}_wm_best_ema.pt'}")
    print(f"  Latest checkpoint:   {NCL_MODEL_DIR / f'v8_1d_{feat_tag}_wm_latest.pt'}")

    if not gate_passed:
        print("\n  [WARN] Diversity model did not pass validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")
    else:
        print("\n  [OK] V8.3.D Diversity model ready for downstream use.")

    if ic_tracker.history["epoch"]:
        print("\n  Shuffled IC History:")
        for i, ep in enumerate(ic_tracker.history["epoch"]):
            c_ic = ic_tracker.history["contiguous_ic"][i]
            s_ic = ic_tracker.history["shuffled_ic"][i]
            gap = ic_tracker.history["ic_gap"][i]
            print(f"    Epoch {ep+1:3d}: Contiguous={c_ic:.4f} "
                  f"Shuffled={s_ic:.4f} Gap={gap:.4f}")

    return gate_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V8.3.D Diversity World Model Trainer")
    parser.add_argument(
        "--load-backbone", type=str, default=None,
        help="Path to V8.3 checkpoint to initialize backbone weights from"
    )
    parser.add_argument(
        "--freeze-backbone", action="store_true",
        help="Freeze backbone and only train diversity heads"
    )
    parser.add_argument("--features", type=int, choices=[13, 18, 30, 34, 37, 41], default=18,
                        help="Number of input features (14 or 19)")
    args = parser.parse_args()

    success = train_diversity_model(
        load_backbone=args.load_backbone,
        freeze_backbone=args.freeze_backbone,
        n_features=args.features,
    )
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
