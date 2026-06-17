"""
V8.1 World Model Trainer -- Neural ODE Edition + XD Anti-Memorization

Architecture: Neural ODE with RK4 solver for continuous-time market dynamics
Version: v8_1_neural_ode_antifragile

Supports --features 13|18|30|37 for feature ablation:
  - 13: Base features only (base_dim == input_dim, no XD paths)
  - 18: Full features with XD anti-memorization defenses (dropout+noise)

Anti-fragile training features:
  - Walk-forward cross-validation with purge gap (eliminates temporal leakage)
  - Shuffled IC as primary model selection metric (detects memorization)
  - Rich augmentation: noise, feature dropout, temporal jitter, mixup, block swap
  - Regime-balanced sampling (prevents neutral-bias)
  - Overfitting monitor (IC gap detection with auto-stop)

Production features (preserved):
  - Full checkpoint state for resumability
  - EMA model for stable validation and final weights
  - LR Schedule: Linear warmup (10 epochs) + cosine decay
  - Gradient norm logging, NaN detection, memory cleanup
  - Windows-compatible (NUM_WORKERS=0)
  - Per-horizon IC calculation
  - Dynamics regularization logging
"""
import torch
import torch.optim as optim
import numpy as np
import sys
import math
import copy
import gc
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import NeuralODEWorldModel, count_parameters
from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, OverfitMonitor, AntifragileDataset,
compute_regime_weights,
    make_predict_fn, print_antifragile_header
)
# Read-side contract: load_full_data goes through data_api so future
# pipeline changes only touch one module. See src/data_api/__init__.py.
from data_api import load_full_data_for_training as load_full_data
from log_utils import setup_logging, teardown_logging
from revin import RevIN


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(batch):
    """Custom collate for dict targets."""
    obs = torch.stack([b[0] for b in batch])
    asset = torch.stack([b[2] for b in batch])
    targets = {}
    for h in REWARD_HORIZONS:
        targets[h] = torch.stack([b[1][h] for b in batch])
    # Include precomputed regime labels if available
    if "regime_label" in batch[0][1]:
        targets["regime_label"] = torch.stack([b[1]["regime_label"] for b in batch])
    return obs, targets, asset


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_mask_ratio(epoch: int) -> float:
    """Linearly ramp mask ratio from start to end over ramp epochs."""
    if epoch >= WM_MASK_RAMP_EPOCHS:
        return WM_MASK_RATIO_END
    progress = epoch / WM_MASK_RAMP_EPOCHS
    return WM_MASK_RATIO_START + progress * (WM_MASK_RATIO_END - WM_MASK_RATIO_START)


def set_lr(optimizer, lr: float):
    """Manually set learning rate for all parameter groups."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def update_ema(model, ema_model, decay=EMA_DECAY):
    """Exponential moving average update of model weights."""
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class CheckpointManager:
    """Manages model checkpoints with full resumability."""

    def __init__(self, save_dir: Path, keep_top_k: int = 3, prefix: str = "v8_1_f18"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.prefix = prefix
        self.history = []

    def save(self, model, optimizer, scaler, epoch, val_loss,
             best_val_loss, patience_counter, gate_passed, best_shuffled_ic=-float("inf"),
             ema_model=None, revin=None):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_shuffled_ic": best_shuffled_ic,
            "patience_counter": patience_counter,
            "gate_passed": gate_passed,
            "n_features": getattr(self, "_expected_n_features", None),
            "version": "v8_1_neural_ode_antifragile",
        }
        if ema_model is not None:
            state["ema_state_dict"] = ema_model.state_dict()
        if revin is not None:
            state["revin_state_dict"] = revin.state_dict()
        torch.save(state, self.save_dir / f"{self.prefix}_wm_latest.pt")
        torch.save(model.state_dict(), self.save_dir / f"{self.prefix}_wm_weights.pt")

        ep_path = self.save_dir / f"{self.prefix}_wm_epoch_{epoch}.pt"
        torch.save(state, ep_path)
        self.history.append((val_loss, epoch, ep_path))
        self.history.sort(key=lambda x: x[0])

        while len(self.history) > self.keep_top_k:
            _, _, old_path = self.history.pop()
            if old_path.exists() and old_path != ep_path:
                old_path.unlink(missing_ok=True)

    def load_latest(self, model, optimizer, scaler, ema_model=None, revin=None):
        path = self.save_dir / f"{self.prefix}_wm_latest.pt"
        if not path.exists():
            return 0, float("inf"), 0, False, -float("inf")

        print(f"  Resuming from {path.name}")
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            version = ckpt.get("version", "unknown")
            if "v8" not in version:
                print(f"  [WARN] Version mismatch: {version}. Starting fresh.")
                return 0, float("inf"), 0, False, -float("inf")

            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            if ema_model is not None:
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
                else:
                    ema_model.load_state_dict(model.state_dict())

            if revin is not None and "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])

            start_epoch = ckpt.get("epoch", 0)
            best_val = ckpt.get("best_val_loss", float("inf"))
            patience = ckpt.get("patience_counter", 0)
            gate = ckpt.get("gate_passed", False)
            best_shic = ckpt.get("best_shuffled_ic", -float("inf"))


            # -- Checkpoint collision guard --
            ckpt_n_feat = ckpt.get("n_features")
            if ckpt_n_feat is not None and hasattr(self, '_expected_n_features'):
                if ckpt_n_feat != self._expected_n_features:
                    raise RuntimeError(
                        f"Checkpoint trained with --features {ckpt_n_feat} but "
                        f"current invocation uses --features {self._expected_n_features}. "
                        f"Delete {path.name} or use matching --features.")
            print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}, "
                  f"best_shIC={best_shic:.4f}, patience={patience}")
            return start_epoch, best_val, patience, gate, best_shic

        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            return 0, float("inf"), 0, False, -float("inf")


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, revin=None):
    """Full validation over ALL val data with per-horizon IC."""
    model.eval()

    metrics = {"rec": [], "kl": [], "regime": [], "total": [], "dynamics_reg": []}
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
            _, loss_dict, outputs = model.get_loss(obs, asset, targets_gpu, mask_ratio=0.0, regime_labels=targets_gpu.get("regime_label"))

        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])

        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(outputs["return_logits"][h])
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    result = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    for h in REWARD_HORIZONS:
        if ic_data[h]["preds"]:
            all_preds = np.concatenate(ic_data[h]["preds"])
            all_reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(all_preds) & np.isfinite(all_reals)
            if mask.sum() > 100:
                result[f"ic_{h}"] = float(
                    np.corrcoef(all_preds[mask], all_reals[mask])[0, 1]
                )
            else:
                result[f"ic_{h}"] = 0.0
        else:
            result[f"ic_{h}"] = 0.0

    result["ic_mean"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)
    # NOTE: Don't call model.train() here -- caller passes ema_model which must
    # stay in eval mode. The training model is set to train() at the top of each
    # epoch loop anyway. Matches V1/V2 pattern.
    return result


def check_gate(val_metrics: dict, shuffled_ic: float = None,
               train_loss: float = None) -> tuple:
    """Check validation gate criteria including anti-fragile shuffled IC."""
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

    # Anti-fragile gate: shuffled IC absolute threshold
    if shuffled_ic is not None and shuffled_ic < GATE_IC_MIN:
        reasons.append(f"Shuffled IC={shuffled_ic:.4f} < {GATE_IC_MIN} (memorizing)")

    # Anti-fragile gate: shuffled IC / contiguous IC ratio
    if shuffled_ic is not None and ic > 0:
        ratio = shuffled_ic / ic
        if ratio < GATE_SHUFFLED_IC_RATIO_MIN:
            reasons.append(
                f"ShIC/IC ratio={ratio:.3f} < {GATE_SHUFFLED_IC_RATIO_MIN} "
                f"(temporal memorization)"
            )

    # Overfitting gate: train/val loss ratio
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


# =============================================================================
# MAIN TRAINING LOOP (Anti-Fragile)
# =============================================================================

def train_world_model(n_features: int = 18, use_revin: bool = False):
    """
    Anti-fragile training loop for V8.1 NeuralODE-RSSM.

    Args:
        n_features: Number of features to use: 14 (base only) or 19 (full).
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    feat_tag = f"f{n_features}"
    ckpt_prefix = f"v8_1_{feat_tag}"
    log_path = setup_logging(LOG_DIR, f"v8_1_{feat_tag}_train")
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V8.1 WORLD MODEL TRAINER (Neural ODE | Anti-Fragile)")
    print("=" * 70)
    print(f"  Platform:         {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Device:           {DEVICE}")
    print(f"  Features:         {n_features} ({len(feature_list)} in feature_list)")
    print(f"  Base Dim:         {base_dim} (posterior/decoder restricted)")
    print(f"  Horizons:         {REWARD_HORIZONS}")
    print(f"  Seq Length:       {WM_SEQ_LEN}")
    print(f"  Batch Size:       {WM_BATCH_SIZE}")
    print(f"  Epochs:           {WM_TOTAL_EPOCHS}")
    print(f"  LR:               {WM_LR} -> {WM_MIN_LR} (cosine)")
    print(f"  Warmup Epochs:    {WM_WARMUP_EPOCHS}")
    print(f"  Weight Decay:     {WM_WEIGHT_DECAY}")
    print(f"  D_MODEL:          {WM_D_MODEL}")
    print(f"  ODE Hidden:       {ODE_HIDDEN_LAYERS}")
    print(f"  ODE Method:       {ODE_METHOD}")
    print(f"  ODE Step Size:    {ODE_STEP_SIZE}")
    print(f"  Lambda Dynamics:  {LAMBDA_DYNAMICS}")
    print(f"  RSSM Latent:      {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  Ckpt prefix:      {ckpt_prefix}")
    print_antifragile_header("V8.1", af_config)

    # -- Load Full Data --------------------------------------------------------
    print("\nLoading full data...")
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

    print(f"\n  Train sequences: {len(train_ds):,}")
    print(f"  Val sequences:   {len(val_ds):,}")

    sampler = train_ds.get_sampler()
    train_loader = DataLoader(
        train_ds,
        batch_size=WM_BATCH_SIZE,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=WM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=NUM_WORKERS > 0,
    )

    # -- Initialize Model ------------------------------------------------------
    model = NeuralODEWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    param_count = count_parameters(model)
    print(f"\n  Model Parameters: {param_count:,}")

    if param_count > 20_000_000:
        print(f"  [WARN] Model has {param_count:,} params -- may be too large for 8GB VRAM")

    # -- RevIN (distribution shift normalization) --------------------------------
    revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None
    # -- Optimizer (SOTA: betas=(0.9, 0.95) from LLaMA/Chinchilla) ------------
    optimizer = optim.AdamW(
        list(model.parameters()) + list(revin.parameters()),
        lr=WM_MIN_LR,
        weight_decay=WM_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # -- Anti-Fragile Components -----------------------------------------------
    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="rssm", revin=revin)
    latest_shuffled_ic = None
    shic_decline_count = 0

    # -- Checkpoint ------------------------------------------------------------
    ckpt_mgr = CheckpointManager(BASE_MODEL_DIR, keep_top_k=3, prefix=ckpt_prefix)
    ckpt_mgr._expected_n_features = n_features
    start_epoch, best_val_loss, patience_counter, gate_passed, best_shuffled_ic = \
        ckpt_mgr.load_latest(model, optimizer, scaler, ema_model=ema_model, revin=revin)

    nan_count = 0
    max_nan_per_epoch = 20

    print(f"\n  Starting from epoch {start_epoch}")
    print(f"  ATME:            temporal_ctx_drop={TEMPORAL_CTX_DROP}, seq_shuffle={SEQ_SHUFFLE_PROB}")
    print("-" * 70)

    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        epoch_start = time.time()
        model.train()

        # -- LR (warmup + cosine) ----------------------------------------------
        current_lr = get_lr_for_epoch(epoch)
        set_lr(optimizer, current_lr)

        mask_ratio = get_mask_ratio(epoch)

        kl_anneal = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) if KL_ANNEAL_EPOCHS > 0 else 1.0

        # Gumbel tau annealing: linear decay from START to END over ANNEAL epochs
        gumbel_tau = GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * min(1.0, (epoch + 1) / GUMBEL_TAU_ANNEAL_EPOCHS)

        epoch_keys = ["total", "rec", "kl", "regime", "regime_acc", "dynamics_reg"] + \
                     [f"ret_{h}" for h in REWARD_HORIZONS]
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        epoch_nan_count = 0

        train_iter = iter(train_loader)
        pbar = tqdm(
            range(WM_STEPS_PER_EPOCH),
            desc=f"Epoch {epoch+1:3d}/{WM_TOTAL_EPOCHS}",
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

            # -- Mixup augmentation (batch-level) ------------------------------
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # -- Sequence shuffling (anti-temporal-memorization) ---------------
            if SEQ_SHUFFLE_PROB > 0:
                B_sz = obs.shape[0]
                for b in range(B_sz):
                    if torch.rand(1).item() < SEQ_SHUFFLE_PROB:
                        perm = torch.randperm(obs.shape[1], device=obs.device)
                        obs[b] = obs[b][perm]
                        for h in targets_gpu:
                            targets_gpu[h][b] = targets_gpu[h][b][perm]

            # -- RevIN normalization (distribution shift) ----------------------
            obs = revin(obs, mode='norm')

            # -- Forward pass with AMP -----------------------------------------
            with torch.amp.autocast("cuda"):
                loss, loss_dict, _ = model.get_loss(
                    obs, asset, targets_gpu,
                    mask_ratio=mask_ratio,
                    block_mask=True,
                    kl_anneal=kl_anneal,
                    gumbel_tau=gumbel_tau,
                    temporal_ctx_drop=TEMPORAL_CTX_DROP,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            # -- NaN detection -------------------------------------------------
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 500:
                optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= max_nan_per_epoch:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf losses in epoch {epoch+1}")
                    break
                continue

            # -- Backward pass -------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(revin.parameters()), WM_GRAD_CLIP
            )
            grad_norms.append(grad_norm.item())

            scaler.step(optimizer)
            scaler.update()

            # -- EMA update ----------------------------------------------------
            update_ema(model, ema_model)
            ema_model._gumbel_tau = gumbel_tau  # Sync Gumbel tau to EMA model

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
                    r64=f"{loss_dict.get('ret_64', 0):.3f}",
                    gn=f"{grad_norms[-1]:.2f}" if grad_norms else "N/A",
                )

        # -- Epoch Summary -----------------------------------------------------
        epoch_time = time.time() - epoch_start
        avg_stats = {k: float(np.mean(v)) for k, v in epoch_stats.items() if v}
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        ret_str = " | ".join([
            f"r{h}:{avg_stats.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS
        ])
        nan_str = f" | NaN:{epoch_nan_count}" if epoch_nan_count > 0 else ""

        # Log effective Kendall weights (verify corridors work)
        with torch.no_grad():
            _s = model.log_vars.clamp(-6.0, 6.0)
            _s_rec = _s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN).item()
            _s_r1 = _s[2].clamp(max=RETURN_LOG_VAR_CLAMP_MAX).item()
            _w_rec = math.exp(-_s_rec)
            _w_r1 = math.exp(-_s_r1)

        print(
            f"  Ep {epoch+1:3d} | "
            f"Loss:{avg_stats.get('total', 0):.4f} | "
            f"Rec:{avg_stats.get('rec', 0):.4f} | "
            f"KL:{avg_stats.get('kl', 0):.2f} | "
            f"DynReg:{avg_stats.get('dynamics_reg', 0):.4f} | "
            f"RegAcc:{avg_stats.get('regime_acc', 0):.3f} | "
            f"{ret_str} | "
            f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} | "
            f"Mask:{mask_ratio:.2f} | "
            f"LR:{current_lr:.1e} | "
            f"GradN:{avg_grad_norm:.2f} | "
            f"{epoch_time:.0f}s{nan_str}"
        )

        # -- Validation --------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_start = time.time()
            val_metrics = validate(ema_model, val_loader, revin=revin)
            val_time = time.time() - val_start
            val_loss = val_metrics.get("total", 999)
            contiguous_ic = val_metrics.get("ic_1", 0)  # h1 to match ShIC horizon

            # -- Shuffled IC (every N epochs) ----------------------------------
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
                    }, BASE_MODEL_DIR / f"{ckpt_prefix}_wm_best_ema.pt")
                    print(f"  [NEW BEST SHUFFLED IC] {shuffled_ic:.4f}")
                    shic_decline_count = 0
                else:
                    shic_drop = best_shuffled_ic - shuffled_ic
                    if shic_drop > SHUFFLED_IC_MIN_DECLINE:
                        shic_decline_count += 1
                        print(f"  [ShIC decline #{shic_decline_count}] {shuffled_ic:.4f} (best={best_shuffled_ic:.4f}, drop={shic_drop:.4f})")
                    else:
                        print(f"  [ShIC flat] {shuffled_ic:.4f} (best={best_shuffled_ic:.4f}, drop={shic_drop:.4f} < {SHUFFLED_IC_MIN_DECLINE}) -- not counted")
                    if shic_decline_count >= SHUFFLED_IC_PATIENCE:
                        print(f"\n  [SHIC STOP] ShIC declining for "
                              f"{shic_decline_count} consecutive checks "
                              f"(best={best_shuffled_ic:.4f}, "
                              f"current={shuffled_ic:.4f})")
                        break

            # -- Gate check (with shuffled IC) ---------------------------------
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
                save_marker = " ** BEST **"
            else:
                patience_counter += WM_VAL_EVERY

            ckpt_mgr.save(
                model, optimizer, scaler,
                epoch + 1, val_loss,
                best_val_loss, patience_counter, gate_passed, best_shuffled_ic,
                ema_model=ema_model, revin=revin,
            )

            ic_str = " | ".join([
                f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
            ])
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL | "
                f"Loss:{val_loss:.4f} | "
                f"Rec:{val_metrics.get('rec', 0):.4f} | "
                f"DynReg:{val_metrics.get('dynamics_reg', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"IC_avg:{contiguous_ic:.4f} | "
                f"KL:{val_metrics.get('kl', 0):.2f} | "
                f"{gate_status}{save_marker} | "
                f"{val_time:.0f}s"
            )

            if not passed:
                print(f"       Reason: {reason}")

            if patience_counter >= WM_PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} (patience={WM_PATIENCE})")
                break

        # -- Memory cleanup ----------------------------------------------------
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Final Report ----------------------------------------------------------
    print()
    print("=" * 70)
    print("  TRAINING COMPLETE (Anti-Fragile)")
    print("=" * 70)
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Best Shuffled IC: {best_shuffled_ic:.4f}")
    print(f"  Gate Status:      {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  Weights saved:    {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_best_ema.pt'}")
    print(f"  Checkpoint:       {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_latest.pt'}")

    if not gate_passed:
        print()
        print("  [WARN] World model did not pass validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")

    # -- Print shuffled IC history ---------------------------------------------
    if ic_tracker.history["epoch"]:
        print("\n  Shuffled IC History:")
        for i, ep in enumerate(ic_tracker.history["epoch"]):
            c_ic = ic_tracker.history["contiguous_ic"][i]
            s_ic = ic_tracker.history["shuffled_ic"][i]
            gap = ic_tracker.history["ic_gap"][i]
            print(f"    Epoch {ep+1:3d}: Contiguous={c_ic:.4f} "
                  f"Shuffled={s_ic:.4f} Gap={gap:.4f}")

    return gate_passed


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V8.1 World Model Trainer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    # Choices match settings.py get_feature_config configs dict (no named SUPPORTED_FEATURE_COUNTS
    # in this sub-version settings; hardcoded list is kept in sync with that dict).
    parser.add_argument("--features", type=int,
                        choices=[13, 18, 30, 34, 37, 41, 127, 133, 154, 161], default=37,
                        help="Feature count: 13/18/30/37 (default: 37 = 30 base + 7 XD)")
    parser.add_argument("--revin", action="store_true", default=False,
                        help="Enable RevIN normalization (disabled by default, causes memorization)")
    parser.add_argument("--loss-type", type=str, choices=["ce", "crps"], default="ce",
                        help="Return loss type: ce (cross-entropy, default) or crps (ordinal-aware CRPS)")
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    import settings
    settings.RETURN_LOSS_TYPE = args.loss_type
    success = train_world_model(n_features=args.features, use_revin=args.revin)
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
