"""
V6.2 World Model Trainer -- Causal JEPA + Adversarial + Anti-Memorization (Anti-Fragile Edition)

Architecture: Causal GRU JEPA + VICReg + Adversarial Time Discriminator + Feature Decoupling
Version: v6_2_causal_jepa_adversarial_antimem

Key V6.2 innovations (added vs V6):
  - Feature-decoupled decoder: reconstruction target is base features only [0:base_dim]
  - XD feature dropout (70%) + noise (0.3) during training: forces cross-sectional learning
  - --features flag: 14 (base only) or 19 (full, XD augmented)
  - Dynamic checkpoint prefix: v6_2_f13 or v6_2_f18

V6 features (preserved):
  - CausalGRU (unidirectional) prevents temporal overfitting from V2's BiGRU
  - Time Discriminator adversarially penalizes temporal dependence in latents
  - DUAL optimizer training: main encoder + separate discriminator optimizer
  - EMA target encoder updated EVERY step (same as V2)

Training loop specifics:
  - get_loss() returns 4 values: (total_loss, loss_dict, l_disc, outputs)
  - Main optimizer: all params except discriminator (AdamW, betas=(0.9, 0.95))
  - Disc optimizer: discriminator params only (AdamW, lr * DISC_LR_MULT, wd=0.0)
  - Forward -> main backward (retain_graph) + scaler_main.update() -> disc backward + scaler_disc.update()
  - EMA update after EVERY optimizer step

Production features:
  - Per-timestep InfoNCE (memory-safe [B x B])
  - VICReg variance/covariance regularization
  - Full checkpoint state for resumability (incl. disc_optimizer)
  - Windows-aware (NUM_WORKERS=0)
  - NaN detection, gradient norm logging, memory cleanup
"""
import copy
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import CausalJEPAWorldModel, count_parameters
from revin import RevIN
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
    """Curriculum masking: gradually increase difficulty."""
    if epoch >= WM_MASK_RAMP_EPOCHS:
        return WM_MASK_RATIO_END
    progress = epoch / WM_MASK_RAMP_EPOCHS
    return WM_MASK_RATIO_START + progress * (WM_MASK_RATIO_END - WM_MASK_RATIO_START)


def set_lr(optimizer, lr: float):
    """Set learning rate for all parameter groups."""
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def update_ema(model, ema_model, decay=EMA_DECAY):
    """Update exponential moving average of full model (for stable validation/ShIC)."""
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class CheckpointManager:
    """Manages model checkpoints with FULL training state for resumability.

    V6.2-specific: also saves/loads discriminator optimizer state.
    Uses dynamic prefix (e.g. v6_2_f18, v6_2_f13) for feature ablation.
    """

    def __init__(self, save_dir: Path, keep_top_k: int = 3, prefix: str = "v6_1"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.prefix = prefix
        self.history = []

    def save(self, model, optimizer, disc_optimizer, scaler_main, scaler_disc,
             epoch, val_loss, best_val_loss, patience_counter, gate_passed,
             best_shuffled_ic=-float("inf"), ema_model=None, revin=None,
             shic_decline_count=0):
        """Save complete training state for full resumability."""
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            "scaler_main_state_dict": scaler_main.state_dict(),
            "scaler_disc_state_dict": scaler_disc.state_dict(),
            "best_val_loss": best_val_loss,
            "best_shuffled_ic": best_shuffled_ic,
            "patience_counter": patience_counter,
            "gate_passed": gate_passed,
            "n_features": getattr(self, "_expected_n_features", None),
            "shic_decline_count": shic_decline_count,
            "version": "v6_2_causal_jepa_adversarial_antimem",
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

    def load_latest(self, model, optimizer, disc_optimizer, scaler_main, scaler_disc,
                    ema_model=None, revin=None):
        """Load latest checkpoint. Returns (start_epoch, best_val_loss, patience, gate, best_shuffled_ic, shic_decline_count)."""
        path = self.save_dir / f"{self.prefix}_wm_latest.pt"
        if not path.exists():
            return 0, float("inf"), 0, False, -float("inf"), 0

        print(f"  Resuming from {path.name}")
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "disc_optimizer_state_dict" in ckpt:
                disc_optimizer.load_state_dict(ckpt["disc_optimizer_state_dict"])
            # Dual scaler loading (backward compat with old single-scaler checkpoints)
            if "scaler_main_state_dict" in ckpt:
                scaler_main.load_state_dict(ckpt["scaler_main_state_dict"])
            elif "scaler_state_dict" in ckpt:
                scaler_main.load_state_dict(ckpt["scaler_state_dict"])
            if "scaler_disc_state_dict" in ckpt:
                scaler_disc.load_state_dict(ckpt["scaler_disc_state_dict"])

            if ema_model is not None:
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
                else:
                    ema_model.load_state_dict(model.state_dict())
            if revin is not None and "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])

            # -- Checkpoint collision guard --
            ckpt_n_feat = ckpt.get("n_features")
            if ckpt_n_feat is not None and hasattr(self, '_expected_n_features'):
                if ckpt_n_feat != self._expected_n_features:
                    print(f"  [WARN] Checkpoint n_features={ckpt_n_feat} != model n_features={self._expected_n_features}. Starting fresh.")
                    return 0, float("inf"), 0, False, -float("inf"), 0

            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            gate_passed = ckpt.get("gate_passed", False)
            best_shuffled_ic = ckpt.get("best_shuffled_ic", -float("inf"))
            shic_decline_count = ckpt.get("shic_decline_count", 0)

            print(f"  Resumed: epoch={start_epoch}, best_val={best_val_loss:.4f}, "
                  f"best_shIC={best_shuffled_ic:.4f}, "
                  f"patience={patience_counter}, gate={'PASS' if gate_passed else 'fail'}")
            return start_epoch, best_val_loss, patience_counter, gate_passed, best_shuffled_ic, shic_decline_count

        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            return 0, float("inf"), 0, False, -float("inf"), 0


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, revin=None):
    """V6 JEPA+Adversarial validation with embedding diversity and per-horizon IC."""
    model.eval()

    metric_keys = [
        "total", "contrastive", "contrastive_acc", "vicreg", "recon", "regime",
        "adv", "disc", "direct_ret",
    ]
    for h in REWARD_HORIZONS:
        metric_keys.append(f"ret_{h}")
    metrics = {k: [] for k in metric_keys}

    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}
    all_embeddings = []

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}
        if revin is not None:
            obs = revin(obs, mode='norm')

        with torch.amp.autocast("cuda"):
            # V6 get_loss returns 4 values: (total_loss, loss_dict, l_disc, outputs)
            _, loss_dict, _, outputs = model.get_loss(obs, asset, targets_gpu, mask_ratio=0.0, regime_labels=targets_gpu.get("regime_label"))

        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])

        # Embedding statistics (subsample every 4 timesteps for memory)
        emb = outputs["ctx_latent"][:, ::4, :]
        all_embeddings.append(emb.cpu().float().numpy())

        # Per-horizon return prediction IC
        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(outputs["return_logits"][h])
            ic_data[h]["preds"].append(pred_ret.cpu().float().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    # Aggregate metrics
    result = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    # Compute IC per horizon
    for h in REWARD_HORIZONS:
        all_preds = np.concatenate(ic_data[h]["preds"])
        all_reals = np.concatenate(ic_data[h]["reals"])
        mask = np.isfinite(all_preds) & np.isfinite(all_reals)
        if mask.sum() > 100:
            corr = np.corrcoef(all_preds[mask], all_reals[mask])[0, 1]
            result[f"ic_{h}"] = float(corr) if np.isfinite(corr) else 0.0
        else:
            result[f"ic_{h}"] = 0.0

    result["ic_mean"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)

    # Embedding health statistics
    all_emb = np.concatenate(all_embeddings, axis=0).reshape(-1, WM_D_LATENT)
    result["embed_std"] = float(np.std(all_emb))
    result["embed_mean_abs"] = float(np.mean(np.abs(all_emb)))

    # NOTE: Don't call model.train() here -- caller passes ema_model which must
    # stay in eval mode. The training model is set to train() at the top of each
    # epoch loop anyway. Matches V1/V2 pattern.
    return result


def check_gate(val_metrics: dict, shuffled_ic: float = None,
               train_loss: float = None) -> tuple:
    """V6 JEPA+Adversarial gate criteria including anti-fragile shuffled IC."""
    reasons = []

    contrastive_acc = val_metrics.get("contrastive_acc", 0)
    ic = val_metrics.get("ic", 0)
    embed_std = val_metrics.get("embed_std", 0)
    val_loss = val_metrics.get("total", 999)

    if contrastive_acc < GATE_CONTRASTIVE_MIN:
        reasons.append(f"ConAcc={contrastive_acc:.4f} < {GATE_CONTRASTIVE_MIN}")
    if ic < GATE_IC_MIN:
        reasons.append(f"IC={ic:.4f} < {GATE_IC_MIN}")
    if embed_std < GATE_EMBED_STD_MIN:
        reasons.append(f"EmbStd={embed_std:.4f} < {GATE_EMBED_STD_MIN} (collapse)")

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
    Anti-fragile training loop for V6.2 Causal JEPA + Adversarial + Anti-Memorization.

    Args:
        n_features: Number of features to use: 14 (base only) or 19 (full, XD augmented).
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    feat_tag = f"f{n_features}"
    ckpt_prefix = f"v6_2_{feat_tag}"
    log_path = setup_logging(LOG_DIR, f"v6_2_{feat_tag}_train")
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V6.2 WORLD MODEL TRAINER (Causal JEPA + Adversarial + Anti-Memorization | Anti-Fragile)")
    print("=" * 70)
    print(f"  Device:        {DEVICE}")
    print(f"  Platform:      {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:      {n_features} ({len(feature_list)} in feature_list, base_dim={base_dim})")
    print(f"  Ckpt prefix:   {ckpt_prefix}")
    print(f"  Horizons:      {REWARD_HORIZONS}")
    print(f"  Architecture:  Causal JEPA + Adversarial (d_model={WM_D_MODEL}, d_latent={WM_D_LATENT}, "
          f"n_layers={WM_N_LAYERS}, predictor={WM_PREDICTOR_LAYERS})")
    print(f"  Discriminator: hidden={DISC_HIDDEN}, lambda_adv={LAMBDA_ADV}, "
          f"lr_mult={DISC_LR_MULT}, grad_penalty={DISC_GRAD_PENALTY}")
    print(f"  Anti-Mem:      XD_DROPOUT={XD_DROPOUT_RATE}, XD_NOISE={XD_NOISE_STD}, "
          f"recon_dim={base_dim}")
    print(f"  EMA Decay:     {JEPA_EMA_DECAY}")
    print(f"  Batch size:    {WM_BATCH_SIZE}")
    print(f"  LR:            {WM_LR} (warmup={WM_WARMUP_EPOCHS}ep, cosine decay)")
    print(f"  Regularization: dropout={WM_DROPOUT}, weight_decay={WM_WEIGHT_DECAY}")
    print(f"  Augmentation:  noise_std={AUG_NOISE_STD}, feat_drop={AUG_FEAT_DROP}")
    print_antifragile_header("V6.2", af_config)

    # -- Load Full Data --------------------------------------------------------
    print("\n--- Loading full data ---")
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
    print(f"  Steps/epoch:     {WM_STEPS_PER_EPOCH}")

    train_loader = DataLoader(
        train_ds,
        batch_size=WM_BATCH_SIZE,
        sampler=train_ds.get_sampler(),
        shuffle=train_ds.get_sampler() is None,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=WM_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=False,
    )

    # -- Initialize Model ------------------------------------------------------
    model = CausalJEPAWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
    n_params = count_parameters(model)
    param_mb = n_params * 4 / 1024 / 1024
    print(f"\n  Parameters:      {n_params:,} ({param_mb:.1f} MB)")

    # -- EMA Model (for stable validation/ShIC) --------------------------------
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    # -- RevIN (Reversible Instance Normalization) ------------------------------
    revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None
    # -- Optimizers (V6.2: single optimizer, no discriminator) -----------------
    _use_disc = getattr(model, "use_discriminator", True)
    if _use_disc:
        disc_params = set(model.discriminator.parameters())
        main_params = [p for p in model.parameters() if p not in disc_params and p.requires_grad]
        disc_optimizer = optim.AdamW(
            model.discriminator.parameters(),
            lr=WM_MIN_LR * DISC_LR_MULT,
            weight_decay=0.0, betas=(0.9, 0.95),
        )
        scaler_disc = torch.amp.GradScaler("cuda")
        disc_n_params = sum(p.numel() for p in model.discriminator.parameters() if p.requires_grad)
    else:
        main_params = [p for p in model.parameters() if p.requires_grad]
        disc_optimizer = None
        scaler_disc = None
        disc_n_params = 0
    optimizer = optim.AdamW(
        list(main_params) + list(revin.parameters()),
        lr=WM_MIN_LR,
        weight_decay=WM_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )
    scaler_main = torch.amp.GradScaler("cuda")

    main_n_params = sum(p.numel() for p in main_params)
    print(f"  Main params:     {main_n_params:,}")
    print(f"  Disc params:     {disc_n_params:,}")

    # -- Anti-Fragile Components -----------------------------------------------
    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="jepa", revin=revin)
    latest_shuffled_ic = None
    shic_decline_count = 0

    # -- Checkpoint ------------------------------------------------------------
    ckpt_mgr = CheckpointManager(BASE_MODEL_DIR, keep_top_k=3, prefix=ckpt_prefix)
    ckpt_mgr._expected_n_features = n_features
    start_epoch, best_val_loss, patience_counter, gate_passed, best_shuffled_ic, shic_decline_count = \
        ckpt_mgr.load_latest(model, optimizer, disc_optimizer, scaler_main, scaler_disc,
                             ema_model=ema_model, revin=revin)


    # -- Training State --------------------------------------------------------
    print(f"\n  Starting from epoch {start_epoch}")
    print(f"  Best val loss:   {best_val_loss:.4f}")
    print(f"  Best shuffled IC: {best_shuffled_ic:.4f}")
    print(f"  Patience:        {patience_counter}/{WM_PATIENCE}")
    print(f"  Gate passed:     {gate_passed}")
    print(f"  ATME:            seq_shuffle={SEQ_SHUFFLE_PROB} (temporal_ctx_drop=N/A for JEPA)")
    print("-" * 70)

    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        epoch_start_time = time.time()
        model.train()

        # -- Set LR (warmup + cosine) ------------------------------------------
        current_lr = get_lr_for_epoch(epoch)
        set_lr(optimizer, current_lr)
        if disc_optimizer is not None:
            set_lr(disc_optimizer, current_lr * DISC_LR_MULT)

        mask_ratio = get_mask_ratio(epoch)

        epoch_keys = (
            ["total", "contrastive", "contrastive_acc", "vicreg", "recon", "regime",
             "regime_acc", "adv", "disc", "direct_ret"]
            + [f"ret_{h}" for h in REWARD_HORIZONS]
        )
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0

        train_iter = iter(train_loader)
        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc=f"Epoch {epoch+1:3d}", leave=False)

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

            # -- RevIN normalization -------------------------------------------
            obs = revin(obs, mode='norm')

            # -- Forward pass with AMP -----------------------------------------
            with torch.amp.autocast("cuda"):
                loss, loss_dict, l_disc, _ = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            # -- NaN detection -------------------------------------------------
            has_nan = False
            for k, v in loss_dict.items():
                if not np.isfinite(v):
                    has_nan = True
                    break

            if has_nan or torch.isnan(loss) or torch.isinf(loss) or loss.item() > 500:
                optimizer.zero_grad(set_to_none=True)
                if disc_optimizer is not None:
                    disc_optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                if nan_count > 20:
                    print(f"\n  [ERROR] Too many NaN batches ({nan_count}). Aborting epoch.")
                    break
                continue

            # -- Main encoder backward -----------------------------------------
            optimizer.zero_grad(set_to_none=True)
            _retain = _use_disc  # only retain graph when disc needs separate backward
            scaler_main.scale(loss).backward(retain_graph=_retain)
            scaler_main.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                main_params + list(revin.parameters()), WM_GRAD_CLIP
            )
            grad_norms.append(grad_norm.item())
            scaler_main.step(optimizer)
            scaler_main.update()

            # -- Discriminator backward (V6.2: skipped when no disc) -----------
            if disc_optimizer is not None:
                disc_optimizer.zero_grad(set_to_none=True)
                scaler_disc.scale(l_disc).backward()
                scaler_disc.unscale_(disc_optimizer)
                torch.nn.utils.clip_grad_norm_(model.discriminator.parameters(), WM_GRAD_CLIP)
                scaler_disc.step(disc_optimizer)
                scaler_disc.update()

            # -- CRITICAL: Update target encoder via EMA EVERY step ------------
            model.update_target_encoder(momentum=JEPA_EMA_DECAY)

            # -- Update full-model EMA (for stable validation/ShIC) -----------
            update_ema(model, ema_model)

            # -- Collect metrics -----------------------------------------------
            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L=f"{loss_dict['total']:.3f}",
                    Con=f"{loss_dict['contrastive']:.3f}",
                    Acc=f"{loss_dict['contrastive_acc']:.2f}",
                    adv=f"{loss_dict.get('adv', 0):.3f}",
                    disc=f"{loss_dict.get('disc', 0):.3f}",
                    r1=f"{loss_dict.get('ret_1', 0):.3f}",
                    gn=f"{grad_norm.item():.2f}",
                )

        # -- Epoch Summary -----------------------------------------------------
        epoch_time = time.time() - epoch_start_time
        avg_stats = {k: float(np.mean(v)) for k, v in epoch_stats.items() if v}
        avg_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0

        ret_str = " | ".join(
            [f"r{h}:{avg_stats.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS]
        )
        # Log effective Kendall weights (verify corridors work)
        with torch.no_grad():
            _s = model.log_vars.clamp(-6.0, 6.0)
            _s_con = _s[0].clamp(min=CONTRASTIVE_LOG_VAR_CLAMP_MIN).item()
            _s_r1 = _s[3].clamp(max=RETURN_LOG_VAR_CLAMP_MAX).item()
            _w_con = math.exp(-_s_con)
            _w_r1 = math.exp(-_s_r1)

        print(
            f"  Ep {epoch+1:3d} | "
            f"Loss:{avg_stats.get('total', 0):.4f} | "
            f"Con:{avg_stats.get('contrastive', 0):.3f} "
            f"Acc:{avg_stats.get('contrastive_acc', 0):.3f} | "
            f"VIC:{avg_stats.get('vicreg', 0):.2f} | "
            f"Rec:{avg_stats.get('recon', 0):.4f} | "
            f"{ret_str} | "
            f"Reg:{avg_stats.get('regime_acc', 0):.3f} | "
            f"adv:{avg_stats.get('adv', 0):.3f} "
            f"disc:{avg_stats.get('disc', 0):.3f} | "
            f"GN:{avg_grad_norm:.2f} | "
            f"Mask:{mask_ratio:.2f} LR:{current_lr:.1e} | "
            f"w_con:{_w_con:.2f} w_r1:{_w_r1:.1f} | "
            f"{epoch_time:.0f}s"
            + (f" [{nan_count} NaN]" if nan_count > 0 else "")
        )

        # -- Validation --------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_start = time.time()
            val_metrics = validate(ema_model, val_loader, revin=revin)
            val_time = time.time() - val_start
            val_loss = val_metrics.get("total", 999.0)
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

                # Track best shuffled IC -- save EMA model for stable inference
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
            gate_status = "GATE PASS" if passed else "gate fail"
            if latest_shuffled_ic is not None:
                gate_passed = passed  # Only meaningful after ShIC is measured

            save_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_marker = " *BEST_LOSS*"
            else:
                patience_counter += WM_VAL_EVERY

            # Save full checkpoint (incl. disc_optimizer + EMA state + RevIN)
            ckpt_mgr.save(
                model, optimizer, disc_optimizer, scaler_main, scaler_disc,
                epoch + 1, val_loss, best_val_loss,
                patience_counter, gate_passed, best_shuffled_ic,
                ema_model=ema_model, revin=revin,
                shic_decline_count=shic_decline_count,
            )

            ic_str = " | ".join(
                [f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS]
            )
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL | "
                f"Loss:{val_loss:.4f} | "
                f"ConAcc:{val_metrics.get('contrastive_acc', 0):.3f} | "
                f"VIC:{val_metrics.get('vicreg', 0):.2f} | "
                f"Rec:{val_metrics.get('recon', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"EmbStd:{val_metrics.get('embed_std', 0):.4f} | "
                f"adv:{val_metrics.get('adv', 0):.3f} "
                f"disc:{val_metrics.get('disc', 0):.3f} | "
                f"{gate_status}{save_marker} | "
                f"{val_time:.1f}s"
            )

            if not passed:
                print(f"       Reason: {reason}")

            if patience_counter >= WM_PATIENCE:
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} "
                      f"(no improvement for {WM_PATIENCE} epochs)")
                break

        # -- Memory cleanup every 10 epochs ------------------------------------
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Final Report ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE (Anti-Fragile)")
    print("=" * 70)
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Best Shuffled IC: {best_shuffled_ic:.4f}")
    print(f"  Gate Status:      {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  Weights saved:    {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_best_ema.pt'}")
    print(f"  Checkpoint:       {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_latest.pt'}")

    if not gate_passed:
        print("\n  [WARN] World model did not pass validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")
    else:
        print("\n  Model ready for agent training.")

    # -- Print shuffled IC history ---------------------------------------------
    if ic_tracker.history["epoch"]:
        print("\n  Shuffled IC History:")
        for i, ep in enumerate(ic_tracker.history["epoch"]):
            c_ic = ic_tracker.history["contiguous_ic"][i]
            s_ic = ic_tracker.history["shuffled_ic"][i]
            gap = ic_tracker.history["ic_gap"][i]
            print(f"    Epoch {ep+1:3d}: Contiguous={c_ic:.4f} "
                  f"Shuffled={s_ic:.4f} Gap={gap:.4f}")

    teardown_logging()
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
    parser = argparse.ArgumentParser(description="V6.2 World Model Trainer (Anti-Memorization)")
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
