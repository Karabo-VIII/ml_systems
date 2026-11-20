"""
V1.4 World Model Trainer -- Transformer-RSSM Multi-Horizon (Anti-Fragile Edition)

Based on V1/TMP_2 proven trainer (ShIC=0.0305, GATE PASS, 105 epochs).
Extended to 18 features with RevIN and verbose logging.
V1.4 adds FeatureAttentionBlock (iTransformer-style cross-feature attention).

Architecture: Causal Transformer + RSSM Latents
Version: v1_4_transformer_rssm_antifragile

Anti-fragile training features:
  - Walk-forward cross-validation with purge gap (eliminates temporal leakage)
  - Shuffled IC as primary model selection metric (detects memorization)
  - Rich augmentation: noise, feature dropout, temporal jitter, mixup
  - Regime-balanced sampling (prevents neutral-bias)
  - Overfitting monitor (IC gap detection with auto-stop)

Production features:
  - Full resumability: persists epoch, best_val_loss, patience, gate, shuffled IC
  - Windows-safe DataLoader (num_workers=0 on Windows)
  - LR schedule: linear warmup then cosine decay
  - EMA model tracking (decay=0.995)
  - Gradient norm logging and NaN detection
  - RevIN disabled by default (causes memorization; enable with --revin)
  - Label smoothing (0.05) for TwoHot cross-entropy
  - ShIC-based early stopping (consecutive decline detection with noise threshold)

Usage:
    python train_world_model.py                    # 18 features, no RevIN (default)
    python train_world_model.py --revin            # 18 features + RevIN (causes memorization)
    python train_world_model.py --seed 42          # Reproducible run
    python train_world_model.py --features 13      # 13 base features only
"""
import os
# OOM mitigation (2026-04-30): cap CUDA caching allocator splits at 128MB
# to prevent long-run fragmentation (V1.0 ep99 OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import torch.optim as optim
import numpy as np
import sys
import copy
import gc
import math
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import TransformerWorldModel, count_parameters
from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, OverfitMonitor, AntifragileDataset,
    compute_regime_weights,
    make_predict_fn, print_antifragile_header,
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

from log_utils import setup_logging, teardown_logging
from revin import RevIN
from diagnostics.feature_autopsy import FeatureAutopsy


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
    # Include precomputed regime labels if available
    if "regime_label" in batch[0][1]:
        targets["regime_label"] = torch.stack([b[1]["regime_label"] for b in batch])
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
    metrics = {"rec": [], "kl": [], "kl_raw": [], "regime": [], "regime_acc": [], "pairwise": [], "total": []}
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

    result["ic_mean"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)
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


# ==============================================================================
# MAIN TRAINING LOOP (Anti-Fragile)
# ==============================================================================

def train_world_model(use_revin: bool = False, n_features: int = 18, args=None):
    """
    Anti-fragile training loop with walk-forward validation, shuffled IC
    monitoring, rich augmentation, regime-balanced sampling, and overfitting
    detection.

    Args:
        use_revin: If True, apply RevIN normalization (causes memorization;
                   checkpoints saved as v1_4_*_revin_wm_*). Default: False.
        n_features: Feature count: 13 (base only) or 18 (base + cross-asset).
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)

    # Run-tag suffix isolates parallel-batch variants' checkpoints + logs.
    # Without it, multi-variant batches via run_flag_batch.py clobber each other.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""

    # Checkpoint prefix: "v1_4_f18" or "v1_4_f13" (no suffix = no RevIN)
    feat_tag = f"f{n_features}"
    revin_tag = "_revin" if use_revin else ""
    ckpt_prefix = f"v1_4_{feat_tag}{revin_tag}{run_tag_str}"
    log_suffix = f"v1_4_{feat_tag}{revin_tag}{run_tag_str}_train"
    log_path = setup_logging(LOG_DIR, log_suffix)
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V1.4 WORLD MODEL TRAINER (Transformer-RSSM | Anti-Fragile)")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:     {n_features} ({len(feature_list)} in feature_list)")
    print(f"  FeatureAttn:  d_feat={FEAT_ATTN_D_FEAT}, heads={FEAT_ATTN_N_HEADS}")
    print(f"  Horizons:     {REWARD_HORIZONS}")
    print(f"  Architecture: d_model={WM_D_MODEL}, layers={WM_N_LAYERS}, "
          f"heads={WM_N_HEADS}, d_ff={WM_D_FF}")
    print(f"  RSSM:         {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  Regularization: dropout={WM_DROPOUT}, weight_decay={WM_WEIGHT_DECAY}")
    print(f"  Augmentation: noise={AUG_NOISE_STD}, feat_drop={AUG_FEAT_DROP}")
    print(f"  LR Schedule:  warmup={WM_WARMUP_EPOCHS}ep, peak={WM_LR}, min={WM_MIN_LR}")
    print(f"  KL:           free-nats max formulation (always >= {WM_FREE_NATS})")
    print(f"  RevIN:        {'enabled' if use_revin else 'DISABLED'} ({input_dim} features)")
    print(f"  Ckpt prefix:  {ckpt_prefix}")
    print_antifragile_header("V1.4", af_config)

    # -- Load Full Data (no pre-split) -----------------------------------------
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
    model = TransformerWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)

    # -- Frontier-ML model-side upgrades ---------------------------------------
    upgrade_flags_active = (args is not None) and (
        getattr(args, "mtp", False)
        or getattr(args, "adaptive_bins", False)
        or getattr(args, "mdn", False)
    )
    if upgrade_flags_active:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.integration import apply_v1_upgrades
        apply_v1_upgrades(
            model,
            use_mtp=getattr(args, "mtp", False),
            use_adaptive_bins=getattr(args, "adaptive_bins", False),
            adaptive_bins_mode=getattr(args, "adaptive_bins_mode", "log_spaced"),
            use_mdn=getattr(args, "mdn", False),
            mdn_mode=getattr(args, "mdn_mode", "normal"),
            mdn_components=getattr(args, "mdn_components", 3),
            verbose=True,
        )

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    print(f"  Parameters: {count_parameters(model):,}")

    # -- RevIN (distribution shift normalization) --------------------------------
    revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None

    # -- Optimizer (includes RevIN affine params when enabled) -------------------
    all_params = list(model.parameters())
    if revin is not None:
        all_params += list(revin.parameters())
    optimizer = optim.AdamW(
        all_params,
        lr=WM_LR,
        weight_decay=WM_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # -- Frontier-ML upgrade context (SAM/FrAug/PCGrad/label-noise/logit-clip) -
    upgrade_ctx = None
    if args is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.trainer_helpers import (
            build_upgrade_context, install_logit_clip_hooks,
        )
        upgrade_ctx = build_upgrade_context(
            model, optimizer, args, device=DEVICE,
            grad_clip=WM_GRAD_CLIP, horizons=tuple(REWARD_HORIZONS),
            verbose=True,
        )
        install_logit_clip_hooks(upgrade_ctx, model, verbose=True)
        optimizer = upgrade_ctx.optimizer
    use_amp = upgrade_ctx.use_amp if upgrade_ctx is not None else True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -- Anti-Fragile Components -----------------------------------------------
    augmentor = AntifragileAugmentor(af_config)
    ic_tracker = ShuffledICTracker(af_config)
    overfit_monitor = OverfitMonitor(af_config)
    device_obj = torch.device(DEVICE)
    predict_fn = make_predict_fn(WM_SEQ_LEN, device_obj, model_type="rssm", revin=revin)
    best_shuffled_ic = -float("inf")
    latest_shuffled_ic = None
    shic_decline_count = 0

    # -- Checkpoint: load if exists --------------------------------------------
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    gate_passed = False

    ckpt_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"
    if ckpt_path.exists():
        print(f"\n  [RESUME] Loading from {ckpt_path.name}")
        try:
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            # load_latest_collision: hard-fail if ckpt n_features != current
            # (per CLAUDE.md Code Change Verification #11)
            ckpt_n_feat = ckpt.get("n_features")
            if ckpt_n_feat is not None and ckpt_n_feat != INPUT_DIM:
                raise RuntimeError(
                    f"load_latest_collision: checkpoint trained at n_features="
                    f"{ckpt_n_feat} but trainer is INPUT_DIM={INPUT_DIM}. "
                    f"Delete {ckpt_path.name} or rerun with matching --features."
                )
            _missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if _missing:
                print(f"    [INFO] {len(_missing)} new keys (random init): {_missing[:3]}{'...' if len(_missing) > 3 else ''}")
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            patience_counter = ckpt.get("patience_counter", 0)
            gate_passed = ckpt.get("gate_passed", False)
            best_shuffled_ic = ckpt.get("best_shuffled_ic", -float("inf"))
            shic_decline_count = ckpt.get("shic_decline_count", 0)

            if "ema_state_dict" in ckpt:
                ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
            else:
                ema_model.load_state_dict(model.state_dict())

            if revin is not None and "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])

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

    # -- Feature Autopsy (non-console diagnostics) ----------------------------
    autopsy_path = LOG_DIR / f"{ckpt_prefix}_autopsy_{log_path.stem.split('_train_')[-1]}.jsonl"
    autopsy = FeatureAutopsy(
        feature_list=feature_list,
        base_dim=base_dim,
        log_path=autopsy_path,
        horizons=REWARD_HORIZONS,
        device=DEVICE,
    )

    print(f"\n  Starting from epoch {start_epoch}")
    print(f"  Autopsy log: {autopsy_path.name}")
    print("-" * 70)

    # ==========================================================================
    # TRAINING LOOP
    # ==========================================================================

    nan_recovery_count = 0

    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        model.train()

        # -- Set LR for this epoch ---------------------------------------------
        current_lr = get_lr_for_epoch(epoch)
        # NOTE: ShIC LR reduction REMOVED -- locks in memorized weights.
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        mask_ratio = get_mask_ratio(epoch)

        epoch_keys = ["total", "rec", "kl", "kl_raw", "regime", "regime_acc",
                      "direct_ret", "pairwise"] + [f"ret_{h}" for h in REWARD_HORIZONS]
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0

        train_iter = iter(train_loader)

        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc=f"Epoch {epoch+1:3d}", leave=False)

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

            # -- B007 E2: calibrated label-noise on regression targets ---------
            if upgrade_ctx is not None and upgrade_ctx.use_label_noise:
                regime_lbl = targets_gpu.get("regime_label")
                for _hk, _tv in list(targets_gpu.items()):
                    if _hk == "regime_label":
                        continue
                    if _tv.dtype.is_floating_point:
                        targets_gpu[_hk] = upgrade_ctx.label_noise_injector(
                            _tv, regime_label=regime_lbl,
                        )

            # -- Mixup augmentation (batch-level) ------------------------------
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # -- FrAug frequency-domain augmentation (B003 R2; opt-in) --------
            if upgrade_ctx is not None and upgrade_ctx.fraug_module is not None:
                upgrade_ctx.fraug_module.train()
                obs = upgrade_ctx.fraug_module(obs)

            # -- RevIN normalization (distribution shift) ----------------------
            if revin is not None:
                obs = revin(obs, mode='norm')

            # -- Forward pass with AMP -----------------------------------------
            need_components = upgrade_ctx is not None and upgrade_ctx.use_pcgrad
            with torch.amp.autocast("cuda", enabled=use_amp):
                if need_components:
                    loss, loss_dict, _, components = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                        regime_labels=targets_gpu.get("regime_label"),
                        return_components=True,
                    )
                else:
                    loss, loss_dict, _ = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                        regime_labels=targets_gpu.get("regime_label"),
                    )
                    components = None

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

            # -- Backward pass: route via upgrade context --------------------
            optimizer.zero_grad(set_to_none=True)
            clip_params = list(model.parameters())
            if revin is not None:
                clip_params += list(revin.parameters())

            if upgrade_ctx is not None and upgrade_ctx.use_sam:
                def _sam_recompute():
                    if need_components:
                        l2, _, _, c2 = model.get_loss(
                            obs, asset, targets_gpu, mask_ratio=mask_ratio,
                            block_mask=True,
                            regime_labels=targets_gpu.get("regime_label"),
                            return_components=True,
                        )
                        return l2, c2
                    l2, _, _ = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=mask_ratio,
                        block_mask=True,
                        regime_labels=targets_gpu.get("regime_label"),
                    )
                    return l2, None
                from frontier_ml.v1_upgrades.trainer_helpers import step_backward_and_update
                grad_norm_val = step_backward_and_update(
                    upgrade_ctx, loss, components, clip_params, scaler,
                    sam_recompute_fn=_sam_recompute,
                )
                if math.isfinite(grad_norm_val):
                    grad_norms.append(grad_norm_val)
            elif upgrade_ctx is not None and upgrade_ctx.use_pcgrad:
                from frontier_ml.v1_upgrades.trainer_helpers import step_backward_and_update
                grad_norm_val = step_backward_and_update(
                    upgrade_ctx, loss, components, clip_params, scaler,
                )
                if math.isfinite(grad_norm_val):
                    grad_norms.append(grad_norm_val)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(clip_params, WM_GRAD_CLIP)
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

        # Log effective Kendall weights (verbose: all 4 weights)
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

        regime_acc_pct = avg_stats.get('regime_acc', 0) * 100
        print(
            f"  Ep {epoch+1:3d} | "
            f"Loss: {avg_stats.get('total', 0):.4f} | "
            f"Rec: {avg_stats.get('rec', 0):.4f} | "
            f"KL: {avg_stats.get('kl', 0):.2f} raw:{avg_stats.get('kl_raw', 0):.3f} | "
            f"{ret_str} | "
            f"Reg:{avg_stats.get('regime', 0):.3f} Acc:{regime_acc_pct:.0f}% | "
            f"GN: {avg_grad_norm:.2f} | "
            f"Mask: {mask_ratio:.2f} | LR: {current_lr:.1e} | "
            f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} w_kl:{_w_kl:.2f} w_reg:{_w_reg:.1f}"
            f"{nan_str}"
        )

        # -- NaN collapse recovery ------------------------------------------------
        nan_frac = nan_count / WM_STEPS_PER_EPOCH
        if nan_frac > 0.5:
            nan_recovery_count += 1
            if nan_recovery_count > 3:
                print(f"  [FATAL] NaN collapse after {nan_recovery_count} recovery attempts. Aborting.")
                break
            new_seed = 42 + nan_recovery_count * 1000
            print(f"  [NaN RECOVERY] {nan_frac:.0%} NaN batches at epoch {epoch+1}. "
                  f"Reinitializing with seed {new_seed} (attempt {nan_recovery_count}/3)")
            torch.manual_seed(new_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(new_seed)
            for m in model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            if hasattr(model, 'log_vars'):
                model.log_vars.data.zero_()
            optimizer = optim.AdamW(
                list(model.parameters()) + (list(revin.parameters()) if revin is not None else []),
                lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                betas=(0.9, 0.95),
            )
            scaler = torch.amp.GradScaler("cuda")
            ema_model = copy.deepcopy(model)
            best_val_loss = float("inf")
            best_shuffled_ic = -float("inf")
            patience_counter = 0
            print(f"  [NaN RECOVERY] Model reinitialized. Continuing from epoch {epoch+2}.")
            continue

        # -- Memory cleanup every 10 epochs -----------------------------------
        if (epoch + 1) % 5 == 0:  # OOM fix 2026-04-30 (was 10; ep99 V1.0 OOM)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -- Validation --------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_metrics = validate(ema_model, val_loader, revin=revin)
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
                # OOM mitigation (2026-04-30): ShIC compute is the memory peak
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
                    shic_decline_count = 0
                    best_ema_ckpt = {"model_state_dict": ema_model.state_dict()}
                    if revin is not None:
                        best_ema_ckpt["revin_state_dict"] = revin.state_dict()
                    torch.save(best_ema_ckpt, BASE_MODEL_DIR / f"{ckpt_prefix}_wm_best_ema.pt")
                    print(f"  [NEW BEST SHUFFLED IC] {shuffled_ic:.4f}")
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

            # -- Gate check (with shuffled IC + train/val ratio) ----------------
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

            # -- Save checkpoint with full state -------------------------------
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
                "use_revin": use_revin,
                "n_features": n_features,
                "base_dim": base_dim,
                "shic_decline_count": shic_decline_count,
                "version": "v1_4_transformer_rssm_antifragile",
            }
            if revin is not None:
                state["revin_state_dict"] = revin.state_dict()
            torch.save(state, BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt")
            # Save standalone weights (matches best_ema format)
            weights_ckpt = {"model_state_dict": model.state_dict()}
            if revin is not None:
                weights_ckpt["revin_state_dict"] = revin.state_dict()
            torch.save(weights_ckpt, BASE_MODEL_DIR / f"{ckpt_prefix}_wm_weights.pt")

            ep_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_epoch_{epoch+1}.pt"
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

            # -- Print validation results (verbose) ----------------------------
            ic_str = " | ".join([
                f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
            ])
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            val_regime_acc = val_metrics.get('regime_acc', 0) * 100
            print(
                f"  -- VAL | "
                f"Loss: {val_loss:.4f} | "
                f"Rec: {val_metrics.get('rec', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"KL: {val_metrics.get('kl', 0):.2f} raw:{val_metrics.get('kl_raw', 0):.3f} | "
                f"Reg:{val_regime_acc:.0f}% | "
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
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} "
                      f"(patience={WM_PATIENCE} exhausted)")
                break

    # -- OOS Evaluation (post-training) ----------------------------------------
    oos_metrics = {}
    if oos_segments and gate_passed:
        print("\n  Running OOS evaluation...")
        oos_ds = AntifragileDataset(
            oos_segments,
            seq_len=WM_SEQ_LEN,
            reward_horizons=REWARD_HORIZONS,
            augment=False,
            config=af_config,
        )
        oos_loader = DataLoader(
            oos_ds,
            batch_size=WM_BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=persistent,
        )
        oos_metrics = validate(ema_model, oos_loader, revin=revin)
        oos_loss = oos_metrics.get("total", 999)
        oos_ic_str = " | ".join([
            f"IC{h}:{oos_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
        ])
        print(f"  -- OOS | Loss: {oos_loss:.4f} | {oos_ic_str}")

    # -- Final Report ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  V1.4 TRAINING COMPLETE (Anti-Fragile)")
    print("=" * 70)
    print(f"  Best Val Loss:       {best_val_loss:.4f}")
    print(f"  Best Shuffled IC:    {best_shuffled_ic:.4f}")
    if oos_metrics:
        print(f"  OOS IC (h=1):        {oos_metrics.get('ic_1', 0):.4f}")
    print(f"  Gate Status:         {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  EMA weights:         {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_best_ema.pt'}")
    print(f"  Latest checkpoint:   {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_latest.pt'}")

    if not gate_passed:
        print("\n  [WARN] World model did not pass validation gate.")
        print("  Do NOT proceed to agent training until gate criteria are met.")
    else:
        print("\n  [OK] Model ready for downstream use.")

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
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args

    parser = argparse.ArgumentParser(description="V1.4 World Model Trainer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--revin", action="store_true", help="Enable RevIN normalization (off by default; causes memorization)")
    parser.add_argument("--features", type=int, choices=sorted(SUPPORTED_FEATURE_COUNTS_V1_4), default=37,
                        help="Feature count: 13/18/21/25/30/37 (default: 37 = 30 base + 7 XD)")
    parser.add_argument("--loss-type", type=str, choices=["ce", "crps"], default="ce",
                        help="Return loss type: ce (cross-entropy, default) or crps (ordinal-aware CRPS)")
    add_upgrade_args(parser)  # +6 frontier-ML upgrade flags (default OFF)
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    import settings
    settings.RETURN_LOSS_TYPE = args.loss_type
    success = train_world_model(use_revin=args.revin, n_features=args.features, args=args)
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
