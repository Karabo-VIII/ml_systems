"""
V1.1 World Model Trainer -- Transformer-RSSM Multi-Horizon (Anti-Fragile Edition)

Based on V1/TMP_2 proven architecture (ShIC=0.0305, GATE PASS, 105 epochs).
37 features (30 base + 7 cross-asset), multi-head feature ablation support.

Architecture: Causal Transformer + RSSM Latents
Version: v1_1_transformer_rssm_antifragile

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
  - ShIC-based early stopping (consecutive decline detection with noise threshold)
  - Multi-head feature ablation (--ablation): per-subset return heads for marginal
    contribution analysis. Trains separate return MLPs per feature subset in parallel.

Usage:
    python train_world_model.py                    # 37 features (default)
    python train_world_model.py --features 13      # 13 core base features only
    python train_world_model.py --features 18      # 18 features (13 base + 5 XD)
    python train_world_model.py --features 37 --ablation  # f37 + ablation heads
    python train_world_model.py --revin            # Enable RevIN (causes memorization)
    python train_world_model.py --seed 42          # Reproducible run
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
from settings import get_feature_config, get_ablation_subsets, ABLATION_LOSS_WEIGHT
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
    # Forward-regime labels (V1_FORWARD_REGIME flag; absent when flag is OFF)
    if "fwd_bear" in batch[0][1]:
        targets["forward_regime_labels"] = {
            "bear":  torch.stack([b[1]["fwd_bear"]  for b in batch]),
            "trend": torch.stack([b[1]["fwd_trend"] for b in batch]),
            "move":  torch.stack([b[1]["fwd_move"]  for b in batch]),
        }
    return obs, targets, asset


def _targets_to_device(targets, device):
    """Move a targets dict to `device`, recursing one level into nested label dicts.

    Plain tensors -> .to(device). A nested dict (the V1_FORWARD_REGIME
    `forward_regime_labels` = {"bear","trend","move"}) is recursed one level so its
    inner tensors land on GPU (forward_regime_aux_loss runs F.cross_entropy on GPU
    logits and needs the labels there too). When V1_FORWARD_REGIME is OFF there is NO
    nested dict, so this behaves identically to the old flat comprehension.
    Mirrors V13/V23 `_targets_to_device`.
    """
    out = {}
    for k, v in targets.items():
        if isinstance(v, dict):
            out[k] = {ik: iv.to(device, non_blocking=True) for ik, iv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


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
    """Validate model on the FULL holdout set with per-horizon IC.

    If the model has ablation heads, also computes per-head IC for each
    feature subset.
    """
    model.eval()
    metrics = {"rec": [], "kl": [], "kl_raw": [], "regime": [], "regime_acc": [], "total": [], "direct_ret": [], "pairwise": []}
    for h in REWARD_HORIZONS:
        metrics[f"ret_{h}"] = []

    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    # Ablation IC tracking
    has_ablation = bool(getattr(model, 'ablation_subsets', None))
    abl_ic_data = {}
    if has_ablation:
        for name in model.ablation_subsets:
            abl_ic_data[name] = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = _targets_to_device(targets, DEVICE)

        if revin is not None:
            obs = revin(obs, mode='norm')

        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(obs, asset, targets_gpu, mask_ratio=0.0, regime_labels=targets_gpu.get("regime_label"))

            # Ablation forward (inside autocast for AMP consistency)
            if has_ablation:
                abl_results = model.ablation_forward(obs, asset, targets_gpu)

        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])

        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(outputs["return_logits"][h])
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

        # Collect ablation predictions
        if has_ablation:
            for name, abl in abl_results.items():
                for h in REWARD_HORIZONS:
                    if h in abl["return_logits"]:
                        pred_abl = model.bucketer.decode(abl["return_logits"][h])
                        abl_ic_data[name][h]["preds"].append(pred_abl.cpu().numpy().flatten())
                        abl_ic_data[name][h]["reals"].append(targets[h].cpu().numpy().flatten())

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

    # Compute ablation per-head ICs
    if has_ablation:
        for name in model.ablation_subsets:
            for h in REWARD_HORIZONS:
                if abl_ic_data[name][h]["preds"]:
                    all_p = np.concatenate(abl_ic_data[name][h]["preds"])
                    all_r = np.concatenate(abl_ic_data[name][h]["reals"])
                    m = np.isfinite(all_p) & np.isfinite(all_r)
                    if m.sum() > 100:
                        result[f"abl_{name}_ic_{h}"] = float(np.corrcoef(all_p[m], all_r[m])[0, 1])
                    else:
                        result[f"abl_{name}_ic_{h}"] = 0.0
                else:
                    result[f"abl_{name}_ic_{h}"] = 0.0
            result[f"abl_{name}_ic"] = float(np.mean(
                [result.get(f"abl_{name}_ic_{h}", 0) for h in REWARD_HORIZONS]
            ))

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

def train_world_model(use_revin: bool = False, n_features: int = 18,
                      use_ablation: bool = False,
                      use_sam: bool = False,
                      use_fraug: bool = False,
                      use_pcgrad: bool = False,
                      use_mtp: bool = False,
                      use_headline: bool = False,
                      use_adaptive_bins: bool = False,
                      use_mdn: bool = False,
                      use_label_noise: bool = False,
                      use_logit_clip: bool = False,
                      sam_rho: float = 0.7,
                      fraug_mask_ratio: float = 0.10,
                      fraug_p: float = 0.5,
                      adaptive_bins_mode: str = "log_spaced",
                      mdn_mode: str = "normal",
                      mdn_components: int = 3,
                      label_noise_ratio: float = 0.5,
                      label_noise_sigma_residual: float = 0.02,
                      logit_clip_tau: float = 4.0,
                      run_tag: str = ""):
    """
    Anti-fragile training loop with walk-forward validation, shuffled IC
    monitoring, rich augmentation, regime-balanced sampling, and overfitting
    detection.

    Args:
        use_revin:    If True, apply RevIN normalization (causes memorization;
                      checkpoints saved as v1_1_*_revin_wm_*). Default: False.
        n_features:   Number of features to use: 13/17/18/20/22/25.
        use_ablation: If True, train with multi-head feature ablation.
                      Creates separate return heads for each feature subset
                      smaller than n_features. Enables per-head IC tracking.
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    feat_tag = f"f{n_features}"
    revin_tag = "_revin" if use_revin else ""
    abl_tag = "_abl" if use_ablation else ""
    # Run-tag isolates checkpoint paths for parallel-batch info-first experimentation.
    # Each variant in a flag-iteration batch should pass a distinct --run-tag (e.g.
    # "sam", "mtp", "mdn_skewt") so checkpoints DO NOT clobber each other.
    run_tag_str = f"_{run_tag}" if run_tag else ""
    ckpt_prefix = f"v1_1_{feat_tag}{revin_tag}{abl_tag}{run_tag_str}"

    # Build ablation subsets: layout-aware (non-contiguous f18/f22 handled correctly)
    ablation_subsets = None
    if use_ablation:
        ablation_subsets = get_ablation_subsets(n_features)
        if not ablation_subsets:
            print(f"  [WARN] No valid ablation subsets for {n_features} features. "
                  f"Disabling ablation.")
            use_ablation = False
            ablation_subsets = None
    log_suffix = f"v1_1_{feat_tag}{revin_tag}{abl_tag}{run_tag_str}_train"
    log_path = setup_logging(LOG_DIR, log_suffix)
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V1.1 WORLD MODEL TRAINER (Transformer-RSSM | Anti-Fragile)")
    print("=" * 70)
    print(f"  Device:       {DEVICE}")
    print(f"  Platform:     {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Features:     {n_features} ({len(feature_list)} in feature_list)")
    print(f"  Horizons:     {REWARD_HORIZONS} (active: {ACTIVE_HORIZONS})")
    print(f"  Architecture: d_model={WM_D_MODEL}, layers={WM_N_LAYERS}, "
          f"heads={WM_N_HEADS}, d_ff={WM_D_FF}")
    print(f"  RSSM:         {RSSM_LATENT_DIM}x{RSSM_CLASSES} = {FLAT_DIM}")
    print(f"  Regularization: dropout={WM_DROPOUT}, weight_decay={WM_WEIGHT_DECAY}")
    print(f"  Augmentation: noise={AUG_NOISE_STD}, feat_drop={AUG_FEAT_DROP}")
    print(f"  LR Schedule:  warmup={WM_WARMUP_EPOCHS}ep, peak={WM_LR}, min={WM_MIN_LR}")
    print(f"  KL:           free-nats max formulation (always >= {WM_FREE_NATS})")
    print(f"  RevIN:        {'enabled' if use_revin else 'DISABLED'} ({input_dim} features)")
    if use_ablation:
        abl_names = list(ablation_subsets.keys())
        print(f"  Ablation:     ENABLED ({len(abl_names)} heads: {abl_names})")
        print(f"  Abl weight:   {ABLATION_LOSS_WEIGHT} (non-primary head loss multiplier)")
    else:
        print(f"  Ablation:     disabled")
    print(f"  Ckpt prefix:  {ckpt_prefix}")
    print_antifragile_header("V1.1", af_config)

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
    model = TransformerWorldModel(
        input_dim=input_dim, base_dim=base_dim,
        ablation_subsets=ablation_subsets,
    ).to(DEVICE)

    # -- Frontier-ML model-side upgrades (B002 R1 MTP, B001 R3 adaptive bins,
    # B003 R3 MDN) ---
    # Applied BEFORE EMA copy so EMA inherits the upgraded structure.
    if use_mtp or use_adaptive_bins or use_mdn:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.integration import apply_v1_upgrades
        apply_v1_upgrades(
            model,
            use_mtp=use_mtp,
            use_adaptive_bins=use_adaptive_bins,
            adaptive_bins_mode=adaptive_bins_mode,
            use_mdn=use_mdn, mdn_mode=mdn_mode, mdn_components=mdn_components,
            verbose=True,
        )

    # -- Headline-tier components (CC-H1/H2/H5/H6/H7) -- applied BEFORE EMA copy --
    if use_headline:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.headline_integration import apply_headline_upgrades
        apply_headline_upgrades(
            model, use_multires=True, use_linattn=True, use_quantile=True,
            use_regime_cond=True, use_dream=True, horizons=tuple(REWARD_HORIZONS), verbose=True,
        )

    # -- Forward-regime head (Layer-C wiring, 2026-06-10) ----------------------
    # Activated by env flag V1_FORWARD_REGIME=1 (default "0" = OFF).
    # When OFF: no attachment, no label computation, base path byte-for-byte unchanged.
    # When ON:  attach head to model + ema_model, pre-compute per-segment forward labels,
    #           pack them into targets via collate_fn -> get_loss adds 0.10 * aux_loss.
    _use_forward_regime = os.environ.get("V1_FORWARD_REGIME", "0") == "1"
    if _use_forward_regime:
        from forward_regime_head import attach_forward_regime_head as _attach_frh
        from regime_targets import (
            forward_bear_label as _fwd_bear_lbl,
            forward_trend_label as _fwd_trend_lbl,
            move_onset_label as _fwd_move_lbl,
        )
        _attach_frh(model)
        # Pre-compute forward labels per segment (train + val sets share the loop).
        # Labels are stored on the segment dicts; AntifragileDataset.__getitem__ picks
        # them up by key (fwd_bear/fwd_trend/fwd_move) just like regime_label.
        # Close proxy: cumprod(1 + target_return_1) -- sufficient for forward-label builders
        # because the labels depend only on RELATIVE price movements (ratios), not levels.
        print("  [V1_FORWARD_REGIME] Pre-computing forward labels on all segments ...")
        _all_label_segs = all_segments  # full dataset (includes train+val splits)
        for _seg in _all_label_segs:
            _ret1 = _seg.get("target_return_1")
            if _ret1 is None:
                # Fallback: skip label for this segment (head will see all-NaN -> 0 loss)
                _seg["fwd_bear"]  = np.full(len(_seg["features"]), np.nan, dtype=np.float32)
                _seg["fwd_trend"] = np.full(len(_seg["features"]), np.nan, dtype=np.float32)
                _seg["fwd_move"]  = np.full(len(_seg["features"]), np.nan, dtype=np.float32)
                continue
            # Reconstruct a strictly-positive close proxy: shift so close[0]=1, close[t+1]=close[t]*(1+ret[t])
            # ret[t] = close[t+1]/close[t]-1, so close[t] = prod_{j<t}(1+ret[j]).
            # We build N+1 points then use close[0..N-1] aligned to features.
            _cret = np.clip(_ret1.astype(np.float64), -0.99, 10.0)   # guard against -100% bars
            _close_ext = np.empty(len(_ret1) + 1, dtype=np.float64)
            _close_ext[0] = 1.0
            np.cumprod(1.0 + _cret, out=_close_ext[1:])
            # Ensure positivity (rounding / edge case guard)
            _close_ext = np.maximum(_close_ext, 1e-8)
            # K=64 is the default forward horizon (matches the head's training doc)
            _seg["fwd_bear"]  = _fwd_bear_lbl(_close_ext[:-1],  K=64, dd_thresh=0.05)
            _seg["fwd_trend"] = _fwd_trend_lbl(_close_ext[:-1], K=64)
            _seg["fwd_move"]  = _fwd_move_lbl(_close_ext[:-1],  a=1,  b=64)
        print(f"  [V1_FORWARD_REGIME] Labels ready on {len(_all_label_segs)} segments "
              f"(K=64, bear/trend/move, NaN tail masked in loss)")
        del _all_label_segs  # let GC reclaim the reference

    # -- VSN flag report (V1_VSN, combinable with V1_FORWARD_REGIME) ----------
    # Model was already built with vsn wired in (env var read at __init__ time).
    # We just confirm the flag here so the training log is self-describing.
    _use_vsn = os.environ.get("V1_VSN", "0") == "1"
    if _use_vsn:
        vsn_params = sum(p.numel() for p in model.vsn.parameters()) if model.vsn is not None else 0
        print(f"  [V1_VSN] Variable Selection Network ENABLED  "
              f"({vsn_params} params, gate [B,T,{input_dim}] sigmoid, causal per-timestep)")
        print(f"          Inspect gate weights post-train: model.get_vsn_weights(obs_seq).mean((0,1))")
    else:
        print(f"  [V1_VSN] VSN disabled (default). Set V1_VSN=1 to enable.")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    print(f"  Parameters: {count_parameters(model):,}")

    # -- torch.compile DISABLED for V1.1 -----------------------------------------
    # V1.1 f13 NaN collapse under torch.compile (4/4 runs, epochs 3-5).
    # V1.0 f13 with torch.compile works fine. Root cause: compile interaction
    # with V1.1's base_dim/input_dim branching. Eager mode is stable.
    print("  [INFO] torch.compile disabled (V1.1 stability fix)")

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

    # -- Frontier-ML upgrade hooks (B002 + B003) ------------------------------
    # Each is opt-in via flag; defaults are OFF so baseline behavior preserved.
    fraug_module = None
    if use_fraug:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.fraug import FrAug
        fraug_module = FrAug(mask_ratio=fraug_mask_ratio, mode="random",
                              p_aug=fraug_p).to(DEVICE)
        print(f"  [B003 R2] FrAug ENABLED  mask_ratio={fraug_mask_ratio} p_aug={fraug_p}")
    use_amp = True
    if use_sam:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.sam import SAM
        optimizer = SAM(all_params, optimizer, rho=sam_rho)
        # AMP+SAM coexistence is delicate; first revision disables AMP under SAM
        # (trades the AMP 2x speedup but SAM already doubles wallclock; net 4x).
        # Future optimization can re-enable AMP under a careful unscale_+second_step
        # pattern. For now, eager fp32 keeps the math clean.
        use_amp = False
        print(f"  [B003 R1] SAM ENABLED  rho={sam_rho}  (AMP disabled for first revision)")
    pcgrad_module = None
    if use_pcgrad:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.pcgrad import PCGrad
        pcgrad_module = PCGrad()
        print(f"  [B003 4.6] PCGrad ENABLED  -- gradient surgery across "
              f"[aux, ret_1, ret_4, ret_16, ret_64]")
    if use_mtp:
        # MTP swap was performed by apply_v1_upgrades above; just confirm.
        if getattr(model, "_use_mtp", False):
            print(f"  [B002 R1] MTP head ACTIVE on model (sequential causal-chain)")
        else:
            print(f"  [B002 R1] WARN: --mtp flag set but model._use_mtp is False; "
                  f"apply_v1_upgrades may have failed silently. Check logs above.")
    if use_adaptive_bins:
        if hasattr(model, "_original_bucketer"):
            print(f"  [B001 R3] AdaptiveBucketer ACTIVE on model "
                  f"({model.bucketer.num_bins} bins, {adaptive_bins_mode})")
        else:
            print(f"  [B001 R3] WARN: --adaptive-bins flag set but model.bucketer "
                  f"appears unchanged. Check logs above.")

    # B007 E2: calibrated label-noise injector (train-only target perturbation)
    label_noise_injector = None
    if use_label_noise:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.label_noise import LabelNoiseInjector
        label_noise_injector = LabelNoiseInjector(
            sigma_residual=float(label_noise_sigma_residual),
            noise_ratio=float(label_noise_ratio),
        )
        print(f"  [B007 E2] LabelNoise ENABLED  sigma_label="
              f"{label_noise_injector.sigma_label:.5f} "
              f"(ratio={label_noise_ratio} * res_std={label_noise_sigma_residual})")

    # B007 §5.2: LogitClip via forward hook on the model's bin-head logits.
    logit_clip_module = None
    if use_logit_clip:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from frontier_ml.v1_upgrades.logit_clip import LogitClip
        logit_clip_module = LogitClip(tau=float(logit_clip_tau)).to(DEVICE)

        def _logit_clip_hook(_mod, _inp, out):
            if model.training and logit_clip_module is not None:
                return logit_clip_module(out)
            return out
        # Attach hook to each per-horizon return head (skips when MTP/MDN replaces them)
        if hasattr(model, "return_heads") and isinstance(model.return_heads, torch.nn.ModuleDict):
            for h_key, head in model.return_heads.items():
                head.register_forward_hook(_logit_clip_hook)
            print(f"  [B007 S5.2] LogitClip ENABLED  tau={logit_clip_tau} "
                  f"on {len(model.return_heads)} return heads")
        else:
            print(f"  [B007 S5.2] WARN: --logit-clip set but model.return_heads is not a "
                  f"ModuleDict (likely MTP or MDN active). LogitClip skipped to avoid "
                  f"double-clip on alternative heads.")

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

            # -- Checkpoint collision guard (load_latest_collision per CLAUDE.md §11) --
            ckpt_n_feat = ckpt.get("n_features")
            ckpt_ablation = ckpt.get("use_ablation", False)
            if ckpt_n_feat is not None and ckpt_n_feat != n_features:
                raise RuntimeError(
                    f"Checkpoint was trained with --features {ckpt_n_feat} but "
                    f"current invocation uses --features {n_features}. "
                    f"Delete {ckpt_path.name} or use matching --features."
                )
            if ckpt_ablation != use_ablation:
                mode_ckpt = "ablation" if ckpt_ablation else "standalone"
                mode_now = "ablation" if use_ablation else "standalone"
                raise RuntimeError(
                    f"Checkpoint was trained in {mode_ckpt} mode but "
                    f"current invocation is {mode_now} mode. "
                    f"These produce different model architectures. "
                    f"Delete {ckpt_path.name} or use matching --ablation flag."
                )

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
    elif use_ablation:
        # -- Warm-start ablation from standalone checkpoint --
        # If no ablation checkpoint exists but a standalone one does,
        # load the shared encoder/RSSM weights (ablation heads get random init).
        standalone_prefix = f"v1_1_{feat_tag}{revin_tag}"
        standalone_ckpt = BASE_MODEL_DIR / f"{standalone_prefix}_wm_latest.pt"
        if standalone_ckpt.exists():
            print(f"\n  [WARM-START] Loading shared weights from {standalone_ckpt.name}")
            try:
                ckpt = torch.load(standalone_ckpt, map_location=DEVICE, weights_only=False)
                _missing, _ = model.load_state_dict(ckpt["model_state_dict"], strict=False)
                abl_keys = [k for k in _missing if "ablation" in k]
                other_keys = [k for k in _missing if "ablation" not in k]
                print(f"    Ablation heads (random init): {len(abl_keys)} keys")
                if other_keys:
                    print(f"    [WARN] {len(other_keys)} non-ablation keys missing: {other_keys[:3]}")
                ema_model.load_state_dict(model.state_dict())
                print(f"    Encoder warm-started from standalone f{n_features} checkpoint. "
                      f"Training from epoch 0.")
            except Exception as e:
                print(f"  [WARN] Warm-start failed: {e}. Starting fresh.")

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
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        mask_ratio = get_mask_ratio(epoch)

        epoch_keys = ["total", "rec", "kl", "kl_raw", "regime", "regime_acc",
                      "direct_ret"] + [f"ret_{h}" for h in REWARD_HORIZONS]
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
            targets_gpu = _targets_to_device(targets, DEVICE)

            # -- B007 E2: calibrated label-noise on regression targets ---------
            # Applied BEFORE mixup so mixup mixes already-noisified targets.
            # regime_label is integer-typed and skipped (no noise on classification).
            # forward_regime_labels is a nested dict of per-bar labels -> skipped
            # (noise applies to the return targets only; cannot .dtype a dict).
            if label_noise_injector is not None:
                regime_lbl = targets_gpu.get("regime_label")
                for h_key, t_val in list(targets_gpu.items()):
                    if h_key == "regime_label" or isinstance(t_val, dict):
                        continue
                    if t_val.dtype.is_floating_point:
                        targets_gpu[h_key] = label_noise_injector(t_val, regime_label=regime_lbl)

            # -- Mixup augmentation (batch-level) ------------------------------
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # -- FrAug frequency-domain augmentation (B003 R2; opt-in) ---------
            if fraug_module is not None:
                fraug_module.train()
                obs = fraug_module(obs)

            # -- RevIN normalization (distribution shift) ----------------------
            if revin is not None:
                obs = revin(obs, mode='norm')

            # -- Forward pass with AMP -----------------------------------------
            # Decide whether we need per-component split (PCGrad) or just total.
            need_components = use_pcgrad
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

                # -- Ablation head losses (feature subset comparison) ----------
                if use_ablation:
                    abl_results = model.ablation_forward(obs, asset, targets_gpu)
                    for abl_name, abl in abl_results.items():
                        loss = loss + ABLATION_LOSS_WEIGHT * abl["loss"]
                        for ah, av in abl["losses"].items():
                            loss_dict[f"abl_{abl_name}_r{ah}"] = av
                        # Ablation losses fold into the aux task for PCGrad
                        if components is not None:
                            components["aux"] = components["aux"] + ABLATION_LOSS_WEIGHT * abl["loss"]

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
            clip_params = list(model.parameters())
            if revin is not None:
                clip_params += list(revin.parameters())

            if use_sam:
                # SAM two-step pattern (eager fp32 per upgrade-hook block above).
                # Step 1: backward at w; ascend to w + epsilon.
                if pcgrad_module is not None:
                    pc_losses = [components["aux"]] + [components[f"ret_{h}"] for h in REWARD_HORIZONS]
                    pcgrad_module.pc_backward(pc_losses, model)
                else:
                    loss.backward()
                grad_norm = clip_grad_norm_(clip_params, WM_GRAD_CLIP)
                if math.isfinite(grad_norm.item()):
                    grad_norms.append(grad_norm.item())
                optimizer.first_step(zero_grad=True)
                # Step 2: recompute loss at w + epsilon; backward; descend.
                if need_components:
                    loss2, _, _, comp2 = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                        regime_labels=targets_gpu.get("regime_label"),
                        return_components=True,
                    )
                else:
                    loss2, _, _ = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                        regime_labels=targets_gpu.get("regime_label"),
                    )
                    comp2 = None
                if use_ablation:
                    abl_results2 = model.ablation_forward(obs, asset, targets_gpu)
                    for abl_name2, abl2 in abl_results2.items():
                        loss2 = loss2 + ABLATION_LOSS_WEIGHT * abl2["loss"]
                        if comp2 is not None:
                            comp2["aux"] = comp2["aux"] + ABLATION_LOSS_WEIGHT * abl2["loss"]
                if pcgrad_module is not None:
                    pc_losses2 = [comp2["aux"]] + [comp2[f"ret_{h}"] for h in REWARD_HORIZONS]
                    pcgrad_module.pc_backward(pc_losses2, model)
                else:
                    loss2.backward()
                clip_grad_norm_(clip_params, WM_GRAD_CLIP)
                optimizer.second_step(zero_grad=False)
            else:
                # Standard path (AMP unless --pcgrad disables it for grad surgery).
                if pcgrad_module is not None:
                    # PCGrad needs separate per-task backwards; AMP scaler doesn't
                    # play well with multi-backward. Disable AMP under --pcgrad.
                    pc_losses = [components["aux"]] + [components[f"ret_{h}"] for h in REWARD_HORIZONS]
                    pcgrad_module.pc_backward(pc_losses, model)
                    grad_norm = clip_grad_norm_(clip_params, WM_GRAD_CLIP)
                    if math.isfinite(grad_norm.item()):
                        grad_norms.append(grad_norm.item())
                    optimizer.step()
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
        # If >50% of batches in an epoch produced NaN, the model is dying.
        # Reinitialize weights and continue — effectively restart with new seed.
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
            # Reinitialize model weights
            for m in model.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            # Reset log_vars (Kendall weights)
            if hasattr(model, 'log_vars'):
                model.log_vars.data.zero_()
            # Reset optimizer and scaler state
            all_params = list(model.parameters())
            if revin is not None:
                all_params += list(revin.parameters())
            optimizer = torch.optim.AdamW(
                all_params,
                lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
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
                "shic_decline_count": shic_decline_count,
                "use_revin": use_revin,
                "n_features": n_features,
                "base_dim": base_dim,
                "version": "v1_1_transformer_rssm_antifragile",
                "use_ablation": use_ablation,
                "ablation_subsets": {k: v for k, v in (ablation_subsets or {}).items()},
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

            # -- Ablation per-head IC summary ----------------------------------
            if use_ablation:
                abl_parts = []
                for abl_name in ablation_subsets:
                    abl_ic = val_metrics.get(f"abl_{abl_name}_ic", 0)
                    abl_ic1 = val_metrics.get(f"abl_{abl_name}_ic_1", 0)
                    abl_parts.append(f"{abl_name}:IC1={abl_ic1:.4f}/avg={abl_ic:.4f}")
                print(f"  -- ABL | {' | '.join(abl_parts)}")

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
        oos_loss = oos_metrics.get("total", float("nan"))
        oos_ic_str = " | ".join([
            f"IC{h}:{oos_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS
        ])
        print(f"  -- OOS | Loss: {oos_loss:.4f} | {oos_ic_str}")

    # -- Final Report ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE (Anti-Fragile)")
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
    parser = argparse.ArgumentParser(description="V1.1 World Model Trainer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--revin", action="store_true", help="Enable RevIN normalization (off by default; causes memorization)")
    parser.add_argument("--features", type=int,
                        choices=sorted(SUPPORTED_FEATURE_COUNTS_V1_1), default=37,
                        help="Feature count. 29 = Pattern-P-cleaned (f34 minus 5 dead "
                             "features, see CLAUDE.md). Default 37 = 30 base + 7 XD.")
    parser.add_argument("--run-tag", type=str, default="",
                        help="Optional checkpoint suffix to isolate parallel-batch runs. "
                             "Example: --run-tag sam writes v1_1_f29_sam_wm_*.pt instead "
                             "of clobbering the baseline v1_1_f29_wm_*.pt. Required when "
                             "running flag-variant batches; without it variants overwrite.")
    parser.add_argument("--ablation", action="store_true",
                        help="Enable multi-head feature ablation (trains separate return heads "
                             "per feature subset for marginal contribution analysis)")
    parser.add_argument("--loss-type", type=str, choices=["ce", "crps"], default="ce",
                        help="Return loss type: ce (cross-entropy, default) or crps (ordinal-aware CRPS)")
    # Frontier-ML upgrade flags (default OFF; baseline behavior preserved)
    parser.add_argument("--sam", action="store_true",
                        help="[B003 R1] Wrap optimizer with Sharpness-Aware Minimization. "
                             "Doubles wallclock per step; targets ShIC + IC simultaneously.")
    parser.add_argument("--sam-rho", type=float, default=0.7,
                        help="SAM perturbation radius. Default 0.7 per SAMformer "
                             "official run.py (VERIFIED 2026-05-02 via "
                             "github.com/romilbert/samformer). Foret 2020 used 0.05 "
                             "for vision; 0.7 is the time-series-validated value.")
    parser.add_argument("--fraug", action="store_true",
                        help="[B003 R2] Frequency-domain augmentation (FFT mask). "
                             "~0 marginal cost; targets ShIC.")
    parser.add_argument("--fraug-mask-ratio", type=float, default=0.10,
                        help="Fraction of frequency components to mask (default 0.10).")
    parser.add_argument("--fraug-p", type=float, default=0.5,
                        help="Probability of applying FrAug per batch (default 0.5).")
    parser.add_argument("--pcgrad", action="store_true",
                        help="[B003 4.6] PCGrad gradient surgery across "
                             "[aux, ret_1, ret_4, ret_16, ret_64]. Disables AMP "
                             "(per-task backward incompatible with scaler).")
    parser.add_argument("--mtp", action="store_true",
                        help="[B002 R1] Multi-Token Prediction sequential causal-chain head. "
                             "Replaces independent {h1,h4,h16,h64} heads with chained MTPHead.")
    parser.add_argument("--headline", action="store_true",
                        help="[Headline-tier] Wire CC-H1 multi-res + CC-H2 linear-attn (forward) + "
                             "CC-H5 quantile + CC-H6 regime-cond + CC-H7 dream (aux losses). "
                             "Targets IC>0.10/ShIC>0.05. Stable (multi-res residual); base path unchanged when off.")
    parser.add_argument("--adaptive-bins", action="store_true",
                        help="[B001 R3] Replace TwoHotSymlog 255-uniform bins with "
                             "log-spaced (default) for h=1 5-min crypto returns. "
                             "Same bin COUNT, denser PLACEMENT near zero.")
    parser.add_argument("--adaptive-bins-mode", default="log_spaced",
                        choices=["log_spaced", "quantile"],
                        help="Adaptive-bin mode: log_spaced (default) or quantile (needs train data).")
    parser.add_argument("--mdn", action="store_true",
                        help="[B003 R3] Replace TwoHot bin heads with K-component MDN "
                             "(mutually exclusive with --mtp). Module-attached but "
                             "end-to-end loss-path wiring is a known follow-up; warn message issued.")
    parser.add_argument("--mdn-mode", default="normal", choices=["normal", "skewed_t"],
                        help="MDN distribution: normal (Bishop) or skewed_t (LSTM-MDN paper).")
    parser.add_argument("--mdn-components", type=int, default=3,
                        help="MDN mixture components (default 3).")
    parser.add_argument("--label-noise", action="store_true",
                        help="[B007 E2] Calibrated Gaussian label noise on regression targets. "
                             "Suppresses noise memorization in low-SNR regression "
                             "(arXiv 2510.17526). ~0 marginal compute.")
    parser.add_argument("--label-noise-ratio", type=float, default=0.5,
                        help="sigma_label = ratio * sigma_residual. Default 0.5 per B007 E2.")
    parser.add_argument("--label-noise-sigma-residual", type=float, default=0.02,
                        help="Residual std proxy (default 0.02 = h=1 crypto return scale).")
    parser.add_argument("--logit-clip", action="store_true",
                        help="[B007 §5.2] LogitClip on bin head logits during training "
                             "(arXiv 2212.04055). Bounds logit-vector L2 norm.")
    parser.add_argument("--logit-clip-tau", type=float, default=4.0,
                        help="LogitClip max L2 norm (default 4.0, conservative for 255-bin TwoHot).")
    parser.add_argument("--vsn", action="store_true",
                        help="[V1_VSN] Variable Selection Network: per-timestep learnable "
                             "feature gate g_t=sigmoid(W*x_t) applied before obs_encoder. "
                             "Causal (per-timestep only). ~input_dim^2 extra params (~1681 for f41). "
                             "Equivalent to setting env var V1_VSN=1. "
                             "Combinable with --forward-regime (both ON = full world-class candidate). "
                             "OFF by default: base input path byte-for-byte unchanged.")
    args = parser.parse_args()
    # Apply --vsn flag as env var BEFORE model construction (env var is read in __init__).
    if args.vsn:
        os.environ["V1_VSN"] = "1"
    if args.seed is not None:
        set_seed(args.seed)
    # Apply loss type override from CLI
    import settings
    settings.RETURN_LOSS_TYPE = args.loss_type
    success = train_world_model(
        use_revin=args.revin, n_features=args.features,
        use_ablation=args.ablation,
        use_sam=args.sam, sam_rho=args.sam_rho,
        use_fraug=args.fraug, fraug_mask_ratio=args.fraug_mask_ratio,
        fraug_p=args.fraug_p,
        use_pcgrad=args.pcgrad, use_mtp=args.mtp,
        use_headline=args.headline,
        use_adaptive_bins=args.adaptive_bins,
        adaptive_bins_mode=args.adaptive_bins_mode,
        use_mdn=args.mdn, mdn_mode=args.mdn_mode, mdn_components=args.mdn_components,
        use_label_noise=args.label_noise,
        label_noise_ratio=args.label_noise_ratio,
        label_noise_sigma_residual=args.label_noise_sigma_residual,
        use_logit_clip=args.logit_clip,
        logit_clip_tau=args.logit_clip_tau,
        run_tag=args.run_tag,
    )
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
