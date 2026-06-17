"""
V8 World Model Trainer -- Neural ODE Edition (Anti-Fragile)

Architecture: Neural ODE with RK4 solver for continuous-time market dynamics
Version: v8_neural_ode_antifragile

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
import os
# OOM mitigation (2026-05-10 upgrade from 2026-04-30 max_split_size_mb:128):
# expandable_segments=True is the V25-era fix (commit 134039d). V8 was using
# the older setting; V4 hit OOM under the same; upgraded all three together.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
from diagnostics.feature_autopsy import FeatureAutopsy

# 2026-05-10 fix: add_meta_args was missing (regression from commit
# e5c17ef "--meta wired across all 14 trainers" sweep).
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
try:
    from meta_runtime import MetaRuntime, add_meta_args  # noqa: E402
except Exception:
    MetaRuntime = None
    def add_meta_args(parser):
        parser.add_argument("--meta", type=str, default="",
                             help="NO-OP for V8 (only V25 implements meta-learners)")
        parser.add_argument("--meta-distill-alpha", type=float, default=0.0,
                             help="NO-OP for V8 (paired with --meta)")


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
    # Forward-regime labels (V8_FORWARD_REGIME flag; absent when flag is OFF)
    if "fwd_bear" in batch[0][1]:
        targets["forward_regime_labels"] = {
            "bear":  torch.stack([b[1]["fwd_bear"]  for b in batch]),
            "trend": torch.stack([b[1]["fwd_trend"] for b in batch]),
            "move":  torch.stack([b[1]["fwd_move"]  for b in batch]),
        }
    return obs, targets, asset


def _targets_to_device(targets, device):
    """Move a targets dict to `device`, recursing one level into nested label dicts.

    Plain tensors -> .to(device). A nested dict (the V8_FORWARD_REGIME
    `forward_regime_labels` = {"bear","trend","move"}) is recursed one level so its
    inner tensors land on GPU. When V8_FORWARD_REGIME is OFF there is NO nested dict,
    so this behaves identically to the old flat comprehension. Mirrors V13/V23.
    """
    out = {}
    for k, v in targets.items():
        if isinstance(v, dict):
            out[k] = {ik: iv.to(device, non_blocking=True) for ik, iv in v.items()}
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


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

    def __init__(self, save_dir: Path, keep_top_k: int = 3, prefix: str = "v8"):
        self.save_dir = save_dir
        self.keep_top_k = keep_top_k
        self.prefix = prefix
        self.history = []

    def save(self, model, optimizer, scaler, epoch, val_loss,
             best_val_loss, patience_counter, gate_passed, best_shuffled_ic=-float("inf"),
             ema_model=None, revin=None, n_features=None, shic_decline_count=0):
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_shuffled_ic": best_shuffled_ic,
            "patience_counter": patience_counter,
            "gate_passed": gate_passed,
            "version": "v8_neural_ode_antifragile",
            "n_features": n_features,
            "shic_decline_count": shic_decline_count,
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

    def load_latest(self, model, optimizer, scaler, ema_model=None, revin=None, n_features=None):
        path = self.save_dir / f"{self.prefix}_wm_latest.pt"
        if not path.exists():
            return 0, float("inf"), 0, False, -float("inf"), 0

        print(f"  Resuming from {path.name}")
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            version = ckpt.get("version", "unknown")
            if "v8" not in version:
                print(f"  [WARN] Version mismatch: {version}. Starting fresh.")
                return 0, float("inf"), 0, False, -float("inf"), 0

            # load_latest_collision: n_features mismatch = stale checkpoint
            # (per CLAUDE.md Code Change Verification #11)
            ckpt_nf = ckpt.get("n_features")
            if ckpt_nf is not None and n_features is not None and ckpt_nf != n_features:
                print(f"  [WARN] Checkpoint n_features={ckpt_nf} != model n_features={n_features}. Starting fresh.")
                return 0, float("inf"), 0, False, -float("inf"), 0

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
            shic_decline = ckpt.get("shic_decline_count", 0)

            print(f"  Resumed at epoch {start_epoch}, best_val={best_val:.4f}, "
                  f"best_shIC={best_shic:.4f}, patience={patience}, shic_declines={shic_decline}")
            return start_epoch, best_val, patience, gate, best_shic, shic_decline

        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            return 0, float("inf"), 0, False, -float("inf"), 0


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def validate(model, val_loader, revin=None):
    """Full validation over ALL val data with per-horizon IC."""
    model.eval()

    metrics = {"rec": [], "kl": [], "kl_raw": [], "regime": [], "regime_acc": [], "total": [], "dynamics_reg": []}
    for h in REWARD_HORIZONS:
        metrics[f"ret_{h}"] = []

    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = _targets_to_device(targets, DEVICE)
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

def train_world_model(use_revin: bool = False, n_features: int = 37, args=None):
    # Run-tag suffix isolates parallel-batch variants (V1 standard; 2026-05-03).
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""

    log_path = setup_logging(LOG_DIR, f"v8_f{n_features}{run_tag_str}_train")
    torch.set_float32_matmul_precision("medium")

    # -- Feature configuration -------------------------------------------------
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    import settings as _s
    _s.FEATURE_LIST = feature_list
    _s.INPUT_DIM = input_dim
    _s.BASE_DIM = base_dim

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 70)
    print("  V8 WORLD MODEL TRAINER (Neural ODE | Anti-Fragile)")
    print("=" * 70)
    print(f"  Platform:         {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    print(f"  Device:           {DEVICE}")
    print(f"  Features:         {input_dim} (f{n_features})")
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
    print_antifragile_header("V8", af_config)

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
    model = NeuralODEWorldModel(input_dim=input_dim).to(DEVICE)

    # -- Forward-regime head (V8_FORWARD_REGIME flag, 2026-06-10) --------------
    # Activated by env flag V8_FORWARD_REGIME=1 (default "0" = OFF).
    # When OFF: no attachment, no label computation, base path byte-for-byte unchanged.
    # When ON:  attach head to model (before EMA copy), pre-compute per-segment forward
    #           labels, pack them into targets via collate_fn -> get_loss adds 0.10 * aux.
    import numpy as _np_frl
    _use_forward_regime = os.environ.get("V8_FORWARD_REGIME", "0") == "1"
    if _use_forward_regime:
        _shared_fr_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
        if _shared_fr_path not in sys.path:
            sys.path.insert(0, _shared_fr_path)
        from forward_regime_head import attach_forward_regime_head as _attach_frh
        from regime_targets import (
            forward_bear_label as _fwd_bear_lbl,
            forward_trend_label as _fwd_trend_lbl,
            move_onset_label as _fwd_move_lbl,
        )
        _attach_frh(model)
        print("  [V8_FORWARD_REGIME] Pre-computing forward labels on all segments ...")
        for _seg in all_segments:
            _ret1 = _seg.get("target_return_1")
            if _ret1 is None:
                _seg["fwd_bear"]  = _np_frl.full(len(_seg["features"]), _np_frl.nan, dtype=_np_frl.float32)
                _seg["fwd_trend"] = _np_frl.full(len(_seg["features"]), _np_frl.nan, dtype=_np_frl.float32)
                _seg["fwd_move"]  = _np_frl.full(len(_seg["features"]), _np_frl.nan, dtype=_np_frl.float32)
                continue
            _cret = _np_frl.clip(_ret1.astype(_np_frl.float64), -0.99, 10.0)
            _close_ext = _np_frl.empty(len(_ret1) + 1, dtype=_np_frl.float64)
            _close_ext[0] = 1.0
            _np_frl.cumprod(1.0 + _cret, out=_close_ext[1:])
            _close_ext = _np_frl.maximum(_close_ext, 1e-8)
            _seg["fwd_bear"]  = _fwd_bear_lbl(_close_ext[:-1],  K=64, dd_thresh=0.05)
            _seg["fwd_trend"] = _fwd_trend_lbl(_close_ext[:-1], K=64)
            _seg["fwd_move"]  = _fwd_move_lbl(_close_ext[:-1],  a=1,  b=64)
        print(f"  [V8_FORWARD_REGIME] Labels ready on {len(all_segments)} segments "
              f"(K=64, bear/trend/move, NaN tail masked in loss)")

    # -- VSN flag report (V8_VSN, combinable with V8_FORWARD_REGIME) -----------
    # Model was already built with vsn wired in (env var read at __init__ time).
    _use_vsn = os.environ.get("V8_VSN", "0") == "1"
    if _use_vsn:
        vsn_params = sum(p.numel() for p in model.vsn.parameters()) if model.vsn is not None else 0
        print(f"  [V8_VSN] Variable Selection Network ENABLED  "
              f"({vsn_params} params, gate [B,T,{input_dim}] sigmoid, causal per-timestep)")
    else:
        print(f"  [V8_VSN] VSN disabled (default). Set V8_VSN=1 to enable.")

    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    param_count = count_parameters(model)
    print(f"\n  Model Parameters: {param_count:,}")

    if param_count > 20_000_000:
        print(f"  [WARN] Model has {param_count:,} params -- may be too large for 8GB VRAM")

    # -- RevIN (disabled by default; causes temporal memorization) ---------------
    revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None
    revin_tag = "_revin" if use_revin else ""

    # -- Optimizer (SOTA: betas=(0.9, 0.95) from LLaMA/Chinchilla) ------------
    all_params = list(model.parameters())
    if revin is not None:
        all_params += list(revin.parameters())
    optimizer = optim.AdamW(
        all_params,
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
    ckpt_mgr = CheckpointManager(BASE_MODEL_DIR, keep_top_k=3, prefix=f"v8_f{n_features}{revin_tag}{run_tag_str}")
    start_epoch, best_val_loss, patience_counter, gate_passed, best_shuffled_ic, shic_decline_count = \
        ckpt_mgr.load_latest(model, optimizer, scaler, ema_model=ema_model, revin=revin, n_features=n_features)

    nan_count = 0
    max_nan_per_epoch = 100  # ODE solver fp16 can produce occasional NaN; 20 was too aggressive

    # -- Feature Autopsy (non-console diagnostics) ----------------------------
    autopsy_path = LOG_DIR / f"v8_f{n_features}{revin_tag}{run_tag_str}_autopsy_{log_path.stem.split('_train_')[-1]}.jsonl"
    autopsy = FeatureAutopsy(
        feature_list=feature_list,
        base_dim=input_dim,
        log_path=autopsy_path,
        horizons=REWARD_HORIZONS,
        device=DEVICE,
    )

    print(f"\n  Starting from epoch {start_epoch}")
    print(f"  Autopsy log: {autopsy_path.name}")
    print(f"  ATME:         temporal_ctx_drop={TEMPORAL_CTX_DROP}, seq_shuffle={SEQ_SHUFFLE_PROB}")
    print("-" * 70)

    # 2026-05-10: --max-epochs CLI override for short validation runs
    _max_epochs = WM_TOTAL_EPOCHS
    if args is not None and getattr(args, "max_epochs", None) is not None:
        _max_epochs = args.max_epochs
    for epoch in range(start_epoch, _max_epochs):
        epoch_start = time.time()
        model.train()

        # -- LR (warmup + cosine) ----------------------------------------------
        current_lr = get_lr_for_epoch(epoch)
        set_lr(optimizer, current_lr)

        mask_ratio = get_mask_ratio(epoch)

        kl_anneal = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) if KL_ANNEAL_EPOCHS > 0 else 1.0

        # Gumbel tau annealing: linear decay from START to END over ANNEAL epochs
        gumbel_tau = GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * min(1.0, (epoch + 1) / GUMBEL_TAU_ANNEAL_EPOCHS)

        epoch_keys = ["total", "rec", "kl", "kl_raw", "regime", "regime_acc", "dynamics_reg"] + \
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
            targets_gpu = _targets_to_device(targets, DEVICE)

            # -- Mixup augmentation (batch-level) ------------------------------
            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # -- Sequence shuffling (anti-temporal-memorization) ---------------
            # forward_regime_labels (nested dict of per-bar labels) is NOT shuffled:
            # the perm applies to the return targets so they stay aligned with the
            # shuffled obs; the forward labels are moved-only (see _targets_to_device).
            if SEQ_SHUFFLE_PROB > 0:
                B_sz = obs.shape[0]
                for b in range(B_sz):
                    if torch.rand(1).item() < SEQ_SHUFFLE_PROB:
                        perm = torch.randperm(obs.shape[1], device=obs.device)
                        obs[b] = obs[b][perm]
                        for h in targets_gpu:
                            if isinstance(targets_gpu[h], dict):
                                continue
                            targets_gpu[h][b] = targets_gpu[h][b][perm]

            # -- RevIN normalization (opt-in) ----------------------------------
            if revin is not None:
                obs = revin(obs, mode='norm')

            # -- Forward pass with AMP -----------------------------------------
            with torch.amp.autocast("cuda"):
                loss, loss_dict, base_outputs = model.get_loss(
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

            clip_params = list(model.parameters())
            if revin is not None:
                clip_params += list(revin.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(
                clip_params, WM_GRAD_CLIP
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

        # Log effective Kendall weights (all 4: rec, kl, r1, regime)
        with torch.no_grad():
            _s = model.log_vars.clamp(-6.0, 6.0)
            _s_rec = _s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN).item()
            _s_r1 = _s[2].clamp(max=RETURN_LOG_VAR_CLAMP_MAX).item()
            _w_rec = math.exp(-_s_rec)
            _w_kl = math.exp(-_s[1].item())
            _w_r1 = math.exp(-_s_r1)
            _regime_idx = 2 + len(REWARD_HORIZONS)
            _s_reg = _s[_regime_idx].clamp(max=REGIME_LOG_VAR_CLAMP_MAX).item()
            _w_reg = math.exp(-_s_reg)

        print(
            f"  Ep {epoch+1:3d} | "
            f"Loss: {avg_stats.get('total', 0):.4f} | "
            f"Rec: {avg_stats.get('rec', 0):.4f} | "
            f"KL: {avg_stats.get('kl', 0):.2f} raw:{avg_stats.get('kl_raw', 0):.3f} | "
            f"DynReg:{avg_stats.get('dynamics_reg', 0):.4f} | "
            f"Reg:{avg_stats.get('regime', 0):.3f} Acc:{avg_stats.get('regime_acc', 0)*100:.0f}% | "
            f"{ret_str} | "
            f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} w_kl:{_w_kl:.2f} w_reg:{_w_reg:.1f} | "
            f"Mask:{mask_ratio:.2f} | "
            f"LR:{current_lr:.1e} | "
            f"GN:{avg_grad_norm:.2f} | "
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
                # OOM mitigation (2026-04-30): ShIC compute is the memory peak
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                should_stop, reason = overfit_monitor.check_overfit(
                    contiguous_ic, shuffled_ic, epoch,
                )
                if should_stop:
                    print(f"\n  [OVERFIT STOP] {reason}")
                    break

                if shuffled_ic > best_shuffled_ic:
                    best_shuffled_ic = shuffled_ic
                    best_ema_dict = {"model_state_dict": ema_model.state_dict()}
                    if revin is not None:
                        best_ema_dict["revin_state_dict"] = revin.state_dict()
                    torch.save(best_ema_dict, BASE_MODEL_DIR / f"v8_f{n_features}{revin_tag}{run_tag_str}_wm_best_ema.pt")
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
                n_features=n_features, shic_decline_count=shic_decline_count,
            )

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
                f"DynReg:{val_metrics.get('dynamics_reg', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"KL: {val_metrics.get('kl', 0):.2f} raw:{val_metrics.get('kl_raw', 0):.3f} | "
                f"Reg:{val_regime_acc:.0f}% | "
                f"{gate_status}{save_marker} | "
                f"{val_time:.0f}s"
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
                print(f"\n  [STOP] Early stopping at epoch {epoch+1} (patience={WM_PATIENCE})")
                break

        # -- Memory cleanup ----------------------------------------------------
        if (epoch + 1) % 5 == 0:  # OOM fix 2026-04-30 (was 10; ep99 V1.0 OOM)
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
    print(f"  Weights saved:    {BASE_MODEL_DIR / f'v8_f{n_features}{revin_tag}{run_tag_str}_wm_best_ema.pt'}")
    print(f"  Checkpoint:       {BASE_MODEL_DIR / 'v8_wm_latest.pt'}")

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
    parser = argparse.ArgumentParser(description="V8 World Model Trainer")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override WM_TOTAL_EPOCHS for short validation runs")
    parser.add_argument("--revin", action="store_true",
                        help="Enable RevIN normalization (off by default; causes memorization)")
    parser.add_argument("--loss-type", type=str, choices=["ce", "crps"], default="ce",
                        help="Return loss type: ce (cross-entropy, default) or crps (ordinal-aware CRPS)")
    parser.add_argument("--features", type=int, choices=sorted(SUPPORTED_FEATURE_COUNTS_V8), default=13,
                        help="Number of features. 29 = Pattern-P-cleaned (default: 13).")
    # V1-standard frontier-ML upgrade flags (--sam, --mtp, --mdn, --label-noise,
    # --logit-clip, --run-tag, etc.) -- only --run-tag is loss-path-wired in V8;
    # the rest are exposed for argparse uniformity.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args
    add_upgrade_args(parser)
    add_meta_args(parser)
    args = parser.parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    import settings
    settings.RETURN_LOSS_TYPE = args.loss_type
    import world_model as _wm_mod  # propagate to get_loss's `from settings import *`
    _wm_mod.RETURN_LOSS_TYPE = args.loss_type  # binding (mutating settings alone was a silent no-op)
    success = train_world_model(use_revin=args.revin, n_features=args.features, args=args)
    if not success:
        print("\n  Exiting with gate failure status.")
        sys.exit(1)
