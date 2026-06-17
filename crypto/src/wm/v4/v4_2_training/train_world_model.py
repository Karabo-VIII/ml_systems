"""
V4.2 World Model Trainer -- Mamba-RSSM + XD Anti-Memorization

Architecture: Mamba Selective State Space Model + RSSM Latents
Version: v4_2_mamba_rssm_antifragile

Supports --features 14|19 for ablation testing:
  - 13: Base features only (base_dim == input_dim, no XD paths)
  - 18: Full features with XD anti-memorization defenses (dropout+noise)

Anti-fragile training features:
  - Walk-forward cross-validation with purge gap (eliminates temporal leakage)
  - Shuffled IC as primary model selection metric (detects memorization)
  - Rich augmentation: noise, feature dropout, temporal jitter, mixup, block swap
  - Regime-balanced sampling (prevents neutral-bias)
  - Overfitting monitor (IC gap detection with auto-stop)

Production features (preserved):
  - Full resumability with persistent training state
  - Data augmentation (noise + feature dropout + anti-fragile augmentations)
  - LR warmup + cosine annealing
  - Windows-safe DataLoader (num_workers=0)
  - EMA model tracking
  - Gradient norm logging, NaN detection
  - Per-horizon IC calculation
"""
import torch
import torch.optim as optim
import numpy as np
import math
import sys
import copy
import gc
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import MambaWorldModel, count_parameters
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
    if epoch >= WM_MASK_RAMP_EPOCHS:
        return WM_MASK_RATIO_END
    progress = epoch / WM_MASK_RAMP_EPOCHS
    return WM_MASK_RATIO_START + progress * (WM_MASK_RATIO_END - WM_MASK_RATIO_START)


def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class CheckpointManager:
    """Full-state checkpoint manager with training state persistence."""

    def __init__(self, save_dir: Path, keep_top_k: int = 3, prefix: str = "v4_2_f18"):
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
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "best_shuffled_ic": best_shuffled_ic,
            "patience_counter": patience_counter,
            "gate_passed": gate_passed,
            "n_features": getattr(self, "_expected_n_features", None),
            "version": f"v4_2_{self.prefix}_mamba_rssm_antifragile",
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
        print(f"  [RESUME] Loading from {path.name}")
        try:
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            # load_latest_collision guard + strict=False (CLAUDE.md §11).
            ckpt_nf = ckpt.get("n_features")
            cur_nf = getattr(model, "input_dim", None)
            if ckpt_nf is not None and cur_nf is not None and ckpt_nf != cur_nf:
                raise RuntimeError(
                    f"load_latest_collision: ckpt n_features={ckpt_nf} != "
                    f"model input_dim={cur_nf}. Delete {path.name} or rerun with matching --features."
                )
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            if ema_model is not None:
                if "ema_state_dict" in ckpt:
                    ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
                else:
                    ema_model.load_state_dict(model.state_dict(), strict=False)

            if revin is not None and "revin_state_dict" in ckpt:
                revin.load_state_dict(ckpt["revin_state_dict"])

            epoch = ckpt.get("epoch", 0) + 1
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
            print(f"    Epoch {epoch}, best_val={best_val:.4f}, "
                  f"best_shIC={best_shic:.4f}, patience={patience}")
            return epoch, best_val, patience, gate, best_shic
        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            return 0, float("inf"), 0, False, -float("inf")


# ==============================================================================
# VALIDATION
# ==============================================================================

@torch.no_grad()
def validate(model, val_loader, revin=None):
    model.eval()
    metrics = {"rec": [], "kl": [], "regime": [], "total": []}
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
        all_preds = np.concatenate(ic_data[h]["preds"])
        all_reals = np.concatenate(ic_data[h]["reals"])
        mask = np.isfinite(all_preds) & np.isfinite(all_reals)
        if mask.sum() > 50:
            result[f"ic_{h}"] = float(np.corrcoef(all_preds[mask], all_reals[mask])[0, 1])
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
    reasons = []
    rec = val_metrics.get("rec", 999)
    ic = val_metrics.get("ic", 0)
    kl = val_metrics.get("kl", 0)
    val_loss = val_metrics.get("total", 999)

    if rec > GATE_REC_MSE_MAX:
        reasons.append(f"Rec={rec:.4f}>{GATE_REC_MSE_MAX}")
    if ic < GATE_IC_MIN:
        reasons.append(f"IC={ic:.4f}<{GATE_IC_MIN}")
    if kl < GATE_KL_MIN:
        reasons.append(f"KL={kl:.4f} collapse")
    if kl > GATE_KL_MAX:
        reasons.append(f"KL={kl:.4f} explosion")

    # Anti-fragile gate: shuffled IC absolute threshold
    if shuffled_ic is not None and shuffled_ic < GATE_IC_MIN:
        reasons.append(f"Shuffled IC={shuffled_ic:.4f}<{GATE_IC_MIN} (memorizing)")

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

def train_world_model(n_features: int = 18, use_revin: bool = False):
    """
    Anti-fragile training loop for V4.2 Mamba-RSSM.

    Args:
        n_features: Number of features to use: 14 (base only) or 19 (full).
    """
    feature_list, input_dim, base_dim = get_feature_config(n_features)
    feat_tag = f"f{n_features}"
    ckpt_prefix = f"v4_2_{feat_tag}"
    log_path = setup_logging(LOG_DIR, f"v4_2_{feat_tag}_train")
    torch.set_float32_matmul_precision("medium")

    # -- Anti-fragile configuration --------------------------------------------
    af_config = AntifragileConfig()

    print("=" * 60)
    print("  V4.2 WORLD MODEL TRAINER (Mamba-RSSM | Anti-Fragile)")
    print("=" * 60)
    print(f"  Device: {DEVICE} | Windows: {IS_WINDOWS}")
    print(f"  Features: {n_features} ({len(feature_list)} in feature_list)")
    print(f"  Base Dim: {base_dim} (posterior/decoder restricted)")
    print(f"  Horizons: {REWARD_HORIZONS}")
    print(f"  Ckpt prefix:  {ckpt_prefix}")
    print_antifragile_header("V4.2", af_config)

    # -- Load Full Data --------------------------------------------------------
    all_segments = load_full_data(
        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if all_segments is None:
        print("  [ERROR] No valid data. Exiting.")
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

    print(f"  Train: {len(train_ds):,} seqs | Val: {len(val_ds):,} seqs")

    sampler = train_ds.get_sampler()
    train_loader = DataLoader(
        train_ds, batch_size=WM_BATCH_SIZE,
        sampler=sampler, shuffle=sampler is None,
        num_workers=NUM_WORKERS, pin_memory=(not IS_WINDOWS), drop_last=True,
        collate_fn=collate_fn, persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=WM_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(not IS_WINDOWS), drop_last=False,
        collate_fn=collate_fn, persistent_workers=False,
    )

    # -- Initialize Model ------------------------------------------------------
    model = MambaWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad = False
    ema_model.eval()

    revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None
    print(f"  Parameters: {count_parameters(model):,}")

    # -- Optimizer -------------------------------------------------------------
    optimizer = optim.AdamW(
        list(model.parameters()) + list(revin.parameters()), lr=WM_LR,
        weight_decay=WM_WEIGHT_DECAY, betas=(0.9, 0.95)
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

    print(f"\n  Starting from epoch {start_epoch + 1} / {WM_TOTAL_EPOCHS}")
    print(f"  ATME:            temporal_ctx_drop={TEMPORAL_CTX_DROP}, seq_shuffle={SEQ_SHUFFLE_PROB}")
    print("-" * 60)

    for epoch in range(start_epoch, WM_TOTAL_EPOCHS):
        model.train()
        mask_ratio = get_mask_ratio(epoch)

        # Warmup + cosine LR
        lr = get_lr_for_epoch(epoch)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # KL annealing: ramp from 0 to 1 over KL_ANNEAL_EPOCHS
        kl_anneal = min(1.0, (epoch + 1) / KL_ANNEAL_EPOCHS) if KL_ANNEAL_EPOCHS > 0 else 1.0

        # Gumbel tau annealing: linear decay from START to END over ANNEAL epochs
        gumbel_tau = GUMBEL_TAU_START - (GUMBEL_TAU_START - GUMBEL_TAU_END) * min(1.0, (epoch + 1) / GUMBEL_TAU_ANNEAL_EPOCHS)

        epoch_keys = ["total", "rec", "kl", "regime", "regime_acc"] + [f"ret_{h}" for h in REWARD_HORIZONS]
        epoch_stats = {k: [] for k in epoch_keys}
        grad_norms = []
        nan_count = 0
        max_nan_per_epoch = 20
        train_iter = iter(train_loader)

        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc=f"Ep {epoch+1:3d}", leave=False)

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

            obs = revin(obs, mode='norm')

            with torch.amp.autocast("cuda"):
                loss, loss_dict, _ = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=mask_ratio, block_mask=True,
                    kl_anneal=kl_anneal, gumbel_tau=gumbel_tau,
                    temporal_ctx_drop=TEMPORAL_CTX_DROP,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            # -- NaN detection -------------------------------------------------
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 500:
                optimizer.zero_grad(set_to_none=True)
                nan_count += 1
                if nan_count >= max_nan_per_epoch:
                    print(f"\n  [ABORT] {nan_count} NaN/Inf losses in epoch {epoch+1}")
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(revin.parameters()), WM_GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()

            update_ema(model, ema_model)
            ema_model._gumbel_tau = gumbel_tau  # Sync Gumbel tau to EMA model

            grad_norms.append(gn.item() if torch.isfinite(gn) else 0.0)

            for k, v in loss_dict.items():
                if k in epoch_stats:
                    epoch_stats[k].append(v)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(
                    L=f"{loss_dict['total']:.3f}",
                    R=f"{loss_dict['rec']:.3f}",
                    KL=f"{loss_dict['kl']:.2f}",
                    gn=f"{gn.item():.2f}" if torch.isfinite(gn) else "nan",
                )

        # -- Epoch Summary -----------------------------------------------------
        avg = {k: np.mean(v) for k, v in epoch_stats.items() if v}
        avg_gn = np.mean(grad_norms) if grad_norms else 0

        ret_str = " | ".join([f"r{h}:{avg.get(f'ret_{h}', 0):.3f}" for h in REWARD_HORIZONS])
        nan_str = f" | NaN:{nan_count}" if nan_count > 0 else ""

        # Log effective Kendall weights (verify corridors work)
        with torch.no_grad():
            _s = model.log_vars.clamp(-6.0, 6.0)
            _s_rec = _s[0].clamp(min=REC_LOG_VAR_CLAMP_MIN).item()
            _s_r1 = _s[2].clamp(max=RETURN_LOG_VAR_CLAMP_MAX).item()
            _w_rec = math.exp(-_s_rec)
            _w_r1 = math.exp(-_s_r1)

        print(
            f"  Ep {epoch+1:3d} | "
            f"L:{avg.get('total', 0):.4f} | "
            f"R:{avg.get('rec', 0):.4f} | "
            f"KL:{avg.get('kl', 0):.2f} | "
            f"Reg:{avg.get('regime', 0):.3f} Acc:{avg.get('regime_acc', 0)*100:.0f}% | "
            f"{ret_str} | "
            f"w_rec:{_w_rec:.2f} w_r1:{_w_r1:.1f} | "
            f"gn:{avg_gn:.2f} | mask:{mask_ratio:.2f} | lr:{lr:.1e}{nan_str}"
        )

        # -- Validation --------------------------------------------------------
        if (epoch + 1) % WM_VAL_EVERY == 0 or epoch == WM_TOTAL_EPOCHS - 1:
            val_metrics = validate(ema_model, val_loader, revin=revin)
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
                train_loss=avg.get("total"),
            )
            gate_str = "PASS" if passed else "fail"
            if latest_shuffled_ic is not None:
                gate_passed = passed  # Only meaningful after ShIC is measured

            save_marker = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_marker = " *BEST*"
            else:
                patience_counter += WM_VAL_EVERY

            ckpt_mgr.save(
                model, optimizer, scaler, epoch,
                val_loss, best_val_loss, patience_counter, gate_passed, best_shuffled_ic,
                ema_model=ema_model, revin=revin,
            )

            ic_str = " | ".join([f"IC{h}:{val_metrics.get(f'ic_{h}', 0):.4f}" for h in REWARD_HORIZONS])
            shuffled_str = ""
            if shuffled_ic is not None:
                ic_gap = contiguous_ic - shuffled_ic
                shuffled_str = f" | ShIC:{shuffled_ic:.4f} Gap:{ic_gap:.4f}"

            print(
                f"  -- VAL | "
                f"L:{val_loss:.4f} | "
                f"R:{val_metrics.get('rec', 0):.4f} | "
                f"{ic_str}{shuffled_str} | "
                f"KL:{val_metrics.get('kl', 0):.2f} | "
                f"[{gate_str}]{save_marker}"
            )

            if not passed:
                print(f"       {reason}")

            if patience_counter >= WM_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

        # Memory cleanup
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Final Report ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE (Anti-Fragile)")
    print("=" * 60)
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print(f"  Best Shuffled IC: {best_shuffled_ic:.4f}")
    print(f"  Gate Status:      {'PASSED' if gate_passed else 'NOT PASSED'}")
    print(f"  Best EMA:         {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_best_ema.pt'}")
    print(f"  Resume from:      {BASE_MODEL_DIR / f'{ckpt_prefix}_wm_latest.pt'}")

    if not gate_passed:
        print("\n  World model did not pass validation gate.")
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
    parser = argparse.ArgumentParser(description="V4.2 World Model Trainer")
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
