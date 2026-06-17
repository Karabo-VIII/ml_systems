"""
V6.x Adapter Trainer -- Rolling Window Adaptation

Trains a lightweight FiLM adapter on top of a frozen V6 world model.
The adapter learns regime-specific corrections to V6's return predictions.

Pipeline:
  1. Load frozen V6 base model (best checkpoint)
  2. Load data, split train/val via walk-forward
  3. Build context vector from TRAINING data (not val -- prevents leakage)
  4. Train adapter on-the-fly: each step runs frozen V6 + adapter forward
  5. Validate: adapted IC vs base IC per horizon
  6. Save adapter if it improves over base

On-the-fly design (no pre-caching):
  - Each training step runs frozen V6 forward (no_grad) then adapter forward
  - Avoids 40GB+ memory for cached representations
  - V6 forward under no_grad is fast (~20ms per batch on RTX 4060)

V6-SPECIFIC:
  - V6 uses CausalJEPAWorldModel, not TransformerWorldModel
  - V6 get_loss returns 4 values: (total, loss_dict, l_disc, outputs)
  - Adapter only needs ONE optimizer (no discriminator in adapter)
  - Adapter operates on ctx_latent [B, T, 192] only

RED TEAM fixes applied:
  - Loss: TwoHot cross-entropy on logits (O(1) gradients, not smooth_l1 on decoded preds)
  - Context: computed from TRAINING data tail, not validation data (no look-ahead bias)
  - Context augmentation: noise injection so adapter sees varying context during training
  - max_shift: hard-clamped to 0.05 ceiling + softplus (no runaway)
  - Regime sampling: use WeightedRandomSampler when available
  - Base model hash: saved for compatibility verification
  - No AMP: unnecessary for 15K params, adds quantization noise
  - Shift regularization: penalize large max_shift
  - NaN-safe IC: checks std on both predictions and actuals
"""
import argparse
import hashlib
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import gc
import math
import time
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import CausalJEPAWorldModel, count_parameters
from adapter import (
    AdaptiveResidualAdapter, AdaptedWorldModel,
    ContextComputer, DriftMonitor,
)
from anti_fragile import (
    load_full_data, WalkForwardSplitter, AntifragileDataset,
    compute_regime_weights,
)
from log_utils import setup_logging, teardown_logging
from diagnostics.feature_autopsy import FeatureAutopsy
from revin import RevIN


# =============================================================================
# HELPERS
# =============================================================================

def compute_base_model_hash(model: torch.nn.Module) -> str:
    """Compute a fingerprint of the base model weights for compatibility."""
    checksum = 0.0
    for p in model.parameters():
        checksum += float(p.sum().item())
    return hashlib.md5(str(checksum).encode()).hexdigest()[:12]


def safe_ic(pred: np.ndarray, actual: np.ndarray) -> float:
    """NaN-safe Pearson correlation (IC)."""
    if len(pred) < 30 or np.std(pred) < 1e-10 or np.std(actual) < 1e-10:
        return 0.0
    ic = float(np.corrcoef(pred, actual)[0, 1])
    return ic if np.isfinite(ic) else 0.0


def cosine_lr(epoch: int, total_epochs: int, base_lr: float, min_lr: float = 1e-5) -> float:
    """Cosine decay learning rate schedule."""
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))


# =============================================================================
# COLLATE
# =============================================================================

def collate_fn(batch):
    """Custom collate for AntifragileDataset."""
    obs = torch.stack([b[0] for b in batch])
    asset = torch.stack([b[2] for b in batch])
    targets = {}
    for h in REWARD_HORIZONS:
        targets[h] = torch.stack([b[1][h] for b in batch])
    return obs, targets, asset


# =============================================================================
# CONTEXT COMPUTATION (from training data, not validation)
# =============================================================================

def compute_context_from_segments(
    base_model: CausalJEPAWorldModel,
    segments: list,
    device: torch.device,
    lookback: int = ADAPTER_CONTEXT_LOOKBACK,
    revin: RevIN = None,
) -> np.ndarray:
    """
    Compute 12-dim context vector from the tail of given segments.

    Uses TRAINING data tail to avoid look-ahead bias (RED TEAM fix).
    """
    ctx_computer = ContextComputer(lookback=lookback)

    for seg in segments:
        n = len(seg["features"])
        if n < WM_SEQ_LEN:
            continue
        # Use last `lookback` bars of each segment
        start = max(0, n - lookback)
        feats_window = seg["features"][start:]
        asset_idx = seg["asset_idx"]

        for i in range(0, len(feats_window) - WM_SEQ_LEN, WM_SEQ_LEN):
            obs_np = feats_window[i:i + WM_SEQ_LEN]
            obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=device)
            obs_input = revin(obs, mode='norm') if revin is not None else obs

            with torch.no_grad():
                ctx_latent, return_preds = base_model.encode_sequence(obs_input, asset)

            preds, acts = {}, {}
            for h in REWARD_HORIZONS:
                pred = return_preds[h].cpu().numpy().flatten()
                tgt_key = f"target_return_{h}"
                actual = seg.get(tgt_key, np.zeros(n))[start + i:start + i + WM_SEQ_LEN]
                preds[h] = pred
                acts[h] = actual

            # V6 has regime_head - compute regime_probs
            # We need to do a forward_train to get regime_logits
            with torch.no_grad():
                full_out = base_model.forward_train(obs_input, asset)
                regime_probs = F.softmax(
                    full_out["regime_logits"], dim=-1
                ).cpu().numpy().mean(axis=(0, 1))

            vol = float(obs_np[:, 0].mean())
            ctx_computer.update(preds, acts, regime_probs, vol)

    return ctx_computer.get_context(), ctx_computer.n_samples


# =============================================================================
# ADAPTER VALIDATION
# =============================================================================

@torch.no_grad()
def validate_adapter(
    adapted_model: AdaptedWorldModel,
    base_model: CausalJEPAWorldModel,
    val_segments: list,
    context: torch.Tensor,
    device: torch.device,
    revin: RevIN = None,
) -> dict:
    """
    Compare adapted vs base IC on validation data.

    Uses the adapted_model.forward_train() (not reimplemented inline)
    to ensure training and validation use the same code path.
    """
    adapted_model.eval()
    base_model.eval()

    base_preds = {h: [] for h in REWARD_HORIZONS}
    adapted_preds = {h: [] for h in REWARD_HORIZONS}
    actuals = {h: [] for h in REWARD_HORIZONS}

    for seg in val_segments:
        feats_np = seg["features"]
        asset_idx = seg["asset_idx"]
        n = len(feats_np)

        indices = list(range(0, n - WM_SEQ_LEN, WM_SEQ_LEN))
        if not indices and n >= WM_SEQ_LEN:
            indices = [0]

        for batch_start in range(0, len(indices), 64):
            batch_indices = indices[batch_start:batch_start + 64]
            obs_list = [feats_np[i:i + WM_SEQ_LEN] for i in batch_indices]

            obs = torch.from_numpy(np.stack(obs_list)).float().to(device)
            asset = torch.full((len(obs_list),), asset_idx, dtype=torch.long, device=device)
            ctx = context.unsqueeze(0).expand(len(obs_list), -1).to(device)

            if revin is not None:
                obs = revin(obs, mode='norm')

            # Use adapted_model which returns both adapted and base logits
            adapted_out = adapted_model.forward_train(obs, asset, ctx)

            for h in REWARD_HORIZONS:
                # Decode from adapted and base logits (no duplicate forward pass)
                ap = base_model.bucketer.decode(adapted_out["return_logits"][h]).cpu().numpy()
                bp = base_model.bucketer.decode(adapted_out["base_return_logits"][h]).cpu().numpy()

                for b, idx in enumerate(batch_indices):
                    for t in range(WM_SEQ_LEN):
                        if idx + t < n:
                            tgt_key = f"target_return_{h}"
                            if tgt_key in seg and idx + t < len(seg[tgt_key]):
                                base_preds[h].append(bp[b, t])
                                adapted_preds[h].append(ap[b, t])
                                actuals[h].append(seg[tgt_key][idx + t])

    results = {}
    for h in REWARD_HORIZONS:
        bp = np.array(base_preds[h])
        ap = np.array(adapted_preds[h])
        act = np.array(actuals[h])

        mask = np.isfinite(bp) & np.isfinite(ap) & np.isfinite(act)
        bp, ap, act = bp[mask], ap[mask], act[mask]

        results[f"base_ic_{h}"] = safe_ic(bp, act)
        results[f"adapted_ic_{h}"] = safe_ic(ap, act)
        results[f"ic_delta_{h}"] = results[f"adapted_ic_{h}"] - results[f"base_ic_{h}"]

        if len(ap) > 0:
            results[f"mean_correction_{h}"] = float(np.mean(np.abs(ap - bp)))

    results["base_ic_mean"] = float(np.mean([results[f"base_ic_{h}"] for h in REWARD_HORIZONS]))
    results["adapted_ic_mean"] = float(np.mean([results[f"adapted_ic_{h}"] for h in REWARD_HORIZONS]))
    results["ic_delta_mean"] = results["adapted_ic_mean"] - results["base_ic_mean"]

    return results


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=int, choices=[13, 18, 30, 34, 37, 41], default=None,
                        help="Feature count override (must match base model)")
    parser.add_argument("--revin", action="store_true",
                        help="Load and apply RevIN from base checkpoint (only if base was trained with RevIN)")
    args = parser.parse_args()
    use_revin = args.revin

    # -- Feature config (runtime override via --features) ---------------------
    n_features = getattr(args, 'features', None)
    if n_features is not None:
        feature_list, input_dim, base_dim = get_feature_config(n_features)
    else:
        feature_list, input_dim, base_dim = FEATURE_LIST, INPUT_DIM, INPUT_DIM
        n_features = INPUT_DIM


    print("=" * 70)
    print("  V6.x ADAPTER TRAINER (FiLM Modulation | Rolling Window)")
    print("=" * 70)
    print(f"  Device:         {DEVICE}")
    print(f"  Base model:     CausalJEPAWorldModel (frozen)")
    print(f"  Adapter:        FiLM bottleneck={ADAPTER_BOTTLENECK}, hidden={ADAPTER_FILM_HIDDEN}")
    print(f"  Context dim:    {ADAPTER_CONTEXT_DIM}")
    print(f"  Scale range:    [{1 - ADAPTER_MAX_SCALE_RANGE:.1f}, {1 + ADAPTER_MAX_SCALE_RANGE:.1f}]")
    print(f"  Training:       {ADAPTER_EPOCHS} epochs, {ADAPTER_STEPS_PER_EPOCH} steps, batch={ADAPTER_BATCH_SIZE}")
    print(f"  LR:             {ADAPTER_LR} (cosine decay)")
    print(f"  Loss:           TwoHot cross-entropy (logit-level gradients)")
    print(f"  RevIN:          {'enabled' if use_revin else 'disabled (--revin to enable)'}")

    device_obj = torch.device(DEVICE)

    # -- Setup logging --------------------------------------------------------
    setup_logging(LOG_DIR, "v6x_adapter_train")

    # -- Load frozen V6 base model -------------------------------------------
    base_ckpt_path = BASE_MODEL_DIR / "v6_wm_best.pt"
    if not base_ckpt_path.exists():
        base_ckpt_path = BASE_MODEL_DIR / "v6_wm_weights.pt"
    if not base_ckpt_path.exists():
        print(f"\n  [ERROR] No V6 checkpoint found in {BASE_MODEL_DIR}")
        print("  Train V6 first: python src/wm/v6/v6_training/train_world_model.py")
        teardown_logging()
        return

    print(f"\n  Loading base model from {base_ckpt_path.name}")
    base_model = CausalJEPAWorldModel(input_dim=input_dim).to(device_obj)
    state = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        base_model.load_state_dict(state["model_state_dict"])
    else:
        base_model.load_state_dict(state)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # -- RevIN (disabled by default; load from base checkpoint if --revin) ------
    revin = None
    if use_revin:
        revin = RevIN(num_features=INPUT_DIM).to(device_obj)
        if isinstance(state, dict) and "revin_state_dict" in state:
            revin.load_state_dict(state["revin_state_dict"])
            print(f"  RevIN loaded from V6 checkpoint")
        else:
            print(f"  [WARN] No revin_state_dict in V6 checkpoint -- using fresh RevIN")
        revin.eval()
        for p in revin.parameters():
            p.requires_grad = False

    base_hash = compute_base_model_hash(base_model)
    print(f"  Base model loaded (frozen, hash={base_hash})")

    # -- Create adapter -------------------------------------------------------
    adapter = AdaptiveResidualAdapter().to(device_obj)
    adapter_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Adapter created: {adapter_params:,} params")

    # -- Create composite model -----------------------------------------------
    adapted_model = AdaptedWorldModel(base_model, adapter).to(device_obj)

    # -- Load data ------------------------------------------------------------
    print(f"\n  Loading data from {DATA_DIR}")
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        print("  [ERROR] No data found")
        teardown_logging()
        return

    # -- 4-Way Split: 50/20/20/10 (train/val/oos/unseen) ------------------------
    splitter = WalkForwardSplitter()
    train_segments, val_segments, oos_segments, unseen_segments = \
        splitter.split_four_way(all_segments)

    total_train = sum(len(s["features"]) for s in train_segments)
    total_val = sum(len(s["features"]) for s in val_segments)
    total_oos = sum(len(s["features"]) for s in oos_segments)
    total_unseen = sum(len(s["features"]) for s in unseen_segments)
    total_all = total_train + total_val + total_oos + total_unseen
    print(f"  Train:  {total_train:>12,} bars ({total_train/total_all*100:.1f}%)")
    print(f"  Val:    {total_val:>12,} bars ({total_val/total_all*100:.1f}%)")
    print(f"  OOS:    {total_oos:>12,} bars ({total_oos/total_all*100:.1f}%)")
    print(f"  Unseen: {total_unseen:>12,} bars ({total_unseen/total_all*100:.1f}%) [held out]")

    # -- Build context from TRAINING data tail (RED TEAM: no val leakage) -----
    print(f"\n  Computing context from training data tail...")
    context_np, ctx_n_samples = compute_context_from_segments(
        base_model, train_segments, device_obj, lookback=ADAPTER_CONTEXT_LOOKBACK,
        revin=revin,
    )
    context_tensor = torch.from_numpy(context_np).float().to(device_obj)
    print(f"  Context ({ctx_n_samples} samples):")
    labels = ["IC_r1", "IC_r4", "IC_r16", "IC_r64",
              "bias_r1", "bias_r4", "bias_r16", "bias_r64",
              "regime_bear", "regime_neutral", "regime_bull", "volatility"]
    for i, (label, val) in enumerate(zip(labels, context_np)):
        print(f"    [{i:2d}] {label:16s} = {val:+.4f}")

    # -- Create dataset and loader --------------------------------------------
    regime_weights = compute_regime_weights(train_segments)
    train_dataset = AntifragileDataset(
        train_segments, seq_len=WM_SEQ_LEN,
        reward_horizons=REWARD_HORIZONS,
        sample_weights=regime_weights,
    )
    print(f"\n  Train windows: {len(train_dataset):,}")

    # Use regime-weighted sampler if available (RED TEAM fix)
    sampler = train_dataset.get_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=ADAPTER_BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
    )
    if sampler is not None:
        print(f"  Using regime-weighted sampling")

    # -- Optimizer ------------------------------------------------------------
    optimizer = optim.AdamW(
        adapter.parameters(),
        lr=ADAPTER_LR,
        weight_decay=ADAPTER_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # -- Resume from existing adapter checkpoint ------------------------------
    adapter_ckpt_path = ADAPTER_MODEL_DIR / "v6_adapter_latest.pt"
    start_epoch = 0
    best_ic_delta = -float("inf")

    if adapter_ckpt_path.exists():
        print(f"\n  [RESUME] Loading adapter from {adapter_ckpt_path.name}")
        try:
            ckpt = torch.load(adapter_ckpt_path, map_location=DEVICE, weights_only=False)
            # Verify base model compatibility
            saved_hash = ckpt.get("base_model_hash", "")
            if saved_hash and saved_hash != base_hash:
                print(f"  [WARN] Adapter was trained on different base model "
                      f"(saved={saved_hash}, current={base_hash}). Starting fresh.")
                start_epoch = 0
            else:
                adapter.load_state_dict(ckpt["adapter_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0)
                best_ic_delta = ckpt.get("best_ic_delta", -float("inf"))
                print(f"    Resumed at epoch {start_epoch}, best_ic_delta={best_ic_delta:.4f}")
        except Exception as e:
            print(f"  [WARN] Resume failed: {e}. Starting fresh.")
            start_epoch = 0

    # -- Training loop --------------------------------------------------------
    print(f"\n  Starting from epoch {start_epoch}")
    print("-" * 70)

    for epoch in range(start_epoch, ADAPTER_EPOCHS):
        adapter.train()
        epoch_losses = []
        train_iter = iter(train_loader)

        # Cosine LR decay (RED TEAM fix: add LR schedule)
        lr = cosine_lr(epoch, ADAPTER_EPOCHS, ADAPTER_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        pbar = tqdm(
            range(ADAPTER_STEPS_PER_EPOCH),
            desc=f"Epoch {epoch + 1:3d}",
            leave=False,
            file=sys.stderr,
        )

        for step in pbar:
            try:
                obs_batch, targets, asset_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                obs_batch, targets, asset_batch = next(train_iter)

            obs_batch = obs_batch.to(device_obj)
            asset_batch = asset_batch.to(device_obj)
            targets = {h: t.to(device_obj) for h, t in targets.items()}

            # Context augmentation: add noise so adapter sees varying context
            # (RED TEAM fix: static context was a major issue)
            ctx_noise = torch.randn_like(context_tensor) * 0.1
            ctx_batch = (context_tensor + ctx_noise).unsqueeze(0).expand(
                obs_batch.size(0), -1
            )

            optimizer.zero_grad(set_to_none=True)

            # RevIN normalization (frozen, matches V6 training distribution)
            if revin is not None:
                obs_batch = revin(obs_batch, mode='norm')

            # Run frozen V6 forward (no gradients on base model)
            with torch.no_grad():
                base_out = base_model.forward_train(obs_batch, asset_batch)

            # Get V6's ctx_latent and trunk
            feat = base_out["ctx_latent"].detach()
            ret_trunk = base_out["ret_trunk"].detach()

            # Adapter: compute modulations from feat + context
            modulations = adapter(feat, ctx_batch)

            # Apply modulation to trunk, run frozen return heads
            # Loss: TwoHot cross-entropy on logits (RED TEAM fix: O(1) gradients)
            total_loss = torch.tensor(0.0, device=device_obj)
            for h in REWARD_HORIZONS:
                scale, shift = modulations[h]
                modulated_trunk = ret_trunk * scale + shift
                logits = base_model.return_heads[str(h)](modulated_trunk)
                # TwoHot cross-entropy (not smooth_l1 on decoded preds)
                loss_h = base_model.bucketer.compute_loss(
                    logits.reshape(-1, logits.size(-1)),
                    targets[h].reshape(-1),
                )
                total_loss = total_loss + loss_h

            # Shift regularization (RED TEAM fix: penalize max_shift growth)
            shift_reg = 0.1 * adapter.max_shift.abs()
            total_loss = total_loss + shift_reg

            total_loss.backward()
            grad_norm = clip_grad_norm_(adapter.parameters(), ADAPTER_GRAD_CLIP)
            optimizer.step()

            epoch_losses.append(total_loss.item())

            if step % 50 == 0:
                pbar.set_postfix(loss=f"{total_loss.item():.4f}", gn=f"{grad_norm:.2f}")

        # -- Epoch summary ----------------------------------------------------
        avg_loss = float(np.mean(epoch_losses))
        effective_shift = float(torch.clamp(
            F.softplus(adapter.max_shift), max=adapter.max_shift_ceiling
        ).item())
        print(f"  Ep {epoch + 1:3d} | Loss: {avg_loss:.4f} | "
              f"eff_shift: {effective_shift:.4f} | lr: {lr:.6f}")

        # -- Validation -------------------------------------------------------
        if (epoch + 1) % ADAPTER_VAL_EVERY == 0 or epoch == ADAPTER_EPOCHS - 1:
            val_results = validate_adapter(
                adapted_model, base_model, val_segments,
                context_tensor, device_obj, revin=revin,
            )

            ic_str = " | ".join([
                f"r{h}: {val_results[f'adapted_ic_{h}']:.4f} "
                f"({val_results[f'ic_delta_{h}']:+.4f})"
                for h in REWARD_HORIZONS
            ])
            print(f"  -- VAL | {ic_str}")
            print(f"         | Base IC mean: {val_results['base_ic_mean']:.4f}")
            print(f"         | Adapted IC mean: {val_results['adapted_ic_mean']:.4f}")
            print(f"         | Delta: {val_results['ic_delta_mean']:+.4f}")

            corr_str = " | ".join([
                f"r{h}: {val_results.get(f'mean_correction_{h}', 0):.6f}"
                for h in REWARD_HORIZONS
            ])
            print(f"         | Corrections: {corr_str}")

            save_marker = ""
            if val_results["ic_delta_mean"] > best_ic_delta:
                best_ic_delta = val_results["ic_delta_mean"]
                best_state = {
                    "adapter_state_dict": adapter.state_dict(),
                    "base_model_hash": base_hash,
                    "base_ckpt_name": base_ckpt_path.name,
                    "context": context_np.tolist(),
                    "ic_delta": best_ic_delta,
                    "version": "v6x_film_adapter_v1",
                }
                torch.save(best_state, ADAPTER_MODEL_DIR / "v6_adapter_best.pt")
                save_marker = " *BEST*"

            ckpt_state = {
                "epoch": epoch + 1,
                "adapter_state_dict": adapter.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_ic_delta": best_ic_delta,
                "context": context_np.tolist(),
                "val_results": val_results,
                "version": "v6x_film_adapter_v1",
                "base_model_hash": base_hash,
                "base_ckpt_name": base_ckpt_path.name,
            }
            torch.save(ckpt_state, ADAPTER_MODEL_DIR / "v6_adapter_latest.pt")
            print(f"         | Saved checkpoint{save_marker}")

        # -- Memory cleanup ---------------------------------------------------
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -- Final summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  V6.x ADAPTER TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best IC delta:       {best_ic_delta:+.4f}")
    print(f"  Adapter params:      {adapter_params:,}")
    print(f"  Base model hash:     {base_hash}")
    print(f"  Adapter checkpoint:  {ADAPTER_MODEL_DIR / 'v6_adapter_latest.pt'}")
    print(f"  Best adapter:        {ADAPTER_MODEL_DIR / 'v6_adapter_best.pt'}")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"  VRAM: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")

    teardown_logging()


if __name__ == "__main__":
    main()
