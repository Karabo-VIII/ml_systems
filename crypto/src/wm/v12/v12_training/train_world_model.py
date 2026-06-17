#!/usr/bin/env python
"""
V12 Training -- Cross-Asset Attention Model
=============================================

Trains in single-asset mode (V1-compatible) by default.
Each asset is processed independently through the shared WaveNet encoder,
then cross-asset attention is applied during multi-asset evaluation.

Usage:
    python src/wm/v12/v12_training/train_world_model.py --features 25
    python src/wm/v12/v12_training/train_world_model.py --features 13
"""
import argparse
import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from settings import *

# Import world model using importlib to avoid path conflicts
import importlib.util
_wm_spec = importlib.util.spec_from_file_location(
    "v12_world_model", str(Path(__file__).resolve().parent / "world_model.py"))
_wm_mod = importlib.util.module_from_spec(_wm_spec)
_wm_spec.loader.exec_module(_wm_mod)
CrossAssetWorldModel = _wm_mod.CrossAssetWorldModel
count_parameters = _wm_mod.count_parameters

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anti_fragile import (
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, AntifragileDataset,
compute_regime_weights,
    make_predict_fn, print_antifragile_header
)
# Read-side contract: load_full_data goes through data_api so future
# pipeline changes only touch one module. See src/data_api/__init__.py.
from data_api import load_full_data_for_training as load_full_data


def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p, ep in zip(model.parameters(), ema_model.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1 - decay)


def collate_fn(batch):
    obs_list, tgt_list, asset_list = [], [], []
    for obs, tgt, asset_idx in batch:
        obs_list.append(obs)
        tgt_list.append(tgt)
        asset_list.append(asset_idx)
    obs = torch.stack(obs_list)
    asset = torch.tensor(asset_list, dtype=torch.long)
    targets = {}
    for key in tgt_list[0]:
        targets[key] = torch.stack([t[key] for t in tgt_list])
    return obs, targets, asset


def save_checkpoint(model, ema_model, optimizer, epoch, val_loss,
                    patience, gate, best_shic, shic_decline, n_features, path,
                    ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        torch.save({"model_state_dict": ema_model.state_dict(),
                     "n_features": n_features, "version": "v12"}, path)
    else:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "patience_counter": patience,
            "gate_status": gate, "best_shic": best_shic,
            "shic_decline_count": shic_decline, "n_features": n_features,
            "version": "v12",
        }, path)


def load_latest(model, ema_model, optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    latest_path = ckpt_dir / ("v12_f%d%s_wm_latest.pt" % (n_features, run_tag_str))
    ema_path = ckpt_dir / ("v12_f%d%s_wm_best_ema.pt" % (n_features, run_tag_str))
    ckpt_path = latest_path if latest_path.exists() else ema_path
    if not ckpt_path.exists():
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    print("  Loading from %s" % ckpt_path.name)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if ckpt.get("n_features", n_features) != n_features:
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    # ema_state_dict may be absent in ema_only checkpoints (best_ema.pt)
    if "ema_state_dict" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state_dict"], strict=False)
    else:
        ema_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception:
        pass
    return (ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf")),
            ckpt.get("patience_counter", 0), ckpt.get("gate_status", "PENDING"),
            ckpt.get("best_shic", 0.0), ckpt.get("shic_decline_count", 0))


@torch.no_grad()
def validate(model, val_loader):
    model.eval()
    metrics = {"total": [], "direct_ret": [], "regime": [], "regime_acc": []}
    for h in REWARD_HORIZONS:
        metrics["ret_%d" % h] = []
    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}
        with torch.amp.autocast("cuda"):
            _, loss_dict, outputs = model.get_loss(
                obs, asset, targets_gpu, mask_ratio=0.0,
                regime_labels=targets_gpu.get("regime_label"))
        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])
        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(outputs["return_logits"][h].reshape(-1, NUM_BINS))
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    result = {k: np.mean(v) for k, v in metrics.items() if v}
    for h in REWARD_HORIZONS:
        if ic_data[h]["preds"]:
            preds = np.concatenate(ic_data[h]["preds"])
            reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(preds) & np.isfinite(reals)
            if mask.sum() > 100:
                result["ic_%d" % h] = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
    result["ic_mean"] = float(np.mean([result.get("ic_%d" % h, 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])  # Gate on h=1 (only generalizing horizon)
    return result


def run_training(n_features=None, resume=True, args=None):
    if n_features is None:
        n_features = INPUT_DIM
    feature_list, input_dim, _ = get_feature_config(n_features)

    # 2026-05-07 SOTA gate (per @browser no-silent-failures): V12's raison
    # d'etre is cross-asset attention. The standard runner uses single-asset
    # forward (forward_train -> forward_single_asset -> WaveNet -> VIB), which
    # bypasses the cross-asset attention entirely. As trained that way, V12
    # is a regression of V11/V1.x. Refuse to train in degraded mode.
    if not HEADLINE_MODE:
        print("=" * 70)
        print("  V12 TRAINING ABORTED -- single-asset path is degraded mode")
        print("=" * 70)
        print("  V12's cross-asset attention is dead code in single-asset path.")
        print("  Training in single-asset mode is structurally guaranteed to")
        print("  underperform V1.0/V1.1/V1.4/V1.6 (all use RSSM bottleneck;")
        print("  V12 single-asset has only WaveNet+VIB, weaker prior).")
        print()
        print("  To train V12 properly (per docs/WM_HEADLINE_UPGRADE_PLAN S14):")
        print("  1. Set V12_HEADLINE_MODE=1 to activate forward_multi_asset path")
        print("  2. Wire MultiAssetDataset emitting [B, A, T, F] synchronized batches")
        print("     (requires wall-clock floored panel data, e.g. v51 1d cadence")
        print("     or a wall-clock-aligned dollar-bar resample layer)")
        print("  3. Replace get_loss with get_multi_loss in this trainer")
        print()
        print("  Estimated unlock: 1-2 days harness work. Until then V12 is gated.")
        sys.exit(2)

    # -- HEADLINE_MODE: multi-asset path (Layer-C wiring, 2026-06-10) -----------
    # MultiAssetDataset + build_multi_asset_collate_fn + get_multi_loss are wired here.
    # The single-asset abort guard above keeps the sys.exit(2) for non-HEADLINE mode.
    print("=" * 70)
    print("  V12 HEADLINE_MODE -- multi-asset forward_multi_asset + get_multi_loss")
    print("=" * 70)

    # Import the multi-asset dataset + collate from the shared primitive.
    _shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
    if _shared_path not in sys.path:
        sys.path.insert(0, _shared_path)
    from multi_asset_dataset import MultiAssetDataset, AnchorSchedule, build_multi_asset_collate_fn

    # Run-tag suffix isolates parallel-batch variants' checkpoints + logs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""

    n_feat_str = "f%d" % input_dim
    print("  Features: %d | d_model: %d | Device: %s" % (input_dim, WM_D_MODEL, DEVICE))
    print("  Multi-asset: %d assets | seq_len: %d" % (NUM_ASSETS, WM_SEQ_LEN))

    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        return False

    # Build synchronized multi-asset datasets for train + val.
    # AnchorSchedule "btc_pace" uses BTC (asset_idx=0) timestamps as anchors.
    # min_assets_present=1 keeps any anchor with at least one asset (partial-asset
    # batches are fine; the mask handles absent slots in get_multi_loss).
    _anchor_train = AnchorSchedule(strategy="btc_pace", btc_asset_idx=0)
    _anchor_val = AnchorSchedule(strategy="btc_pace", btc_asset_idx=0)

    # FIX 1 (2026-06-10): use split_four_way_dated instead of split_four_way.
    # split_four_way._slice_seg does NOT copy the 'timestamp' key, so its output
    # segments are missing 'timestamp'.  MultiAssetDataset.__init__ requires
    # 'timestamp' in every segment -> ValueError -> sys.exit(2) crash.
    # split_four_way_dated._slice_seg_dated propagates 'timestamp', and also
    # freezes ALL assets at the same calendar boundaries (eliminates cross-asset
    # calendar-overlap leakage where one asset's val bars overlap another's train).
    from anti_fragile import AntifragileConfig, WalkForwardSplitter, compute_regime_weights
    af_config = AntifragileConfig()
    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way_dated(all_segments)

    try:
        train_ds = MultiAssetDataset(
            train_seg, seq_len=WM_SEQ_LEN, reward_horizons=REWARD_HORIZONS,
            anchor_schedule=_anchor_train, n_assets=NUM_ASSETS, min_assets_present=1,
        )
        val_ds = MultiAssetDataset(
            val_seg, seq_len=WM_SEQ_LEN, reward_horizons=REWARD_HORIZONS,
            anchor_schedule=_anchor_val, n_assets=NUM_ASSETS, min_assets_present=1,
        )
    except ValueError as e:
        print("  [V12 HEADLINE_MODE] MultiAssetDataset construction failed: %s" % e)
        print("  Ensure DATA_DIR has synchronized multi-asset chimera data.")
        sys.exit(2)

    _collate = build_multi_asset_collate_fn(REWARD_HORIZONS)
    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True,
                             collate_fn=_collate)

    print("  Train samples: %d | Val samples: %d" % (len(train_ds), len(val_ds)))
    # ANCHOR (2026-06-10): the cross-asset path now HAS a recon-anchor + VIB KL
    # bottleneck (forward_multi_asset returns recon/vib_mu/vib_logvar;
    # get_multi_loss includes masked recon MSE + annealed VIB KL). This closes the
    # V22/V25 memorization trap the previous warning flagged. ShIC remains an
    # independent in-training detector; watch rec/kl alongside it.
    print("  [V12 HEADLINE] recon+VIB anchor ACTIVE on cross-asset path "
          "(VIB_Z_DIM=%d, KL_WEIGHT=%.3f, anneal_epochs=%d). "
          "Watch rec/kl/ShIC every %d epochs." %
          (VIB_Z_DIM, VIB_KL_WEIGHT, VIB_KL_ANNEAL_EPOCHS, SHUFFLED_IC_CHECK_INTERVAL))

    model = CrossAssetWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = CrossAssetWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False
    print("  Parameters: %d" % count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                   betas=(0.9, 0.95))
    use_amp = True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, patience, gate, best_shic, shic_decline = \
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str) if resume else \
        (0, float("inf"), 0, "PENDING", 0.0, 0)

    # FIX 2 (2026-06-10): ShIC tracker for the HEADLINE loop.
    # Mirrors the single-asset path (lines ~463-585).  ShuffledICTracker and
    # make_predict_fn are already imported at module top.
    shic_tracker = ShuffledICTracker(af_config)

    # FIX 1 (2026-06-10): VIB-collapse guard counter.
    # Counts consecutive epochs where val_kl < KL_COLLAPSE_FLOOR after anneal is done.
    kl_collapse_count = 0

    # HEADLINE_MODE uses HEADLINE_STEPS_PER_EPOCH=200 (C1-equivalent data throughput).
    # Each step processes B*A=32*10=320 asset-samples vs C1's B=32; 200*320=64k=C1-equiv.
    # WM_STEPS_PER_EPOCH stays at 2000 (CDAP invariant); the HEADLINE path reads this var.
    _steps_per_epoch = HEADLINE_STEPS_PER_EPOCH  # 200
    import math as _math
    def _lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(_steps_per_epoch * WM_EPOCHS, 1)
        return 0.5 * (1.0 + _math.cos(_math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    for epoch in range(start_epoch, WM_EPOCHS):
        model.train()
        kl_anneal = min(1.0, (epoch + 1) / VIB_KL_ANNEAL_EPOCHS) if VIB_KL_ANNEAL_EPOCHS > 0 else 1.0
        train_iter = iter(train_loader)
        epoch_nan = 0
        pbar = tqdm(range(_steps_per_epoch), desc="Epoch %3d" % (epoch + 1), leave=False)

        for step in pbar:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            multi_obs = batch["obs"].to(DEVICE, non_blocking=True)           # [B,A,T,F]
            multi_ids = batch["asset_ids"].to(DEVICE, non_blocking=True)     # [B,A]
            mask = batch["mask"].to(DEVICE, non_blocking=True)               # [B,A,T]
            tgts_gpu = {h: batch["targets"][h].to(DEVICE, non_blocking=True)
                        for h in REWARD_HORIZONS}

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                # kl_anneal ramps 0->1 over VIB_KL_ANNEAL_EPOCHS (computed above).
                # Wires the HEADLINE recon/VIB anchor (2026-06-10): get_multi_loss
                # now includes the masked recon MSE + annealed VIB KL bottleneck.
                loss, loss_dict, _ = model.get_multi_loss(
                    multi_obs, multi_ids, tgts_gpu, mask, return_components=False,
                    kl_anneal=kl_anneal)

            if not torch.isfinite(loss):
                epoch_nan += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(model, ema_model)

            if step % LOG_FREQ == 0:
                pbar.set_postfix({"loss": "%.4f" % loss_dict["total"],
                                  "rec": "%.4f" % loss_dict.get("rec", 0.0),
                                  "kl": "%.4f" % loss_dict.get("kl", 0.0),
                                  "nan": epoch_nan})

        # -- Validation (multi-asset get_multi_loss) --
        model.eval()
        val_metrics = {"total": [], "direct_ret": [], "rec": [], "kl": []}
        for h in REWARD_HORIZONS:
            val_metrics["ret_%d" % h] = []
        with torch.no_grad():
            for batch in val_loader:
                m_obs = batch["obs"].to(DEVICE, non_blocking=True)
                m_ids = batch["asset_ids"].to(DEVICE, non_blocking=True)
                m_mask = batch["mask"].to(DEVICE, non_blocking=True)
                t_gpu = {h: batch["targets"][h].to(DEVICE, non_blocking=True)
                         for h in REWARD_HORIZONS}
                with torch.amp.autocast("cuda"):
                    _, vdict, _ = model.get_multi_loss(m_obs, m_ids, t_gpu, m_mask)
                for k in val_metrics:
                    if k in vdict:
                        val_metrics[k].append(vdict[k])
        val_total = float(np.mean(val_metrics["total"])) if val_metrics["total"] else float("inf")

        val_rec = float(np.mean(val_metrics["rec"])) if val_metrics["rec"] else 0.0
        val_kl = float(np.mean(val_metrics["kl"])) if val_metrics["kl"] else 0.0
        print("  [V12 HEADLINE] epoch %d  val_loss=%.4f  rec=%.4f  kl=%.4f  nan=%d" % (
            epoch + 1, val_total, val_rec, val_kl, epoch_nan))

        # FIX 1 (2026-06-10): VIB-collapse guard.
        # Once the KL anneal ramp is complete (kl_anneal >= 1.0), the bottleneck
        # should stay active (kl >= KL_COLLAPSE_FLOOR). If it collapses, the cross-
        # asset path can memorize temporal order in a way the ShIC tracker (which
        # watches the per-asset encoder only) may not catch.
        if kl_anneal >= 1.0:
            if val_kl < KL_COLLAPSE_FLOOR:
                kl_collapse_count += 1
            else:
                kl_collapse_count = 0
            if kl_collapse_count >= KL_COLLAPSE_K:
                print("  *** VIB COLLAPSE: kl=%.4f < KL_COLLAPSE_FLOOR=%.4f for %d consecutive "
                      "epochs -> bottleneck open, memorization risk. EARLY STOP. ***" % (
                          val_kl, KL_COLLAPSE_FLOOR, kl_collapse_count))
                break

        if val_total < best_val:
            best_val = val_total
            patience = 0
            save_checkpoint(model, ema_model, optimizer, epoch, best_val,
                            patience, gate, best_shic, shic_decline, input_dim,
                            MODEL_DIR / ("v12_%s%s_wm_best_ema.pt" % (n_feat_str, run_tag_str)),
                            ema_only=True)
        else:
            patience += 1

        save_checkpoint(model, ema_model, optimizer, epoch, val_total,
                        patience, gate, best_shic, shic_decline, input_dim,
                        MODEL_DIR / ("v12_%s%s_wm_latest.pt" % (n_feat_str, run_tag_str)))

        # FIX 2 (2026-06-10): ShIC memorization detector for HEADLINE loop.
        # Mirrors the single-asset path (lines ~574-585 in the legacy path).
        # Runs every SHUFFLED_IC_CHECK_INTERVAL epochs on val_seg (flat per-asset
        # segments propagated by split_four_way_dated with 'timestamp' intact).
        # make_predict_fn calls forward_train (shared per-asset encoder) which is
        # valid for memorization detection: if the encoder memorizes temporal order
        # ShIC collapses to ~0, same as V22/V25.  Contiguous IC is not re-computed
        # here (no single-asset validate() in HEADLINE mode); the ShIC trend alone
        # is the signal (decline_count guards against V22/V25 failure mode).
        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_seg, predict_fn, horizon=1)
            print("  [V12 HEADLINE ShIC] epoch %d  ShIC=%.4f  best=%.4f  decline=%d/%d" % (
                epoch + 1, shic, best_shic, shic_decline, SHUFFLED_IC_PATIENCE))
            if shic > best_shic:
                best_shic = shic
                shic_decline = 0
            else:
                shic_decline += 1
            if shic_decline >= SHUFFLED_IC_PATIENCE:
                print("  [V12 HEADLINE] EARLY STOP: ShIC patience exhausted "
                      "(shic_decline=%d >= SHUFFLED_IC_PATIENCE=%d) at epoch %d" % (
                          shic_decline, SHUFFLED_IC_PATIENCE, epoch + 1))
                break

        # Early-stop guard (added 2026-06-10: HEADLINE path previously ran full WM_EPOCHS
        # unconditionally -- no patience check existed). WM_PATIENCE=20 epochs.
        if patience >= WM_PATIENCE:
            print("  [V12 HEADLINE] Early stop at epoch %d (patience=%d >= WM_PATIENCE=%d)" % (
                epoch + 1, patience, WM_PATIENCE))
            break

    print("  [V12 HEADLINE] Training complete. best_val=%.4f" % best_val)
    return True

    # ---- Below: the legacy single-asset path (kept for reference, unreachable) ----

    # Run-tag suffix isolates parallel-batch variants' checkpoints + logs.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import make_run_tag_suffix
    run_tag_str = make_run_tag_suffix(args) if args is not None else ""

    print("=" * 70)
    print("  V12 TRAINING -- Cross-Asset Attention")
    print("  Features: %d | d_model: %d | Device: %s" % (input_dim, WM_D_MODEL, DEVICE))
    print("  Single-asset mode (V1-compatible). Cross-asset at eval time.")
    print("=" * 70)

    af_config = AntifragileConfig()
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        return False

    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way(all_segments)
    regime_weights = compute_regime_weights(train_seg)

    train_ds = AntifragileDataset(train_seg, seq_len=WM_SEQ_LEN,
                                   reward_horizons=REWARD_HORIZONS,
                                   augment=True, config=af_config,
                                   sample_weights=regime_weights)
    val_ds = AntifragileDataset(val_seg, seq_len=WM_SEQ_LEN,
                                 reward_horizons=REWARD_HORIZONS,
                                 augment=False, config=af_config)

    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE,
                               sampler=train_ds.get_sampler(),
                               num_workers=NUM_WORKERS, pin_memory=True,
                               drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS,
                             pin_memory=True, collate_fn=collate_fn)

    model = CrossAssetWorldModel(input_dim=input_dim).to(DEVICE)

    # -- Frontier-ML model-side upgrades (B002 R1 MTP, B001 R3 adaptive, B003 R3 MDN)
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

    ema_model = CrossAssetWorldModel(input_dim=input_dim).to(DEVICE)
    if upgrade_flags_active:
        from frontier_ml.v1_upgrades.integration import apply_v1_upgrades as _apply
        _apply(
            ema_model,
            use_mtp=getattr(args, "mtp", False),
            use_adaptive_bins=getattr(args, "adaptive_bins", False),
            adaptive_bins_mode=getattr(args, "adaptive_bins_mode", "log_spaced"),
            use_mdn=getattr(args, "mdn", False),
            mdn_mode=getattr(args, "mdn_mode", "normal"),
            mdn_components=getattr(args, "mdn_components", 3),
            verbose=False,
        )
    ema_model.load_state_dict(model.state_dict())
    print("  Parameters: %d" % count_parameters(model))

    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                    betas=(0.9, 0.95))

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
        optimizer = upgrade_ctx.optimizer
        install_logit_clip_hooks(upgrade_ctx, model, verbose=True)
    use_amp = upgrade_ctx.use_amp if upgrade_ctx is not None else True
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, patience, gate, best_shic, shic_decline = \
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str) if resume else \
        (0, float("inf"), 0, "PENDING", 0.0, 0)

    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    # LR scheduler (cosine with warmup)
    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    for epoch in range(start_epoch, WM_EPOCHS):
        model.train()
        # VIB KL annealing: ramp from 0 to 1 over VIB_KL_ANNEAL_EPOCHS
        kl_anneal = min(1.0, (epoch + 1) / VIB_KL_ANNEAL_EPOCHS) if VIB_KL_ANNEAL_EPOCHS > 0 else 1.0
        train_iter = iter(train_loader)
        epoch_nan_count = 0
        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc="Epoch %3d" % (epoch + 1), leave=False)

        for step in pbar:
            try:
                obs, targets, asset = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                obs, targets, asset = next(train_iter)

            obs = obs.to(DEVICE, non_blocking=True)
            asset = asset.to(DEVICE, non_blocking=True)
            targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}

            # B007 E2: calibrated label-noise on regression targets (BEFORE mixup)
            if upgrade_ctx is not None and upgrade_ctx.use_label_noise:
                regime_lbl = targets_gpu.get("regime_label")
                for _hk, _tv in list(targets_gpu.items()):
                    if _hk == "regime_label":
                        continue
                    if _tv.dtype.is_floating_point:
                        targets_gpu[_hk] = upgrade_ctx.label_noise_injector(
                            _tv, regime_label=regime_lbl,
                        )

            obs, targets_gpu = augmentor.mixup_batch(obs, targets_gpu)

            # FrAug (B003 R2; opt-in)
            if upgrade_ctx is not None and upgrade_ctx.fraug_module is not None:
                upgrade_ctx.fraug_module.train()
                obs = upgrade_ctx.fraug_module(obs)

            need_components = upgrade_ctx is not None and upgrade_ctx.use_pcgrad
            with torch.amp.autocast("cuda", enabled=use_amp):
                if need_components:
                    loss, loss_dict, _, components = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                        regime_labels=targets_gpu.get("regime_label"),
                        kl_anneal=kl_anneal, return_components=True)
                else:
                    loss, loss_dict, _ = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                        regime_labels=targets_gpu.get("regime_label"),
                        kl_anneal=kl_anneal)
                    components = None

            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= 100:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf in epoch {epoch+1}")
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            clip_params = list(model.parameters())

            if upgrade_ctx is not None and upgrade_ctx.use_sam:
                def _sam_recompute():
                    if need_components:
                        l2, _, _, c2 = model.get_loss(
                            obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                            regime_labels=targets_gpu.get("regime_label"),
                            kl_anneal=kl_anneal, return_components=True)
                        return l2, c2
                    l2, _, _ = model.get_loss(
                        obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                        regime_labels=targets_gpu.get("regime_label"),
                        kl_anneal=kl_anneal)
                    return l2, None
                from frontier_ml.v1_upgrades.trainer_helpers import step_backward_and_update
                step_backward_and_update(upgrade_ctx, loss, components, clip_params,
                                          scaler, sam_recompute_fn=_sam_recompute)
            elif upgrade_ctx is not None and upgrade_ctx.use_pcgrad:
                from frontier_ml.v1_upgrades.trainer_helpers import step_backward_and_update
                step_backward_and_update(upgrade_ctx, loss, components, clip_params, scaler)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            update_ema(model, ema_model)
            scheduler.step()

            if step % LOG_FREQ == 0:
                pbar.set_postfix(L="%.3f" % loss_dict["total"],
                                 r1="%.3f" % loss_dict.get("ret_1", 0))

        val_metrics = validate(ema_model, val_loader)
        val_loss = val_metrics.get("total", float("inf"))
        ic_str = " ".join(["ic%d=%.4f" % (h, val_metrics.get("ic_%d" % h, 0)) for h in ACTIVE_HORIZONS])
        print("  Ep %3d | val=%.3f | %s" % (epoch + 1, val_loss, ic_str))

        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_seg, predict_fn, horizon=1)
            print("    ShIC=%.4f | best=%.4f" % (shic, best_shic))
            if shic > best_shic:
                best_shic = shic
                shic_decline = 0
            else:
                shic_decline += 1
            if shic_decline >= SHUFFLED_IC_PATIENCE:
                print("    [STOP] ShIC patience exhausted.")
                break

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            ckpt = MODEL_DIR / ("v12_f%d%s_wm_best_ema.pt" % (input_dim, run_tag_str))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ckpt,
                            ema_only=True)
            print("    [SAVE] %s" % ckpt.name)

        # Latest (full state, every epoch)
        latest = MODEL_DIR / ("v12_f%d%s_wm_latest.pt" % (input_dim, run_tag_str))
        save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                        patience, gate, best_shic, shic_decline, input_dim, latest)
        if (epoch + 1) % 5 == 0:
            ep = MODEL_DIR / ("v12_f%d%s_wm_epoch_%d.pt" % (input_dim, run_tag_str, epoch + 1))
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ep)

    print("\n  TRAINING COMPLETE (Anti-Fragile)  -- V12")
    return True


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from frontier_ml.v1_upgrades.trainer_helpers import add_upgrade_args

    parser = argparse.ArgumentParser(description="V12 Training")
    parser.add_argument("--features", type=int, default=25, choices=sorted(SUPPORTED_FEATURE_COUNTS_V12))
    parser.add_argument("--resume", action="store_true")
    add_upgrade_args(parser)  # +6 frontier-ML upgrade flags (default OFF)
    args = parser.parse_args()
    run_training(n_features=args.features, resume=args.resume, args=args)
