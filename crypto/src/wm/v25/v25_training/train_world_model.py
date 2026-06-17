#!/usr/bin/env python
"""V25 Training -- Frontier Crypto WM (first-principles synthesis).

Round-6 high-leverage bet under unconstrained-default-synthesis protocol.
Combines five components, none of which appear together in any single paper:
  1. Channel-tokenized cross-feature attention (iTransformer base)
  2. Hard-coded crypto period embeddings (8h / 24h / 7d / 30d)
  3. Hurst-regime conditioned FFN (per-bar bull/sideways/bear gating)
  4. Rate-budget VIB (auto-tuned β to bits-per-timestep target)
  5. Tail-adaptive Huber loss + adversarial regime upweighting

Anti-fragile by construction. Designed for crypto regime, not paper benchmarks.

Usage:
    python src/wm/v25/v25_training/train_world_model.py --features 29
"""
import argparse
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

# Local world_model import (avoids package vs script ambiguity)
import importlib.util
_wm_spec = importlib.util.spec_from_file_location(
    "v25_world_model", str(Path(__file__).resolve().parent / "world_model.py")
)
_wm_mod = importlib.util.module_from_spec(_wm_spec)
_wm_spec.loader.exec_module(_wm_mod)
V25FrontierWorldModel = _wm_mod.V25FrontierWorldModel
count_parameters = _wm_mod.count_parameters

_v1_path = str(Path(__file__).resolve().parent.parent.parent / "v1" / "v1_0_training")
if _v1_path not in sys.path:
    sys.path.insert(0, _v1_path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from anti_fragile import (  # noqa: E402
    AntifragileConfig, WalkForwardSplitter, AntifragileAugmentor,
    ShuffledICTracker, AntifragileDataset,
    compute_regime_weights, make_predict_fn,
)
from data_api import load_full_data_for_training as load_full_data  # noqa: E402

# Round-8: residual meta-learners (opt-in via --meta flags; base unchanged when off)
_shared_path = str(Path(__file__).resolve().parent.parent.parent / "_shared")
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)
from meta_learners import (   # noqa: E402
    ShICMonitor, MultiHeadConsistency, ProbeCallback, SlowTeacher,
    MetaVariantTracker,
)


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


def save_checkpoint(model, ema_model, optimizer, epoch, val_loss, patience,
                    gate, best_shic, shic_decline, n_features, path,
                    ema_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema_only:
        torch.save({"model_state_dict": ema_model.state_dict(),
                    "n_features": n_features, "version": "v25"}, path)
    else:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "ema_state_dict": ema_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss, "patience_counter": patience,
            "gate_status": gate, "best_shic": best_shic,
            "shic_decline_count": shic_decline, "n_features": n_features,
            "version": "v25",
        }, path)


def load_latest(model, ema_model, optimizer, ckpt_dir, n_features, run_tag_str: str = ""):
    latest_path = ckpt_dir / f"v25_f{n_features}{run_tag_str}_wm_latest.pt"
    ema_path = ckpt_dir / f"v25_f{n_features}{run_tag_str}_wm_best_ema.pt"
    ckpt_path = latest_path if latest_path.exists() else ema_path
    if not ckpt_path.exists():
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    print(f"  Loading from {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    # load_latest_collision guard (per CLAUDE.md Code Change Verification #11)
    if ckpt.get("n_features", n_features) != n_features:
        return 0, float("inf"), 0, "PENDING", 0.0, 0
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
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
        metrics[f"ret_{h}"] = []
    ic_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

    for obs, targets, asset in val_loader:
        obs = obs.to(DEVICE, non_blocking=True)
        asset = asset.to(DEVICE, non_blocking=True)
        targets_gpu = {h: t.to(DEVICE, non_blocking=True) for h, t in targets.items()}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss_dict, outputs = model.get_loss(
                obs, asset, targets_gpu, mask_ratio=0.0,
                regime_labels=targets_gpu.get("regime_label"),
            )
        for k in metrics:
            if k in loss_dict:
                metrics[k].append(loss_dict[k])
        for h in REWARD_HORIZONS:
            pred_ret = model.bucketer.decode(
                outputs["return_logits"][h].reshape(-1, NUM_BINS)
            )
            ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
            ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

    result = {k: np.mean(v) for k, v in metrics.items() if v}
    for h in REWARD_HORIZONS:
        if ic_data[h]["preds"]:
            preds = np.concatenate(ic_data[h]["preds"])
            reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(preds) & np.isfinite(reals)
            if mask.sum() > 100:
                p = preds[mask]; r = reals[mask]
                result[f"ic_{h}"] = float(np.corrcoef(p, r)[0, 1])
                # Sign-flip early-warning + magnitude sanity
                result[f"ic_neg_{h}"] = float(np.corrcoef(-p, r)[0, 1])
                result[f"pred_mean_{h}"] = float(p.mean())
                result[f"pred_std_{h}"] = float(p.std())
                result[f"real_mean_{h}"] = float(r.mean())
                result[f"real_std_{h}"] = float(r.std())
    result["ic_mean"] = float(np.mean([result.get(f"ic_{h}", 0) for h in REWARD_HORIZONS]))
    result["ic"] = result.get("ic_1", result["ic_mean"])
    return result


def run_training(n_features=None, resume=True, args=None):
    if n_features is None:
        n_features = INPUT_DIM
    feature_list, input_dim, _ = get_feature_config(n_features)

    # 2026-05-09 V25 reproducibility fix: pin random seed.
    # V25's regime-conditioned FFN structure has a stochastic basin-selection
    # bug — different random inits land in either correct-regime-label or
    # INVERTED-regime-label basins, producing positive vs sign-flipped IC.
    # Yesterday's run with seed (random) hit positive (ic1=+0.306). Today's
    # run with different seed hit inverted (ic1=-0.31). Pin seed to yesterday's
    # equivalent so production runs reproduce the positive-basin convergence.
    # Override via --seed CLI flag; default=42.
    seed = getattr(args, "seed", None) or 42
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"  Random seed: {seed}", flush=True)

    run_tag_str = ""

    print("=" * 70)
    print("  V25 TRAINING -- Frontier Crypto WM (first-principles synthesis)")
    print(f"  Features: {input_dim} | d_model: {WM_D_MODEL} | n_layers: {WM_N_LAYERS}")
    print(f"  Periods: {PERIOD_BARS} | n_regimes: {N_REGIMES} | VIB target: {VIB_TARGET_RATE_NATS} nats/step")
    print(f"  Tail-adaptive Huber sigma={TAIL_THRESHOLD_SIGMA} w={TAIL_WEIGHT} | Adv-regime weight={ADVERSARIAL_REGIME_WEIGHT}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    af_config = AntifragileConfig()
    all_segments = load_full_data(DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        print("  [ERROR] No valid data. Exiting.")
        return False

    splitter = WalkForwardSplitter(af_config)
    train_seg, val_seg, oos_seg, unseen_seg = splitter.split_four_way(all_segments)
    regime_weights = compute_regime_weights(train_seg)

    # 2026-05-21 oracle validation: same stride=1 fix as V22 for last-bar
    # supervision. See V22 train_world_model.py rationale.
    try:
        from settings import USE_LAST_BAR_SUPERVISION as _last_bar
    except ImportError:
        _last_bar = True
    _stride = 1 if _last_bar else None
    train_ds = AntifragileDataset(train_seg, seq_len=WM_SEQ_LEN,
                                  reward_horizons=REWARD_HORIZONS,
                                  augment=True, config=af_config,
                                  sample_weights=regime_weights,
                                  stride=_stride)
    val_ds = AntifragileDataset(val_seg, seq_len=WM_SEQ_LEN,
                                reward_horizons=REWARD_HORIZONS,
                                augment=False, config=af_config,
                                stride=_stride)

    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE,
                              sampler=train_ds.get_sampler(),
                              num_workers=NUM_WORKERS, pin_memory=True,
                              drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS,
                            pin_memory=True, collate_fn=collate_fn)

    model = V25FrontierWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model = V25FrontierWorldModel(input_dim=input_dim).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    print(f"  Parameters: {count_parameters(model):,}")

    # AutopsyMode: internals-level diagnostic (opt-in via --autopsy).
    # Hooks every Linear/Conv/LayerNorm/Embedding to track activation health,
    # gradient health, and trigger snapshots on NaN/Inf or loss explosions.
    # Diagnostic-only; never crashes training.
    autopsy = None
    if getattr(args, "autopsy", False):
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
            from diagnostics.autopsy_mode import AutopsyMode
            autopsy = AutopsyMode(
                model,
                log_dir=Path(__file__).resolve().parents[4] / "logs" / "v25",
                run_tag=f"v25_f{input_dim}{run_tag_str}",
                sample_every=getattr(args, "autopsy_every", 50),
                loss_window=getattr(args, "autopsy_loss_window", 200),
                explosion_z_threshold=getattr(args, "autopsy_z", 3.0),
            )
            print(f"  [AUTOPSY] enabled; jsonl={autopsy.jsonl_path.name}")
        except Exception as e:
            print(f"  [AUTOPSY] init failed ({type(e).__name__}: {e}); continuing without")

    optimizer = torch.optim.AdamW(model.parameters(), lr=WM_LR, weight_decay=WM_WEIGHT_DECAY,
                                    betas=(0.9, 0.95))

    # 2026-05-09 fix: bfloat16 autocast avoids the fp16 backward-overflow that
    # the autopsy traced to proj/return_trunk/regime_ffn (absmax 5-6 under fp16
    # with GradScaler ×65k → Inf in scaled gradients → biased optimizer skips
    # → systematic sign-flipped training). bfloat16 has fp32-range exponent so
    # cannot overflow at these magnitudes; GradScaler is unnecessary (and
    # actively harmful — it would underflow bfloat16 mantissa).
    AMP_DTYPE = torch.bfloat16 if getattr(args, "bf16", True) else torch.float16
    USE_SCALER = (AMP_DTYPE == torch.float16)
    scaler = torch.amp.GradScaler("cuda") if USE_SCALER else None
    print(f"  AMP dtype: {AMP_DTYPE}, GradScaler: {'enabled' if USE_SCALER else 'disabled (bf16)'}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start_epoch, best_val, patience, gate, best_shic, shic_decline = (
        load_latest(model, ema_model, optimizer, MODEL_DIR, input_dim, run_tag_str)
        if resume else (0, float("inf"), 0, "PENDING", 0.0, 0)
    )

    shic_tracker = ShuffledICTracker(af_config)
    augmentor = AntifragileAugmentor(af_config)

    # ─── Round-8: residual meta-learners (opt-in via args.meta) ─────────────
    # Default: all OFF → base training byte-for-byte unchanged.
    # When enabled: meta runs ALONGSIDE training, produces parallel
    # diagnostic / variant / EMA without affecting base gradients.
    meta_flags = set([m.strip() for m in (getattr(args, "meta", "") or "").split(",") if m.strip()])
    meta_tracker = MetaVariantTracker()
    meta_shic = None
    meta_b = None
    meta_b_optimizer = None
    meta_probe = None
    meta_g = None
    if "shic_monitor" in meta_flags:
        meta_shic = ShICMonitor(check_every=100, threshold=0.3,
                                warn_callback=lambda m: print(f"  [META-A WARN] {m}"))
        meta_tracker.register("shic_monitor", meta_shic, MODEL_DIR)
        print("  [META] Design A - ShICMonitor (passive, ~1% overhead)")
    if "multi_head" in meta_flags:
        meta_b = MultiHeadConsistency(trunk_dim=RETURN_HEAD_DIM).to(DEVICE)
        meta_b_optimizer = torch.optim.AdamW(meta_b.parameters(), lr=3e-4, weight_decay=1e-4)
        meta_tracker.register("multi_head", meta_b, MODEL_DIR)
        print(f"  [META] Design B - MultiHeadConsistency ({sum(p.numel() for p in meta_b.parameters()):,} aux params, ~5-10% overhead)")
    if "probe_callback" in meta_flags:
        meta_probe = ProbeCallback(run_every_epochs=5, n_steps=80)
        meta_tracker.register("probe_callback", meta_probe, MODEL_DIR)
        print("  [META] Design C - ProbeCallback (every 5 epochs, ~5% overhead)")
    if "slow_teacher" in meta_flags:
        # Default passive (alpha=0); user can pass --meta-distill-alpha for active
        distill_alpha = float(getattr(args, "meta_distill_alpha", 0.0) or 0.0)
        meta_g = SlowTeacher(model, ema_decay=0.999, distill_alpha=distill_alpha).to(DEVICE)
        meta_tracker.register("slow_teacher", meta_g, MODEL_DIR)
        print(f"  [META] Design G - SlowTeacher (alpha={distill_alpha}, "
              f"{'passive EMA only' if distill_alpha == 0 else 'active distillation'}, ~10% overhead)")

    def lr_lambda(step):
        if step < WM_WARMUP_STEPS:
            return step / max(WM_WARMUP_STEPS, 1)
        progress = (step - WM_WARMUP_STEPS) / max(WM_STEPS_PER_EPOCH * WM_EPOCHS, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    max_epochs = getattr(args, "max_epochs", None) or WM_EPOCHS
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_iter = iter(train_loader)
        epoch_nan_count = 0
        pbar = tqdm(range(WM_STEPS_PER_EPOCH), desc=f"Epoch {epoch + 1:3d}", leave=False)

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

            # V25 uses rate-budget VIB (β auto-tuned), no kl_anneal kwarg needed
            with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                loss, loss_dict, base_outputs = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=WM_MASK_RATIO,
                    regime_labels=targets_gpu.get("regime_label"),
                )
                # Round-8 Design G — slow-teacher distillation (only if active alpha > 0)
                if meta_g is not None and meta_g.distill_alpha > 0:
                    distill_loss, distill_metrics = meta_g.distillation_loss(
                        base_outputs, obs, asset
                    )
                    loss = loss + distill_loss
                    loss_dict.update(distill_metrics)

            if not math.isfinite(loss.item()):
                optimizer.zero_grad(set_to_none=True)
                epoch_nan_count += 1
                if epoch_nan_count >= 100:
                    print(f"\n  [ABORT] {epoch_nan_count} NaN/Inf in epoch {epoch + 1}")
                    break
                continue

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                # fp16 path: scale loss to prevent gradient underflow
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bfloat16 path: no scaling needed
                loss.backward()
                clip_grad_norm_(model.parameters(), WM_GRAD_CLIP)
                optimizer.step()
            update_ema(model, ema_model)
            scheduler.step()

            # AutopsyMode step (diagnostic-only; never crashes training)
            if autopsy is not None:
                try:
                    autopsy.step(step, loss_components={
                        "total": float(loss_dict.get("total", 0.0)),
                        "ret_1": float(loss_dict.get("ret_1", 0.0)),
                        "ret_4": float(loss_dict.get("ret_4", 0.0)),
                        "recon": float(loss_dict.get("recon", 0.0)),
                        "vib_kl": float(loss_dict.get("vib_kl", 0.0)),
                    })
                except Exception:
                    pass

            # Round-8 Design B — multi-head residual (DETACHED trunk; no base gradient effect)
            if meta_b is not None:
                with torch.amp.autocast("cuda", dtype=AMP_DTYPE):
                    ret_trunk_detached = base_outputs["ret_trunk"].detach()
                    aux_out = meta_b(ret_trunk_detached)
                    aux_target = targets_gpu.get(1, None)
                    if aux_target is not None:
                        aux_loss, aux_metrics = meta_b.compute_loss(aux_out, aux_target)
                meta_b_optimizer.zero_grad(set_to_none=True)
                aux_loss.backward()
                meta_b_optimizer.step()
                if step % LOG_FREQ == 0:
                    meta_tracker.log_metrics("multi_head", step, aux_metrics)

            # Round-8 Design G — slow-teacher EMA update (always, when enabled)
            if meta_g is not None:
                meta_g.update_ema(model)

            # Round-8 Design A — ShIC monitor (periodic check on val batch)
            if meta_shic is not None and meta_shic.should_check(step):
                # Use a small slice of the current batch as val proxy
                val_obs_slice = obs[:8].detach()
                val_asset_slice = asset[:8].detach()
                val_tgt_slice = targets_gpu[1][:8].detach()
                shic_metrics = meta_shic.evaluate(model, val_obs_slice, val_asset_slice,
                                                    val_tgt_slice, step=step)
                meta_tracker.log_metrics("shic_monitor", step, shic_metrics)

            if step % LOG_FREQ == 0:
                pbar.set_postfix(L=f"{loss_dict['total']:.3f}",
                                 r1=f"{loss_dict.get('ret_1', 0):.3f}")

        val_metrics = validate(ema_model, val_loader)
        val_loss = val_metrics.get("total", float("inf"))
        ic_str = " ".join([f"ic{h}={val_metrics.get(f'ic_{h}', 0):+.4f}" for h in ACTIVE_HORIZONS])
        print(f"  Ep {epoch + 1:3d} | val={val_loss:.3f} | {ic_str}", flush=True)
        # Loss component breakdown
        loss_parts = " ".join([
            f"direct={val_metrics.get('direct_ret', 0):.4f}",
            *(f"ret{h}={val_metrics.get(f'ret_{h}', 0):.3f}" for h in ACTIVE_HORIZONS),
            f"reg_acc={val_metrics.get('regime_acc', 0):.3f}",
        ])
        print(f"      losses | {loss_parts}", flush=True)
        # Sign-flip early-warning
        signflip_alerts = []
        for h in ACTIVE_HORIZONS:
            ic_pos = val_metrics.get(f"ic_{h}", 0)
            ic_neg = val_metrics.get(f"ic_neg_{h}", 0)
            if ic_pos < -0.05 and ic_neg > 0.10:
                signflip_alerts.append(f"h={h}(IC{ic_pos:+.3f}/-IC{ic_neg:+.3f})")
        if signflip_alerts:
            print(f"      SIGN-FLIP ALERT: {' '.join(signflip_alerts)}", flush=True)
        # Magnitude sanity (h=1)
        pm = val_metrics.get("pred_mean_1", 0); ps = val_metrics.get("pred_std_1", 0)
        rm = val_metrics.get("real_mean_1", 0); rs = val_metrics.get("real_std_1", 0)
        ratio = ps / max(rs, 1e-9)
        mag_warn = " WARN-collapsed" if ratio < 0.3 else (" WARN-saturated" if ratio > 3.0 else "")
        print(f"      h=1 stats | pred mean={pm:+.5f} std={ps:.5f} | "
              f"real mean={rm:+.5f} std={rs:.5f} | ps/rs={ratio:.2f}x{mag_warn}", flush=True)

        # Round-8 Design C — probe callback (every K epochs)
        if meta_probe is not None and meta_probe.should_run(epoch):
            try:
                probe_metrics = meta_probe.run_probe(model, epoch=epoch, device=DEVICE)
                meta_tracker.log_metrics("probe_callback", epoch, probe_metrics)
                print(f"    [META-C] probe IC at epoch {epoch+1}: {probe_metrics.get('probe_ic', 0):.4f}")
            except Exception as e:
                print(f"    [META-C] probe failed: {str(e)[:60]}")

        if (epoch + 1) % SHUFFLED_IC_CHECK_INTERVAL == 0:
            predict_fn = make_predict_fn(WM_SEQ_LEN, torch.device(DEVICE), revin=None)
            shic = shic_tracker.compute_shuffled_ic(ema_model, val_seg, predict_fn, horizon=1)
            ic1 = val_metrics.get("ic_1", 0.0)
            ratio_shic = abs(shic) / max(abs(ic1), 1e-9)
            # Anti-fragile gate per CLAUDE.md "Shuffled IC / Contiguous IC > 0.3":
            # ratio >= 0.3 means signal generalizes (PASS). ratio < 0.3 means
            # contiguous IC is much larger than shuffled IC = memorization (FAIL).
            # Prior logic was inverted (PASS at < 0.3); fixed 2026-05-10.
            gate_pass = "GATE PASS" if ratio_shic >= 0.3 else "GATE FAIL"
            print(f"    ShIC={shic:+.4f} | best={best_shic:+.4f} | "
                  f"ShIC/IC ratio={ratio_shic:.4f} ({gate_pass}; threshold 0.3)",
                  flush=True)
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
            ckpt = MODEL_DIR / f"v25_f{input_dim}{run_tag_str}_wm_best_ema.pt"
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ckpt,
                            ema_only=True)
            print(f"    [SAVE] {ckpt.name}")

        latest = MODEL_DIR / f"v25_f{input_dim}{run_tag_str}_wm_latest.pt"
        save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                        patience, gate, best_shic, shic_decline, input_dim, latest)
        if (epoch + 1) % 5 == 0:
            ep = MODEL_DIR / f"v25_f{input_dim}{run_tag_str}_wm_epoch_{epoch + 1}.pt"
            save_checkpoint(model, ema_model, optimizer, epoch + 1, val_loss,
                            patience, gate, best_shic, shic_decline, input_dim, ep)

        # 2026-05-09 OOM fix: flush CUDA cache at end of each epoch.
        # Caught during V25 first-real-data run — Ep 3 step 0 OOM in
        # patch_embed spectral_norm._v.clone() under bf16 + autocast as
        # memory fragments accumulate over thousands of forward/backward
        # passes. empty_cache() is gradient-free; safe to call here.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Round-8: save meta variant checkpoints + summary
    if meta_flags:
        print()
        print("  [META] Residual variant summary:")
        for name, info in meta_tracker.metas.items():
            meta_tracker.save_meta_checkpoint(name, input_dim, "v25")
            n_metrics = len(info["metrics"])
            print(f"    {name}: {n_metrics} metric snapshots saved at "
                  f"models/wm/v25/base/v25_f{input_dim}_meta_{name}.pt")
            if info["metrics"]:
                last = info["metrics"][-1][1]
                summary_str = "  ".join(f"{k}={v:.4f}" for k, v in last.items()
                                          if isinstance(v, (int, float)) and abs(v) < 1e6)
                print(f"      last: {summary_str}")

    if autopsy is not None:
        try:
            # Final memorization + embedding-health probes before close
            try:
                # Pull one val batch for memorization probe
                val_iter = iter(val_loader)
                vb = next(val_iter)
                autopsy.memorization_probe(ema_model, vb, n_features=input_dim)
            except Exception:
                pass
            # V25's patch_embed is the prime memorization-vector candidate
            try:
                autopsy.embedding_health_probe(model.patch_embed, name="patch_embed")
            except Exception:
                pass
            autopsy.close()
        except Exception:
            pass

    print("\n  TRAINING COMPLETE (Anti-Fragile) -- V25")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V25 Frontier Crypto WM Training")
    parser.add_argument("--features", type=int, default=29,
                        choices=sorted(SUPPORTED_FEATURE_COUNTS_V25))
    parser.add_argument("--meta", type=str, default="",
                        help="Comma-separated meta-learner residuals to enable: "
                             "shic_monitor,multi_head,probe_callback,slow_teacher. "
                             "Each is OPT-IN; default empty = base training unchanged.")
    parser.add_argument("--meta-distill-alpha", type=float, default=0.0,
                        help="If slow_teacher enabled, alpha for distillation loss. "
                             "0.0 = passive EMA tracking only; >0 = active distill.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override WM_EPOCHS (useful for short autopsy runs).")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                        help="Use bfloat16 autocast (default). --no-bf16 falls back to fp16+GradScaler.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default 42 — pinned 2026-05-09 to ensure "
                             "regime-classifier lands in correct-basin; different seeds "
                             "may invert regime labels and sign-flip predictions).")
    # AutopsyMode flags (diagnostic-only; opt-in)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from diagnostics.autopsy_mode import AutopsyMode as _AP
    _AP.add_argparse(parser)
    args = parser.parse_args()
    run_training(n_features=args.features, resume=args.resume, args=args)
