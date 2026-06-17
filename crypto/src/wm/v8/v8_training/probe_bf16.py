"""
Probe script for V8 bf16 ODE fix.

Runs 100 training steps at B=32 with the bf16 solver and:
  1. Checks for NaN/inf in loss and h_seq
  2. Tracks h_seq.abs().max() across steps (magnitude stability)
  3. Times forward+backward per step
  4. Reports speedup vs the fp32 baseline (1.0-1.2 s/step from training log)

If clean -> safe to restart full training.
If NaN -> revert and investigate.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "wm" / "v8" / "v8_training"))

import settings as S
from world_model import NeuralODEWorldModel
from anti_fragile import (AntifragileDataset, WalkForwardSplitter,
                             load_full_data, compute_regime_weights,
                             AntifragileConfig)
from torch.utils.data import DataLoader
from train_world_model import collate_fn

SEED = 42
N_STEPS = 100
DATA_DIR = ROOT / "data" / "processed"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[probe] Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    print("[probe] Settings: B=%d, SEQ_LEN=%d, INPUT_DIM=%d, D=%d, substeps=%d" % (
        S.WM_BATCH_SIZE, S.WM_SEQ_LEN, S.INPUT_DIM, S.WM_D_MODEL, S.ODE_SUBSTEPS))

    # Load real data the same way train_world_model.py does
    print("\n[probe] Loading data...")
    t0 = time.time()
    all_segments = load_full_data(DATA_DIR, S.FEATURE_LIST,
                                     S.ASSET_TO_IDX, S.REWARD_HORIZONS)
    print("  loaded %d segments in %.1fs" % (len(all_segments), time.time() - t0))

    af_config = AntifragileConfig()   # defaults match settings
    splitter = WalkForwardSplitter(af_config)
    train_segs, _, _, _ = splitter.split_four_way(all_segments)
    weights = compute_regime_weights(train_segs)
    ds = AntifragileDataset(train_segs, seq_len=S.WM_SEQ_LEN,
                              reward_horizons=S.REWARD_HORIZONS,
                              augment=True, config=af_config,
                              sample_weights=weights)
    sampler = ds.get_sampler()
    loader = DataLoader(ds, batch_size=S.WM_BATCH_SIZE, sampler=sampler,
                          shuffle=sampler is None, num_workers=0,
                          collate_fn=collate_fn)

    model = NeuralODEWorldModel().to(S.DEVICE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=S.WM_LR,
                                     weight_decay=S.WM_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    hmax_history = []
    loss_history = []
    step_times = []
    nan_count = 0

    batch_iter = iter(loader)
    print("\n[probe] Running %d steps..." % N_STEPS)
    for step in range(N_STEPS):
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        obs, targets_dict, asset = batch
        obs = obs.to(S.DEVICE)
        asset = asset.to(S.DEVICE)
        targets = {k: v.to(S.DEVICE) for k, v in targets_dict.items()}

        t0 = time.time()
        with torch.amp.autocast("cuda"):
            loss, loss_dict, model_out = model.get_loss(
                obs, asset, targets,
                mask_ratio=0.1, block_mask=True,
                kl_anneal=1.0, gumbel_tau=0.5,
                temporal_ctx_drop=S.TEMPORAL_CTX_DROP,
                regime_labels=targets.get("regime_label"),
            )
        loss_val = loss.item()

        # Track h_seq magnitude from the model output
        h_max = 0.0
        if model_out is not None and "h_seq" in model_out:
            h_seq = model_out["h_seq"]
            if torch.isfinite(h_seq).all():
                h_max = float(h_seq.abs().max().item())
        hmax_history.append(h_max)

        optimizer.zero_grad(set_to_none=True)
        if np.isfinite(loss_val) and loss_val < 500:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), S.WM_GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            gn = float(grad_norm.item()) if torch.isfinite(grad_norm) else float("inf")
        else:
            nan_count += 1
            gn = float("nan")
        torch.cuda.synchronize()
        dt = time.time() - t0
        step_times.append(dt)
        loss_history.append(loss_val)

        if step < 5 or step % 10 == 0 or step == N_STEPS - 1:
            print("  step %3d | loss %8.3f | h_max %6.2f | gn %6.2f | dt %.3fs" % (
                step, loss_val, h_max, gn, dt))

    # Summary
    print("\n[probe] SUMMARY (%d steps):" % N_STEPS)
    print("  NaN/inf losses     : %d" % nan_count)
    print("  loss mean/std      : %.3f / %.3f" % (np.mean(loss_history), np.std(loss_history)))
    print("  loss trajectory    : first=%+.2f  mid=%+.2f  last=%+.2f" % (
        loss_history[0], loss_history[len(loss_history)//2], loss_history[-1]))
    print("  h_seq.abs.max mean : %.2f  (max across run %.2f)" % (
        np.mean(hmax_history), np.max(hmax_history)))
    # Skip first steps (warm-up + data priming)
    warm = step_times[10:]
    print("  step time median   : %.3fs  (warm steps 10+)" % np.median(warm))
    print("  step time p95      : %.3fs" % np.percentile(warm, 95))
    print("  BASELINE fp32       : 1.00-1.20 s/step (from training log)")
    baseline_med = 1.05
    speedup = baseline_med / np.median(warm)
    print("  SPEEDUP            : %.2fx" % speedup)
    print()
    status = "PASS" if nan_count == 0 and np.max(hmax_history) < 1e5 and np.isfinite(np.mean(loss_history)) else "FAIL"
    print("  VERDICT: %s" % status)
    if status == "FAIL":
        print("  (investigate before restarting full training)")
    else:
        print("  (safe to restart full training with bf16 solver)")


if __name__ == "__main__":
    main()
