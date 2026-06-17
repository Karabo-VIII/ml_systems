"""
V1.E Ensemble Gating Trainer -- XD-Conditioned Snapshot Weighting

Trains a tiny Linear(5, K) gating network on frozen snapshot models.
The gating learns optimal per-sample seed weighting as a function of
cross-asset features (BTC return, BTC vol, funding spread, cross-return mean, cross-vol mean).

~5*K parameters. Trains in <1 minute.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import math
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from world_model import TransformerWorldModel
from snapshot_ensemble import SnapshotEnsemble, EnsembleGating
from anti_fragile import load_full_data, WalkForwardSplitter, AntifragileDataset
from train_world_model import collate_fn


def train_ensemble_gating():
    """Train XD-conditioned gating for snapshot ensemble."""
    print("=" * 60)
    print("V1.E Ensemble Gating Trainer")
    print("=" * 60)

    # -- Load snapshot ensemble (frozen) ---
    snap_dir = MODEL_DIR / "snapshots"
    snap_paths = sorted(snap_dir.glob("v1_snapshot_*.pt"))
    if not snap_paths:
        print("[ERROR] No snapshots found. Run train_snapshot.py first.")
        return False

    # Load master checkpoint for snapshot ICs
    master_ckpt_path = MODEL_DIR / "v1e_snapshot_latest.pt"
    snapshot_ics = None
    if master_ckpt_path.exists():
        master_ckpt = torch.load(master_ckpt_path, map_location=DEVICE, weights_only=False)
        snapshot_ics = master_ckpt.get("snapshot_ics", None)

    ensemble = SnapshotEnsemble(
        snapshot_paths=snap_paths, top_k=SNAPSHOT_TOP_K, snapshot_ics=snapshot_ics
    ).to(DEVICE)
    ensemble.eval()
    for p in ensemble.parameters():
        p.requires_grad = False
    K = ensemble.n_models
    print(f"  Loaded {K} snapshot models (frozen)")

    # -- Build gating network ---
    gating = EnsembleGating(xd_dim=XD_DIM, n_models=K).to(DEVICE)
    print(f"  Gating params: {sum(p.numel() for p in gating.parameters())} ({XD_DIM} -> {K})")

    # -- Load data (18 features for XD) ---
    print(f"\n  Loading data with {FULL_INPUT_DIM} features...")
    all_segments = load_full_data(DATA_DIR, FULL_FEATURE_LIST, ASSET_TO_IDX, REWARD_HORIZONS)
    if all_segments is None:
        print("[ERROR] No valid data.")
        return False

    splitter = WalkForwardSplitter()
    train_segments, val_segments, oos_segments, unseen_segments = \
        splitter.split_four_way_dated(all_segments)

    train_ds = AntifragileDataset(train_segments, seq_len=WM_SEQ_LEN, reward_horizons=REWARD_HORIZONS, augment=False)
    val_ds = AntifragileDataset(val_segments, seq_len=WM_SEQ_LEN, reward_horizons=REWARD_HORIZONS, augment=False)

    train_loader = DataLoader(train_ds, batch_size=WM_BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=WM_BATCH_SIZE, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # -- Training ---
    GATING_EPOCHS = 20
    GATING_LR = 1e-3

    optimizer = optim.AdamW(gating.parameters(), lr=GATING_LR, weight_decay=0.01)

    best_val_ic = -float("inf")

    for epoch in range(GATING_EPOCHS):
        gating.train()
        epoch_losses = []

        # Cosine LR
        lr = GATING_LR * 0.5 * (1 + math.cos(math.pi * epoch / GATING_EPOCHS))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        for obs, targets, asset in tqdm(train_loader, desc=f"  Ep {epoch+1}/{GATING_EPOCHS}", leave=False):
            obs = obs.to(DEVICE)
            asset = asset.to(DEVICE)
            targets_gpu = {h: t.to(DEVICE) for h, t in targets.items()}

            base_obs = obs[:, :, :INPUT_DIM]   # [B, T, 13]
            xd_last = obs[:, -1, INPUT_DIM:]   # [B, 5]

            # Run all frozen snapshots
            with torch.no_grad():
                all_logits = {}  # {h: [K, B, T, NUM_BINS]}
                for i, model in enumerate(ensemble.models):
                    out = model.forward_train(base_obs, asset)
                    for h in REWARD_HORIZONS:
                        if h not in all_logits:
                            all_logits[h] = []
                        all_logits[h].append(out["return_logits"][h].detach())

            # Gating: XD -> weights
            weights = gating(xd_last)  # [B, K]

            # Weighted average of logits
            total_loss = torch.tensor(0.0, device=DEVICE)
            for h in REWARD_HORIZONS:
                stacked = torch.stack(all_logits[h], dim=1)  # [B, K, T, NUM_BINS]
                w = weights.unsqueeze(-1).unsqueeze(-1)  # [B, K, 1, 1]
                weighted = (stacked * w).sum(dim=1)  # [B, T, NUM_BINS]

                loss_h = ensemble.bucketer.compute_loss(
                    weighted.reshape(-1, NUM_BINS),
                    targets_gpu[h].reshape(-1),
                )
                total_loss = total_loss + loss_h

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(total_loss.item())

        # -- Validation ---
        gating.eval()
        uniform_preds = {h: [] for h in REWARD_HORIZONS}
        gated_preds = {h: [] for h in REWARD_HORIZONS}
        actuals_all = {h: [] for h in REWARD_HORIZONS}

        with torch.no_grad():
            for obs, targets, asset in val_loader:
                obs = obs.to(DEVICE)
                asset = asset.to(DEVICE)

                base_obs = obs[:, :, :INPUT_DIM]
                xd_last = obs[:, -1, INPUT_DIM:]

                all_logits = {}
                for i, model in enumerate(ensemble.models):
                    out = model.forward_train(base_obs, asset)
                    for h in REWARD_HORIZONS:
                        if h not in all_logits:
                            all_logits[h] = []
                        all_logits[h].append(out["return_logits"][h])

                weights = gating(xd_last)

                for h in REWARD_HORIZONS:
                    stacked = torch.stack(all_logits[h], dim=1)

                    # Uniform average
                    uniform_logits = stacked.mean(dim=1)
                    uniform_pred = ensemble.bucketer.decode(uniform_logits).cpu().numpy().flatten()
                    uniform_preds[h].append(uniform_pred)

                    # Gated average
                    w = weights.unsqueeze(-1).unsqueeze(-1)
                    gated_logits = (stacked * w).sum(dim=1)
                    gated_pred = ensemble.bucketer.decode(gated_logits).cpu().numpy().flatten()
                    gated_preds[h].append(gated_pred)

                    actuals_all[h].append(targets[h].numpy().flatten())

        # Compute ICs
        avg_loss = float(np.mean(epoch_losses))
        ic_str_parts = []
        gated_ics = []
        for h in REWARD_HORIZONS:
            up = np.concatenate(uniform_preds[h])
            gp = np.concatenate(gated_preds[h])
            act = np.concatenate(actuals_all[h])

            mask = np.isfinite(up) & np.isfinite(gp) & np.isfinite(act)
            up, gp, act = up[mask], gp[mask], act[mask]

            if len(up) > 30 and np.std(up) > 1e-10 and np.std(gp) > 1e-10 and np.std(act) > 1e-10:
                uniform_ic = float(np.corrcoef(up, act)[0, 1])
                gated_ic = float(np.corrcoef(gp, act)[0, 1])
            else:
                uniform_ic = gated_ic = 0.0

            ic_str_parts.append(f"r{h}: {gated_ic:.4f} ({gated_ic - uniform_ic:+.4f})")
            gated_ics.append(gated_ic)

        mean_gated_ic = float(np.mean(gated_ics))
        print(f"  Ep {epoch+1:2d} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | {' | '.join(ic_str_parts)}")

        save_marker = ""
        if mean_gated_ic > best_val_ic:
            best_val_ic = mean_gated_ic
            torch.save({
                "gating_state_dict": gating.state_dict(),
                "n_models": K,
                "xd_dim": XD_DIM,
                "best_ic": best_val_ic,
                "version": "v1e_ensemble_gating_v1",
            }, MODEL_DIR / "ensemble_gating.pt")
            save_marker = " *BEST*"

        if save_marker:
            print(f"         {save_marker}")

    print(f"\n  Best gated IC: {best_val_ic:.4f}")
    print(f"  Saved to: {MODEL_DIR / 'ensemble_gating.pt'}")
    return True


if __name__ == "__main__":
    success = train_ensemble_gating()
    sys.exit(0 if success else 1)
