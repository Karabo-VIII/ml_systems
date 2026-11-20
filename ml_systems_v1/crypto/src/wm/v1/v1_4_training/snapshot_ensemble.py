"""
V1.E Snapshot Ensemble -- Inference wrapper for ensembling snapshot models.

Loads K snapshots from cyclical cosine training, runs each on input,
averages return predictions for improved IC via diversity.

IC_ensemble = IC_single * sqrt(K / (1 + (K-1) * rho))
With rho ~ 0.3-0.5 between snapshots, K=3-5 gives ~30-50% IC boost.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from components import TwoHotSymlog
from world_model import TransformerWorldModel


class SnapshotEnsemble(nn.Module):
    """
    Ensemble of K snapshot models from cyclical training.

    All snapshots share the same architecture (TransformerWorldModel),
    loaded from different checkpoints (different local minima).

    Predictions are averaged across snapshots (equal weighting).
    """

    def __init__(self, snapshot_paths: list = None, top_k: int = SNAPSHOT_TOP_K,
                 snapshot_ics: dict = None):
        super().__init__()
        if snapshot_paths is None:
            snap_dir = MODEL_DIR / "snapshots"
            snapshot_paths = sorted(snap_dir.glob("v1_4_snapshot_*.pt"))

        if not snapshot_paths:
            raise FileNotFoundError("No snapshots found. Run train_snapshot.py first.")

        # Rank snapshots by ShIC if available, otherwise by recency
        if snapshot_ics and len(snapshot_paths) > top_k:
            # Build (path, shic) pairs, sort by ShIC descending
            ranked = []
            for p in snapshot_paths:
                try:
                    cidx = int(p.stem.split("_")[-1])
                    shic = snapshot_ics.get(cidx, 0.0)
                except (ValueError, IndexError):
                    shic = 0.0
                ranked.append((p, shic))
            ranked.sort(key=lambda x: x[1], reverse=True)
            snapshot_paths = [r[0] for r in ranked[:top_k]]
            print(f"  [OK] Selected top-{top_k} snapshots by ShIC:")
            for p, ic in ranked[:top_k]:
                print(f"       - {p.name} (ShIC={ic:.4f})")
        elif len(snapshot_paths) > top_k:
            snapshot_paths = snapshot_paths[-top_k:]

        self.n_models = len(snapshot_paths)
        self.models = nn.ModuleList()
        self.bucketer = TwoHotSymlog(NUM_BINS, BIN_MIN, BIN_MAX, DEVICE)

        for path in snapshot_paths:
            model = TransformerWorldModel()
            model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.models.append(model)

        print(f"  [OK] Loaded {self.n_models} snapshot models for ensemble")
        if not snapshot_ics:
            for p in snapshot_paths:
                print(f"       - {p.name}")

    @torch.no_grad()
    def forward_train(self, obs_seq, asset_id, masked_obs_seq=None):
        """
        Run all snapshots, average return predictions.
        Other outputs (recon, regime, latent) from first model only.

        Args:
            obs_seq:        [B, T, INPUT_DIM] -- feature sequences
            asset_id:       [B] -- integer asset indices
            masked_obs_seq: [B, T, INPUT_DIM] -- masked observations (optional)

        Returns:
            Dict with averaged return_logits and first model's other outputs.
        """
        all_outputs = []
        for model in self.models:
            out = model.forward_train(obs_seq, asset_id, masked_obs_seq)
            all_outputs.append(out)

        # Average return logits across snapshots
        avg_return_logits = {}
        for h in REWARD_HORIZONS:
            logits_stack = torch.stack([out["return_logits"][h] for out in all_outputs])
            avg_return_logits[h] = logits_stack.mean(dim=0)

        # Use first model's non-return outputs
        base = all_outputs[0]
        return {
            "recon": base["recon"],
            "return_logits": avg_return_logits,
            "regime_logits": base["regime_logits"],
            "prior_logits": base["prior_logits"],
            "post_logits": base["post_logits"],
            "h_seq": base["h_seq"],
            "z_post": base["z_post"],
            "ret_trunk": base["ret_trunk"],
        }

    @torch.no_grad()
    def encode_sequence(self, obs_seq, asset_id):
        """
        Encode with ensemble averaging for predictions.

        Args:
            obs_seq:  [B, T, INPUT_DIM]
            asset_id: [B]

        Returns:
            h_seq:        [B, T, d_model] -- from first model
            z_post:       [B, T, flat_dim] -- from first model
            return_preds: dict of {horizon: [B, T]} -- ensemble averaged
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in REWARD_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["h_seq"], outputs["z_post"], return_preds

    def get_loss(self, obs_seq, asset_id, target_returns, mask_ratio=0.0, block_mask=False):
        """
        Compute loss using ensemble predictions (for validation only).

        Args:
            obs_seq:        [B, T, INPUT_DIM]
            asset_id:       [B]
            target_returns: dict of {int_horizon: [B, T]} tensors
            mask_ratio:     fraction of timesteps to mask (default 0 for eval)
            block_mask:     if True, mask contiguous blocks

        Returns:
            (total_loss, loss_dict, outputs)
        """
        outputs = self.forward_train(obs_seq, asset_id)

        total_loss = torch.tensor(0.0, device=obs_seq.device)
        loss_dict = {}

        # Use first model's bucketer for loss computation
        bucketer = self.models[0].bucketer

        for h in REWARD_HORIZONS:
            logits = outputs["return_logits"][h].reshape(-1, NUM_BINS)
            targets = target_returns[h].reshape(-1)
            l_h = bucketer.compute_loss(logits, targets)
            loss_dict[f"ret_{h}"] = l_h.item()
            total_loss = total_loss + l_h

        # Reconstruction (decoder outputs base_dim features, clip obs_seq to match)
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        l_rec = F.mse_loss(outputs["recon"], recon_target)
        loss_dict["rec"] = l_rec.item()
        loss_dict["total"] = total_loss.item()
        loss_dict["kl"] = 0.0  # Not meaningful for ensemble

        return total_loss, loss_dict, outputs


if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    try:
        ensemble = SnapshotEnsemble().to(DEVICE)
        print(f"Ensemble with {ensemble.n_models} snapshots")

        # Test forward
        B, T = 4, WM_SEQ_LEN
        obs = torch.randn(B, T, INPUT_DIM).to(DEVICE)
        asset = torch.randint(0, NUM_ASSETS, (B,)).to(DEVICE)

        outputs = ensemble.forward_train(obs, asset)
        for h in REWARD_HORIZONS:
            print(f"  Return t+{h}: {outputs['return_logits'][h].shape}")

        print("[OK] Snapshot ensemble sanity check passed.")
    except FileNotFoundError as e:
        print(f"[WARN] {e}")
        print("Run train_snapshot.py first to generate snapshots.")
