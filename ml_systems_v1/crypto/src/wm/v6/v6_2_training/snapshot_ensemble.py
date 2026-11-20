"""
V6.E Seed Ensemble -- Inference wrapper for multi-seed ensemble models.

Loads K independently-trained seed models, runs each on input,
averages return predictions for improved IC via diversity.

IC_ensemble = IC_single * sqrt(K / (1 + (K-1) * rho))
With rho ~ 0.3-0.5 between seeds, K=3-5 gives ~30-50% IC boost.

V6-specific:
  - Uses CausalJEPAWorldModel (CausalGRU + contrastive + VICReg + TimeDiscriminator)
  - encode_sequence returns (ctx_latent, return_preds) -- JEPA-style (no RSSM)
  - No prior/posterior logits (JEPA has no stochastic latent)
  - Output dict has ctx_latent, tgt_latent, pred_latent instead of h_seq, z_post
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
from world_model import CausalJEPAWorldModel
from revin import RevIN


class SnapshotEnsemble(nn.Module):
    """
    Ensemble of K seed models from independent training runs.

    All models share the same architecture (CausalJEPAWorldModel),
    loaded from different checkpoints (different random seeds -> different basins).

    Predictions are averaged across seeds (equal weighting).
    """

    def __init__(self, snapshot_paths: list = None, top_k: int = ENSEMBLE_TOP_K,
                 snapshot_ics: dict = None):
        super().__init__()
        if snapshot_paths is None:
            snap_dir = ENSEMBLE_MODEL_DIR
            # Prefer best-ShIC snapshots (peak anti-memorization performance)
            snapshot_paths = sorted(snap_dir.glob("v6_2e_seed_*_best.pt"))
            if not snapshot_paths:
                # Fallback to final models (backward compatibility)
                snapshot_paths = sorted(snap_dir.glob("v6_2_seed_*.pt"))

        if not snapshot_paths:
            raise FileNotFoundError("No seed models found. Run train_snapshot.py first.")

        # Rank snapshots by IC if available, otherwise by recency
        if snapshot_ics and len(snapshot_paths) > top_k:
            ranked = []
            for p in snapshot_paths:
                try:
                    cidx = int(p.stem.split("_")[-1])
                    ic_val = snapshot_ics.get(cidx, 0.0)
                except (ValueError, IndexError):
                    ic_val = 0.0
                ranked.append((p, ic_val))
            ranked.sort(key=lambda x: x[1], reverse=True)
            snapshot_paths = [r[0] for r in ranked[:top_k]]
            print(f"  [OK] Selected top-{top_k} snapshots by IC:")
            for p, ic in ranked[:top_k]:
                print(f"       - {p.name} (IC={ic:.4f})")
        elif len(snapshot_paths) > top_k:
            snapshot_paths = snapshot_paths[-top_k:]

        self.n_models = len(snapshot_paths)
        self.models = nn.ModuleList()
        self._revins = []  # Per-snapshot RevIN (may be None for legacy snapshots)
        self._revin_modules = nn.ModuleList()  # For .to(device) support
        self.bucketer = TwoHotSymlog(NUM_BINS, BIN_MIN, BIN_MAX, DEVICE)

        for path in snapshot_paths:
            model = CausalJEPAWorldModel()
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

            # Handle both old (bare state_dict) and new (dict with keys) formats
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                if "revin_state_dict" in ckpt:
                    rv = RevIN(num_features=INPUT_DIM)
                    rv.load_state_dict(ckpt["revin_state_dict"])
                    rv.eval()
                    for p_rv in rv.parameters():
                        p_rv.requires_grad = False
                    self._revins.append(rv)
                    self._revin_modules.append(rv)
                else:
                    self._revins.append(None)
            else:
                # Legacy bare state_dict (pre-RevIN snapshots)
                model.load_state_dict(ckpt)
                self._revins.append(None)

            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            self.models.append(model)

        print(f"  [OK] Loaded {self.n_models} seed models for ensemble")
        n_with_revin = sum(1 for r in self._revins if r is not None)
        if n_with_revin > 0:
            print(f"       {n_with_revin}/{self.n_models} seeds have RevIN")
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
        for i, model in enumerate(self.models):
            # Apply per-snapshot RevIN if available
            obs_in = obs_seq
            masked_in = masked_obs_seq
            if self._revins[i] is not None:
                obs_in = self._revins[i](obs_seq, mode='norm')
                if masked_obs_seq is not None:
                    masked_in = self._revins[i](masked_obs_seq, mode='norm')
            out = model.forward_train(obs_in, asset_id, masked_in)
            all_outputs.append(out)

        # Average return logits across snapshots
        avg_return_logits = {}
        for h in REWARD_HORIZONS:
            logits_stack = torch.stack([out["return_logits"][h] for out in all_outputs])
            avg_return_logits[h] = logits_stack.mean(dim=0)

        # Use first model's non-return outputs (JEPA latent keys)
        base = all_outputs[0]
        return {
            "recon": base["recon"],
            "return_logits": avg_return_logits,
            "regime_logits": base["regime_logits"],
            "ctx_latent": base["ctx_latent"],
            "tgt_latent": base["tgt_latent"],
            "pred_latent": base["pred_latent"],
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
            ctx_latent:   [B, T, d_latent] -- from first model
            return_preds: dict of {horizon: [B, T]} -- ensemble averaged
        """
        outputs = self.forward_train(obs_seq, asset_id)
        return_preds = {}
        for h in REWARD_HORIZONS:
            return_preds[h] = self.bucketer.decode(outputs["return_logits"][h])
        return outputs["ctx_latent"], return_preds

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

        # Reconstruction: match recon dim (base_dim may < INPUT_DIM)
        recon_dim = outputs["recon"].shape[-1]
        recon_target = obs_seq if recon_dim == obs_seq.shape[-1] else obs_seq[:, :, :recon_dim]
        if self._revins[0] is not None:
            recon_target = self._revins[0](recon_target, mode='norm')
        l_rec = F.mse_loss(outputs["recon"], recon_target)
        loss_dict["rec"] = l_rec.item()
        loss_dict["total"] = total_loss.item()
        loss_dict["contrastive"] = 0.0      # Not meaningful for ensemble
        loss_dict["contrastive_acc"] = 0.0  # Not meaningful for ensemble
        loss_dict["vicreg"] = 0.0           # Not meaningful for ensemble
        loss_dict["regime"] = 0.0           # Not computed for ensemble
        loss_dict["regime_acc"] = 0.0       # Not computed for ensemble

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
