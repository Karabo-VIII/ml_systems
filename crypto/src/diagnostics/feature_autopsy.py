"""
Feature Autopsy System -- Per-feature diagnostics for world model training.

Writes structured JSONL logs (NOT console) with per-feature metrics.
Enables model "autopsy" at any training epoch to diagnose which features
drive temporal memorization vs cross-sectional signal.

Metrics computed:
  CHEAP (every validation, ~15 seconds):
    1. Per-feature reconstruction MSE -- which features the model learns to
       reconstruct (temporal signal proxy: better reconstruction = stronger
       temporal pattern learned)
    2. Per-feature input gradient magnitude -- which features the model
       "attends to" for return prediction (gradient w.r.t. r1 loss)

  MEDIUM (configurable, ~3 minutes):
    3. Feature group ablation IC -- zero out feature groups, measure IC change.
       Groups: vol, returns, microstructure, regime, new_base, xd.
       IC drop = causal importance. If zeroing vol features doesn't drop ShIC
       but drops IC, vol features contribute only temporal signal.

  ONCE (at training start, ~30 seconds):
    4. Per-feature raw IC -- direct correlation of each input feature with
       h=1 target. Baseline importance without model.

Output: JSONL file (one JSON record per epoch), machine-readable for
        plotting and automated analysis.
"""
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Semantic feature groups for ablation analysis.
# Keys are group names, values are feature column names.
# Only features present in the model's FEATURE_LIST are used.
FEATURE_GROUPS = {
    "vol": ["norm_hl_spread", "norm_vol_cluster", "norm_deviation"],
    "returns": ["norm_return_1", "norm_return_4", "norm_return_16"],
    "microstructure": [
        "norm_vpin", "norm_flow_imbalance", "norm_spread_bps",
        "norm_tick_count", "norm_log_volume",
    ],
    "regime": ["norm_fd_close", "hurst_regime", "norm_funding", "norm_oi_change"],
    "new_base": ["norm_whale", "norm_efficiency"],
    "xd": [
        "xd_btc_return", "xd_btc_volatility", "xd_funding_spread",
        "xd_cross_return_mean", "xd_cross_vol_mean", "xd_ma_distance",
    ],
}


class FeatureAutopsy:
    """Non-console diagnostic logger for per-feature model analysis.

    All output goes to a JSONL file. Nothing is printed to console.
    Designed for minimal compute overhead when integrated into training loops.
    """

    def __init__(
        self,
        feature_list: List[str],
        base_dim: int,
        log_path,
        horizons: List[int] = None,
        device: str = "cuda",
    ):
        """
        Args:
            feature_list: Names of input features (ordered as in model input).
            base_dim:     Number of base features (decoder reconstructs these).
            log_path:     Path for JSONL output file.
            horizons:     Return prediction horizons (default [1, 4, 16, 64]).
            device:       Torch device.
        """
        self.feature_list = list(feature_list)
        self.base_dim = base_dim
        self.n_features = len(feature_list)
        self.log_path = Path(log_path)
        self.horizons = horizons or [1, 4, 16, 64]
        self.device = device

        # Build feature name -> index maps for groups
        self.feat_to_idx = {name: i for i, name in enumerate(feature_list)}
        self.group_indices = {}
        for group_name, group_feats in FEATURE_GROUPS.items():
            indices = [self.feat_to_idx[f] for f in group_feats
                       if f in self.feat_to_idx]
            if indices:
                self.group_indices[group_name] = indices

    def _write_record(self, record: dict):
        """Append one JSON record to the JSONL log file."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")

    # ------------------------------------------------------------------
    # Metric 1: Per-feature reconstruction MSE
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_per_feature_rec_mse(
        self, model, val_loader, revin=None, max_batches: int = 50,
    ) -> Dict[str, float]:
        """Decompose reconstruction MSE by feature dimension.

        The decoder reconstructs base_dim features. Per-feature MSE shows
        which features the model has learned temporal patterns for (low MSE =
        strong temporal prediction of that feature).
        """
        model.eval()
        per_feat_sum = torch.zeros(self.base_dim, device=self.device)
        n_batches = 0

        for i, (obs, targets, asset) in enumerate(val_loader):
            if i >= max_batches:
                break
            obs = obs.to(self.device)
            asset = asset.to(self.device)
            targets_gpu = {h: t.to(self.device) for h, t in targets.items()}

            if revin is not None:
                obs = revin(obs, mode="norm")

            with torch.amp.autocast("cuda"):
                _, _, outputs = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=0.0,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            recon = outputs["recon"].float()          # [B, T, base_dim]
            actual = obs[:, :, :self.base_dim].float()
            per_feat = ((recon - actual) ** 2).mean(dim=(0, 1))  # [base_dim]
            per_feat_sum += per_feat
            n_batches += 1

        if n_batches > 0:
            per_feat_sum /= n_batches

        result = {}
        for i in range(self.base_dim):
            fname = self.feature_list[i] if i < len(self.feature_list) else f"feat_{i}"
            result[fname] = round(per_feat_sum[i].item(), 6)
        return result

    # ------------------------------------------------------------------
    # Metric 2: Per-feature gradient magnitude
    # ------------------------------------------------------------------
    def compute_per_feature_gradient(
        self, model, val_loader, revin=None, n_batches: int = 3,
    ) -> Dict[str, float]:
        """Gradient magnitude w.r.t. each input feature from h=1 return loss.

        High gradient = the model relies on this feature for return prediction.
        Tracks which features drive the return prediction pathway.

        Uses AMP for forward pass (memory efficient) with fp32 input gradients.
        Only processes n_batches to limit compute cost.
        """
        model.eval()
        grad_accum = torch.zeros(self.n_features, device=self.device)
        n_computed = 0

        for i, (obs, targets, asset) in enumerate(val_loader):
            if i >= n_batches:
                break

            obs = obs.to(self.device).float()
            obs.requires_grad_(True)
            asset = asset.to(self.device)
            targets_gpu = {h: t.to(self.device) for h, t in targets.items()}

            if revin is not None:
                obs_in = revin(obs, mode="norm")
            else:
                obs_in = obs

            # Forward with AMP, backward in fp32 for input gradients
            with torch.amp.autocast("cuda"):
                _, _, outputs = model.get_loss(
                    obs_in, asset, targets_gpu, mask_ratio=0.0,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            # Compute gradient from h=1 return loss specifically
            r1_logits = outputs["return_logits"][1]
            r1_logits_flat = r1_logits.reshape(-1, r1_logits.shape[-1]).float()
            r1_targets_flat = targets_gpu[1].reshape(-1)
            r1_loss = model.bucketer.compute_loss(r1_logits_flat, r1_targets_flat)
            r1_loss.backward()

            if obs.grad is not None:
                # Per-feature: mean absolute gradient over (B, T)
                grad_per_feat = obs.grad.abs().mean(dim=(0, 1))  # [n_features]
                grad_accum += grad_per_feat.detach()
                n_computed += 1

            # Cleanup
            model.zero_grad()
            if obs.grad is not None:
                obs.grad = None

        if n_computed > 0:
            grad_accum /= n_computed

        result = {}
        for i, fname in enumerate(self.feature_list):
            result[fname] = round(grad_accum[i].item(), 6)
        return result

    # ------------------------------------------------------------------
    # Metric 3: Feature group ablation IC
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_group_ablation_ic(
        self, model, val_loader, revin=None,
    ) -> Dict[str, dict]:
        """Zero out each feature group and measure IC change.

        IC drop when zeroing a group = that group's causal importance for
        return prediction. Compare with ShIC trajectory to determine if
        the group contributes temporal vs cross-sectional signal.
        """
        model.eval()

        # Baseline IC with all features
        baseline_ic = self._compute_ic(model, val_loader, revin, mask_indices=None)

        results = {
            "baseline": {f"ic_{h}": round(baseline_ic.get(f"ic_{h}", 0), 6)
                         for h in self.horizons},
        }

        for group_name, indices in self.group_indices.items():
            ablated_ic = self._compute_ic(model, val_loader, revin, mask_indices=indices)
            results[group_name] = {
                "ic_1": round(ablated_ic.get("ic_1", 0), 6),
                "ic_1_drop": round(
                    baseline_ic.get("ic_1", 0) - ablated_ic.get("ic_1", 0), 6
                ),
                "ic_avg_drop": round(
                    np.mean([baseline_ic.get(f"ic_{h}", 0) for h in self.horizons])
                    - np.mean([ablated_ic.get(f"ic_{h}", 0) for h in self.horizons]),
                    6,
                ),
                "features_zeroed": [self.feature_list[idx] for idx in indices],
            }

        return results

    @torch.no_grad()
    def _compute_ic(
        self, model, val_loader, revin=None, mask_indices=None,
    ) -> Dict[str, float]:
        """Run validation with optional feature zeroing, return per-horizon IC."""
        ic_data = {h: {"preds": [], "reals": []} for h in self.horizons}

        for obs, targets, asset in val_loader:
            obs = obs.to(self.device)
            asset = asset.to(self.device)
            targets_gpu = {h: t.to(self.device) for h, t in targets.items()}

            if mask_indices is not None:
                obs = obs.clone()
                for idx in mask_indices:
                    obs[:, :, idx] = 0.0

            if revin is not None:
                obs = revin(obs, mode="norm")

            with torch.amp.autocast("cuda"):
                _, _, outputs = model.get_loss(
                    obs, asset, targets_gpu, mask_ratio=0.0,
                    regime_labels=targets_gpu.get("regime_label"),
                )

            for h in self.horizons:
                pred_ret = model.bucketer.decode(outputs["return_logits"][h])
                ic_data[h]["preds"].append(pred_ret.cpu().numpy().flatten())
                ic_data[h]["reals"].append(targets[h].cpu().numpy().flatten())

        result = {}
        for h in self.horizons:
            all_preds = np.concatenate(ic_data[h]["preds"])
            all_reals = np.concatenate(ic_data[h]["reals"])
            mask = np.isfinite(all_preds) & np.isfinite(all_reals)
            if mask.sum() > 100:
                result[f"ic_{h}"] = float(
                    np.corrcoef(all_preds[mask], all_reals[mask])[0, 1]
                )
            else:
                result[f"ic_{h}"] = 0.0
        return result

    # ------------------------------------------------------------------
    # Metric 4: Raw feature-target correlation (no model)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_raw_feature_ic(self, val_loader) -> Dict[str, float]:
        """Correlate each input feature directly with h=1 target.

        This is the "floor" importance -- how much each feature correlates
        with the target without any model. Features with high raw IC are
        the ones the model can most easily exploit.
        """
        all_features = []
        all_targets = []

        for obs, targets, asset in val_loader:
            # Keep on CPU for memory efficiency
            all_features.append(obs.numpy().reshape(-1, obs.shape[-1]))
            if 1 in targets:
                all_targets.append(targets[1].numpy().flatten())

        if not all_targets:
            return {}

        features = np.concatenate(all_features, axis=0)
        tgt = np.concatenate(all_targets, axis=0)
        valid_tgt = np.isfinite(tgt)

        result = {}
        for i, fname in enumerate(self.feature_list):
            feat_vals = features[valid_tgt, i]
            tgt_vals = tgt[valid_tgt]
            valid = np.isfinite(feat_vals)
            if valid.sum() > 100:
                result[fname] = round(
                    float(np.corrcoef(feat_vals[valid], tgt_vals[valid])[0, 1]),
                    6,
                )
            else:
                result[fname] = 0.0
        return result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        model,
        val_loader,
        epoch: int,
        revin=None,
        do_ablation: bool = False,
        do_raw_ic: bool = False,
    ) -> dict:
        """Run autopsy diagnostics and write to JSONL log.

        Args:
            model:        The world model (eval mode expected).
            val_loader:   Validation DataLoader.
            epoch:        Current epoch number.
            revin:        Optional RevIN module.
            do_ablation:  Run feature group ablation IC (expensive, ~3 min).
            do_raw_ic:    Run raw feature-target IC (once at start).

        Returns:
            The autopsy record dict (also written to JSONL file).
        """
        record = {
            "epoch": epoch,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_features": self.n_features,
            "base_dim": self.base_dim,
        }

        try:
            # CHEAP: per-feature reconstruction MSE (~10 sec)
            record["rec_mse_per_feat"] = self.compute_per_feature_rec_mse(
                model, val_loader, revin,
            )
        except Exception as e:
            record["rec_mse_error"] = str(e)

        try:
            # CHEAP: per-feature gradient norm (~10 sec)
            record["grad_norm_per_feat"] = self.compute_per_feature_gradient(
                model, val_loader, revin,
            )
        except Exception as e:
            record["grad_norm_error"] = str(e)

        if do_ablation:
            try:
                # MEDIUM: feature group ablation IC (~3 min)
                record["group_ablation"] = self.compute_group_ablation_ic(
                    model, val_loader, revin,
                )
            except Exception as e:
                record["ablation_error"] = str(e)

        if do_raw_ic:
            try:
                # ONCE: raw feature-target IC (~30 sec)
                record["raw_feature_ic"] = self.compute_raw_feature_ic(val_loader)
            except Exception as e:
                record["raw_ic_error"] = str(e)

        self._write_record(record)
        return record


def _json_default(obj):
    """JSON serializer for numpy/torch types."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")
