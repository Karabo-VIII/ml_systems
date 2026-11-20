"""
Conformal Prediction Calibrator for World Model TwoHot Outputs.

Post-hoc calibration that produces distribution-free prediction intervals
with finite-sample coverage guarantees. Does NOT require retraining.

Usage:
    # 1. Calibrate on a held-out calibration set
    calibrator = ConformalCalibrator(alpha=0.10)
    calibrator.calibrate(model, calibration_dataloader, bucketer, device)

    # 2. At inference, get intervals and abstention decisions
    interval = calibrator.predict_interval(logits, bucketer)
    should_trade = calibrator.should_trade(logits, bucketer, width_threshold=0.005)

    # 3. Evaluate conditional IC on traded vs abstained bars
    report = calibrator.evaluate(model, test_dataloader, bucketer, device)

Design:
    - Uses nonconformity score: |predicted_return - actual_return|
    - Conformal quantile computed on calibration set
    - Prediction interval = point_prediction +/- quantile
    - Abstention: interval_width > threshold means low confidence
    - Adaptive Conformal Inference (ACI) for non-stationary data:
      online adjustment of alpha to maintain target coverage

References:
    - Vovk et al. (2005): Algorithmic Learning in a Random World
    - Gibbs & Candes (2021): Adaptive Conformal Inference Under Distribution Shift
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ConformalCalibrator:
    """
    Distribution-free conformal prediction for TwoHot world model outputs.

    Provides:
        - Calibrated prediction intervals with (1-alpha) coverage
        - Width-based abstention signal for trading decisions
        - Conditional IC analysis (traded vs abstained subsets)
        - Optional ACI for online recalibration under distribution shift
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Miscoverage rate. 0.10 = 90% prediction intervals.
        """
        self.alpha = alpha
        self.q_hat = None           # Conformal quantile (set by calibrate())
        self.scores = None          # Stored calibration nonconformity scores
        self.calibrated = False

        # ACI state (for online recalibration)
        self._aci_alpha = alpha     # Adaptive alpha (updated online)
        self._aci_gamma = 0.005     # ACI step size

    def calibrate(self, model, dataloader, bucketer, device, horizon: int = 1):
        """
        Compute conformal quantile on a calibration set.

        Args:
            model: Trained world model (in eval mode)
            dataloader: Calibration data (must NOT overlap with train or test)
            bucketer: TwoHotSymlog instance
            device: torch device
            horizon: Which return horizon to calibrate (default: h=1)
        """
        model.eval()
        all_scores = []

        with torch.no_grad():
            for batch in dataloader:
                obs = batch["obs"].to(device)
                asset_ids = batch["asset_id"].to(device)
                target_key = f"target_return_{horizon}"

                if target_key not in batch:
                    continue

                targets = batch[target_key].to(device)  # [B, T]

                # Forward pass
                outputs = model.forward_train(obs, asset_ids)

                if horizon in outputs.get("return_logits", {}):
                    logits = outputs["return_logits"][horizon]  # [B, T, 255]
                    preds = bucketer.decode(logits)             # [B, T]

                    # Nonconformity score: absolute residual
                    scores = (preds - targets).abs()            # [B, T]

                    # Flatten, filter out NaN/padding
                    scores_flat = scores.reshape(-1)
                    valid = ~torch.isnan(scores_flat) & ~torch.isinf(scores_flat)
                    all_scores.append(scores_flat[valid].cpu().numpy())

        if not all_scores:
            raise ValueError("No valid calibration scores computed. Check dataloader and model.")

        self.scores = np.concatenate(all_scores)
        n = len(self.scores)

        # Conformal quantile: ceil((n+1)(1-alpha)) / n quantile
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self.q_hat = float(np.quantile(self.scores, q_level))
        self.calibrated = True

        print(f"  [Conformal] Calibrated on {n:,} samples")
        print(f"  [Conformal] alpha={self.alpha}, q_hat={self.q_hat:.6f}")
        print(f"  [Conformal] Score stats: mean={self.scores.mean():.6f}, "
              f"median={np.median(self.scores):.6f}, p90={np.percentile(self.scores, 90):.6f}")

        return self.q_hat

    def predict_interval(self, logits: torch.Tensor, bucketer) -> dict:
        """
        Compute prediction interval from TwoHot logits.

        Args:
            logits: [*, 255] raw logits from return head
            bucketer: TwoHotSymlog instance

        Returns:
            dict with keys:
                point: [*] point prediction
                lower: [*] lower interval bound
                upper: [*] upper interval bound
                width: [*] interval width
                distribution_width: [*] width from softmax distribution (model uncertainty)
        """
        if not self.calibrated:
            raise RuntimeError("Must call calibrate() before predict_interval()")

        point = bucketer.decode(logits)  # [*]

        # Conformal interval
        lower = point - self.q_hat
        upper = point + self.q_hat
        width = torch.full_like(point, 2 * self.q_hat)

        # Distribution-based width: interquartile range of the softmax
        # This captures model-specific uncertainty per prediction
        probs = F.softmax(logits.float(), dim=-1)
        cdf = torch.cumsum(probs, dim=-1)

        # Find 10th and 90th percentile bins
        p10_idx = (cdf >= 0.10).float().argmax(dim=-1)  # first bin where CDF >= 0.10
        p90_idx = (cdf >= 0.90).float().argmax(dim=-1)

        # Convert bin indices to values via bucket positions
        buckets = bucketer.buckets  # [255]
        p10_val = bucketer.symexp(buckets[p10_idx.long()])
        p90_val = bucketer.symexp(buckets[p90_idx.long()])
        dist_width = (p90_val - p10_val).abs()

        return {
            "point": point,
            "lower": lower,
            "upper": upper,
            "width": width,
            "distribution_width": dist_width,
        }

    def should_trade(self, logits: torch.Tensor, bucketer,
                     width_threshold: Optional[float] = None,
                     use_distribution_width: bool = True) -> torch.Tensor:
        """
        Abstention decision: trade only when prediction is confident.

        Args:
            logits: [*, 255] raw logits
            bucketer: TwoHotSymlog instance
            width_threshold: Max interval width to accept trade.
                If None, uses median distribution_width from calibration.
            use_distribution_width: If True, use per-prediction softmax width
                (model-specific). If False, use conformal interval width (global).

        Returns:
            [*] boolean tensor: True = trade, False = abstain
        """
        interval = self.predict_interval(logits, bucketer)

        if use_distribution_width:
            width = interval["distribution_width"]
        else:
            width = interval["width"]

        if width_threshold is None:
            # Default: trade top 30% most confident predictions
            # (narrowest distribution width)
            if self.scores is not None:
                width_threshold = float(np.percentile(self.scores, 30))
            else:
                width_threshold = self.q_hat

        return width <= width_threshold

    def update_aci(self, covered: bool):
        """
        Adaptive Conformal Inference: update alpha online based on coverage.

        Call after each prediction with whether the true value fell in the interval.
        Adjusts alpha to maintain target coverage under distribution shift.

        Args:
            covered: True if actual value was within predicted interval
        """
        # ACI update: alpha_t+1 = alpha_t + gamma * (alpha - err_t)
        err_t = 0.0 if covered else 1.0
        self._aci_alpha = self._aci_alpha + self._aci_gamma * (self.alpha - err_t)
        self._aci_alpha = max(0.001, min(0.50, self._aci_alpha))  # clamp

        # Recompute q_hat with adapted alpha
        if self.scores is not None:
            n = len(self.scores)
            q_level = np.ceil((n + 1) * (1 - self._aci_alpha)) / n
            q_level = min(q_level, 1.0)
            self.q_hat = float(np.quantile(self.scores, q_level))

    def evaluate(self, model, dataloader, bucketer, device, horizon: int = 1,
                 width_percentiles: list = None) -> dict:
        """
        Evaluate conditional IC on traded vs abstained subsets at various thresholds.

        Args:
            model: Trained world model (eval mode)
            dataloader: Test set dataloader
            bucketer: TwoHotSymlog instance
            device: torch device
            horizon: Return horizon (default: h=1)
            width_percentiles: Distribution width percentiles to test as thresholds.
                Default: [10, 20, 30, 50, 70, 100] (100 = no abstention baseline)

        Returns:
            dict with per-threshold IC analysis
        """
        if width_percentiles is None:
            width_percentiles = [10, 20, 30, 50, 70, 100]

        model.eval()
        all_preds = []
        all_targets = []
        all_widths = []

        with torch.no_grad():
            for batch in dataloader:
                obs = batch["obs"].to(device)
                asset_ids = batch["asset_id"].to(device)
                target_key = f"target_return_{horizon}"

                if target_key not in batch:
                    continue

                targets = batch[target_key].to(device)
                outputs = model.forward_train(obs, asset_ids)

                if horizon in outputs.get("return_logits", {}):
                    logits = outputs["return_logits"][horizon]
                    interval = self.predict_interval(logits, bucketer)

                    preds_flat = interval["point"].reshape(-1).cpu().numpy()
                    targets_flat = targets.reshape(-1).cpu().numpy()
                    widths_flat = interval["distribution_width"].reshape(-1).cpu().numpy()

                    valid = ~np.isnan(preds_flat) & ~np.isnan(targets_flat) & ~np.isinf(widths_flat)
                    all_preds.append(preds_flat[valid])
                    all_targets.append(targets_flat[valid])
                    all_widths.append(widths_flat[valid])

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        widths = np.concatenate(all_widths)

        # Compute IC at each threshold
        results = {}
        for pct in width_percentiles:
            if pct >= 100:
                mask = np.ones(len(preds), dtype=bool)
                threshold = float("inf")
            else:
                threshold = np.percentile(widths, pct)
                mask = widths <= threshold

            n_traded = mask.sum()
            trade_rate = n_traded / len(preds)

            if n_traded < 10:
                ic = float("nan")
            else:
                # Pearson IC
                p_sel = preds[mask]
                t_sel = targets[mask]
                if p_sel.std() < 1e-12 or t_sel.std() < 1e-12:
                    ic = 0.0
                else:
                    ic = float(np.corrcoef(p_sel, t_sel)[0, 1])

            results[f"p{pct}"] = {
                "threshold": threshold,
                "n_traded": int(n_traded),
                "trade_rate": float(trade_rate),
                "ic": ic,
            }

        # Print summary
        print(f"\n  [Conformal] Conditional IC analysis (h={horizon}, n={len(preds):,})")
        print(f"  {'Percentile':>10} {'Threshold':>10} {'Traded':>8} {'Rate':>6} {'IC':>8}")
        print(f"  {'-'*44}")
        for key, val in results.items():
            print(f"  {key:>10} {val['threshold']:>10.6f} {val['n_traded']:>8,} "
                  f"{val['trade_rate']:>6.1%} {val['ic']:>8.4f}")

        return results

    def save(self, path: str):
        """Save calibration state to disk."""
        np.savez(
            path,
            scores=self.scores,
            q_hat=self.q_hat,
            alpha=self.alpha,
            aci_alpha=self._aci_alpha,
            calibrated=self.calibrated,
        )

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        """Load calibration state from disk."""
        data = np.load(path)
        cal = cls(alpha=float(data["alpha"]))
        cal.scores = data["scores"]
        cal.q_hat = float(data["q_hat"])
        cal._aci_alpha = float(data["aci_alpha"])
        cal.calibrated = bool(data["calibrated"])
        return cal
