"""
Robust Validation Framework for World Models

Prevents overfitting detection via multiple validation strategies:
  1. Temporal Forward Walk (expanding window)
  2. Shuffled K-Fold (non-temporal split)
  3. Regime-Specific Holdout
  4. Stability Analysis (prediction variance across folds)

Usage:
    from validation_utils import RobustValidator

    validator = RobustValidator(model, data_loader)
    results = validator.run_comprehensive_validation()

    if results["hallucination_score"] > 0.3:
        print("WARNING: Model likely overfitting!")
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from scipy import stats as scipy_stats
from tqdm import tqdm


@dataclass
class ValidationConfig:
    """Configuration for robust validation."""

    # Temporal forward walk
    n_forward_splits: int = 5          # Number of expanding window splits
    min_train_ratio: float = 0.50      # Minimum training data for first split

    # Shuffled K-fold
    n_shuffled_folds: int = 5          # K-fold splits (non-temporal)

    # Regime-specific holdout
    regime_holdout_pct: float = 0.20   # Hold out 20% of each regime

    # Stability thresholds
    max_ic_std_ratio: float = 0.40     # Max IC std/mean ratio (stability check)
    max_mse_coeff_var: float = 0.30    # Max MSE coefficient of variation

    # Hallucination detection
    hallucination_threshold: float = 0.3  # Score > 0.3 = likely overfitting


@dataclass
class RobustValidationResults:
    """Comprehensive validation results."""

    # Temporal forward walk
    forward_walk_ics: List[float]
    forward_walk_mses: List[float]

    # Shuffled K-fold
    shuffled_ics: List[float]
    shuffled_mses: List[float]

    # Regime-specific
    regime_ics: Dict[str, float]
    regime_mses: Dict[str, float]

    # Stability metrics
    ic_stability: float          # 1 - (std/mean) of ICs across folds
    mse_stability: float         # 1 - (std/mean) of MSEs

    # Hallucination score
    hallucination_score: float   # 0-1, higher = more suspicious
    hallucination_reasons: List[str]

    # Comparison to baseline split
    baseline_ic: float
    baseline_mse: float
    ic_degradation: float        # (baseline - robust) / baseline

    def passes_robustness_gates(self, config: ValidationConfig) -> bool:
        """Check if model passes all robustness gates."""
        gates = {
            "IC Stability": self.ic_stability > (1 - config.max_ic_std_ratio),
            "MSE Stability": self.mse_stability > (1 - config.max_mse_coeff_var),
            "Hallucination": self.hallucination_score < config.hallucination_threshold,
            "IC Degradation": self.ic_degradation < 0.50,  # IC shouldn't drop >50%
        }
        return all(gates.values()), gates


class RobustValidator:
    """
    Comprehensive validation framework to detect overfitting and hallucination.

    Detects common failure modes:
      - Memorization: High IC on contiguous split, low IC on shuffled splits
      - Temporal leakage: IC degrades significantly in forward walk
      - Regime brittleness: Strong IC in one regime, weak in others
      - High variance: Predictions unstable across different validation folds
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: ValidationConfig = None,
    ):
        self.model = model
        self.config = config or ValidationConfig()
        self.device = next(model.parameters()).device

    def compute_ic_and_mse(
        self,
        preds: np.ndarray,
        reals: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute IC and MSE for predictions vs reals."""
        # Handle shape mismatch due to sliding window
        min_len = min(len(preds), len(reals))
        preds = preds[:min_len]
        reals = reals[:min_len]

        mask = np.isfinite(preds) & np.isfinite(reals)
        p, r = preds[mask], reals[mask]

        if len(p) < 30:
            return 0.0, float('inf')

        ic = float(np.corrcoef(p, r)[0, 1]) if np.std(p) > 1e-10 and np.std(r) > 1e-10 else 0.0
        mse = float(np.mean((p - r) ** 2))

        return ic, mse

    def temporal_forward_walk(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        asset_idx: int,
        predict_fn: Callable,
    ) -> Tuple[List[float], List[float]]:
        """
        Expanding window validation (simulates production deployment).

        Train on [0, 50%], validate on [50%, 60%]
        Train on [0, 60%], validate on [60%, 70%]
        Train on [0, 70%], validate on [70%, 80%]
        Train on [0, 80%], validate on [80%, 90%]
        Train on [0, 90%], validate on [90%, 100%]

        If model is robust, IC should be stable across splits.
        If overfitting, IC degrades as validation moves further from training.
        """
        n_samples = len(data)
        min_train = int(n_samples * self.config.min_train_ratio)
        step_size = int(n_samples * (1 - self.config.min_train_ratio) / self.config.n_forward_splits)

        ics, mses = [], []

        for i in range(self.config.n_forward_splits):
            val_start = min_train + i * step_size
            val_end = min(val_start + step_size, n_samples)

            if val_end - val_start < 96:  # Need at least one sequence
                continue

            val_data = data[val_start:val_end]
            val_targets = targets[val_start:val_end]

            preds = predict_fn(val_data, asset_idx)
            ic, mse = self.compute_ic_and_mse(preds, val_targets)

            ics.append(ic)
            mses.append(mse)

        return ics, mses

    def shuffled_kfold(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        asset_idx: int,
        predict_fn: Callable,
        seed: int = 42,
    ) -> Tuple[List[float], List[float]]:
        """
        K-fold validation with shuffled splits (breaks temporal structure).

        If model is truly learning patterns (not memorizing temporal structure),
        shuffled IC should be similar to temporal IC.

        If shuffled IC << temporal IC, model is exploiting autocorrelation.
        """
        n_samples = len(data)
        fold_size = n_samples // self.config.n_shuffled_folds

        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        ics, mses = [], []

        for fold in range(self.config.n_shuffled_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < self.config.n_shuffled_folds - 1 else n_samples

            val_indices = indices[val_start:val_end]
            val_data = data[val_indices]
            val_targets = targets[val_indices]

            preds = predict_fn(val_data, asset_idx)
            ic, mse = self.compute_ic_and_mse(preds, val_targets)

            ics.append(ic)
            mses.append(mse)

        return ics, mses

    def regime_specific_validation(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        asset_idx: int,
        predict_fn: Callable,
        seed: int = 42,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Validate on held-out samples from each regime separately.

        Detects if model only works in specific market conditions.
        Robust model should have positive IC in all regimes.
        """
        # Classify regimes
        ret_std = np.std(targets) + 1e-6
        regimes = np.ones(len(targets), dtype=int)
        regimes[targets > ret_std * 0.5] = 2   # Bullish
        regimes[targets < -ret_std * 0.5] = 0  # Bearish

        regime_names = {0: "bearish", 1: "neutral", 2: "bullish"}
        regime_ics, regime_mses = {}, {}

        rng = np.random.default_rng(seed)

        for regime_id, regime_name in regime_names.items():
            mask = regimes == regime_id
            if mask.sum() < 96:  # Need minimum samples
                regime_ics[regime_name] = 0.0
                regime_mses[regime_name] = float('inf')
                continue

            # Randomly hold out 20% of this regime
            regime_indices = np.where(mask)[0]
            rng.shuffle(regime_indices)
            n_holdout = int(len(regime_indices) * self.config.regime_holdout_pct)
            holdout_indices = regime_indices[:n_holdout]

            val_data = data[holdout_indices]
            val_targets = targets[holdout_indices]

            preds = predict_fn(val_data, asset_idx)
            ic, mse = self.compute_ic_and_mse(preds, val_targets)

            regime_ics[regime_name] = ic
            regime_mses[regime_name] = mse

        return regime_ics, regime_mses

    def detect_hallucination(
        self,
        baseline_ic: float,
        forward_ics: List[float],
        shuffled_ics: List[float],
        regime_ics: Dict[str, float],
        ic_stability: float,
    ) -> Tuple[float, List[str]]:
        """
        Compute hallucination score (0-1, higher = more suspicious).

        Red flags:
          1. Forward walk IC degrades >50%
          2. Shuffled IC much lower than baseline IC
          3. IC only positive in one regime
          4. High variance across folds (instability)
          5. Baseline IC is absurdly high (>0.8 on crypto returns)
        """
        score = 0.0
        reasons = []

        # Check 1: Forward walk degradation
        if len(forward_ics) > 0:
            avg_forward = np.mean(forward_ics)
            degradation = (baseline_ic - avg_forward) / max(baseline_ic, 1e-6)
            if degradation > 0.5:
                score += 0.25
                reasons.append(f"Forward walk IC degraded {degradation*100:.1f}% ({baseline_ic:.3f} → {avg_forward:.3f})")

        # Check 2: Shuffled IC much lower
        if len(shuffled_ics) > 0:
            avg_shuffled = np.mean(shuffled_ics)
            shuffle_drop = (baseline_ic - avg_shuffled) / max(baseline_ic, 1e-6)
            if shuffle_drop > 0.4:
                score += 0.25
                reasons.append(f"Shuffled IC dropped {shuffle_drop*100:.1f}% (temporal leakage suspected)")

        # Check 3: Regime brittleness
        positive_regimes = sum(1 for ic in regime_ics.values() if ic > 0.02)
        if positive_regimes <= 1:
            score += 0.20
            reasons.append(f"Only {positive_regimes}/3 regimes have positive IC (brittle)")

        # Check 4: Instability
        if ic_stability < 0.6:
            score += 0.15
            reasons.append(f"IC stability is {ic_stability:.2f} (high variance across folds)")

        # Check 5: Absurdly high baseline IC
        if baseline_ic > 0.80:
            score += 0.15
            reasons.append(f"Baseline IC={baseline_ic:.3f} is suspiciously high for crypto")

        return min(score, 1.0), reasons

    def baseline_validation(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        asset_idx: int,
        predict_fn: Callable,
        split: float = 0.90,
    ) -> Tuple[float, float]:
        """Standard contiguous last-N% validation (for comparison)."""
        split_idx = int(len(data) * split)
        val_data = data[split_idx:]
        val_targets = targets[split_idx:]

        preds = predict_fn(val_data, asset_idx)
        return self.compute_ic_and_mse(preds, val_targets)

    def run_comprehensive_validation(
        self,
        data_segments: List[Tuple[np.ndarray, np.ndarray, int, str]],
        predict_fn: Callable,
        horizon: int = 1,
    ) -> Dict[str, RobustValidationResults]:
        """
        Run all validation strategies across all assets.

        Args:
            data_segments: List of (features, targets_dict, asset_idx, asset_name)
            predict_fn: Function(data, asset_idx) -> predictions
            horizon: Which return horizon to validate (1, 4, 16, 64)

        Returns:
            Dict mapping asset_name -> RobustValidationResults
        """
        all_results = {}

        for feats, targets_dict, asset_idx, asset_name in data_segments:
            if horizon not in targets_dict:
                continue

            targets = targets_dict[horizon]

            print(f"\n  Robust validation: {asset_name}")

            # Baseline (standard last-10% split)
            baseline_ic, baseline_mse = self.baseline_validation(
                feats, targets, asset_idx, predict_fn
            )
            print(f"    Baseline IC: {baseline_ic:+.4f}  MSE: {baseline_mse:.6f}")

            # Forward walk
            print(f"    Running forward walk ({self.config.n_forward_splits} splits)...")
            forward_ics, forward_mses = self.temporal_forward_walk(
                feats, targets, asset_idx, predict_fn
            )
            print(f"    Forward IC: {np.mean(forward_ics):+.4f} ± {np.std(forward_ics):.4f}")

            # Shuffled K-fold
            print(f"    Running shuffled K-fold ({self.config.n_shuffled_folds} folds)...")
            shuffled_ics, shuffled_mses = self.shuffled_kfold(
                feats, targets, asset_idx, predict_fn
            )
            print(f"    Shuffled IC: {np.mean(shuffled_ics):+.4f} ± {np.std(shuffled_ics):.4f}")

            # Regime-specific
            print(f"    Running regime-specific holdout...")
            regime_ics, regime_mses = self.regime_specific_validation(
                feats, targets, asset_idx, predict_fn
            )
            for regime, ic in regime_ics.items():
                print(f"      {regime:<10} IC: {ic:+.4f}")

            # Compute stability
            all_ics = forward_ics + shuffled_ics
            ic_mean = np.mean(all_ics)
            ic_std = np.std(all_ics)
            ic_stability = max(0.0, 1.0 - (ic_std / max(abs(ic_mean), 1e-6)))

            all_mses = forward_mses + shuffled_mses
            mse_mean = np.mean(all_mses)
            mse_std = np.std(all_mses)
            mse_stability = max(0.0, 1.0 - (mse_std / max(mse_mean, 1e-6)))

            # Detect hallucination
            hallucination_score, reasons = self.detect_hallucination(
                baseline_ic, forward_ics, shuffled_ics, regime_ics, ic_stability
            )

            # IC degradation
            avg_robust_ic = np.mean(forward_ics + shuffled_ics)
            ic_degradation = (baseline_ic - avg_robust_ic) / max(baseline_ic, 1e-6)

            results = RobustValidationResults(
                forward_walk_ics=forward_ics,
                forward_walk_mses=forward_mses,
                shuffled_ics=shuffled_ics,
                shuffled_mses=shuffled_mses,
                regime_ics=regime_ics,
                regime_mses=regime_mses,
                ic_stability=ic_stability,
                mse_stability=mse_stability,
                hallucination_score=hallucination_score,
                hallucination_reasons=reasons,
                baseline_ic=baseline_ic,
                baseline_mse=baseline_mse,
                ic_degradation=ic_degradation,
            )

            all_results[asset_name] = results

            # Print summary
            print(f"    IC Stability: {ic_stability:.3f}")
            print(f"    Hallucination Score: {hallucination_score:.3f}")
            if reasons:
                for reason in reasons:
                    print(f"      [!] {reason}")

        return all_results

    def print_aggregate_report(
        self,
        results: Dict[str, RobustValidationResults],
    ):
        """Print comprehensive report across all assets."""
        print(f"\n{'='*70}")
        print(f"  ROBUSTNESS VALIDATION SUMMARY")
        print(f"{'='*70}")

        n_assets = len(results)

        # Aggregate metrics
        avg_baseline_ic = np.mean([r.baseline_ic for r in results.values()])
        avg_forward_ic = np.mean([np.mean(r.forward_walk_ics) for r in results.values()])
        avg_shuffled_ic = np.mean([np.mean(r.shuffled_ics) for r in results.values()])
        avg_ic_stability = np.mean([r.ic_stability for r in results.values()])
        avg_hallucination = np.mean([r.hallucination_score for r in results.values()])
        avg_ic_degradation = np.mean([r.ic_degradation for r in results.values()])

        print(f"\n  Averaged across {n_assets} assets:")
        print(f"    Baseline IC (last 10%):     {avg_baseline_ic:+.4f}")
        print(f"    Forward Walk IC:            {avg_forward_ic:+.4f}  (degradation: {avg_ic_degradation*100:+.1f}%)")
        print(f"    Shuffled IC:                {avg_shuffled_ic:+.4f}")
        print(f"    IC Stability:               {avg_ic_stability:.3f}")
        print(f"    Hallucination Score:        {avg_hallucination:.3f}  {'[FAIL] SUSPICIOUS' if avg_hallucination > 0.3 else '[PASS] OK'}")

        # Check gates
        # NOTE: The Forward IC threshold (0.015) is the unadjusted per-model threshold.
        # When comparing across all 9 models (V1-V9) simultaneously, apply Bonferroni
        # correction: 0.015 / 9 = 0.00167. See anti_fragile.IC_THRESHOLD_BONFERRONI.
        # This gate uses the unadjusted threshold as a minimum quality bar for
        # individual model validation. Multi-model selection should use the corrected
        # threshold to control family-wise error rate.
        print(f"\n  ROBUSTNESS GATES:")
        gates = {
            "IC Stability": avg_ic_stability > 0.6,
            "IC Degradation": avg_ic_degradation < 0.50,
            "Hallucination": avg_hallucination < 0.3,
            "Forward IC": avg_forward_ic > 0.015,
        }

        for gate_name, passed in gates.items():
            status = "PASS" if passed else "FAIL"
            print(f"    [{status}] {gate_name}")

        all_pass = all(gates.values())

        print(f"\n  {'='*40}")
        if all_pass:
            print(f"  VERDICT: ROBUST - Model generalizes well")
        else:
            print(f"  VERDICT: OVERFITTING DETECTED")
            print(f"  Model may be memorizing temporal patterns")
        print(f"  {'='*40}")

        # Per-asset details
        print(f"\n  Per-Asset Robustness:")
        print(f"    {'Asset':<10} {'Baseline':>9} {'Forward':>9} {'Shuffled':>9} {'Stab':>6} {'Hall':>6}")
        print(f"    {'-'*70}")
        for name, r in results.items():
            avg_fwd = np.mean(r.forward_walk_ics)
            avg_shuf = np.mean(r.shuffled_ics)
            print(f"    {name:<10} {r.baseline_ic:>+9.4f} {avg_fwd:>+9.4f} {avg_shuf:>+9.4f} "
                  f"{r.ic_stability:>6.3f} {r.hallucination_score:>6.3f}")


def example_predict_function(model, seq_len=96):
    """
    Example prediction function wrapper for RobustValidator.

    Usage:
        predict_fn = lambda data, asset_idx: example_predict_function(
            model, seq_len=96
        )(data, asset_idx)

        validator = RobustValidator(model)
        results = validator.run_comprehensive_validation(
            data_segments, predict_fn, horizon=1
        )
    """
    @torch.no_grad()
    def predict(data: np.ndarray, asset_idx: int) -> np.ndarray:
        """Run model on data, return predictions for all timesteps."""
        model.eval()
        device = next(model.parameters()).device

        predictions = []
        # FIX: Use non-overlapping stride to prevent IC inflation
        indices = list(range(0, len(data) - seq_len, seq_len))
        if not indices:
            indices = [0]

        for i in indices:
            obs_np = data[i:i+seq_len]
            if len(obs_np) < seq_len:
                break

            obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                # Adapt to your model's forward signature
                outputs = model.forward_train(obs, asset)
                # Extract t+1 predictions (adapt horizon as needed)
                logits = outputs["return_logits"][1]  # horizon=1
                preds = model.bucketer.decode(logits).cpu().numpy().flatten()

            predictions.append(preds)

        return np.concatenate(predictions) if predictions else np.array([])

    return predict
