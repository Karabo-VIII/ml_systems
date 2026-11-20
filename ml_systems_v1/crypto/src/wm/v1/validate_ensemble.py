"""
V1.E Cross-Model Ensemble Validation

Evaluates the CrossModelEnsemble on real validation data.
Computes IC, RankIC, DirAcc per horizon per asset, plus global ShIC.
Compares ensemble to individual model baselines to quantify IC boost.

Usage:
    python validate_ensemble.py
    python validate_ensemble.py --models v1_0 v1_1_f13 v1_2
    python validate_ensemble.py --gating     # Enable XD-conditioned gating
"""
import torch
import numpy as np
import polars as pl
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy import stats as scipy_stats

_THIS_DIR = Path(__file__).resolve().parent           # src/wm/v1/
_V1_GROUP = _THIS_DIR                                 # src/wm/v1/ (ensemble lives at group root)
_SRC_DIR = _V1_GROUP.parent                           # src/
_PROJECT_ROOT = _SRC_DIR.parent                       # project root
_V1_0_DIR = _V1_GROUP / "v1_0_training"               # V1.0 base (settings, components)

for _p in [str(_V1_0_DIR), str(_V1_GROUP), str(_SRC_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from settings import (
    DEVICE, DATA_DIR, LOG_DIR, REWARD_HORIZONS, ASSET_LIST, ASSET_TO_IDX,
    WM_SEQ_LEN, GATE_IC_MIN, GATE_SHUFFLED_IC_RATIO_MIN,
)
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets
from cross_ensemble import CrossModelEnsemble, _V1_FAMILY, ENSEMBLE_FEATURE_LIST

# Ensemble output directory (not per-variant, shared across ensemble)
ENSEMBLE_LOG_DIR = _PROJECT_ROOT / "logs" / "v1" / "ensemble"
ENSEMBLE_LOG_DIR.mkdir(parents=True, exist_ok=True)


# ENSEMBLE_FEATURE_LIST imported from cross_ensemble.py (single source of truth)

# Default ensemble: active V1 models only (V1.2-V1.5, V1.7 archived)
DEFAULT_MODEL_KEYS = ["v1_0", "v1_1_f13", "v1_1", "v1_6"]

PURGE_GAP_BARS = 400  # Must match anti_fragile.py


# ===========================================================================
# Ensemble Validator
# ===========================================================================
class EnsembleValidator:
    def __init__(self, model_keys=None, use_gating=False):
        self.model_keys = model_keys or DEFAULT_MODEL_KEYS
        self.use_gating = use_gating

        # Determine feature list based on max n_features across requested models
        # Models with f22+ need indices 19-22 (V51 new base); V1.5 also needs 18 (xd_ma_distance)
        max_nf = max(_V1_FAMILY[k]["n_features"] for k in self.model_keys)
        if max_nf > 22:
            self.feature_list = ENSEMBLE_FEATURE_LIST  # Full 27 features (incl baseline preds)
        elif max_nf > 18:
            self.feature_list = ENSEMBLE_FEATURE_LIST[:23]  # 23 features (no baseline preds)
        else:
            self.feature_list = ENSEMBLE_FEATURE_LIST[:18]  # 18 features
        self.n_features = len(self.feature_list)

        print(f"\n{'='*70}")
        print(f"  V1.E CROSS-MODEL ENSEMBLE VALIDATION")
        print(f"  Models:   {', '.join(self.model_keys)}")
        print(f"  Features: {self.n_features}")
        print(f"  Gating:   {'XD-conditioned' if use_gating else 'uniform'}")
        print(f"  Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Load ensemble
        self.ensemble = CrossModelEnsemble(
            model_keys=self.model_keys,
            use_gating=use_gating,
            device=DEVICE,
        )

        # Print model info
        print(f"\n  Ensemble composition:")
        total_params = 0
        for info in self.ensemble.get_model_info():
            print(f"    {info['key']:<12} f{info['n_features']}  {info['params']:>10,} params")
            total_params += info["params"]
        print(f"    {'TOTAL':<12}      {total_params:>10,} params (all frozen)")

    def load_validation_data(self):
        """Load last-10% validation split from chimera parquets."""
        files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
        segments = []

        for f in files:
            asset_name = f.stem.split("_")[0].upper()
            if asset_name not in ASSET_TO_IDX:
                continue

            asset_idx = ASSET_TO_IDX[asset_name]
            df = pl.read_parquet(f)
            df = selective_drop_nulls(df, self.feature_list, REWARD_HORIZONS, asset_name)

            # FIX 2026-05-29: was 0.90 -> evaluated the ensemble on the UNSEEN
            # held-out segment (consuming the held-out budget). Val = [50%+purge, 70%].
            val_start = int(len(df) * 0.50) + PURGE_GAP_BARS
            val_end = int(len(df) * 0.70)
            if val_start >= val_end:
                val_start = int(len(df) * 0.50)
            df_val = df.slice(val_start, val_end - val_start)

            feats, targets = extract_features_targets(
                df_val, self.feature_list, REWARD_HORIZONS, asset_name
            )

            segments.append((feats, targets, asset_idx, asset_name))
            print(f"    {asset_name}: {len(feats):,} validation bars")

        return segments

    @torch.no_grad()
    def _evaluate_asset(self, feats, targets, asset_idx):
        """Evaluate ensemble on one asset. Returns per-horizon IC, RankIC, DirAcc."""
        seq_len = WM_SEQ_LEN
        returns_data = {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS}

        # Non-overlapping windows
        indices = list(range(0, len(feats) - seq_len, seq_len))
        if not indices and len(feats) >= seq_len:
            indices = [0]

        for i in tqdm(indices, desc="  Evaluating", leave=False):
            obs_np = feats[i:i+seq_len]
            obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                outputs = self.ensemble.forward_train(obs, asset)

            # Decode ensemble-averaged logits
            for h in REWARD_HORIZONS:
                logits_h = outputs["return_logits"][h]
                pred_h = self.ensemble.bucketer.decode(logits_h).cpu().numpy().flatten()
                real_h = targets[h][i:i+seq_len]
                returns_data[h]["preds"].append(pred_h)
                returns_data[h]["reals"].append(real_h)

        # Aggregate per-horizon metrics
        result = {"returns": {}}
        for h in REWARD_HORIZONS:
            preds = np.concatenate(returns_data[h]["preds"])
            reals = np.concatenate(returns_data[h]["reals"])
            mask = np.isfinite(preds) & np.isfinite(reals)
            p, r = preds[mask], reals[mask]

            if len(p) > 50:
                ic = float(np.corrcoef(p, r)[0, 1])
                rank_ic = float(scipy_stats.spearmanr(p, r).statistic)
                dir_acc = float(np.mean(np.sign(p) == np.sign(r)))

                # Bootstrap CI
                ic_lo, ic_hi = self._bootstrap_ic(p, r)

                # IC p-value
                n = len(p)
                t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
                p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2)))
            else:
                ic = rank_ic = 0.0
                dir_acc = 0.5
                ic_lo = ic_hi = 0.0
                p_value = 1.0

            result["returns"][h] = {
                "ic": ic, "rank_ic": rank_ic, "dir_acc": dir_acc,
                "ic_95_lo": ic_lo, "ic_95_hi": ic_hi, "p_value": p_value,
                "n_samples": len(p),
            }

        return result

    @torch.no_grad()
    def _compute_global_shuffled_ic(self, segments, n_seeds=5, batch_size=64):
        """Compute IC on globally-shuffled data (anti-memorization gate)."""
        all_shuffled_ics = []

        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset * 1000
            rng = np.random.default_rng(seed)
            all_preds, all_reals = [], []

            for feats, targets, asset_idx, asset_name in segments:
                n = len(feats)
                seq_len = WM_SEQ_LEN
                if n < seq_len * 2:
                    continue

                indices = np.arange(n)
                rng.shuffle(indices)

                shuffled_feats = feats[indices]
                shuffled_targets_1 = targets[1][indices]

                window_starts = list(range(0, n - seq_len, seq_len))

                for batch_start in range(0, len(window_starts), batch_size):
                    batch_ws = window_starts[batch_start:batch_start + batch_size]
                    obs_list = []
                    real_list = []

                    for ws in batch_ws:
                        obs_list.append(shuffled_feats[ws:ws+seq_len])
                        real_list.append(shuffled_targets_1[ws:ws+seq_len])

                    obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
                    asset = torch.full(
                        (len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE
                    )

                    with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                        outputs = self.ensemble.forward_train(obs, asset)

                    logits = outputs["return_logits"][1]
                    preds = self.ensemble.bucketer.decode(logits).cpu().numpy()

                    for b, real in enumerate(real_list):
                        all_preds.extend(preds[b].flatten())
                        all_reals.extend(real)

            preds_arr = np.array(all_preds)
            reals_arr = np.array(all_reals)
            mask = np.isfinite(preds_arr) & np.isfinite(reals_arr)
            if mask.sum() > 50:
                ic = float(np.corrcoef(preds_arr[mask], reals_arr[mask])[0, 1])
                if np.isfinite(ic):
                    all_shuffled_ics.append(ic)

        return float(np.mean(all_shuffled_ics)) if all_shuffled_ics else 0.0

    def _bootstrap_ic(self, preds, reals, n_bootstrap=1000):
        """Bootstrap 95% CI for IC."""
        n = len(preds)
        if n < 30:
            return 0.0, 0.0
        ics = []
        rng = np.random.default_rng(42)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            p, r = preds[idx], reals[idx]
            if np.std(p) > 1e-10 and np.std(r) > 1e-10:
                ics.append(float(np.corrcoef(p, r)[0, 1]))
        if len(ics) < 10:
            return 0.0, 0.0
        return float(np.percentile(ics, 2.5)), float(np.percentile(ics, 97.5))

    def run_validation(self):
        """Full ensemble validation pipeline."""
        print(f"\n  Loading validation data from {DATA_DIR}")
        segments = self.load_validation_data()
        if not segments:
            print("  [ERROR] No validation data found.")
            return False

        # -- Per-asset evaluation --
        all_results = {}
        for feats, targets, asset_idx, asset_name in segments:
            print(f"\n{'-'*70}")
            print(f"  {asset_name}")
            print(f"{'-'*70}")
            result = self._evaluate_asset(feats, targets, asset_idx)
            all_results[asset_name] = result

            # Print per-asset results
            for h in REWARD_HORIZONS:
                ret = result["returns"][h]
                sig = "***" if ret["p_value"] < 0.001 else "**" if ret["p_value"] < 0.01 else "*" if ret["p_value"] < 0.05 else ""
                print(f"    t+{h:<3} IC:{ret['ic']:+.4f}{sig:<4} RankIC:{ret['rank_ic']:+.4f}  "
                      f"Dir:{ret['dir_acc']*100:.1f}%  "
                      f"CI:[{ret['ic_95_lo']:+.4f},{ret['ic_95_hi']:+.4f}]  "
                      f"n={ret['n_samples']}")

        # -- Aggregate --
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS ({len(all_results)} assets, {self.ensemble.n_models} models)")
        print(f"{'='*70}")

        avg_ics = {}
        for h in REWARD_HORIZONS:
            ics = [r["returns"][h]["ic"] for r in all_results.values()]
            rank_ics = [r["returns"][h]["rank_ic"] for r in all_results.values()]
            dirs = [r["returns"][h]["dir_acc"] for r in all_results.values()]
            avg_ic = float(np.mean(ics))
            avg_ics[h] = avg_ic
            print(f"    t+{h:<3} IC:{avg_ic:+.4f}  RankIC:{np.mean(rank_ics):+.4f}  "
                  f"Dir:{np.mean(dirs)*100:.1f}%  [range: {min(ics):+.4f} to {max(ics):+.4f}]")

        mean_ic = float(np.mean(list(avg_ics.values())))
        print(f"\n    Mean IC across horizons: {mean_ic:+.4f}  "
              f"{'PASS' if mean_ic > GATE_IC_MIN else 'FAIL'}")

        # Per-asset summary table
        print(f"\n  Per-Asset Summary:")
        print(f"    {'Asset':<10} {'IC(1)':>8} {'IC(4)':>8} {'IC(16)':>8} {'IC(64)':>8} {'Dir(1)':>7}")
        print(f"    {'-'*55}")
        for name, r in all_results.items():
            print(f"    {name:<10} "
                  f"{r['returns'][1]['ic']:>+8.4f} "
                  f"{r['returns'][4]['ic']:>+8.4f} "
                  f"{r['returns'][16]['ic']:>+8.4f} "
                  f"{r['returns'][64]['ic']:>+8.4f} "
                  f"{r['returns'][1]['dir_acc']*100:>6.1f}%")

        # -- Shuffled IC --
        contiguous_ic = mean_ic
        print(f"\n  Computing GLOBAL shuffled IC (anti-memorization gate)...")
        shuffled_ic = self._compute_global_shuffled_ic(segments)
        print(f"    Contiguous IC:       {contiguous_ic:+.4f}")
        print(f"    Global Shuffled IC:  {shuffled_ic:+.4f}")
        if abs(contiguous_ic) > 1e-6:
            ratio = shuffled_ic / contiguous_ic
            print(f"    Ratio:               {ratio:.3f} (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")
        else:
            ratio = 0.0
            print(f"    Ratio:               N/A (contiguous IC near zero)")

        # -- Gates --
        print(f"\n{'='*70}")
        print(f"  VALIDATION GATES")
        print(f"{'='*70}")

        gates = {
            "Mean IC": (mean_ic > GATE_IC_MIN, f"{mean_ic:+.4f} > {GATE_IC_MIN}"),
        }
        if abs(contiguous_ic) > 1e-6:
            gates["Shuffled IC Ratio"] = (
                ratio > GATE_SHUFFLED_IC_RATIO_MIN,
                f"{ratio:.3f} > {GATE_SHUFFLED_IC_RATIO_MIN}"
            )

        all_pass = True
        for gate_name, (passed, desc) in gates.items():
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    [{status}] {gate_name:<25} {desc}")

        print(f"\n  {'='*40}")
        if all_pass:
            print(f"  VERDICT: ALL GATES PASSED")
            print(f"  Ensemble is ready for agent training.")
        else:
            print(f"  VERDICT: GATE(S) FAILED")
            print(f"  Ensemble needs adjustment.")
        print(f"  {'='*40}")

        # -- Individual model baselines (from validation JSONs) --
        self._print_baseline_comparison(mean_ic, shuffled_ic)

        # -- Save results --
        self._save_results(all_results, all_pass, mean_ic, shuffled_ic, ratio)

        return all_pass

    def _print_baseline_comparison(self, ensemble_ic, ensemble_shic):
        """Load individual model validation JSONs and compare."""
        print(f"\n{'='*70}")
        print(f"  ENSEMBLE vs INDIVIDUAL MODEL COMPARISON")
        print(f"{'='*70}")

        log_root = _PROJECT_ROOT / "logs" / "v1"
        baseline_ics = {}

        # Map model keys to their log directories and validation JSON patterns
        key_to_log = {
            "v1_0": "v1_0",
            "v1_1_f13": "v1_1",
            "v1_1": "v1_1",
            "v1_6": "v1_6",
        }

        for key in self.model_keys:
            log_dir_name = key_to_log.get(key)
            if not log_dir_name:
                continue

            log_dir = log_root / log_dir_name
            # Find validation JSONs
            jsons = sorted(log_dir.glob("validation_*.json")) if log_dir.exists() else []
            # Also check archive subdirectories
            if log_dir.exists():
                for subdir in log_dir.iterdir():
                    if subdir.is_dir():
                        jsons.extend(sorted(subdir.glob("validation_*.json")))

            if not jsons:
                print(f"    {key:<12} -- no validation JSON found")
                continue

            # Use most recent validation JSON
            latest = jsons[-1]
            try:
                with open(latest) as f:
                    data = json.load(f)
                results = data.get("results", {})
                # Compute mean IC across horizons and assets
                all_ics = []
                for asset_data in results.values():
                    for h_str, h_data in asset_data.get("returns", {}).items():
                        all_ics.append(h_data.get("ic", 0.0))
                if all_ics:
                    baseline_ics[key] = float(np.mean(all_ics))
            except Exception as e:
                print(f"    {key:<12} -- error reading JSON: {e}")

        if not baseline_ics:
            print(f"    No baseline validation JSONs found for comparison.")
            return

        best_individual = max(baseline_ics.values())
        mean_individual = float(np.mean(list(baseline_ics.values())))

        print(f"\n    {'Model':<12} {'Mean IC':>10}")
        print(f"    {'-'*25}")
        for key, ic in baseline_ics.items():
            print(f"    {key:<12} {ic:>+10.4f}")
        print(f"    {'-'*25}")
        print(f"    {'Best indiv.':<12} {best_individual:>+10.4f}")
        print(f"    {'Mean indiv.':<12} {mean_individual:>+10.4f}")
        print(f"    {'ENSEMBLE':<12} {ensemble_ic:>+10.4f}")
        print(f"    {'-'*25}")

        if best_individual != 0:
            boost_vs_best = (ensemble_ic - best_individual) / abs(best_individual) * 100
            boost_vs_mean = (ensemble_ic - mean_individual) / abs(mean_individual) * 100
            print(f"    IC boost vs best:  {boost_vs_best:+.1f}%")
            print(f"    IC boost vs mean:  {boost_vs_mean:+.1f}%")

        if ensemble_shic > 0:
            print(f"\n    Ensemble ShIC: {ensemble_shic:+.4f}")
            print(f"    (Individual ShIC ceiling: ~0.022 from prior analysis)")
            if ensemble_shic > 0.022:
                print(f"    [OK] Ensemble ShIC exceeds individual ceiling -- genuine signal boost")
            else:
                print(f"    [INFO] Ensemble ShIC within individual range -- diversity benefit is robustness")

    def _save_results(self, all_results, gate_pass, mean_ic, shuffled_ic, shic_ratio):
        """Save validation results to JSON."""
        output = {
            "version": "v1.E",
            "model": f"ensemble_{'_'.join(self.model_keys)}",
            "n_models": self.ensemble.n_models,
            "model_keys": self.model_keys,
            "gating": "xd" if self.use_gating else "uniform",
            "timestamp": datetime.now().isoformat(),
            "gate_passed": gate_pass,
            "mean_ic": mean_ic,
            "shuffled_ic": shuffled_ic,
            "shic_ratio": shic_ratio,
            "results": {},
        }

        for name, r in all_results.items():
            output["results"][name] = {
                "returns": {str(h): {
                    "ic": r["returns"][h]["ic"],
                    "rank_ic": r["returns"][h]["rank_ic"],
                    "dir_acc": r["returns"][h]["dir_acc"],
                } for h in REWARD_HORIZONS},
            }

        out_dir = ENSEMBLE_LOG_DIR
        out_path = out_dir / f"validation_ensemble_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path}")


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="V1.E Ensemble Validation")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Model keys to include (default: {DEFAULT_MODEL_KEYS})")
    parser.add_argument("--gating", action="store_true",
                        help="Enable XD-conditioned gating (default: uniform averaging)")
    args = parser.parse_args()

    model_keys = args.models or DEFAULT_MODEL_KEYS

    # Validate model keys
    for key in model_keys:
        if key not in _V1_FAMILY:
            print(f"[ERROR] Unknown model key: {key}")
            print(f"  Available: {list(_V1_FAMILY.keys())}")
            sys.exit(1)

    validator = EnsembleValidator(
        model_keys=model_keys,
        use_gating=args.gating,
    )
    gate_pass = validator.run_validation()
    sys.exit(0 if gate_pass else 1)


if __name__ == "__main__":
    main()
