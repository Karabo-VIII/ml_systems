"""
V6.2 World Model Validation -- Comprehensive Diagnostics Suite (Causal JEPA + Adversarial + Anti-Memorization)

Run this BEFORE proceeding to agent training.
Evaluates: representation health, contrastive quality, return prediction (all horizons),
auxiliary reconstruction (base features only), regime classification, collapse detection, per-asset breakdown.

V6.2-Specific Tests:
  - Embedding collapse detection (VICReg variance/covariance)
  - Contrastive alignment quality
  - Effective rank of representations
  - Dead dimension analysis
  - NO dream coherence (V6.2 has no dream_step)
  - NO imagination masking (JEPA uses masking differently -- evaluated via reconstruction)
  - get_loss returns 4 values: (total_loss, loss_dict, l_disc, outputs) -- V6 4-return pattern
  - Reconstruction is over base features only [0:base_dim], NOT full feature vector

Usage:
    python validate_world.py                    # Best model (18 features)
    python validate_world.py --features 13      # Validate 13-feature checkpoint
    python validate_world.py --latest           # Latest checkpoint
    python validate_world.py --both             # Run both and compare
    python validate_world.py --model path.pt    # Specific checkpoint
    python validate_world.py --robust           # Run robust validation (detects overfitting)
"""
import torch
import torch.nn.functional as F
import numpy as np
import polars as pl
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import CausalJEPAWorldModel, count_parameters
from revin import RevIN
from log_utils import setup_logging, teardown_logging
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets

try:
    from validation_utils import RobustValidator, ValidationConfig
    ROBUST_VALIDATION_AVAILABLE = True
except ImportError:
    ROBUST_VALIDATION_AVAILABLE = False


# ===============================================================================
# VALIDATOR
# ===============================================================================

class WorldModelValidator:
    """Comprehensive diagnostics for V6.2 Causal JEPA + Adversarial + Anti-Memorization World Model."""

    def __init__(self, model_path: Path = None, n_features: int = 22):
        self.n_features = n_features
        feature_list, input_dim, base_dim = get_feature_config(n_features)
        self.feature_list = feature_list
        self.input_dim = input_dim
        self.base_dim = base_dim
        feat_tag = f"f{n_features}"
        self.ckpt_prefix = f"v6_2_{feat_tag}"
        self.model = CausalJEPAWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
        self.model_name = "unknown"
        self.revin = None  # Created only if checkpoint has revin_state_dict
        self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: Path):
        if model_path is None:
            model_path = BASE_MODEL_DIR / f"{self.ckpt_prefix}_wm_best_ema.pt"
        if not model_path.exists():
            model_path = BASE_MODEL_DIR / f"{self.ckpt_prefix}_wm_best.pt"
        if not model_path.exists():
            model_path = BASE_MODEL_DIR / f"{self.ckpt_prefix}_wm_latest.pt"
        if not model_path.exists():
            print(f"  [ERROR] No checkpoint found in {BASE_MODEL_DIR} (prefix={self.ckpt_prefix})")
            sys.exit(1)

        self.model_name = model_path.stem
        print(f"  Loading: {model_path.name}")
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            print(f"  Checkpoint from epoch {epoch}")
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and any(
            k.startswith("obs_proj") or k.startswith("context_encoder") for k in ckpt
        ):
            self.model.load_state_dict(ckpt)
        else:
            self.model.load_state_dict(ckpt)
        if isinstance(ckpt, dict) and "revin_state_dict" in ckpt:
            self.revin.load_state_dict(ckpt["revin_state_dict"])
        print(f"  Parameters: {count_parameters(self.model):,}")

    # --- Data Loading ---------------------------------------------------------

    # Purge gap: Hurst R/S (200) + rolling z-score (200) = 400-bar cascading dependency
    PURGE_GAP_BARS = 400

    def load_validation_data(self, use_full_data=False):
        """
        Load validation data.

        Args:
            use_full_data: If True, load all data (for robust validation).
                          If False, load last 10% only (standard validation).
        """
        files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
        segments = []

        for f in files:
            asset_name = f.stem.split("_")[0].upper()
            if asset_name not in ASSET_TO_IDX:
                continue

            asset_idx = ASSET_TO_IDX[asset_name]
            df = pl.read_parquet(f)
            df = selective_drop_nulls(df, self.feature_list, REWARD_HORIZONS, asset_name)

            if not use_full_data:
                # FIX 2026-05-29: was 0.90 -> evaluated on the UNSEEN held-out
                # segment. Val = [50%+purge, 70%] (mirrors v1.0, the correct one).
                val_start = int(len(df) * 0.50) + self.PURGE_GAP_BARS
                val_end = int(len(df) * 0.70)
                if val_start >= val_end:
                    val_start = int(len(df) * 0.50)
                df_val = df.slice(val_start, val_end - val_start)
            else:
                df_val = df  # Use full dataset for robust validation

            feats, targets = extract_features_targets(
                df_val, self.feature_list, REWARD_HORIZONS, asset_name
            )

            segments.append((feats, targets, asset_idx, asset_name))
            print(f"    {asset_name}: {len(feats):,} "
                  f"{'total' if use_full_data else 'validation'} bars")

        return segments

    # --- Train/Val Loss Computation (for overfitting gate) -------------------

    @torch.no_grad()
    def _compute_split_loss(self, segments):
        """Compute average total loss on a set of data segments for GATE_LOSS_RATIO_MAX.
        V6 JEPA get_loss returns 3 values: (total_loss, loss_dict, l_disc)."""
        seq_len = WM_SEQ_LEN
        all_losses = []
        for feats, targets, asset_idx, asset_name in segments:
            indices = list(range(0, len(feats) - seq_len, seq_len))
            if not indices:
                continue
            for i in indices:
                obs = torch.from_numpy(feats[i:i+seq_len]).unsqueeze(0).float().to(DEVICE)
                asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
                targ = {h: torch.from_numpy(targets[h][i:i+seq_len]).unsqueeze(0).float().to(DEVICE)
                        for h in REWARD_HORIZONS}
                if self.revin is not None:

                    obs = self.revin(obs, mode='norm')
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    # V6 returns 4 values: (total_loss, loss_dict, l_disc, outputs)
                    _, loss_dict, _, _ = self.model.get_loss(obs, asset, targ, mask_ratio=0.0)
                if "total" in loss_dict and np.isfinite(loss_dict["total"]):
                    all_losses.append(loss_dict["total"])
        return float(np.mean(all_losses)) if all_losses else 0.0

    def _load_train_split(self):
        """Load training split (first 90%) for loss ratio computation."""
        files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
        segments = []
        for f in files:
            asset_name = f.stem.split("_")[0].upper()
            if asset_name not in ASSET_TO_IDX:
                continue
            asset_idx = ASSET_TO_IDX[asset_name]
            df = pl.read_parquet(f)
            df = selective_drop_nulls(df, self.feature_list, REWARD_HORIZONS, asset_name)
            split = int(len(df) * 0.50)  # was 0.90: leaked OOS+unseen into train baseline
            df_train = df.slice(0, split)
            feats, targets = extract_features_targets(
                df_train, self.feature_list, REWARD_HORIZONS, asset_name
            )
            segments.append((feats, targets, asset_idx, asset_name))
        return segments

    # --- Main Diagnostics -----------------------------------------------------

    @torch.no_grad()
    def run_diagnostics(self):
        print(f"\n{'='*70}")
        print(f"  V6.2 CAUSAL JEPA + ADVERSARIAL + ANTI-MEMORIZATION COMPREHENSIVE VALIDATION")
        print(f"  Model: {self.model_name}")
        print(f"  Features: {self.n_features} (base_dim={self.base_dim})")
        print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        print(f"\n  Loading validation data from {DATA_DIR}")

        segments = self.load_validation_data()
        if not segments:
            print("  [ERROR] No validation data found.")
            return False

        all_results = {}

        for feats, targets, asset_idx, asset_name in segments:
            print(f"\n{'-'*70}")
            print(f"  {asset_name} DIAGNOSTICS")
            print(f"{'-'*70}")
            results = self._evaluate_asset(feats, targets, asset_idx)
            all_results[asset_name] = results
            self._print_asset_report(results, asset_name)

        self._print_aggregate_report(all_results)

        # -- Shuffled IC (anti-memorization gate) --------------------------
        print(f"\n  Computing shuffled IC (anti-memorization check)...")
        contiguous_ic = np.mean([
            np.mean([r["returns"][h]["ic"] for r in all_results.values()])
            for h in REWARD_HORIZONS
        ])
        shuffled_ic = self._compute_shuffled_ic(segments)
        print(f"    Contiguous IC: {contiguous_ic:+.4f}")
        print(f"    Shuffled IC:   {shuffled_ic:+.4f}")
        if abs(contiguous_ic) > 1e-6:
            ratio = shuffled_ic / contiguous_ic
            print(f"    Ratio:         {ratio:.3f} (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")
        else:
            ratio = 0.0
            print(f"    Ratio:         N/A (contiguous IC near zero)")

        # -- Train/Val Loss Ratio (overfitting gate) -----------------------
        print(f"\n  Computing train/val loss ratio (overfitting check)...")
        train_segments_data = self._load_train_split()
        train_loss = self._compute_split_loss(train_segments_data) if train_segments_data else 0.0
        val_loss = self._compute_split_loss(segments)
        loss_ratio = val_loss / train_loss if train_loss > 1e-8 else 0.0  # val/train: >2.0 = overfitting
        print(f"    Train Loss:    {train_loss:.4f}")
        print(f"    Val Loss:      {val_loss:.4f}")
        print(f"    Ratio:         {loss_ratio:.3f} (gate: < {GATE_LOSS_RATIO_MAX})")

        ic_h1 = np.mean([r["returns"][1]["ic"] for r in all_results.values()])
        gate_pass = self._check_gates(all_results, shuffled_ic=shuffled_ic, loss_ratio=loss_ratio, ic_h1_for_shic=ic_h1)
        self._save_results(all_results, gate_pass)

        return gate_pass

    # --- Per-Asset Evaluation -------------------------------------------------

    def _evaluate_asset(self, feats, targets, asset_idx):
        seq_len = WM_SEQ_LEN
        # Reconstruction only covers base features [0:base_dim]
        base_feature_list = self.feature_list[:self.base_dim]
        results = {
            "reconstruction": {
                "total_mse": [],
                "per_feature_mse": [[] for _ in base_feature_list],
                "per_feature_r2": [[] for _ in base_feature_list],
            },
            "returns": {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS},
            "representation": {
                "ctx_std": [], "tgt_std": [],
                "ctx_dim_stds": [], "tgt_dim_stds": [],
                "contrastive_acc": [],
                "ctx_cov_off_diag": [], "tgt_cov_off_diag": [],
            },
            "loss_components": {
                "contrastive": [], "vicreg": [], "recon": [],
                "adv": [], "disc": [], "regime": [],
            },
            "regime": {"preds": [], "labels": []},
        }

        # FIX: Use non-overlapping stride (seq_len) to prevent IC inflation
        # from overlapping windows. Previously seq_len // 2 caused data reuse.
        indices = list(range(0, len(feats) - seq_len, seq_len))
        if len(indices) == 0:
            indices = [0] if len(feats) >= seq_len else []

        for i in tqdm(indices, desc="  Evaluating", leave=False):
            obs_np = feats[i:i+seq_len]
            obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

            targ_tensors = {}
            for h in REWARD_HORIZONS:
                t = targets[h][i:i+seq_len]
                targ_tensors[h] = torch.from_numpy(t).unsqueeze(0).float().to(DEVICE)

            # -- RevIN normalization ----------------------------------------
            if self.revin is not None:

                obs = self.revin(obs, mode='norm')

            # -- Clean forward pass ----------------------------------------
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                outputs = self.model.forward_train(obs, asset)

            ctx_latent = outputs["ctx_latent"]    # [1, T, d_latent]
            tgt_latent = outputs["tgt_latent"]    # [1, T, d_latent]
            pred_latent = outputs["pred_latent"]  # [1, T, d_latent]
            recon = outputs["recon"]              # [1, T, input_dim]

            # -- Loss component tracking (V6 returns 4 values) -----------
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                _, loss_dict, _, _ = self.model.get_loss(
                    obs, asset, targ_tensors, mask_ratio=0.0
                )
            for lk in results["loss_components"]:
                if lk in loss_dict:
                    results["loss_components"][lk].append(loss_dict[lk])

            # -- 1. Auxiliary Reconstruction (base features only) ----------
            # recon has shape [1, T, base_dim]; compare to obs[:, :, :base_dim]
            obs_base = obs[:, :, :self.base_dim]
            rec_mse = F.mse_loss(recon, obs_base).item()
            results["reconstruction"]["total_mse"].append(rec_mse)

            for fi in range(self.base_dim):
                feat_recon = recon[..., fi].reshape(-1).cpu().numpy()
                feat_real = obs_base[..., fi].reshape(-1).cpu().numpy()
                mse_f = float(np.mean((feat_recon - feat_real) ** 2))
                var_f = float(np.var(feat_real)) + 1e-10
                r2_f = 1.0 - mse_f / var_f
                results["reconstruction"]["per_feature_mse"][fi].append(mse_f)
                results["reconstruction"]["per_feature_r2"][fi].append(r2_f)

            # -- 2. Return Prediction --------------------------------------
            for h in REWARD_HORIZONS:
                logits_h = outputs["return_logits"][h]
                pred_h = self.model.bucketer.decode(logits_h).cpu().numpy().flatten()
                real_h = targ_tensors[h].cpu().numpy().flatten()
                results["returns"][h]["preds"].append(pred_h)
                results["returns"][h]["reals"].append(real_h)

            # -- 3. Representation Health ----------------------------------
            ctx_flat = ctx_latent.reshape(-1, WM_D_LATENT).cpu().numpy()
            tgt_flat = tgt_latent.reshape(-1, WM_D_LATENT).cpu().numpy()

            # Per-dimension std
            ctx_dim_std = np.std(ctx_flat, axis=0)
            tgt_dim_std = np.std(tgt_flat, axis=0)
            results["representation"]["ctx_dim_stds"].append(ctx_dim_std)
            results["representation"]["tgt_dim_stds"].append(tgt_dim_std)

            # Overall std
            results["representation"]["ctx_std"].append(float(np.mean(ctx_dim_std)))
            results["representation"]["tgt_std"].append(float(np.mean(tgt_dim_std)))

            # Contrastive accuracy (per-timestep cosine similarity)
            pred_norm = F.normalize(pred_latent, dim=-1)    # [1, T, D]
            tgt_norm = F.normalize(tgt_latent, dim=-1)       # [1, T, D]
            sim = (pred_norm * tgt_norm).sum(-1).mean().item()
            results["representation"]["contrastive_acc"].append(sim)

            # Off-diagonal covariance (collapse indicator)
            ctx_c = ctx_flat - ctx_flat.mean(0)
            cov = (ctx_c.T @ ctx_c) / max(len(ctx_c) - 1, 1)
            np.fill_diagonal(cov, 0)
            off_diag = float(np.mean(np.abs(cov)))
            results["representation"]["ctx_cov_off_diag"].append(off_diag)

            tgt_c = tgt_flat - tgt_flat.mean(0)
            cov_t = (tgt_c.T @ tgt_c) / max(len(tgt_c) - 1, 1)
            np.fill_diagonal(cov_t, 0)
            results["representation"]["tgt_cov_off_diag"].append(
                float(np.mean(np.abs(cov_t)))
            )

            # -- 4. Regime Classification ----------------------------------
            regime_logits = outputs["regime_logits"].reshape(-1, 3)
            regime_preds = regime_logits.argmax(dim=-1).cpu().numpy()

            ret_1 = targ_tensors[1].reshape(-1).cpu().numpy()
            ret_std = np.std(ret_1) + 1e-6
            regime_labels = np.ones_like(ret_1, dtype=np.int64)
            regime_labels[ret_1 > ret_std * 0.5] = 2
            regime_labels[ret_1 < -ret_std * 0.5] = 0
            results["regime"]["preds"].append(regime_preds)
            results["regime"]["labels"].append(regime_labels)

        return self._aggregate_results(results)

    def _aggregate_results(self, raw):
        agg = {}

        # Reconstruction
        agg["rec_mse"] = float(np.mean(raw["reconstruction"]["total_mse"]))
        agg["per_feature_mse"] = [
            float(np.mean(v)) if v else 0.0
            for v in raw["reconstruction"]["per_feature_mse"]
        ]
        agg["per_feature_r2"] = [
            float(np.mean(v)) if v else 0.0
            for v in raw["reconstruction"]["per_feature_r2"]
        ]

        # Loss components
        agg["loss_components"] = {}
        for lk, vals in raw["loss_components"].items():
            agg["loss_components"][lk] = float(np.mean(vals)) if vals else 0.0

        # Return predictions per horizon
        agg["returns"] = {}
        for h in REWARD_HORIZONS:
            preds = np.concatenate(raw["returns"][h]["preds"])
            reals = np.concatenate(raw["returns"][h]["reals"])
            mask = np.isfinite(preds) & np.isfinite(reals)
            p, r = preds[mask], reals[mask]

            if len(p) > 50:
                ic = float(np.corrcoef(p, r)[0, 1])
                rank_ic = float(scipy_stats.spearmanr(p, r).statistic)
                dir_acc = float(np.mean(np.sign(p) == np.sign(r)))
                mae = float(np.mean(np.abs(p - r)))
                ic_lo, ic_hi = self._bootstrap_ic(p, r)

                sorted_idx = np.argsort(p)
                decile = max(1, len(p) // 10)
                top_real = float(np.mean(r[sorted_idx[-decile:]]))
                bot_real = float(np.mean(r[sorted_idx[:decile]]))
                spread = top_real - bot_real

                n = len(p)
                t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)
                p_value = float(2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2)))
            else:
                ic = rank_ic = 0.0
                dir_acc = 0.5
                mae = float("inf")
                ic_lo = ic_hi = 0.0
                spread = 0.0
                p_value = 1.0

            agg["returns"][h] = {
                "ic": ic, "rank_ic": rank_ic, "dir_acc": dir_acc,
                "mae": mae, "ic_95_lo": ic_lo, "ic_95_hi": ic_hi,
                "decile_spread": spread, "p_value": p_value, "n_samples": len(p),
            }

        # Representation health
        agg["ctx_embed_std"] = float(np.mean(raw["representation"]["ctx_std"]))
        agg["tgt_embed_std"] = float(np.mean(raw["representation"]["tgt_std"]))
        agg["contrastive_alignment"] = float(
            np.mean(raw["representation"]["contrastive_acc"])
        )
        agg["ctx_cov_off_diag"] = float(
            np.mean(raw["representation"]["ctx_cov_off_diag"])
        )
        agg["tgt_cov_off_diag"] = float(
            np.mean(raw["representation"]["tgt_cov_off_diag"])
        )

        # Per-dimension analysis
        if raw["representation"]["ctx_dim_stds"]:
            all_ctx_stds = np.stack(raw["representation"]["ctx_dim_stds"])
            mean_dim_stds = np.mean(all_ctx_stds, axis=0)
            agg["dead_dims"] = int(np.sum(mean_dim_stds < 0.01))
            agg["low_var_dims"] = int(np.sum(mean_dim_stds < 0.05))
            agg["dim_std_min"] = float(np.min(mean_dim_stds))
            agg["dim_std_max"] = float(np.max(mean_dim_stds))
            agg["dim_std_median"] = float(np.median(mean_dim_stds))

            # Effective rank (proxy for collapse)
            norms = np.linalg.svd(all_ctx_stds.T, compute_uv=False)
            if norms.sum() > 0:
                p_norm = norms / norms.sum()
                p_norm = p_norm[p_norm > 1e-10]
                agg["effective_rank"] = float(
                    np.exp(-np.sum(p_norm * np.log(p_norm)))
                )
            else:
                agg["effective_rank"] = 0.0
        else:
            agg["dead_dims"] = agg["low_var_dims"] = 0
            agg["dim_std_min"] = agg["dim_std_max"] = agg["dim_std_median"] = 0.0
            agg["effective_rank"] = 0.0

        # Regime classification
        all_preds = np.concatenate(raw["regime"]["preds"])
        all_labels = np.concatenate(raw["regime"]["labels"])
        agg["regime_acc"] = float(np.mean(all_preds == all_labels))
        agg["regime_per_class"] = {}
        for c, name in enumerate(["bearish", "neutral", "bullish"]):
            mask_c = all_labels == c
            if mask_c.sum() > 0:
                agg["regime_per_class"][name] = float(np.mean(all_preds[mask_c] == c))
            else:
                agg["regime_per_class"][name] = 0.0

        return agg

    def _bootstrap_ic(self, preds, reals, n_bootstrap=1000):
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

    # --- Reporting ------------------------------------------------------------

    def _print_asset_report(self, r, name):
        # Auxiliary Reconstruction (base features only)
        base_feature_list = self.feature_list[:self.base_dim]
        print(f"\n  AUXILIARY RECONSTRUCTION (base features only, {self.base_dim} of {self.n_features})")
        print(f"    Total MSE:       {r['rec_mse']:.6f}")
        print(f"    Per-Feature Breakdown:")
        for i, feat_name in enumerate(base_feature_list):
            mse = r["per_feature_mse"][i]
            r2 = r["per_feature_r2"][i]
            bar = "#" * max(0, min(20, int(r2 * 20))) if r2 > 0 else ""
            status = "PASS" if mse < 0.5 else "WARN" if mse < 1.0 else "FAIL"
            print(f"      {status} {feat_name:<22} MSE:{mse:.5f}  R2:{r2:+.3f}  [{bar}]")

        # Representation Health (V6.2-specific)
        print(f"\n  REPRESENTATION HEALTH (Causal JEPA + Adversarial + Anti-Mem)")
        print(f"    Context Embed Std:    {r['ctx_embed_std']:.4f}  "
              f"{'PASS' if r['ctx_embed_std'] > GATE_EMBED_STD_MIN else 'FAIL (collapse?)'}")
        print(f"    Target Embed Std:     {r['tgt_embed_std']:.4f}")
        print(f"    Contrastive Align:    {r['contrastive_alignment']:.4f}")
        print(f"    Ctx Off-Diag Cov:     {r['ctx_cov_off_diag']:.6f}  "
              f"(lower=more decorrelated)")
        print(f"    Dead Dimensions:      {r['dead_dims']}/{WM_D_LATENT}  "
              f"{'PASS' if r['dead_dims'] < WM_D_LATENT // 4 else 'FAIL'}")
        print(f"    Low-Var Dimensions:   {r['low_var_dims']}/{WM_D_LATENT}")
        print(f"    Dim Std Range:        [{r['dim_std_min']:.4f}, {r['dim_std_max']:.4f}]  "
              f"median={r['dim_std_median']:.4f}")
        print(f"    Effective Rank:       {r['effective_rank']:.1f}/{WM_D_LATENT}")

        # Loss Components (V6.2-specific adversarial + anti-mem)
        lc = r.get("loss_components", {})
        if lc:
            print(f"\n  LOSS COMPONENTS (V6.2 Adversarial + Anti-Memorization)")
            print(f"    Contrastive:     {lc.get('contrastive', 0):.4f}")
            print(f"    VICReg:          {lc.get('vicreg', 0):.4f}")
            print(f"    Reconstruction:  {lc.get('recon', 0):.4f}")
            print(f"    Adversarial:     {lc.get('adv', 0):.4f}  "
                  f"(encoder fools discriminator)")
            print(f"    Discriminator:   {lc.get('disc', 0):.4f}  "
                  f"(separates real/shuffled)")
            print(f"    Regime:          {lc.get('regime', 0):.4f}")

        # Return Prediction
        print(f"\n  RETURN PREDICTION")
        for h in REWARD_HORIZONS:
            ret = r["returns"][h]
            sig = ("***" if ret["p_value"] < 0.001
                   else "**" if ret["p_value"] < 0.01
                   else "*" if ret["p_value"] < 0.05
                   else "")
            print(
                f"    t+{h:<3} IC:{ret['ic']:+.4f}{sig:<4} "
                f"RankIC:{ret['rank_ic']:+.4f}  "
                f"Dir:{ret['dir_acc']*100:.1f}%  MAE:{ret['mae']:.6f}  "
                f"CI:[{ret['ic_95_lo']:+.4f},{ret['ic_95_hi']:+.4f}]  "
                f"Spread:{ret['decile_spread']:+.6f}  n={ret['n_samples']}"
            )

        # Regime
        print(f"\n  REGIME CLASSIFICATION")
        print(f"    Overall Accuracy: {r['regime_acc']*100:.1f}%")
        for cls_name, acc in r["regime_per_class"].items():
            print(f"    {cls_name:<10} accuracy: {acc*100:.1f}%")

    def _print_aggregate_report(self, all_results):
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS ACROSS ALL ASSETS")
        print(f"{'='*70}")

        n = len(all_results)

        # Reconstruction
        avg_rec = np.mean([r["rec_mse"] for r in all_results.values()])
        print(f"\n  Auxiliary Reconstruction:")
        print(f"    Avg MSE:         {avg_rec:.6f}")

        # Representation Health
        avg_ctx_std = np.mean([r["ctx_embed_std"] for r in all_results.values()])
        avg_dead = np.mean([r["dead_dims"] for r in all_results.values()])
        avg_rank = np.mean([r["effective_rank"] for r in all_results.values()])
        print(f"\n  Representation Health:")
        print(f"    Avg Embed Std:   {avg_ctx_std:.4f}  "
              f"{'PASS' if avg_ctx_std > GATE_EMBED_STD_MIN else 'FAIL'}")
        print(f"    Avg Dead Dims:   {avg_dead:.1f}/{WM_D_LATENT}")
        print(f"    Avg Eff. Rank:   {avg_rank:.1f}/{WM_D_LATENT}")

        # Adversarial Health
        avg_adv = np.mean([
            r.get("loss_components", {}).get("adv", 0)
            for r in all_results.values()
        ])
        avg_disc = np.mean([
            r.get("loss_components", {}).get("disc", 0)
            for r in all_results.values()
        ])
        print(f"\n  Adversarial Health:")
        print(f"    Avg Adv Loss:    {avg_adv:.4f}  (encoder fools disc)")
        print(f"    Avg Disc Loss:   {avg_disc:.4f}  (disc separates real/shuffled)")

        # Returns
        print(f"\n  Return Prediction (averaged across {n} assets):")
        avg_ics = {}
        for h in REWARD_HORIZONS:
            ics = [r["returns"][h]["ic"] for r in all_results.values()]
            rank_ics = [r["returns"][h]["rank_ic"] for r in all_results.values()]
            dirs = [r["returns"][h]["dir_acc"] for r in all_results.values()]
            avg_ic = np.mean(ics)
            avg_ics[h] = avg_ic
            print(
                f"    t+{h:<3} IC:{avg_ic:+.4f}  "
                f"RankIC:{np.mean(rank_ics):+.4f}  "
                f"Dir:{np.mean(dirs)*100:.1f}%  "
                f"[range: {min(ics):+.4f} to {max(ics):+.4f}]"
            )

        mean_ic = np.mean(list(avg_ics.values()))
        print(f"    Mean IC across horizons: {mean_ic:+.4f}  "
              f"{'PASS' if mean_ic > GATE_IC_MIN else 'FAIL'}")

        # Per-asset summary
        print(f"\n  Per-Asset Summary:")
        print(f"    {'Asset':<10} {'RecMSE':>8} {'IC(1)':>8} {'IC(4)':>8} "
              f"{'IC(16)':>8} {'IC(64)':>8} {'EmbStd':>7} {'Dir(1)':>7}")
        print(f"    {'-'*70}")
        for name, r in all_results.items():
            print(
                f"    {name:<10} "
                f"{r['rec_mse']:>8.5f} "
                f"{r['returns'][1]['ic']:>+8.4f} "
                f"{r['returns'][4]['ic']:>+8.4f} "
                f"{r['returns'][16]['ic']:>+8.4f} "
                f"{r['returns'][64]['ic']:>+8.4f} "
                f"{r['ctx_embed_std']:>7.4f} "
                f"{r['returns'][1]['dir_acc']*100:>6.1f}%"
            )

    @torch.no_grad()
    def _compute_shuffled_ic(self, segments, n_seeds=3):
        """
        Compute IC on temporally-shuffled sequences to detect memorization.
        If model memorized temporal patterns, shuffled IC drops to ~0.
        If model learned feature->return signal, shuffled IC remains positive.
        """
        all_shuffled_ics = []

        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset * 1000
            rng = np.random.default_rng(seed)
            all_preds, all_reals = [], []

            for feats, targets, asset_idx, asset_name in segments:
                seq_len = WM_SEQ_LEN
                indices = list(range(0, len(feats) - seq_len, seq_len))
                if not indices:
                    continue

                for i in indices:
                    obs_np = feats[i:i+seq_len].copy()
                    perm = rng.permutation(seq_len)
                    obs_shuffled = obs_np[perm]

                    obs = torch.from_numpy(obs_shuffled).unsqueeze(0).float().to(DEVICE)
                    asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
                    if self.revin is not None:

                        obs = self.revin(obs, mode='norm')

                    with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                        outputs = self.model.forward_train(obs, asset)

                    logits = outputs["return_logits"][1]
                    pred = self.model.bucketer.decode(logits).cpu().numpy().flatten()
                    real = targets[1][i:i+seq_len]
                    all_preds.extend(pred)
                    all_reals.extend(real)

            preds_arr = np.array(all_preds)
            reals_arr = np.array(all_reals)
            mask = np.isfinite(preds_arr) & np.isfinite(reals_arr)
            if mask.sum() > 50:
                ic = float(np.corrcoef(preds_arr[mask], reals_arr[mask])[0, 1])
                if np.isfinite(ic):
                    all_shuffled_ics.append(ic)

        return float(np.mean(all_shuffled_ics)) if all_shuffled_ics else 0.0

    def _check_gates(self, all_results, shuffled_ic=None, loss_ratio=None, ic_h1_for_shic=None):
        print(f"\n{'='*70}")
        print(f"  VALIDATION GATES")
        print(f"{'='*70}")

        avg_ctx_std = np.mean([r["ctx_embed_std"] for r in all_results.values()])
        avg_ics = {
            h: np.mean([r["returns"][h]["ic"] for r in all_results.values()])
            for h in REWARD_HORIZONS
        }
        mean_ic = np.mean(list(avg_ics.values()))

        # V6 uses contrastive accuracy gate
        avg_align = np.mean([
            r["contrastive_alignment"] for r in all_results.values()
        ])

        gates = {
            "Embedding Std (collapse)": (
                avg_ctx_std > GATE_EMBED_STD_MIN,
                f"{avg_ctx_std:.4f} > {GATE_EMBED_STD_MIN}",
            ),
            "Mean IC": (
                mean_ic > GATE_IC_MIN,
                f"{mean_ic:+.4f} > {GATE_IC_MIN}",
            ),
            "Contrastive Alignment": (
                avg_align > GATE_CONTRASTIVE_MIN,
                f"{avg_align:.4f} > {GATE_CONTRASTIVE_MIN}",
            ),
        }

        # Shuffled IC gate (anti-memorization)
        if shuffled_ic is not None:
            shic_denom = ic_h1_for_shic if ic_h1_for_shic is not None else mean_ic
            if abs(shic_denom) > 1e-6:
                shuffled_ratio = shuffled_ic / shic_denom
                gates["Shuffled IC Ratio"] = (
                    shuffled_ratio > GATE_SHUFFLED_IC_RATIO_MIN,
                    f"{shuffled_ratio:.3f} > {GATE_SHUFFLED_IC_RATIO_MIN}"
                )
            else:
                gates["Shuffled IC Ratio"] = (False, "contiguous IC near zero")

        # FIX: Enforce GATE_LOSS_RATIO_MAX (was dead code, now active)
        if loss_ratio is not None and loss_ratio > 0:
            gates["Train/Val Loss Ratio"] = (
                loss_ratio < GATE_LOSS_RATIO_MAX,
                f"{loss_ratio:.3f} < {GATE_LOSS_RATIO_MAX}"
            )

        all_pass = True
        for gate_name, (passed, desc) in gates.items():
            status = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
            print(f"    [{status}] {gate_name:<30} {desc}")

        print(f"\n  {'='*40}")
        if all_pass:
            print(f"  VERDICT: ALL GATES PASSED")
            print(f"  World model is ready for agent training.")
        else:
            print(f"  VERDICT: GATE(S) FAILED")
            print(f"  World model needs more training or tuning.")
        print(f"  {'='*40}")

        return all_pass

    def _save_results(self, all_results, gate_pass):
        output = {
            "version": "v6_2_causal_jepa_adversarial_antimem",
            "model": self.model_name,
            "n_features": self.n_features,
            "base_dim": self.base_dim,
            "timestamp": datetime.now().isoformat(),
            "gate_passed": gate_pass,
            "results": {},
        }
        for name, r in all_results.items():
            output["results"][name] = {
                "rec_mse": r["rec_mse"],
                "ctx_embed_std": r["ctx_embed_std"],
                "contrastive_alignment": r["contrastive_alignment"],
                "dead_dims": r["dead_dims"],
                "effective_rank": r["effective_rank"],
                "loss_components": r.get("loss_components", {}),
                "returns": {
                    str(h): {
                        "ic": r["returns"][h]["ic"],
                        "rank_ic": r["returns"][h]["rank_ic"],
                        "dir_acc": r["returns"][h]["dir_acc"],
                    }
                    for h in REWARD_HORIZONS
                },
                "regime_acc": r["regime_acc"],
            }

        out_path = LOG_DIR / (
            f"validation_{self.model_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path.name}")

    # --- Robust Validation ---------------------------------------------------

    @torch.no_grad()
    def run_robust_validation(self, horizon=1):
        """
        Run comprehensive robustness validation to detect overfitting.

        Tests:
          1. Temporal forward walk (expanding window)
          2. Shuffled K-fold (non-temporal)
          3. Regime-specific holdout
          4. Stability analysis

        Detects hallucination/overfitting if:
          - IC degrades significantly in forward walk
          - Shuffled IC << baseline IC
          - High variance across folds
          - Only works in specific regimes
        """
        if not ROBUST_VALIDATION_AVAILABLE:
            print("  [ERROR] validation_utils not found. Cannot run robust validation.")
            return False

        print(f"\n{'='*70}")
        print(f"  V6.2 CAUSAL JEPA + ADVERSARIAL + ANTI-MEM ROBUST VALIDATION (Overfitting Detection)")
        print(f"  Model: {self.model_name}")
        print(f"  Features: {self.n_features} (base_dim={self.base_dim})")
        print(f"  Horizon: t+{horizon}")
        print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Load full dataset
        print(f"\n  Loading full dataset from {DATA_DIR}")
        segments = self.load_validation_data(use_full_data=True)

        if not segments:
            print("  [ERROR] No validation data found.")
            return False

        # Create prediction function
        def predict_fn(data: np.ndarray, asset_idx: int) -> np.ndarray:
            """Run model on data sequences, return predictions aligned with targets."""
            seq_len = WM_SEQ_LEN
            n_samples = len(data)

            # Initialize prediction array (NaN for unpredicted timesteps)
            predictions = np.full(n_samples, np.nan, dtype=np.float32)

            # FIX: Use non-overlapping stride to prevent IC inflation
            indices = list(range(0, n_samples - seq_len, seq_len))
            if not indices and n_samples >= seq_len:
                indices = [0]

            for i in indices:
                obs_np = data[i:i+seq_len]
                if len(obs_np) < seq_len:
                    break

                obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
                asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
                if self.revin is not None:

                    obs = self.revin(obs, mode='norm')

                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    outputs = self.model.forward_train(obs, asset)
                    logits = outputs["return_logits"][horizon]
                    preds = self.model.bucketer.decode(logits).cpu().numpy().flatten()

                # Assign predictions to corresponding timesteps
                # For overlapping windows, take average
                for j, pred in enumerate(preds):
                    timestep = i + j
                    if timestep < n_samples:
                        if np.isnan(predictions[timestep]):
                            predictions[timestep] = pred
                        else:
                            # Average overlapping predictions
                            predictions[timestep] = (
                                predictions[timestep] + pred
                            ) / 2.0

            return predictions

        # Run robust validation
        config = ValidationConfig()
        validator = RobustValidator(self.model, config)

        robust_results = validator.run_comprehensive_validation(
            segments, predict_fn, horizon=horizon
        )

        # Print aggregate report
        validator.print_aggregate_report(robust_results)

        # Check gates
        all_passed = True
        for asset_name, results in robust_results.items():
            passed, gates = results.passes_robustness_gates(config)
            if not passed:
                all_passed = False

        # Save results
        output = {
            "version": "v6_2_causal_jepa_adversarial_antimem",
            "model": self.model_name,
            "n_features": self.n_features,
            "base_dim": self.base_dim,
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon,
            "robust_validation": True,
            "robustness_passed": all_passed,
            "results": {},
        }

        for asset_name, r in robust_results.items():
            output["results"][asset_name] = {
                "baseline_ic": r.baseline_ic,
                "forward_walk_ics": r.forward_walk_ics,
                "shuffled_ics": r.shuffled_ics,
                "regime_ics": r.regime_ics,
                "ic_stability": r.ic_stability,
                "hallucination_score": r.hallucination_score,
                "hallucination_reasons": r.hallucination_reasons,
                "ic_degradation": r.ic_degradation,
            }

        out_path = LOG_DIR / (
            f"robust_validation_{self.model_name}_h{horizon}_"
            f"{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Robust validation results saved to: {out_path.name}")

        return all_passed


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    log_path = setup_logging(LOG_DIR, "v6_2_validate")
    parser = argparse.ArgumentParser(
        description="V6.2 Causal JEPA + Adversarial + Anti-Memorization World Model Comprehensive Validation"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific checkpoint")
    parser.add_argument("--latest", action="store_true",
                        help="Use latest checkpoint")
    parser.add_argument("--both", action="store_true",
                        help="Run both best and latest")
    parser.add_argument("--robust", action="store_true",
                        help="Run robust validation (detects overfitting)")
    parser.add_argument("--horizon", type=int, default=1,
                        help="Horizon for robust validation (1,4,16,64)")
    parser.add_argument("--features", type=int, choices=[13, 17, 18, 22], default=22,
                        help="Number of features: 13 (base only) or 18 (full)")
    args = parser.parse_args()

    n_features = args.features
    feat_tag = f"f{n_features}"
    ckpt_prefix = f"v6_2_{feat_tag}"

    if args.both:
        print("\n  Running validation on BOTH checkpoints...\n")
        for name, path in [
            ("best_ema", BASE_MODEL_DIR / f"{ckpt_prefix}_wm_best_ema.pt"),
            ("latest", BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"),
        ]:
            if path.exists():
                print(f"\n{'#'*70}")
                print(f"  CHECKPOINT: {name} ({n_features}f)")
                print(f"{'#'*70}")
                v = WorldModelValidator(path, n_features=n_features)
                if args.robust:
                    v.run_robust_validation(horizon=args.horizon)
                else:
                    v.run_diagnostics()
    elif args.robust:
        # Robust validation mode
        model_path = Path(args.model) if args.model else None
        if args.latest and not model_path:
            model_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"
        v = WorldModelValidator(model_path, n_features=n_features)
        passed = v.run_robust_validation(horizon=args.horizon)
        teardown_logging()
        sys.exit(0 if passed else 1)
    elif args.model:
        v = WorldModelValidator(Path(args.model), n_features=n_features)
        passed = v.run_diagnostics()
        teardown_logging()
        sys.exit(0 if passed else 1)
    elif args.latest:
        v = WorldModelValidator(BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt",
                                n_features=n_features)
        passed = v.run_diagnostics()
        teardown_logging()
        sys.exit(0 if passed else 1)
    else:
        v = WorldModelValidator(n_features=n_features)
        passed = v.run_diagnostics()
        teardown_logging()
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
