"""
V3.3 World Model Validation — WaveNet-GRU-RSSM + XD Anti-Memorization

Supports --features 13|18|30|37.

Usage:
    python validate_world.py --features 13      # Legacy 13 features
    python validate_world.py --features 37      # Full 37 features (default)
    python validate_world.py --latest           # Latest checkpoint
    python validate_world.py --both             # Run both and compare
    python validate_world.py --model path.pt    # Specific checkpoint
"""
import torch
import torch.nn.functional as F
import torch.distributions as D
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
from world_model import WaveNetGRUWorldModel, count_parameters
from log_utils import setup_logging, teardown_logging
from revin import RevIN
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
    """Comprehensive diagnostics for V3.3 WaveNet-GRU World Model."""

    def __init__(self, model_path: Path = None, n_features: int = 22):
        feature_list, input_dim, base_dim = get_feature_config(n_features)
        self.feature_list = feature_list
        self.input_dim = input_dim
        self.base_dim = base_dim
        self.n_features = n_features
        self.model = WaveNetGRUWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
        self.model_name = "unknown"
        self.revin = None  # Created only if checkpoint has revin_state_dict
        self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path: Path):
        feat_tag = f"f{self.n_features}"
        prefix = f"v3_3_{feat_tag}"
        if model_path is None:
            model_path = BASE_MODEL_DIR / f"{prefix}_wm_best_ema.pt"
        if not model_path.exists():
            model_path = BASE_MODEL_DIR / f"{prefix}_wm_latest.pt"
        if not model_path.exists():
            print(f"  [ERROR] No checkpoint found in {BASE_MODEL_DIR}")
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
        elif isinstance(ckpt, dict) and any(k.startswith("obs_encoder") or k.startswith("wavenet") for k in ckpt):
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
        """Load validation data. If use_full_data=True, load all data for robust validation."""
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
                df_val = df

            feats, targets = extract_features_targets(
                df_val, self.feature_list, REWARD_HORIZONS, asset_name
            )

            segments.append((feats, targets, asset_idx, asset_name))
            print(f"    {asset_name}: {len(feats):,} {'total' if use_full_data else 'validation'} bars")

        return segments

    # --- Train/Val Loss Computation (for overfitting gate) -------------------

    @torch.no_grad()
    def _compute_split_loss(self, segments):
        """Compute average total loss on a set of data segments for GATE_LOSS_RATIO_MAX."""
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
                    _, loss_dict, _ = self.model.get_loss(obs, asset, targ, mask_ratio=0.0)
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
        print(f"  V3.3 WAVENET-GRU WORLD MODEL COMPREHENSIVE VALIDATION")
        print(f"  Model: {self.model_name}")
        print(f"  Features: {self.n_features} (input_dim={self.input_dim}, base_dim={self.base_dim})")
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
        # FIX: Enforce GATE_LOSS_RATIO_MAX which was previously dead code
        print(f"\n  Computing train/val loss ratio (overfitting check)...")
        train_segments_data = self._load_train_split()
        train_loss = self._compute_split_loss(train_segments_data) if train_segments_data else 0.0
        val_loss = self._compute_split_loss(segments)
        loss_ratio = val_loss / train_loss if train_loss > 1e-8 else 0.0  # val/train: >2.0 = overfitting
        print(f"    Train Loss:    {train_loss:.4f}")
        print(f"    Val Loss:      {val_loss:.4f}")
        print(f"    Ratio:         {loss_ratio:.3f} (gate: < {GATE_LOSS_RATIO_MAX})")

        self._print_aggregate_report(all_results)
        ic_h1 = np.mean([r["returns"][1]["ic"] for r in all_results.values()])
        gate_pass = self._check_gates(all_results, shuffled_ic=shuffled_ic, loss_ratio=loss_ratio, ic_h1_for_shic=ic_h1)
        self._save_results(all_results, gate_pass)

        return gate_pass

    # --- Per-Asset Evaluation -------------------------------------------------

    def _evaluate_asset(self, feats, targets, asset_idx):
        seq_len = WM_SEQ_LEN
        recon_features = self.feature_list[:self.base_dim]  # Only base features reconstructed
        results = {
            "reconstruction": {"total_mse": [], "per_feature_mse": [[] for _ in recon_features],
                                "per_feature_r2": [[] for _ in recon_features], "cosine_sim": []},
            "returns": {h: {"preds": [], "reals": []} for h in REWARD_HORIZONS},
            "latent": {"kl": [], "post_entropy": [], "prior_entropy": [],
                       "utilization": [], "prior_post_cosine": []},
            "imagination": {mr: [] for mr in [0.0, 0.25, 0.50, 0.75]},
            "regime": {"preds": [], "labels": []},
            "dream": {"ic_by_step": {s: {"preds": [], "reals": []} for s in range(1, 6)}},
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

            # -- Clean forward pass ----------------------------------------
            if self.revin is not None:

                obs = self.revin(obs, mode='norm')
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                outputs = self.model.forward_train(obs, asset)

            recon = outputs["recon"]
            prior_logits = outputs["prior_logits"]
            post_logits = outputs["post_logits"]

            # -- 1. Reconstruction (base features only) --------------------
            obs_base = obs[:, :, :self.base_dim]
            rec_mse = F.mse_loss(recon, obs_base).item()
            results["reconstruction"]["total_mse"].append(rec_mse)

            cos_sim = F.cosine_similarity(recon.reshape(-1), obs_base.reshape(-1), dim=0).item()
            results["reconstruction"]["cosine_sim"].append(cos_sim)

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

            # -- 3. Latent Health ------------------------------------------
            prior = prior_logits.view(-1, RSSM_LATENT_DIM, RSSM_CLASSES)
            post = post_logits.view(-1, RSSM_LATENT_DIM, RSSM_CLASSES)

            kl = D.kl_divergence(
                D.Categorical(logits=post), D.Categorical(logits=prior)
            ).mean().item()
            results["latent"]["kl"].append(kl)

            post_probs = F.softmax(post.float(), dim=-1)
            prior_probs = F.softmax(prior.float(), dim=-1)

            post_ent = -(post_probs * post_probs.clamp(min=1e-8).log()).sum(-1).mean().item()
            prior_ent = -(prior_probs * prior_probs.clamp(min=1e-8).log()).sum(-1).mean().item()
            results["latent"]["post_entropy"].append(post_ent)
            results["latent"]["prior_entropy"].append(prior_ent)

            uniform = 1.0 / RSSM_CLASSES
            active = (post_probs.mean(0) > uniform * 0.5).float().mean().item()
            results["latent"]["utilization"].append(active)

            pp_cos = F.cosine_similarity(
                prior_probs.reshape(prior.shape[0], -1),
                post_probs.reshape(post.shape[0], -1), dim=-1
            ).mean().item()
            results["latent"]["prior_post_cosine"].append(pp_cos)

            # -- 4. Imagination (varying mask levels) ----------------------
            for mask_ratio in [0.0, 0.25, 0.50, 0.75]:
                masked = obs.clone()
                if mask_ratio > 0:
                    block_size = max(4, int(seq_len * 0.10))
                    num_blocks = max(1, int((seq_len * mask_ratio) / block_size))
                    for _ in range(num_blocks):
                        start = np.random.randint(0, max(1, seq_len - block_size))
                        masked[:, start:start + block_size, :] = 0.0

                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    imag_out = self.model.forward_train(obs, asset, masked)
                imag_mse = F.mse_loss(imag_out["recon"], obs[:, :, :self.base_dim]).item()
                results["imagination"][mask_ratio].append(imag_mse)

            # -- 5. Regime Classification ----------------------------------
            regime_logits = outputs["regime_logits"].reshape(-1, 3)
            regime_preds = regime_logits.argmax(dim=-1).cpu().numpy()

            ret_1 = targ_tensors[1].reshape(-1).cpu().numpy()
            ret_std = np.std(ret_1) + 1e-6
            regime_labels = np.ones_like(ret_1, dtype=np.int64)
            regime_labels[ret_1 > ret_std * 0.5] = 2
            regime_labels[ret_1 < -ret_std * 0.5] = 0
            results["regime"]["preds"].append(regime_preds)
            results["regime"]["labels"].append(regime_labels)

            # -- 6. Dream Coherence (V3: GRU-based multi-step) ------------
            if i + seq_len + 5 < len(feats):
                h_seq = outputs["h_seq"]
                z_post = outputs["z_post"]
                h_last = h_seq[:, -1, :]
                z_last = z_post[:, -1, :]
                gru_hidden = None  # V3 dream_step manages GRU hidden state

                for step in range(1, 6):
                    h_last, z_next, gru_hidden, dream_rets = self.model.dream_step(
                        h_last, z_last, gru_hidden
                    )
                    z_last = z_next

                    future_idx = i + seq_len + step - 1
                    if future_idx < len(targets[1]):
                        results["dream"]["ic_by_step"][step]["preds"].append(
                            dream_rets[1].cpu().numpy().flatten()[0]
                        )
                        results["dream"]["ic_by_step"][step]["reals"].append(
                            targets[1][future_idx]
                        )

        return self._aggregate_results(results)

    def _aggregate_results(self, raw):
        agg = {}

        # Reconstruction
        agg["rec_mse"] = float(np.mean(raw["reconstruction"]["total_mse"]))
        agg["rec_cosine"] = float(np.mean(raw["reconstruction"]["cosine_sim"]))
        agg["per_feature_mse"] = [float(np.mean(v)) if v else 0.0
                                    for v in raw["reconstruction"]["per_feature_mse"]]
        agg["per_feature_r2"] = [float(np.mean(v)) if v else 0.0
                                   for v in raw["reconstruction"]["per_feature_r2"]]

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

        # Latent health
        agg["kl_mean"] = float(np.mean(raw["latent"]["kl"]))
        agg["kl_std"] = float(np.std(raw["latent"]["kl"]))
        agg["kl_min"] = float(np.min(raw["latent"]["kl"])) if raw["latent"]["kl"] else 0.0
        agg["kl_max"] = float(np.max(raw["latent"]["kl"])) if raw["latent"]["kl"] else 0.0
        agg["post_entropy"] = float(np.mean(raw["latent"]["post_entropy"]))
        agg["prior_entropy"] = float(np.mean(raw["latent"]["prior_entropy"]))
        agg["utilization"] = float(np.mean(raw["latent"]["utilization"]))
        agg["prior_post_cosine"] = float(np.mean(raw["latent"]["prior_post_cosine"]))
        max_entropy = float(np.log(RSSM_CLASSES))
        agg["entropy_ratio"] = agg["post_entropy"] / max_entropy if max_entropy > 0 else 0.0

        # Imagination degradation
        agg["imagination"] = {}
        for mr in [0.0, 0.25, 0.50, 0.75]:
            agg["imagination"][mr] = float(np.mean(raw["imagination"][mr]))
        base = max(agg["imagination"][0.0], 1e-9)
        agg["imag_degradation_50"] = agg["imagination"][0.50] / base
        agg["imag_degradation_75"] = agg["imagination"][0.75] / base

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

        # Dream coherence
        agg["dream_ic"] = {}
        for step in range(1, 6):
            p = np.array(raw["dream"]["ic_by_step"][step]["preds"])
            r = np.array(raw["dream"]["ic_by_step"][step]["reals"])
            if len(p) > 20:
                mask = np.isfinite(p) & np.isfinite(r)
                if mask.sum() > 20:
                    agg["dream_ic"][step] = float(np.corrcoef(p[mask], r[mask])[0, 1])
                else:
                    agg["dream_ic"][step] = 0.0
            else:
                agg["dream_ic"][step] = 0.0

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
        # Reconstruction
        print(f"\n  RECONSTRUCTION")
        print(f"    Total MSE:       {r['rec_mse']:.6f}  {'PASS' if r['rec_mse'] < GATE_REC_MSE_MAX else 'FAIL'}")
        print(f"    Cosine Sim:      {r['rec_cosine']:.4f}")
        print(f"    Per-Feature Breakdown (base features, {self.base_dim}D):")
        for i, feat_name in enumerate(self.feature_list[:self.base_dim]):
            mse = r["per_feature_mse"][i]
            r2 = r["per_feature_r2"][i]
            bar = "#" * max(0, min(20, int(r2 * 20))) if r2 > 0 else ""
            status = "PASS" if mse < 0.5 else "WARN" if mse < 1.0 else "FAIL"
            print(f"      {status} {feat_name:<22} MSE:{mse:.5f}  R2:{r2:+.3f}  [{bar}]")

        # Return Prediction
        print(f"\n  RETURN PREDICTION")
        for h in REWARD_HORIZONS:
            ret = r["returns"][h]
            sig = "***" if ret["p_value"] < 0.001 else "**" if ret["p_value"] < 0.01 else "*" if ret["p_value"] < 0.05 else ""
            print(f"    t+{h:<3} IC:{ret['ic']:+.4f}{sig:<4} RankIC:{ret['rank_ic']:+.4f}  "
                  f"Dir:{ret['dir_acc']*100:.1f}%  MAE:{ret['mae']:.6f}  "
                  f"CI:[{ret['ic_95_lo']:+.4f},{ret['ic_95_hi']:+.4f}]  "
                  f"Spread:{ret['decile_spread']:+.6f}  n={ret['n_samples']}")

        # Latent Health
        print(f"\n  LATENT HEALTH")
        print(f"    KL Divergence:   {r['kl_mean']:.4f} +/- {r['kl_std']:.4f}  "
              f"[{r['kl_min']:.4f}, {r['kl_max']:.4f}]  "
              f"{'PASS' if GATE_KL_MIN < r['kl_mean'] < GATE_KL_MAX else 'FAIL'}")
        print(f"    Post Entropy:    {r['post_entropy']:.4f}  "
              f"({r['entropy_ratio']*100:.1f}% of max {np.log(RSSM_CLASSES):.2f})")
        print(f"    Prior Entropy:   {r['prior_entropy']:.4f}")
        print(f"    Utilization:     {r['utilization']*100:.1f}%")
        print(f"    Prior-Post Cos:  {r['prior_post_cosine']:.4f}")

        # Imagination
        print(f"\n  IMAGINATION (reconstruction under masking)")
        for mr in [0.0, 0.25, 0.50, 0.75]:
            label = f"{int(mr*100)}% masked"
            print(f"    {label:<14} MSE: {r['imagination'][mr]:.6f}")
        print(f"    Degradation @50%: {r['imag_degradation_50']:.2f}x")
        print(f"    Degradation @75%: {r['imag_degradation_75']:.2f}x")

        # Regime
        print(f"\n  REGIME CLASSIFICATION")
        print(f"    Overall Accuracy: {r['regime_acc']*100:.1f}%")
        for cls_name, acc in r["regime_per_class"].items():
            print(f"    {cls_name:<10} accuracy: {acc*100:.1f}%")

        # Dream
        if any(v != 0.0 for v in r["dream_ic"].values()):
            print(f"\n  DREAM COHERENCE (GRU-based multi-step rollout)")
            for step, ic_val in r["dream_ic"].items():
                bar = "#" * max(0, int(abs(ic_val) * 50))
                print(f"    Step {step}: IC = {ic_val:+.4f}  [{bar}]")

    def _print_aggregate_report(self, all_results):
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS ACROSS ALL ASSETS")
        print(f"{'='*70}")

        n = len(all_results)

        # Reconstruction
        avg_rec = np.mean([r["rec_mse"] for r in all_results.values()])
        avg_cos = np.mean([r["rec_cosine"] for r in all_results.values()])
        print(f"\n  Reconstruction:")
        print(f"    Avg MSE:         {avg_rec:.6f}  {'PASS' if avg_rec < GATE_REC_MSE_MAX else 'FAIL'}")
        print(f"    Avg Cosine:      {avg_cos:.4f}")

        # Returns
        print(f"\n  Return Prediction (averaged across {n} assets):")
        avg_ics = {}
        for h in REWARD_HORIZONS:
            ics = [r["returns"][h]["ic"] for r in all_results.values()]
            rank_ics = [r["returns"][h]["rank_ic"] for r in all_results.values()]
            dirs = [r["returns"][h]["dir_acc"] for r in all_results.values()]
            avg_ic = np.mean(ics)
            avg_ics[h] = avg_ic
            print(f"    t+{h:<3} IC:{avg_ic:+.4f}  RankIC:{np.mean(rank_ics):+.4f}  "
                  f"Dir:{np.mean(dirs)*100:.1f}%  [range: {min(ics):+.4f} to {max(ics):+.4f}]")

        mean_ic = np.mean(list(avg_ics.values()))
        print(f"    Mean IC across horizons: {mean_ic:+.4f}  "
              f"{'PASS' if mean_ic > GATE_IC_MIN else 'FAIL'}")

        # Latent
        avg_kl = np.mean([r["kl_mean"] for r in all_results.values()])
        avg_util = np.mean([r["utilization"] for r in all_results.values()])
        avg_ent = np.mean([r["entropy_ratio"] for r in all_results.values()])
        print(f"\n  Latent Health:")
        print(f"    Avg KL:          {avg_kl:.4f}  "
              f"{'PASS' if GATE_KL_MIN < avg_kl < GATE_KL_MAX else 'FAIL'}")
        print(f"    Avg Utilization: {avg_util*100:.1f}%")
        print(f"    Avg Entropy:     {avg_ent*100:.1f}% of max")

        # Imagination
        avg_imag50 = np.mean([r["imag_degradation_50"] for r in all_results.values()])
        avg_imag75 = np.mean([r["imag_degradation_75"] for r in all_results.values()])
        print(f"\n  Imagination:")
        print(f"    Avg Degradation @50%: {avg_imag50:.2f}x")
        print(f"    Avg Degradation @75%: {avg_imag75:.2f}x")

        # Regime
        avg_regime = np.mean([r["regime_acc"] for r in all_results.values()])
        print(f"\n  Regime: {avg_regime*100:.1f}% accuracy")

        # Per-asset summary table
        print(f"\n  Per-Asset Summary:")
        print(f"    {'Asset':<10} {'RecMSE':>8} {'IC(1)':>8} {'IC(4)':>8} {'IC(16)':>8} {'IC(64)':>8} {'KL':>6} {'Dir(1)':>7}")
        print(f"    {'-'*70}")
        for name, r in all_results.items():
            print(f"    {name:<10} "
                  f"{r['rec_mse']:>8.5f} "
                  f"{r['returns'][1]['ic']:>+8.4f} "
                  f"{r['returns'][4]['ic']:>+8.4f} "
                  f"{r['returns'][16]['ic']:>+8.4f} "
                  f"{r['returns'][64]['ic']:>+8.4f} "
                  f"{r['kl_mean']:>6.3f} "
                  f"{r['returns'][1]['dir_acc']*100:>6.1f}%")

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
                # Use non-overlapping windows for efficiency
                indices = list(range(0, len(feats) - seq_len, seq_len))
                if not indices:
                    continue

                for i in indices:
                    obs_np = feats[i:i+seq_len].copy()
                    # Shuffle temporal order within each window
                    perm = rng.permutation(seq_len)
                    obs_shuffled = obs_np[perm]

                    obs = torch.from_numpy(obs_shuffled).unsqueeze(0).float().to(DEVICE)
                    asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

                    if self.revin is not None:


                        obs = self.revin(obs, mode='norm')
                    with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                        outputs = self.model.forward_train(obs, asset)

                    # Use horizon 1 for shuffled IC check
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

        avg_rec = np.mean([r["rec_mse"] for r in all_results.values()])
        avg_kl = np.mean([r["kl_mean"] for r in all_results.values()])
        avg_ics = {h: np.mean([r["returns"][h]["ic"] for r in all_results.values()])
                   for h in REWARD_HORIZONS}
        mean_ic = np.mean(list(avg_ics.values()))

        gates = {
            "Reconstruction MSE": (avg_rec < GATE_REC_MSE_MAX,
                                    f"{avg_rec:.5f} < {GATE_REC_MSE_MAX}"),
            "Mean IC": (mean_ic > GATE_IC_MIN,
                        f"{mean_ic:+.4f} > {GATE_IC_MIN}"),
            "KL Lower Bound": (avg_kl > GATE_KL_MIN,
                                f"{avg_kl:.4f} > {GATE_KL_MIN}"),
            "KL Upper Bound": (avg_kl < GATE_KL_MAX,
                                f"{avg_kl:.4f} < {GATE_KL_MAX}"),
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
            print(f"    [{status}] {gate_name:<25} {desc}")

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
            "version": "v3_1",
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "gate_passed": gate_pass,
            "results": {},
        }
        for name, r in all_results.items():
            output["results"][name] = {
                "rec_mse": r["rec_mse"],
                "kl_mean": r["kl_mean"],
                "returns": {str(h): {
                    "ic": r["returns"][h]["ic"],
                    "rank_ic": r["returns"][h]["rank_ic"],
                    "dir_acc": r["returns"][h]["dir_acc"],
                } for h in REWARD_HORIZONS},
                "regime_acc": r["regime_acc"],
            }

        out_path = LOG_DIR / f"validation_{self.model_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {out_path.name}")

    # --- Robust Validation ---------------------------------------------------

    @torch.no_grad()
    def run_robust_validation(self, horizon=1):
        """
        Run robust validation to detect overfitting via:
          1. Temporal forward walk
          2. Shuffled K-fold
          3. Regime-specific holdout
          4. Hallucination detection
        """
        if not ROBUST_VALIDATION_AVAILABLE:
            print("  [ERROR] validation_utils not found. Cannot run robust validation.")
            return False

        print(f"\n{'='*70}")
        print(f"  V3.3 WAVENET-GRU-RSSM ROBUST VALIDATION (Overfitting Detection)")
        print(f"  Model: {self.model_name}")
        print(f"  Horizon: t+{horizon}")
        print(f"  Features: {self.n_features} (input_dim={self.input_dim}, base_dim={self.base_dim})")
        print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        print(f"\n  Loading full dataset from {DATA_DIR}")
        segments = self.load_validation_data(use_full_data=True)
        if not segments:
            print("  [ERROR] No validation data found.")
            return False

        def predict_fn(data: np.ndarray, asset_idx: int) -> np.ndarray:
            seq_len = WM_SEQ_LEN
            n_samples = len(data)
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

                for j, pred in enumerate(preds):
                    timestep = i + j
                    if timestep < n_samples:
                        if np.isnan(predictions[timestep]):
                            predictions[timestep] = pred
                        else:
                            predictions[timestep] = (predictions[timestep] + pred) / 2.0
            return predictions

        config = ValidationConfig()
        validator = RobustValidator(self.model, config)
        robust_results = validator.run_comprehensive_validation(
            segments, predict_fn, horizon=horizon
        )
        validator.print_aggregate_report(robust_results)

        all_passed = True
        for asset_name, results in robust_results.items():
            passed, gates = results.passes_robustness_gates(config)
            if not passed:
                all_passed = False

        output = {
            "version": "v3_1", "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "horizon": horizon, "robust_validation": True,
            "robustness_passed": all_passed, "results": {},
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

        out_path = LOG_DIR / f"robust_validation_{self.model_name}_h{horizon}_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Robust validation results saved to: {out_path.name}")
        return all_passed


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="V3.3 WaveNet-GRU World Model Validation")
    parser.add_argument("--model", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint")
    parser.add_argument("--both", action="store_true", help="Run both best and latest")
    parser.add_argument("--robust", action="store_true", help="Run robust validation (detects overfitting)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon for robust validation (1,4,16,64)")
    parser.add_argument("--features", type=int, choices=[13, 18, 30, 37], default=37,
                        help="Number of features: 13 (legacy), 18 (extended), 30 (all base), 37 (full+XD)")
    args = parser.parse_args()

    feat_tag = f"f{args.features}"
    prefix = f"v3_3_{feat_tag}"
    log_path = setup_logging(LOG_DIR, f"v3_3_{feat_tag}_validate")

    if args.both:
        print("\n  Running validation on BOTH checkpoints...\n")
        for name, path in [("best_ema", BASE_MODEL_DIR / f"{prefix}_wm_best_ema.pt"),
                           ("latest", BASE_MODEL_DIR / f"{prefix}_wm_latest.pt")]:
            if path.exists():
                print(f"\n{'#'*70}")
                print(f"  CHECKPOINT: {name}")
                print(f"{'#'*70}")
                v = WorldModelValidator(path, n_features=args.features)
                if args.robust:
                    v.run_robust_validation(horizon=args.horizon)
                else:
                    v.run_diagnostics()
    elif args.robust:
        model_path = Path(args.model) if args.model else None
        if args.latest and not model_path:
            model_path = BASE_MODEL_DIR / f"{prefix}_wm_latest.pt"
        v = WorldModelValidator(model_path, n_features=args.features)
        passed = v.run_robust_validation(horizon=args.horizon)
        sys.exit(0 if passed else 1)
    elif args.model:
        v = WorldModelValidator(Path(args.model), n_features=args.features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)
    elif args.latest:
        v = WorldModelValidator(BASE_MODEL_DIR / f"{prefix}_wm_latest.pt", n_features=args.features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)
    else:
        v = WorldModelValidator(n_features=args.features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
