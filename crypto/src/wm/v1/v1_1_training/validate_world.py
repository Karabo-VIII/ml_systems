"""
V1.1 World Model Validation -- Comprehensive Diagnostics Suite

Run this BEFORE proceeding to agent training.
Evaluates: reconstruction, return prediction (all horizons), latent health,
imagination quality, regime classification, dream coherence, per-asset breakdown.

Usage:
    python validate_world.py                    # Best EMA model (25 features)
    python validate_world.py --features 13      # Validate 13-feature checkpoint
    python validate_world.py --features 25      # Validate 25-feature checkpoint
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
from world_model import TransformerWorldModel, count_parameters
from log_utils import setup_logging, teardown_logging
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets
from revin import RevIN

try:
    from validation_utils import RobustValidator, ValidationConfig
    ROBUST_VALIDATION_AVAILABLE = True
except ImportError:
    ROBUST_VALIDATION_AVAILABLE = False


# ===============================================================================
# VALIDATOR
# ===============================================================================

class WorldModelValidator:
    """Comprehensive diagnostics for V1.1 Transformer-RSSM World Model."""

    def __init__(self, model_path: Path = None, use_revin: bool = False, n_features: int = 25):
        self.n_features = n_features
        feature_list, input_dim, base_dim = get_feature_config(n_features)
        self.feature_list = feature_list
        self.input_dim = input_dim
        self.model = TransformerWorldModel(input_dim=input_dim, base_dim=base_dim).to(DEVICE)
        self.use_revin = use_revin
        self.revin = RevIN(num_features=input_dim).to(DEVICE) if use_revin else None
        self.model_name = "unknown"
        self._load_model(model_path)
        self.model.eval()
        if self.revin is not None:
            self.revin.eval()

    def _load_model(self, model_path: Path):
        feat_tag = f"f{self.n_features}"
        revin_tag = "_revin" if self.use_revin else ""
        ckpt_prefix = f"v1_1_{feat_tag}{revin_tag}"
        if model_path is None:
            model_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_best_ema.pt"
        if not model_path.exists():
            model_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"
        if not model_path.exists():
            print(f"  [ERROR] No checkpoint found in {BASE_MODEL_DIR}")
            sys.exit(1)

        self.model_name = model_path.stem
        print(f"  Loading: {model_path.name}")
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            epoch = ckpt.get("epoch", "?")
            print(f"  Checkpoint from epoch {epoch}")
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and any(k.startswith("obs_encoder") for k in ckpt):
            state_dict = ckpt
        else:
            state_dict = ckpt

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  [INFO] {len(missing)} new keys (random init): {missing[:3]}{'...' if len(missing) > 3 else ''}")
        if unexpected:
            print(f"  [WARN] {len(unexpected)} unexpected keys: {unexpected[:3]}")

        # Load RevIN state (backward compatible with pre-RevIN checkpoints)
        if self.revin is not None and isinstance(ckpt, dict) and "revin_state_dict" in ckpt:
            self.revin.load_state_dict(ckpt["revin_state_dict"])

        print(f"  Parameters: {count_parameters(self.model):,}")

    # --- Data Loading ---------------------------------------------------------

    # Purge gap: must match anti_fragile.py AntifragileConfig.purge_gap_bars
    # Hurst R/S (window=200) + rolling z-score (window=200) = 400-bar cascading dependency.
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
                # Use val segment (60-80%) matching training split
                val_start = int(len(df) * 0.50) + self.PURGE_GAP_BARS
                val_end = int(len(df) * 0.70)
                if val_start >= val_end:
                    val_start = int(len(df) * 0.50)
                    val_end = int(len(df) * 0.70)
                df_val = df.slice(val_start, val_end - val_start)
            else:
                df_val = df

            feats, targets = extract_features_targets(
                df_val, self.feature_list, REWARD_HORIZONS, asset_name
            )

            # Also load raw return targets for cross-domain IC check.
            # extract_features_targets auto-detects voladj; raw targets let us
            # verify signal transfers to actual returns (not just voladj space).
            raw_targets = {}
            for h in REWARD_HORIZONS:
                raw_col = f"target_return_{h}"
                if raw_col in df_val.columns:
                    raw_targets[h] = df_val[raw_col].to_numpy().astype(np.float32)

            segments.append((feats, targets, asset_idx, asset_name, raw_targets))
            print(f"    {asset_name}: {len(feats):,} {'total' if use_full_data else 'validation'} bars")

        return segments

    # --- Train/Val Loss Computation (for overfitting gate) -------------------

    @torch.no_grad()
    def _compute_split_loss(self, segments):
        """
        Compute average total loss on a set of data segments.
        Used to compute train/val loss ratio for GATE_LOSS_RATIO_MAX.
        """
        seq_len = WM_SEQ_LEN
        all_losses = []

        for feats, targets, asset_idx, asset_name, *_extra in segments:
            indices = list(range(0, len(feats) - seq_len, seq_len))
            if not indices:
                continue

            for i in indices:
                obs_np = feats[i:i+seq_len]
                obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
                asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

                targ_tensors = {}
                for h in REWARD_HORIZONS:
                    t = targets[h][i:i+seq_len]
                    targ_tensors[h] = torch.from_numpy(t).unsqueeze(0).float().to(DEVICE)

                if self.revin is not None:
                    obs = self.revin(obs, mode='norm')
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    _, loss_dict, _ = self.model.get_loss(
                        obs, asset, targ_tensors, mask_ratio=0.0
                    )

                if "total" in loss_dict and np.isfinite(loss_dict["total"]):
                    all_losses.append(loss_dict["total"])

        return float(np.mean(all_losses)) if all_losses else 0.0

    # --- Main Diagnostics -----------------------------------------------------

    @torch.no_grad()
    def run_diagnostics(self):
        print(f"\n{'='*70}")
        print(f"  V1.1 WORLD MODEL COMPREHENSIVE VALIDATION")
        print(f"  Model: {self.model_name}")
        print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Purge Gap: {self.PURGE_GAP_BARS} bars (matches anti_fragile.py)")
        print(f"{'='*70}")
        print(f"\n  Loading validation data from {DATA_DIR}")

        segments = self.load_validation_data()
        if not segments:
            print("  [ERROR] No validation data found.")
            return False

        all_results = {}

        for feats, targets, asset_idx, asset_name, *_extra in segments:
            print(f"\n{'-'*70}")
            print(f"  {asset_name} DIAGNOSTICS")
            print(f"{'-'*70}")
            results = self._evaluate_asset(feats, targets, asset_idx)
            all_results[asset_name] = results
            self._print_asset_report(results, asset_name)

        # -- Shuffled IC (anti-memorization gate) --------------------------
        contiguous_ic = np.mean([
            np.mean([r["returns"][h]["ic"] for r in all_results.values()])
            for h in REWARD_HORIZONS
        ])

        # Global shuffle (GATED metric -- matches training ShIC from anti_fragile.py)
        print(f"\n  Computing GLOBAL shuffled IC (gated metric)...")
        global_shuffled_ic = self._compute_global_shuffled_ic(segments)
        print(f"    Contiguous IC:       {contiguous_ic:+.4f}")
        print(f"    Global Shuffled IC:  {global_shuffled_ic:+.4f}")
        if abs(contiguous_ic) > 1e-6:
            global_ratio = global_shuffled_ic / contiguous_ic
            print(f"    Global Ratio:        {global_ratio:.3f} (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")
        else:
            global_ratio = 0.0
            print(f"    Global Ratio:        N/A (contiguous IC near zero)")

        # Within-sequence shuffle (DIAGNOSTIC -- stricter test, not gated)
        print(f"\n  Computing within-sequence shuffled IC (diagnostic)...")
        within_seq_shuffled_ic = self._compute_within_seq_shuffled_ic(segments)
        print(f"    Within-Seq Shuffled IC: {within_seq_shuffled_ic:+.4f}")
        if abs(contiguous_ic) > 1e-6:
            within_ratio = within_seq_shuffled_ic / contiguous_ic
            print(f"    Within-Seq Ratio:       {within_ratio:.3f} (diagnostic, not gated)")
        else:
            print(f"    Within-Seq Ratio:       N/A (contiguous IC near zero)")

        # -- Train/Val Loss Ratio (overfitting gate) -----------------------
        print(f"\n  Computing train/val loss ratio (overfitting check)...")
        train_segments_data = self._load_train_split()
        train_loss = self._compute_split_loss(train_segments_data) if train_segments_data else 0.0
        val_loss = self._compute_split_loss(segments)
        loss_ratio = val_loss / train_loss if train_loss > 1e-8 else 0.0  # val/train: >2.0 = overfitting
        print(f"    Train Loss:    {train_loss:.4f}")
        print(f"    Val Loss:      {val_loss:.4f}")
        print(f"    Ratio:         {loss_ratio:.3f} (gate: < {GATE_LOSS_RATIO_MAX})")

        # -- Raw Return IC (vol-shortcut diagnostic) -----------------------
        # Correlates voladj predictions with raw return targets (no re-inference).
        # If raw IC << voladj IC, the model is predicting vol, not returns.
        print(f"\n  RAW RETURN IC (cross-domain diagnostic)")
        has_raw = any(len(seg) > 4 and seg[4] for seg in segments)
        if has_raw:
            for h in REWARD_HORIZONS:
                raw_preds_all, raw_reals_all = [], []
                for feats, targets, asset_idx, asset_name, *_extra in segments:
                    raw_tgts = _extra[0] if _extra else {}
                    if h not in raw_tgts:
                        continue
                    raw_h = raw_tgts[h]
                    # Reuse already-computed predictions (stored in _aggregate_results)
                    voladj_preds = all_results[asset_name]["returns"][h].get("_preds")
                    if voladj_preds is None:
                        continue
                    # voladj_preds are flattened contiguous windows; raw_h is full val array.
                    # Align: predictions cover indices [0:seq, seq:2*seq, ...] of val data.
                    seq_len = WM_SEQ_LEN
                    indices = list(range(0, len(feats) - seq_len, seq_len))
                    n_pred_bars = len(indices) * seq_len
                    if n_pred_bars != len(voladj_preds):
                        n_pred_bars = min(n_pred_bars, len(voladj_preds))
                    raw_aligned = np.concatenate([raw_h[i:i+seq_len] for i in indices])[:n_pred_bars]
                    raw_preds_all.extend(voladj_preds[:n_pred_bars])
                    raw_reals_all.extend(raw_aligned)

                p = np.array(raw_preds_all)
                r_arr = np.array(raw_reals_all)
                mask = np.isfinite(p) & np.isfinite(r_arr)
                if mask.sum() > 50:
                    raw_ic = float(np.corrcoef(p[mask], r_arr[mask])[0, 1])
                    raw_rank_ic = float(scipy_stats.spearmanr(p[mask], r_arr[mask]).statistic)
                    voladj_ic = np.mean([all_results[a]["returns"][h]["ic"] for a in all_results])
                    print(f"    t+{h:<3} Raw IC:{raw_ic:+.4f}  Raw RankIC:{raw_rank_ic:+.4f}  "
                          f"(voladj IC:{voladj_ic:+.4f})")
                else:
                    print(f"    t+{h:<3} Insufficient data for raw IC")
        else:
            print(f"    [SKIP] No raw return targets in chimera (pre-V51 data?)")

        # -- Aggregate Report ----------------------------------------------
        self._print_aggregate_report(all_results)

        # Use h=1 IC for ShIC ratio (not mean across all horizons)
        # ShIC is computed on h=1 only, so denominator must match
        ic_h1 = np.mean([r["returns"][1]["ic"] for r in all_results.values()])
        gate_pass = self._check_gates(
            all_results, shuffled_ic=global_shuffled_ic, loss_ratio=loss_ratio,
            ic_h1_for_shic=ic_h1
        )

        # -- Save results --------------------------------------------------
        self._save_results(all_results, gate_pass)

        return gate_pass

    def _load_train_split(self):
        """Load training split (first 90%) for loss ratio computation.
        Uses the same 90% cutoff as training -- no purge gap needed here
        since we're measuring train loss on the actual training data."""
        files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
        segments = []

        for f in files:
            asset_name = f.stem.split("_")[0].upper()
            if asset_name not in ASSET_TO_IDX:
                continue

            asset_idx = ASSET_TO_IDX[asset_name]
            df = pl.read_parquet(f)
            df = selective_drop_nulls(df, self.feature_list, REWARD_HORIZONS, asset_name)

            split = int(len(df) * 0.50)  # Train segment only (60%)
            df_train = df.slice(0, split)

            feats, targets = extract_features_targets(
                df_train, self.feature_list, REWARD_HORIZONS, asset_name
            )

            segments.append((feats, targets, asset_idx, asset_name))

        return segments

    # --- Per-Asset Evaluation -------------------------------------------------

    def _evaluate_asset(self, feats, targets, asset_idx):
        seq_len = WM_SEQ_LEN
        results = {
            "reconstruction": {"total_mse": [], "per_feature_mse": [[] for _ in range(self.model.base_dim)],
                                "per_feature_r2": [[] for _ in range(self.model.base_dim)], "cosine_sim": []},
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

            # Prepare targets for this window
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

            # -- 1. Reconstruction -----------------------------------------
            # Decoder only reconstructs base_dim features (first 13)
            obs_base = obs[:, :, :self.model.base_dim]
            rec_mse = F.mse_loss(recon, obs_base).item()
            results["reconstruction"]["total_mse"].append(rec_mse)

            cos_sim = F.cosine_similarity(recon.reshape(-1), obs_base.reshape(-1), dim=0).item()
            results["reconstruction"]["cosine_sim"].append(cos_sim)

            for fi in range(self.model.base_dim):
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

            # Utilization: fraction of categories with prob > uniform + epsilon
            uniform = 1.0 / RSSM_CLASSES
            active = (post_probs.mean(0) > uniform * 0.5).float().mean().item()
            results["latent"]["utilization"].append(active)

            # Prior-posterior agreement
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
                imag_mse = F.mse_loss(imag_out["recon"], obs[:, :, :self.model.base_dim]).item()
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

            # -- 6. Dream Coherence (multi-step) --------------------------
            if i + seq_len + 5 < len(feats):
                h_seq = outputs["h_seq"]
                z_post = outputs["z_post"]
                h_last = h_seq[:, -1, :]
                z_last = z_post[:, -1, :]

                gru_hidden = None
                for step in range(1, 6):
                    h_last, z_next, gru_hidden, dream_rets = self.model.dream_step(h_last, z_last, gru_hidden)
                    z_last = z_next

                    future_idx = i + seq_len + step - 1
                    if future_idx < len(targets[1]):
                        for h in [1]:
                            results["dream"]["ic_by_step"][step]["preds"].append(
                                dream_rets[h].cpu().numpy().flatten()[0]
                            )
                            results["dream"]["ic_by_step"][step]["reals"].append(
                                targets[h][future_idx]
                            )

        # -- Compute aggregate metrics -------------------------------------
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

                # Top/bottom decile analysis
                sorted_idx = np.argsort(p)
                decile = max(1, len(p) // 10)
                top_real = float(np.mean(r[sorted_idx[-decile:]]))
                bot_real = float(np.mean(r[sorted_idx[:decile]]))
                spread = top_real - bot_real

                # IC p-value (t-test approximation)
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
                "_preds": p, "_reals": r,  # kept for raw-return IC (no re-inference)
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
        """Bootstrap 95% confidence interval for IC."""
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
        # Reconstruction (base_dim features only — decoder doesn't reconstruct XD)
        base_features = self.feature_list[:self.model.base_dim]
        print(f"\n  RECONSTRUCTION (base {self.model.base_dim} features)")
        print(f"    Total MSE:       {r['rec_mse']:.6f}  {'PASS' if r['rec_mse'] < GATE_REC_MSE_MAX else 'FAIL'}")
        print(f"    Cosine Sim:      {r['rec_cosine']:.4f}")
        print(f"    Per-Feature Breakdown:")
        for i, feat_name in enumerate(base_features):
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
            print(f"\n  DREAM COHERENCE (multi-step prior-only rollout)")
            for step, ic_val in r["dream_ic"].items():
                bar = "#" * max(0, int(abs(ic_val) * 50))
                print(f"    Step {step}: IC = {ic_val:+.4f}  [{bar}]")

    def _print_aggregate_report(self, all_results):
        print(f"\n{'='*70}")
        print(f"  AGGREGATE RESULTS ACROSS ALL ASSETS")
        print(f"{'='*70}")

        assets = list(all_results.keys())
        n = len(assets)

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
    def _compute_global_shuffled_ic(self, segments, n_seeds=5, batch_size=64):
        """
        Compute IC on globally-shuffled data (matches anti_fragile.py training metric).

        Shuffles ALL bars for each asset, then creates seq_len windows from the
        shuffled data. This tests whether the model learned feature->return signal
        (preserved under global shuffle) vs temporal patterns (destroyed).

        This is the GATED metric -- it matches the ShIC used during training.
        """
        all_shuffled_ics = []

        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset * 1000
            rng = np.random.default_rng(seed)
            all_preds, all_reals = [], []

            for feats, targets, asset_idx, asset_name, *_extra in segments:
                n = len(feats)
                seq_len = WM_SEQ_LEN
                if n < seq_len * 2:
                    continue

                # Globally shuffle all bar indices for this asset
                indices = np.arange(n)
                rng.shuffle(indices)

                shuffled_feats = feats[indices]
                shuffled_targets_1 = targets[1][indices]

                # Create non-overlapping seq_len windows from shuffled data
                window_starts = list(range(0, n - seq_len, seq_len))

                for batch_start in range(0, len(window_starts), batch_size):
                    batch_ws = window_starts[batch_start:batch_start + batch_size]
                    obs_list = []
                    real_list = []

                    for i in batch_ws:
                        obs_list.append(shuffled_feats[i:i+seq_len])
                        real_list.append(shuffled_targets_1[i:i+seq_len])

                    obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
                    asset = torch.full(
                        (len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE
                    )

                    if self.revin is not None:
                        obs = self.revin(obs, mode='norm')
                    with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                        outputs = self.model.forward_train(obs, asset)

                    logits = outputs["return_logits"][1]
                    preds = self.model.bucketer.decode(logits).cpu().numpy()

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

    @torch.no_grad()
    def _compute_within_seq_shuffled_ic(self, segments, n_seeds=5, batch_size=64):
        """
        Compute IC on within-sequence temporally-shuffled data (DIAGNOSTIC only).

        Shuffles the temporal order WITHIN each 96-bar window. This is a stricter
        test than global shuffle -- it tests whether the model needs temporal ordering
        within a window (expected: yes, so this IC should be near zero).

        NOT gated -- used as a diagnostic to understand model behavior.
        """
        all_shuffled_ics = []

        for seed_offset in range(n_seeds):
            seed = 42 + seed_offset * 1000
            rng = np.random.default_rng(seed)
            all_preds, all_reals = [], []

            for feats, targets, asset_idx, asset_name, *_extra in segments:
                seq_len = WM_SEQ_LEN
                # Use non-overlapping windows for efficiency
                indices = list(range(0, len(feats) - seq_len, seq_len))
                if not indices:
                    continue

                # Batch processing for GPU efficiency
                for batch_start in range(0, len(indices), batch_size):
                    batch_indices = indices[batch_start:batch_start + batch_size]
                    obs_list = []
                    real_list = []

                    for i in batch_indices:
                        obs_np = feats[i:i+seq_len].copy()
                        perm = rng.permutation(seq_len)
                        obs_list.append(obs_np[perm])
                        real_list.append(targets[1][i:i+seq_len])

                    obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
                    asset = torch.full(
                        (len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE
                    )

                    if self.revin is not None:
                        obs = self.revin(obs, mode='norm')
                    with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                        outputs = self.model.forward_train(obs, asset)

                    logits = outputs["return_logits"][1]
                    preds = self.model.bucketer.decode(logits).cpu().numpy()

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
        # Use h=1 IC as denominator since ShIC is computed on h=1 only.
        # Using mean IC across all horizons inflates the ratio when h16/h64
        # have memorized temporal patterns (IC_h1 >> mean_all).
        if shuffled_ic is not None:
            shic_denom = ic_h1_for_shic if ic_h1_for_shic is not None else mean_ic
            if abs(shic_denom) > 1e-6:
                shuffled_ratio = shuffled_ic / shic_denom
                gates["Shuffled IC Ratio (h1)"] = (
                    shuffled_ratio > GATE_SHUFFLED_IC_RATIO_MIN,
                    f"{shuffled_ratio:.3f} > {GATE_SHUFFLED_IC_RATIO_MIN} "
                    f"(ShIC={shuffled_ic:.4f} / IC_h1={shic_denom:.4f})"
                )
            else:
                gates["Shuffled IC Ratio (h1)"] = (False, "h1 IC near zero")

        # Overfitting gate: val_loss/train_loss > 2.0 means val loss is much higher
        # than train loss, indicating the model has memorized training patterns
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
        """Save validation results to JSON for tracking."""
        output = {
            "version": "v1.1",
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
        print(f"  V1.1 TRANSFORMER-RSSM ROBUST VALIDATION (Overfitting Detection)")
        print(f"  Model: {self.model_name}")
        print(f"  Horizon: t+{horizon}")
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
            "version": "v1.1", "model": self.model_name,
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
    log_path = setup_logging(LOG_DIR, "v1_1_validate")
    parser = argparse.ArgumentParser(description="V1.1 World Model Comprehensive Validation")
    parser.add_argument("--model", type=str, default=None, help="Path to specific checkpoint")
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint")
    parser.add_argument("--both", action="store_true", help="Run both best and latest")
    parser.add_argument("--robust", action="store_true", help="Run robust validation (detects overfitting)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon for robust validation (1,4,16,64)")
    parser.add_argument("--revin", action="store_true", help="Enable RevIN (validate revin checkpoint; off by default)")
    parser.add_argument("--features", type=int, choices=[13, 17, 18, 20, 22, 25], default=25,
                        help="Feature count: 13/17/18/20/22/25 (default: 25 = 20 base + 5 XD)")
    args = parser.parse_args()

    use_revin = args.revin
    n_features = args.features
    feat_tag = f"f{n_features}"
    revin_tag = "_revin" if use_revin else ""
    ckpt_prefix = f"v1_1_{feat_tag}{revin_tag}"

    if args.both:
        print("\n  Running validation on BOTH checkpoints...\n")
        for name, path in [("best_ema", BASE_MODEL_DIR / f"{ckpt_prefix}_wm_best_ema.pt"),
                           ("latest", BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt")]:
            if path.exists():
                print(f"\n{'#'*70}")
                print(f"  CHECKPOINT: {name} ({n_features}f, {'RevIN' if use_revin else 'no-RevIN'})")
                print(f"{'#'*70}")
                v = WorldModelValidator(path, use_revin=use_revin, n_features=n_features)
                if args.robust:
                    v.run_robust_validation(horizon=args.horizon)
                else:
                    v.run_diagnostics()
    elif args.robust:
        model_path = Path(args.model) if args.model else None
        if args.latest and not model_path:
            model_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"
        v = WorldModelValidator(model_path, use_revin=use_revin, n_features=n_features)
        passed = v.run_robust_validation(horizon=args.horizon)
        sys.exit(0 if passed else 1)
    elif args.model:
        v = WorldModelValidator(Path(args.model), use_revin=use_revin, n_features=n_features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)
    elif args.latest:
        model_path = BASE_MODEL_DIR / f"{ckpt_prefix}_wm_latest.pt"
        v = WorldModelValidator(model_path, use_revin=use_revin, n_features=n_features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)
    else:
        v = WorldModelValidator(use_revin=use_revin, n_features=n_features)
        passed = v.run_diagnostics()
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
