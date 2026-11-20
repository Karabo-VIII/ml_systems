#!/usr/bin/env python
"""Architectural-Ceiling Probe — synthetic signal recovery + ShIC degradation test.

Compensates for the inability to run real training in-session: empirically
probes each architecture's REACH on a synthetic task with KNOWN signal
strength, in 30-90 seconds per model. Outputs: (a) max recoverable IC,
(b) ShIC degradation under shuffled-features condition, (c) anti-fragile
ratio (ShIC/IC).

DESIGN:
  - Generate synthetic [B*epochs, T, F] features from real-data summary stats
    (mean, std, autocorr) so distribution roughly matches crypto regime.
  - Plant a predictive signal: target[t] = α · Σⱼ wⱼ · feature[t-lag, j] + ε
    where (w, lag) are random sparse weights and α controls SNR.
  - Train architecture for `n_steps` (300-500 default).
  - Track IC trajectory at h=1.
  - ShIC: re-run with features time-shuffled per sample → if architecture is
    relying on temporal patterns rather than instantaneous features, IC drops.
  - Anti-fragile filter: ShIC/IC > 0.3 (CLAUDE.md mandate) at α-target.

CALIBRATION INSIGHT:
  If architecture A recovers γ × planted signal at SNR S in 300 steps, it can
  recover ~γ × real signal at similar SNR with proportionally more steps. The
  key signal is RELATIVE recovery between architectures, not absolute IC.

LIMITATIONS (state honestly):
  - Synthetic data has no regime shifts, no fat tails, no cross-asset structure
  - Real data is harder; this gives an UPPER BOUND on real-data performance
  - Architectures that fail synthetic probe DEFINITELY fail real-data probe
  - Architectures that pass synthetic probe LIKELY (not guaranteed) hit target
  - Necessary, not sufficient. Useful as a FILTER.

USAGE:
  python scripts/probe_architectural_ceiling.py --target-ic 0.10 --n-steps 400
  python scripts/probe_architectural_ceiling.py --models v4,v22,v25
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Architecture registry — paths to (settings, world_model, model_class_name)
# =============================================================================

ARCHITECTURE_REGISTRY = {
    # ROUND-5 baselines (post-VIB additions)
    "v4":  ("src/wm/v4/v4_training",   "MambaWorldModel"),
    "v13": ("src/wm/v13/v13_training", "TFTWorldModel"),
    "v22": ("src/wm/v22/v22_training", "iTransformerWorldModel"),
    "v23": ("src/wm/v23/v23_training", "xLSTMWorldModel"),
    "v24": ("src/wm/v24/v24_training", "TimesNetWorldModel"),
    # ROUND-6 frontier
    "v25": ("src/wm/v25/v25_training", "V25FrontierWorldModel"),
    # Reference baselines (NOT modified in rounds 5-6 — anchor for comparison)
    "v1_1": ("src/wm/v1/v1_1_training", "TransformerWorldModel"),
}


def load_architecture(version: str):
    """Load (settings_module, model_class) for a given V-version."""
    if version not in ARCHITECTURE_REGISTRY:
        raise ValueError(f"Unknown version {version}; available: {list(ARCHITECTURE_REGISTRY)}")
    rel_path, cls_name = ARCHITECTURE_REGISTRY[version]
    abs_path = PROJECT_ROOT / rel_path
    # Clear conflicting modules
    for mod in ("settings", "world_model", "components"):
        sys.modules.pop(mod, None)
    # Need V1.x components on path for the imports
    v1_path = str(PROJECT_ROOT / "src" / "wm" / "v1" / "v1_0_training")
    if v1_path not in sys.path:
        sys.path.insert(0, v1_path)
    if str(abs_path) not in sys.path:
        sys.path.insert(0, str(abs_path))
    import settings as st
    import world_model as wm
    return st, getattr(wm, cls_name)


# =============================================================================
# Synthetic data generator — crypto-flavored stats
# =============================================================================

@dataclass
class SyntheticConfig:
    n_assets: int = 10
    seq_len: int = 96
    n_features: int = 29
    target_horizons: tuple = (1, 4, 16, 64)
    signal_lag: int = 1                # Predictive lag (target[t] depends on x[t-lag])
    signal_sparsity: int = 5           # Number of features that carry signal
    feature_autocorr: float = 0.7      # AR(1) coefficient on feature dynamics
    noise_std: float = 0.5             # Residual noise std (added after signal)
    fat_tail_kurt: float = 5.0         # Approximate target kurtosis (Student-t)
    seed: int = 42
    # Round-8: semi-synthetic — use real-data feature stats if loaded
    real_feature_mean: np.ndarray = None    # [F] per-feature mean from real data
    real_feature_std: np.ndarray = None     # [F] per-feature std
    real_feature_ar1: np.ndarray = None     # [F] per-feature AR(1) coefficient
    use_real_stats: bool = False


def load_real_feature_stats(feature_list: list, max_rows: int = 20000,
                              n_assets: int = 3) -> dict:
    """Load per-feature mean/std/AR(1) from chimera_legacy/dollar real data.

    Returns dict with arrays (mean, std, ar1) of shape [F]. Used to make
    synthetic data REALISTIC by sampling from the same distribution as real
    crypto features.
    """
    data_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    if not data_dir.exists():
        return None
    try:
        import polars as pl
    except Exception:
        return None
    sym_candidates = ["btcusdt", "ethusdt", "solusdt"]
    all_features = []
    for sym in sym_candidates[:n_assets]:
        cands = sorted(data_dir.glob(f"{sym}_v50_chimera_*.parquet"))
        if not cands:
            continue
        df = pl.read_parquet(cands[-1]).tail(max_rows)
        avail = [c for c in feature_list if c in df.columns]
        if not avail:
            continue
        arr = df.select(avail).to_numpy().astype(np.float32)
        # Drop NaN rows
        arr = arr[np.isfinite(arr).all(axis=1)]
        all_features.append(arr)
    if not all_features:
        return None
    feats = np.concatenate(all_features, axis=0)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0) + 1e-6
    # AR(1) coefficient per feature: corr(x[t], x[t-1]) on standardized
    feats_std = (feats - mean) / std
    ar1 = np.array([
        float(np.corrcoef(feats_std[:-1, i], feats_std[1:, i])[0, 1])
        for i in range(feats_std.shape[1])
    ], dtype=np.float32)
    ar1 = np.nan_to_num(ar1, nan=0.5).clip(0.0, 0.99)
    return {"mean": mean, "std": std, "ar1": ar1, "n_rows": len(feats)}


def make_signal_weights(F_dim: int, sparsity: int, seed: int) -> np.ndarray:
    """Deterministic signal weights — must be CONSISTENT across all batches.

    The probe planted a signal that needs to be the SAME for the model to
    learn across batches. Use a fixed seed independent of per-batch seeds.
    """
    rng = np.random.default_rng(seed * 31 + 7919)  # Distinct from batch seeds
    w = np.zeros(F_dim)
    signal_idx = rng.choice(F_dim, size=sparsity, replace=False)
    w[signal_idx] = rng.standard_normal(sparsity)
    w /= np.linalg.norm(w) + 1e-6
    return w


def generate_synthetic_batch(B: int, T: int, F_dim: int, signal_strength: float,
                              cfg: SyntheticConfig, device: torch.device,
                              shuffled: bool = False,
                              signal_weights: np.ndarray | None = None,
                              batch_seed: int | None = None) -> tuple[torch.Tensor, dict]:
    """Generate [B, T, F] features + target dict with planted signal of strength α.

    Signal model: y[t] = α · w · x[t-lag] + ε
    where w is FIXED across batches (passed in as signal_weights) and ε is
    fat-tailed.

    If signal_weights is None, derived deterministically from cfg.seed —
    this means the SAME signal pattern is planted in every batch, allowing
    the model to actually learn it.

    If shuffled=True: time-shuffle features per sample (breaks temporal signal).
    """
    if signal_weights is None:
        signal_weights = make_signal_weights(F_dim, cfg.signal_sparsity, cfg.seed)

    # Per-batch RNG for the FEATURES (different each call to give learning data)
    bs = batch_seed if batch_seed is not None else cfg.seed
    rng = np.random.default_rng(bs)

    # AR(1) features. Use REAL per-feature AR(1) coefficients if available
    # (semi-synthetic mode); otherwise scalar default.
    eps_feat = rng.standard_t(df=4, size=(B, T, F_dim)) * 0.3
    x = np.zeros((B, T, F_dim))
    x[:, 0] = eps_feat[:, 0]
    if cfg.use_real_stats and cfg.real_feature_ar1 is not None:
        ar1 = cfg.real_feature_ar1[:F_dim]    # per-feature AR(1) coeff
        innov_scale = np.sqrt(1 - ar1**2)     # per-feature
        for t in range(1, T):
            x[:, t] = ar1 * x[:, t - 1] + innov_scale * eps_feat[:, t]
    else:
        # Scalar AR(1) (legacy path)
        for t in range(1, T):
            x[:, t] = cfg.feature_autocorr * x[:, t - 1] + math.sqrt(1 - cfg.feature_autocorr**2) * eps_feat[:, t]
    # Standardize per-feature
    x = (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1e-6)
    # Round-8: shift+scale to real-data per-feature mean/std (semi-synthetic)
    if cfg.use_real_stats and cfg.real_feature_mean is not None and cfg.real_feature_std is not None:
        x = x * cfg.real_feature_std[:F_dim] + cfg.real_feature_mean[:F_dim]
        # Re-standardize globally to keep optimization stable
        x = (x - x.mean(axis=(0, 1), keepdims=True)) / (x.std(axis=(0, 1), keepdims=True) + 1e-6)

    # Targets per horizon: target[t] uses x[t-lag]
    targets = {}
    lag = cfg.signal_lag
    for h in cfg.target_horizons:
        signal = np.einsum("bti,i->bt", x, signal_weights)        # [B, T]
        noise = rng.standard_t(df=4, size=(B, T)) * cfg.noise_std
        if lag > 0:
            signal = np.pad(signal, ((0, 0), (lag, 0)), mode="constant")[:, :-lag]
        h_decay = math.exp(-h / 32.0)
        # Scale signal to have meaningful magnitude in [-1, 1] TwoHot bin range
        # signal_strength=0.15 -> targets in roughly +-0.15 range (19 bins)
        y = signal_strength * h_decay * signal + noise * 0.05
        targets[h] = torch.tensor(y, dtype=torch.float32, device=device)

    if shuffled:
        # Shuffle features ALONG TIME AXIS per sample — breaks temporal patterns
        # Note: this shuffles AFTER target was computed, so target-feature
        # relationship at t is broken. ShIC measures whether model relies on
        # temporal patterns.
        for b in range(B):
            perm = rng.permutation(T)
            x[b] = x[b, perm]

    obs = torch.tensor(x, dtype=torch.float32, device=device)
    return obs, targets


# =============================================================================
# Probe — train architecture for n_steps, measure IC trajectory
# =============================================================================

@dataclass
class ProbeResult:
    version: str
    target_alpha: float
    n_steps: int
    final_ic_h1: float = 0.0
    best_ic_h1: float = 0.0
    final_shic_h1: float = 0.0
    shic_ratio: float = 0.0           # final_shic / final_ic
    can_hit_target: bool = False      # final_ic >= 0.5 * target_alpha (50% recovery)
    anti_fragile: bool = False        # shic_ratio >= 0.3 (CLAUDE.md mandate)
    ic_trajectory: list = field(default_factory=list)
    elapsed_s: float = 0.0
    final_loss: float = 0.0
    failure_mode: str = ""
    n_params: int = 0
    # Per-asset IC breakdown (info-max axis 3)
    per_asset_ic: dict = field(default_factory=dict)
    ic_std_across_assets: float = 0.0  # higher = regime-dependent / asset-specific learning


def compute_ic(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Spearman-style IC: correlation between predicted and actual return."""
    p = predicted.flatten().detach().cpu().numpy()
    t = target.flatten().detach().cpu().numpy()
    mask = np.isfinite(p) & np.isfinite(t)
    if mask.sum() < 50:
        return 0.0
    pc = (p[mask] - p[mask].mean()) / (p[mask].std() + 1e-8)
    tc = (t[mask] - t[mask].mean()) / (t[mask].std() + 1e-8)
    return float(np.mean(pc * tc))


def probe_architecture(version: str, target_alpha: float, n_steps: int = 400,
                       cfg: SyntheticConfig = None, device: str = "cuda") -> ProbeResult:
    """Run architectural-ceiling probe on one version.

    Procedure:
      1. Load architecture
      2. Generate synthetic batch with signal at strength α
      3. Train for n_steps using AdamW + AMP
      4. Compute IC at h=1 every 50 steps
      5. Generate SHUFFLED batch (time-shuffled features); compute IC on it
         using the same trained model. If model relied on temporal patterns,
         shuffled IC drops sharply.
      6. Return ProbeResult with trajectory, can_hit_target, anti_fragile flags
    """
    cfg = cfg or SyntheticConfig()
    result = ProbeResult(version=version, target_alpha=target_alpha, n_steps=n_steps)
    t0 = time.time()
    try:
        st, ModelCls = load_architecture(version)
    except Exception as e:
        result.failure_mode = f"load: {str(e)[:80]}"
        return result

    # Try multiple constructor signatures (V1.x needs base_dim too;
    # V4/V13/V22-V25 accept input_dim only; V0/V19 may have different shape).
    # Derive base_dim from the feature config (V1.x XD-split: base = total - 7 XD).
    model = None
    try:
        # get_feature_config returns (feature_list, input_dim, base_dim) per CLAUDE.md
        _, _, base_dim_resolved = st.get_feature_config(cfg.n_features)
    except Exception:
        base_dim_resolved = getattr(st, "BASE_DIM", cfg.n_features)
    init_attempts = [
        lambda: ModelCls(input_dim=cfg.n_features, base_dim=base_dim_resolved),
        lambda: ModelCls(input_dim=cfg.n_features),
        lambda: ModelCls(),
    ]
    init_err = None
    for attempt in init_attempts:
        try:
            model = attempt().to(device)
            break
        except TypeError as e:
            init_err = str(e)
            continue
        except Exception as e:
            init_err = f"{type(e).__name__}: {str(e)}"
            continue
    if model is None:
        result.failure_mode = f"init: {init_err[:80]}"
        return result

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result.n_params = n_params
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    B = 32
    asset = torch.randint(0, cfg.n_assets, (B,), device=device)

    # CRITICAL: signal weights are FIXED across all batches (train + val).
    # Without this, each batch has different "signal" and the model can't learn.
    signal_weights = make_signal_weights(cfg.n_features, cfg.signal_sparsity, cfg.seed)

    # Pre-generate validation set (held out from training)
    val_obs, val_tgt = generate_synthetic_batch(
        B, cfg.seq_len, cfg.n_features, target_alpha, cfg, device,
        shuffled=False, signal_weights=signal_weights, batch_seed=cfg.seed + 99999
    )
    val_asset = torch.randint(0, cfg.n_assets, (B,), device=device)

    # Training loop
    model.train()
    for step in range(n_steps):
        # Fresh batch each step but SAME signal_weights (consistent task)
        obs, tgt = generate_synthetic_batch(
            B, cfg.seq_len, cfg.n_features, target_alpha, cfg, device,
            shuffled=False, signal_weights=signal_weights,
            batch_seed=cfg.seed + step + 1000
        )
        try:
            with torch.amp.autocast("cuda"):
                # Try standard get_loss signature; fall back if model needs special kwargs
                try:
                    loss, ld, _ = model.get_loss(obs, asset, tgt, mask_ratio=0.15)
                except TypeError:
                    # V4 needs temporal_ctx_drop kwarg
                    loss, ld, _ = model.get_loss(obs, asset, tgt, mask_ratio=0.15,
                                                  temporal_ctx_drop=0.15)
            if not torch.isfinite(loss).item():
                result.failure_mode = f"NaN/Inf at step {step}"
                break
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        except Exception as e:
            result.failure_mode = f"train step {step}: {str(e)[:80]}"
            break

        # IC checkpoint every 50 steps on validation set
        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                try:
                    val_out = model.forward_train(val_obs, val_asset)
                    pred_logits = val_out["return_logits"][1]
                    nb = getattr(model, "_num_bins", None) or pred_logits.shape[-1]
                    pred = model.bucketer.decode(pred_logits.reshape(-1, nb))
                    ic = compute_ic(pred, val_tgt[1])
                    result.ic_trajectory.append((step + 1, ic))
                    if ic > result.best_ic_h1:
                        result.best_ic_h1 = ic
                except Exception as e:
                    pass
            model.train()

    # Final IC + ShIC
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        try:
            # Final IC on clean validation
            out_clean = model.forward_train(val_obs, val_asset)
            nb = getattr(model, "_num_bins", None) or out_clean["return_logits"][1].shape[-1]
            pred_clean = model.bucketer.decode(
                out_clean["return_logits"][1].reshape(-1, nb)
            )
            result.final_ic_h1 = compute_ic(pred_clean, val_tgt[1])

            # Per-asset IC breakdown (info-max: detect regime/asset-specific learning)
            pred_2d = pred_clean.reshape(B, cfg.seq_len)
            tgt_2d = val_tgt[1]
            asset_cpu = val_asset.detach().cpu().numpy()
            ic_per_asset = []
            for a in range(cfg.n_assets):
                mask = (asset_cpu == a)
                if mask.sum() > 0:
                    p = pred_2d[mask].flatten()
                    t = tgt_2d[mask].flatten()
                    ic_a = compute_ic(p, t)
                    result.per_asset_ic[a] = ic_a
                    ic_per_asset.append(ic_a)
            if len(ic_per_asset) > 1:
                result.ic_std_across_assets = float(np.std(ic_per_asset))

            # ShIC: shuffle features along time axis per sample.
            # Use same signal_weights so target structure is consistent.
            sh_obs, sh_tgt = generate_synthetic_batch(
                B, cfg.seq_len, cfg.n_features, target_alpha, cfg, device,
                shuffled=True, signal_weights=signal_weights,
                batch_seed=cfg.seed + 88888
            )
            out_sh = model.forward_train(sh_obs, val_asset)
            pred_sh = model.bucketer.decode(
                out_sh["return_logits"][1].reshape(-1, nb)
            )
            result.final_shic_h1 = compute_ic(pred_sh, sh_tgt[1])
            if abs(result.final_ic_h1) > 1e-6:
                result.shic_ratio = result.final_shic_h1 / result.final_ic_h1
            result.can_hit_target = result.final_ic_h1 >= 0.5 * target_alpha
            result.anti_fragile = result.shic_ratio >= 0.3
        except Exception as e:
            result.failure_mode = f"final eval: {str(e)[:80]}"

    result.elapsed_s = time.time() - t0
    return result


def format_result_row(r: ProbeResult) -> str:
    flags = []
    if r.can_hit_target:
        flags.append("HIT")
    if r.anti_fragile:
        flags.append("AF")
    if r.failure_mode:
        flags.append(f"FAIL:{r.failure_mode[:30]}")
    flag_str = " ".join(flags) or "-"
    return (f"  {r.version:<6} {r.n_params:>10,} {r.best_ic_h1:>+7.4f} "
            f"{r.final_ic_h1:>+7.4f} {r.final_shic_h1:>+7.4f} "
            f"{r.shic_ratio:>+7.3f} {r.ic_std_across_assets:>+7.4f} "
            f"{r.elapsed_s:>5.1f}s  {flag_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-ic", type=float, default=0.15,
                        help="Synthetic signal strength alpha (~ target IC)")
    parser.add_argument("--alpha-curve", type=str, default=None,
                        help="Sweep alpha values, e.g. '0.05,0.10,0.15,0.20,0.30'. "
                             "When set, --target-ic is ignored.")
    parser.add_argument("--n-steps", type=int, default=300)
    parser.add_argument("--models", type=str, default="v1_1,v4,v13,v22,v23,v24,v25",
                        help="Comma-separated list of versions to probe")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Multi-seed averaging (variance estimate)")
    parser.add_argument("--use-real-stats", action="store_true",
                        help="Semi-synthetic: load per-feature mean/std/AR(1) from "
                             "chimera_legacy/dollar so synthetic data has real-crypto "
                             "noise distribution. Signal still planted (controllable).")
    parser.add_argument("--feature-list-name", type=str, default="FEATURE_LIST_29",
                        help="Feature list to derive real stats from")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available; falling back to CPU (slower)")
        args.device = "cpu"

    versions = [v.strip() for v in args.models.split(",") if v.strip()]

    # Build alpha sweep list
    if args.alpha_curve:
        alphas = [float(x.strip()) for x in args.alpha_curve.split(",")]
    else:
        alphas = [args.target_ic]

    seeds = [args.seed + i for i in range(args.n_seeds)]

    # Round-8: semi-synthetic — load real-data feature stats once
    real_stats = None
    if args.use_real_stats:
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from feature_sets import (
                FEATURE_LIST_13, FEATURE_LIST_18, FEATURE_LIST_25, FEATURE_LIST_29,
                FEATURE_LIST_30, FEATURE_LIST_34, FEATURE_LIST_37, FEATURE_LIST_41,
            )
            feat_lists = {"FEATURE_LIST_13": FEATURE_LIST_13, "FEATURE_LIST_18": FEATURE_LIST_18,
                          "FEATURE_LIST_25": FEATURE_LIST_25, "FEATURE_LIST_29": FEATURE_LIST_29,
                          "FEATURE_LIST_30": FEATURE_LIST_30, "FEATURE_LIST_34": FEATURE_LIST_34,
                          "FEATURE_LIST_37": FEATURE_LIST_37, "FEATURE_LIST_41": FEATURE_LIST_41}
            feat_list = feat_lists.get(args.feature_list_name, FEATURE_LIST_29)
            real_stats = load_real_feature_stats(feat_list)
            if real_stats:
                print(f"  [Semi-synthetic] Loaded real-data stats from {real_stats['n_rows']} rows "
                      f"(mean range {real_stats['mean'].min():.3f}..{real_stats['mean'].max():.3f}, "
                      f"AR(1) range {real_stats['ar1'].min():.3f}..{real_stats['ar1'].max():.3f})")
            else:
                print("  [WARN] real-stats requested but data load failed; using pure synthetic")
        except Exception as e:
            print(f"  [WARN] real-stats load error: {str(e)[:80]}; using pure synthetic")

    print("=" * 100)
    print(f"  ARCHITECTURAL CEILING PROBE -- synthetic signal recovery")
    print(f"  Alphas: {alphas}  Seeds: {seeds}  Steps: {args.n_steps}  Device: {args.device}")
    print(f"  Anti-fragile gate: ShIC/IC >= 0.3 (CLAUDE.md)")
    print(f"  Versions: {versions}")
    print("=" * 100)
    print(f"  {'Ver':<6} {'alpha':>5} {'seed':>4} {'Params':>10} {'BestIC':>8} {'FinIC':>8} {'ShIC':>8} {'ShIC/IC':>8} {'IC_std':>8} {'Time':>6}  Flags")
    print("-" * 100)

    all_results = []
    for v in versions:
        for alpha in alphas:
            for seed in seeds:
                if args.device == "cuda":
                    torch.cuda.empty_cache()
                torch.manual_seed(seed)
                cfg = SyntheticConfig(seed=seed)
                if real_stats is not None:
                    cfg.real_feature_mean = real_stats["mean"]
                    cfg.real_feature_std = real_stats["std"]
                    cfg.real_feature_ar1 = real_stats["ar1"]
                    cfg.use_real_stats = True
                r = probe_architecture(v, alpha, n_steps=args.n_steps,
                                        cfg=cfg, device=args.device)
                all_results.append(r)
                # Add alpha+seed to the row prefix
                row = (f"  {r.version:<6} {alpha:>5.2f} {seed:>4} "
                       f"{r.n_params:>10,} {r.best_ic_h1:>+7.4f} "
                       f"{r.final_ic_h1:>+7.4f} {r.final_shic_h1:>+7.4f} "
                       f"{r.shic_ratio:>+7.3f} {r.ic_std_across_assets:>+7.4f} "
                       f"{r.elapsed_s:>5.1f}s")
                flags = []
                if r.can_hit_target: flags.append("HIT")
                if r.anti_fragile: flags.append("AF")
                if r.failure_mode: flags.append(f"FAIL:{r.failure_mode[:25]}")
                row += "  " + (" ".join(flags) or "-")
                print(row, flush=True)

    print("=" * 100)
    print()
    # Summary across all (version, alpha, seed) tuples
    n_total = len(all_results)
    n_hit = sum(1 for r in all_results if r.can_hit_target)
    n_af = sum(1 for r in all_results if r.anti_fragile)
    n_both = sum(1 for r in all_results if r.can_hit_target and r.anti_fragile)
    print(f"  Production candidates (HIT + AF): {n_both}/{n_total}")
    print(f"  Hit target only:                  {n_hit}/{n_total}")
    print(f"  Anti-fragile only:                {n_af}/{n_total}")

    # Per-version aggregate (mean BestIC across alphas, seeds)
    print()
    print(f"  Per-version aggregate ({len(alphas)} alphas x {len(seeds)} seeds = {len(alphas)*len(seeds)} runs each):")
    by_version = {}
    for r in all_results:
        if r.failure_mode:
            continue
        by_version.setdefault(r.version, []).append(r)
    for v, rs in by_version.items():
        if not rs:
            continue
        mean_best = np.mean([r.best_ic_h1 for r in rs])
        mean_final = np.mean([r.final_ic_h1 for r in rs])
        mean_shic_r = np.mean([r.shic_ratio for r in rs])
        n_hit_v = sum(1 for r in rs if r.can_hit_target)
        print(f"    {v:<6}: mean BestIC={mean_best:+.4f} | mean FinalIC={mean_final:+.4f} | "
              f"mean ShIC/IC={mean_shic_r:+.3f} | HIT={n_hit_v}/{len(rs)}")

    return all_results


if __name__ == "__main__":
    main()
