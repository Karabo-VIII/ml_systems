#!/usr/bin/env python
"""Real-Data Architectural-Ceiling Probe — measures IC ceiling on actual crypto data.

Companion to probe_architectural_ceiling.py (synthetic). The synthetic probe is
a NECESSARY filter — architectures that fail it definitely fail real. This
probe is the SUFFICIENCY check — actual IC ceiling on chimera_legacy slices
in <2 min per architecture.

Procedure:
  1. Load N assets from chimera_legacy/dollar (default: 3 small BTC/ETH/SOL)
  2. Build [B, T, F] windows with feature_set (default f29)
  3. Walk-forward 70/30 split (purge gap = 200 bars)
  4. Train each architecture for n_steps with AdamW + AMP autocast
  5. Track val IC every 50 steps; report BestIC / FinalIC
  6. ShIC: re-eval with features time-shuffled per sample
  7. Per-asset IC breakdown (info-max axis 3)

Reports:
  - BestIC at h=1 on real validation
  - ShIC ratio (anti-fragile gate)
  - Per-asset IC variance (regime / asset-specific learning)
  - HIT (BestIC >= 0.05, our V1.x baseline tier)
  - HEADLINE (BestIC >= 0.10, target tier)
  - AF (ShIC/IC >= 0.3)

Usage:
  python scripts/probe_real_data_ceiling.py --assets BTC,ETH,SOL --n-steps 500 --models v1_1,v22,v25
  python scripts/probe_real_data_ceiling.py --models v22 --n-steps 1000  # single arch deep probe
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# Reuse architecture loading logic
from probe_architectural_ceiling import (   # noqa: E402
    ARCHITECTURE_REGISTRY, load_architecture as _default_load, compute_ic
)


def load_architecture_with_source(version: str, source: str = "current"):
    """Load architecture from either current src/ or a backup directory.

    `source` options:
      - "current"  → src/wm/ (default; post-round-7 state)
      - "backup_pre_sota_gaps"  → backups/BKP_20260507_PRE_SOTA_GAPS/wm/
                                   (pre-round-6/7 state — A/B baseline)
    """
    if source == "current":
        return _default_load(version)
    if source == "backup_pre_sota_gaps":
        # Re-route the architecture's path to the backup tree
        if version not in ARCHITECTURE_REGISTRY:
            raise ValueError(f"Unknown version: {version}")
        rel_path, cls_name = ARCHITECTURE_REGISTRY[version]
        # Replace src/wm with backups/BKP_20260507_PRE_SOTA_GAPS/wm
        backup_rel = rel_path.replace("src/wm", "backups/BKP_20260507_PRE_SOTA_GAPS/wm")
        abs_path = PROJECT_ROOT / backup_rel
        if not abs_path.exists():
            raise FileNotFoundError(f"Backup path not found: {abs_path}")
        # Clear conflicting modules
        for mod in ("settings", "world_model", "components"):
            sys.modules.pop(mod, None)
        # V1.x components from backup too (if backup has its own; else fall back)
        v1_path = str(PROJECT_ROOT / "backups" / "BKP_20260507_PRE_SOTA_GAPS" / "wm" / "v1" / "v1_0_training")
        if Path(v1_path).exists() and v1_path not in sys.path:
            sys.path.insert(0, v1_path)
        if str(abs_path) not in sys.path:
            sys.path.insert(0, str(abs_path))
        import settings as st
        import world_model as wm
        return st, getattr(wm, cls_name)
    raise ValueError(f"Unknown source: {source}")


# =============================================================================
# Real data loader — chimera_legacy/dollar slice
# =============================================================================

def load_real_data_slice(asset_symbols: list[str], feature_list: list[str],
                          target_horizon: int = 1,
                          max_rows: int = 50000) -> dict:
    """Load chimera_legacy/dollar slices for the requested assets.

    Returns dict {asset_idx: {features: np.ndarray [N, F], targets: np.ndarray [N]}}.
    Each asset's data is independently sliced (max_rows per asset).
    """
    data_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    out = {}
    for idx, sym in enumerate(asset_symbols):
        sym_lower = sym.lower()
        # Find the most recent file
        candidates = sorted(data_dir.glob(f"{sym_lower}usdt_v50_chimera_*.parquet"))
        if not candidates:
            print(f"  [WARN] no file for {sym}; skipping")
            continue
        path = candidates[-1]   # latest
        df = pl.read_parquet(path).tail(max_rows)
        # Extract features that exist in the dataframe
        avail_feats = [c for c in feature_list if c in df.columns]
        if len(avail_feats) < len(feature_list):
            missing = set(feature_list) - set(avail_feats)
            print(f"  [WARN] {sym}: missing {len(missing)} features; using {len(avail_feats)}")
        target_col = f"target_return_{target_horizon}"
        if target_col not in df.columns:
            print(f"  [WARN] {sym}: missing {target_col}; skipping")
            continue
        feat_arr = df.select(avail_feats).to_numpy().astype(np.float32)
        tgt_arr = df.select(target_col).to_numpy().flatten().astype(np.float32)
        # Drop NaN rows
        mask = np.isfinite(feat_arr).all(axis=1) & np.isfinite(tgt_arr)
        feat_arr = feat_arr[mask]
        tgt_arr = tgt_arr[mask]
        out[idx] = {
            "features": feat_arr,
            "targets": tgt_arr,
            "symbol": sym,
            "n_rows": len(feat_arr),
        }
        print(f"  loaded {sym}: {len(feat_arr)} rows, {len(avail_feats)} features")
    return out


def build_windowed_dataset(real_data: dict, seq_len: int = 96,
                           train_frac: float = 0.7, purge_bars: int = 200,
                           seed: int = 42):
    """Build windowed [N, T, F] tensors with walk-forward train/val split.

    Returns:
        train_obs, train_tgt, train_asset, val_obs, val_tgt, val_asset
        where each is a numpy/tensor with N as the leading dim.
    """
    rng = np.random.default_rng(seed)
    train_x, train_y, train_a = [], [], []
    val_x, val_y, val_a = [], [], []

    for asset_idx, d in real_data.items():
        feats = d["features"]   # [N, F]
        tgts = d["targets"]     # [N]
        n_rows = len(feats)
        if n_rows < seq_len + purge_bars + 100:
            continue
        # Build all valid windows: each window starts at t, ends at t+seq_len-1, target at t+seq_len-1
        max_start = n_rows - seq_len
        # Walk-forward split
        train_end = int(train_frac * max_start)
        val_start = train_end + purge_bars
        # Sample windows
        train_starts = rng.choice(np.arange(0, train_end), size=min(500, train_end), replace=False)
        val_starts = rng.choice(np.arange(val_start, max_start), size=min(200, max_start - val_start), replace=False)
        for s in train_starts:
            train_x.append(feats[s:s+seq_len])
            train_y.append(tgts[s:s+seq_len])
            train_a.append(asset_idx)
        for s in val_starts:
            val_x.append(feats[s:s+seq_len])
            val_y.append(tgts[s:s+seq_len])
            val_a.append(asset_idx)

    train_obs = np.stack(train_x)        # [N_train, T, F]
    train_tgt = np.stack(train_y)        # [N_train, T]
    train_asset = np.array(train_a, dtype=np.int64)
    val_obs = np.stack(val_x)
    val_tgt = np.stack(val_y)
    val_asset = np.array(val_a, dtype=np.int64)

    return (train_obs, train_tgt, train_asset, val_obs, val_tgt, val_asset)


# =============================================================================
# Probe one architecture on real data
# =============================================================================

@dataclass
class RealDataProbeResult:
    version: str
    n_steps: int
    final_ic_h1: float = 0.0
    best_ic_h1: float = 0.0
    final_shic_h1: float = 0.0
    shic_ratio: float = 0.0
    hit_baseline: bool = False        # >= 0.05 (V1.x tier)
    hit_headline: bool = False        # >= 0.10 (target tier)
    anti_fragile: bool = False        # ShIC/IC >= 0.3
    per_asset_ic: dict = field(default_factory=dict)
    ic_std_across_assets: float = 0.0
    elapsed_s: float = 0.0
    failure_mode: str = ""
    n_params: int = 0


def probe_real_data(version: str, real_data: dict, train_set, val_set,
                    n_features: int, n_steps: int = 500,
                    seed: int = 42, device: str = "cuda",
                    source: str = "current") -> RealDataProbeResult:
    """Run probe on real data. `source` toggles current vs backup (A/B harness)."""
    result = RealDataProbeResult(version=version, n_steps=n_steps)
    t0 = time.time()
    try:
        st, ModelCls = load_architecture_with_source(version, source=source)
    except Exception as e:
        result.failure_mode = f"load: {str(e)[:80]}"
        return result

    model = None
    try:
        _, _, base_dim = st.get_feature_config(n_features)
    except Exception:
        base_dim = getattr(st, "BASE_DIM", n_features)
    init_attempts = [
        lambda: ModelCls(input_dim=n_features, base_dim=base_dim),
        lambda: ModelCls(input_dim=n_features),
        lambda: ModelCls(),
    ]
    for attempt in init_attempts:
        try:
            model = attempt().to(device)
            break
        except Exception:
            continue
    if model is None:
        result.failure_mode = "init failed"
        return result

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result.n_params = n_params
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    train_obs_np, train_tgt_np, train_asset_np = train_set
    val_obs_np, val_tgt_np, val_asset_np = val_set

    val_obs = torch.from_numpy(val_obs_np).to(device)
    val_tgt = torch.from_numpy(val_tgt_np).to(device)
    val_asset = torch.from_numpy(val_asset_np).to(device)

    rng = np.random.default_rng(seed)
    B = 32
    n_train = len(train_obs_np)

    model.train()
    for step in range(n_steps):
        idx = rng.choice(n_train, size=B, replace=(n_train < B))
        obs = torch.from_numpy(train_obs_np[idx]).to(device)
        tgt = torch.from_numpy(train_tgt_np[idx]).to(device)
        asset = torch.from_numpy(train_asset_np[idx]).to(device)
        targets = {1: tgt, 4: tgt, 16: tgt, 64: tgt}   # use h=1 target as proxy for all
        try:
            with torch.amp.autocast("cuda"):
                try:
                    loss, ld, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15)
                except TypeError:
                    loss, ld, _ = model.get_loss(obs, asset, targets, mask_ratio=0.15,
                                                  temporal_ctx_drop=0.15)
            if not torch.isfinite(loss).item():
                result.failure_mode = f"NaN at step {step}"
                break
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        except Exception as e:
            result.failure_mode = f"step {step}: {str(e)[:60]}"
            break

        if (step + 1) % 50 == 0:
            model.eval()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                try:
                    val_out = model.forward_train(val_obs, val_asset)
                    pred_logits = val_out["return_logits"][1]
                    nb = getattr(model, "_num_bins", None) or pred_logits.shape[-1]
                    pred = model.bucketer.decode(pred_logits.reshape(-1, nb))
                    ic = compute_ic(pred, val_tgt)
                    if ic > result.best_ic_h1:
                        result.best_ic_h1 = ic
                except Exception:
                    pass
            model.train()

    # Final IC + ShIC + per-asset
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda"):
        try:
            val_out = model.forward_train(val_obs, val_asset)
            nb = getattr(model, "_num_bins", None) or val_out["return_logits"][1].shape[-1]
            pred = model.bucketer.decode(val_out["return_logits"][1].reshape(-1, nb))
            result.final_ic_h1 = compute_ic(pred, val_tgt)

            # ShIC: time-shuffle val features per sample
            val_obs_sh = val_obs.clone()
            for b in range(len(val_obs_sh)):
                perm = torch.randperm(val_obs_sh.shape[1], device=val_obs_sh.device)
                val_obs_sh[b] = val_obs_sh[b, perm]
            val_out_sh = model.forward_train(val_obs_sh, val_asset)
            pred_sh = model.bucketer.decode(val_out_sh["return_logits"][1].reshape(-1, nb))
            result.final_shic_h1 = compute_ic(pred_sh, val_tgt)
            if abs(result.final_ic_h1) > 1e-6:
                result.shic_ratio = result.final_shic_h1 / result.final_ic_h1

            # Per-asset IC (info-max)
            B_val, T_val = val_obs.shape[:2]
            pred_2d = pred.reshape(B_val, T_val)
            asset_cpu = val_asset.detach().cpu().numpy()
            ic_per_asset = []
            for a in real_data.keys():
                mask = (asset_cpu == a)
                if mask.sum() > 0:
                    p = pred_2d[mask].flatten()
                    t = val_tgt[mask].flatten()
                    ic_a = compute_ic(p, t)
                    result.per_asset_ic[a] = ic_a
                    ic_per_asset.append(ic_a)
            if len(ic_per_asset) > 1:
                result.ic_std_across_assets = float(np.std(ic_per_asset))

            result.hit_baseline = result.best_ic_h1 >= 0.05
            result.hit_headline = result.best_ic_h1 >= 0.10
            result.anti_fragile = result.shic_ratio >= 0.3
        except Exception as e:
            result.failure_mode = f"eval: {str(e)[:60]}"

    result.elapsed_s = time.time() - t0
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=str, default="BTC,ETH,SOL",
                        help="Comma-separated asset symbols")
    parser.add_argument("--n-features", type=int, default=29,
                        choices=[13, 18, 25, 29, 34, 37, 41])
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--max-rows", type=int, default=30000,
                        help="Max rows per asset (most recent)")
    parser.add_argument("--models", type=str, default="v1_1,v22,v25",
                        help="Comma-separated versions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--source", type=str, default="current",
                        choices=["current", "backup_pre_sota_gaps", "ab"],
                        help="'current' = post-round-7; 'backup_pre_sota_gaps' = "
                             "pre-round-6/7; 'ab' = run BOTH and report delta")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Load shared feature list directly from feature_sets (avoids polluting sys.path
    # with one architecture's settings; load_architecture handles per-version paths)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from feature_sets import get_feature_config as _get_fc   # noqa: E402
    feature_list, _, _ = _get_fc(args.n_features)
    # Clean up to avoid affecting per-architecture settings imports later
    sys.path.pop(0)

    print("=" * 100)
    print("  REAL-DATA ARCHITECTURAL CEILING PROBE")
    print(f"  Assets: {args.assets}  Features: f{args.n_features}  Steps: {args.n_steps}")
    print(f"  Models: {args.models}")
    print("=" * 100)

    asset_symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
    real_data = load_real_data_slice(asset_symbols, feature_list, max_rows=args.max_rows)
    if not real_data:
        print("[ERROR] No data loaded. Exiting.")
        return

    print()
    print("Building windows...")
    train_set = build_windowed_dataset(real_data, seq_len=96, train_frac=0.7,
                                         purge_bars=200, seed=args.seed)
    train_obs_np, train_tgt_np, train_asset_np, val_obs_np, val_tgt_np, val_asset_np = train_set
    print(f"  Train: {len(train_obs_np)} windows  Val: {len(val_obs_np)} windows")
    print()

    versions = [v.strip() for v in args.models.split(",") if v.strip()]
    sources = ["current"] if args.source != "ab" else ["backup_pre_sota_gaps", "current"]

    print(f"  {'Ver':<6} {'src':<8} {'Params':>10} {'BestIC':>8} {'FinIC':>8} {'ShIC':>8} {'ShIC/IC':>8} {'IC_std':>8} {'Time':>6}  Flags")
    print("-" * 110)

    results = []
    for v in versions:
        for src in sources:
            if args.device == "cuda":
                torch.cuda.empty_cache()
            r = probe_real_data(
                v, real_data,
                train_set=(train_obs_np, train_tgt_np, train_asset_np),
                val_set=(val_obs_np, val_tgt_np, val_asset_np),
                n_features=args.n_features,
                n_steps=args.n_steps,
                seed=args.seed,
                device=args.device,
                source=src,
            )
            r.source = src   # tag for A/B comparison
            results.append(r)
            flags = []
            if r.hit_headline:
                flags.append("HEADLINE")
            elif r.hit_baseline:
                flags.append("BASELINE")
            if r.anti_fragile:
                flags.append("AF")
            if r.failure_mode:
                flags.append(f"FAIL:{r.failure_mode[:25]}")
            flag_str = " ".join(flags) or "-"
            src_label = "post-r7" if src == "current" else "pre-r6"
            print(f"  {r.version:<6} {src_label:<8} {r.n_params:>10,} {r.best_ic_h1:>+7.4f} "
                  f"{r.final_ic_h1:>+7.4f} {r.final_shic_h1:>+7.4f} "
                  f"{r.shic_ratio:>+7.3f} {r.ic_std_across_assets:>+7.4f} "
                  f"{r.elapsed_s:>5.1f}s  {flag_str}", flush=True)

    print("=" * 100)
    print()
    n_baseline = sum(1 for r in results if r.hit_baseline)
    n_headline = sum(1 for r in results if r.hit_headline)
    n_af = sum(1 for r in results if r.anti_fragile)
    n_prod = sum(1 for r in results if r.hit_headline and r.anti_fragile)
    print(f"  HEADLINE-tier (BestIC >= 0.10): {n_headline}/{len(results)}")
    print(f"  Baseline-tier (BestIC >= 0.05): {n_baseline}/{len(results)}")
    print(f"  Anti-fragile (ShIC/IC >= 0.3):  {n_af}/{len(results)}")
    print(f"  PRODUCTION (HEADLINE + AF):     {n_prod}/{len(results)}")
    print()
    print("  Per-asset IC breakdown:")
    for r in results:
        if r.per_asset_ic and not r.failure_mode:
            asset_strs = [f"{a}={ic:+.3f}" for a, ic in sorted(r.per_asset_ic.items())]
            src_label = "post-r7" if getattr(r, "source", "current") == "current" else "pre-r6"
            print(f"    {r.version} ({src_label}): {' '.join(asset_strs)}")

    # A/B delta if both sources ran
    if args.source == "ab":
        print()
        print("  A/B delta (post-r7 minus pre-r6):")
        by_v = {}
        for r in results:
            by_v.setdefault(r.version, {})[getattr(r, "source", "current")] = r
        for v, rs in by_v.items():
            if "current" in rs and "backup_pre_sota_gaps" in rs:
                cur, base = rs["current"], rs["backup_pre_sota_gaps"]
                d_ic = cur.best_ic_h1 - base.best_ic_h1
                d_shic = cur.shic_ratio - base.shic_ratio
                tag = "GOOD" if d_ic > 0 else ("NEUTRAL" if abs(d_ic) < 0.005 else "BAD")
                print(f"    {v}: dBestIC={d_ic:+.4f}  dShIC/IC={d_shic:+.3f}  -> {tag}")

    return results


if __name__ == "__main__":
    main()
