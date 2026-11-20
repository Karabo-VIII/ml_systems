"""
V4 OOS Evaluation -- Does memorization hinder out-of-sample performance?

Tests the hypothesis: a model with low ShIC (memorizing) may still produce
useful OOS predictions if the temporal patterns it memorized generalize.

Evaluates V4 checkpoints across all 4 data segments (50/20/20/10 split)
to measure how IC degrades from train -> val -> OOS -> unseen.

Usage:
    python evaluate_v4_oos.py                        # Best EMA checkpoint
    python evaluate_v4_oos.py --latest               # Latest checkpoint
    python evaluate_v4_oos.py --model path.pt        # Specific checkpoint
    python evaluate_v4_oos.py --all-epochs            # Compare all saved epoch checkpoints
    python evaluate_v4_oos.py --features 13          # Feature count (default: 13)
    python evaluate_v4_oos.py --no-shic              # Skip ShIC (much faster)
    python evaluate_v4_oos.py --quick                # Quick mode (subsample, no ShIC)
"""
import torch
import numpy as np
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from settings import get_feature_config
from world_model import MambaWorldModel, count_parameters
from anti_fragile import load_full_data

PURGE_GAP = 400
MAX_WINDOWS_PER_SEGMENT = 5000  # Cap forward passes per asset-segment (~8 min total)


def split_four_way(n_bars):
    """Return (train_end, val_end, oos_end) indices for 50/20/20/10 split with purge gaps."""
    train_end = int(n_bars * 0.60)
    val_start = train_end + PURGE_GAP
    val_end = int(n_bars * 0.80)
    oos_start = val_end + PURGE_GAP
    oos_end = int(n_bars * 0.90)
    unseen_start = oos_end + PURGE_GAP
    return {
        "train": (0, train_end),
        "val": (val_start, val_end),
        "oos": (oos_start, oos_end),
        "unseen": (unseen_start, n_bars),
    }


def subsample_indices(indices, max_windows):
    """Subsample indices if there are too many, keeping uniform spacing."""
    if len(indices) <= max_windows:
        return indices
    step = len(indices) / max_windows
    return [indices[int(i * step)] for i in range(max_windows)]


@torch.no_grad()
def compute_ic_on_segment(model, feats, targets_h1, asset_idx, device, max_windows=MAX_WINDOWS_PER_SEGMENT):
    """Compute IC(h=1) on a data segment using non-overlapping windows."""
    seq_len = WM_SEQ_LEN
    all_preds, all_reals = [], []

    indices = list(range(0, len(feats) - seq_len, seq_len))
    if not indices:
        return float("nan"), 0

    indices = subsample_indices(indices, max_windows)

    for i in indices:
        obs = torch.from_numpy(feats[i:i+seq_len]).unsqueeze(0).float().to(device)
        asset = torch.tensor([asset_idx], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model.forward_train(obs, asset)
            logits = outputs["return_logits"][1]
            pred = model.bucketer.decode(logits).cpu().numpy().flatten()

        real = targets_h1[i:i+seq_len]
        all_preds.extend(pred)
        all_reals.extend(real)

    preds = np.array(all_preds)
    reals = np.array(all_reals)
    mask = np.isfinite(preds) & np.isfinite(reals)
    if mask.sum() < 50:
        return float("nan"), int(mask.sum())

    ic = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
    return ic, int(mask.sum())


@torch.no_grad()
def compute_shuffled_ic_on_segment(model, feats, targets_h1, asset_idx, device, n_seeds=5, max_windows=MAX_WINDOWS_PER_SEGMENT):
    """Compute ShIC on a specific data segment."""
    seq_len = WM_SEQ_LEN
    all_ics = []

    for seed_offset in range(n_seeds):
        rng = np.random.default_rng(42 + seed_offset * 1000)
        all_preds, all_reals = [], []

        indices = list(range(0, len(feats) - seq_len, seq_len))
        if not indices:
            continue

        indices = subsample_indices(indices, max_windows)

        for i in indices:
            obs_np = feats[i:i+seq_len].copy()
            perm = rng.permutation(seq_len)
            obs_shuffled = obs_np[perm]

            obs = torch.from_numpy(obs_shuffled).unsqueeze(0).float().to(device)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=device)

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model.forward_train(obs, asset)
                logits = outputs["return_logits"][1]
                pred = model.bucketer.decode(logits).cpu().numpy().flatten()

            real = targets_h1[i:i+seq_len][perm]  # Shuffle targets same as features
            all_preds.extend(pred)
            all_reals.extend(real)

        preds = np.array(all_preds)
        reals = np.array(all_reals)
        mask = np.isfinite(preds) & np.isfinite(reals)
        if mask.sum() > 50:
            ic = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
            if np.isfinite(ic):
                all_ics.append(ic)

    return float(np.mean(all_ics)) if all_ics else float("nan")


@torch.no_grad()
def compute_all_horizon_ic(model, feats, targets, asset_idx, device, max_windows=MAX_WINDOWS_PER_SEGMENT):
    """Compute IC for all horizons on a segment."""
    seq_len = WM_SEQ_LEN
    horizon_preds = {h: [] for h in REWARD_HORIZONS}
    horizon_reals = {h: [] for h in REWARD_HORIZONS}

    indices = list(range(0, len(feats) - seq_len, seq_len))
    if not indices:
        return {h: float("nan") for h in REWARD_HORIZONS}

    indices = subsample_indices(indices, max_windows)

    for i in indices:
        obs = torch.from_numpy(feats[i:i+seq_len]).unsqueeze(0).float().to(device)
        asset = torch.tensor([asset_idx], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model.forward_train(obs, asset)

        for h in REWARD_HORIZONS:
            logits = outputs["return_logits"][h]
            pred = model.bucketer.decode(logits).cpu().numpy().flatten()
            real = targets[h][i:i+seq_len]
            horizon_preds[h].extend(pred)
            horizon_reals[h].extend(real)

    result = {}
    for h in REWARD_HORIZONS:
        preds = np.array(horizon_preds[h])
        reals = np.array(horizon_reals[h])
        mask = np.isfinite(preds) & np.isfinite(reals)
        if mask.sum() > 50:
            result[h] = float(np.corrcoef(preds[mask], reals[mask])[0, 1])
        else:
            result[h] = float("nan")
    return result


def load_model(model_path, input_dim, device):
    """Load a V4 model from checkpoint."""
    model = MambaWorldModel(input_dim=input_dim).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        epoch = ckpt.get("epoch", "?")
        shic = ckpt.get("best_shuffled_ic", "?")
        print(f"  Loaded: {model_path.name} (epoch={epoch}, best_shIC={shic})")
    elif isinstance(ckpt, dict):
        # Raw state dict (weights-only file)
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded: {model_path.name} (weights only)")
    else:
        model.load_state_dict(ckpt, strict=False)
        print(f"  Loaded: {model_path.name}")

    model.eval()
    return model


def evaluate_checkpoint(model, all_segments, feature_list, device, compute_shic=True, quick=False):
    """Evaluate one checkpoint across all 4 segments and all assets."""
    segment_names = ["train", "val", "oos", "unseen"]
    max_win = 1000 if quick else MAX_WINDOWS_PER_SEGMENT
    shic_seeds = 2 if quick else 3

    # Per-segment, per-asset results
    results = {seg: {"ics_h1": [], "shics": [], "all_horizon_ics": {h: [] for h in REWARD_HORIZONS},
                      "n_samples": 0} for seg in segment_names}

    # Build asset index for progress display
    idx_to_asset = {v: k for k, v in ASSET_TO_IDX.items()}
    n_assets = len(all_segments)
    t0 = time.time()

    for ai, seg_data in enumerate(all_segments):
        feats = seg_data["features"]
        asset_idx = seg_data["asset_idx"]
        asset_name = idx_to_asset.get(asset_idx, f"asset_{asset_idx}")
        n_bars = len(feats)

        targets_h1 = seg_data["target_return_1"]
        targets_all = {h: seg_data[f"target_return_{h}"] for h in REWARD_HORIZONS}

        splits = split_four_way(n_bars)

        for seg_name in segment_names:
            start, end = splits[seg_name]
            if end <= start or end - start < WM_SEQ_LEN:
                continue

            seg_feats = feats[start:end]
            seg_tgt_h1 = targets_h1[start:end]
            seg_tgt_all = {h: targets_all[h][start:end] for h in REWARD_HORIZONS}
            n_windows = max(0, (end - start - WM_SEQ_LEN) // WM_SEQ_LEN)
            used_windows = min(n_windows, max_win)

            # IC h=1
            ic, n = compute_ic_on_segment(model, seg_feats, seg_tgt_h1, asset_idx, device, max_windows=max_win)
            if np.isfinite(ic):
                results[seg_name]["ics_h1"].append(ic)
            results[seg_name]["n_samples"] += n

            # All horizon ICs
            h_ics = compute_all_horizon_ic(model, seg_feats, seg_tgt_all, asset_idx, device, max_windows=max_win)
            for h in REWARD_HORIZONS:
                if np.isfinite(h_ics.get(h, float("nan"))):
                    results[seg_name]["all_horizon_ics"][h].append(h_ics[h])

            ic_str = f"{ic:+.4f}" if np.isfinite(ic) else "  N/A"
            print(f"    [{ai+1}/{n_assets}] {asset_name:<12} {seg_name:<8} "
                  f"IC1={ic_str} ({used_windows} windows, {end-start} bars)", flush=True)

            # ShIC (only on val/oos/unseen to save time, train is always high)
            if compute_shic and seg_name != "train":
                shic = compute_shuffled_ic_on_segment(
                    model, seg_feats, seg_tgt_h1, asset_idx, device,
                    n_seeds=shic_seeds, max_windows=max_win
                )
                if np.isfinite(shic):
                    results[seg_name]["shics"].append(shic)
                    print(f"                          ShIC={shic:+.4f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Evaluation completed in {elapsed:.0f}s", flush=True)
    return results


def print_results(results, model_name=""):
    """Print evaluation results."""
    segment_names = ["train", "val", "oos", "unseen"]

    print(f"\n{'='*80}")
    print(f"  V4 OOS EVALUATION: {model_name}")
    print(f"  Hypothesis: Does memorization (low ShIC) hinder OOS performance?")
    print(f"{'='*80}")

    # Header
    print(f"\n  {'Segment':<10} {'IC(h1)':>8} {'ShIC':>8} {'Ratio':>8} "
          f"{'IC(h4)':>8} {'IC(h16)':>8} {'IC(h64)':>8} {'N':>8}")
    print(f"  {'-'*74}")

    for seg in segment_names:
        r = results[seg]
        ic = np.mean(r["ics_h1"]) if r["ics_h1"] else float("nan")
        shic = np.mean(r["shics"]) if r["shics"] else float("nan")
        ratio = shic / ic if np.isfinite(shic) and np.isfinite(ic) and abs(ic) > 1e-6 else float("nan")

        ic4 = np.mean(r["all_horizon_ics"][4]) if r["all_horizon_ics"][4] else float("nan")
        ic16 = np.mean(r["all_horizon_ics"][16]) if r["all_horizon_ics"][16] else float("nan")
        ic64 = np.mean(r["all_horizon_ics"][64]) if r["all_horizon_ics"][64] else float("nan")

        ratio_str = f"{ratio:.3f}" if np.isfinite(ratio) else "  N/A"
        shic_str = f"{shic:+.4f}" if np.isfinite(shic) else "   N/A"

        print(f"  {seg:<10} {ic:+.4f}   {shic_str}  {ratio_str}  "
              f"{ic4:+.4f}   {ic16:+.4f}   {ic64:+.4f}   {r['n_samples']:>6}")

    # Key diagnostic
    train_ic = np.mean(results["train"]["ics_h1"]) if results["train"]["ics_h1"] else 0
    oos_ic = np.mean(results["oos"]["ics_h1"]) if results["oos"]["ics_h1"] else 0
    unseen_ic = np.mean(results["unseen"]["ics_h1"]) if results["unseen"]["ics_h1"] else 0
    val_shic = np.mean(results["val"]["shics"]) if results["val"]["shics"] else 0
    oos_shic = np.mean(results["oos"]["shics"]) if results["oos"]["shics"] else 0

    print(f"\n  KEY DIAGNOSTICS:")
    if abs(train_ic) > 1e-6:
        print(f"    IC retention (train->OOS):    {oos_ic/train_ic*100:.1f}%")
        print(f"    IC retention (train->unseen): {unseen_ic/train_ic*100:.1f}%")
    print(f"    OOS ShIC:                    {oos_shic:+.4f} (gate: > 0.015)")
    print(f"    Val ShIC:                    {val_shic:+.4f}")

    # Verdict
    print(f"\n  VERDICT:")
    if oos_ic > 0.015:
        if oos_shic > 0.010:
            print(f"    Model generalizes AND has cross-sectional signal OOS.")
            print(f"    Memorization is NOT hindering performance.")
        else:
            print(f"    Model has positive OOS IC ({oos_ic:+.4f}) but low ShIC ({oos_shic:+.4f}).")
            print(f"    -> Temporal patterns learned in-sample may be carrying over to OOS.")
            print(f"    -> This is fragile: works only if future resembles the past.")
    elif oos_ic > 0:
        print(f"    Weak OOS IC ({oos_ic:+.4f}). Memorization IS hurting generalization.")
    else:
        print(f"    Zero or negative OOS IC. Model is purely memorizing.")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="V4 OOS Evaluation -- memorization vs generalization")
    parser.add_argument("--features", type=int, choices=[13, 18, 30, 37], default=13)
    parser.add_argument("--model", type=str, default=None, help="Specific checkpoint path")
    parser.add_argument("--latest", action="store_true", help="Use latest checkpoint")
    parser.add_argument("--all-epochs", action="store_true", help="Compare all saved epoch checkpoints")
    parser.add_argument("--no-shic", action="store_true", help="Skip ShIC computation (faster)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (1000 max windows, fewer ShIC seeds)")
    args = parser.parse_args()

    feature_list, input_dim, base_dim = get_feature_config(args.features)
    feat_tag = f"f{args.features}"
    device = torch.device(DEVICE)

    mode_str = "QUICK" if args.quick else ("no-ShIC" if args.no_shic else "full")
    print(f"\n  Loading data ({args.features} features, mode={mode_str})...")
    all_segments = load_full_data(
        DATA_DIR, feature_list, ASSET_TO_IDX, REWARD_HORIZONS
    )
    if not all_segments:
        print("  [ERROR] No data loaded.")
        sys.exit(1)

    if args.all_epochs:
        # Find all epoch checkpoints
        ckpt_pattern = f"v4_{feat_tag}_wm_epoch_*.pt"
        ckpt_files = sorted(BASE_MODEL_DIR.glob(ckpt_pattern))

        # Also include best_ema and latest
        for name in [f"v4_{feat_tag}_wm_best_ema.pt", f"v4_{feat_tag}_wm_latest.pt"]:
            p = BASE_MODEL_DIR / name
            if p.exists() and p not in ckpt_files:
                ckpt_files.append(p)

        if not ckpt_files:
            print(f"  [ERROR] No checkpoints found matching {ckpt_pattern}")
            sys.exit(1)

        print(f"\n  Found {len(ckpt_files)} checkpoints. Evaluating each...")
        print(f"  {'Checkpoint':<35} {'Train IC':>10} {'Val IC':>10} {'OOS IC':>10} "
              f"{'Unseen IC':>10} {'Val ShIC':>10} {'OOS ShIC':>10}")
        print(f"  {'-'*95}")

        for ckpt_path in ckpt_files:
            model = load_model(ckpt_path, input_dim, device)
            results = evaluate_checkpoint(model, all_segments, feature_list, device,
                                         compute_shic=not args.no_shic, quick=args.quick)

            train_ic = np.mean(results["train"]["ics_h1"]) if results["train"]["ics_h1"] else float("nan")
            val_ic = np.mean(results["val"]["ics_h1"]) if results["val"]["ics_h1"] else float("nan")
            oos_ic = np.mean(results["oos"]["ics_h1"]) if results["oos"]["ics_h1"] else float("nan")
            unseen_ic = np.mean(results["unseen"]["ics_h1"]) if results["unseen"]["ics_h1"] else float("nan")
            val_shic = np.mean(results["val"]["shics"]) if results["val"]["shics"] else float("nan")
            oos_shic = np.mean(results["oos"]["shics"]) if results["oos"]["shics"] else float("nan")

            shic_str_val = f"{val_shic:+.4f}" if np.isfinite(val_shic) else "  N/A"
            shic_str_oos = f"{oos_shic:+.4f}" if np.isfinite(oos_shic) else "  N/A"

            print(f"  {ckpt_path.name:<35} {train_ic:+10.4f} {val_ic:+10.4f} "
                  f"{oos_ic:+10.4f} {unseen_ic:+10.4f} {shic_str_val:>10} {shic_str_oos:>10}")

            del model
            torch.cuda.empty_cache()

    else:
        # Single checkpoint evaluation
        if args.model:
            model_path = Path(args.model)
        elif args.latest:
            model_path = BASE_MODEL_DIR / f"v4_{feat_tag}_wm_latest.pt"
        else:
            model_path = BASE_MODEL_DIR / f"v4_{feat_tag}_wm_best_ema.pt"
            if not model_path.exists():
                model_path = BASE_MODEL_DIR / f"v4_{feat_tag}_wm_latest.pt"

        if not model_path.exists():
            print(f"  [ERROR] Checkpoint not found: {model_path}")
            sys.exit(1)

        model = load_model(model_path, input_dim, device)
        results = evaluate_checkpoint(model, all_segments, feature_list, device,
                                     compute_shic=not args.no_shic, quick=args.quick)
        print_results(results, model_path.name)


if __name__ == "__main__":
    main()
