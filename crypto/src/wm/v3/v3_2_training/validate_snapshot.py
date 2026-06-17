"""
V3.E Snapshot Ensemble Validation

Usage:
    python validate_snapshot.py              # Ensemble from best snapshots
    python validate_snapshot.py --model X    # Specific checkpoint dir
"""
import torch
import torch.nn.functional as F
import numpy as np
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from snapshot_ensemble import SnapshotEnsemble, ENSEMBLE_MODEL_DIR
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets


# ===============================================================================
# DATA LOADING
# ===============================================================================

PURGE_GAP_BARS = 400


def load_validation_data():
    """Load 10% validation split with purge gap from all chimera files."""
    files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
    segments = []
    try:
        import polars as pl
    except ImportError:
        print("  [FAIL] polars not installed")
        sys.exit(1)

    for f in files:
        asset_name = f.stem.split("_")[0].upper()
        if asset_name not in ASSET_TO_IDX:
            continue
        asset_idx = ASSET_TO_IDX[asset_name]
        df = pl.read_parquet(f)
        df = selective_drop_nulls(df, FEATURE_LIST, REWARD_HORIZONS, asset_name)

        # FIX 2026-05-29: was 0.90 -> evaluated on the UNSEEN held-out segment.
        # Val window = [50%+purge, 70%] (mirrors v1.0, the correct one).
        val_start = int(len(df) * 0.50) + PURGE_GAP_BARS
        val_end = int(len(df) * 0.70)
        if val_start >= val_end:
            val_start = int(len(df) * 0.50)
        df_val = df.slice(val_start, val_end - val_start)

        feats, targets = extract_features_targets(df_val, FEATURE_LIST, REWARD_HORIZONS, asset_name)
        segments.append((feats, targets, asset_idx, asset_name))
        print(f"    {asset_name}: {len(feats):,} validation bars")

    return segments


# ===============================================================================
# IC COMPUTATION
# ===============================================================================

@torch.no_grad()
def compute_asset_ic(model, feats, targets, asset_idx):
    """
    Compute IC per horizon for one asset using non-overlapping windows.
    model.forward_train handles RevIN internally (SnapshotEnsemble).
    Returns dict {horizon: ic}.
    """
    seq_len = WM_SEQ_LEN
    asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
    indices = list(range(0, len(feats) - seq_len, seq_len))
    if not indices:
        return {h: 0.0 for h in REWARD_HORIZONS}

    all_preds = {h: [] for h in REWARD_HORIZONS}
    all_reals = {h: [] for h in REWARD_HORIZONS}

    for i in indices:
        obs_np = feats[i:i + seq_len]
        obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)

        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            outputs = model.forward_train(obs, asset)

        for h in REWARD_HORIZONS:
            logits_h = outputs["return_logits"][h]
            pred_h = model.bucketer.decode(logits_h).cpu().numpy().flatten()
            real_h = targets[h][i:i + seq_len]
            all_preds[h].append(pred_h)
            all_reals[h].append(real_h)

    ics = {}
    for h in REWARD_HORIZONS:
        p = np.concatenate(all_preds[h])
        r = np.concatenate(all_reals[h])
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() > 50:
            ic = float(np.corrcoef(p[mask], r[mask])[0, 1])
            ics[h] = ic if np.isfinite(ic) else 0.0
        else:
            ics[h] = 0.0

    return ics


@torch.no_grad()
def compute_single_snapshot_ic1(snap_model, revin, feats, targets, asset_idx):
    """
    Compute IC(1) for a single snapshot model (not the ensemble).
    Applies RevIN manually if provided.
    """
    seq_len = WM_SEQ_LEN
    asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
    indices = list(range(0, len(feats) - seq_len, seq_len))
    if not indices:
        return 0.0

    preds, reals = [], []
    for i in indices:
        obs_np = feats[i:i + seq_len]
        obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)

        if revin is not None:
            obs = revin(obs, mode='norm')

        with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            outputs = snap_model.forward_train(obs, asset)

        logits = outputs["return_logits"][1]
        pred = snap_model.bucketer.decode(logits).cpu().numpy().flatten()
        preds.append(pred)
        reals.append(targets[1][i:i + seq_len])

    p = np.concatenate(preds)
    r = np.concatenate(reals)
    mask = np.isfinite(p) & np.isfinite(r)
    if mask.sum() > 50:
        ic = float(np.corrcoef(p[mask], r[mask])[0, 1])
        return ic if np.isfinite(ic) else 0.0
    return 0.0


@torch.no_grad()
def compute_global_shuffled_ic(model, segments, n_seeds=5, batch_size=32):
    """
    IC on globally-shuffled data (anti-memorization gate).
    SnapshotEnsemble handles RevIN internally -- no external RevIN applied.
    """
    all_ics = []
    for seed_offset in range(n_seeds):
        rng = np.random.default_rng(42 + seed_offset * 1000)
        all_preds, all_reals = [], []

        for feats, targets, asset_idx, _ in segments:
            n = len(feats)
            if n < WM_SEQ_LEN * 2:
                continue
            idx = np.arange(n)
            rng.shuffle(idx)
            shuffled_feats = feats[idx]
            shuffled_targets_1 = targets[1][idx]

            window_starts = list(range(0, n - WM_SEQ_LEN, WM_SEQ_LEN))
            for batch_start in range(0, len(window_starts), batch_size):
                batch_ws = window_starts[batch_start:batch_start + batch_size]
                obs_list = [shuffled_feats[i:i + WM_SEQ_LEN] for i in batch_ws]
                real_list = [shuffled_targets_1[i:i + WM_SEQ_LEN] for i in batch_ws]

                obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
                asset = torch.full((len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE)

                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    outputs = model.forward_train(obs, asset)

                preds = model.bucketer.decode(outputs["return_logits"][1]).cpu().numpy()
                for b, real in enumerate(real_list):
                    all_preds.extend(preds[b].flatten())
                    all_reals.extend(real)

        p = np.array(all_preds)
        r = np.array(all_reals)
        mask = np.isfinite(p) & np.isfinite(r)
        if mask.sum() > 50:
            ic = float(np.corrcoef(p[mask], r[mask])[0, 1])
            if np.isfinite(ic):
                all_ics.append(ic)

    return float(np.mean(all_ics)) if all_ics else 0.0


# ===============================================================================
# GATE CHECKS
# ===============================================================================

def check_gates(avg_rec_mse, mean_ic, shuffled_ic, ic_h1=None):
    """Print gate pass/fail table. Returns True if all gates pass."""
    print(f"\n  {'='*60}")
    print(f"  VALIDATION GATES")
    print(f"  {'='*60}")

    if abs(mean_ic) > 1e-6:
        shic_denom = ic_h1 if ic_h1 is not None else mean_ic
        shic_ratio = shuffled_ic / shic_denom
    else:
        shic_ratio = 0.0

    gates = [
        ("Reconstruction MSE",      avg_rec_mse < GATE_REC_MSE_MAX,
         f"{avg_rec_mse:.5f} < {GATE_REC_MSE_MAX}"),
        ("Mean IC (all horizons)",   mean_ic > GATE_IC_MIN,
         f"{mean_ic:+.4f} > {GATE_IC_MIN}"),
        ("Shuffled IC Ratio",        shic_ratio > GATE_SHUFFLED_IC_RATIO_MIN,
         f"{shic_ratio:.3f} > {GATE_SHUFFLED_IC_RATIO_MIN}"),
    ]

    all_pass = True
    for name, passed, desc in gates:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"    [{status}] {name:<28} {desc}")

    verdict = "ALL GATES PASSED" if all_pass else "GATE(S) FAILED"
    print(f"\n  VERDICT: {verdict}")
    return all_pass


# ===============================================================================
# MAIN VALIDATION
# ===============================================================================

def run_validation(model_path=None):
    print(f"\n{'='*65}")
    print(f"  V3.E SNAPSHOT ENSEMBLE VALIDATION")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}")

    # --- Load ensemble -------------------------------------------------------
    print(f"\n  Loading seed ensemble from {ENSEMBLE_MODEL_DIR}")

    # Try to load seed ICs from master checkpoint for top-K selection
    snapshot_ics = None
    master_path = ENSEMBLE_MODEL_DIR / "v3_2e_master.pt"
    if master_path.exists():
        try:
            master = torch.load(master_path, map_location=DEVICE, weights_only=False)
            if isinstance(master, dict) and "seed_metrics" in master:
                # Build {seed_idx: best_ic} dict for top-K ranking
                snapshot_ics = {
                    idx: m.get("best_ic", 0.0)
                    for idx, m in master["seed_metrics"].items()
                }
                print(f"  [OK] Loaded seed ICs from v3e_master.pt")
        except Exception:
            pass

    if model_path:
        snap_dir = Path(model_path)
        # Prefer best-ShIC snapshots (peak anti-memorization performance)
        snapshot_paths = sorted(snap_dir.glob("v3_2e_seed_*_best.pt"))
        if not snapshot_paths:
            # Fallback to final models (backward compatibility)
            snapshot_paths = sorted(snap_dir.glob("v3_2_seed_*.pt"))
    else:
        snapshot_paths = None  # SnapshotEnsemble will auto-discover

    try:
        model = SnapshotEnsemble(
            snapshot_paths=snapshot_paths,
            top_k=ENSEMBLE_TOP_K,
            snapshot_ics=snapshot_ics,
        ).to(DEVICE)
        model.eval()
    except FileNotFoundError as e:
        print(f"  [FAIL] {e}")
        sys.exit(1)

    print(f"  Ensemble: {model.n_models} seed models loaded")

    # --- Load validation data ------------------------------------------------
    print(f"\n  Loading validation data from {DATA_DIR}")
    segments = load_validation_data()
    if not segments:
        print("  [FAIL] No validation data found.")
        sys.exit(1)

    # --- Ensemble IC per asset -----------------------------------------------
    print(f"\n  Computing ensemble IC...")
    print(f"  {'Asset':<10} {'IC(1)':>7} {'IC(4)':>7} {'IC(16)':>8} {'IC(64)':>8} {'RecMSE':>8}")
    print(f"  {'-'*53}")

    asset_results = {}
    for feats, targets, asset_idx, asset_name in tqdm(segments, desc="  Assets", leave=True):
        ics = compute_asset_ic(model, feats, targets, asset_idx)

        # Reconstruction MSE from first non-overlapping window
        # NOTE: SnapshotEnsemble applies RevIN internally per-snapshot, so
        # recon is in RevIN-normalized space. We must normalize obs to match.
        obs_np = feats[:WM_SEQ_LEN]
        obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
        asset_t = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            out = model.forward_train(obs, asset_t)
        recon = out["recon"]
        # Apply first snapshot's RevIN to obs for fair comparison
        if model._revins[0] is not None:
            obs_norm = model._revins[0](obs, mode='norm')
            rec_mse = F.mse_loss(recon, obs_norm).item()
        else:
            rec_mse = F.mse_loss(recon, obs).item()

        asset_results[asset_name] = {"ics": ics, "rec_mse": rec_mse}
        print(f"  {asset_name:<10} "
              f"{ics[1]:>+7.4f} {ics[4]:>+7.4f} {ics[16]:>+8.4f} "
              f"{ics[64]:>+8.4f} {rec_mse:>8.5f}")

    # --- Per-seed IC(1) comparison --------------------------------------------
    print(f"\n  Per-seed IC(1) (averaged across assets):")
    snap_ic1s = []
    for i, snap_model in enumerate(model.models):
        snap_model.eval()
        revin_i = model._revins[i] if i < len(model._revins) else None
        if revin_i is not None:
            revin_i = revin_i.to(DEVICE)

        per_asset_ic1 = []
        for feats, targets, asset_idx, _ in segments:
            ic1 = compute_single_snapshot_ic1(snap_model, revin_i, feats, targets, asset_idx)
            per_asset_ic1.append(ic1)
        mean_ic1 = float(np.mean(per_asset_ic1))
        snap_ic1s.append(mean_ic1)
        print(f"    Seed {i:2d}: IC(1) = {mean_ic1:+.4f}")

    best_snap = int(np.argmax(snap_ic1s))
    print(f"  Best seed: {best_snap} (IC(1)={snap_ic1s[best_snap]:+.4f})")

    # --- Ensemble average IC -------------------------------------------------
    all_horizon_ics = []
    for h in REWARD_HORIZONS:
        h_ics = [r["ics"][h] for r in asset_results.values()]
        mean_h = float(np.mean(h_ics))
        all_horizon_ics.append(mean_h)
        print(f"  Mean IC(t+{h:<2}): {mean_h:+.4f}  "
              f"[range: {min(h_ics):+.4f} to {max(h_ics):+.4f}]")

    mean_ic = float(np.mean(all_horizon_ics))
    avg_rec_mse = float(np.mean([r["rec_mse"] for r in asset_results.values()]))
    print(f"\n  Mean IC across all horizons: {mean_ic:+.4f}")
    print(f"  Avg Rec MSE:                 {avg_rec_mse:.5f}")

    # --- Shuffled IC ---------------------------------------------------------
    print(f"\n  Computing global shuffled IC (anti-memorization)...")
    shuffled_ic = compute_global_shuffled_ic(model, segments)
    # Use h1 IC as denominator (ShIC is h1-only, so ratio must be h1 vs h1)
    shic_denom = ic_h1 if ic_h1 is not None else mean_ic
    shic_ratio = shuffled_ic / shic_denom if abs(shic_denom) > 1e-6 else 0.0
    print(f"    Contiguous IC:  {mean_ic:+.4f}")
    print(f"    Shuffled IC:    {shuffled_ic:+.4f}")
    print(f"    Ratio:          {shic_ratio:.3f}  (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")

    # --- Gate checks ---------------------------------------------------------
    ic_h1 = float(np.mean([r["ics"][1] for r in asset_results.values()]))
    all_pass = check_gates(avg_rec_mse, mean_ic, shuffled_ic, ic_h1=ic_h1)

    # --- Save results --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "version": "v3e",
        "variant": "seed_ensemble",
        "timestamp": datetime.now().isoformat(),
        "n_seeds": model.n_models,
        "gate_passed": all_pass,
        "mean_ic": mean_ic,
        "avg_rec_mse": avg_rec_mse,
        "shuffled_ic": shuffled_ic,
        "shic_ratio": shic_ratio,
        "per_asset": {
            name: {"rec_mse": r["rec_mse"],
                   "ics": {str(h): r["ics"][h] for h in REWARD_HORIZONS}}
            for name, r in asset_results.items()
        },
        "per_seed_ic1": snap_ic1s,
    }
    out_path = LOG_DIR / f"validate_snapshot_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result_data, fh, indent=2)
    print(f"\n  Results saved to: {out_path.name}")

    return all_pass


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="V3.E Snapshot Ensemble Validation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to directory containing snapshot checkpoints")
    args = parser.parse_args()

    passed = run_validation(model_path=args.model)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
