"""
V4.D NCL Diversity Validation -- Per-head IC + ensemble IC + diversity metrics

Validates the V4.3.D DiversityWorldModel checkpoint from NCL_MODEL_DIR.
Reports ensemble IC, per-head IC, head correlation matrix, and head std diversity.
Gates on IC > GATE_IC_MIN and shuffled IC ratio > GATE_SHUFFLED_IC_RATIO_MIN.

Usage:
    python validate_ncl.py                  # Best NCL checkpoint
    python validate_ncl.py --model path.pt  # Specific checkpoint
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from settings import *
from ncl_model import DiversityWorldModel
from revin import RevIN
from pipeline.data_integrity import selective_drop_nulls, extract_features_targets

PURGE_GAP_BARS = 400


# =============================================================================
# DATA LOADING
# =============================================================================

def load_validation_data():
    """Load last 10% of each asset's chimera file, with 400-bar purge gap."""
    files = sorted(DATA_DIR.glob("*_v51_chimera*.parquet"))
    segments = []
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


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path=None):
    """Load DiversityWorldModel + RevIN from NCL checkpoint."""
    model = DiversityWorldModel().to(DEVICE)
    revin = RevIN(num_features=INPUT_DIM).to(DEVICE)

    if model_path is None:
        ckpt_path = NCL_MODEL_DIR / "v4_1d_ncl_best_ema.pt"
        if not ckpt_path.exists():
            ckpt_path = NCL_MODEL_DIR / "v4_1d_wm_best_ema.pt"
        if not ckpt_path.exists():
            ckpt_path = NCL_MODEL_DIR / "v4_1d_ncl_latest.pt"
        if not ckpt_path.exists():
            ckpt_path = NCL_MODEL_DIR / "v4_1d_wm_latest.pt"
    else:
        ckpt_path = Path(model_path)

    if not ckpt_path.exists():
        print(f"  [FAIL] No NCL checkpoint found: {ckpt_path}")
        sys.exit(1)

    print(f"  Loading: {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}")
        elif "ema_state_dict" in ckpt:
            model.load_state_dict(ckpt["ema_state_dict"])
        else:
            model.load_state_dict(ckpt)
        if "revin_state_dict" in ckpt:
            revin.load_state_dict(ckpt["revin_state_dict"])
            print(f"  [OK] RevIN state loaded from checkpoint")
    else:
        model.load_state_dict(ckpt)

    model.eval()
    revin.eval()
    K = model.n_diversity_heads
    print(f"  [OK] V4.D loaded  |  heads={K}  |  NCL lambda={model.ncl_lambda}")
    return model, revin, ckpt_path.stem


# =============================================================================
# IC COMPUTATION
# =============================================================================

def compute_ic(preds, reals):
    """Pearson IC with finite-value guard. Returns 0.0 if insufficient data."""
    mask = np.isfinite(preds) & np.isfinite(reals)
    p, r = preds[mask], reals[mask]
    if len(p) < 50:
        return 0.0
    if np.std(p) < 1e-10 or np.std(r) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, r)[0, 1])


# =============================================================================
# MAIN EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(model, revin, segments):
    """
    Evaluate ensemble IC, per-head IC, and diversity metrics across all assets.

    Returns dict with per-asset and aggregate results.
    """
    K = model.n_diversity_heads
    seq_len = WM_SEQ_LEN

    ensemble_preds = {h: [] for h in REWARD_HORIZONS}
    ensemble_reals = {h: [] for h in REWARD_HORIZONS}
    head_preds = {k: {h: [] for h in REWARD_HORIZONS} for k in range(K)}

    asset_results = {}

    for feats, targets, asset_idx, asset_name in tqdm(segments, desc="  Assets", leave=True):
        indices = list(range(0, len(feats) - seq_len, seq_len))
        if not indices:
            continue

        a_ens_preds = {h: [] for h in REWARD_HORIZONS}
        a_ens_reals = {h: [] for h in REWARD_HORIZONS}
        a_head_preds = {k: {h: [] for h in REWARD_HORIZONS} for k in range(K)}

        for i in indices:
            obs_np = feats[i:i + seq_len]
            obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
            asset = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)

            obs = revin(obs, mode='norm')
            with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                outputs = model.forward_train(obs, asset)

            for h in REWARD_HORIZONS:
                ens_logits = outputs["return_logits"][h]
                ens_pred = model.bucketer.decode(ens_logits).cpu().numpy().flatten()
                real_h = targets[h][i:i + seq_len]
                a_ens_preds[h].append(ens_pred)
                a_ens_reals[h].append(real_h)
                ensemble_preds[h].append(ens_pred)
                ensemble_reals[h].append(real_h)

            all_rl = outputs["all_return_logits"]  # list of K dicts {h: [1,T,bins]}
            for k in range(K):
                for h in REWARD_HORIZONS:
                    head_logits = all_rl[k][h]
                    head_pred = model.bucketer.decode(head_logits).cpu().numpy().flatten()
                    a_head_preds[k][h].append(head_pred)
                    head_preds[k][h].append(head_pred)

        asset_ens_ic = {}
        for h in REWARD_HORIZONS:
            p = np.concatenate(a_ens_preds[h])
            r = np.concatenate(a_ens_reals[h])
            asset_ens_ic[h] = compute_ic(p, r)

        asset_head_ic = {}
        for k in range(K):
            asset_head_ic[k] = {}
            for h in REWARD_HORIZONS:
                p = np.concatenate(a_head_preds[k][h])
                r = np.concatenate(a_ens_reals[h])
                asset_head_ic[k][h] = compute_ic(p, r)

        asset_results[asset_name] = {
            "ensemble_ic": asset_ens_ic,
            "head_ic": asset_head_ic,
        }

        mean_ens_ic = np.mean(list(asset_ens_ic.values()))
        print(f"    {asset_name:<10}  ens_IC={mean_ens_ic:+.4f}  "
              f"per-head(h=1): " +
              "  ".join(f"k{k}={asset_head_ic[k][1]:+.4f}" for k in range(K)))

    return asset_results, ensemble_preds, ensemble_reals, head_preds


@torch.no_grad()
def compute_global_shuffled_ic(model, revin, segments, n_seeds=5, batch_size=64):
    """IC on globally shuffled data. Tests feature->return signal vs temporal memorization."""
    all_ics = []
    for seed_offset in range(n_seeds):
        rng = np.random.default_rng(42 + seed_offset * 1000)
        all_preds, all_reals = [], []
        seq_len = WM_SEQ_LEN

        for feats, targets, asset_idx, _ in segments:
            n = len(feats)
            if n < seq_len * 2:
                continue
            idx = np.arange(n)
            rng.shuffle(idx)
            shuf_feats = feats[idx]
            shuf_reals = targets[1][idx]
            window_starts = list(range(0, n - seq_len, seq_len))

            for bs in range(0, len(window_starts), batch_size):
                batch_ws = window_starts[bs:bs + batch_size]
                obs_list = [shuf_feats[i:i + seq_len] for i in batch_ws]
                real_list = [shuf_reals[i:i + seq_len] for i in batch_ws]
                obs = torch.from_numpy(np.stack(obs_list)).float().to(DEVICE)
                asset = torch.full((len(obs_list),), asset_idx, dtype=torch.long, device=DEVICE)
                obs = revin(obs, mode='norm')
                with torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
                    out = model.forward_train(obs, asset)
                preds = model.bucketer.decode(out["return_logits"][1]).cpu().numpy()
                for b, real in enumerate(real_list):
                    all_preds.extend(preds[b].flatten())
                    all_reals.extend(real)

        p_arr, r_arr = np.array(all_preds), np.array(all_reals)
        mask = np.isfinite(p_arr) & np.isfinite(r_arr)
        if mask.sum() > 50:
            ic = float(np.corrcoef(p_arr[mask], r_arr[mask])[0, 1])
            if np.isfinite(ic):
                all_ics.append(ic)

    return float(np.mean(all_ics)) if all_ics else 0.0


def compute_diversity_metrics(head_preds, ensemble_reals, K):
    """
    Compute pairwise head correlation matrix and per-timestep head std for horizon=1.

    Returns:
        avg_off_diag: average off-diagonal Pearson correlation (lower = more diverse)
        mean_head_std: mean std across heads per timestep (higher = more diverse)
    """
    preds_by_head = []
    for k in range(K):
        p = np.concatenate(head_preds[k][1])
        preds_by_head.append(p)

    min_len = min(len(p) for p in preds_by_head)
    preds_by_head = [p[:min_len] for p in preds_by_head]

    stacked = np.stack(preds_by_head)  # [K, N]

    corr_matrix = np.corrcoef(stacked)  # [K, K]
    off_diag_sum = corr_matrix.sum() - np.trace(corr_matrix)
    avg_off_diag = off_diag_sum / (K * (K - 1)) if K > 1 else 0.0

    mean_head_std = float(np.mean(np.std(stacked, axis=0)))

    return corr_matrix, float(avg_off_diag), mean_head_std


# =============================================================================
# REPORTING + GATES
# =============================================================================

def check_gates(mean_ic, shuffled_ic, ic_h1=None):
    """Check validation gates. Returns (all_pass, gate_results dict)."""
    gates = {}

    ic_pass = mean_ic > GATE_IC_MIN
    gates["Mean IC"] = (ic_pass, f"{mean_ic:+.4f} > {GATE_IC_MIN}")

    shic_denom = ic_h1 if ic_h1 is not None else mean_ic
    if abs(shic_denom) > 1e-6:
        ratio = shuffled_ic / shic_denom
        shic_pass = ratio > GATE_SHUFFLED_IC_RATIO_MIN
        gates["Shuffled IC Ratio (h1)"] = (shic_pass, f"{ratio:.3f} > {GATE_SHUFFLED_IC_RATIO_MIN}")
    else:
        gates["Shuffled IC Ratio (h1)"] = (False, "h1 IC near zero")

    all_pass = all(v[0] for v in gates.values())
    return all_pass, gates


def print_report(asset_results, ensemble_preds, ensemble_reals, head_preds,
                 shuffled_ic, model_name, K):
    print(f"\n{'='*70}")
    print(f"  V4.D NCL DIVERSITY VALIDATION")
    print(f"  Model: {model_name}")
    print(f"  Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Heads: {K}  |  Horizons: {REWARD_HORIZONS}")
    print(f"{'='*70}")

    print(f"\n  Per-Asset Ensemble IC:")
    print(f"    {'Asset':<10} " + "  ".join(f"{'IC('+str(h)+')':>9}" for h in REWARD_HORIZONS))
    print(f"    {'-'*60}")
    for name, r in asset_results.items():
        row = f"    {name:<10} "
        row += "  ".join(f"{r['ensemble_ic'][h]:>+9.4f}" for h in REWARD_HORIZONS)
        print(row)

    print(f"\n  Global Ensemble IC (all assets concatenated):")
    global_ics = {}
    for h in REWARD_HORIZONS:
        p = np.concatenate(ensemble_preds[h])
        r = np.concatenate(ensemble_reals[h])
        ic = compute_ic(p, r)
        global_ics[h] = ic
        print(f"    t+{h:<3}  IC = {ic:+.4f}")
    mean_ic = float(np.mean(list(global_ics.values())))
    print(f"    Mean IC across horizons: {mean_ic:+.4f}")

    print(f"\n  Per-Head IC at Horizon=1 (global):")
    head_ics_h1 = []
    for k in range(K):
        p = np.concatenate(head_preds[k][1])
        r = np.concatenate(ensemble_reals[1])
        min_len = min(len(p), len(r))
        ic = compute_ic(p[:min_len], r[:min_len])
        head_ics_h1.append(ic)
        best_marker = " <-- best" if ic == max(head_ics_h1) and k == K - 1 else ""
        print(f"    Head {k}: IC = {ic:+.4f}{best_marker}")
    best_head = int(np.argmax(head_ics_h1))
    print(f"    Best head: k={best_head}  IC={head_ics_h1[best_head]:+.4f}")

    print(f"\n  Per-Head IC (all horizons, global):")
    print(f"    {'Head':<6} " + "  ".join(f"{'IC('+str(h)+')':>9}" for h in REWARD_HORIZONS))
    print(f"    {'-'*55}")
    for k in range(K):
        row = f"    k={k:<4} "
        for h in REWARD_HORIZONS:
            p = np.concatenate(head_preds[k][h])
            r = np.concatenate(ensemble_reals[h])
            min_len = min(len(p), len(r))
            ic = compute_ic(p[:min_len], r[:min_len])
            row += f"  {ic:>+9.4f}"
        print(row)

    corr_matrix, avg_off_diag, mean_head_std = compute_diversity_metrics(
        head_preds, ensemble_reals, K
    )
    print(f"\n  Head Diversity Metrics (Horizon=1):")
    print(f"    Avg pairwise correlation: {avg_off_diag:+.4f}  (lower = more diverse)")
    print(f"    Mean per-timestep std:    {mean_head_std:.4f}  (higher = more diverse)")
    print(f"\n  Head Correlation Matrix (Horizon=1):")
    header = "         " + "".join(f"  k{k:>2}" for k in range(K))
    print(f"    {header}")
    for i in range(K):
        row = f"    k{i}:    " + "".join(f"  {corr_matrix[i, j]:>5.3f}" for j in range(K))
        print(row)

    print(f"\n  Anti-Memorization (Global Shuffled IC):")
    print(f"    Contiguous IC:   {mean_ic:+.4f}")
    print(f"    Shuffled IC:     {shuffled_ic:+.4f}")
    if abs(mean_ic) > 1e-6:
        ratio = shuffled_ic / mean_ic
        print(f"    Ratio:           {ratio:.3f}  (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")
    else:
        print(f"    Ratio:           N/A (contiguous IC near zero)")

    ic_h1 = global_ics.get(1, mean_ic)
    all_pass, gates = check_gates(mean_ic, shuffled_ic, ic_h1=ic_h1)
    print(f"\n{'='*70}")
    print(f"  VALIDATION GATES")
    print(f"{'='*70}")
    for gate_name, (passed, desc) in gates.items():
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] {gate_name:<25} {desc}")

    print(f"\n  {'='*40}")
    if all_pass:
        print(f"  VERDICT: ALL GATES PASSED")
        print(f"  V4.D NCL model is validated.")
    else:
        print(f"  VERDICT: GATE(S) FAILED")
        print(f"  V4.D NCL model needs more training or tuning.")
    print(f"  {'='*40}")

    return mean_ic, all_pass, gates


def save_results(asset_results, global_ics, shuffled_ic, head_ics_h1,
                 avg_off_diag, mean_head_std, gate_pass, model_name):
    """Save NCL validation results to JSON."""
    output = {
        "version": "v4d",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "gate_passed": gate_pass,
        "global_ensemble_ic": {str(h): global_ics[h] for h in REWARD_HORIZONS},
        "mean_ic": float(np.mean(list(global_ics.values()))),
        "shuffled_ic": shuffled_ic,
        "per_head_ic_h1": {str(k): float(ic) for k, ic in enumerate(head_ics_h1)},
        "diversity": {
            "avg_pairwise_corr": avg_off_diag,
            "mean_head_std_h1": mean_head_std,
        },
        "per_asset": {
            name: {
                "ensemble_ic": {str(h): r["ensemble_ic"][h] for h in REWARD_HORIZONS},
            }
            for name, r in asset_results.items()
        },
    }
    out_path = LOG_DIR / f"ncl_validation_{model_name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {out_path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="V4.D NCL Diversity Validation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific NCL checkpoint")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  V4.D NCL DIVERSITY VALIDATOR")
    print(f"  NCL_MODEL_DIR: {NCL_MODEL_DIR}")
    print(f"  DEVICE: {DEVICE}")
    print(f"{'='*70}\n")

    model, revin, model_name = load_model(args.model)
    K = model.n_diversity_heads

    print(f"\n  Loading validation data from {DATA_DIR}")
    segments = load_validation_data()
    if not segments:
        print("  [FAIL] No validation data found.")
        sys.exit(1)

    print(f"\n  Evaluating {K} heads across {len(segments)} assets...")
    asset_results, ensemble_preds, ensemble_reals, head_preds = evaluate(
        model, revin, segments
    )

    global_ics = {}
    for h in REWARD_HORIZONS:
        p = np.concatenate(ensemble_preds[h])
        r = np.concatenate(ensemble_reals[h])
        global_ics[h] = compute_ic(p, r)
    mean_ic = float(np.mean(list(global_ics.values())))

    head_ics_h1 = []
    for k in range(K):
        p = np.concatenate(head_preds[k][1])
        r = np.concatenate(ensemble_reals[1])
        min_len = min(len(p), len(r))
        head_ics_h1.append(compute_ic(p[:min_len], r[:min_len]))

    corr_matrix, avg_off_diag, mean_head_std = compute_diversity_metrics(
        head_preds, ensemble_reals, K
    )

    print(f"\n  Computing global shuffled IC...")
    shuffled_ic = compute_global_shuffled_ic(model, revin, segments)

    mean_ic, all_pass, gates = print_report(
        asset_results, ensemble_preds, ensemble_reals, head_preds,
        shuffled_ic, model_name, K
    )

    save_results(
        asset_results, global_ics, shuffled_ic, head_ics_h1,
        avg_off_diag, mean_head_std, all_pass, model_name
    )

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
