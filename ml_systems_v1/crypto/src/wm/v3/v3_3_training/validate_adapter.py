"""
V3.X Adapter Validation -- Compares adapted vs base model

Usage:
    python validate_adapter.py               # Best adapter + best base
    python validate_adapter.py --model X     # Specific adapter checkpoint
    python validate_adapter.py --base-model X  # Specific base checkpoint
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
from world_model import WaveNetGRUWorldModel, count_parameters
from adapter import AdaptiveResidualAdapter, AdaptedWorldModel
from revin import RevIN
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
# MODEL LOADING
# ===============================================================================

def load_base_model(model_path=None):
    """Load WaveNetGRUWorldModel + RevIN from checkpoint."""
    if model_path is None:
        model_path = BASE_MODEL_DIR / "v3_3_wm_best_ema.pt"
    if not model_path.exists():
        model_path = BASE_MODEL_DIR / "v3_3_wm_latest.pt"
    if not model_path.exists():
        print(f"  [FAIL] No base checkpoint found in {BASE_MODEL_DIR}")
        sys.exit(1)

    print(f"  Base model: {model_path.name}")
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = WaveNetGRUWorldModel().to(DEVICE)
    revin = RevIN(num_features=INPUT_DIM).to(DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch", "?")
        print(f"  Base checkpoint epoch: {epoch}")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("wavenet") for k in ckpt):
        state_dict = ckpt
    else:
        state_dict = ckpt

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print(f"  [WARN] Architecture mismatch, doing filtered load...")
            model_state = model.state_dict()
            loaded, skipped = 0, []
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded += 1
                else:
                    skipped.append(k)
            model.load_state_dict(model_state)
            print(f"  [WARN] Loaded {loaded}/{loaded + len(skipped)} keys")
        else:
            raise

    if isinstance(ckpt, dict) and "revin_state_dict" in ckpt:
        revin.load_state_dict(ckpt["revin_state_dict"])
        print(f"  [OK] RevIN loaded from checkpoint")

    model.eval()
    revin.eval()
    print(f"  Base parameters: {count_parameters(model):,}")
    return model, revin


def load_adapter_model(base_model, adapter_path=None):
    """Load AdaptiveResidualAdapter + wrap in AdaptedWorldModel."""
    if adapter_path is None:
        # Glob for adapter checkpoints (train saves as v3_3_{feat_tag}_adapter_*.pt)
        best_candidates = sorted(ADAPTER_MODEL_DIR.glob("v3_3_*_adapter_best.pt"))
        latest_candidates = sorted(ADAPTER_MODEL_DIR.glob("v3_3_*_adapter_latest.pt"))
        if best_candidates:
            adapter_path = best_candidates[-1]
        elif latest_candidates:
            adapter_path = latest_candidates[-1]

    if adapter_path is None or not adapter_path.exists():
        print(f"  [FAIL] No adapter checkpoint found in {ADAPTER_MODEL_DIR}")
        sys.exit(1)

    print(f"  Adapter: {adapter_path.name}")
    ckpt = torch.load(adapter_path, map_location=DEVICE, weights_only=False)

    adapter = AdaptiveResidualAdapter().to(DEVICE)
    if isinstance(ckpt, dict) and "adapter_state_dict" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Adapter checkpoint epoch: {epoch}")
    else:
        adapter.load_state_dict(ckpt)

    adapter.eval()
    adapted_model = AdaptedWorldModel(base_model, adapter).to(DEVICE)
    adapted_model.eval()
    return adapted_model


# ===============================================================================
# IC COMPUTATION
# ===============================================================================

@torch.no_grad()
def compute_asset_ic(model, revin, feats, targets, asset_idx, use_revin=True):
    """
    Compute IC per horizon for one asset using non-overlapping windows.

    revin: RevIN instance (shared between base and adapted models).
    use_revin=True: apply RevIN normalization before calling model.forward_train.
    AdaptedWorldModel expects normalized obs (RevIN applied before forward_train).

    V3 WaveNet-GRU: outputs contain h_seq, z_post (RSSM-style latents).
    bucketer is accessed via model.bucketer (delegated in AdaptedWorldModel).
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

        if use_revin and revin is not None:
            obs = revin(obs, mode='norm')

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
def compute_global_shuffled_ic(model, revin, segments, n_seeds=5, batch_size=32):
    """IC on globally-shuffled data (anti-memorization gate)."""
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
                if revin is not None:
                    obs = revin(obs, mode='norm')
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
    print(f"  VALIDATION GATES (ADAPTED MODEL)")
    print(f"  {'='*60}")

    # Use h1 IC as denominator (ShIC is h1-only, so ratio must be h1 vs h1)
    shic_denom = ic_h1 if ic_h1 is not None else mean_ic
    shic_ratio = shuffled_ic / shic_denom if abs(shic_denom) > 1e-6 else 0.0

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

def run_validation(base_model_path=None, adapter_path=None):
    print(f"\n{'='*65}")
    print(f"  V3.X ADAPTER VALIDATION (vs Base)")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*65}")

    # --- Load models ---------------------------------------------------------
    print(f"\n  Loading base model...")
    base_model, revin = load_base_model(
        Path(base_model_path) if base_model_path else None
    )

    print(f"\n  Loading adapter...")
    adapted_model = load_adapter_model(
        base_model,
        Path(adapter_path) if adapter_path else None
    )

    # --- Load validation data ------------------------------------------------
    print(f"\n  Loading validation data from {DATA_DIR}")
    segments = load_validation_data()
    if not segments:
        print("  [FAIL] No validation data found.")
        sys.exit(1)

    # --- Base IC per asset ---------------------------------------------------
    print(f"\n  Computing base model IC...")
    base_results = {}
    for feats, targets, asset_idx, asset_name in tqdm(segments, desc="  Base IC", leave=True):
        ics = compute_asset_ic(base_model, revin, feats, targets, asset_idx, use_revin=True)
        base_results[asset_name] = ics

    # --- Adapted IC per asset ------------------------------------------------
    print(f"  Computing adapted model IC...")
    adapted_results = {}
    for feats, targets, asset_idx, asset_name in tqdm(segments, desc="  Adapted IC", leave=True):
        # Apply RevIN before forward_train -- AdaptedWorldModel expects normalized obs.
        # RevIN is shared with base model; both see identical normalisation.
        ics = compute_asset_ic(adapted_model, revin, feats, targets, asset_idx, use_revin=True)
        adapted_results[asset_name] = ics

    # --- Side-by-side comparison table ---------------------------------------
    print(f"\n  {'='*65}")
    print(f"  IC COMPARISON: BASE vs ADAPTED (delta = adapted - base)")
    print(f"  {'='*65}")

    for h in REWARD_HORIZONS:
        print(f"\n  Horizon t+{h}:")
        print(f"  {'Asset':<10} {'Base':>8} {'Adapted':>8} {'Delta':>8}")
        print(f"  {'-'*40}")
        deltas = []
        for asset_name in sorted(base_results.keys()):
            base_ic = base_results[asset_name][h]
            adapted_ic = adapted_results[asset_name][h]
            delta = adapted_ic - base_ic
            deltas.append(delta)
            arrow = "+" if delta > 0.001 else ("-" if delta < -0.001 else " ")
            print(f"  {asset_name:<10} {base_ic:>+8.4f} {adapted_ic:>+8.4f} "
                  f"{delta:>+8.4f} {arrow}")
        mean_delta = float(np.mean(deltas))
        print(f"  {'Mean':<10} "
              f"{np.mean([base_results[a][h] for a in base_results]):>+8.4f} "
              f"{np.mean([adapted_results[a][h] for a in adapted_results]):>+8.4f} "
              f"{mean_delta:>+8.4f}")

    # --- Aggregate summary ---------------------------------------------------
    base_horizon_ics = []
    adapted_horizon_ics = []
    for h in REWARD_HORIZONS:
        base_h = float(np.mean([base_results[a][h] for a in base_results]))
        adapted_h = float(np.mean([adapted_results[a][h] for a in adapted_results]))
        base_horizon_ics.append(base_h)
        adapted_horizon_ics.append(adapted_h)

    base_mean_ic = float(np.mean(base_horizon_ics))
    adapted_mean_ic = float(np.mean(adapted_horizon_ics))
    overall_delta = adapted_mean_ic - base_mean_ic

    print(f"\n  Mean IC (all horizons, all assets):")
    print(f"    Base:    {base_mean_ic:+.4f}")
    print(f"    Adapted: {adapted_mean_ic:+.4f}")
    print(f"    Delta:   {overall_delta:+.4f}")

    # --- Reconstruction MSE (adapted) ----------------------------------------
    # V3 WaveNet-GRU: recon head maps z_post -> input features.
    # Compare out["recon"] vs obs_norm (RevIN-normalized input).
    rec_mses = []
    for feats, targets, asset_idx, asset_name in segments:
        obs_np = feats[:WM_SEQ_LEN]
        obs = torch.from_numpy(obs_np).unsqueeze(0).float().to(DEVICE)
        obs_norm = revin(obs, mode='norm')
        asset_t = torch.tensor([asset_idx], dtype=torch.long, device=DEVICE)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE == "cuda"):
            out = adapted_model.forward_train(obs_norm, asset_t)
        rec_mse = F.mse_loss(out["recon"], obs_norm).item()
        rec_mses.append(rec_mse)
    avg_rec_mse = float(np.mean(rec_mses))
    print(f"    Rec MSE (adapted): {avg_rec_mse:.5f}")

    # --- Shuffled IC (adapted model) -----------------------------------------
    print(f"\n  Computing global shuffled IC (adapted model)...")
    shuffled_ic = compute_global_shuffled_ic(adapted_model, revin, segments)
    adapted_ic_h1 = float(np.mean([adapted_results[a][1] for a in adapted_results]))
    shic_denom_inline = adapted_ic_h1 if adapted_ic_h1 is not None else adapted_mean_ic
    shic_ratio = shuffled_ic / shic_denom_inline if abs(shic_denom_inline) > 1e-6 else 0.0
    print(f"    Contiguous IC:  {adapted_mean_ic:+.4f}")
    print(f"    Shuffled IC:    {shuffled_ic:+.4f}")
    print(f"    Ratio:          {shic_ratio:.3f}  (gate: > {GATE_SHUFFLED_IC_RATIO_MIN})")

    # --- Gate checks ---------------------------------------------------------
    adapted_ic_h1 = float(np.mean([adapted_results[a][1] for a in adapted_results]))
    all_pass = check_gates(avg_rec_mse, adapted_mean_ic, shuffled_ic, ic_h1=adapted_ic_h1)

    # --- Save results --------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_data = {
        "version": "v3x",
        "variant": "adapter",
        "timestamp": datetime.now().isoformat(),
        "gate_passed": all_pass,
        "base_mean_ic": base_mean_ic,
        "adapted_mean_ic": adapted_mean_ic,
        "ic_delta": overall_delta,
        "avg_rec_mse": avg_rec_mse,
        "shuffled_ic": shuffled_ic,
        "shic_ratio": shic_ratio,
        "per_asset_base": {
            name: {str(h): base_results[name][h] for h in REWARD_HORIZONS}
            for name in base_results
        },
        "per_asset_adapted": {
            name: {str(h): adapted_results[name][h] for h in REWARD_HORIZONS}
            for name in adapted_results
        },
    }
    out_path = LOG_DIR / f"validate_adapter_{timestamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result_data, fh, indent=2)
    print(f"\n  Results saved to: {out_path.name}")

    return all_pass


# ===============================================================================
# ENTRY POINT
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(description="V3.X Adapter Validation vs Base")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific adapter checkpoint (.pt)")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Path to specific base checkpoint (default: best EMA)")
    args = parser.parse_args()

    passed = run_validation(
        base_model_path=args.base_model,
        adapter_path=args.model,
    )
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
