"""Linear-probe IC evaluator — first SOTA validation gate.

Two IC readings on the held-out OOS segment (the 30% suffix per asset
that the dataset reserved with train_frac=0.7):

  intrinsic_ic
      Decode the foundation model's TwoHot return logits at h=1 directly
      to an expected return; correlate with target_return_1. This is the
      model's RAW predictive power without any downstream tuning.

  linear_probe_ic
      Fit a 1-layer linear regression on h_seq[:,-1,:] (frozen) -> target,
      train half / eval half (split the OOS segment). Standard linear
      probe per Chronos / MOMENT eval protocol.

Comparison anchor: V1.0 baseline IC = 0.066 (per CLAUDE.md scoresheet).

Usage:
    python src/frontier_ml/foundation/eval_probe.py --universe u10 \
        --ckpt models/frontier_ml/foundation/latest.pt --n-windows 2000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.stats import spearmanr

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from frontier_ml.foundation.harmony import apply_harmony  # noqa: E402
apply_harmony(verbose=False)

from frontier_ml.foundation.backbone import FoundationBackbone, DEFAULT_CONFIG  # noqa: E402
from frontier_ml.foundation.data_loader import FoundationDataset  # noqa: E402
from frontier_ml.foundation.objectives import make_bucketer  # noqa: E402


def sample_oos_windows(
    ds: FoundationDataset,
    n_windows: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample random windows from the OOS suffix (30% per asset).

    OOS = bars [n_train_bars[a], n_bars[a]) for each asset.

    Returns x (N,S,F), asset_ids (N,), target_h1 (N,).
    """
    rng = np.random.default_rng(seed)
    S = ds.seq_len
    F = ds.n_features
    max_h = max(ds.horizons)
    n_assets = len(ds.asset_ids)

    x = np.empty((n_windows, S, F), dtype=np.float32)
    asset_ids = np.empty(n_windows, dtype=np.int64)
    target_h1 = np.empty(n_windows, dtype=np.float32)
    h1_idx = list(ds.horizons).index(1)

    for i in range(n_windows):
        # Draw asset uniformly; require enough OOS room
        for _ in range(20):
            a = int(rng.integers(0, n_assets))
            n_total = ds.n_bars[a]
            n_train = ds.n_train_bars[a]
            oos_lo = n_train
            oos_hi = n_total - S - max_h
            if oos_hi - oos_lo > 100:
                break
        if oos_hi - oos_lo <= 100:
            # All assets too small -- fall back to whatever
            a = 0
            oos_lo = ds.n_train_bars[0]
            oos_hi = ds.n_bars[0] - S - max_h
        start = int(rng.integers(oos_lo, oos_hi))
        x[i] = ds.features_arr[a][start:start + S].astype(np.float32)
        asset_ids[i] = a
        target_h1[i] = ds.targets_arr[a][start + S - 1, h1_idx]

    return x, asset_ids, target_h1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="u10")
    ap.add_argument("--ckpt", default=str(_PROJECT_ROOT / "models" / "frontier_ml" / "foundation" / "latest.pt"))
    ap.add_argument("--n-windows", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70, flush=True)
    print(f"FRONTIER FOUNDATION LINEAR-PROBE EVAL", flush=True)
    print(f"  ckpt:     {args.ckpt}", flush=True)
    print(f"  universe: {args.universe}", flush=True)
    print(f"  device:   {device}", flush=True)
    print("=" * 70, flush=True)

    # Load dataset (uses cached slim arrays; ~5s)
    ds = FoundationDataset(universe=args.universe, seq_len=args.seq_len, seed=args.seed)
    n_features = ds.n_features

    # Load model
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = state.get("config", DEFAULT_CONFIG)
    model = FoundationBackbone(n_features=n_features, config=cfg).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print(f"[eval] loaded ckpt at step {state.get('step', '?')}; "
          f"params {model.num_params():,}", flush=True)

    bucketer = make_bucketer(num_bins=cfg["num_bins"], device=device)

    # Sample OOS windows
    print(f"[eval] sampling {args.n_windows} OOS windows (suffix 30% per asset)", flush=True)
    x_np, asset_ids_np, tgt_h1 = sample_oos_windows(ds, args.n_windows, seed=args.seed)

    # Forward in batches
    print(f"[eval] forwarding @ batch={args.batch_size}", flush=True)
    pred_returns = np.empty(args.n_windows, dtype=np.float32)
    h_pool = np.empty((args.n_windows, cfg["d_model"]), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, args.n_windows, args.batch_size):
            xb = torch.from_numpy(x_np[i:i + args.batch_size]).to(device, non_blocking=True)
            ab = torch.from_numpy(asset_ids_np[i:i + args.batch_size]).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(xb, asset_ids=ab)
            # Decode h1 logits -> expected return via TwoHot bucketer
            logits_h1 = out["return_logits"]["h1"].float()
            decoded = bucketer.decode(logits_h1).cpu().numpy()
            pred_returns[i:i + args.batch_size] = decoded[:, 0] if decoded.ndim > 1 else decoded
            # Pool h_seq mean for linear-probe features
            h_pool[i:i + args.batch_size] = out["h_seq"].mean(dim=1).float().cpu().numpy()
    elapsed = time.time() - t0
    print(f"[eval] forward done in {elapsed:.1f}s ({args.n_windows / elapsed:.0f} win/s)",
          flush=True)

    # ---- Intrinsic IC -----------------------------------------------------
    rho_intrinsic, p_intrinsic = spearmanr(pred_returns, tgt_h1)

    # ---- Linear-probe IC --------------------------------------------------
    # Train half / eval half (random split). Standard MOMENT/Chronos protocol.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(args.n_windows)
    half = args.n_windows // 2
    train_idx = perm[:half]
    eval_idx = perm[half:]

    # Closed-form OLS
    X_tr = h_pool[train_idx]
    y_tr = tgt_h1[train_idx]
    X_ev = h_pool[eval_idx]
    y_ev = tgt_h1[eval_idx]
    # Add bias column
    X_tr_b = np.concatenate([X_tr, np.ones((len(X_tr), 1), dtype=np.float32)], axis=1)
    X_ev_b = np.concatenate([X_ev, np.ones((len(X_ev), 1), dtype=np.float32)], axis=1)
    # Regularized solve to avoid singular X^TX
    lam = 1.0
    A = X_tr_b.T @ X_tr_b + lam * np.eye(X_tr_b.shape[1], dtype=np.float32)
    b = X_tr_b.T @ y_tr
    w = np.linalg.solve(A.astype(np.float64), b.astype(np.float64)).astype(np.float32)
    y_pred = X_ev_b @ w
    rho_probe, p_probe = spearmanr(y_pred, y_ev)

    # ---- Shuffled IC (sanity) --------------------------------------------
    shuffled = rng.permutation(tgt_h1)
    rho_shuf, _ = spearmanr(pred_returns, shuffled)

    # ---- Report ----------------------------------------------------------
    print("\n" + "=" * 70, flush=True)
    print("RESULTS  (Spearman rank-IC at h=1)", flush=True)
    print("=" * 70, flush=True)
    print(f"  intrinsic IC      = {rho_intrinsic:+.4f}  (p = {p_intrinsic:.2e})", flush=True)
    print(f"  linear-probe IC   = {rho_probe:+.4f}  (p = {p_probe:.2e})", flush=True)
    print(f"  shuffled-IC ctrl  = {rho_shuf:+.4f}", flush=True)
    print(f"  V1.0 baseline IC  = +0.0660 (per CLAUDE.md scoresheet)", flush=True)
    print(f"  V1.1 record IC    = +0.0674", flush=True)
    print(f"  Headline target   = > +0.10", flush=True)
    print("=" * 70, flush=True)

    # Persist
    out_path = _PROJECT_ROOT / "logs" / "frontier_ml" / f"eval_probe_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump({
            "ckpt": str(args.ckpt),
            "ckpt_step": int(state.get("step", -1)),
            "universe": args.universe,
            "n_windows": args.n_windows,
            "intrinsic_ic": float(rho_intrinsic),
            "intrinsic_p": float(p_intrinsic),
            "linear_probe_ic": float(rho_probe),
            "linear_probe_p": float(p_probe),
            "shuffled_ic": float(rho_shuf),
            "v1_0_baseline_ic": 0.066,
            "headline_target_ic": 0.10,
        }, fp, indent=2)
    print(f"[eval] result written to {out_path}", flush=True)


if __name__ == "__main__":
    main()
