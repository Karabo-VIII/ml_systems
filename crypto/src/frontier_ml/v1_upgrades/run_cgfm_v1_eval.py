"""B007 E3 -- CGFM residual flow on V1.1 best_ema.

Pipeline:
  1. Load V1.1 best_ema (auto-detect input_dim/base_dim).
  2. Forward over VAL slice, capture per-bar (h_state, bin_probs[h=1], point_pred, y).
  3. Split capture 80/20 into CGFM-train / CGFM-eval (within VAL only;
     OOS reserved for final strategy backtest).
  4. Train CGFM on residual = y - point_pred conditioned on
     [h_state || bin_probs || point_pred].
  5. Evaluate on the held-out 20%:
        - CRPS_cgfm vs CRPS_gaussian_baseline (sigma = empirical residual std)
        - IC at h=1: corr(point_pred, y) vs corr(cgfm_mean_sample, y)

Decision gate per B007 RESPONSE §10 E3:
    CRPS lift >= 5%  AND  IC lift >= +0.003  -> ship CGFM as V1.x
    distributional add-on. If CRPS improves but IC flat, ship for sizing
    only, not as primary signal.

VAL slice for gate-checks; OOS reserved per @browser correction.

Usage:
    python -m frontier_ml.v1_upgrades.run_cgfm_v1_eval \
        --asset BTC --max-windows 20000 --cgfm-iters 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "wm" / "v1" / "v1_1_training"))


def _load_model_and_settings(features, ckpt_path):
    """Reuses the auto-detect approach from run_aci_v1_eval.py."""
    from frontier_ml.v1_upgrades.run_aci_v1_eval import (
        _load_model_and_settings as _ld,
    )
    return _ld(features, ckpt_path)


def _build_obs_tensor(df: pl.DataFrame, feat_list, S, asset_idx: int, horizon: int = 1):
    target_col = f"target_return_{horizon}"
    needed = list(feat_list) + [target_col]
    if "regime_label" in df.columns:
        needed.append("regime_label")
    sub = df.select(needed).drop_nulls()
    feats = sub.select(feat_list).to_numpy().astype(np.float32)
    y = sub.get_column(target_col).to_numpy().astype(np.float32)
    regime = (sub.get_column("regime_label").to_numpy().astype(np.int64)
              if "regime_label" in df.columns
              else np.full(len(sub), 1, dtype=np.int64))
    obs = torch.from_numpy(feats).to(S.DEVICE)
    return obs, y, regime, asset_idx


@torch.no_grad()
def _capture_windows(
    model, S, obs: torch.Tensor, asset_idx: int,
    seq_len: int, batch_size: int, max_windows: int | None,
    horizon: int = 1,
):
    """Capture per-window (h_state, bin_probs[h], point_pred[h]) for the last bar of each window."""
    T = obs.shape[0]
    n_total = T - seq_len + 1
    n = min(n_total, max_windows) if max_windows else n_total

    centers_raw = _bin_centers_raw(S.NUM_BINS, S.BIN_MIN, S.BIN_MAX)
    centers_raw_t = torch.tensor(centers_raw, dtype=torch.float32, device=S.DEVICE)

    h_states = np.zeros((n, model.d_model), dtype=np.float32)
    bin_probs = np.zeros((n, S.NUM_BINS), dtype=np.float32)
    point_preds = np.zeros(n, dtype=np.float32)

    asset_t_full = torch.full((batch_size,), asset_idx, dtype=torch.long, device=S.DEVICE)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start
        windows = torch.stack([obs[k:k + seq_len] for k in range(start, end)], dim=0)
        asset_t = asset_t_full[:bs]
        outputs = model.forward_train(windows, asset_t)
        h_seq = outputs["h_seq"][:, -1, :]                                  # (bs, d_model)
        logits = outputs["return_logits"][horizon]
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        probs = F.softmax(logits.float(), dim=-1)                           # (bs, num_bins)
        pp = (probs * centers_raw_t.unsqueeze(0)).sum(dim=-1)              # (bs,)
        h_states[start:end] = h_seq.detach().cpu().numpy()
        bin_probs[start:end] = probs.detach().cpu().numpy()
        point_preds[start:end] = pp.detach().cpu().numpy()
        if (start // batch_size) % 50 == 0:
            print(f"[cgfm-eval]   capture {start}/{n}")
    return h_states, bin_probs, point_preds


def _bin_centers_raw(num_bins: int, bin_min: float, bin_max: float) -> np.ndarray:
    sym = np.linspace(bin_min, bin_max, num_bins)
    return np.sign(sym) * (np.exp(np.abs(sym)) - 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC")
    p.add_argument("--features", type=int, default=None)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--slice", default="val")
    p.add_argument("--horizon", type=int, default=1, choices=[1, 4, 16, 64],
                   help="V1.x horizon to evaluate CGFM on. Default h=1; "
                        "h=64 has wider return distribution where bin-tail capture matters more.")
    p.add_argument("--max-windows", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--cgfm-iters", type=int, default=3000)
    p.add_argument("--cgfm-batch", type=int, default=512)
    p.add_argument("--cgfm-hidden", type=int, default=256)
    p.add_argument("--cgfm-layers", type=int, default=3)
    p.add_argument("--cgfm-lr", type=float, default=2e-3)
    p.add_argument("--cgfm-samples", type=int, default=64)
    p.add_argument("--cgfm-steps", type=int, default=20)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "logs" / "frontier_ml" / "cgfm_eval"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    model, S, feat_list, base_dim, input_dim = _load_model_and_settings(
        args.features, Path(args.ckpt) if args.ckpt else None
    )
    model.eval()

    from frontier_ml.v1_upgrades.run_aci_v1_eval import _load_slice
    df_slice = _load_slice(args.asset, slice_name=args.slice)
    asset_u = args.asset if args.asset.endswith("USDT") else args.asset + "USDT"
    asset_idx = S.ASSET_TO_IDX.get(asset_u.replace("USDT", ""), 0)
    obs, y_full, regime_full, _ = _build_obs_tensor(
        df_slice, feat_list, S, asset_idx, horizon=args.horizon,
    )

    print(f"[cgfm-eval] horizon=h{args.horizon}, capturing windows (max={args.max_windows}, batch={args.batch_size})...")
    h_states, bin_probs, point_preds = _capture_windows(
        model, S, obs, asset_idx,
        seq_len=S.WM_SEQ_LEN, batch_size=args.batch_size,
        max_windows=args.max_windows, horizon=args.horizon,
    )
    n = h_states.shape[0]
    y_align = y_full[S.WM_SEQ_LEN - 1:S.WM_SEQ_LEN - 1 + n]
    regime_align = regime_full[S.WM_SEQ_LEN - 1:S.WM_SEQ_LEN - 1 + n]

    # 80/20 within-VAL split for CGFM-train / CGFM-eval
    n_train = int(n * args.train_frac)
    print(f"[cgfm-eval] within-VAL split: {n_train} train / {n - n_train} eval")

    # Build conditioning tensor [h || bin_probs || point_pred]
    cond_np = np.concatenate(
        [h_states, bin_probs, point_preds[:, None]], axis=-1
    ).astype(np.float32)
    cond_dim = cond_np.shape[1]
    residuals = (y_align - point_preds).astype(np.float32)

    cond_train = torch.from_numpy(cond_np[:n_train]).to(S.DEVICE)
    cond_eval = torch.from_numpy(cond_np[n_train:]).to(S.DEVICE)
    r_train = torch.from_numpy(residuals[:n_train]).to(S.DEVICE)
    pp_eval = point_preds[n_train:]
    y_eval = y_align[n_train:]
    regime_eval = regime_align[n_train:]

    # Train CGFM
    from frontier_ml.v1_upgrades.cgfm_residual import CGFMResidualHead
    head = CGFMResidualHead(
        cond_dim=cond_dim, residual_dim=1,
        hidden=args.cgfm_hidden, n_layers=args.cgfm_layers,
    ).to(S.DEVICE)
    optim = torch.optim.AdamW(head.parameters(), lr=args.cgfm_lr)
    print(f"[cgfm-eval] training CGFM cond_dim={cond_dim} hidden={args.cgfm_hidden} layers={args.cgfm_layers} iters={args.cgfm_iters}")
    losses = []
    n_train_t = cond_train.shape[0]
    for it in range(args.cgfm_iters):
        idx = torch.randint(0, n_train_t, (args.cgfm_batch,), device=S.DEVICE)
        loss = head.fm_loss(target=r_train[idx].unsqueeze(-1), cond=cond_train[idx])
        optim.zero_grad()
        loss.backward()
        optim.step()
        if it % max(1, args.cgfm_iters // 10) == 0:
            losses.append(float(loss.item()))
    print(f"[cgfm-eval] training losses (every {args.cgfm_iters // 10}): {losses}")

    # Sample on eval
    head.eval()
    print(f"[cgfm-eval] sampling {args.cgfm_samples}x{args.cgfm_steps}-step ODE on {len(pp_eval)} eval bars...")
    samples_r = head.sample(
        cond=cond_eval, n_samples=args.cgfm_samples, n_steps=args.cgfm_steps,
    ).detach().cpu().numpy()
    if samples_r.ndim == 3:
        samples_r = samples_r.squeeze(-1)
    samples_y = pp_eval[:, None] + samples_r                                # (n_eval, K)

    # Decision metrics
    pp_eval_t = pp_eval.astype(np.float64)
    y_eval_t = y_eval.astype(np.float64)
    cgfm_mean = samples_y.mean(axis=1).astype(np.float64)

    # IC: spearman of point estimator vs y_true
    rho_baseline, _ = spearmanr(pp_eval_t, y_eval_t)
    rho_cgfm, _ = spearmanr(cgfm_mean, y_eval_t)

    # CRPS comparison
    samples_y_t = torch.from_numpy(samples_y)
    y_eval_torch = torch.from_numpy(y_eval_t).float()
    crps_cgfm = head.crps(samples_y_t, y_eval_torch).item()

    # Two baselines:
    #  (a) Gaussian(pp, sigma_residual) -- standard but unfair on tiny-residual h=1
    #  (b) V1.1 bin distribution itself -- fair distributional baseline (multi-modal,
    #      asymmetric, same support as CGFM samples)
    sigma_residual = float(np.std(residuals[:n_train]))
    np.random.seed(0)
    z = np.random.randn(samples_y.shape[0], samples_y.shape[1]).astype(np.float32)
    samples_y_gauss = pp_eval[:, None] + sigma_residual * z
    crps_gauss = head.crps(torch.from_numpy(samples_y_gauss), y_eval_torch).item()

    # Bin-distribution sampling (categorical over bin centers in raw space)
    centers_raw = _bin_centers_raw(S.NUM_BINS, S.BIN_MIN, S.BIN_MAX).astype(np.float32)
    bin_probs_eval = bin_probs[n_train:]
    n_eval_bars = bin_probs_eval.shape[0]
    M = samples_y.shape[1]
    # Vectorized categorical: cumsum + uniform
    cdf = np.cumsum(bin_probs_eval, axis=-1)
    u = np.random.rand(n_eval_bars, M).astype(np.float32)
    idx_bin = (u[..., None] > cdf[:, None, :]).sum(axis=-1)  # (n_eval, M)
    idx_bin = np.clip(idx_bin, 0, S.NUM_BINS - 1)
    samples_y_bin = centers_raw[idx_bin]
    crps_bin = head.crps(torch.from_numpy(samples_y_bin), y_eval_torch).item()

    crps_baseline = crps_bin  # use the FAIR baseline as primary
    crps_lift_pct = (crps_baseline - crps_cgfm) / max(crps_baseline, 1e-12)
    crps_lift_pct_gauss = (crps_gauss - crps_cgfm) / max(crps_gauss, 1e-12)

    crps_lift_pct = (crps_baseline - crps_cgfm) / max(crps_baseline, 1e-12)
    ic_lift = float(rho_cgfm - rho_baseline)

    summary = {
        "asset": args.asset,
        "slice": args.slice,
        "input_dim": int(input_dim),
        "base_dim": int(base_dim),
        "n_capture": int(n),
        "n_train": int(n_train),
        "n_eval": int(n - n_train),
        "cgfm_iters": args.cgfm_iters,
        "cgfm_hidden": args.cgfm_hidden,
        "cond_dim": int(cond_dim),
        "sigma_residual": sigma_residual,
        "ic_baseline_h1": float(rho_baseline),
        "ic_cgfm_h1": float(rho_cgfm),
        "ic_lift": ic_lift,
        "crps_baseline_bin": float(crps_baseline),
        "crps_baseline_gauss": float(crps_gauss),
        "crps_cgfm": float(crps_cgfm),
        "crps_lift_vs_bin_pct": float(crps_lift_pct),
        "crps_lift_vs_gauss_pct": float(crps_lift_pct_gauss),
        "decision_gate": {
            "crps_lift_ge_5pct": bool(crps_lift_pct >= 0.05),
            "ic_lift_ge_0p003": bool(ic_lift >= 0.003),
            "primary_signal_ship": bool(crps_lift_pct >= 0.05 and ic_lift >= 0.003),
            "sizing_only_ship": bool(crps_lift_pct >= 0.05 and ic_lift < 0.003),
        },
    }

    summary["horizon"] = args.horizon
    out_path = out_dir / f"cgfm_{args.asset}_{args.slice}_h{args.horizon}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[cgfm-eval] IC baseline={rho_baseline:+.4f}  CGFM-mean={rho_cgfm:+.4f}  lift={ic_lift:+.5f}")
    print(f"[cgfm-eval] CRPS bin-baseline={crps_baseline:.5f}  Gauss-baseline={crps_gauss:.5f}  CGFM={crps_cgfm:.5f}")
    print(f"[cgfm-eval] CRPS lift  vs-bin={crps_lift_pct*100:+.2f}%  vs-gauss={crps_lift_pct_gauss*100:+.2f}%")
    print(f"[cgfm-eval] decision_gate = {summary['decision_gate']}")
    print(f"[cgfm-eval] summary written to {out_path}")
    return summary


if __name__ == "__main__":
    main()
