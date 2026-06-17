"""Run E1 (Adaptive Conformal Inference) on a V1.1 best-EMA checkpoint.

Wires the ACI online wrapper around V1.1's h=1 TwoHot bin distribution
on the OOS slice. Reports:
    - empirical coverage rate (target 0.90)
    - per-regime coverage (bear / chop / bull from regime_label)
    - interval width statistics (mean / std / p95)
    - Spearman correlation: ACI width vs |target_return_1|
      -- the regime-stress signal usefulness for sizing.

This is the ground-truth probe for B007 E1. Decision gate per RESPONSE:
    coverage holds within 88-92% per regime AND width-aware sizing lifts
    Sortino by >= 0.3 vs flat sizing -> ship as default inference layer.

Sortino lift is downstream of this script (sizing-strategy backtest);
this script reports the upstream signals.

Usage:
    python -m frontier_ml.v1_upgrades.run_aci_v1_eval --asset BTC --features 29
    python -m frontier_ml.v1_upgrades.run_aci_v1_eval --target-coverage 0.80

Outputs JSON summary to logs/frontier_ml/aci_eval/<asset>_<features>_<ts>.json.
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


def _detect_dims_from_state(sd: dict) -> tuple[int, int]:
    """Read obs_encoder.0.weight (input_dim) and decoder.2.weight (base_dim) from state."""
    enc = sd.get("obs_encoder.0.weight")
    dec = sd.get("decoder.2.weight")
    if enc is None or dec is None:
        raise KeyError("checkpoint missing obs_encoder.0.weight or decoder.2.weight")
    asset_emb_dim = sd["asset_embedding.weight"].shape[1]
    input_dim = int(enc.shape[1] - asset_emb_dim)
    base_dim = int(dec.shape[0])
    return input_dim, base_dim


def _resolve_feature_list(base_dim: int, input_dim: int, S) -> list:
    """Pick the FEATURE_LIST_<N> whose len matches input_dim; fall back to base_dim."""
    from settings import get_feature_config

    # Try input_dim first, then base_dim
    for n in (input_dim, base_dim):
        try:
            fl, idim, bdim = get_feature_config(n)
            if idim == input_dim and bdim == base_dim:
                return fl
        except Exception:
            pass
    # Fallback: use the master FEATURE_LIST truncated to input_dim
    fl_master = list(getattr(S, "FEATURE_LIST", []))
    if len(fl_master) >= input_dim:
        return fl_master[:input_dim]
    raise ValueError(
        f"cannot resolve feature_list for input_dim={input_dim} base_dim={base_dim}"
    )


def _load_model_and_settings(features: int | None, ckpt_path: Path | None):
    """Construct V1.1 model with dims auto-detected from checkpoint; load best-EMA."""
    import settings as S
    from world_model import TransformerWorldModel

    if ckpt_path is None:
        candidates = [
            S.MODEL_DIR / "v1e_best_ema.pt",
        ]
        if features is not None:
            candidates.insert(0, S.BASE_MODEL_DIR / f"v1_1_f{features}_wm_best_ema.pt")
        for c in candidates:
            if c.exists():
                ckpt_path = c
                break
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No V1.1 best-EMA checkpoint found in {S.MODEL_DIR} or {S.BASE_MODEL_DIR}"
        )
    print(f"[aci-eval] loading checkpoint {ckpt_path.name}")
    ckpt = torch.load(ckpt_path, map_location=S.DEVICE, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    input_dim, base_dim = _detect_dims_from_state(sd)
    print(f"[aci-eval] auto-detected input_dim={input_dim} base_dim={base_dim}")

    feat_list = _resolve_feature_list(base_dim, input_dim, S)
    print(f"[aci-eval] feature_list len={len(feat_list)} (head: {feat_list[:4]})")

    model = TransformerWorldModel(input_dim=input_dim, base_dim=base_dim).to(S.DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[aci-eval] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model, S, feat_list, base_dim, input_dim


def _bin_centers_raw(num_bins: int, bin_min: float, bin_max: float) -> np.ndarray:
    """Bin centers in raw return space.

    The TwoHotSymlog bucketer places linearly-spaced buckets in symlog space.
    Decoding is symexp(sum(probs * buckets)). For ACI we need raw-return bin
    centers so the discrete-CDF maps to raw returns directly.
    """
    sym = np.linspace(bin_min, bin_max, num_bins)
    raw = np.sign(sym) * (np.exp(np.abs(sym)) - 1.0)
    return raw


# Trainer split: 50% train / 20% val / 20% oos / 10% unseen.
# (Matches CLAUDE.md "Data Split (50/20/20/10)" and validate_world.py.)
# Purge-gap boundaries are not honored here; this is a fractional approximation
# good enough for ACI gate-checks. Use VAL for any gate decision; OOS is reserved
# for strategy backtests; UNSEEN must be untouched until deploy.
_SLICE_FRACTIONS = {
    "train":  (0.00, 0.50),
    "val":    (0.50, 0.70),
    "oos":    (0.70, 0.90),
    "unseen": (0.90, 1.00),
}


def _load_slice(asset: str, slice_name: str = "val") -> pl.DataFrame:
    """Load chimera for `asset` and return the named split (val/oos/unseen).

    Uses ChimeraLoader (cadence='dollar') which auto-resolves v50/v51.
    """
    from pipeline.chimera_loader import ChimeraLoader

    if slice_name not in _SLICE_FRACTIONS:
        raise ValueError(f"unknown slice: {slice_name}; known={list(_SLICE_FRACTIONS)}")
    lo, hi = _SLICE_FRACTIONS[slice_name]
    loader = ChimeraLoader()
    df = loader.load(asset, cadence="dollar")
    n = len(df)
    start = int(n * lo)
    end = int(n * hi)
    df_slice = df.slice(start, end - start)
    print(f"[aci-eval] {slice_name.upper()} slice: rows [{start:,}..{end:,}) of {n:,} "
          f"({len(df_slice):,} bars) for {asset}")
    return df_slice


def _build_obs_tensor(df: pl.DataFrame, feat_list, S, asset_idx: int):
    """Build (T, F) tensor of features and (T,) tensor of target_return_1 + regime_label.

    Drops rows missing required cols. Returns (obs, y_h1, regime, asset_id).
    """
    needed = list(feat_list) + ["target_return_1"]
    if "regime_label" in df.columns:
        needed.append("regime_label")
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"chimera missing cols: {missing}")
    sub = df.select(needed).drop_nulls()
    feats = sub.select(feat_list).to_numpy().astype(np.float32)
    y = sub.get_column("target_return_1").to_numpy().astype(np.float32)
    regime = sub.get_column("regime_label").to_numpy().astype(np.int64) if "regime_label" in df.columns else np.full(len(sub), 1, dtype=np.int64)
    obs = torch.from_numpy(feats).to(S.DEVICE)
    return obs, y, regime, asset_idx


@torch.no_grad()
def _bin_probs_h1_stream(
    model, S, obs: torch.Tensor, asset_idx: int, seq_len: int = 96,
    batch_size: int = 64, max_windows: int | None = None,
):
    """Batched forward over sliding 96-bar windows; return per-window h=1 probs.

    Uses model.forward_train (V1.1 entry). Window k spans obs[k:k+seq_len];
    its prediction targets the bar at obs[k+seq_len-1] (last bar of window).
    Aligned target_return_1 lives at index k+seq_len-1.
    """
    T = obs.shape[0]
    n_win = T - seq_len + 1
    if max_windows is not None and max_windows < n_win:
        n_win = max_windows
    out_probs = np.zeros((n_win, S.NUM_BINS), dtype=np.float32)
    asset_t_template = torch.full(
        (batch_size,), asset_idx, dtype=torch.long, device=S.DEVICE,
    )
    for start in range(0, n_win, batch_size):
        end = min(start + batch_size, n_win)
        bs = end - start
        # Stack windows: each row is obs[k:k+seq_len], k = start..end-1
        windows = torch.stack(
            [obs[k:k + seq_len] for k in range(start, end)], dim=0
        )  # (bs, seq_len, F)
        asset_t = asset_t_template[:bs]
        outputs = model.forward_train(windows, asset_t)
        logits = outputs["return_logits"][1]  # (bs, seq_len, num_bins) typically
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        p = F.softmax(logits.float(), dim=-1).cpu().numpy()
        out_probs[start:end] = p
        if (start // batch_size) % 50 == 0:
            print(f"[aci-eval]   forward {start}/{n_win}")
    return out_probs


def _per_regime_coverage(errors: np.ndarray, regime: np.ndarray) -> dict:
    out = {}
    for rid, name in [(0, "bear"), (1, "chop"), (2, "bull")]:
        mask = regime == rid
        if mask.sum() == 0:
            continue
        out[name] = {
            "n": int(mask.sum()),
            "coverage": float(1.0 - errors[mask].mean()),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="BTC")
    p.add_argument("--features", type=int, default=None,
                   help="V1.x feature count for ckpt naming convention. "
                        "If unset, auto-detect from checkpoint state-dict.")
    p.add_argument("--target-coverage", type=float, default=0.90)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--slice", default="val", choices=list(_SLICE_FRACTIONS.keys()),
                   help="Which split to use. VAL for gate-checks (default); "
                        "OOS is reserved for strategy backtests; UNSEEN untouched until deploy.")
    p.add_argument("--max-windows", type=int, default=None,
                   help="Cap number of forward-pass windows (smoke speed-up).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default=str(PROJECT_ROOT / "logs" / "frontier_ml" / "aci_eval"))
    p.add_argument("--save-parquet", action="store_true",
                   help="Persist per-bar (y, point_pred, L, U, width, regime) to a parquet "
                        "for downstream sizing-strategy backtests.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    model, S, feat_list, base_dim, input_dim = _load_model_and_settings(
        args.features, Path(args.ckpt) if args.ckpt else None
    )

    df_slice = _load_slice(args.asset, slice_name=args.slice)
    asset_u = args.asset if args.asset.endswith("USDT") else args.asset + "USDT"
    asset_idx = S.ASSET_TO_IDX.get(asset_u.replace("USDT", ""), 0)
    obs, y_full, regime_full, _ = _build_obs_tensor(df_slice, feat_list, S, asset_idx)

    n_total_windows = obs.shape[0] - S.WM_SEQ_LEN + 1
    n_planned = min(n_total_windows, args.max_windows) if args.max_windows else n_total_windows
    print(f"[aci-eval] forward pass over {n_planned}/{n_total_windows} windows (batch={args.batch_size})...")
    probs_arr = _bin_probs_h1_stream(
        model, S, obs, asset_idx,
        seq_len=S.WM_SEQ_LEN, batch_size=args.batch_size, max_windows=args.max_windows,
    )
    # The window at index k predicts the bar at obs[k + seq_len - 1]; target_return_1 at that
    # bar is the realized next-bar return. So the y_target aligned with probs_arr[k] is y_full[k + seq_len - 1].
    n = probs_arr.shape[0]
    y_align = y_full[S.WM_SEQ_LEN - 1:S.WM_SEQ_LEN - 1 + n]
    regime_align = regime_full[S.WM_SEQ_LEN - 1:S.WM_SEQ_LEN - 1 + n]
    assert len(y_align) == n, f"alignment off: {len(y_align)} vs {n}"

    # Build raw-return bin centers
    bin_min = float(S.BIN_MIN)
    bin_max = float(S.BIN_MAX)
    centers_raw = _bin_centers_raw(S.NUM_BINS, bin_min, bin_max)

    # Run ACI online over the stream
    from frontier_ml.v1_upgrades.adaptive_conformal import (
        AdaptiveConformalInference,
        bin_probs_to_quantile_fn,
    )

    aci = AdaptiveConformalInference(target_coverage=float(args.target_coverage))
    widths = np.zeros(n, dtype=np.float64)
    errors = np.zeros(n, dtype=np.int64)
    Ls = np.zeros(n, dtype=np.float64)
    Us = np.zeros(n, dtype=np.float64)
    # Point estimate per bar: expectation under the bin distribution in raw space.
    point_preds = np.zeros(n, dtype=np.float64)
    for t in range(n):
        q = bin_probs_to_quantile_fn(probs_arr[t], centers_raw)
        L, U = aci.predict_interval(q)
        Ls[t] = L
        Us[t] = U
        widths[t] = U - L
        point_preds[t] = float((probs_arr[t] * centers_raw).sum())
        y_t = float(y_align[t])
        errors[t] = 0 if (L <= y_t <= U) else 1
        aci.update(y_t, L, U)

    coverage = float(1.0 - errors.mean())
    per_reg = _per_regime_coverage(errors, regime_align)

    # Width-as-stress signal: Spearman of width vs |y|
    abs_y = np.abs(y_align).astype(np.float64)
    rho_width_absy, p_value = spearmanr(widths, abs_y)

    summary = {
        "asset": args.asset,
        "features_arg": args.features,
        "input_dim": int(input_dim),
        "base_dim": int(base_dim),
        "n_windows": int(n),
        "target_coverage": float(args.target_coverage),
        "empirical_coverage": coverage,
        "per_regime_coverage": per_reg,
        "width_stats": {
            "mean": float(widths.mean()),
            "std": float(widths.std()),
            "p50": float(np.median(widths)),
            "p95": float(np.percentile(widths, 95)),
        },
        "spearman_width_vs_abs_y": {
            "rho": float(rho_width_absy),
            "p_value": float(p_value),
            "sample_size": int(n),
        },
        "decision_gate": {
            "coverage_in_band": bool(0.88 <= coverage <= 0.92) if args.target_coverage == 0.90 else None,
            "per_regime_in_band": bool(all(
                0.85 <= v["coverage"] <= 0.95 for v in per_reg.values()
            )),
            "width_predicts_volatility": bool(rho_width_absy > 0.05 and p_value < 0.05),
        },
    }

    feat_tag = f"f{args.features}" if args.features is not None else f"input{input_dim}base{base_dim}"
    summary["slice"] = args.slice
    out_path = out_dir / f"aci_{args.asset}_{args.slice}_{feat_tag}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    if args.save_parquet:
        # Try to align timestamps (may not exist in all chimera shapes; skip if absent).
        ts_col = None
        if "timestamp" in df_slice.columns:
            ts_full = df_slice.get_column("timestamp").to_numpy()
            ts_col = ts_full[S.WM_SEQ_LEN - 1:S.WM_SEQ_LEN - 1 + n]
        bar_df = pl.DataFrame({
            "y_true": y_align[:n].astype(np.float64),
            "point_pred": point_preds,
            "L": Ls,
            "U": Us,
            "width": widths,
            "regime_label": regime_align[:n].astype(np.int64),
            "covered": (1 - errors).astype(np.int8),
            **({"timestamp": ts_col} if ts_col is not None else {}),
        })
        pq_path = out_dir / f"aci_{args.asset}_{args.slice}_{feat_tag}_{ts}.parquet"
        bar_df.write_parquet(pq_path)
        print(f"[aci-eval] per-bar parquet: {pq_path} ({len(bar_df)} rows)")
    print(f"[aci-eval] empirical coverage = {coverage:.4f}  (target {args.target_coverage})")
    print(f"[aci-eval] per-regime coverage: {per_reg}")
    print(f"[aci-eval] width: mean={widths.mean():.5f} p95={np.percentile(widths, 95):.5f}")
    print(f"[aci-eval] Spearman(width, |y|) = {rho_width_absy:.4f}  p={p_value:.4g}")
    print(f"[aci-eval] decision_gate = {summary['decision_gate']}")
    print(f"[aci-eval] summary written to {out_path}")
    return summary


if __name__ == "__main__":
    main()
