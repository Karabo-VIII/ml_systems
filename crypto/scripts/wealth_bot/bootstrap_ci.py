"""Bootstrap CIs on bot/ensemble trade returns + monthly sub-window stability.

Reads:
  - Paper-trade JSONL journal (per-trade entries + exits with realized_pnl_pct).
  - Or: engine all-in trade returns reconstructed from audit_ensemble preds + actions
    (for the engine-ceiling +65.8% ensemble compound).

Outputs:
  - data/bootstrap_ci.json with:
      bootstrap_compound: dict[seg -> p05/p50/p95]  (block-bootstrap of trade returns)
      monthly_subwindow: dict[seg -> list of (month, n_trades, compound_pct)]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from wealth_bot.framework.config import load_config
from wealth_bot.framework.data_loader import prepare
from wealth_bot.framework.repro import CANONICAL_SEEDS, SCHEMA_VERSION, build_repro_block
from wealth_bot.framework.upgrades import apply_threshold


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def trade_returns_from_engine(cfg, df_lag, signals, fwd_ret, ensemble_preds, threshold):
    """Reconstruct per-trade returns from engine preds (all-in, non-overlapping)."""
    actions, chosen = apply_threshold(ensemble_preds, signals, cfg.fwd_bars, threshold)
    n = len(actions)
    rets = []
    timestamps_idx = []
    i = 0
    while i < n:
        if actions[i] == 1 and not np.isnan(fwd_ret[i]):
            rets.append(float(fwd_ret[i]))
            timestamps_idx.append(i)
            i += cfg.fwd_bars
        else:
            i += 1
    return np.array(rets), np.array(timestamps_idx)


def bootstrap_compound(returns: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    """Sample-with-replacement bootstrap of compound returns."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "mean": 0.0, "std": 0.0, "n_trades": 0}
    compounds = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        compounds[b] = (np.prod(1 + sample) - 1) * 100
    return {
        "p05": float(np.percentile(compounds, 5)),
        "p25": float(np.percentile(compounds, 25)),
        "p50": float(np.percentile(compounds, 50)),
        "p75": float(np.percentile(compounds, 75)),
        "p95": float(np.percentile(compounds, 95)),
        "mean": float(compounds.mean()),
        "std": float(compounds.std()),
        "n_trades": int(n),
        "actual_compound": float((np.prod(1 + returns) - 1) * 100),
        "n_boot": n_boot,
    }


def stationary_block_bootstrap_compound(
    returns: np.ndarray,
    n_boot: int = 5000,
    avg_block_size: int = 5,
    seed: int = 0,
) -> dict:
    """Politis-Romano stationary block bootstrap — preserves temporal correlation
    in returns (better for trade sequences where consecutive trades may be regime-correlated)."""
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0, "n_trades": 0}
    p = 1.0 / avg_block_size
    compounds = np.zeros(n_boot)
    for b in range(n_boot):
        sample = np.empty(n)
        i = 0
        while i < n:
            start = int(rng.integers(0, n))
            length = max(1, int(rng.geometric(p)))
            length = min(length, n - i)
            for j in range(length):
                sample[i + j] = returns[(start + j) % n]
            i += length
        compounds[b] = (np.prod(1 + sample) - 1) * 100
    return {
        "p05": float(np.percentile(compounds, 5)),
        "p25": float(np.percentile(compounds, 25)),
        "p50": float(np.percentile(compounds, 50)),
        "p75": float(np.percentile(compounds, 75)),
        "p95": float(np.percentile(compounds, 95)),
        "n_trades": int(n),
        "avg_block_size": avg_block_size,
        "n_boot": n_boot,
    }


def monthly_subwindow(
    cfg,
    df,
    returns: np.ndarray,
    trade_bar_idx: np.ndarray,
) -> list[dict]:
    """Group trades by calendar month and compute per-month compound."""
    months = []
    if len(returns) == 0:
        return months
    dates = df["date"].values
    trade_dates = dates[trade_bar_idx]
    df_t = pd.DataFrame({"date": trade_dates, "ret": returns})
    df_t["month"] = pd.to_datetime(df_t["date"]).dt.to_period("M")
    for month, g in df_t.groupby("month"):
        r = g["ret"].values
        compound = (np.prod(1 + r) - 1) * 100
        months.append({
            "month": str(month),
            "n_trades": int(len(r)),
            "compound_pct": float(compound),
            "win_rate": float((r > 0).mean()) if len(r) else 0.0,
            "mean_trade_pct": float(r.mean() * 100) if len(r) else 0.0,
        })
    return months


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--audit-json", required=True,
                    help="audit_ensemble_10seed_with_preds.json (must have sidecar .npz)")
    ap.add_argument("--segments", nargs="+", default=["UNSEEN"],
                    help="Segments to analyze (default: UNSEEN)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0,
                    help="Bootstrap RNG seed (default 0 for backward compat; "
                         "use --seed 42 for canonical reproducible runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log(f"Config: asset={cfg.asset} cadence={cfg.cadence}")
    df, df_lag, signals, fwd_ret, masks = prepare(cfg)

    # Load ensemble preds
    audit_path = Path(args.audit_json)
    with open(audit_path) as fp:
        audit = json.load(fp)
    npz_name = audit.get("_preds_npz")
    if not npz_name:
        log("ERROR: audit JSON has no sidecar .npz")
        return 2
    npz_path = audit_path.parent / npz_name
    z = np.load(npz_path, allow_pickle=False)
    ensemble_preds = z["ensemble_preds"]
    threshold = float(z["ensemble_best_threshold"]) if "ensemble_best_threshold" in z.files else 0.0
    log(f"loaded preds shape={ensemble_preds.shape} threshold={threshold:+.4f}")

    _chimera_paths = [
        str(p) for p in (ROOT / "data").rglob(
            f"*{cfg.asset.lower()}usdt*{cfg.cadence}*.parquet"
        )
    ]
    results = {
        "reproducibility": build_repro_block(
            command_line=" ".join(sys.argv),
            config_path=args.config,
            chimera_paths=_chimera_paths,
            seeds=CANONICAL_SEEDS,
            extra={"bootstrap_seed": args.seed},
        ),
        "wall_clock": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "n_boot": args.n_boot,
        "bootstrap_seed": args.seed,
        "segments": {},
    }

    for seg in args.segments:
        log(f"=== Segment: {seg} ===")
        seg_mask = masks[seg]

        # Engine all-in returns (from ensemble preds applied within seg only)
        # We mask signals outside the segment to zero so apply_threshold respects seg.
        sig_masked = signals.copy()
        sig_masked[~seg_mask] = 0
        actions, chosen = apply_threshold(ensemble_preds, sig_masked, cfg.fwd_bars, threshold)
        # Per-trade returns
        trade_returns = []
        trade_idx = []
        n_total = len(actions)
        i = 0
        while i < n_total:
            if actions[i] == 1 and not np.isnan(fwd_ret[i]):
                trade_returns.append(float(fwd_ret[i]))
                trade_idx.append(i)
                i += cfg.fwd_bars
            else:
                i += 1
        trade_returns = np.array(trade_returns)
        trade_idx = np.array(trade_idx)
        log(f"  {seg}: {len(trade_returns)} trades, actual compound "
            f"{(np.prod(1 + trade_returns) - 1) * 100:+.1f}%")

        # IID bootstrap
        iid = bootstrap_compound(trade_returns, n_boot=args.n_boot, seed=args.seed)
        log(f"  IID bootstrap: p05={iid['p05']:+.1f}% p50={iid['p50']:+.1f}% p95={iid['p95']:+.1f}%")

        # Block bootstrap
        block = stationary_block_bootstrap_compound(
            trade_returns, n_boot=args.n_boot, avg_block_size=5, seed=args.seed
        )
        log(f"  Block bootstrap (avg_block=5): p05={block['p05']:+.1f}% p50={block['p50']:+.1f}% p95={block['p95']:+.1f}%")

        # Monthly sub-windows
        monthly = monthly_subwindow(cfg, df, trade_returns, trade_idx)
        log(f"  Monthly sub-windows ({len(monthly)} months):")
        for m in monthly:
            log(f"    {m['month']}: n={m['n_trades']:>2} compound={m['compound_pct']:+6.1f}% "
                f"WR={m['win_rate']:.2f}")

        results["segments"][seg] = {
            "n_trades": int(len(trade_returns)),
            "actual_compound_pct": float(iid["actual_compound"]),
            "iid_bootstrap": iid,
            "block_bootstrap": block,
            "monthly_subwindow": monthly,
            "trade_returns": [float(x) for x in trade_returns],
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    log(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
