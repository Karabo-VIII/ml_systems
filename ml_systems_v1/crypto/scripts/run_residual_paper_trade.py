#!/usr/bin/env python
"""Strat-layer Round-12 orchestrator: run base + N residuals on real chimera data.

Closes the missing orchestration script for `src/strategy/_shared/strat_residuals.py`
(StratResidualManager + KVariantResidual + RankerResidual). Runs N parallel
paper-trade equity curves on the same daily chimera_legacy panel and emits a
leaderboard JSON.

Two residual designs wired:
  - KVariantResidual: K=3, K=5, K=7, K=10, K=15 (top-K) using a single ranker
  - RankerResidual:   momentum / vol / random rankers at fixed K

Usage:
  python scripts/run_residual_paper_trade.py --start 2024-01-01 --end 2026-05-01
  python scripts/run_residual_paper_trade.py --ranker momentum --k-variants 3,5,7,10
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from strategy._shared.strat_residuals import (
    KVariantResidual, RankerResidual, StratResidualManager,
)


# Reused from probe_strategy_ceiling.py
def load_chimera_panel(start_date: str, end_date: str) -> pl.DataFrame:
    """Build per-(day, asset) panel from chimera_legacy/dollar."""
    data_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    files = sorted(data_dir.glob("*_v50_chimera_*.parquet"))
    start_ms = int(datetime.fromisoformat(start_date)
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_date)
                  .replace(tzinfo=timezone.utc).timestamp() * 1000)

    rows = []
    cols_to_select = ["timestamp", "target_return_1"]
    for fp in files:
        sym = fp.stem.split("_v50_chimera_")[0].replace("usdt", "").upper()
        try:
            df = (pl.read_parquet(fp)
                    .select(cols_to_select)
                    .filter((pl.col("timestamp") >= start_ms) &
                            (pl.col("timestamp") < end_ms)))
            if len(df) < 50:
                continue
            df = df.with_columns([
                (pl.col("timestamp") // 86_400_000).alias("day_id"),
                pl.lit(sym).alias("asset"),
            ])
            daily = (df.group_by(["day_id", "asset"])
                       .agg([pl.col("target_return_1").sum().alias("daily_ret"),
                             pl.len().alias("n_bars")])
                       .filter(pl.col("n_bars") >= 5))
            rows.append(daily)
        except Exception as e:
            print(f"  [skip] {sym}: {type(e).__name__}: {e}", flush=True)
            continue
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows)


def run_paper_trade(start: str, end: str, k_variants: list[int],
                     ranker: str, top_k_for_ranker: int,
                     out_dir: Path) -> dict:
    panel = load_chimera_panel(start, end)
    if len(panel) == 0:
        print("[residual] HARD FAIL: no panel data", flush=True)
        sys.exit(2)
    print(f"[residual] panel rows={len(panel)} "
          f"days={panel['day_id'].n_unique()} assets={panel['asset'].n_unique()}",
          flush=True)

    panel = (panel.sort(["asset", "day_id"])
                  .with_columns(
                      pl.col("daily_ret").shift(1).over("asset").alias("lag_ret")))

    mgr = StratResidualManager(state_dir=out_dir)
    mgr.register("k_variants", KVariantResidual(k_variants=k_variants))
    mgr.register("rankers", RankerResidual(
        ranker_names=["momentum", "vol", "random"], top_k=top_k_for_ranker))
    rng = np.random.default_rng(42)

    days = sorted(panel["day_id"].unique().to_list())
    n_steps = 0
    for d in days:
        sub = panel.filter(pl.col("day_id") == d).to_pandas()
        if len(sub) < max(k_variants) * 2:
            continue
        scoreable = sub.dropna(subset=["lag_ret"]).copy()
        if len(scoreable) < max(k_variants):
            continue
        scoreable["abs_lag"] = scoreable["lag_ret"].abs()
        ret_lookup = dict(zip(sub["asset"], sub["daily_ret"]))

        # KVariantResidual: each K reads ranker = momentum top-K
        per_k_returns: dict[int, float] = {}
        for k in k_variants:
            picks = scoreable.nlargest(k, "lag_ret")["asset"].tolist()
            per_k_returns[k] = float(np.mean(
                [ret_lookup.get(a, 0.0) for a in picks]))
        mgr.get("k_variants").step(d, per_k_returns)

        # RankerResidual: fixed K, three ranker variants
        ranker_picks = {
            "momentum": scoreable.nlargest(top_k_for_ranker,
                                             "lag_ret")["asset"].tolist(),
            "vol":      scoreable.nlargest(top_k_for_ranker,
                                             "abs_lag")["asset"].tolist(),
            "random":   rng.choice(sub["asset"].tolist(),
                                     size=min(top_k_for_ranker, len(sub)),
                                     replace=False).tolist(),
        }
        # Realized top-K (oracle) for hit-rate
        sub["abs_ret"] = sub["daily_ret"].abs()
        realized_top_k = sub.nlargest(top_k_for_ranker,
                                        "abs_ret")["asset"].tolist()
        mgr.get("rankers").step(d, ranker_picks, realized_top_k, ret_lookup)
        n_steps += 1

    print(f"[residual] stepped {n_steps} day(s); writing report", flush=True)
    report = mgr.save_report(fname="residual_report.json")
    return {"n_steps": n_steps, "report": report}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-05-01")
    parser.add_argument("--k-variants", type=str, default="3,5,7,10,15",
                        help="Comma-separated K values for KVariantResidual.")
    parser.add_argument("--ranker", type=str, default="momentum",
                        help="Label only; rankers are fixed in this orchestrator.")
    parser.add_argument("--ranker-k", type=int, default=5,
                        help="Top-K used by RankerResidual.")
    parser.add_argument("--out-dir", type=str,
                        default="logs/strat_residuals")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else (
        PROJECT_ROOT / args.out_dir)
    k_variants = [int(x) for x in args.k_variants.split(",") if x.strip()]

    print("=" * 100, flush=True)
    print(f"  RESIDUAL PAPER-TRADE - K={k_variants}  ranker_K={args.ranker_k}",
          flush=True)
    print(f"  Window: {args.start} -> {args.end}", flush=True)
    print(f"  Output: {out_dir}", flush=True)
    print("=" * 100, flush=True)

    result = run_paper_trade(args.start, args.end, k_variants,
                              args.ranker, args.ranker_k, out_dir)

    # Summary print
    rep = result["report"]
    print("\n  K-VARIANT LEADERBOARD (top-K by lagged-momentum ranker):", flush=True)
    if "k_variants" in rep:
        rows = []
        for name, stats in rep["k_variants"].items():
            if isinstance(stats, dict) and "sharpe" in stats:
                rows.append((name, stats.get("cum_ret", 0.0),
                              stats.get("sharpe", 0.0),
                              stats.get("max_dd", 0.0),
                              stats.get("mean_daily", 0.0)))
        rows.sort(key=lambda r: -r[1])
        print(f"    {'variant':<10} {'cum_ret':>10} {'sharpe':>8} "
              f"{'max_dd':>8} {'mean/day':>10}", flush=True)
        for name, cum, sh, dd, m in rows:
            print(f"    {name:<10} {cum*100:>+9.2f}% {sh:>+8.2f} "
                  f"{dd*100:>+7.2f}% {m*100:>+9.4f}%", flush=True)

    print("\n  RANKER LEADERBOARD (fixed K, three rankers):", flush=True)
    if "rankers" in rep:
        rows = []
        for name, stats in rep["rankers"].items():
            if isinstance(stats, dict) and "mean_hit_rate" in stats:
                rows.append((name, stats.get("mean_hit_rate", 0.0),
                              stats.get("mean_daily_return", 0.0),
                              stats.get("cum_return", 0.0)))
        rows.sort(key=lambda r: -r[3])
        print(f"    {'ranker':<10} {'hit_rate':>10} {'mean/day':>10} "
              f"{'cum_ret':>10}", flush=True)
        for name, hr, m, cum in rows:
            print(f"    {name:<10} {hr*100:>+9.2f}% {m*100:>+9.4f}% "
                  f"{cum*100:>+9.2f}%", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
