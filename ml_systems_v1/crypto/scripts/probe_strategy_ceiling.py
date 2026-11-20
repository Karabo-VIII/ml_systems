#!/usr/bin/env python
"""Strategy-Ceiling Probe (Round-11, 2026-05-08).

Analog to scripts/probe_real_data_ceiling.py for the WM layer. Empirically
measures a strategy's selection quality + drawdown response in <2 min,
without running a full backtest.

Three modes:
  - synthetic: planted regime shift + known top-mover distribution; measures
    selection accuracy
  - real: held-out chimera_legacy slice; measures realized vs target ROI
  - stress: scenarios (BTC -30% flash, funding spike, regime shift, exchange
    outage); measures drawdown + recovery

For our 1-5% daily target, real mode reports:
  - per-day top-K selection hit rate
  - daily ROI distribution
  - days hitting >=1%, >=2%, >=5%

Usage:
  python scripts/probe_strategy_ceiling.py --mode real --strategy xsec_K5_5 \
    --top-k 5 --start 2026-01-01 --end 2026-04-30

  python scripts/probe_strategy_ceiling.py --mode stress --scenario btc_flash_crash \
    --strategy xsec_K7_7
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StrategyProbeResult:
    mode: str
    strategy: str
    n_days: int = 0
    days_hit_1pct: int = 0
    days_hit_2pct: int = 0
    days_hit_5pct: int = 0
    mean_daily_ret: float = 0.0
    median_daily_ret: float = 0.0
    cum_ret: float = 0.0
    sharpe: float = 0.0
    max_dd: float = 0.0
    top_k_hit_rate: float = 0.0
    failure_mode: str = ""


# =============================================================================
# REAL MODE — measure on chimera_legacy slice
# =============================================================================

def load_chimera_panel(start_date: str, end_date: str,
                       feature_subset: list = None) -> pl.DataFrame:
    """Build per-day per-asset panel from chimera_legacy/dollar."""
    data_dir = PROJECT_ROOT / "data" / "processed" / "chimera_legacy" / "dollar"
    files = sorted(data_dir.glob("*_v50_chimera_*.parquet"))
    start_ms = int(datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc).timestamp() * 1000)

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
        except Exception:
            continue
    if not rows:
        return pl.DataFrame()
    return pl.concat(rows)


def probe_real(strategy_name: str, top_k: int, start_date: str, end_date: str,
                random_seed: int = 42, ranker: str = "momentum") -> StrategyProbeResult:
    """Measure top-K selection + daily ROI on real data slice.

    Rankers:
      - "momentum": top-K by t-1 daily return (real, no look-ahead)
      - "random":   uniform random top-K (null-baseline for comparison)
      - "vol":      top-K by t-1 absolute return (vol-chasing)

    Real V22/V25 trained-ranker integration plugs in by adding another
    branch that calls model.predict on the per-day feature panel.
    """
    result = StrategyProbeResult(mode="real", strategy=strategy_name)
    panel = load_chimera_panel(start_date, end_date)
    if len(panel) == 0:
        result.failure_mode = "no panel data"
        return result

    # Build per-(asset) lagged-return lookup so ranker has a real predictor
    # without look-ahead. shift(1) within each asset; drop t-1 NaNs.
    panel = (panel.sort(["asset", "day_id"])
                  .with_columns(
                      pl.col("daily_ret").shift(1).over("asset").alias("lag_ret")))

    days = sorted(panel["day_id"].unique().to_list())
    rng = np.random.default_rng(random_seed)
    daily_strategy_returns = []
    hit_count = 0
    realized_top_total = 0
    n_skipped = 0

    for d in days:
        sub = panel.filter(pl.col("day_id") == d)
        if len(sub) < top_k * 2:
            n_skipped += 1
            continue
        sub_pd = sub.to_pandas()
        # Realized top-K movers by absolute return (oracle benchmark)
        sub_pd["abs_ret"] = sub_pd["daily_ret"].abs()
        realized_top_k = sub_pd.nlargest(top_k, "abs_ret")["asset"].tolist()

        # Pick assets per ranker
        if ranker == "momentum":
            scoreable = sub_pd.dropna(subset=["lag_ret"])
            if len(scoreable) < top_k:
                n_skipped += 1
                continue
            picks = scoreable.nlargest(top_k, "lag_ret")["asset"].tolist()
        elif ranker == "vol":
            scoreable = sub_pd.dropna(subset=["lag_ret"]).copy()
            if len(scoreable) < top_k:
                n_skipped += 1
                continue
            scoreable["abs_lag"] = scoreable["lag_ret"].abs()
            picks = scoreable.nlargest(top_k, "abs_lag")["asset"].tolist()
        elif ranker == "random":
            picks = rng.choice(sub_pd["asset"].tolist(),
                                size=min(top_k, len(sub_pd)),
                                replace=False).tolist()
        else:
            result.failure_mode = f"unknown ranker: {ranker}"
            return result

        ret_lookup = dict(zip(sub_pd["asset"], sub_pd["daily_ret"]))
        strat_ret = float(np.mean([ret_lookup.get(a, 0.0) for a in picks]))
        daily_strategy_returns.append(strat_ret)
        hit_count += len(set(picks) & set(realized_top_k))
        realized_top_total += top_k

    if not daily_strategy_returns:
        result.failure_mode = "no valid days"
        return result
    rets = np.array(daily_strategy_returns)
    eq = np.cumprod(1 + rets)

    result.n_days = len(rets)
    result.days_hit_1pct = int((rets >= 0.01).sum())
    result.days_hit_2pct = int((rets >= 0.02).sum())
    result.days_hit_5pct = int((rets >= 0.05).sum())
    result.mean_daily_ret = float(rets.mean())
    result.median_daily_ret = float(np.median(rets))
    result.cum_ret = float(eq[-1] - 1)
    if rets.std() > 1e-9:
        result.sharpe = float(rets.mean() / rets.std() * math.sqrt(365))
    cmax = np.maximum.accumulate(eq)
    result.max_dd = float(((eq - cmax) / cmax).min())
    result.top_k_hit_rate = hit_count / max(realized_top_total, 1)
    return result


# =============================================================================
# STRESS MODE — scenarios
# =============================================================================

STRESS_SCENARIOS = {
    "btc_flash_crash": {
        "description": "BTC drops 30% in 48h",
        "btc_shock": -0.30,
        "duration_days": 2,
        "altcoin_shock_factor": 1.5,   # alts drop 1.5x BTC
    },
    "funding_spike": {
        "description": "Funding rate spikes to 200% APR",
        "btc_shock": 0.0,
        "funding_shock_apr": 2.0,
        "duration_days": 3,
    },
    "regime_shift": {
        "description": "Bull -> bear in 3 days",
        "btc_shock": -0.20,
        "duration_days": 3,
        "regime_correlation_break": True,
    },
    "exchange_outage": {
        "description": "1 of 3 exchanges down for 6h",
        "btc_shock": 0.0,
        "execution_failure_rate": 0.33,
        "duration_days": 0.25,
    },
}


def probe_stress(strategy_name: str, scenario_name: str) -> StrategyProbeResult:
    """Run synthetic stress scenario; measure strategy drawdown + recovery."""
    result = StrategyProbeResult(mode="stress", strategy=strategy_name)
    scenario = STRESS_SCENARIOS.get(scenario_name)
    if scenario is None:
        result.failure_mode = f"unknown scenario: {scenario_name}"
        return result
    # STUB: simulated strategy response
    # Real integration: hook the strategy's `plan_day` and feed shocked panel
    np.random.seed(42)
    n_days = 30
    daily_rets = np.random.normal(0.005, 0.015, n_days)   # baseline
    # Apply shock
    shock_start = 10
    shock_dur = int(scenario.get("duration_days", 1))
    btc_shock = scenario.get("btc_shock", 0.0)
    altcoin_factor = scenario.get("altcoin_shock_factor", 1.0)
    daily_rets[shock_start:shock_start + shock_dur] = btc_shock * altcoin_factor / shock_dur

    eq = np.cumprod(1 + daily_rets)
    cmax = np.maximum.accumulate(eq)
    dd = (eq - cmax) / cmax
    max_dd = float(dd.min())
    # Recovery time: bars to return within 5% of peak
    recovery_idx = None
    for i in range(shock_start + shock_dur, n_days):
        if eq[i] >= cmax[shock_start - 1] * 0.95:
            recovery_idx = i - shock_start
            break

    result.n_days = n_days
    result.max_dd = max_dd
    result.cum_ret = float(eq[-1] - 1)
    result.failure_mode = (f"scenario={scenario['description']} | "
                            f"max_dd={max_dd*100:.1f}% | "
                            f"recovery_days={recovery_idx if recovery_idx else 'none'}")
    return result


# =============================================================================
# Main
# =============================================================================

def format_result(r: StrategyProbeResult) -> str:
    flags = []
    if r.failure_mode:
        flags.append(f"FAIL/INFO: {r.failure_mode[:50]}")
    if r.mean_daily_ret >= 0.01:
        flags.append("HIT_1PCT")
    if r.mean_daily_ret >= 0.02:
        flags.append("HIT_2PCT")
    flag_str = "  ".join(flags) or "-"
    return (
        f"  {r.strategy:<20} mode={r.mode:<10} n={r.n_days:>4} "
        f"mean={r.mean_daily_ret:+.4f}/day  median={r.median_daily_ret:+.4f}  "
        f"days>=1%={r.days_hit_1pct}  >=2%={r.days_hit_2pct}  >=5%={r.days_hit_5pct}  "
        f"sharpe={r.sharpe:+.2f}  cum={r.cum_ret*100:+.1f}%  dd={r.max_dd*100:+.1f}%  "
        f"top_k_hit={r.top_k_hit_rate:.2f}  {flag_str}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="real",
                        choices=["real", "synthetic", "stress"])
    parser.add_argument("--strategy", type=str, default="xsec_K5_5",
                        help="Strategy label for output. Ranker chosen by --ranker.")
    parser.add_argument("--ranker", type=str, default="momentum",
                        choices=["momentum", "vol", "random"],
                        help="momentum=top-K by t-1 ret; vol=top-K by |t-1 ret|; "
                             "random=null-baseline.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default="2026-04-30")
    parser.add_argument("--scenario", type=str, default="btc_flash_crash")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 100)
    print(f"  STRATEGY CEILING PROBE — mode={args.mode} strategy={args.strategy}")
    print("=" * 100)

    if args.mode == "real":
        r = probe_real(args.strategy, args.top_k, args.start, args.end,
                        args.seed, ranker=args.ranker)
    elif args.mode == "stress":
        r = probe_stress(args.strategy, args.scenario)
    else:
        print(f"  [TODO] synthetic mode")
        return

    print(format_result(r), flush=True)
    print()
    if r.n_days > 0:
        print(f"  Days hitting target tiers (n={r.n_days}):")
        print(f"    >=1%: {r.days_hit_1pct} ({100*r.days_hit_1pct/r.n_days:.1f}%)")
        print(f"    >=2%: {r.days_hit_2pct} ({100*r.days_hit_2pct/r.n_days:.1f}%)")
        print(f"    >=5%: {r.days_hit_5pct} ({100*r.days_hit_5pct/r.n_days:.1f}%)")
        print(f"  Top-{args.top_k} hit rate: {r.top_k_hit_rate*100:.1f}%")
        print(f"  Cumulative return: {r.cum_ret*100:+.2f}%")
        print(f"  Sharpe: {r.sharpe:+.2f}  MaxDD: {r.max_dd*100:+.1f}%")


if __name__ == "__main__":
    main()
