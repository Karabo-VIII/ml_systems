"""Divergence monitor: live paper PnL vs audit baseline projection.

Concept: each blend has an audited backtest CAGR (from config/deployment_ranking.yaml
measured_cagr_pct). Once paper trading starts on day T0, we expect the live equity
curve to hug that CAGR line. Significant deviation after N weeks -> something
broke (data, regime, fill model, model stale).

Reads:
    logs/portfolio_aggregator/<blend>_daily.csv
    config/deployment_ranking.yaml (measured_cagr_pct per blend)

Compares:
    Live cumulative return since --since-date vs ex-ante projection
    CAGR_window * days / 365 = expected return over the window.

Usage:
    python scripts/divergence_monitor.py --blend recommended_2sleeve --since-date 2026-04-23
    python scripts/divergence_monitor.py --all

Flags:
    --alert-threshold-pct X    Warn if |live - projected| > X (default 20%)
    --since-date YYYY-MM-DD    Paper-trading epoch. Default: 4 weeks ago.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "deployment_ranking.yaml"
AGG_DIR = ROOT / "logs" / "portfolio_aggregator"
OUT_DIR = ROOT / "logs" / "deployment" / "divergence"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_blend_equity(blend: str) -> pd.DataFrame | None:
    p = AGG_DIR / f"{blend}_daily.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def measure(blend: str, since_date: str, alert_pct: float) -> dict:
    cfg = load_config()
    bspec = cfg.get("deployment", {}).get(blend, {})
    # Baseline CAGR from measured field
    baseline_cagr = bspec.get("measured_cagr_pct")
    baseline_sharpe = bspec.get("measured_sharpe")
    baseline_dd = bspec.get("measured_dd_pct")
    if baseline_cagr is None:
        # Fall back to preFriction expected
        baseline_cagr = bspec.get("expected_cagr_preFriction_pct")
    if baseline_cagr is None:
        return {"blend": blend, "status": "no_baseline_cagr_in_config"}

    df = load_blend_equity(blend)
    if df is None or df.empty:
        return {"blend": blend, "status": "no_equity_csv", "expected_path": str(AGG_DIR / f"{blend}_daily.csv")}

    since = pd.to_datetime(since_date).date()
    sub = df[df["date"] >= since].copy()
    if len(sub) < 2:
        return {
            "blend": blend, "status": "insufficient_live_days",
            "baseline_cagr_pct": baseline_cagr,
            "first_live_day": str(df["date"].iloc[0]),
            "last_live_day": str(df["date"].iloc[-1]),
        }

    eq = sub["portfolio_equity"].values
    live_ret_pct = float((eq[-1] / eq[0] - 1) * 100)
    days_span = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days or 1
    projected_ret_pct = ((1 + baseline_cagr / 100.0) ** (days_span / 365.0) - 1) * 100
    divergence_pct = live_ret_pct - projected_ret_pct
    divergence_abs = abs(divergence_pct)
    # Also live max DD vs baseline DD
    cum_max = pd.Series(eq).cummax().values
    live_dd_pct = float(((eq - cum_max) / cum_max).min() * 100)
    status = "OK"
    flags = []
    if divergence_abs > alert_pct:
        status = "ALERT"
        flags.append(f"ret_divergence_{divergence_abs:.1f}pp")
    if baseline_dd is not None and live_dd_pct < baseline_dd - 5:
        status = "ALERT"
        flags.append(f"dd_worse_{live_dd_pct:.1f}_vs_{baseline_dd:.1f}")
    return {
        "blend": blend,
        "status": status,
        "flags": flags,
        "baseline_cagr_pct": baseline_cagr,
        "baseline_sharpe": baseline_sharpe,
        "baseline_dd_pct": baseline_dd,
        "live_days": len(sub),
        "period_start": str(sub["date"].iloc[0]),
        "period_end": str(sub["date"].iloc[-1]),
        "live_ret_pct": live_ret_pct,
        "projected_ret_pct": projected_ret_pct,
        "divergence_pct": divergence_pct,
        "live_dd_pct": live_dd_pct,
        "alert_threshold_pct": alert_pct,
    }


def print_report(records: list):
    print("\n" + "=" * 95)
    print("DIVERGENCE MONITOR -- live paper PnL vs audited-baseline projection")
    print("=" * 95)
    print(f"{'blend':<36} {'status':<8} {'days':>5} {'live%':>8} {'proj%':>8} {'diff pp':>8} {'liveDD%':>8}")
    for r in records:
        if r.get("status") in ("no_baseline_cagr_in_config", "no_equity_csv"):
            print(f"{r['blend']:<36} {'N/A':<8}  -- {r['status']}")
            continue
        if r.get("status") == "insufficient_live_days":
            print(f"{r['blend']:<36} {'PENDING':<8}  -- {r.get('last_live_day')} (need more days)")
            continue
        print(f"{r['blend']:<36} {r['status']:<8} {r['live_days']:>5} "
              f"{r['live_ret_pct']:>+7.2f} {r['projected_ret_pct']:>+7.2f} "
              f"{r['divergence_pct']:>+7.2f} {r['live_dd_pct']:>+7.2f}")
        if r.get("flags"):
            print(f"      flags: {', '.join(r['flags'])}")
    print("=" * 95)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blend", help="Single blend to check")
    ap.add_argument("--all", action="store_true",
                    help="Check all blends in config/deployment_ranking.yaml deployment section")
    ap.add_argument("--since-date", default=None,
                    help="Paper trading epoch date (YYYY-MM-DD). Default: 28 days ago.")
    ap.add_argument("--alert-threshold-pct", type=float, default=20.0,
                    help="Percentage points of return divergence that triggers ALERT.")
    args = ap.parse_args()

    since = args.since_date or str((datetime.now(timezone.utc).date() - timedelta(days=28)))

    if args.blend:
        blends = [args.blend]
    elif args.all:
        cfg = load_config()
        blends = list(cfg.get("deployment", {}).keys())
    else:
        blends = ["recommended_2sleeve"]

    records = [measure(b, since, args.alert_threshold_pct) for b in blends]
    print_report(records)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"divergence_{datetime.now(timezone.utc).date()}.json"
    with open(out_path, "w") as f:
        json.dump({"run_utc": datetime.now(timezone.utc).isoformat(),
                   "since_date": since, "alert_pct": args.alert_threshold_pct,
                   "records": records}, f, indent=2, default=str)
    print(f"\n[info] report: {out_path}")

    has_alert = any(r.get("status") == "ALERT" for r in records)
    sys.exit(1 if has_alert else 0)


if __name__ == "__main__":
    main()
