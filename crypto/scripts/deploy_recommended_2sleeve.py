"""Deployment + paper-trading driver for recommended_2sleeve blend.

Wires together the two sleeves that comprise the PRIMARY 2026-04-23 recommendation
(docs/DEPLOYMENT_RECOMMENDATION_2026_04_23.md):

    Book A (70%): xsec_K5_5_FULL_dneut (scratch/xsec_variants_daily_equity.py)
    Book B (30%): frontier_dib_flow_both (src/frontier/strategies/dib_flow_duo.py)

    Measured aligned 2025-01-01 -> 2026-04-13:
      CAGR 117.55% | Sharpe 4.96 | DD -5.05% | Calmar 23.26
      Time-to-10x preFriction 2.96y | postFriction (0.5x haircut) 5.4y

Why a driver? Each sleeve is its own full-replay script (not paper_trader_v2's
incremental state model). This driver extends both to the latest available data,
then runs the portfolio aggregator to produce blended equity, metrics, and a
DD-halt check. Running this daily IS the paper-trading loop.

Daily operator flow:
    1.  Optionally refresh Binance data:
            python src/pipeline/fetch_all.py
            python src/pipeline/make_dataset_legacy.py
        Or pass --refresh to do both automatically.
    2.  Optionally refresh DIB bars:
            python src/frontier/pipeline/dib_bars_fast.py --asset BTCUSDT --year 2025
            python src/frontier/pipeline/dib_bars_fast.py --asset ETHUSDT --year 2025
        Or pass --rebuild-dib.
    3.  Run this driver:
            python scripts/deploy_recommended_2sleeve.py
        (default: regenerate both sleeve snapshots + aggregate + report)

Exit codes (propagated from aggregator):
    0 = OK
    1 = missing sleeves (data problem)
    2 = DD halt triggered (portfolio DD < halt threshold -- PAUSE DEPLOYMENT)
    3 = unexpected error

Flags:
    --capital N             Capital display scale ($10K baseline used internally)
    --blend NAME            Override blend name (default: recommended_2sleeve)
    --dd-halt X             Override DD halt % (default: -15; yaml: -15)
    --test-end YYYY-MM-DD   Freeze simulation end date (default: all available)
    --test-start YYYY-MM-DD Override simulation start (default: 2025-01-01)
    --refresh               Run fetch_all + make_dataset_legacy BEFORE sim
    --rebuild-dib           Rebuild BTC+ETH DIB bars BEFORE sim
    --skip-xsec             Skip xsec sleeve regen (use existing snapshot)
    --skip-dib              Skip dib sleeve regen (use existing snapshot)
    --aggregate-only        Skip both regens, just run aggregator
    --dry-run               Show what WOULD run, do not execute
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

XSEC_SCRIPT = ROOT / "scratch" / "xsec_variants_daily_equity.py"
DIB_SCRIPT = ROOT / "src" / "frontier" / "strategies" / "dib_flow_duo.py"
DIB_BARS_SCRIPT = ROOT / "src" / "frontier" / "pipeline" / "dib_bars_fast.py"
FETCH_SCRIPT = ROOT / "src" / "pipeline" / "fetch_all.py"
DATASET_SCRIPT = ROOT / "src" / "pipeline" / "make_dataset_legacy.py"
AGGREGATOR_SCRIPT = ROOT / "src" / "analysis" / "portfolio_aggregator.py"

DEPLOY_LOG_DIR = ROOT / "logs" / "deployment"


def _run(cmd: list, env: dict | None = None, dry_run: bool = False) -> int:
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n[RUN] {pretty}")
    if dry_run:
        print("      (dry-run, skipped)")
        return 0
    t0 = time.time()
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    rc = subprocess.call(cmd, env=full_env)
    dt = time.time() - t0
    print(f"[DONE] rc={rc}  ({dt:.1f}s)")
    return rc


def refresh_data(dry_run: bool) -> int:
    print("\n>>> Phase 1: refresh Binance data (fetch_all + make_dataset_legacy)")
    rc = _run([PYTHON, str(FETCH_SCRIPT)], dry_run=dry_run)
    if rc != 0:
        print(f"[WARN] fetch_all exited rc={rc}; continuing to dataset build")
    return _run([PYTHON, str(DATASET_SCRIPT)], dry_run=dry_run)


def rebuild_dib(dry_run: bool) -> int:
    print("\n>>> Phase 2: rebuild DIB bars (BTC + ETH)")
    rc1 = _run([PYTHON, str(DIB_BARS_SCRIPT), "--asset", "BTCUSDT", "--year", "2025"],
               dry_run=dry_run)
    rc2 = _run([PYTHON, str(DIB_BARS_SCRIPT), "--asset", "ETHUSDT", "--year", "2025"],
               dry_run=dry_run)
    return max(rc1, rc2)


def run_xsec(test_start: str | None, test_end: str | None, dry_run: bool) -> int:
    print("\n>>> Phase 3a: regenerate xsec_K5_5_FULL_dneut snapshot (70% weight)")
    env = {}
    if test_start:
        env["PT_TEST_START"] = test_start
    if test_end:
        env["PT_TEST_END"] = test_end
    return _run([PYTHON, str(XSEC_SCRIPT)], env=env, dry_run=dry_run)


def run_dib(test_start: str | None, test_end: str | None, dry_run: bool) -> int:
    print("\n>>> Phase 3b: regenerate frontier_dib_flow_both snapshot (30% weight)")
    env = {}
    if test_start:
        env["PT_TEST_START"] = test_start
    if test_end:
        env["PT_TEST_END"] = test_end
    return _run([PYTHON, str(DIB_SCRIPT)], env=env, dry_run=dry_run)


def run_aggregator(blend: str, dd_halt: float, dry_run: bool) -> int:
    print(f"\n>>> Phase 4: aggregate blend={blend} dd_halt={dd_halt}%")
    return _run([PYTHON, str(AGGREGATOR_SCRIPT),
                 "--blend", blend, "--dd-halt", str(dd_halt)],
                dry_run=dry_run)


def log_run(blend: str, capital: float, agg_rc: int, phases_run: list):
    DEPLOY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now(timezone.utc).date()
    run_dir = DEPLOY_LOG_DIR / str(today)
    run_dir.mkdir(parents=True, exist_ok=True)
    # Pull aggregator summary for this blend
    summary_path = ROOT / "logs" / "portfolio_aggregator" / f"{blend}_summary.json"
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    record = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "blend": blend,
        "capital_display": capital,
        "phases_run": phases_run,
        "aggregator_rc": agg_rc,
        "halt_triggered": bool(summary.get("halt_triggered", False)),
        "summary": summary,
    }
    out = run_dir / f"{blend}_run.json"
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n[deploy] run record: {out}")
    return record


def print_operator_summary(record: dict, capital: float):
    s = record.get("summary", {})
    if not s:
        print("[deploy] WARNING: no aggregator summary available")
        return
    scale = capital / 10000.0
    print("\n" + "=" * 80)
    print(f"OPERATOR SUMMARY -- {record['blend']}")
    print("=" * 80)
    print(f"  Period:            {s.get('period_start', '?')} -> {s.get('period_end', '?')} "
          f"({s.get('n_days', '?')} days)")
    print(f"  Capital (display): ${capital:,.2f}  (internal baseline $10,000)")
    nav = s.get("ending_nav", 0.0) * scale
    print(f"  Ending NAV:        ${nav:,.2f}  ({s.get('total_return_pct', 0):+.2f}%)")
    print(f"  CAGR:              {s.get('cagr_pct', 0):+.2f}%/yr")
    print(f"  Sharpe / Sortino:  {s.get('sharpe', 0):.2f} / "
          f"{s.get('sortino') if s.get('sortino') is not None else 'inf'}")
    print(f"  Max DD:            {s.get('max_dd_pct', 0):+.2f}%  "
          f"(halt threshold {s.get('dd_halt_threshold_pct', 0):.0f}%)")
    print(f"  Calmar:            {s.get('calmar') if s.get('calmar') is not None else 'inf'}")
    if record["halt_triggered"]:
        print("\n  [!!! HALT TRIGGERED !!!] Pause all deployment pending review.")
    else:
        print("\n  [OK] Within DD halt threshold. Continue daily updates.")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capital", type=float, default=10000.0,
                    help="Capital display scale (internal sim uses $10K). Default 10000.")
    ap.add_argument("--blend", default="recommended_2sleeve",
                    help="Blend name from config/deployment_ranking.yaml. Default recommended_2sleeve.")
    ap.add_argument("--dd-halt", type=float, default=-15.0,
                    help="Portfolio DD halt threshold %%. Default -15 (matches yaml for 2sleeve).")
    ap.add_argument("--test-start", type=str, default=None,
                    help="Override PT_TEST_START (default 2025-01-01 from script).")
    ap.add_argument("--test-end", type=str, default=None,
                    help="Override PT_TEST_END (default: all available data).")
    ap.add_argument("--refresh", action="store_true", help="Run fetch_all + make_dataset_legacy first")
    ap.add_argument("--rebuild-dib", action="store_true", help="Rebuild BTC+ETH DIB bars first")
    ap.add_argument("--skip-xsec", action="store_true", help="Skip xsec sleeve regen")
    ap.add_argument("--skip-dib", action="store_true", help="Skip dib sleeve regen")
    ap.add_argument("--aggregate-only", action="store_true", help="Skip regens; aggregate only")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = ap.parse_args()

    print("=" * 80)
    print(f"DEPLOYMENT DRIVER -- recommended_2sleeve paper trading")
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    print(f"Blend: {args.blend}  |  Capital (display): ${args.capital:,.2f}  |  DD halt: {args.dd_halt}%")
    print(f"Test window: {args.test_start or '2025-01-01'} -> {args.test_end or 'ALL_AVAILABLE'}")
    print("=" * 80)

    phases_run = []

    if args.refresh and not args.aggregate_only:
        rc = refresh_data(args.dry_run)
        phases_run.append({"phase": "refresh_data", "rc": rc})
        if rc != 0:
            print("[deploy] refresh_data failed; aborting before sleeve regens")
            sys.exit(3)

    if args.rebuild_dib and not args.aggregate_only:
        rc = rebuild_dib(args.dry_run)
        phases_run.append({"phase": "rebuild_dib", "rc": rc})
        if rc != 0:
            print("[deploy] rebuild_dib failed; continuing (sleeves will use stale bars)")

    if not args.aggregate_only and not args.skip_xsec:
        rc = run_xsec(args.test_start, args.test_end, args.dry_run)
        phases_run.append({"phase": "xsec", "rc": rc})
        if rc != 0:
            print("[deploy] xsec regen failed; aggregator will use stale snapshot")

    if not args.aggregate_only and not args.skip_dib:
        rc = run_dib(args.test_start, args.test_end, args.dry_run)
        phases_run.append({"phase": "dib", "rc": rc})
        if rc != 0:
            print("[deploy] dib regen failed; aggregator will use stale snapshot")

    agg_rc = run_aggregator(args.blend, args.dd_halt, args.dry_run)
    phases_run.append({"phase": "aggregator", "rc": agg_rc})

    if not args.dry_run:
        record = log_run(args.blend, args.capital, agg_rc, phases_run)
        print_operator_summary(record, args.capital)

    sys.exit(agg_rc)


if __name__ == "__main__":
    main()
