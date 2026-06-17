"""Deploy + paper-trade ALL top PROD-ready strategies, aggregate, and report.

What this is:
    A single driver that extends every ship-listed sleeve forward using current
    data, aggregates the blends they roll up into, and produces a consolidated
    performance report over a user-chosen window.

Strategies covered (PROD-ready, post 2026-04-23 audit):
    STANDALONE SLEEVES (script-driven; regen'd here)
      xsec_K5_5_FULL_dneut   -- best Sharpe sleeve      70% weight in primary blend
      xgb_K3_long_WEALTH40   -- wealth amplifier        long-only 3-pick cross-sec
      cat_K1_stop_no_macro   -- aggressive, DD -67%     1-pick top-rank + 10% stop
      frontier_dib_flow_both -- flow-triggered BTC/ETH  30% weight in primary blend
      frontier_dib_flow_any  -- looser trigger variant  audit reference
      frontier_dib_flow_avg  -- averaged trigger        audit reference

    BLENDS (aggregator'd here; require the sleeves above to be up to date)
      recommended_2sleeve              -- PRIMARY (Sh 4.96, CAGR 117%, DD -5%)
      recommended_3sleeve_wealth_max   -- WEALTH (CAGR 142%, DD -33%, halt -40%)
      recommended_3sleeve_conservative -- CONSERVATIVE (Sh 5.30, DD -3.9%)
           ^ includes prod_meta_combined which needs paper_trader_v2 --update run
             separately -- driver warns and skips aggregator if stale.
      full_stack_v7_frontier           -- 8-sleeve reference (higher blend, more ops)

Usage:
    # Full pipeline (fetch -> rebuild bars -> regen all sleeves -> aggregate ALL)
    python scripts/deploy_all_top_strats.py --refresh --rebuild-dib

    # Skip data refresh; regen sleeves + aggregate with current data
    python scripts/deploy_all_top_strats.py

    # Restrict xsec ranker to 10-asset core universe (sets PT_UNIVERSE)
    python scripts/deploy_all_top_strats.py --universe-10

    # Freeze the window via env
    python scripts/deploy_all_top_strats.py --test-start 2026-04-01 --test-end 2026-04-23

    # Generate per-window summary without regen (aggregate + report only)
    python scripts/deploy_all_top_strats.py --aggregate-only

Output:
    logs/deployment/<UTC-date>/all_top_strats_report.json  -- consolidated record
    logs/portfolio_aggregator/<blend>_{daily.csv,summary.json}  -- per-blend
    logs/paper_trader_v2/seeds/pt_*/daily_snapshot.csv     -- per-sleeve

Exit codes:
    0 OK  |  1 missing sleeves  |  2 HALT triggered (some blend DD breached)
    3 unexpected error
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
PY = sys.executable

XSEC_SCRIPT = ROOT / "scratch" / "xsec_variants_daily_equity.py"
DIB_SCRIPT = ROOT / "src" / "frontier" / "strategies" / "dib_flow_duo.py"
DIB_BARS_SCRIPT = ROOT / "src" / "frontier" / "pipeline" / "dib_bars_fast.py"
FETCH_SCRIPT = ROOT / "src" / "pipeline" / "fetch_all.py"
DATASET_SCRIPT = ROOT / "src" / "pipeline" / "make_dataset_legacy.py"
AGGREGATOR_SCRIPT = ROOT / "src" / "analysis" / "portfolio_aggregator.py"

CORE_10 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]
CORE_10_PAIRS = [f"{a}/USDT" for a in CORE_10]

# Standalone sleeves with the seed dir they write to.
# NOTE: (profile, seed, regen_type) — regen_type determines who writes to the seed:
#   "xsec"        -> scratch/xsec_variants_daily_equity.py
#   "dib"         -> src/frontier/strategies/dib_flow_duo.py
#   "frontier"    -> scripts/refresh_frontier_overlays.py (stable + etf)
#   "paper_v2"    -> src/analysis/paper_trader_v2.py --update --seed <seed>
#                    (pre-existing; may go stale if its update path is broken)
SLEEVES = [
    # Top PROD candidates -- xsec ranker family
    ("xsec_K5_5_FULL_dneut",   "pt_xsec_K5_5_FULL_dneut",   "xsec"),
    ("xgb_K3_long_WEALTH40",   "pt_xgb_K3_long_WEALTH40",   "xsec"),
    ("cat_K1_stop_no_macro",   "pt_cat_K1_stop_no_macro",   "xsec"),
    # DIB flow duo (BTC+ETH, 3 trigger variants)
    ("frontier_dib_flow_both", "pt_frontier_dib_flow_both", "dib"),
    ("frontier_dib_flow_any",  "pt_frontier_dib_flow_any",  "dib"),
    ("frontier_dib_flow_avg",  "pt_frontier_dib_flow_avg",  "dib"),
    # Meta-labeled paper_trader_v2 seeds (S-tier + A-tier champions)
    ("prod_meta_combined",     "pt_meta_combined",          "paper_v2"),
    ("prod_meta_full",         "pt_meta_full",              "paper_v2"),
    # Frontier overlays (flow-based, canonical champion variants)
    ("frontier_stable_flow",   "pt_frontier_stable_flow",   "frontier"),
    ("frontier_etf_flow",      "pt_frontier_etf_flow",      "frontier"),
]

# Blends to aggregate. dd_halt keys match config yaml defaults where possible.
BLENDS_TO_AGGREGATE = [
    ("recommended_2sleeve",              -15.0,  "PRIMARY"),
    ("recommended_3sleeve_wealth_max",   -40.0,  "WEALTH-MAX"),
    ("recommended_3sleeve_conservative", -12.0,  "CONSERVATIVE"),
    ("full_stack_v7_frontier",           -25.0,  "REFERENCE"),
]

DEPLOY_LOG_DIR = ROOT / "logs" / "deployment"


def _run(cmd: list, env_override: dict | None = None, dry_run: bool = False) -> int:
    pretty = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
    print(f"\n[RUN] {pretty}")
    if dry_run:
        print("      (dry-run, skipped)")
        return 0
    t0 = time.time()
    env = os.environ.copy()
    if env_override:
        env.update(env_override)
        print(f"      env: {env_override}")
    rc = subprocess.call(cmd, env=env)
    dt = time.time() - t0
    print(f"[DONE] rc={rc}  ({dt:.1f}s)")
    return rc


def refresh_data(dry_run: bool, pairs: list[str]) -> int:
    print("\n>>> Phase 1a: Binance data refresh")
    rc = _run([PY, str(FETCH_SCRIPT), "--assets", *pairs], dry_run=dry_run)
    if rc != 0:
        print(f"[WARN] fetch_all rc={rc}; continuing to dataset rebuild")
    print("\n>>> Phase 1b: rebuild chimera dataset")
    return _run([PY, str(DATASET_SCRIPT)], dry_run=dry_run)


def rebuild_dib(dry_run: bool, assets: list[str], start: str, end: str) -> int:
    print("\n>>> Phase 2: rebuild DIB bars")
    pairs = [f"{a}USDT" for a in assets]
    return _run([PY, str(DIB_BARS_SCRIPT), "--assets", *pairs,
                 "--start", start, "--end", end], dry_run=dry_run)


def regen_xsec(env_overrides: dict, dry_run: bool) -> int:
    print("\n>>> Phase 3a: regen xsec variants (xsec_K5_5_FULL_dneut, xgb_K3_long_WEALTH40, cat_K1_stop_no_macro)")
    return _run([PY, str(XSEC_SCRIPT)], env_override=env_overrides, dry_run=dry_run)


def regen_dib(env_overrides: dict, dry_run: bool) -> int:
    print("\n>>> Phase 3b: regen frontier_dib_flow_{both,any,avg}")
    return _run([PY, str(DIB_SCRIPT)], env_override=env_overrides, dry_run=dry_run)


def run_aggregator(blend: str, dd_halt: float, dry_run: bool) -> int:
    print(f"\n>>> Aggregate blend={blend}  dd_halt={dd_halt}%")
    return _run([PY, str(AGGREGATOR_SCRIPT),
                 "--blend", blend, "--dd-halt", str(dd_halt)], dry_run=dry_run)


def load_sleeve_snapshot(seed_name: str) -> list | None:
    """Return list of dict rows or None if not readable."""
    import pandas as pd
    p = ROOT / "logs" / "paper_trader_v2" / "seeds" / seed_name / "daily_snapshot.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    return df.to_dict("records")


def slice_and_metrics(rows: list, start_date: str, end_date: str):
    """Slice equity rows to [start, end] and compute period metrics."""
    import numpy as np
    import pandas as pd

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    sub = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if len(sub) < 2:
        return {"n_days": len(sub), "period_start": start_date, "period_end": end_date}
    eq = sub["total_equity"].values.astype(float)
    start_nav = eq[0]
    end_nav = eq[-1]
    total_ret = (end_nav / start_nav - 1) * 100
    n_days = len(eq)
    # Period days approximation: actual calendar span
    actual_start = sub["date"].iloc[0]
    actual_end = sub["date"].iloc[-1]
    days_span = (pd.to_datetime(actual_end) - pd.to_datetime(actual_start)).days or 1
    cagr = ((end_nav / start_nav) ** (365.0 / days_span) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0
    cum_max = np.maximum.accumulate(eq)
    dd = (eq - cum_max) / cum_max
    max_dd = dd.min() * 100
    open_pos = int(sub.get("swing_open_positions", pd.Series([0])).astype(int).max())
    return {
        "n_days": n_days,
        "period_start": actual_start,
        "period_end": actual_end,
        "start_nav": float(start_nav),
        "end_nav": float(end_nav),
        "total_ret_pct": float(total_ret),
        "cagr_pct": float(cagr),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd),
        "bars_with_open_pos": int((sub.get("swing_open_positions", pd.Series([0])).astype(int) > 0).sum()),
        "max_open_pos": open_pos,
    }


def load_blend_summary(blend: str) -> dict:
    p = ROOT / "logs" / "portfolio_aggregator" / f"{blend}_summary.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def load_blend_daily(blend: str):
    import pandas as pd
    p = ROOT / "logs" / "portfolio_aggregator" / f"{blend}_daily.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def window_metrics_from_blend_daily(blend: str, start_date: str, end_date: str):
    import numpy as np
    import pandas as pd
    df = load_blend_daily(blend)
    if df is None:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    sub = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    if len(sub) < 2:
        return {"n_days": len(sub)}
    eq = sub["portfolio_equity"].values.astype(float)
    total_ret = (eq[-1] / eq[0] - 1) * 100
    actual_start = sub["date"].iloc[0]
    actual_end = sub["date"].iloc[-1]
    days_span = (pd.to_datetime(actual_end) - pd.to_datetime(actual_start)).days or 1
    cagr = ((eq[-1] / eq[0]) ** (365.0 / days_span) - 1) * 100
    dr = np.diff(eq) / eq[:-1]
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0
    cum_max = np.maximum.accumulate(eq)
    max_dd = ((eq - cum_max) / cum_max).min() * 100
    return {
        "n_days": len(eq),
        "period_start": actual_start,
        "period_end": actual_end,
        "start_nav": float(eq[0]),
        "end_nav": float(eq[-1]),
        "total_ret_pct": float(total_ret),
        "cagr_pct": float(cagr),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd),
    }


def print_report(report: dict, window_start: str, window_end: str, capital: float):
    scale = capital / 10000.0
    print("\n" + "=" * 100)
    print(f"CONSOLIDATED REPORT -- window {window_start} -> {window_end}   (display capital ${capital:,.2f})")
    print("=" * 100)

    print("\nPER-SLEEVE STANDALONE PERFORMANCE (window slice):")
    print("-" * 100)
    print(f"{'sleeve':<28} {'days':>5} {'start$':>10} {'end$':>10} {'ret%':>8} {'Sh':>6} {'DD%':>7} {'posBars':>8}")
    for s in report["sleeves"]:
        m = s.get("window_metrics") or {}
        if not m or "start_nav" not in m:
            print(f"{s['seed']:<28} {m.get('n_days','?'):>5}  (insufficient data)")
            continue
        print(f"{s['seed']:<28} {m['n_days']:>5} {m['start_nav']*scale:>10.2f} {m['end_nav']*scale:>10.2f} "
              f"{m['total_ret_pct']:>+7.2f} {m['sharpe']:>+5.2f} {m['max_dd_pct']:>+6.2f} {m.get('bars_with_open_pos',0):>8}")

    print("\nPER-BLEND AGGREGATED PERFORMANCE (full-history baseline + window slice):")
    print("-" * 100)
    print(f"{'blend':<35} {'status':<14} {'full Sh':>8} {'full DD%':>10} {'win days':>9} "
          f"{'win ret%':>10} {'win Sh':>8} {'win DD%':>9}")
    for b in report["blends"]:
        full = b.get("full_summary") or {}
        win = b.get("window_metrics") or {}
        if not full:
            print(f"{b['blend']:<35} {'MISSING':<14}")
            continue
        wr = win.get("total_ret_pct", 0.0)
        ws = win.get("sharpe", 0.0)
        wd = win.get("max_dd_pct", 0.0)
        wn = win.get("n_days", 0)
        fs = full.get("sharpe", 0.0)
        fd = full.get("max_dd_pct", 0.0)
        halt = " HALT" if full.get("halt_triggered") else ""
        print(f"{b['blend']:<35} {b['tag']:<14} {fs:>+7.2f} {fd:>+9.2f}{halt:<5} {wn:>9} "
              f"{wr:>+9.2f} {ws:>+7.2f} {wd:>+8.2f}")

    # Champion
    print("\n" + "=" * 100)
    print("WINDOW CHAMPIONS")
    print("-" * 100)
    sleeves_valid = [s for s in report["sleeves"] if s.get("window_metrics", {}).get("start_nav")]
    blends_valid = [b for b in report["blends"] if b.get("window_metrics", {}).get("start_nav")]
    if sleeves_valid:
        cs = max(sleeves_valid, key=lambda s: s["window_metrics"]["total_ret_pct"])
        print(f"  Best sleeve by return:  {cs['seed']:<30} {cs['window_metrics']['total_ret_pct']:+.2f}% "
              f"Sh {cs['window_metrics']['sharpe']:+.2f}")
        cs = max(sleeves_valid, key=lambda s: s["window_metrics"]["sharpe"])
        print(f"  Best sleeve by Sharpe:  {cs['seed']:<30} Sh {cs['window_metrics']['sharpe']:+.2f} "
              f"ret {cs['window_metrics']['total_ret_pct']:+.2f}%")
    if blends_valid:
        cb = max(blends_valid, key=lambda b: b["window_metrics"]["total_ret_pct"])
        print(f"  Best blend by return:   {cb['blend']:<30} {cb['window_metrics']['total_ret_pct']:+.2f}% "
              f"Sh {cb['window_metrics']['sharpe']:+.2f}")
        cb = max(blends_valid, key=lambda b: b["window_metrics"]["sharpe"])
        print(f"  Best blend by Sharpe:   {cb['blend']:<30} Sh {cb['window_metrics']['sharpe']:+.2f} "
              f"ret {cb['window_metrics']['total_ret_pct']:+.2f}%")
    print("=" * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="Binance data fetch + chimera rebuild")
    ap.add_argument("--rebuild-dib", action="store_true", help="Rebuild DIB bars for 10-core")
    ap.add_argument("--skip-xsec", action="store_true")
    ap.add_argument("--skip-dib", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip sleeve regen; aggregate + report only")
    ap.add_argument("--test-start", default=None, help="PT_TEST_START override")
    ap.add_argument("--test-end", default=None, help="PT_TEST_END override")
    ap.add_argument("--window-start", default="2026-04-01", help="Reporting window start")
    ap.add_argument("--window-end", default=None,
                    help="Reporting window end (default: today UTC)")
    ap.add_argument("--universe-10", action="store_true",
                    help="Restrict xsec ranker to 10-core via PT_UNIVERSE")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    today = datetime.now(timezone.utc).date()
    window_end = args.window_end or str(today)

    print("=" * 100)
    print("DEPLOY ALL TOP STRATS -- kick-off report")
    print(f"UTC now: {datetime.now(timezone.utc).isoformat()}")
    print(f"Window:  {args.window_start} -> {window_end}")
    print(f"Capital (display): ${args.capital:,.2f}   internal baseline $10,000")
    print(f"Universe: {'10-core ' + ','.join(CORE_10) if args.universe_10 else 'full (all chimera files)'}")
    print("=" * 100)

    phases = []

    if args.refresh and not args.aggregate_only:
        rc = refresh_data(args.dry_run, CORE_10_PAIRS)
        phases.append({"phase": "refresh", "rc": rc})

    if args.rebuild_dib and not args.aggregate_only:
        rc = rebuild_dib(args.dry_run, CORE_10, "2025-01-01", str(today))
        phases.append({"phase": "rebuild_dib", "rc": rc})

    sleeve_env = {}
    if args.test_start:
        sleeve_env["PT_TEST_START"] = args.test_start
    if args.test_end:
        sleeve_env["PT_TEST_END"] = args.test_end
    if args.universe_10:
        sleeve_env["PT_UNIVERSE"] = ",".join(CORE_10)

    if not args.aggregate_only and not args.skip_xsec:
        rc = regen_xsec(sleeve_env, args.dry_run)
        phases.append({"phase": "xsec", "rc": rc})

    if not args.aggregate_only and not args.skip_dib:
        rc = regen_dib(sleeve_env, args.dry_run)
        phases.append({"phase": "dib", "rc": rc})

    # Aggregate each blend
    blend_records = []
    worst_rc = 0
    for blend, halt, tag in BLENDS_TO_AGGREGATE:
        rc = run_aggregator(blend, halt, args.dry_run)
        worst_rc = max(worst_rc, rc if rc != 1 else 0)  # missing sleeves -> don't fail whole run
        full_summary = {} if args.dry_run else load_blend_summary(blend)
        win_metrics = {} if args.dry_run else (window_metrics_from_blend_daily(blend, args.window_start, window_end) or {})
        blend_records.append({
            "blend": blend,
            "tag": tag,
            "dd_halt_pct": halt,
            "aggregator_rc": rc,
            "full_summary": full_summary,
            "window_metrics": win_metrics,
        })

    # Per-sleeve window slicing
    sleeve_records = []
    for profile, seed, regen_type in SLEEVES:
        rows = load_sleeve_snapshot(seed)
        metrics = slice_and_metrics(rows, args.window_start, window_end) if rows else None
        sleeve_records.append({
            "profile": profile,
            "seed": seed,
            "regen_type": regen_type,
            "window_metrics": metrics,
        })

    report = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "window_start": args.window_start,
        "window_end": window_end,
        "capital_display": args.capital,
        "phases": phases,
        "sleeves": sleeve_records,
        "blends": blend_records,
    }

    if not args.dry_run:
        DEPLOY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        run_dir = DEPLOY_LOG_DIR / str(today)
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "all_top_strats_report.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[deploy] consolidated report: {out}")

    print_report(report, args.window_start, window_end, args.capital)

    halt_any = any(b.get("full_summary", {}).get("halt_triggered") for b in blend_records)
    sys.exit(2 if halt_any else worst_rc)


if __name__ == "__main__":
    main()
