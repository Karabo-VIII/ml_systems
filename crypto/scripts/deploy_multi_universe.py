"""Multi-universe backtest driver: U10, U50, U100.

Runs the universe-sensitive xsec ranker + frontier overlays under three
progressively larger asset universes, then compares metrics to show how
each strategy scales (or collapses) with universe size.

Universes:
    U10    -> core 10 (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, LTC)
    U50    -> src/strategy/universe.py UNIVERSE_50 (the official 50-list)
    U100   -> all available chimera files (currently ~72 assets; U100
              aspirational until more assets are pipelined)

For each universe, runs:
    1. xsec_K5_5_FULL_dneut + xgb_K3_long_WEALTH40 + cat_K1_stop_no_macro
       (via scratch/xsec_variants_daily_equity.py with PT_UNIVERSE set)
    2. DIB duo (universe-invariant; BTC+ETH only, run once)
    3. Frontier stable_flow + etf_flow (uses chimera panel; run once)
    4. Meta seeds (uses UNIVERSE_50 hardcoded; run once)

Snapshots are saved to `pt_xsec_*_U<N>/daily_snapshot.csv` so each
universe has its own record. A comparison report is emitted at the end.

Usage:
    python scripts/deploy_multi_universe.py                # all 3 universes
    python scripts/deploy_multi_universe.py --only 10 50   # subset
    python scripts/deploy_multi_universe.py --skip-universal  # xsec-only (fast)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
XSEC_SCRIPT = ROOT / "scratch" / "xsec_variants_daily_equity.py"
SEEDS_DIR = ROOT / "logs" / "paper_trader_v2" / "seeds"
DEPLOY_LOG = ROOT / "logs" / "deployment"

U10 = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"]


def load_u50() -> list[str]:
    sys.path.insert(0, str(ROOT / "src" / "strategy"))
    from universe import UNIVERSE_50
    return [a.replace("USDT", "") for a in UNIVERSE_50]


def load_u100_available() -> list[str]:
    """Union of all chimera files."""
    import glob
    fps = sorted(glob.glob(str(ROOT / "data" / "processed" / "*_v50_chimera.parquet")))
    return sorted({Path(f).stem.replace("usdt_v50_chimera", "").upper() for f in fps})


def run_xsec_for_universe(universe_label: str, assets: list[str]) -> dict:
    """Run xsec script with PT_UNIVERSE set; rename outputs to include universe label."""
    env = os.environ.copy()
    env["PT_UNIVERSE"] = ",".join(assets)

    print(f"\n>>> xsec {universe_label} ({len(assets)} assets)")
    t0 = time.time()
    out = subprocess.run([PY, str(XSEC_SCRIPT)], env=env,
                         capture_output=True, text=True, timeout=1200,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    dt = time.time() - t0
    rc = out.returncode

    # Parse the 3 variant results from xsec output
    summary = {"universe": universe_label, "n_assets": len(assets),
               "rc": rc, "dt": round(dt, 1)}
    for line in (out.stdout or "").splitlines():
        for variant in ("xsec_K5_5_FULL_dneut", "xgb_K3_long_WEALTH40", "cat_K1_stop_no_macro"):
            if line.strip().startswith(variant):
                summary[variant] = line.strip()

    # Copy the 3 produced snapshots into universe-specific seed dirs
    variants = ["xsec_K5_5_FULL_dneut", "xgb_K3_long_WEALTH40", "cat_K1_stop_no_macro"]
    copies = {}
    for v in variants:
        src = SEEDS_DIR / f"pt_{v}" / "daily_snapshot.csv"
        dst_dir = SEEDS_DIR / f"pt_{v}_{universe_label}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "daily_snapshot.csv"
        if src.exists():
            shutil.copyfile(src, dst)
            copies[v] = str(dst)
    summary["snapshot_copies"] = copies
    print(f"[done] {universe_label}  rc={rc}  dt={dt:.1f}s")
    return summary


def slice_and_metrics(fp: Path, window_start: str, window_end: str) -> dict:
    import numpy as np
    if not fp.exists():
        return {"status": "MISSING"}
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        return {"status": "UNREADABLE", "err": str(e)}
    df["date"] = pd.to_datetime(df["date"])
    df = df.groupby(df["date"].dt.date).last().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    if len(df) < 10:
        return {"status": "SHORT"}
    eq = df["total_equity"].values.astype(float)
    eq_r = eq / eq[0] * 10000.0
    r = np.diff(eq_r) / eq_r[:-1]
    total_ret = (eq_r[-1] / 10000.0 - 1) * 100
    days = (df["date"].iloc[-1] - df["date"].iloc[0]).days or 1
    cagr = ((eq_r[-1] / 10000.0) ** (365.0 / days) - 1) * 100
    sharpe = r.mean() / r.std() * (365 ** 0.5) if r.std() > 0 else 0.0
    cum_max = np.maximum.accumulate(eq_r)
    dd = ((eq_r - cum_max) / cum_max).min() * 100
    result = {
        "status": "OK",
        "n_days": len(df),
        "total_ret_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(dd, 2),
    }
    ws = pd.to_datetime(window_start)
    we = pd.to_datetime(window_end)
    sub = df[(df["date"] >= ws) & (df["date"] <= we)]
    if len(sub) >= 2:
        se = sub["total_equity"].values.astype(float)
        sr = (se[1:] - se[:-1]) / se[:-1]
        result["win_ret_pct"] = round((se[-1] / se[0] - 1) * 100, 2)
        result["win_sharpe"] = round(sr.mean() / sr.std() * (365 ** 0.5) if sr.std() > 0 else 0.0, 3)
        cm = np.maximum.accumulate(se)
        result["win_dd_pct"] = round(((se - cm) / cm).min() * 100, 2)
        result["win_days"] = len(sub)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=["10", "50", "100"],
                    help="Subset of universes to run (default: all 3)")
    ap.add_argument("--window-start", default="2026-04-01")
    ap.add_argument("--window-end", default="2026-04-22")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    universes = {
        "U10":  U10,
        "U50":  load_u50(),
        "U100": load_u100_available(),
    }

    targets = [(lbl, assets) for lbl, assets in universes.items()
               if not args.only or lbl[1:] in args.only]

    print("=" * 90)
    print(f"MULTI-UNIVERSE BACKTEST DRIVER")
    print(f"UTC: {datetime.now(timezone.utc).isoformat()}")
    for lbl, assets in targets:
        print(f"  {lbl}  n_assets={len(assets)}  first 5: {assets[:5]}")
    print("=" * 90)

    if args.dry_run:
        print("(dry-run)")
        return

    t_start = time.time()
    run_records = []
    for lbl, assets in targets:
        rec = run_xsec_for_universe(lbl, assets)
        run_records.append(rec)

    # Compare metrics
    print("\n" + "=" * 120)
    print(f"MULTI-UNIVERSE COMPARISON  (window: {args.window_start} -> {args.window_end})")
    print("=" * 120)
    header = f"{'variant':<28} {'universe':<6} {'days':>5} {'CAGR%':>8} {'Sh':>6} {'DD%':>7} " \
             f"{'winD':>5} {'winR%':>7} {'winSh':>6}"
    print(header)
    print("-" * 120)

    comparison = []
    for v in ["xsec_K5_5_FULL_dneut", "xgb_K3_long_WEALTH40", "cat_K1_stop_no_macro"]:
        for lbl, _ in targets:
            fp = SEEDS_DIR / f"pt_{v}_{lbl}" / "daily_snapshot.csv"
            m = slice_and_metrics(fp, args.window_start, args.window_end)
            if m.get("status") == "OK":
                print(f"{v:<28} {lbl:<6} {m['n_days']:>5} {m['cagr_pct']:>+7.2f} "
                      f"{m['sharpe']:>+5.2f} {m['max_dd_pct']:>+6.2f} "
                      f"{m.get('win_days',0):>5} "
                      f"{m.get('win_ret_pct',0):>+6.2f} {m.get('win_sharpe',0):>+5.2f}")
            comparison.append({"variant": v, "universe": lbl, **m})
        print("-" * 120)

    # Persist record
    today = datetime.now(timezone.utc).date()
    out_dir = DEPLOY_LOG / str(today)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "multi_universe_run.json"
    with open(out_file, "w") as f:
        json.dump({
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "window_start": args.window_start,
            "window_end": args.window_end,
            "universes": {lbl: assets for lbl, assets in targets},
            "xsec_runs": run_records,
            "comparison": comparison,
            "total_elapsed_s": round(time.time() - t_start, 1),
        }, f, indent=2, default=str)
    print(f"\n[saved] {out_file}")


if __name__ == "__main__":
    main()
