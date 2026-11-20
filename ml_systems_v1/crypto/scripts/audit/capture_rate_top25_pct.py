"""capture_rate_top25_pct.py -- per-day top-25% daily mover capture analysis.

Uses the REFRESHED smart-grid top cells (per_asset_smart_grid_profile.parquet)
from master_csv_smart_rebuild.

For each OOS day:
  1. Compute each asset's 1d return today (close[t]/close[t-1] - 1)
  2. Rank assets by today's return; pick top-25%
  3. For each top-25% mover, check three capture definitions on its smart-grid cells:
     - fired_today: cross-up event exactly today on any of its top cells (cadence-aware)
     - fired_recently: cross-up event in last 7 days (we'd be holding through today)
     - active_today: MA-active state today (cell is in long-active state)

Then aggregates:
  - per-day capture rate (overall mean + distribution)
  - per-asset capture rate (which assets we systematically catch / miss)
  - per-regime capture rate (bull/chop/bear/crash)
  - per-bucket capture rate

OUTPUT:
  runs/audit/MASTER_CSV_SMART_REBUILD_2026_05_20/
    capture_top25_per_day.csv
    capture_top25_per_asset.csv
    capture_top25_per_regime.csv
    CAPTURE_TOP25_REPORT.md
"""
from __future__ import annotations
import sys
import math
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))

OUT_DIR = ROOT / "runs" / "audit" / "MASTER_CSV_SMART_REBUILD_2026_05_20"
PROFILE_PATH = OUT_DIR / "per_asset_smart_grid_profile.parquet"
OWN_REGIME = ROOT / "data" / "processed" / "asset_own_regime_panel.parquet"

# Default = TRAIN+VAL window (the period the cells were MINED on; expect higher
# capture than OOS since cells were optimized here)
TRAIN_START = date(2021, 1, 1)
TRAIN_END = date(2024, 5, 15)
OOS_START_DT = date(2024, 5, 16)
OOS_END_DT = date(2025, 3, 15)
# Set via CLI; defaults to TRAIN window
import os
_MODE = os.environ.get("CAPTURE_MODE", "train")  # 'train' or 'oos'
if _MODE == "oos":
    OOS_START = OOS_START_DT
    OOS_END = OOS_END_DT
    OUT_TAG = "OOS"
else:
    OOS_START = TRAIN_START
    OOS_END = TRAIN_END
    OUT_TAG = "TRAIN"
TOP_PCT = 0.25  # top 25%
RECENT_WINDOW_DAYS = 7  # for fired_recently


def _ma(s: np.ndarray, period: int, ma_type: str) -> np.ndarray:
    sr = pd.Series(s)
    if ma_type == "SMA":
        return sr.rolling(period).mean().values
    return sr.ewm(span=period, adjust=False).mean().values


def cell_cross_and_active(closes: np.ndarray, fast: int, slow: int, ma_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (cross_up_bool_array, active_bool_array) per-bar."""
    if len(closes) < slow + 2:
        return np.zeros(len(closes), dtype=bool), np.zeros(len(closes), dtype=bool)
    mf = _ma(closes, fast, ma_type)
    ml = _ma(closes, slow, ma_type)
    cross = np.zeros(len(closes), dtype=bool)
    cross[1:] = (mf[1:] > ml[1:]) & (mf[:-1] <= ml[:-1])
    active = mf > ml
    return cross, active


def main():
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()

    print("=" * 78)
    print("CAPTURE-RATE TOP-25% ANALYSIS (smart-grid refreshed set)")
    print(f"  OOS: {OOS_START} -> {OOS_END}")
    print(f"  top pct: {TOP_PCT*100:.0f}%")
    print(f"  recent window: {RECENT_WINDOW_DAYS} days")
    print("=" * 78)

    if not PROFILE_PATH.exists():
        print(f"ERROR: {PROFILE_PATH} not found")
        return

    profile = pd.read_parquet(PROFILE_PATH)
    print(f"\nSmart-grid profile loaded: {len(profile)} cells across {profile['asset'].nunique()} assets, "
          f"{profile['cadence'].nunique()} cadences")

    own_regime = pl.read_parquet(OWN_REGIME).to_pandas()
    own_regime["date"] = pd.to_datetime(own_regime["date"]).dt.date

    # For each asset, load 1d chimera; compute returns + cross + active states per cell
    print("\n[1/3] Loading 1d chimera + computing per-asset cell states...")
    asset_data: dict = {}
    for asset in profile["asset"].unique():
        try:
            df = cl.load(asset, "1d")
            if df is None:
                continue
            if hasattr(df, "to_pandas"):
                df = df.to_pandas()
            df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
            df = df.sort_values("timestamp").reset_index(drop=True)
            closes = df["close"].values.astype(float)
            dates = df["date"].values
            # compute 1d return (today vs yesterday)
            ret_1d = np.zeros(len(closes), dtype=float)
            ret_1d[1:] = closes[1:] / closes[:-1] - 1
            ret_1d[np.isinf(ret_1d)] = 0
            ret_1d[np.isnan(ret_1d)] = 0
            asset_cells = profile[profile["asset"] == asset]
            cell_states = []
            for _, c in asset_cells.iterrows():
                # Only use 1d cells for this analysis (consistent day-level check)
                if c["cadence"] != "1d":
                    continue
                cross, active = cell_cross_and_active(closes, int(c["fast"]), int(c["slow"]), c["ma_type"])
                cell_states.append({
                    "ma_type": c["ma_type"], "fast": int(c["fast"]), "slow": int(c["slow"]),
                    "cross": cross, "active": active,
                    "rank": c.get("sharpe_train", 0),
                })
            asset_data[asset] = {
                "dates": dates, "closes": closes, "ret_1d": ret_1d,
                "cells": cell_states,
            }
        except Exception as e:
            print(f"  fail {asset}: {e}")
            continue
    print(f"  loaded {len(asset_data)} assets")

    # Build per-day return table + capture indicators
    print("\n[2/3] Computing top-25% per day + capture indicators...")
    cur = OOS_START
    all_dates = []
    while cur <= OOS_END:
        all_dates.append(cur)
        cur += timedelta(days=1)

    own_lookup = {(r["asset"], r["date"]): r["asset_own_regime"]
                   for _, r in own_regime.iterrows()}

    # universe bucket map
    import yaml
    asset_meta = {}
    for p in (ROOT/"config"/"universes"/"u50.yaml", ROOT/"config"/"universes"/"u100.yaml"):
        with open(p) as f:
            doc = yaml.safe_load(f)
        for a in doc.get("assets", []) + doc.get("extra_assets", []):
            if a.get("status", "ready") != "ready":
                continue
            sym = a["symbol"].replace("USDT", "")
            asset_meta[sym] = {"bucket": a.get("dna", "VOLATILE")}

    per_day_records = []
    per_asset_capture = {a: {"n_top25": 0, "n_capt_today": 0, "n_capt_recent": 0,
                              "n_capt_active": 0} for a in asset_data}

    for d in all_dates:
        # Gather per-asset return + capture indicators today
        rows_today = []
        for asset, data in asset_data.items():
            idxs = np.where(data["dates"] == d)[0]
            if not len(idxs):
                continue
            idx = int(idxs[0])
            r = float(data["ret_1d"][idx])
            # Cell-level indicators
            any_cross_today = False
            any_cross_recent = False
            any_active_today = False
            recent_start = max(0, idx - RECENT_WINDOW_DAYS + 1)
            for cell in data["cells"]:
                if idx < len(cell["cross"]):
                    if cell["cross"][idx]:
                        any_cross_today = True
                    if cell["active"][idx]:
                        any_active_today = True
                    # Recent window
                    rs = recent_start
                    re_ = idx + 1
                    if rs < len(cell["cross"]) and re_ <= len(cell["cross"]):
                        if cell["cross"][rs:re_].any():
                            any_cross_recent = True
            rows_today.append({
                "date": d, "asset": asset, "ret_1d": r,
                "cross_today": any_cross_today,
                "cross_recent": any_cross_recent,
                "active_today": any_active_today,
            })
        if not rows_today:
            continue
        df_t = pd.DataFrame(rows_today)
        n_top25 = max(1, math.ceil(len(df_t) * TOP_PCT))
        top = df_t.sort_values("ret_1d", ascending=False).head(n_top25)

        n_capt_today = int(top["cross_today"].sum())
        n_capt_recent = int(top["cross_recent"].sum())
        n_capt_active = int(top["active_today"].sum())
        per_day_records.append({
            "date": d, "n_assets_active": len(df_t), "n_top25": n_top25,
            "n_capt_today": n_capt_today, "n_capt_recent": n_capt_recent, "n_capt_active": n_capt_active,
            "capt_today_pct": 100 * n_capt_today / n_top25,
            "capt_recent_pct": 100 * n_capt_recent / n_top25,
            "capt_active_pct": 100 * n_capt_active / n_top25,
            "top25_assets": ",".join(top["asset"].tolist()),
            "top25_returns_mean": float(top["ret_1d"].mean()),
            "top25_returns_min": float(top["ret_1d"].min()),
            "top25_returns_max": float(top["ret_1d"].max()),
        })
        # per-asset accumulators
        for _, r in top.iterrows():
            a = r["asset"]
            if a in per_asset_capture:
                per_asset_capture[a]["n_top25"] += 1
                if r["cross_today"]: per_asset_capture[a]["n_capt_today"] += 1
                if r["cross_recent"]: per_asset_capture[a]["n_capt_recent"] += 1
                if r["active_today"]: per_asset_capture[a]["n_capt_active"] += 1

    per_day_df = pd.DataFrame(per_day_records)
    per_day_df.to_csv(OUT_DIR / f"capture_top25_per_day_{OUT_TAG}.csv", index=False)
    print(f"  per-day capture: {len(per_day_df)} days")

    # Per-asset capture rate
    per_asset_rows = []
    for a, d in per_asset_capture.items():
        if d["n_top25"] == 0:
            continue
        per_asset_rows.append({
            "asset": a,
            "bucket": asset_meta.get(a, {}).get("bucket", "?"),
            "n_top25_appearances": d["n_top25"],
            "n_capt_today": d["n_capt_today"],
            "n_capt_recent": d["n_capt_recent"],
            "n_capt_active": d["n_capt_active"],
            "capt_today_pct": 100 * d["n_capt_today"] / d["n_top25"],
            "capt_recent_pct": 100 * d["n_capt_recent"] / d["n_top25"],
            "capt_active_pct": 100 * d["n_capt_active"] / d["n_top25"],
        })
    per_asset_df = pd.DataFrame(per_asset_rows).sort_values("n_top25_appearances", ascending=False)
    per_asset_df.to_csv(OUT_DIR / f"capture_top25_per_asset_{OUT_TAG}.csv", index=False)
    print(f"  per-asset capture: {len(per_asset_df)} assets")

    # Per-regime capture (using BTC regime as proxy for market state on the day)
    # Pull from own_regime panel for BTC
    btc_regime_lookup = {}
    btc_regime = own_regime[own_regime["asset"] == "BTC"]
    for _, r in btc_regime.iterrows():
        btc_regime_lookup[r["date"]] = r["asset_own_regime"]
    per_day_df["btc_regime"] = per_day_df["date"].map(btc_regime_lookup).fillna("unknown")
    per_regime_rows = []
    for regime, g in per_day_df.groupby("btc_regime"):
        per_regime_rows.append({
            "btc_regime": regime,
            "n_days": len(g),
            "mean_capt_today_pct": float(g["capt_today_pct"].mean()),
            "median_capt_today_pct": float(g["capt_today_pct"].median()),
            "mean_capt_recent_pct": float(g["capt_recent_pct"].mean()),
            "median_capt_recent_pct": float(g["capt_recent_pct"].median()),
            "mean_capt_active_pct": float(g["capt_active_pct"].mean()),
            "median_capt_active_pct": float(g["capt_active_pct"].median()),
            "mean_top25_ret": float(g["top25_returns_mean"].mean()),
        })
    per_regime_df = pd.DataFrame(per_regime_rows)
    per_regime_df.to_csv(OUT_DIR / f"capture_top25_per_regime_{OUT_TAG}.csv", index=False)

    # Per-bucket capture
    per_bucket_rows = []
    for bucket, g in per_asset_df.groupby("bucket"):
        n_total_top25 = g["n_top25_appearances"].sum()
        n_capt_today = g["n_capt_today"].sum()
        n_capt_recent = g["n_capt_recent"].sum()
        n_capt_active = g["n_capt_active"].sum()
        per_bucket_rows.append({
            "bucket": bucket,
            "n_assets": len(g),
            "n_top25_appearances": int(n_total_top25),
            "capt_today_pct": float(100 * n_capt_today / max(n_total_top25, 1)),
            "capt_recent_pct": float(100 * n_capt_recent / max(n_total_top25, 1)),
            "capt_active_pct": float(100 * n_capt_active / max(n_total_top25, 1)),
        })
    per_bucket_df = pd.DataFrame(per_bucket_rows)

    # Summary
    print("\n[3/3] OVERALL CAPTURE SUMMARY")
    print(f"  total days: {len(per_day_df)}")
    print(f"  mean top-25% capture (cross-up TODAY):   {per_day_df['capt_today_pct'].mean():.2f}%")
    print(f"  mean top-25% capture (cross-up RECENT 7d): {per_day_df['capt_recent_pct'].mean():.2f}%")
    print(f"  mean top-25% capture (MA-active TODAY):  {per_day_df['capt_active_pct'].mean():.2f}%")
    print(f"\n  BTC regime breakdown:")
    for _, r in per_regime_df.iterrows():
        print(f"    {r['btc_regime']:<8} n={r['n_days']:3d}d  "
              f"capt_today={r['mean_capt_today_pct']:.1f}%  "
              f"capt_recent={r['mean_capt_recent_pct']:.1f}%  "
              f"capt_active={r['mean_capt_active_pct']:.1f}%  "
              f"top25_ret={r['mean_top25_ret']*100:+.2f}%/day")

    print(f"\n  bucket breakdown:")
    for _, r in per_bucket_df.iterrows():
        print(f"    {r['bucket']:<10} n={r['n_assets']:2d}a  appearances={r['n_top25_appearances']:4d}  "
              f"capt_today={r['capt_today_pct']:.1f}%  "
              f"capt_recent={r['capt_recent_pct']:.1f}%  "
              f"capt_active={r['capt_active_pct']:.1f}%")

    print(f"\n  TOP-10 assets by top-25% APPEARANCES (most-frequent movers):")
    for _, r in per_asset_df.head(10).iterrows():
        print(f"    {r['asset']:<6} {r['bucket']:<10} appearances={r['n_top25_appearances']:3d}  "
              f"capt_today={r['capt_today_pct']:5.1f}%  "
              f"capt_recent={r['capt_recent_pct']:5.1f}%  "
              f"capt_active={r['capt_active_pct']:5.1f}%")

    print(f"\n  BOTTOM-10 assets by top-25% APPEARANCES (assets we rarely catch movers on):")
    for _, r in per_asset_df.tail(10).iterrows():
        print(f"    {r['asset']:<6} {r['bucket']:<10} appearances={r['n_top25_appearances']:3d}  "
              f"capt_today={r['capt_today_pct']:5.1f}%  "
              f"capt_recent={r['capt_recent_pct']:5.1f}%  "
              f"capt_active={r['capt_active_pct']:5.1f}%")

    # Write markdown report
    lines = ["# Top-25% Daily Mover Capture Analysis (Smart-Grid Refreshed Set)\n",
             f"\n**Date**: 2026-05-20  \n**OOS**: {OOS_START} -> {OOS_END}  ",
             f"\n**Substrate**: smart-grid top cells (per_asset_smart_grid_profile.parquet)  ",
             f"\n**Definitions**:",
             "\n- `cross_today` — at least one of asset's top-3 1d smart-grid cells had a cross-up event ON this day",
             f"\n- `cross_recent` — at least one cross-up event in last {RECENT_WINDOW_DAYS} days (would be in a position)",
             "\n- `active_today` — at least one of asset's top-3 1d smart-grid cells is in MA-active state today\n",
             "\n## Headline\n",
             f"- Mean per-day capture of top-25% movers: cross_today **{per_day_df['capt_today_pct'].mean():.2f}%**  /  cross_recent **{per_day_df['capt_recent_pct'].mean():.2f}%**  /  active_today **{per_day_df['capt_active_pct'].mean():.2f}%**",
             f"- Total days analyzed: {len(per_day_df)}",
             f"- Mean top-25% return per day: {per_day_df['top25_returns_mean'].mean()*100:+.2f}%",
             "\n## By BTC regime\n",
             "| Regime | n_days | Cross today | Cross recent | Active today | Top25 ret |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, r in per_regime_df.iterrows():
        lines.append(f"| {r['btc_regime']} | {r['n_days']} | {r['mean_capt_today_pct']:.2f}% | "
                     f"{r['mean_capt_recent_pct']:.2f}% | {r['mean_capt_active_pct']:.2f}% | "
                     f"{r['mean_top25_ret']*100:+.2f}% |")
    lines.append("\n## By DNA bucket\n")
    lines.append("| Bucket | n_assets | n_top25 | Cross today | Cross recent | Active today |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in per_bucket_df.iterrows():
        lines.append(f"| {r['bucket']} | {r['n_assets']} | {r['n_top25_appearances']} | "
                     f"{r['capt_today_pct']:.2f}% | {r['capt_recent_pct']:.2f}% | "
                     f"{r['capt_active_pct']:.2f}% |")
    lines.append("\n## Per asset (full table) — sorted by top-25% appearances\n")
    lines.append("| Asset | Bucket | n_top25 | Cross today | Cross recent | Active today |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in per_asset_df.iterrows():
        lines.append(f"| {r['asset']} | {r['bucket']} | {r['n_top25_appearances']} | "
                     f"{r['capt_today_pct']:.2f}% | {r['capt_recent_pct']:.2f}% | "
                     f"{r['capt_active_pct']:.2f}% |")
    lines.append("\n## Interpretation\n")
    lines.append("- `Active today` is the loosest capture — being IN a long position today (MA in long state).")
    lines.append(f"- `Cross recent` ({RECENT_WINDOW_DAYS}d) is most production-realistic — would we have ENTERED a position recently enough to be holding?")
    lines.append("- `Cross today` is strictest — did the signal fire EXACTLY today?")
    lines.append("- Capture gap = (100% - Active today%) is the structural ceiling for MA/EMA: we can't catch a mover if we're not even in the long-active state.")
    (OUT_DIR / f"CAPTURE_TOP25_REPORT_{OUT_TAG}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / f'CAPTURE_TOP25_REPORT_{OUT_TAG}.md'}")


if __name__ == "__main__":
    main()
