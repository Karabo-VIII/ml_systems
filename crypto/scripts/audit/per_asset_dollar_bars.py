"""Per-asset dollar bar generator + smart cell experiment.

Universal dollar-bar test was -41.73%. This script does it RIGHT per the
2nd-pass critique: each asset gets its OWN threshold calibrated to its OWN
median daily dollar volume, targeting ~48 bars/day (~30-min equivalent at
median activity).

PIPELINE
  1. Load 15m chimera for asset
  2. median_daily_dvol = median of daily sum(close * volume) across last 365 days
  3. threshold = median_daily_dvol / TARGET_BARS_PER_DAY (default 48)
  4. Walk 15m bars; accumulate cum_dollar_vol; emit new dollar bar when cum >= threshold
  5. Mine smart-grid cells on dollar bars (TRAIN+VAL only)
  6. Generate OOS fires on dollar bars
  7. Honest 4-bound portfolio sim

OUTPUT
  runs/audit/DOLLAR_BARS_2026_05_20/
    per_asset_thresholds.csv
    db_{asset}.parquet              -- per-asset dollar bars
    per_asset_top_cells_db.parquet
    oos_fires_db.parquet
    sim_results_db.json
    REPORT_DB.md

NOTE: this is an ISOLATED experiment. Failure here does not contaminate the
multi-TF confluence main harness. If positive, we can route specific assets
to dollar bars in a future blend.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "pipeline"))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))

from smart_candidate_generator import generate_raw_candidates, empirical_decorrelate
from multi_tf_breakthrough import (
    cross_up_events, cell_score_on_training, _ma,
    simulate_4bound, metrics, _build_panel_idx, _trade_exit,
    TRAIN_VAL_END, OOS_START, OOS_END, MIN_FIRES, TOP_K_CELLS, CORR_THRESHOLD,
)

OUT_DIR = ROOT / "runs" / "audit" / "DOLLAR_BARS_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_BARS_PER_DAY = 48
LOOKBACK_DAYS_FOR_THRESHOLD = 365

# Focus on high-vol assets where dollar bars should help most
VOL_ASSETS = ["PEPE", "FLOKI", "SUI", "ARB", "APT", "OP", "LDO", "JST",
              "DOGE", "SHIB", "AVAX", "FET", "SUPER", "ZEC"]


def build_per_asset_dollar_bars(chimera_loader, asset: str) -> tuple[pd.DataFrame, float]:
    """Returns (dollar_bars_df, threshold_used)."""
    try:
        df = chimera_loader.load(asset, "15m")
    except Exception:
        return pd.DataFrame(), 0.0
    if df is None:
        return pd.DataFrame(), 0.0
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["date"] = df["dt"].dt.date
    df = df.sort_values("dt").reset_index(drop=True)
    if "volume" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(), 0.0

    df["dvol"] = df["close"] * df["volume"]
    # Compute median daily dollar volume on a recent slice (pre-TRAIN_VAL_END to avoid leakage)
    recent = df[(df["date"] < TRAIN_VAL_END)].tail(LOOKBACK_DAYS_FOR_THRESHOLD * 96)  # 96 15m bars/day
    if not len(recent):
        return pd.DataFrame(), 0.0
    daily_dv = recent.groupby("date")["dvol"].sum()
    if not len(daily_dv):
        return pd.DataFrame(), 0.0
    median_daily_dv = float(daily_dv.median())
    if median_daily_dv <= 0:
        return pd.DataFrame(), 0.0
    threshold = median_daily_dv / TARGET_BARS_PER_DAY

    # Walk and accumulate
    cum = 0.0
    o = h = l = c = None
    bar_start_ts = None
    bar_start_dt = None
    db_rows = []
    for _, row in df.iterrows():
        if o is None:
            o = float(row["open"]) if "open" in df.columns else float(row["close"])
            h = float(row["high"])
            l = float(row["low"])
            bar_start_ts = int(row["timestamp"])
            bar_start_dt = row["dt"]
        else:
            h = max(h, float(row["high"]))
            l = min(l, float(row["low"]))
        c = float(row["close"])
        cum += float(row["dvol"])
        if cum >= threshold:
            db_rows.append({
                "timestamp": bar_start_ts,
                "dt": bar_start_dt,
                "date": bar_start_dt.date(),
                "open": o, "high": h, "low": l, "close": c,
                "dollar_volume": cum,
            })
            o = h = l = c = None
            cum = 0.0
            bar_start_ts = None
            bar_start_dt = None
    return pd.DataFrame(db_rows), threshold


def mine_db_cells_for_asset(db: pd.DataFrame, asset: str, candidates: list[tuple[int, int]],
                            train_val_end: date) -> list[dict]:
    if not len(db):
        return []
    closes = db["close"].values.astype(float)
    dates = db["date"].values
    results = []
    for ma_type in ["SMA", "EMA"]:
        for (f, s) in candidates:
            sc = cell_score_on_training(closes, dates, f, s, ma_type, train_val_end)
            sc.update({"asset": asset, "cadence": "db", "ma_type": ma_type, "fast": f, "slow": s})
            results.append(sc)
    qual = [r for r in results if r["n"] >= MIN_FIRES]
    qual.sort(key=lambda x: x["sharpe"], reverse=True)
    return qual[:TOP_K_CELLS]


def generate_db_oos_fires(db_dict: dict[str, pd.DataFrame], top_cells: pd.DataFrame,
                          oos_start: date, oos_end: date) -> pd.DataFrame:
    fires = []
    for asset, db in db_dict.items():
        if not len(db):
            continue
        sub_cells = top_cells[top_cells["asset"] == asset]
        if not len(sub_cells):
            continue
        closes = db["close"].values.astype(float)
        dates = pd.to_datetime(db["dt"]).dt.date.values
        for _, cell in sub_cells.iterrows():
            f, s, mt = int(cell["fast"]), int(cell["slow"]), cell["ma_type"]
            cross = cross_up_events(closes, f, s, mt)
            if not cross.any():
                continue
            mf = _ma(closes, f, mt)
            ml = _ma(closes, s, mt)
            for i in np.where(cross)[0]:
                d = dates[i]
                if d < oos_start or d > oos_end:
                    continue
                strength = (mf[i] - ml[i]) / closes[i] if closes[i] > 0 else 0
                fires.append({
                    "asset": asset, "cadence": "db", "ma_type": mt,
                    "fast": f, "slow": s,
                    "fire_date": d,
                    "signal_strength": float(strength),
                    "confluence_level": 1,
                })
    return pd.DataFrame(fires)


def main():
    from pipeline.chimera_loader import ChimeraLoader
    cl = ChimeraLoader()

    print("=" * 78)
    print("PER-ASSET DOLLAR BARS EXPERIMENT")
    print(f"  assets: {VOL_ASSETS}")
    print(f"  target bars/day: {TARGET_BARS_PER_DAY}")
    print("=" * 78)

    # ---- Build dollar bars per asset
    print("\n[1/4] Building per-asset dollar bars...")
    db_dict = {}
    thresholds = []
    for asset in VOL_ASSETS:
        db, thr = build_per_asset_dollar_bars(cl, asset)
        if not len(db):
            print(f"  {asset}: SKIP (no data or zero dvol)")
            continue
        db.to_parquet(OUT_DIR / f"db_{asset}.parquet", index=False)
        db_dict[asset] = db
        thresholds.append({
            "asset": asset, "threshold_dollar": thr, "n_bars": len(db),
            "first_date": str(db["date"].min()), "last_date": str(db["date"].max()),
            "bars_per_day_actual": len(db) / max((pd.to_datetime(db["dt"]).max() - pd.to_datetime(db["dt"]).min()).days, 1),
        })
        print(f"  {asset}: thr=${thr:,.0f}  {len(db)} bars  {thresholds[-1]['bars_per_day_actual']:.1f} bars/day actual")
    thresh_df = pd.DataFrame(thresholds)
    thresh_df.to_csv(OUT_DIR / "per_asset_thresholds.csv", index=False)

    # ---- Mine smart cells on dollar bars
    print("\n[2/4] Mining smart cells on dollar bars (TRAIN+VAL only)...")
    raw_cands = generate_raw_candidates(100)
    print(f"  raw candidates: {len(raw_cands)}")
    all_cells = []
    for asset, db in db_dict.items():
        # Decorrelate on this asset's TRAIN dollar-bar closes
        train_closes = db.loc[pd.to_datetime(db["date"]) < pd.Timestamp(TRAIN_VAL_END), "close"].values
        if len(train_closes) < 200:
            cands = raw_cands
        else:
            cands = empirical_decorrelate(train_closes, raw_cands, CORR_THRESHOLD, "SMA")
        top = mine_db_cells_for_asset(db, asset, cands, TRAIN_VAL_END)
        all_cells.extend(top)
        print(f"  {asset}: {len(top)} top cells")
    cells_df = pd.DataFrame(all_cells)
    cells_df.to_parquet(OUT_DIR / "per_asset_top_cells_db.parquet", index=False)

    # ---- Generate OOS fires on dollar bars
    print("\n[3/4] Generating OOS fires on dollar bars...")
    fires = generate_db_oos_fires(db_dict, cells_df, OOS_START, OOS_END)
    fires.to_parquet(OUT_DIR / "oos_fires_db.parquet", index=False)
    print(f"  total OOS fires: {len(fires)}")

    # ---- Sim
    print("\n[4/4] Running honest 4-bound sim on dollar-bar fires...")
    # Build 1d panel idx (use 1d chimera for exit MtM, even though entries are
    # at dollar-bar timestamps) — this matches the deployed sim's MtM convention.
    panel_idx = _build_panel_idx(cl, VOL_ASSETS)
    window = (OOS_END - OOS_START).days
    results = {}
    for mode in ["best", "signal", "random", "worst"]:
        daily, trade_log = simulate_4bound(fires, panel_idx, OOS_START, OOS_END,
                                            mode=mode, use_sub_day_entry=False)
        m = metrics(daily, trade_log, window)
        results[mode] = m
        print(f"  {mode:7s}-K: NAV={m['total_pct']:+8.2f}%  Sortino={m['sortino']:+.2f}  "
              f"DD={m['max_dd_pct']:+.1f}%  n={m['n_trades']}  win={m['win_rate_pct']:.1f}%")

    (OUT_DIR / "sim_results_db.json").write_text(json.dumps(results, indent=2, default=str))

    # Report
    lines = ["# Per-Asset Dollar Bars Experiment\n",
             f"\n**Date**: 2026-05-20  \n**Assets**: {VOL_ASSETS}  ",
             f"\n**Target bars/day**: {TARGET_BARS_PER_DAY}  \n",
             "\n## Per-asset thresholds + bar counts\n",
             "| Asset | Threshold ($) | n_bars | bars/day | First | Last |",
             "|---|---:|---:|---:|---|---|"]
    for t in thresholds:
        lines.append(f"| {t['asset']} | {t['threshold_dollar']:,.0f} | {t['n_bars']} | "
                     f"{t['bars_per_day_actual']:.1f} | {t['first_date']} | {t['last_date']} |")
    lines.append("\n## 4-bound sim (OOS)\n")
    lines.append("| Mode | NAV % | Sortino | Max DD | Trades | Win % | 7d>=5.25% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for mode in ["best", "signal", "random", "worst"]:
        m = results.get(mode, {})
        lines.append(f"| {mode} | {m.get('total_pct',0):+.2f} | {m.get('sortino',0):+.2f} | "
                     f"{m.get('max_dd_pct',0):+.2f} | {m.get('n_trades',0)} | "
                     f"{m.get('win_rate_pct',0):.1f} | {m.get('pct_days_7d_above_5_25',0):.1f} |")
    lines.append("\n## Honest interpretation\n")
    lines.append("- Universal dollar-bar test: -41.73% NAV (rejected). This experiment fixes the threshold per-asset.")
    lines.append("- Compare signal-K NAV to baseline +91.40%.")
    lines.append("- If positive: dollar bars are viable on these specific assets.")
    lines.append("- If random-K beats signal-K: ranker is sub-random on dollar bars; gate-only viable.")
    (OUT_DIR / "REPORT_DB.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'REPORT_DB.md'}")


if __name__ == "__main__":
    main()
