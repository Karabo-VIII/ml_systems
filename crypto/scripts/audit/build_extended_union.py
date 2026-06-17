"""build_extended_union.py — P4-8: Extended union including all 8Q WF blends.

Adds to the prior 9-lane union:
  - 5 HAWKES_PERASSET specialists (ALGO_BULL, ARKM_CHOP, HBAR_CHOP, JST_BEAR, WLD_BULL)
  - ORACLE_MA_ML_K5_LO
  - TA_SML_SOLO (u50; treated separately due to universe mismatch)
  - STEALTH_PUMP_SOLO_LO

Re-runs the lookahead-free union composition analysis on the expanded roster.

Outputs:
  runs/audit/union_extended_trade_ledger.parquet
  runs/audit/union_extended_composed.parquet
  runs/audit/P4_8_EXTENDED_UNION_VERDICT_2026_05_19.md (final)
"""
from __future__ import annotations
import os
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
LOGS_DIR = ROOT / "logs" / "strat_audit"


def parse_v3_json_to_trades(json_path: Path, lane: str) -> pl.DataFrame:
    """Parse a v3 paper_trade_replay JSON's trade_ledger into normalized schema."""
    try:
        with open(json_path, "r", encoding="utf-8", errors="replace") as f:
            j = json.load(f)
    except Exception:
        return None
    rows = []
    for t in j.get("trade_ledger", []):
        rows.append({
            "lane": lane,
            "asset": t.get("asset", ""),
            "entry_date": t.get("entry_date", ""),
            "exit_date": t.get("exit_date", ""),
            "size_pct": float(t.get("size_pct", 0.0)),
            "gross_ret_pct": float(t.get("gross_ret_pct", 0.0)),
            "cost_pct": float(t.get("cost_pct", 0.0)),
            "net_pnl_pct": float(t.get("net_pnl_pct", 0.0)),
            "exit_reason": t.get("exit_reason", ""),
        })
    if not rows:
        return None
    df = pl.DataFrame(rows).with_columns([
        pl.col("entry_date").str.to_date(),
        pl.col("exit_date").str.to_date(),
    ])
    return df


def gather_blend_quarters(blend_prefix: str, q_windows: list[str]) -> pl.DataFrame:
    """For each (blend_prefix, quarter_window), find latest JSON and parse trades."""
    all_dfs = []
    for w in q_windows:
        pattern = f"paper_trade_replay_v3_{blend_prefix}_u100_{w}.json"
        candidates = sorted(LOGS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            # Try alternative for u50 (TA_SML)
            alt_pattern = f"paper_trade_replay_v3_{blend_prefix}_u50_{w}.json"
            candidates = sorted(LOGS_DIR.glob(alt_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"  {blend_prefix} {w}: MISSING")
            continue
        df = parse_v3_json_to_trades(candidates[0], blend_prefix)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            print(f"  {blend_prefix} {w}: {len(df)} trades")
    if not all_dfs:
        return None
    return pl.concat(all_dfs)


def main():
    # 8Q window definitions
    Q_WINDOWS = [
        "20240101_20240331",
        "20240401_20240630",
        "20240701_20240930",
        "20241001_20241231",
        "20250101_20250331",
        "20250401_20250630",
        "20250701_20250930",
        "20251001_20251231",
    ]

    print("[ext-union] loading prior 9-lane ledger...")
    prior = pl.read_parquet(str(OUT_DIR / "union_trade_ledger.parquet"))
    print(f"  prior: {len(prior)} trades, {prior['lane'].n_unique()} lanes")

    print("\n[ext-union] gathering new lanes from logs/strat_audit JSONs...")
    new_lanes = []

    # 5 Hawkes per-asset specialists
    for spec in ["HAWKES_PERASSET_ALGO_BULL_LO",
                 "HAWKES_PERASSET_ARKM_CHOP_LO",
                 "HAWKES_PERASSET_HBAR_CHOP_LO",
                 "HAWKES_PERASSET_JST_BEAR_LO",
                 "HAWKES_PERASSET_WLD_BULL_LO"]:
        df = gather_blend_quarters(spec, Q_WINDOWS)
        if df is not None:
            new_lanes.append(df)

    # ORACLE MA ML
    df = gather_blend_quarters("ORACLE_MA_ML_K5_LO", Q_WINDOWS)
    if df is not None:
        new_lanes.append(df)

    # TA_SML SOLO (u50)
    df = gather_blend_quarters("TA_SML_SOLO", Q_WINDOWS)
    if df is not None:
        new_lanes.append(df)

    # STEALTH_PUMP
    df = gather_blend_quarters("STEALTH_PUMP_SOLO_LO", Q_WINDOWS)
    if df is not None:
        new_lanes.append(df)

    if not new_lanes:
        print("\nNo new lanes added — exiting.")
        return

    new_trades = pl.concat(new_lanes)
    # Normalize columns to match prior schema
    new_trades = new_trades.with_columns([
        pl.lit("").alias("quarter"),
        pl.lit("long").alias("side"),
        pl.lit("").alias("sleeve"),
        pl.lit("").alias("strategy_id"),
    ]).select(prior.columns)

    extended = pl.concat([prior, new_trades])
    # Filter 8Q
    extended = extended.filter(
        (pl.col("entry_date") >= date(2024, 1, 1)) & (pl.col("entry_date") <= date(2025, 12, 31))
    )
    extended.write_parquet(str(OUT_DIR / "union_extended_trade_ledger.parquet"))
    print(f"\n[ext-union] extended union: {len(extended)} trades / {extended['lane'].n_unique()} lanes")

    # Per-lane summary
    by_lane = extended.group_by("lane").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("net_pnl_pct").sum().alias("sum_net"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("sum_net", descending=True)
    print(f"\n  {'lane':<36} {'n':>5} {'mean_net':>10} {'sum_net':>10} {'wr':>6}")
    for r in by_lane.iter_rows(named=True):
        print(f"  {r['lane']:<36} {r['n']:>5d} {r['mean_net']:>+9.4f}% {r['sum_net']:>+9.2f}% {r['wr']*100:>5.1f}%")

    # Coverage atlas
    print("\n[ext-union] computing extended coverage atlas...")
    atlas = extended.group_by(["asset", "entry_date"]).agg([
        pl.col("lane").n_unique().alias("n_lanes_firing"),
        pl.col("lane").unique().alias("lanes"),
        pl.col("net_pnl_pct").mean().alias("mean_lane_pnl"),
        pl.col("net_pnl_pct").sum().alias("sum_lane_pnl"),
        pl.col("size_pct").first().alias("size_pct"),
    ]).sort("entry_date")
    print(f"  Unique (asset, date) cells: {len(atlas)}")

    daily = atlas.group_by("entry_date").agg([
        pl.col("asset").n_unique().alias("n_assets_firing"),
    ]).sort("entry_date")
    print(f"  Active days: {len(daily)}/731 ({len(daily)/731*100:.1f}%)")
    print(f"  Mean assets/active-day: {daily['n_assets_firing'].mean():.1f}")
    print(f"  Median: {daily['n_assets_firing'].median():.0f}")
    print(f"  Days with ≥5 assets: {(daily['n_assets_firing'] >= 5).sum()}")
    print(f"  Days with ≥10 assets: {(daily['n_assets_firing'] >= 10).sum()}")

    # Conjunction structure
    print("\n  Extended conjunction structure:")
    by_conj = atlas.group_by("n_lanes_firing").agg([
        pl.len().alias("n_cells"),
        pl.col("mean_lane_pnl").mean().alias("mean_pnl"),
        (pl.col("mean_lane_pnl") > 0).cast(pl.Float64).mean().alias("frac_pos"),
    ]).sort("n_lanes_firing")
    for r in by_conj.iter_rows(named=True):
        print(f"  {r['n_lanes_firing']} lanes: {r['n_cells']:>5d} cells  mean_pnl={r['mean_pnl']:+.4f}%  frac_pos={r['frac_pos']*100:.1f}%")

    # Composed simulation (lookahead-free, same as P4-end-to-end)
    print("\n[ext-union] composed engine simulation:")
    print(f"\n  {'K_cap':>6} {'min_n_lanes':>12} {'COMP':>9} {'mean_d':>10} {'neg_wk':>8} {'wealth':>10}")
    cell_best = atlas.with_columns(pl.col("n_lanes_firing").cast(pl.Float64).alias("priority"))

    def simulate(intents, K, min_conv):
        per_day = intents.filter(pl.col("priority") >= min_conv).sort(
            ["entry_date", "priority"], descending=[False, True]
        ).group_by("entry_date").agg([
            pl.col("mean_lane_pnl").head(K).sum().alias("day_pnl_pct"),
            pl.col("asset").head(K).len().alias("n_entries"),
        ])
        all_dates = pl.date_range(date(2024,1,1), date(2025,12,31), interval="1d", eager=True)
        cal = pl.DataFrame({"entry_date": all_dates}).join(per_day, on="entry_date", how="left").with_columns([
            pl.col("day_pnl_pct").fill_null(0.0),
            pl.col("n_entries").fill_null(0),
        ]).sort("entry_date")
        pnl_arr = cal["day_pnl_pct"].to_numpy()
        comp = float((np.exp(np.log1p(pnl_arr / 100).sum()) - 1) * 100)
        # Weekly via pandas
        cal_wk = cal.with_columns([
            pl.col("entry_date").dt.year().alias("y"),
            pl.col("entry_date").dt.week().alias("w"),
        ]).with_columns(
            (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
        )
        cal_wk_pd = cal_wk.select(["iso_week", "day_pnl_pct"]).to_pandas()
        wk_groups = cal_wk_pd.groupby("iso_week")["day_pnl_pct"].apply(
            lambda x: float(np.exp(np.log1p(x.values / 100).sum()) - 1)
        )
        neg_wk = int((wk_groups < 0).sum())
        return comp, float(np.mean(pnl_arr)), neg_wk

    results = []
    for K in [5, 10, 20]:
        for min_conv in [1, 2, 3, 4]:
            comp, mean_d, neg_wk = simulate(cell_best, K, min_conv)
            wealth = 10000 * (1 + comp / 100)
            results.append({"K": K, "min_conv": min_conv, "COMP": comp, "mean_d": mean_d, "neg_wk": neg_wk, "wealth": wealth})
            print(f"  {K:>6d} {min_conv:>9.0f}    {comp:>+8.2f}% {mean_d:>+9.4f}% {neg_wk:>5d}/104 ${wealth:>9,.0f}")

    # Best by wealth
    best = max(results, key=lambda r: r["wealth"])
    print(f"\n  BEST: K={best['K']} min_conv={best['min_conv']}  COMP={best['COMP']:+.2f}%  wealth=${best['wealth']:,.0f}  neg_wk={best['neg_wk']}/104")

    # Compare to prior (9-lane) and singles
    print(f"\n  Comparison:")
    print(f"    Prior 9-lane union best:           $25,750 / +157.50% / 63 neg wk")
    print(f"    Extended ({extended['lane'].n_unique()}-lane) union best: ${best['wealth']:>9,.0f} / {best['COMP']:>+7.2f}% / {best['neg_wk']} neg wk")
    delta = best['wealth'] - 25750
    print(f"    Delta from extension: ${delta:+,.0f}")
    if best["wealth"] >= 28000:
        print("    1%/wk floor ($28k): CLEARED")
    else:
        gap = 28000 - best["wealth"]
        print(f"    1%/wk floor ($28k): gap ${gap:+,.0f}")

    pl.DataFrame(results).write_parquet(str(OUT_DIR / "union_extended_composed.parquet"))
    return atlas, best, by_lane


if __name__ == "__main__":
    main()
