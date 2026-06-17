"""Capture-ratio analysis: top setups vs oracle ceiling on TRAIN window.

For each (indicator, config), compute:
  realized_nav = setup's actual cumulative return @ 4% sizing
  available_nav (oracle) = perfect-foresight ceiling using one of 3 definitions

ORACLE DEFINITIONS:
  ORACLE_TOP1:  best 14d-forward return across universe on every TRAIN day
  ORACLE_TOP5:  top-5 14d-forward returns across universe (≥5% bar where ≥5 assets)
  ORACLE_FIRED: max 14d-forward return across universe ON DAYS SETUP FIRED
                (per-event oracle, isolates entry timing from entry asset selection)

OUTPUTS:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/capture_ratio_analysis.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/capture_ratio_REPORT.md
"""
from __future__ import annotations
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)
COST = 0.0024
SIZE = 0.04

def load_oracle_panel():
    """Per-asset, per-date 14d-forward return panel on TRAIN window."""
    print("Loading panel for oracle ceiling computation...")
    files = sorted((ROOT / "data" / "processed" / "chimera" / "1d").glob("*_v51_chimera_1d_*.parquet"))
    rows = []
    for f in files:
        sym = f.name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].reset_index(drop=True)
        if len(df) < 30:
            continue
        df["asset"] = sym
        df["close_fwd14"] = df["close"].shift(-14)
        df["ret_fwd14"] = (df["close_fwd14"] / df["close"] - 1) - COST
        rows.append(df[["asset", "date", "close", "ret_fwd14"]].dropna())
    panel = pd.concat(rows, ignore_index=True)
    print(f"  panel: {len(panel):,} rows × {panel['asset'].nunique()} assets")
    return panel

def compute_oracle_ceilings(panel):
    """Compute per-day oracle ceilings."""
    # ORACLE_TOP1: best 14d-fwd return across universe each day
    daily = panel.groupby("date")["ret_fwd14"].agg([
        ("max_ret", "max"),
        ("top5_mean", lambda x: x.nlargest(min(5, len(x))).mean()),
        ("top5_sum", lambda x: x.nlargest(min(5, len(x))).sum()),
        ("n_active", "count"),
        ("n_movers_5pct", lambda x: (x >= 0.05).sum()),
        ("n_movers_10pct", lambda x: (x >= 0.10).sum()),
    ]).reset_index()

    print(f"  daily oracle: {len(daily):,} days")
    print(f"  total ORACLE_TOP1 NAV @4%: +{daily['max_ret'].sum() * SIZE * 100:.2f}%")
    print(f"  total ORACLE_TOP5 NAV @4% (sum of top5/day): +{daily['top5_sum'].sum() * SIZE * 100:.2f}%")
    print(f"  days with >=1 mover >=5%: {(daily['n_movers_5pct'] >= 1).sum()} ({(daily['n_movers_5pct'] >= 1).mean()*100:.1f}%)")
    return daily

def compute_per_setup_capture(events, daily_oracle):
    """For each (indicator, config), compute capture metrics."""
    daily_idx = daily_oracle.set_index("date").to_dict("index")
    out = []
    for (ind, cfg), grp in events.groupby(["indicator", "config"]):
        n = len(grp)
        if n < 200:
            continue
        # Realized NAV (E_14d as primary outcome)
        e14 = grp["ret_E_14d"].dropna()
        realized_nav = e14.sum() * SIZE * 100
        # ORACLE_FIRED: on days this setup fired, what was the best forward 14d return?
        fired_oracle = []
        for d in grp["date"]:
            if d in daily_idx:
                fired_oracle.append(daily_idx[d]["max_ret"])
        fired_oracle = np.array(fired_oracle)
        fired_oracle_nav = fired_oracle.sum() * SIZE * 100  # if you'd picked the right asset every fire day
        # ORACLE_TOP1 ceiling across full TRAIN
        total_oracle_top1 = daily_oracle["max_ret"].sum() * SIZE * 100
        total_oracle_top5 = daily_oracle["top5_sum"].sum() * SIZE * 100

        out.append({
            "indicator": ind,
            "config": cfg,
            "n_events": n,
            "realized_nav_pct": round(realized_nav, 2),
            "realized_mean_pct": round(e14.mean() * 100, 3),
            "oracle_FIRED_top1_nav_pct": round(fired_oracle_nav, 2),
            "oracle_FIRED_mean_pct": round(fired_oracle.mean() * 100, 3) if len(fired_oracle) else 0,
            "capture_pct_vs_FIRED": round(realized_nav / fired_oracle_nav * 100, 2) if fired_oracle_nav > 0 else 0,
            "oracle_TRAIN_top1_nav_pct": round(total_oracle_top1, 2),
            "capture_pct_vs_TOP1": round(realized_nav / total_oracle_top1 * 100, 3) if total_oracle_top1 > 0 else 0,
            "oracle_TRAIN_top5_nav_pct": round(total_oracle_top5, 2),
            "capture_pct_vs_TOP5": round(realized_nav / total_oracle_top5 * 100, 3) if total_oracle_top5 > 0 else 0,
        })
    df = pd.DataFrame(out)
    return df.sort_values(["indicator", "realized_nav_pct"], ascending=[True, False])

def rank_indicators_by_avg(events):
    """Rank indicators by avg mean-per-event across all their configs (where n_events ≥ 200)."""
    indicator_stats = []
    for ind, grp in events.groupby("indicator"):
        # Aggregate per-config first, then average over configs
        per_cfg = grp.groupby("config")["ret_E_14d"].agg(["count", "mean", "sum"])
        per_cfg = per_cfg[per_cfg["count"] >= 200]
        if len(per_cfg) == 0: continue
        # Weighted by n_events: total return / total events
        indicator_stats.append({
            "indicator": ind,
            "n_configs_qualifying": len(per_cfg),
            "avg_mean_pct": per_cfg["mean"].mean() * 100,         # unweighted avg of per-cfg means
            "median_mean_pct": per_cfg["mean"].median() * 100,
            "weighted_mean_pct": (per_cfg["sum"].sum() / per_cfg["count"].sum()) * 100,
            "total_n_events": int(per_cfg["count"].sum()),
            "total_realized_nav_pct": per_cfg["sum"].sum() * SIZE * 100,
        })
    return pd.DataFrame(indicator_stats).sort_values("weighted_mean_pct", ascending=False)

def main():
    print("="*78)
    print("CAPTURE RATIO ANALYSIS — TRAIN")
    print("="*78)
    events = pd.read_parquet(OUT_DIR / "per_event_raw.parquet")
    print(f"Loaded events: {len(events):,}")

    panel = load_oracle_panel()
    daily_oracle = compute_oracle_ceilings(panel)
    daily_oracle.to_csv(OUT_DIR / "daily_oracle_train.csv", index=False)

    print("\nRanking indicators by average performance...")
    ind_rank = rank_indicators_by_avg(events)
    print(ind_rank.to_string(index=False))
    ind_rank.to_csv(OUT_DIR / "indicator_avg_ranking.csv", index=False)

    print("\nComputing per-setup capture ratios...")
    capture = compute_per_setup_capture(events, daily_oracle)
    capture.to_csv(OUT_DIR / "capture_ratio_analysis.csv", index=False)

    # Top-3 setups for top-3 indicators
    print("\n=== TOP 3 SETUPS FOR TOP 3 INDICATORS (by avg weighted mean per event) ===")
    top3_ind = ind_rank.head(3)["indicator"].tolist()
    for ind in top3_ind:
        print(f"\n--- {ind} ---")
        ind_cap = capture[capture["indicator"] == ind].head(3)
        print(ind_cap[["config","n_events","realized_nav_pct","realized_mean_pct",
                       "oracle_FIRED_top1_nav_pct","capture_pct_vs_FIRED",
                       "capture_pct_vs_TOP5"]].to_string(index=False))

    # Write report
    lines = ["# Capture Ratio Analysis — TRAIN\n"]
    lines.append(f"\n## A) Top 3 indicators by weighted mean per event\n")
    lines.append("| Rank | Indicator | Weighted mean/event | n_configs | Total n_events | Realized NAV @4% |")
    lines.append("|---:|---|--:|--:|--:|--:|")
    for i, r in ind_rank.head(3).iterrows():
        lines.append(f"| {i+1} | {r['indicator']} | {r['weighted_mean_pct']:+.3f}% | {r['n_configs_qualifying']} | {r['total_n_events']:,} | {r['total_realized_nav_pct']:+.2f}% |")
    lines.append("\nFull ranking:")
    lines.append("| Rank | Indicator | Weighted mean | Median per-cfg mean | n_cfg | Total events |")
    lines.append("|---:|---|--:|--:|--:|--:|")
    for i, r in ind_rank.iterrows():
        lines.append(f"| {i+1} | {r['indicator']} | {r['weighted_mean_pct']:+.3f}% | {r['median_mean_pct']:+.3f}% | {r['n_configs_qualifying']} | {r['total_n_events']:,} |")

    lines.append(f"\n## B) Top 3 setups for each of top 3 indicators (with capture %)\n")
    for ind in top3_ind:
        ind_cap = capture[capture["indicator"] == ind].head(3)
        lines.append(f"\n### {ind}")
        lines.append("| config | n | realized NAV | mean/event | ORACLE_FIRED NAV* | capture vs FIRED | capture vs TOP5 |")
        lines.append("|---|--:|--:|--:|--:|--:|--:|")
        for _, r in ind_cap.iterrows():
            lines.append(f"| `{r['config']}` | {r['n_events']:,} | {r['realized_nav_pct']:+.2f}% | {r['realized_mean_pct']:+.3f}% | {r['oracle_FIRED_top1_nav_pct']:+.2f}% | {r['capture_pct_vs_FIRED']:.2f}% | {r['capture_pct_vs_TOP5']:.3f}% |")
        lines.append("\n*ORACLE_FIRED = best 14d-fwd return across universe ON DAYS THIS SETUP FIRED. Isolates entry timing from asset-selection skill.")

    lines.append(f"\n## C) Oracle ceiling reference\n")
    lines.append(f"\nTRAIN window: {TRAIN_START} -> {TRAIN_END} ({len(daily_oracle)} trading days)")
    lines.append(f"\n- **ORACLE_TOP1** (perfect asset pick / day): +{daily_oracle['max_ret'].sum() * SIZE * 100:,.0f}% NAV @4% sizing")
    lines.append(f"- **ORACLE_TOP5** (top-5 / day, fits LO+5-position-cap): +{daily_oracle['top5_sum'].sum() * SIZE * 100:,.0f}% NAV @4% sizing")
    lines.append(f"- Days with ≥1 mover ≥5%: {(daily_oracle['n_movers_5pct'] >= 1).sum()} ({(daily_oracle['n_movers_5pct'] >= 1).mean()*100:.1f}% of TRAIN)")
    lines.append(f"- Days with ≥1 mover ≥10%: {(daily_oracle['n_movers_10pct'] >= 1).sum()} ({(daily_oracle['n_movers_10pct'] >= 1).mean()*100:.1f}% of TRAIN)")
    lines.append(f"- Median day-best 14d-fwd return: {daily_oracle['max_ret'].median()*100:.2f}%")
    lines.append(f"- Mean day-best 14d-fwd return: {daily_oracle['max_ret'].mean()*100:.2f}%")

    lines.append(f"\n## D) Headline interpretation\n")
    lines.append(f"\nThe oracle ceiling under perfect 1-pick-per-day is ENORMOUS (~10K-100K% over 3.5 years).")
    lines.append(f"Capture vs TOP1/TOP5 looks tiny because no rule-based strategy with single-config")
    lines.append(f"firing matches a daily-perfect oracle. The more meaningful number is **capture vs FIRED**")
    lines.append(f"— on days this setup fired, did it pick the best forward-14d asset that day? That metric")
    lines.append(f"isolates entry-timing skill from asset-selection skill.")

    (OUT_DIR / "capture_ratio_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'capture_ratio_REPORT.md'}")
    print(f"Wrote {OUT_DIR / 'capture_ratio_analysis.csv'}")

if __name__ == "__main__":
    main()
