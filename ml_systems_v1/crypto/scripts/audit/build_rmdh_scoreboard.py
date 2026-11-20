"""build_rmdh_scoreboard.py — Aggregate 8Q R-MDH v3 outputs into scoreboard.

Parses the JSON outputs from `paper_trade_replay_v3.py --rmdh-*` runs
across 8 quarters and produces per-day + per-quarter + per-week scoreboard
artifacts comparable to the baseline V2 scoreboard.

Inputs (auto-discovered):
  logs/strat_audit/paper_trade_replay_v3_REGIME_ROUTER_STRICT_LO_SETUP60_u100_<startdate>_<enddate>.json
  (8 quarters: 20240101..., 20240401..., 20240701..., 20241001...,
              20250101..., 20250401..., 20250701..., 20251001...)

Outputs:
  runs/audit/rmdh_scoreboard_perday.parquet
  runs/audit/rmdh_scoreboard_pertrade.parquet
  runs/audit/rmdh_scoreboard_quarter.parquet
  Prints: per-quarter + 8Q rollup + comparison to baseline V2
"""
from __future__ import annotations
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
LOGS_DIR = ROOT / "logs" / "strat_audit"
STRATEGY_ID = "REGIME_ROUTER_STRICT_LO_SETUP60"

# R-MDH runner uses (start + days - 1) for end_date.
# 24Q1=90d so end is 2024-03-30 (not -31 like baseline 91-day window).
QUARTERS = [
    ("24Q1", "20240101_20240330"),
    ("24Q2", "20240401_20240630"),
    ("24Q3", "20240701_20240930"),
    ("24Q4", "20241001_20241231"),
    ("25Q1", "20250101_20250331"),
    ("25Q2", "20250401_20250630"),
    ("25Q3", "20250701_20250930"),
    ("25Q4", "20251001_20251231"),
]


def find_latest_json(quarter_window: str) -> Path | None:
    """Find newest JSON matching the quarter window."""
    # Prefer files containing the quarter window string
    candidates = sorted(
        LOGS_DIR.glob(f"paper_trade_replay_v3_{STRATEGY_ID}_u100_{quarter_window}.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def main():
    print(f"[rmdh-scoreboard] scanning {LOGS_DIR} for R-MDH quarter JSONs...")
    per_day_rows, trade_rows, q_summaries = [], [], []
    missing = []
    for q_label, q_window in QUARTERS:
        p = find_latest_json(q_window)
        if p is None:
            print(f"  {q_label}: MISSING JSON")
            missing.append(q_label)
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            j = json.load(f)
        mtime = p.stat().st_mtime
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).isoformat(timespec="minutes")
        print(f"  {q_label}: {p.name}  ret={j['total_pnl_pct']:+.2f}%  sh={j['sharpe_annualized']:+.2f}  dd={j['max_dd_pct']:+.2f}%  n_trades={j['n_closed_total']}  (modified {ts})")

        for d in j.get("per_day", []):
            per_day_rows.append({
                "date": d["date"],
                "quarter": q_label,
                "day_pnl_pct": float(d["day_pnl_pct"]),
                "nav": float(d["nav"]),
                "n_entries": int(d.get("new_entries", 0)),
                "n_open": int(d.get("open_book_after", 0)),
                "n_closed": int(d.get("closed_today", 0)),
                "btc_30d": float(d.get("btc_30d", 0.0)),
            })

        for t in j.get("trade_ledger", []):
            trade_rows.append({
                "quarter": q_label,
                "sleeve": t.get("sleeve", ""),
                "asset": t.get("asset", ""),
                "side": t.get("side", ""),
                "entry_date": t.get("entry_date", ""),
                "exit_date": t.get("exit_date", ""),
                "size_pct": float(t.get("size_pct", 0.0)),
                "gross_ret_pct": float(t.get("gross_ret_pct", 0.0)),
                "cost_pct": float(t.get("cost_pct", 0.0)),
                "net_pnl_pct": float(t.get("net_pnl_pct", 0.0)),
                "exit_reason": t.get("exit_reason", ""),
            })

        q_summaries.append({
            "quarter": q_label,
            "n_days": int(j["n_days"]),
            "q_return_pct": float(j["total_pnl_pct"]),
            "q_sharpe_ann": float(j["sharpe_annualized"]),
            "q_max_dd_pct": float(j["max_dd_pct"]),
            "n_positive_days": int(j["n_positive_days"]),
            "n_closed_total": int(j["n_closed_total"]),
        })

    if missing:
        print(f"\n[WARN] {len(missing)} quarter(s) missing: {missing}")
        if len(missing) == len(QUARTERS):
            print("All quarters missing — cannot aggregate.")
            return

    if not per_day_rows:
        print("No data — exiting.")
        return

    per_day = pl.DataFrame(per_day_rows).with_columns(pl.col("date").str.to_date())
    trades = pl.DataFrame(trade_rows).with_columns([
        pl.col("entry_date").str.to_date(),
        pl.col("exit_date").str.to_date(),
    ]).with_columns([
        (pl.col("exit_date") - pl.col("entry_date")).dt.total_days().alias("hold_days"),
    ])
    q_sum = pl.DataFrame(q_summaries)

    per_day.write_parquet(str(OUT_DIR / "rmdh_scoreboard_perday.parquet"))
    trades.write_parquet(str(OUT_DIR / "rmdh_scoreboard_pertrade.parquet"))
    q_sum.write_parquet(str(OUT_DIR / "rmdh_scoreboard_quarter.parquet"))
    print(f"\n[rmdh-scoreboard] wrote 3 artifacts in {OUT_DIR}")

    # === SUMMARY vs BASELINE ===
    print("\n" + "=" * 80)
    print("R-MDH 8Q ROLLUP vs BASELINE V2")
    print("=" * 80)
    base = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_perday.parquet"))
    base_pertrade = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_pertrade.parquet"))

    # COMP
    q_rets_rmdh = q_sum["q_return_pct"].to_list()
    comp_rmdh = 1.0
    for r in q_rets_rmdh:
        comp_rmdh *= (1 + r / 100)
    comp_base = 1.2087  # from V2 audit

    daily_rmdh = per_day["day_pnl_pct"].to_list()
    daily_base = base["day_pnl_pct"].to_list()
    sh_rmdh = float(np.mean(daily_rmdh)) / float(np.std(daily_rmdh)) * math.sqrt(365)
    sh_base = float(np.mean(daily_base)) / float(np.std(daily_base)) * math.sqrt(365)

    print(f"  {'Metric':>25s}  {'Baseline V2':>14s}  {'R-MDH':>14s}  {'Δ':>10s}")
    print(f"  {'8Q COMP':>25s}  {(comp_base-1)*100:>+13.2f}%  {(comp_rmdh-1)*100:>+13.2f}%  {(comp_rmdh-comp_base)*100:>+9.2f}%")
    print(f"  {'Sharpe (365)':>25s}  {sh_base:>+14.2f}  {sh_rmdh:>+14.2f}  {sh_rmdh-sh_base:>+10.2f}")

    # 2yr wealth
    wealth_rmdh = 10000 * comp_rmdh
    wealth_base = 10000 * comp_base
    print(f"  {'2yr wealth $10k':>25s}  ${wealth_base:>13,.0f}  ${wealth_rmdh:>13,.0f}  ${wealth_rmdh-wealth_base:>+9,.0f}")

    # Per-event
    mean_net_rmdh = float(trades["net_pnl_pct"].mean())
    mean_net_base = float(base_pertrade["net_pnl_pct"].mean())
    wr_rmdh = float((trades["net_pnl_pct"] > 0).cast(pl.Float64).mean())
    wr_base = float((base_pertrade["net_pnl_pct"] > 0).cast(pl.Float64).mean())
    mean_hold_rmdh = float(trades["hold_days"].mean())
    mean_hold_base = float(base_pertrade["hold_days"].mean())
    print(f"  {'Mean net/trade':>25s}  {mean_net_base:>+13.4f}%  {mean_net_rmdh:>+13.4f}%  {mean_net_rmdh-mean_net_base:>+9.4f}%")
    print(f"  {'Win rate':>25s}  {wr_base*100:>13.1f}%  {wr_rmdh*100:>13.1f}%  {(wr_rmdh-wr_base)*100:>+9.1f}pp")
    print(f"  {'Mean hold days':>25s}  {mean_hold_base:>14.2f}  {mean_hold_rmdh:>14.2f}  {mean_hold_rmdh-mean_hold_base:>+10.2f}")

    # Negative weeks
    per_day_wk = per_day.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.week().alias("w"),
    ])
    per_day_wk = per_day_wk.with_columns(
        (pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week")
    )
    weekly = per_day_wk.group_by("iso_week").agg(
        (((pl.col("day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("wk_ret")
    )
    n_neg_rmdh = (weekly["wk_ret"] < 0).sum()
    print(f"  {'Negative weeks':>25s}  {61:>13d}/104  {n_neg_rmdh:>13d}/104  {n_neg_rmdh-61:>+10d}")

    # Asymmetry
    w_rmdh = trades.filter(pl.col("net_pnl_pct") > 0)["net_pnl_pct"]
    l_rmdh = trades.filter(pl.col("net_pnl_pct") < 0)["net_pnl_pct"]
    asym_rmdh = abs(float(w_rmdh.mean()) / float(l_rmdh.mean())) if len(l_rmdh) > 0 else 0
    print(f"  {'Asymmetry':>25s}  {1.37:>14.2f}x  {asym_rmdh:>13.2f}x  {asym_rmdh-1.37:>+10.2f}x")

    # Exit-reason mix
    print("\n  R-MDH exit-reason mix:")
    er = trades.group_by("exit_reason").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("gross_ret_pct").mean().alias("mean_gross"),
        (pl.col("net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("n", descending=True)
    for r in er.iter_rows(named=True):
        pct = r["n"] / len(trades) * 100
        print(f"    {r['exit_reason']:15s}: n={r['n']:>5d} ({pct:5.1f}%)  mean_gross={r['mean_gross']:+.3f}%  mean_net={r['mean_net']:+.4f}%  wr={r['wr']*100:.0f}%")

    # Counterfactual vs realized
    print("\n  vs counterfactual prediction (COMP +75.25%, neg_wk 56):")
    cf_comp = 75.25
    cf_neg = 56
    realized_pct_of_cf = ((comp_rmdh - 1) * 100 / cf_comp) * 100 if cf_comp > 0 else 0
    print(f"    Realized COMP {(comp_rmdh-1)*100:+.2f}% = {realized_pct_of_cf:.0f}% of counterfactual")
    print(f"    Realized neg weeks {n_neg_rmdh}/104 vs counterfactual {cf_neg}/104")


if __name__ == "__main__":
    main()
