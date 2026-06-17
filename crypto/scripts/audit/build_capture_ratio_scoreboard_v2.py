"""build_capture_ratio_scoreboard_v2.py - Phase 1 anchor (V2, corrected).

V2 corrections over V1:
  1. Use v3 JSON trade_ledger (per-trade PnL) instead of stdout parser.
     Enables per-event ROI dimension (charter's 3rd lever).
  2. 25Q4 oracle gap closed (outcome_catalog extended to 2025-12-31).
  3. Add signed-median per-day capture alongside clipped.
  4. Use JSON's authoritative max_dd_pct (running-peak basis).
  5. Cost-normalize note: ideal_k5_1d_ret is ALREADY net of 24bps RT
     (per build_outcome_catalog.py line 151). So oracle vs strategy is
     fair-net-of-cost comparison; gross-vs-net concern resolved.

Outputs:
  runs/audit/capture_ratio_scoreboard_v2_perday.parquet
  runs/audit/capture_ratio_scoreboard_v2_pertrade.parquet
  runs/audit/capture_ratio_scoreboard_v2_quarter.parquet
  runs/audit/CAPTURE_RATIO_SCOREBOARD_V2_2026_05_18.md (next; this script prints summary)
"""
from __future__ import annotations
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = ROOT / "logs" / "strat_audit"
STRATEGY_ID = "REGIME_ROUTER_STRICT_LO_SETUP60"

# Map quarter label -> (start_date, end_date, json_basename_window)
QUARTERS = [
    ("24Q1", "20240101_20240331"),
    ("24Q2", "20240401_20240630"),
    ("24Q3", "20240701_20240930"),
    ("24Q4", "20241001_20241231"),
    ("25Q1", "20250101_20250331"),
    ("25Q2", "20250401_20250630"),
    ("25Q3", "20250701_20250930"),
    ("25Q4", "20251001_20251231"),
]

JSON_PATTERN = "paper_trade_replay_v3_{strat}_u100_{window}.json"


def load_quarter_json(quarter: str, window: str) -> dict:
    path = LOGS_DIR / JSON_PATTERN.format(strat=STRATEGY_ID, window=window)
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON for {quarter}: {path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def extract_per_day(j: dict, quarter: str) -> list[dict]:
    rows = []
    for d in j.get("per_day", []):
        rows.append({
            "date": d["date"],
            "quarter": quarter,
            "strategy_id": STRATEGY_ID,
            "day_pnl_pct": float(d["day_pnl_pct"]),
            "nav": float(d["nav"]),
            "n_entries": int(d.get("new_entries", 0)),
            "n_open": int(d.get("open_book_after", 0)),
            "n_closed": int(d.get("closed_today", 0)),
            "btc_30d": float(d.get("btc_30d", 0.0)),
            "dispatch_ok": bool(d.get("dispatch_ok", False)),
        })
    return rows


def extract_trade_ledger(j: dict, quarter: str) -> list[dict]:
    rows = []
    for t in j.get("trade_ledger", []):
        rows.append({
            "quarter": quarter,
            "strategy_id": STRATEGY_ID,
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
    return rows


def extract_q_summary(j: dict, quarter: str) -> dict:
    return {
        "quarter": quarter,
        "n_days": int(j["n_days"]),
        "window_start": j["window_start"],
        "window_end": j["window_end"],
        "nav_initial": float(j["nav_initial"]),
        "nav_final": float(j["nav_final"]),
        "q_return_pct": float(j["total_pnl_pct"]),
        "q_sharpe_ann": float(j["sharpe_annualized"]),
        "q_max_dd_pct": float(j["max_dd_pct"]),
        "n_positive_days": int(j["n_positive_days"]),
        "n_closed_total": int(j["n_closed_total"]),
        "exit_reason_counts": j.get("exit_reason_counts", {}),
    }


def main():
    print("[v2] loading JSONs for 8 quarters...")
    per_day_rows, trade_rows, q_summaries = [], [], []
    for q_label, q_window in QUARTERS:
        j = load_quarter_json(q_label, q_window)
        pd_rows = extract_per_day(j, q_label)
        tr_rows = extract_trade_ledger(j, q_label)
        per_day_rows.extend(pd_rows)
        trade_rows.extend(tr_rows)
        q_summaries.append(extract_q_summary(j, q_label))
        print(f"  {q_label}: days={len(pd_rows)} trades={len(tr_rows)} ret={j['total_pnl_pct']:+.2f}%  sh={j['sharpe_annualized']:+.2f}  dd={j['max_dd_pct']:+.2f}%")

    per_day = pl.DataFrame(per_day_rows).with_columns(pl.col("date").str.to_date())
    trades = pl.DataFrame(trade_rows).with_columns([
        pl.col("entry_date").str.to_date(),
        pl.col("exit_date").str.to_date(),
    ])
    q_sum = pl.DataFrame([{k: v for k, v in q.items() if k != "exit_reason_counts"} for q in q_summaries])

    # Load oracle (now extended through 25Q4)
    oracle = pl.read_parquet(str(ROOT / "data" / "processed" / "outcome_catalog.parquet")).select([
        "date",
        pl.col("ideal_k5_1d_ret").alias("oracle_k5_1d_net"),  # NET of 24bps RT cost
        pl.col("gross_k5_1d_ret").alias("oracle_k5_1d_gross"),
        pl.col("n_assets_available_1d"),
        pl.col("day_class_1d"),
    ])

    # Join per_day with oracle
    sb = per_day.join(oracle, on="date", how="left")
    sb = sb.with_columns([
        (pl.col("day_pnl_pct") / 100.0).alias("realized_frac"),
    ])
    # Signed capture: realized / oracle_net (can be negative if realized < 0)
    sb = sb.with_columns([
        pl.when(pl.col("oracle_k5_1d_net") != 0)
          .then(pl.col("realized_frac") / pl.col("oracle_k5_1d_net"))
          .otherwise(None)
          .alias("capture_ratio_signed"),
        pl.col("oracle_k5_1d_net").is_not_null().alias("has_oracle"),
    ])
    # Clipped capture for distributional stats (no overshoot, no negatives)
    sb = sb.with_columns(
        pl.col("capture_ratio_signed").clip(0.0, 1.0).alias("capture_ratio_clipped")
    )

    # Per-trade scoreboard - already in trades; just add a few derived cols
    trades = trades.with_columns([
        (pl.col("exit_date") - pl.col("entry_date")).dt.total_days().alias("hold_days"),
        (pl.col("net_pnl_pct") - pl.col("gross_ret_pct") * pl.col("size_pct")).alias("cost_drag_pct"),
        (pl.col("net_pnl_pct") > 0).alias("is_winner"),
    ])

    # Write artifacts
    sb_path = OUT_DIR / "capture_ratio_scoreboard_v2_perday.parquet"
    tr_path = OUT_DIR / "capture_ratio_scoreboard_v2_pertrade.parquet"
    sb.write_parquet(str(sb_path))
    trades.write_parquet(str(tr_path))
    print(f"\n[v2] wrote {sb_path}  rows={len(sb)}")
    print(f"[v2] wrote {tr_path}  rows={len(trades)}")

    # --- 8Q rollup ---
    print("\n" + "=" * 80)
    print("8Q ROLLUP (V2)")
    print("=" * 80)
    q_rets = q_sum["q_return_pct"].to_list()
    q_dds = q_sum["q_max_dd_pct"].to_list()
    q_shs = q_sum["q_sharpe_ann"].to_list()

    comp = 1.0
    for r in q_rets:
        comp *= (1.0 + r / 100.0)
    comp_pct = (comp - 1.0) * 100.0

    worst_dd = min(q_dds)  # JSON's running-peak DD per quarter (more honest)
    n_pos_q = sum(1 for r in q_rets if r > 0)
    daily_pnls = sb["day_pnl_pct"].to_list()
    sh_8q_365 = float(np.mean(daily_pnls)) / float(np.std(daily_pnls)) * math.sqrt(365)
    sh_8q_252 = float(np.mean(daily_pnls)) / float(np.std(daily_pnls)) * math.sqrt(252)
    mean_d = float(np.mean(daily_pnls))

    print(f"  COMP 8Q:               {comp_pct:+.2f}%")
    print(f"  Sharpe (SQRT(365)):    {sh_8q_365:+.2f}    [crypto convention]")
    print(f"  Sharpe (SQRT(252)):    {sh_8q_252:+.2f}    [legacy/published]")
    print(f"  Worst Q DD (running):  {worst_dd:+.2f}%")
    print(f"  Positive quarters:     {n_pos_q}/8")
    print(f"  Mean day PnL:          {mean_d:+.4f}%")

    # --- Per-event ROI ---
    print("\n" + "=" * 80)
    print("PER-EVENT (PER-TRADE) ROI DIMENSION  - NEW IN V2")
    print("=" * 80)
    n_tr = len(trades)
    mean_net = float(trades["net_pnl_pct"].mean())
    median_net = float(trades["net_pnl_pct"].median())
    win_rate = float(trades["is_winner"].cast(pl.Float64).mean())
    mean_winner = float(trades.filter(pl.col("is_winner"))["net_pnl_pct"].mean()) if win_rate > 0 else 0.0
    mean_loser = float(trades.filter(~pl.col("is_winner"))["net_pnl_pct"].mean()) if win_rate < 1 else 0.0
    mean_hold = float(trades["hold_days"].mean())
    mean_cost = float(trades["cost_pct"].mean())
    print(f"  Total trades:          {n_tr}")
    print(f"  Mean net_pnl_pct:      {mean_net:+.4f}%   (per trade, of NAV)")
    print(f"  Median net_pnl_pct:    {median_net:+.4f}%")
    print(f"  Win rate:              {win_rate*100:.1f}%")
    print(f"  Mean winner:           {mean_winner:+.4f}%")
    print(f"  Mean loser:            {mean_loser:+.4f}%")
    print(f"  Win/Loss ratio:        {abs(mean_winner/mean_loser):.2f}x" if mean_loser != 0 else "  Win/Loss ratio: inf")
    print(f"  Mean hold days:        {mean_hold:.2f}")
    print(f"  Mean cost_pct:         {mean_cost:.4f}%   (round-trip)")

    print("\n  Exit reason breakdown:")
    by_reason = trades.group_by("exit_reason").agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("is_winner").cast(pl.Float64).mean().alias("win_rate"),
    ]).sort("n", descending=True)
    for row in by_reason.iter_rows(named=True):
        print(f"    {row['exit_reason']:15s}: n={row['n']:>4d}  mean_net={row['mean_net']:+.4f}%  win_rate={row['win_rate']*100:.1f}%")

    print("\n  Per-quarter trade volume:")
    by_q = trades.group_by("quarter").agg([
        pl.len().alias("n_trades"),
        pl.col("net_pnl_pct").mean().alias("mean_net"),
        pl.col("is_winner").cast(pl.Float64).mean().alias("win_rate"),
    ]).sort("quarter")
    for row in by_q.iter_rows(named=True):
        print(f"    {row['quarter']:5s}: n={row['n_trades']:>4d}  mean_net={row['mean_net']:+.4f}%  win_rate={row['win_rate']*100:.1f}%")

    # --- Capture ratio summary ---
    print("\n" + "=" * 80)
    print("CAPTURE RATIO (V2 - oracle extended through 25Q4)")
    print("=" * 80)
    sb_with_oracle = sb.filter(pl.col("has_oracle"))
    total_realized = float(sb_with_oracle["realized_frac"].sum())
    total_oracle = float(sb_with_oracle["oracle_k5_1d_net"].sum())
    cap_agg_8q = total_realized / total_oracle if total_oracle > 0 else float("nan")
    n_oracle_days = len(sb_with_oracle)

    print(f"  Coverage: {n_oracle_days}/{len(sb)} days have oracle ({n_oracle_days/len(sb)*100:.1f}%)")
    print(f"  Total realized (sum):      {total_realized:+.4f}")
    print(f"  Total oracle (sum, NET):   {total_oracle:+.4f}")
    print(f"  Capture ratio aggregate:   {cap_agg_8q*100:+.3f}%   [8Q WF NET-vs-NET]")

    # Distribution stats
    cap_signed = sb_with_oracle["capture_ratio_signed"].drop_nulls().to_numpy()
    cap_clip = sb_with_oracle["capture_ratio_clipped"].drop_nulls().to_numpy()
    print(f"\n  Per-day capture (n={len(cap_signed)} days with oracle):")
    print(f"    signed mean:           {cap_signed.mean()*100:+.2f}%")
    print(f"    signed median:         {np.median(cap_signed)*100:+.2f}%   *** SIGNED ***")
    print(f"    clipped [0,1] mean:    {cap_clip.mean()*100:+.2f}%")
    print(f"    clipped [0,1] median:  {np.median(cap_clip)*100:+.2f}%")

    # Days with positive oracle
    pos_oracle = sb_with_oracle.filter(pl.col("oracle_k5_1d_net") > 0)
    n_po = len(pos_oracle)
    cap_pos = pos_oracle["capture_ratio_signed"].to_numpy()
    print(f"\n  Subset: oracle_net > 0  (n={n_po} days):")
    print(f"    signed median capture: {np.median(cap_pos)*100:+.2f}%   *** ON OPPORTUNITY DAYS ***")
    print(f"    pct negative capture:  {(cap_pos < 0).sum()/n_po*100:.1f}%")
    print(f"    pct zero capture:      {(np.abs(cap_pos) < 0.001).sum()/n_po*100:.1f}%")
    print(f"    pct >= 10% capture:    {(cap_pos >= 0.10).sum()/n_po*100:.1f}%")
    print(f"    pct >= 25% capture:    {(cap_pos >= 0.25).sum()/n_po*100:.1f}%")
    print(f"    pct >= 50% capture:    {(cap_pos >= 0.50).sum()/n_po*100:.1f}%")

    # Per-quarter capture (including 25Q4 now)
    print("\n  Per-quarter capture:")
    q_cap = sb_with_oracle.group_by("quarter").agg([
        pl.col("realized_frac").sum().alias("realized"),
        pl.col("oracle_k5_1d_net").sum().alias("oracle"),
        pl.len().alias("n_days"),
    ]).with_columns(
        (pl.col("realized") / pl.col("oracle")).alias("cap_q")
    ).sort("quarter")
    for row in q_cap.iter_rows(named=True):
        print(f"    {row['quarter']:5s}: days={row['n_days']:>3d}  realized={row['realized']*100:+7.2f}%  oracle={row['oracle']*100:+7.2f}%  cap={row['cap_q']*100:+7.3f}%")

    # Save quarter rollup
    q_sum = q_sum.join(q_cap.select(["quarter", "cap_q"]), on="quarter", how="left")
    q_path = OUT_DIR / "capture_ratio_scoreboard_v2_quarter.parquet"
    q_sum.write_parquet(str(q_path))
    print(f"\n[v2] wrote {q_path}")

    # --- Acceptance gates (REVISED) ---
    print("\n" + "=" * 80)
    print("ACCEPTANCE GATES (V2 -- against PUBLISHED claim)")
    print("=" * 80)
    g2a = abs(comp_pct - 20.95) <= 3.0
    # NOTE: published 0.63 used SQRT(252); we standardize on SQRT(365) crypto
    g2b_252 = abs(sh_8q_252 - 0.63) <= 0.30
    g2b_365 = abs(sh_8q_365 - 0.76) <= 0.30  # SQRT(365) target
    g2c = n_pos_q == 6
    g2d = -15 <= worst_dd <= -5

    print(f"  COMP 20.95% +/- 3pp:         {comp_pct:+.2f}%   pass={g2a}")
    print(f"  Sharpe 0.63 +/- 0.30 (252):  {sh_8q_252:+.2f}      pass={g2b_252}")
    print(f"  Sharpe 0.76 +/- 0.30 (365):  {sh_8q_365:+.2f}      pass={g2b_365}")
    print(f"  Positive quarters 6/8:       {n_pos_q}/8         pass={g2c}")
    print(f"  Worst DD in [-15,-5]%:       {worst_dd:+.2f}%    pass={g2d}")
    print()
    print(f"  Honest 8Q capture ratio (now with 25Q4): {cap_agg_8q*100:+.3f}%")
    print(f"  Implication: ratchet to 1.0%/d baseline = {1.0/(np.mean(daily_pnls)*0.01):.0f}x")
    print(f"  Implication: ratchet to 1.0%/d capture-equivalent = {16.0/(cap_agg_8q*100):.0f}x" if cap_agg_8q > 0 else "  Cap-equivalent ratchet: undefined (negative aggregate)")

    return {
        "comp": comp_pct, "sh365": sh_8q_365, "sh252": sh_8q_252,
        "worst_dd": worst_dd, "n_pos_q": n_pos_q,
        "cap_8q": cap_agg_8q, "mean_d": mean_d,
        "n_trades": n_tr, "mean_event_pnl": mean_net, "win_rate": win_rate,
    }


if __name__ == "__main__":
    main()
