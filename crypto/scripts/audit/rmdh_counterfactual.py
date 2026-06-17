"""rmdh_counterfactual.py — R-MDH (Multi-Day Hold) counterfactual.

For each existing trade in the 8Q ledger, simulate what would have happened
with extended hold + realistic stop logic.

Exit logic priority (checked daily at close):
  1. Stop-loss     : entry_pct <= -8%  → exit at -8% gross
  2. Take-profit   : entry_pct >= +15% → exit at +15% gross
  3. Trailing stop : peak_pct >= +3% AND pullback >= -3% → exit at peak - 3% gross
  4. Max hold cap  : 3 days from entry → exit at day+3 close
  (Closes-only approximation; conservative on stop fires.)

Side cap: long-only enforced (matches strategy invariant).
Cost: 0.12% RT (current strategy mean cost_pct).

Outputs:
  runs/audit/rmdh_counterfactual_trades.parquet
  runs/audit/rmdh_counterfactual_per_day.parquet
  runs/audit/RMDH_COUNTERFACTUAL_2026_05_18.md (written separately)
"""
from __future__ import annotations
import os
import math
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"


# Stop config
STOP_LOSS_PCT = -0.08
TAKE_PROFIT_PCT = 0.15
TRAILING_TRIGGER_PCT = 0.03
TRAILING_PULLBACK_PCT = 0.03
MAX_HOLD_DAYS = 3
COST_RT_PCT = 0.12  # round-trip cost as %


def load_close_panel():
    CHIMERA_1D = ROOT / "data/processed/chimera/1d"
    asset_map = {}
    for f in os.listdir(CHIMERA_1D):
        if not f.endswith(".parquet"):
            continue
        sym = f.split("usdt")[0].upper()
        path = CHIMERA_1D / f
        if sym not in asset_map or str(path) > str(asset_map[sym]):
            asset_map[sym] = path
    frames = []
    for sym, path in sorted(asset_map.items()):
        df = pl.read_parquet(path, columns=["date", "close"]).rename({"close": sym})
        frames.append(df)
    wide = frames[0]
    for df in frames[1:]:
        wide = wide.join(df, on="date", how="full", coalesce=True)
    return wide.sort("date")


def simulate_trade(entry_close, future_closes):
    """Simulate one trade's outcome given entry + up to 3 future closes.

    Args:
      entry_close: close at entry date
      future_closes: list of [close+1, close+2, close+3] (None if not available)

    Returns:
      dict with hold_days, exit_reason, gross_pct (decimal), peak_pct
    """
    if entry_close is None or entry_close <= 0:
        return None
    peak_pct = 0.0
    for day_idx in range(1, MAX_HOLD_DAYS + 1):
        if day_idx > len(future_closes) or future_closes[day_idx - 1] is None:
            # Data missing — exit at last available
            last_idx = day_idx - 1
            while last_idx > 0 and (last_idx > len(future_closes) or future_closes[last_idx - 1] is None):
                last_idx -= 1
            if last_idx == 0:
                return None  # no data at all
            return {
                "hold_days": last_idx,
                "exit_reason": "data_gap",
                "gross_pct": future_closes[last_idx - 1] / entry_close - 1,
                "peak_pct": peak_pct,
            }

        c = future_closes[day_idx - 1]
        entry_pct = c / entry_close - 1
        if entry_pct > peak_pct:
            peak_pct = entry_pct

        # 1. Stop-loss (check FIRST — conservative)
        if entry_pct <= STOP_LOSS_PCT:
            return {
                "hold_days": day_idx,
                "exit_reason": "stop_loss",
                "gross_pct": STOP_LOSS_PCT,
                "peak_pct": peak_pct,
            }
        # 2. Take-profit
        if entry_pct >= TAKE_PROFIT_PCT:
            return {
                "hold_days": day_idx,
                "exit_reason": "take_profit",
                "gross_pct": TAKE_PROFIT_PCT,
                "peak_pct": peak_pct,
            }
        # 3. Trailing stop (only after trigger)
        if peak_pct >= TRAILING_TRIGGER_PCT:
            pullback_from_peak = peak_pct - entry_pct
            if pullback_from_peak >= TRAILING_PULLBACK_PCT:
                exit_pct = peak_pct - TRAILING_PULLBACK_PCT
                return {
                    "hold_days": day_idx,
                    "exit_reason": "trailing_stop",
                    "gross_pct": exit_pct,
                    "peak_pct": peak_pct,
                }
        # 4. Continue holding
    # Max hold reached
    return {
        "hold_days": MAX_HOLD_DAYS,
        "exit_reason": "max_hold",
        "gross_pct": future_closes[MAX_HOLD_DAYS - 1] / entry_close - 1,
        "peak_pct": peak_pct,
    }


def main():
    print("[rmdh] loading inputs...")
    trades = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_pertrade.parquet"))
    print(f"  trades: {len(trades)}")

    close_panel = load_close_panel()
    print(f"  close_panel: {len(close_panel)} dates")

    # Build asset->date->close lookup
    asset_cols = [c for c in close_panel.columns if c != "date"]
    # Add shift columns
    rows = []
    for asset in asset_cols:
        a_df = close_panel.select(["date", asset]).rename({asset: "close"}).drop_nulls().sort("date")
        a_df = a_df.with_columns([
            pl.col("date").alias("entry_date"),
            pl.col("close").alias("c0"),
            pl.col("close").shift(-1).alias("c1"),
            pl.col("close").shift(-2).alias("c2"),
            pl.col("close").shift(-3).alias("c3"),
            pl.lit(asset).alias("asset"),
        ])
        rows.append(a_df.select(["asset", "entry_date", "c0", "c1", "c2", "c3"]))
    future = pl.concat(rows)

    joined = trades.join(future, on=["asset", "entry_date"], how="left")
    print(f"  joined: {len(joined)} (missing close lookup: {joined.filter(pl.col('c0').is_null()).height})")

    # Apply simulator row-by-row
    print("[rmdh] simulating each trade...")
    sim_results = []
    for r in joined.iter_rows(named=True):
        c0 = r["c0"]
        futures = [r["c1"], r["c2"], r["c3"]]
        sim = simulate_trade(c0, futures)
        if sim is None:
            sim_results.append({
                "rmdh_hold_days": None, "rmdh_exit_reason": "no_data",
                "rmdh_gross_pct": None, "rmdh_peak_pct": None,
                "rmdh_net_pnl_pct": None,
            })
            continue
        gross = sim["gross_pct"]
        net = gross * 100 * r["size_pct"] - COST_RT_PCT * r["size_pct"]  # net as % NAV
        sim_results.append({
            "rmdh_hold_days": sim["hold_days"],
            "rmdh_exit_reason": sim["exit_reason"],
            "rmdh_gross_pct": gross * 100,
            "rmdh_peak_pct": sim["peak_pct"] * 100,
            "rmdh_net_pnl_pct": net,
        })
    sim_df = pl.DataFrame(sim_results)
    combined = pl.concat([joined, sim_df], how="horizontal")

    # Save
    out_trades = OUT_DIR / "rmdh_counterfactual_trades.parquet"
    combined.write_parquet(str(out_trades))
    print(f"[rmdh] wrote {out_trades}")

    # === SUMMARY STATS ===
    print("\n" + "=" * 80)
    print("R-MDH COUNTERFACTUAL — RESULTS")
    print("=" * 80)
    has_data = combined.filter(pl.col("rmdh_net_pnl_pct").is_not_null())
    no_data = combined.filter(pl.col("rmdh_net_pnl_pct").is_null())
    print(f"Trades with valid simulation: {len(has_data)}/{len(combined)}  (missing: {len(no_data)})")
    print()

    # Per-event stats
    print("PER-EVENT (per-trade) — actual vs R-MDH:")
    print(f"{'metric':>22s}  {'actual':>10s}  {'R-MDH':>10s}  {'delta':>10s}")
    actual_mean_gross = float(trades["gross_ret_pct"].mean())
    rmdh_mean_gross = float(has_data["rmdh_gross_pct"].mean())
    actual_mean_net = float(trades["net_pnl_pct"].mean())
    rmdh_mean_net = float(has_data["rmdh_net_pnl_pct"].mean())
    actual_wr = float((trades["net_pnl_pct"] > 0).cast(pl.Float64).mean())
    rmdh_wr = float((has_data["rmdh_net_pnl_pct"] > 0).cast(pl.Float64).mean())
    actual_hold = float(trades["hold_days"].mean())
    rmdh_hold = float(has_data["rmdh_hold_days"].mean())
    print(f"  Mean gross_ret_pct:    {actual_mean_gross:>+9.3f}%  {rmdh_mean_gross:>+9.3f}%  {rmdh_mean_gross-actual_mean_gross:>+9.3f}%")
    print(f"  Mean net_pnl_pct:      {actual_mean_net:>+9.4f}%  {rmdh_mean_net:>+9.4f}%  {rmdh_mean_net-actual_mean_net:>+9.4f}%")
    print(f"  Win rate:              {actual_wr*100:>9.1f}%  {rmdh_wr*100:>9.1f}%  {(rmdh_wr-actual_wr)*100:>+9.1f}pp")
    print(f"  Mean hold days:        {actual_hold:>9.2f}   {rmdh_hold:>9.2f}   {rmdh_hold-actual_hold:>+9.2f}")

    # Asymmetry
    w = has_data.filter(pl.col("rmdh_net_pnl_pct") > 0)["rmdh_net_pnl_pct"]
    l = has_data.filter(pl.col("rmdh_net_pnl_pct") < 0)["rmdh_net_pnl_pct"]
    if len(w) > 0 and len(l) > 0:
        rmdh_asym = abs(float(w.mean()) / float(l.mean()))
        print(f"  Asymmetry (mean):      {1.37:>9.2f}x  {rmdh_asym:>9.2f}x")

    # Exit reason breakdown
    print()
    print("R-MDH exit reason mix:")
    er = has_data.group_by("rmdh_exit_reason").agg([
        pl.len().alias("n"),
        pl.col("rmdh_net_pnl_pct").mean().alias("mean_net"),
        pl.col("rmdh_gross_pct").mean().alias("mean_gross"),
        (pl.col("rmdh_net_pnl_pct") > 0).cast(pl.Float64).mean().alias("wr"),
    ]).sort("n", descending=True)
    for r in er.iter_rows(named=True):
        pct = r["n"] / len(has_data) * 100
        print(f"  {r['rmdh_exit_reason']:15s}: n={r['n']:>4d} ({pct:5.1f}%)  mean_gross={r['mean_gross']:+.3f}%  mean_net={r['mean_net']:+.4f}%  wr={r['wr']*100:.0f}%")

    # Aggregate NAV: sum of net_pnl_pct (assign to entry_date for daily aggregation)
    print()
    print("AGGREGATE NAV IMPACT (sum of net_pnl_pct over 8Q):")
    actual_sum = float(trades["net_pnl_pct"].sum())
    rmdh_sum = float(has_data["rmdh_net_pnl_pct"].sum())
    print(f"  Actual sum net_pnl:   {actual_sum:+.2f}%")
    print(f"  R-MDH sum net_pnl:    {rmdh_sum:+.2f}%   delta: {rmdh_sum-actual_sum:+.2f}%")

    # Per-day aggregation (assign net to entry_date)
    # Then chain into weeks + quarters
    per_day_rmdh = has_data.group_by("entry_date").agg(
        pl.col("rmdh_net_pnl_pct").sum().alias("rmdh_day_pnl_pct")
    ).rename({"entry_date": "date"})

    sb = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_perday.parquet"))
    sb = sb.join(per_day_rmdh, on="date", how="left").with_columns(
        pl.col("rmdh_day_pnl_pct").fill_null(0.0)
    )

    # Save per-day
    sb_out = OUT_DIR / "rmdh_counterfactual_per_day.parquet"
    sb.write_parquet(str(sb_out))
    print(f"\n[rmdh] wrote {sb_out}")

    # ISO week aggregate
    sb = sb.with_columns([
        pl.col("date").dt.year().alias("y"),
        pl.col("date").dt.week().alias("w"),
    ])
    sb = sb.with_columns((pl.col("y").cast(pl.Utf8) + "-W" + pl.col("w").cast(pl.Utf8).str.zfill(2)).alias("iso_week"))

    weekly_actual = sb.group_by("iso_week").agg(
        (((pl.col("day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("wk_actual")
    )
    weekly_rmdh = sb.group_by("iso_week").agg(
        (((pl.col("rmdh_day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("wk_rmdh")
    )
    weekly = weekly_actual.join(weekly_rmdh, on="iso_week").with_columns([
        (pl.col("wk_actual") * 100).alias("wk_actual_pct"),
        (pl.col("wk_rmdh") * 100).alias("wk_rmdh_pct"),
    ])
    neg_actual = (weekly["wk_actual_pct"] < 0).sum()
    neg_rmdh = (weekly["wk_rmdh_pct"] < 0).sum()
    print()
    print("WEEK-LEVEL:")
    print(f"  Negative weeks (actual):  {neg_actual}/104 ({neg_actual/104*100:.1f}%)")
    print(f"  Negative weeks (R-MDH):   {neg_rmdh}/104 ({neg_rmdh/104*100:.1f}%)   delta: {neg_rmdh-neg_actual:+d}")
    print(f"  Median week (actual):     {weekly['wk_actual_pct'].median():+.3f}%")
    print(f"  Median week (R-MDH):      {weekly['wk_rmdh_pct'].median():+.3f}%")
    print(f"  Worst week (actual):      {weekly['wk_actual_pct'].min():+.2f}%")
    print(f"  Worst week (R-MDH):       {weekly['wk_rmdh_pct'].min():+.2f}%")
    print(f"  Best week (actual):       {weekly['wk_actual_pct'].max():+.2f}%")
    print(f"  Best week (R-MDH):        {weekly['wk_rmdh_pct'].max():+.2f}%")

    # Per-quarter 8Q rollup
    q_actual = sb.group_by("quarter").agg(
        (((pl.col("day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("q_actual")
    )
    q_rmdh = sb.group_by("quarter").agg(
        (((pl.col("rmdh_day_pnl_pct") / 100 + 1).log().sum().exp()) - 1).alias("q_rmdh")
    )
    q_combined = q_actual.join(q_rmdh, on="quarter").sort("quarter")

    print()
    print("PER-QUARTER:")
    print(f"  {'Q':5s} {'actual':>10s} {'R-MDH':>10s} {'delta':>10s}")
    actual_comp = 1.0
    rmdh_comp = 1.0
    n_pos_actual = 0
    n_pos_rmdh = 0
    for r in q_combined.iter_rows(named=True):
        actual_comp *= (1 + r["q_actual"])
        rmdh_comp *= (1 + r["q_rmdh"])
        if r["q_actual"] > 0: n_pos_actual += 1
        if r["q_rmdh"] > 0: n_pos_rmdh += 1
        print(f"  {r['quarter']:5s} {r['q_actual']*100:>+9.2f}% {r['q_rmdh']*100:>+9.2f}% {(r['q_rmdh']-r['q_actual'])*100:>+9.2f}%")
    print()
    print(f"  8Q COMP actual:           {(actual_comp-1)*100:+.2f}%")
    print(f"  8Q COMP R-MDH:            {(rmdh_comp-1)*100:+.2f}%   delta: {(rmdh_comp-actual_comp)*100:+.2f}%")
    print(f"  Positive quarters actual: {n_pos_actual}/8")
    print(f"  Positive quarters R-MDH:  {n_pos_rmdh}/8")

    # 8Q wealth path (running-peak DD)
    actual_wealth = 10000.0
    rmdh_wealth = 10000.0
    actual_peak = 10000.0
    rmdh_peak = 10000.0
    actual_max_dd = 0.0
    rmdh_max_dd = 0.0
    for r in q_combined.iter_rows(named=True):
        actual_wealth *= (1 + r["q_actual"])
        rmdh_wealth *= (1 + r["q_rmdh"])
        if actual_wealth > actual_peak: actual_peak = actual_wealth
        if rmdh_wealth > rmdh_peak: rmdh_peak = rmdh_wealth
        dd_a = (actual_wealth / actual_peak - 1) * 100
        dd_r = (rmdh_wealth / rmdh_peak - 1) * 100
        if dd_a < actual_max_dd: actual_max_dd = dd_a
        if dd_r < rmdh_max_dd: rmdh_max_dd = dd_r
    print(f"  2yr wealth (actual):      ${actual_wealth:,.2f}  max_DD={actual_max_dd:+.2f}%")
    print(f"  2yr wealth (R-MDH):       ${rmdh_wealth:,.2f}  max_DD={rmdh_max_dd:+.2f}%")
    print(f"  Wealth delta:             ${rmdh_wealth-actual_wealth:+,.2f}")

    print()
    print("=" * 80)
    print("ACCEPTANCE GATES")
    print("=" * 80)
    g1 = (rmdh_comp - 1) * 100 >= 35
    g2 = neg_rmdh <= 50
    rmdh_asym_final = abs(float(w.mean()) / float(l.mean())) if len(w) > 0 and len(l) > 0 else 0
    g3 = rmdh_asym_final >= 2.0
    g4 = weekly["wk_rmdh_pct"].min() >= -6
    g5 = (rmdh_max_dd >= -10)
    print(f"  G1: 8Q COMP >= +35%:               {(rmdh_comp-1)*100:+.2f}%   pass={g1}")
    print(f"  G2: Neg weeks <= 50/104:           {neg_rmdh}/104    pass={g2}")
    print(f"  G3: Asymmetry >= 2.0x:             {rmdh_asym_final:.2f}x      pass={g3}")
    print(f"  G4: Worst week >= -6%:             {weekly['wk_rmdh_pct'].min():+.2f}%   pass={g4}")
    print(f"  G5: 8Q wealth DD >= -10%:          {rmdh_max_dd:+.2f}%   pass={g5}")
    all_pass = g1 and g2 and g3 and g4 and g5
    print(f"\n  ALL GATES PASS: {all_pass}")


if __name__ == "__main__":
    main()
