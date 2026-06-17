"""leak_attribution.py — Phase 2 of No-Failure 1-5%/d Campaign.

Three analyses:
  A. Per-day leak classification (on oracle-positive days with capture < 50%)
  B. signal_flip premature-exit analysis: of 1211 signal_flip trades, compute
     "what-if-held-1-more-day" gross_ret using chimera 1d close prices.
  C. Per-quarter exit-mix breakdown (why 25Q1/25Q4 lose; how exit-mix correlates)

Inputs:
  runs/audit/capture_ratio_scoreboard_v2_perday.parquet
  runs/audit/capture_ratio_scoreboard_v2_pertrade.parquet
  data/processed/outcome_catalog.parquet
  data/processed/chimera/1d/<sym>usdt_v51_chimera*.parquet (for what-if analysis)

Outputs:
  runs/audit/leak_attribution_per_day.parquet
  runs/audit/leak_attribution_signal_flip.parquet
  runs/audit/LEAK_ATTRIBUTION_2026_05_18.md (written separately)
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from datetime import date

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"


def load_inputs():
    per_day = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_perday.parquet"))
    trades = pl.read_parquet(str(OUT_DIR / "capture_ratio_scoreboard_v2_pertrade.parquet"))
    oracle = pl.read_parquet(str(ROOT / "data" / "processed" / "outcome_catalog.parquet")).select([
        "date", "ideal_k5_1d_ret", "n_assets_available_1d", "winning_picks_1d",
    ])
    return per_day, trades, oracle


# ---------- Analysis A: per-day leak classification ----------

def classify_per_day_leaks(per_day: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """For each day where capture < 50% AND oracle > 0, classify the leak mode.

    Leak modes:
      ENTRY_NOT_FIRED  : oracle had availability but strategy fired 0 entries
      WRONG_DIRECTION  : strategy fired but day_pnl < 0 (lost on positive day)
      COST_BLEED       : fired, day_pnl in [-0.05%, +0.05%] (small ~ cost band)
      UNDERSIZED       : fired, day_pnl positive but capture < 10%
      MIXED            : fired, day_pnl positive in [10%, 50%) capture (real signal, undersized)
      CAPTURE_OK       : capture >= 50% (no leak)
    """
    df = per_day.filter(pl.col("oracle_k5_1d_net") > 0).with_columns([
        pl.when(pl.col("capture_ratio_signed") >= 0.50).then(pl.lit("CAPTURE_OK"))
          .when(pl.col("n_entries") == 0).then(pl.lit("ENTRY_NOT_FIRED"))
          .when(pl.col("day_pnl_pct") < -0.05).then(pl.lit("WRONG_DIRECTION"))
          .when((pl.col("day_pnl_pct") >= -0.05) & (pl.col("day_pnl_pct") <= 0.05)).then(pl.lit("COST_BLEED"))
          .when(pl.col("capture_ratio_signed") < 0.10).then(pl.lit("UNDERSIZED"))
          .otherwise(pl.lit("MIXED"))
          .alias("leak_mode"),
    ])
    return df


# ---------- Analysis B: signal_flip premature-exit ----------

def get_asset_chimera_map():
    """Map short symbol -> chimera 1d parquet."""
    mapping = {}
    for f in os.listdir(CHIMERA_1D):
        if not f.endswith(".parquet"):
            continue
        sym = f.split("usdt")[0].upper()
        path = CHIMERA_1D / f
        if sym not in mapping or str(path) > str(mapping[sym]):
            mapping[sym] = path
    return mapping


def load_close_panel():
    """Load a wide date x asset close panel from chimera 1d."""
    asset_map = get_asset_chimera_map()
    frames = []
    for sym, path in sorted(asset_map.items()):
        df = pl.read_parquet(path, columns=["date", "close"]).rename({"close": sym})
        frames.append(df)
    wide = frames[0]
    for df in frames[1:]:
        wide = wide.join(df, on="date", how="full", coalesce=True)
    return wide.sort("date")


def whatif_hold_longer(sf_trades: pl.DataFrame, close_panel: pl.DataFrame) -> pl.DataFrame:
    """For each signal_flip trade, compute what-if gross_ret had we held 1 more day.

    Strategy:
      - Entry price = trade.entry_price (already in ledger)
      - Actual exit price at exit_date = trade.exit_price (1-day hold typical)
      - What-if exit price at exit_date + 1d = close_panel[asset, exit_date+1]
      - What-if gross_ret = (whatif_exit / entry) - 1
    """
    # Convert close_panel to date-indexed dict per asset for fast lookup
    asset_cols = [c for c in close_panel.columns if c != "date"]
    close_long = close_panel.unpivot(index="date", variable_name="asset", value_name="close")

    # We want close at exit_date+1 (next chimera bar after exit). Use a shift.
    # For each asset, build a (asset, exit_date_plus_1, close_plus_1) index.
    rows = []
    for asset in asset_cols:
        a_df = close_panel.select(["date", asset]).rename({asset: "close"}).drop_nulls().sort("date")
        a_df = a_df.with_columns([
            pl.col("date").alias("exit_date"),
            pl.col("close").alias("close_at_exit"),
            pl.col("close").shift(-1).alias("close_plus_1d"),
            pl.col("close").shift(-2).alias("close_plus_2d"),
            pl.lit(asset).alias("asset"),
        ])
        rows.append(a_df.select(["asset", "exit_date", "close_at_exit", "close_plus_1d", "close_plus_2d"]))
    future_close = pl.concat(rows)

    # Join onto signal_flip trades
    sf = sf_trades.join(future_close, left_on=["asset", "exit_date"], right_on=["asset", "exit_date"], how="left")
    # Derive what-if gross from close_panel deltas:
    #   exit gross_ret_pct: known (in ledger). Asset close at exit = close_at_exit.
    #   one-more-day asset return = close_plus_1d / close_at_exit - 1
    #   whatif gross (compound): (1 + gross/100) * (1 + next_day_chg) - 1, in pct
    sf = sf.with_columns([
        pl.when(pl.col("close_plus_1d").is_not_null() & (pl.col("close_at_exit") > 0))
          .then(((1 + pl.col("gross_ret_pct") / 100.0) * (pl.col("close_plus_1d") / pl.col("close_at_exit")) - 1) * 100)
          .otherwise(None)
          .alias("whatif_gross_pct_plus1d"),
        pl.when(pl.col("close_plus_2d").is_not_null() & (pl.col("close_at_exit") > 0))
          .then(((1 + pl.col("gross_ret_pct") / 100.0) * (pl.col("close_plus_2d") / pl.col("close_at_exit")) - 1) * 100)
          .otherwise(None)
          .alias("whatif_gross_pct_plus2d"),
    ])
    # Premature flag: actual gross_ret < what-if-plus-1d (signal flipped before continuing in direction)
    sf = sf.with_columns([
        (pl.col("whatif_gross_pct_plus1d") - pl.col("gross_ret_pct")).alias("missed_gross_pct_plus1d"),
        (pl.col("whatif_gross_pct_plus2d") - pl.col("gross_ret_pct")).alias("missed_gross_pct_plus2d"),
        # premature: would have done better holding 1 more day
        (pl.col("whatif_gross_pct_plus1d") > pl.col("gross_ret_pct")).alias("premature_plus1d"),
        # capitulation: would have done WORSE; flip was correct
        (pl.col("whatif_gross_pct_plus1d") < pl.col("gross_ret_pct")).alias("capitulation_correct_plus1d"),
    ])
    return sf


# ---------- Analysis C: per-quarter exit-mix ----------

def quarter_exit_mix(trades: pl.DataFrame) -> pl.DataFrame:
    g = trades.group_by(["quarter", "exit_reason"]).agg([
        pl.len().alias("n"),
        pl.col("net_pnl_pct").sum().alias("total_net_pnl_pct"),
        pl.col("net_pnl_pct").mean().alias("mean_net_pnl_pct"),
        pl.col("gross_ret_pct").mean().alias("mean_gross"),
    ])
    # Pivot to wide
    pv = g.pivot(on="exit_reason", index="quarter", values="n").fill_null(0).sort("quarter")
    # Add quarter totals
    totals = trades.group_by("quarter").agg([
        pl.len().alias("total_trades"),
        pl.col("net_pnl_pct").sum().alias("total_pnl_pct"),
    ])
    pv = pv.join(totals, on="quarter")
    return pv, g


# ---------- main ----------

def main():
    print("[leak_attr] loading inputs...")
    per_day, trades, oracle = load_inputs()
    print(f"  per_day: {len(per_day)} rows")
    print(f"  trades: {len(trades)} rows")
    print(f"  oracle: {len(oracle)} rows")

    # ---- Analysis A: per-day leak classification ----
    print("\n[A] per-day leak classification...")
    leaks_day = classify_per_day_leaks(per_day, trades)
    out_a = OUT_DIR / "leak_attribution_per_day.parquet"
    leaks_day.write_parquet(str(out_a))
    print(f"  wrote {out_a}")

    print("\n  Leak-mode distribution (oracle-positive days, n={}):".format(len(leaks_day)))
    leak_dist = leaks_day.group_by("leak_mode").agg([
        pl.len().alias("n"),
        pl.col("day_pnl_pct").mean().alias("mean_day_pnl"),
        pl.col("capture_ratio_signed").mean().alias("mean_capture"),
        pl.col("oracle_k5_1d_net").mean().alias("mean_oracle_avail"),
    ]).sort("n", descending=True)
    for row in leak_dist.iter_rows(named=True):
        pct = row["n"] / len(leaks_day) * 100
        print(f"    {row['leak_mode']:18s}: n={row['n']:>4d} ({pct:5.1f}%)  day_pnl={row['mean_day_pnl']:+.4f}%  cap={row['mean_capture']*100:+7.2f}%  oracle_avail={row['mean_oracle_avail']*100:+.2f}%")

    # Per-quarter breakdown of leak modes
    print("\n  Per-quarter leak-mode counts:")
    q_leak = leaks_day.group_by(["quarter", "leak_mode"]).agg(pl.len().alias("n"))
    pv_leak = q_leak.pivot(on="leak_mode", index="quarter", values="n").fill_null(0).sort("quarter")
    print(pv_leak)

    # ---- Analysis B: signal_flip premature-exit ----
    print("\n[B] signal_flip premature-exit analysis...")
    sf_trades = trades.filter(pl.col("exit_reason") == "signal_flip")
    print(f"  signal_flip trades to analyze: {len(sf_trades)}")
    print(f"  loading close panel...")
    close_panel = load_close_panel()
    print(f"  close panel: {len(close_panel)} dates x {close_panel.shape[1]-1} assets")

    sf_whatif = whatif_hold_longer(sf_trades, close_panel)
    out_b = OUT_DIR / "leak_attribution_signal_flip.parquet"
    sf_whatif.write_parquet(str(out_b))
    print(f"  wrote {out_b}")

    # Statistics
    sf_with_whatif = sf_whatif.filter(pl.col("whatif_gross_pct_plus1d").is_not_null())
    print(f"\n  signal_flip with what-if data: {len(sf_with_whatif)}/{len(sf_whatif)}")

    actual_mean = float(sf_with_whatif["gross_ret_pct"].mean())
    whatif1_mean = float(sf_with_whatif["whatif_gross_pct_plus1d"].mean())
    whatif2_mean = float(sf_with_whatif["whatif_gross_pct_plus2d"].drop_nulls().mean())
    missed1_mean = float(sf_with_whatif["missed_gross_pct_plus1d"].mean())
    missed2_mean = float(sf_with_whatif["missed_gross_pct_plus2d"].drop_nulls().mean())
    pct_premature1 = float(sf_with_whatif["premature_plus1d"].cast(pl.Float64).mean())

    print(f"\n  GROSS RETURN (signal_flip trades):")
    print(f"    Actual gross at signal_flip exit: {actual_mean:+.3f}%")
    print(f"    What-if held +1d:                 {whatif1_mean:+.3f}%   delta={missed1_mean:+.3f}%")
    print(f"    What-if held +2d:                 {whatif2_mean:+.3f}%   delta={missed2_mean:+.3f}%")
    print(f"    % trades where +1d would have been BETTER: {pct_premature1*100:.1f}%")

    # Estimate the leak: count of premature × avg miss × avg size
    n_premature = int(sf_with_whatif["premature_plus1d"].cast(pl.Int64).sum())
    n_total_sf = len(sf_with_whatif)
    # Per-trade NAV impact if held: missed × size
    sf_with_whatif = sf_with_whatif.with_columns(
        (pl.col("missed_gross_pct_plus1d") * pl.col("size_pct")).alias("missed_nav_pct_plus1d")
    )
    total_missed_nav = float(sf_with_whatif["missed_nav_pct_plus1d"].sum())
    avg_missed_nav = float(sf_with_whatif["missed_nav_pct_plus1d"].mean())
    print(f"\n  NAV IMPACT estimate (if all signal_flip held +1d):")
    print(f"    Total NAV missed (cumulative %):  {total_missed_nav:+.3f}%   (across 8Q)")
    print(f"    Per-trade NAV missed (mean):      {avg_missed_nav:+.4f}%")
    print(f"    Premature exits (would have profited more): {n_premature}/{n_total_sf} ({100*n_premature/n_total_sf:.1f}%)")

    # ---- Analysis C: per-quarter exit-mix ----
    print("\n[C] per-quarter exit-mix...")
    pv_exit, exit_stats = quarter_exit_mix(trades)
    print("\n  Exit-mix counts by quarter:")
    print(pv_exit)

    # Bad-quarter vs good-quarter comparison
    bad_q = ["25Q1", "25Q4"]
    good_q = ["24Q1", "24Q3", "25Q3"]
    print("\n  BAD QUARTERS (25Q1, 25Q4) vs GOOD QUARTERS (24Q1, 24Q3, 25Q3):")
    for q_set, label in [(bad_q, "BAD"), (good_q, "GOOD")]:
        sub = exit_stats.filter(pl.col("quarter").is_in(q_set))
        totals = sub.group_by("exit_reason").agg([
            pl.col("n").sum().alias("n_total"),
            pl.col("total_net_pnl_pct").sum().alias("net_pnl_total"),
            pl.col("mean_net_pnl_pct").mean().alias("mean_per_trade"),
        ]).sort("n_total", descending=True)
        total_sum = float(totals["n_total"].sum())
        print(f"\n    {label} quarters ({', '.join(q_set)}):")
        for r in totals.iter_rows(named=True):
            pct = r["n_total"] / total_sum * 100
            print(f"      {r['exit_reason']:15s}: n={r['n_total']:>4d} ({pct:5.1f}%)  net_pnl_sum={r['net_pnl_total']:+7.2f}%  mean={r['mean_per_trade']:+.4f}%")

    print("\n[leak_attr] done.")


if __name__ == "__main__":
    main()
