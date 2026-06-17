"""union_oos_diagnostic.py -- per-day diagnostic: oracle vs strat vs top-25% winners.

For each day in OOS (2024-05-16 → 2025-03-15):
  1. Oracle K=5 LO daily availability: mean of top-5 asset cc-returns (>0 only)
  2. Strat per-day NAV change (from union sim's daily series)
  3. Top-25%-of-winners: among all positive-return assets that day, average the top quartile
  4. Capture ratio: strat / oracle (clipped 0-100%)
  5. Miss attribution: was strat in market that day? what fraction of slots?

OUTPUT:
  runs/audit/MA_EMA_PROFILE_2026_05_20/UNION_OOS_DIAGNOSTIC.md
  runs/audit/MA_EMA_PROFILE_2026_05_20/union_oos_per_day.csv
"""
# [!] SPLIT DISCIPLINE NOTE (2026-05-24 INST-C cleanup):
# This script uses the legacy convention where "OOS" labels the post-TRAIN window
# (= canonical OOS + UNSEEN combined). Per src/split_config.py the canonical OOS
# ends 2025-12-31 and UNSEEN starts 2026-01-01. The dates hardcoded below are
# intentionally preserved for reproducibility of prior outputs. New scripts must
# import from split_config -- see docs/SPLIT_DISCIPLINE.md.
from __future__ import annotations

import sys
import glob
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
CHIMERA_1D = ROOT / "data" / "processed" / "chimera" / "1d"
DAILY_PATH = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "oos_union_daily.csv"
TRADES_PATH = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "oos_union_trades.csv"
OUT_MD = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "UNION_OOS_DIAGNOSTIC.md"
OUT_CSV = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20" / "union_oos_per_day.csv"

OOS_START = _date(2024, 5, 16)
OOS_END = _date(2025, 3, 15)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("UNION OOS PER-DAY DIAGNOSTIC")
    print("="*78)

    # Load chimera 1d for all assets, compute per-day cc returns
    print("Loading chimera 1d closes...")
    asset_returns = {}
    for f in glob.glob(str(CHIMERA_1D / "*_v51_chimera_1d_*.parquet")):
        sym = Path(f).name.split("_")[0].upper().replace("USDT", "")
        try:
            df = pl.read_parquet(f, columns=["timestamp", "close"]).to_pandas()
        except Exception:
            continue
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.sort_values("date").reset_index(drop=True)
        df["ret_1d"] = df["close"].pct_change()
        df = df[(df["date"] >= pd.Timestamp(OOS_START)) & (df["date"] <= pd.Timestamp(OOS_END))]
        if len(df) > 0:
            asset_returns[sym] = df[["date", "ret_1d"]].set_index("date")["ret_1d"]
    print(f"  {len(asset_returns)} assets in OOS window")

    # Build per-day return matrix
    all_dates = sorted(set(d for s in asset_returns.values() for d in s.index))
    ret_matrix = pd.DataFrame({a: s.reindex(all_dates) for a, s in asset_returns.items()})
    print(f"  per-day matrix: {ret_matrix.shape}")

    # Compute per-day metrics
    per_day = []
    for d in all_dates:
        row = ret_matrix.loc[d].dropna()
        if len(row) < 10: continue

        # Sort returns descending
        sorted_rets = row.sort_values(ascending=False)
        positive_rets = sorted_rets[sorted_rets > 0]

        # Oracle K=5 LO: mean of top-5 returns (top-5 long picks)
        if len(sorted_rets) >= 5:
            oracle_top5_mean = float(sorted_rets.head(5).mean()) * 100
        else:
            oracle_top5_mean = float(sorted_rets.mean()) * 100

        # Top 25% of winners: among positive-return assets, top quartile mean
        if len(positive_rets) >= 4:
            top_quartile_n = max(1, int(np.ceil(len(positive_rets) * 0.25)))
            top_q_winners_mean = float(positive_rets.head(top_quartile_n).mean()) * 100
        elif len(positive_rets) >= 1:
            top_q_winners_mean = float(positive_rets.mean()) * 100
        else:
            top_q_winners_mean = 0.0

        # Universe stats
        n_positive = int((row > 0).sum())
        n_negative = int((row < 0).sum())
        n_assets = len(row)
        max_winner = float(sorted_rets.iloc[0]) * 100
        median_ret = float(row.median()) * 100

        per_day.append({
            "date": d.date(),
            "n_assets": n_assets,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "oracle_K5_mean_pct": oracle_top5_mean,
            "top_q_winners_mean_pct": top_q_winners_mean,
            "max_winner_pct": max_winner,
            "median_ret_pct": median_ret,
        })

    per_day_df = pd.DataFrame(per_day)
    print(f"  per-day records: {len(per_day_df)}")

    # Load strat daily NAV
    print("\nLoading strat daily NAV...")
    daily_df = pd.read_csv(DAILY_PATH)
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.date
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    daily_df["strat_ret_pct"] = daily_df["portfolio_value"].pct_change() * 100
    daily_df["strat_ret_pct"] = daily_df["strat_ret_pct"].fillna(0)
    print(f"  daily NAV records: {len(daily_df)}")

    # Merge
    merged = per_day_df.merge(daily_df[["date", "strat_ret_pct", "n_open", "deployed_pct"]],
                                on="date", how="left")
    merged["strat_ret_pct"] = merged["strat_ret_pct"].fillna(0)
    merged["n_open"] = merged["n_open"].fillna(0).astype(int)
    merged["deployed_pct"] = merged["deployed_pct"].fillna(0)

    # Capture ratio: strat_ret / oracle_K5 (clipped)
    merged["capture_ratio_pct"] = np.where(
        merged["oracle_K5_mean_pct"] > 0.1,
        (merged["strat_ret_pct"] / merged["oracle_K5_mean_pct"]) * 100,
        np.nan
    )
    merged["capture_ratio_pct"] = merged["capture_ratio_pct"].clip(-200, 200)

    # Capture vs top-25% winners
    merged["capture_vs_topq_pct"] = np.where(
        merged["top_q_winners_mean_pct"] > 0.1,
        (merged["strat_ret_pct"] / merged["top_q_winners_mean_pct"]) * 100,
        np.nan
    )
    merged["capture_vs_topq_pct"] = merged["capture_vs_topq_pct"].clip(-200, 200)

    # Day classification (against oracle availability)
    def classify_day(row):
        oracle = row["oracle_K5_mean_pct"]
        strat = row["strat_ret_pct"]
        fired = row["n_open"] > 0
        if oracle < 0.5:
            if fired and strat < -0.5: return "DEAD_LOSER"
            if fired: return "DEAD_NEUTRAL"
            return "DEAD_CASH"  # nothing to capture, no fires
        # Oracle has something
        if not fired:
            if oracle >= 3.0: return "MISS_BIG"
            return "MISS_MODEST"
        # We fired
        capture = strat / oracle if oracle > 0 else 0
        if capture >= 0.30: return "CAPTURE_HIT"
        if capture >= 0.10: return "CAPTURE_PARTIAL"
        if capture > 0: return "CAPTURE_LEAK"
        return "ADVERSE"
    merged["day_class"] = merged.apply(classify_day, axis=1)

    # Aggregate stats
    print(f"\n=== OOS DIAGNOSTIC SUMMARY ===")
    n_days = len(merged)
    print(f"  Days analyzed: {n_days}")
    print(f"\n  Oracle (K=5 LO mean per-day):")
    print(f"    mean:   {merged['oracle_K5_mean_pct'].mean():+.3f}%")
    print(f"    median: {merged['oracle_K5_mean_pct'].median():+.3f}%")
    print(f"    days oracle >= +1%:  {(merged['oracle_K5_mean_pct'] >= 1.0).sum()} "
          f"({(merged['oracle_K5_mean_pct'] >= 1.0).mean()*100:.1f}%)")
    print(f"    days oracle >= +3%:  {(merged['oracle_K5_mean_pct'] >= 3.0).sum()} "
          f"({(merged['oracle_K5_mean_pct'] >= 3.0).mean()*100:.1f}%)")
    print(f"    days oracle >= +5%:  {(merged['oracle_K5_mean_pct'] >= 5.0).sum()} "
          f"({(merged['oracle_K5_mean_pct'] >= 5.0).mean()*100:.1f}%)")
    print(f"    sum oracle (sum of K=5 means across all days): {merged['oracle_K5_mean_pct'].sum():+.2f}%")

    print(f"\n  Top-25% winners (per-day avg gain):")
    print(f"    mean:   {merged['top_q_winners_mean_pct'].mean():+.3f}%")
    print(f"    median: {merged['top_q_winners_mean_pct'].median():+.3f}%")
    print(f"    sum:    {merged['top_q_winners_mean_pct'].sum():+.2f}%")

    print(f"\n  Strat per-day NAV change:")
    print(f"    mean:   {merged['strat_ret_pct'].mean():+.4f}%")
    print(f"    median: {merged['strat_ret_pct'].median():+.4f}%")
    print(f"    days strat positive: {(merged['strat_ret_pct'] > 0).sum()} "
          f"({(merged['strat_ret_pct'] > 0).mean()*100:.1f}%)")
    print(f"    sum strat per-day:   {merged['strat_ret_pct'].sum():+.2f}%")

    print(f"\n  Capture ratio (strat / oracle):")
    cap = merged["capture_ratio_pct"].dropna()
    print(f"    mean:    {cap.mean():+.2f}%")
    print(f"    median:  {cap.median():+.2f}%")
    print(f"    25-pct:  {cap.quantile(0.25):+.2f}%")
    print(f"    75-pct:  {cap.quantile(0.75):+.2f}%")
    print(f"    days capture >= 30%: {(cap >= 30).sum()} ({(cap >= 30).mean()*100:.1f}%)")
    print(f"    days capture < 5%:   {(cap < 5).sum()} ({(cap < 5).mean()*100:.1f}%)")

    print(f"\n  Day classification:")
    for cl in ["CAPTURE_HIT", "CAPTURE_PARTIAL", "CAPTURE_LEAK", "MISS_BIG", "MISS_MODEST",
               "ADVERSE", "DEAD_LOSER", "DEAD_NEUTRAL", "DEAD_CASH"]:
        n = (merged["day_class"] == cl).sum()
        pct = n / n_days * 100
        print(f"    {cl:<20} {n:4d} ({pct:5.1f}%)")

    # Miss analysis — MISS_BIG days (oracle big, strat not fired)
    miss_big = merged[merged["day_class"] == "MISS_BIG"].copy()
    print(f"\n=== MISS_BIG analysis ({len(miss_big)} days) ===")
    if len(miss_big):
        print(f"  Avg oracle on miss-big days:      {miss_big['oracle_K5_mean_pct'].mean():+.2f}%")
        print(f"  Avg top-Q winners on miss days:   {miss_big['top_q_winners_mean_pct'].mean():+.2f}%")
        print(f"  Total oracle missed:              {miss_big['oracle_K5_mean_pct'].sum():+.2f}%")

    # CAPTURE_LEAK days (oracle big, strat fired but tiny)
    leak = merged[merged["day_class"] == "CAPTURE_LEAK"].copy()
    print(f"\n=== CAPTURE_LEAK analysis ({len(leak)} days) ===")
    if len(leak):
        print(f"  Avg oracle on leak days:          {leak['oracle_K5_mean_pct'].mean():+.2f}%")
        print(f"  Avg strat on leak days:           {leak['strat_ret_pct'].mean():+.4f}%")
        print(f"  Avg n_open on leak days:          {leak['n_open'].mean():.1f}")
        print(f"  Avg deployed % on leak days:      {leak['deployed_pct'].mean():.1f}%")

    # CAPTURE_HIT days
    hit = merged[merged["day_class"] == "CAPTURE_HIT"].copy()
    print(f"\n=== CAPTURE_HIT analysis ({len(hit)} days) ===")
    if len(hit):
        print(f"  Avg oracle:    {hit['oracle_K5_mean_pct'].mean():+.2f}%")
        print(f"  Avg strat:     {hit['strat_ret_pct'].mean():+.4f}%")
        print(f"  Avg n_open:    {hit['n_open'].mean():.1f}")
        print(f"  Avg deployed:  {hit['deployed_pct'].mean():.1f}%")

    # ADVERSE days (fired but lost)
    adv = merged[merged["day_class"] == "ADVERSE"].copy()
    print(f"\n=== ADVERSE_FIRE analysis ({len(adv)} days) ===")
    if len(adv):
        print(f"  Avg oracle on adverse days:       {adv['oracle_K5_mean_pct'].mean():+.2f}%")
        print(f"  Avg strat on adverse days:        {adv['strat_ret_pct'].mean():+.4f}%")
        print(f"  Avg deployed on adverse:          {adv['deployed_pct'].mean():.1f}%")

    # Top 20 oracle days — what did strat do?
    print(f"\n=== TOP 20 ORACLE DAYS — strat behavior ===")
    top_oracle = merged.sort_values("oracle_K5_mean_pct", ascending=False).head(20)
    print(f"  {'date':<12}{'oracle%':>9}{'top_q%':>9}{'strat%':>9}{'capture%':>10}{'n_open':>7}{'class':>20}")
    for _, r in top_oracle.iterrows():
        cap_s = f"{r['capture_ratio_pct']:+.1f}%" if pd.notna(r["capture_ratio_pct"]) else "n/a"
        print(f"  {str(r['date']):<12}{r['oracle_K5_mean_pct']:>+8.2f}%{r['top_q_winners_mean_pct']:>+8.2f}%"
              f"{r['strat_ret_pct']:>+8.3f}%{cap_s:>10}{r['n_open']:>7d}{r['day_class']:>20}")

    # Save CSV
    merged.to_csv(OUT_CSV, index=False)
    print(f"\n[OK] wrote {OUT_CSV}")

    # Write markdown
    lines = [
        "# UNION OOS Per-Day Diagnostic (2026-05-20)\n",
        f"**Window**: {OOS_START} → {OOS_END} ({len(merged)} days)",
        "",
        "## A. Universe-wide oracle availability (per-day)",
        "",
        f"- Oracle K=5 LO daily mean: {merged['oracle_K5_mean_pct'].mean():+.3f}%",
        f"- Oracle K=5 LO daily median: {merged['oracle_K5_mean_pct'].median():+.3f}%",
        f"- Sum of oracle K=5 across all OOS days: {merged['oracle_K5_mean_pct'].sum():+.2f}%",
        f"- Days oracle ≥ +1%: {(merged['oracle_K5_mean_pct'] >= 1.0).sum()} ({(merged['oracle_K5_mean_pct'] >= 1.0).mean()*100:.1f}%)",
        f"- Days oracle ≥ +3%: {(merged['oracle_K5_mean_pct'] >= 3.0).sum()} ({(merged['oracle_K5_mean_pct'] >= 3.0).mean()*100:.1f}%)",
        f"- Days oracle ≥ +5%: {(merged['oracle_K5_mean_pct'] >= 5.0).sum()} ({(merged['oracle_K5_mean_pct'] >= 5.0).mean()*100:.1f}%)",
        "",
        "## B. Top-25% of winners (per-day avg gain of the winning quartile)",
        "",
        f"- Mean: {merged['top_q_winners_mean_pct'].mean():+.3f}%/day",
        f"- Median: {merged['top_q_winners_mean_pct'].median():+.3f}%/day",
        f"- Sum across OOS: {merged['top_q_winners_mean_pct'].sum():+.2f}%",
        "",
        "## C. Strat per-day performance",
        "",
        f"- Mean daily NAV change: {merged['strat_ret_pct'].mean():+.4f}%",
        f"- Median: {merged['strat_ret_pct'].median():+.4f}%",
        f"- Positive days: {(merged['strat_ret_pct'] > 0).sum()} ({(merged['strat_ret_pct'] > 0).mean()*100:.1f}%)",
        f"- Cumulative (arithmetic sum): {merged['strat_ret_pct'].sum():+.2f}%",
        "",
        "## D. Capture ratio (strat / oracle)",
        "",
        f"- Mean: {cap.mean():+.2f}%",
        f"- Median: {cap.median():+.2f}%",
        f"- Days capture ≥ 30%: {(cap >= 30).sum()} ({(cap >= 30).mean()*100:.1f}%)",
        f"- Days capture < 5%: {(cap < 5).sum()} ({(cap < 5).mean()*100:.1f}%)",
        "",
        "## E. Day classification",
        "",
        "| Class | Days | % | Meaning |",
        "|---|---:|---:|---|",
    ]
    class_meanings = {
        "CAPTURE_HIT": "strat captured ≥ 30% of oracle (good)",
        "CAPTURE_PARTIAL": "strat captured 10-30% of oracle (decent)",
        "CAPTURE_LEAK": "strat fired but captured < 10% of oracle (bled out)",
        "MISS_BIG": "oracle ≥ +3% but strat did not fire (selection fail)",
        "MISS_MODEST": "oracle 0.5-3% and strat did not fire (smaller miss)",
        "ADVERSE": "strat fired and lost despite positive oracle",
        "DEAD_LOSER": "no oracle but strat lost",
        "DEAD_NEUTRAL": "no oracle and strat flat (correct cash-ish)",
        "DEAD_CASH": "no oracle and strat in cash (correct)",
    }
    for cl, meaning in class_meanings.items():
        n = (merged["day_class"] == cl).sum()
        pct = n / n_days * 100
        lines.append(f"| {cl} | {n} | {pct:.1f}% | {meaning} |")
    lines += [
        "",
        f"## F. MISS_BIG detail ({len(miss_big)} days — oracle ≥ +3% but strat didn't fire)",
        "",
    ]
    if len(miss_big):
        lines.append(f"- Avg oracle on miss-big days: **{miss_big['oracle_K5_mean_pct'].mean():+.2f}%**")
        lines.append(f"- Total oracle PnL forgone: **{miss_big['oracle_K5_mean_pct'].sum():+.2f}%**")
        lines.append("")
        lines.append("### Top-10 MISS_BIG days (largest forgone)")
        lines.append("")
        lines.append("| date | oracle K5 % | top-Q winners % | n_open | n_positive | n_negative |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for _, r in miss_big.sort_values("oracle_K5_mean_pct", ascending=False).head(10).iterrows():
            lines.append(f"| {r['date']} | {r['oracle_K5_mean_pct']:+.2f} | {r['top_q_winners_mean_pct']:+.2f} | "
                         f"{int(r['n_open'])} | {int(r['n_positive'])} | {int(r['n_negative'])} |")

    lines += [
        "",
        f"## G. CAPTURE_LEAK detail ({len(leak)} days — strat fired but captured <10% of oracle)",
        "",
    ]
    if len(leak):
        lines.append(f"- Avg oracle on leak days: {leak['oracle_K5_mean_pct'].mean():+.2f}%")
        lines.append(f"- Avg strat: {leak['strat_ret_pct'].mean():+.4f}%")
        lines.append(f"- Avg n_open: {leak['n_open'].mean():.1f}")
        lines.append(f"- Avg deployed %: {leak['deployed_pct'].mean():.1f}%")

    lines += [
        "",
        "## H. TOP 20 oracle days — what strat did",
        "",
        "| date | oracle K5 % | top-Q% | strat % | capture % | n_open | class |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in top_oracle.iterrows():
        cap_s = f"{r['capture_ratio_pct']:+.1f}%" if pd.notna(r["capture_ratio_pct"]) else "n/a"
        lines.append(f"| {r['date']} | {r['oracle_K5_mean_pct']:+.2f} | "
                     f"{r['top_q_winners_mean_pct']:+.2f} | {r['strat_ret_pct']:+.3f} | "
                     f"{cap_s} | {int(r['n_open'])} | {r['day_class']} |")

    # Sum check
    lines += [
        "",
        "## I. Honest sum check",
        "",
        f"- Sum of oracle K=5 means across OOS: **{merged['oracle_K5_mean_pct'].sum():+.2f}%**",
        f"- Sum of top-25% winners across OOS: **{merged['top_q_winners_mean_pct'].sum():+.2f}%**",
        f"- Sum of strat per-day NAV changes (arithmetic): **{merged['strat_ret_pct'].sum():+.2f}%**",
        f"- Strat compound 8Q OOS: **{((daily_df['portfolio_value'].iloc[-1] / daily_df['portfolio_value'].iloc[0] - 1) * 100):+.2f}%**",
        "",
        f"- Strat / oracle capture ratio (arithmetic basis): "
        f"**{(merged['strat_ret_pct'].sum() / merged['oracle_K5_mean_pct'].sum() * 100):.1f}%**",
        f"- Strat / top-Q winners capture ratio (arithmetic basis): "
        f"**{(merged['strat_ret_pct'].sum() / merged['top_q_winners_mean_pct'].sum() * 100):.1f}%**",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
