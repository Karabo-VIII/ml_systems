"""Bear/chop oracle analysis + why-missed framework per regime.

Per user directive (2026-05-20):
  "Solve for the bear/chop market in the sense that if we solve for the
   worst or trickiest conditions, we will win when the bull cycle comes.
   Use research, check oracle numbers: look at those days that top
   performers in a day rally in bear."
  + invoke the why-missed framework for bull/bear/chop and close gaps.

Analysis structure:
  1. Per-regime oracle ceiling (TOP1, TOP5, TOP10 forward 14d returns per day)
  2. Per-regime ensemble firing rate (does the bull-tuned specialist set fire
     at all in bear/chop? what % of regime days have firings?)
  3. Per-regime CAPTURE: realized NAV / oracle NAV
  4. RALLY DAYS IN BEAR/CHOP: days where some asset moves >=5% in forward 14d
     -> did our ensemble catch ANY asset that day? If not, why?
  5. WHY-MISSED CLASSIFIER per regime:
        - CAPTURE_HIT:  ensemble fired + caught a top mover
        - PARTIAL:      fired but mid-tier asset (< top-5 mover)
        - MISS:         didn't fire on the day's top mover
        - DEAD_DAY:     no top movers existed
  6. BEAR/CHOP SPECIALIST CANDIDATES:
        - For missed rallies in bear/chop, which indicator-config DID
          fire on that asset that day (even if not in current specialist
          set)? -> candidates for bear/chop specialist expansion

Outputs in runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/:
  per_regime_oracle_capture.csv
  bear_chop_rally_classification.csv
  bear_chop_specialist_candidates.csv
  BEAR_CHOP_SOLVE_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2023, 7, 1)
COST = 0.0024
BET_FRACTION = 0.08
K_MAX = 8
ASYM_STOP, ASYM_TARGET = -0.04, 0.12

def asymmetric_stop_returns(rets, stop=ASYM_STOP, target=ASYM_TARGET):
    out = np.copy(rets)
    out = np.where(out <= stop, stop, out)
    out = np.where(out >= target, target, out)
    return out

def load_oracle_panel():
    print("Loading oracle panel...")
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
        if len(df) < 30: continue
        df["asset"] = sym
        df["close_fwd14"] = df["close"].shift(-14)
        df["ret_fwd14"] = (df["close_fwd14"] / df["close"] - 1) - COST
        rows.append(df[["asset", "date", "ret_fwd14"]].dropna())
    return pd.concat(rows, ignore_index=True)

def main():
    print("="*78)
    print("BEAR/CHOP ORACLE + WHY-MISSED FRAMEWORK")
    print("="*78)
    events = pd.read_parquet(OUT_DIR / "per_event_enriched.parquet")
    print(f"Loaded {len(events):,} events")

    panel = load_oracle_panel()
    print(f"Oracle panel: {len(panel):,} rows / {panel['asset'].nunique()} assets")

    # 1) Per-regime oracle ceiling
    # First we need per-day regime + per-day TOP-K stats
    # Join regime to oracle panel by date
    date_regime = events[["date", "btc_regime_30d"]].drop_duplicates(subset="date").copy()
    date_regime["date"] = pd.to_datetime(date_regime["date"]).dt.date
    panel = panel.merge(date_regime, on="date", how="left")
    panel["btc_regime_30d"] = panel["btc_regime_30d"].fillna("UNK")

    # Per-day top-K
    daily = panel.groupby(["date", "btc_regime_30d"])["ret_fwd14"].agg([
        ("top1_ret", "max"),
        ("top5_mean", lambda x: x.nlargest(min(5, len(x))).mean()),
        ("top5_sum", lambda x: x.nlargest(min(5, len(x))).sum()),
        ("n_active", "count"),
        ("n_movers_5pct", lambda x: (x >= 0.05).sum()),
        ("n_movers_10pct", lambda x: (x >= 0.10).sum()),
    ]).reset_index()
    print(f"\nDaily oracle: {len(daily)} days")

    # 2) Per-regime ceiling stats
    print("\n=== PER-REGIME ORACLE CEILING ===")
    print(f"{'regime':<8}{'days':<8}{'avg_top1':<12}{'avg_top5_sum':<14}{'pct_days_5pct':<16}{'pct_days_10pct':<14}")
    reg_stats = []
    for reg in ("bull", "chop", "bear", "crash"):
        sub = daily[daily["btc_regime_30d"] == reg]
        if len(sub) == 0: continue
        s = {
            "regime": reg,
            "n_days": int(len(sub)),
            "avg_top1_ret_pct": float(sub["top1_ret"].mean() * 100),
            "median_top1_ret_pct": float(sub["top1_ret"].median() * 100),
            "avg_top5_sum_pct_at4size": float(sub["top5_sum"].mean() * 0.04 * 100),
            "pct_days_with_5pct_mover": float((sub["n_movers_5pct"] >= 1).mean() * 100),
            "pct_days_with_10pct_mover": float((sub["n_movers_10pct"] >= 1).mean() * 100),
            "total_oracle_top5_nav_at4size_pct": float(sub["top5_sum"].sum() * 0.04 * 100),
        }
        reg_stats.append(s)
        print(f"{reg:<8}{s['n_days']:<8}{s['avg_top1_ret_pct']:+.2f}%      {s['avg_top5_sum_pct_at4size']:+.2f}%        {s['pct_days_with_5pct_mover']:>5.1f}%          {s['pct_days_with_10pct_mover']:>5.1f}%")
    reg_df = pd.DataFrame(reg_stats)
    reg_df.to_csv(OUT_DIR / "per_regime_oracle_capture.csv", index=False)

    # 3) Per-regime ensemble firings + capture
    print("\n=== PER-REGIME ENSEMBLE CAPTURE (current bull-tuned set) ===")
    # Use the 50-specialist set from S1
    df = pd.read_csv(OUT_DIR / "small_account_sizing_target12.csv")
    survivors = df[(df["asym_expectancy_pct"] > 1.0) & (df["asym_hit_pct"] >= 40) &
                     (df["asymmetry_ratio"] >= 2.0) & (df["n"] >= 100)]
    top_per_ind = survivors.sort_values(["indicator", "asym_small_account_score"],
                                          ascending=[True, False]).groupby("indicator").head(5)
    specialists = set(zip(top_per_ind["indicator"], top_per_ind["config"], top_per_ind["regime"]))

    spec_filter = events.set_index(["indicator", "config", "btc_regime_30d"]).index.isin(specialists)
    spec_events = events[spec_filter].copy()
    print(f"Bull-tuned specialist firings across regimes:")
    reg_fires = spec_events["btc_regime_30d"].value_counts()
    print(reg_fires)

    # Capture by regime: for each regime, what is the ensemble's nav vs the
    # oracle's nav?
    print("\n=== PER-REGIME CAPTURE ===")
    print(f"{'regime':<8}{'ens_fires':<12}{'ens_NAV':<12}{'oracle_NAV':<14}{'capture%':<10}")
    cap_rows = []
    for reg in ("bull", "chop", "bear", "crash"):
        ens = spec_events[spec_events["btc_regime_30d"] == reg]
        if len(ens) == 0:
            ens_nav = 0
        else:
            asym = asymmetric_stop_returns(ens["ret_E_14d"].fillna(0).values)
            ens_nav = asym.sum() * BET_FRACTION * 100
        oracle_nav = reg_df.loc[reg_df["regime"] == reg, "total_oracle_top5_nav_at4size_pct"].iloc[0]
        cap_pct = (ens_nav / oracle_nav * 100) if oracle_nav > 0 else 0
        cap_rows.append({"regime": reg, "ens_fires": int(len(ens)),
                          "ens_NAV_pct": ens_nav, "oracle_NAV_pct": oracle_nav,
                          "capture_pct": cap_pct})
        print(f"{reg:<8}{len(ens):<12}{ens_nav:+8.2f}%   {oracle_nav:+8.2f}%      {cap_pct:>6.2f}%")
    pd.DataFrame(cap_rows).to_csv(OUT_DIR / "per_regime_ensemble_capture.csv", index=False)

    # 4) BEAR/CHOP RALLY DAYS
    print("\n=== BEAR/CHOP RALLY DAYS (>=10% movers exist in bear/chop) ===")
    bc_rally = daily[(daily["btc_regime_30d"].isin(["bear", "chop"])) & (daily["n_movers_10pct"] >= 1)]
    print(f"Bear+Chop days with >=1 10%-mover: {len(bc_rally)} of {len(daily[daily['btc_regime_30d'].isin(['bear','chop'])])}")
    print(f"  bear: {(bc_rally['btc_regime_30d']=='bear').sum()}")
    print(f"  chop: {(bc_rally['btc_regime_30d']=='chop').sum()}")
    bc_rally.to_csv(OUT_DIR / "bear_chop_rally_days.csv", index=False)

    # For these rally days, classify outcome
    print("\n=== RALLY-DAY OUTCOME CLASSIFICATION (bear+chop) ===")
    # For each (rally day, top-mover asset), did the bull-tuned ensemble fire on this asset?
    panel_top_mover = panel.copy()
    panel_top_mover["date"] = pd.to_datetime(panel_top_mover["date"]).dt.date
    # Identify top mover per day
    panel_top_mover_max = panel_top_mover.loc[panel_top_mover.groupby("date")["ret_fwd14"].idxmax()]
    panel_top_mover_max = panel_top_mover_max.rename(columns={"asset": "top_mover_asset",
                                                                "ret_fwd14": "top_mover_ret"})

    bc_rally_assets = bc_rally.merge(
        panel_top_mover_max[["date", "top_mover_asset", "top_mover_ret"]],
        on="date", how="left",
    )

    # Did ensemble fire on top_mover_asset on rally_date?
    ens_dates_assets = set(zip(spec_events["asset"], pd.to_datetime(spec_events["date"]).dt.date))
    bc_rally_assets["ens_fired_on_top_mover"] = bc_rally_assets.apply(
        lambda r: (r["top_mover_asset"], r["date"]) in ens_dates_assets, axis=1)

    n_caught = bc_rally_assets["ens_fired_on_top_mover"].sum()
    n_total = len(bc_rally_assets)
    print(f"Bear/chop rally days where ensemble fired on top mover: {n_caught}/{n_total} ({n_caught*100/max(n_total,1):.1f}%)")

    # Which OTHER indicator-configs fired on the missed top mover?
    # i.e. bear/chop rally + ensemble missed -> what DID fire?
    missed = bc_rally_assets[~bc_rally_assets["ens_fired_on_top_mover"]].copy()
    missed_assets_dates = set(zip(missed["top_mover_asset"], missed["date"]))
    candidate_fires = events[events.apply(
        lambda r: (r["asset"], r["date"]) in missed_assets_dates, axis=1)]
    candidate_summary = candidate_fires.groupby(["indicator", "config", "btc_regime_30d"]).agg(
        n_fires=("date", "size"),
        mean_ret=("ret_E_14d", "mean"),
        hit=("ret_E_14d", lambda s: (s > 0).mean() * 100),
    ).reset_index().sort_values("mean_ret", ascending=False)
    print(f"\nCandidate indicators/configs that fired on missed bear/chop rallies:")
    print(candidate_summary.head(15).to_string(index=False))
    candidate_summary.to_csv(OUT_DIR / "bear_chop_specialist_candidates.csv", index=False)

    # 5) WHY-MISSED CLASSIFIER (per regime, why is opportunity missed?)
    print("\n=== WHY-MISSED CLASSIFIER (per regime) ===")
    classifications = []
    for reg in ("bull", "chop", "bear", "crash"):
        sub_panel = panel[panel["btc_regime_30d"] == reg].copy()
        sub_panel["date"] = pd.to_datetime(sub_panel["date"]).dt.date
        sub_ens_dates = set(zip(spec_events[spec_events["btc_regime_30d"]==reg]["asset"],
                                  pd.to_datetime(spec_events[spec_events["btc_regime_30d"]==reg]["date"]).dt.date))
        sub_events_dates_assets = set(zip(events[events["btc_regime_30d"]==reg]["asset"],
                                            pd.to_datetime(events[events["btc_regime_30d"]==reg]["date"]).dt.date))
        # Identify top mover per day
        top_per_day = sub_panel.loc[sub_panel.groupby("date")["ret_fwd14"].idxmax()]
        for _, r in top_per_day.iterrows():
            key = (r["asset"], r["date"])
            top_ret = r["ret_fwd14"]
            if top_ret < 0.05:
                cls = "DEAD_DAY"
            elif key in sub_ens_dates:
                cls = "CAPTURE_HIT"
            elif key in sub_events_dates_assets:
                cls = "MISS_BUT_OTHER_INDICATOR_FIRED"  # Something fired, just not specialist
            else:
                cls = "MISS_NO_FIRE"
            classifications.append({"date": r["date"], "regime": reg,
                                      "top_mover_asset": r["asset"],
                                      "top_mover_ret_pct": top_ret * 100,
                                      "class": cls})
    cls_df = pd.DataFrame(classifications)
    print("\nPer-regime classification:")
    summary = cls_df.groupby(["regime", "class"]).size().unstack(fill_value=0)
    summary["TOTAL"] = summary.sum(axis=1)
    for c in summary.columns[:-1]:
        summary[f"{c}_pct"] = (summary[c] / summary["TOTAL"] * 100).round(1)
    print(summary)
    cls_df.to_csv(OUT_DIR / "why_missed_per_regime.csv", index=False)

    # 6) Write report
    lines = ["# Bear/Chop Solve + Why-Missed Framework — TRAIN\n"]
    lines.append("\n## A) Per-regime oracle ceiling\n")
    lines.append("| regime | days | avg top-1 | avg top-5 NAV@4% | days with 5%-mover | days with 10%-mover |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for s in reg_stats:
        lines.append(f"| {s['regime']} | {s['n_days']} | {s['avg_top1_ret_pct']:+.2f}% | {s['avg_top5_sum_pct_at4size']:+.2f}% | {s['pct_days_with_5pct_mover']:.1f}% | {s['pct_days_with_10pct_mover']:.1f}% |")

    lines.append("\n## B) Current ensemble (bull-tuned) capture per regime\n")
    lines.append("| regime | ens fires | ens NAV | oracle NAV | capture % |")
    lines.append("|---|--:|--:|--:|--:|")
    for r in cap_rows:
        lines.append(f"| {r['regime']} | {r['ens_fires']} | {r['ens_NAV_pct']:+.2f}% | {r['oracle_NAV_pct']:+.2f}% | {r['capture_pct']:.2f}% |")

    lines.append("\n## C) Bear/chop rally-day capture\n")
    lines.append(f"- Bear+Chop days with >=1 10%-mover: **{len(bc_rally)}**")
    lines.append(f"- Of those, ensemble fired on top mover: **{n_caught}/{n_total} ({n_caught*100/max(n_total,1):.1f}%)**")
    lines.append(f"- This is the gap to close: when bear/chop has a 10%+ mover, our current ensemble misses it ~{100-n_caught*100/max(n_total,1):.0f}% of the time.")

    lines.append("\n## D) Bear/chop specialist candidates (what DID fire on missed rallies)\n")
    lines.append("Top 15 indicator-configs that fired on missed bear/chop rally days:")
    lines.append("| indicator | config | regime | n_fires | mean_ret | hit % |")
    lines.append("|---|---|---|--:|--:|--:|")
    for _, r in candidate_summary.head(15).iterrows():
        lines.append(f"| {r['indicator']} | `{r['config']}` | {r['btc_regime_30d']} | {int(r['n_fires'])} | {r['mean_ret']*100:+.2f}% | {r['hit']:.1f}% |")

    lines.append("\n## E) Why-missed classification per regime\n")
    lines.append("```")
    lines.append(summary.to_string())
    lines.append("```")
    lines.append("\nLegend:")
    lines.append("- CAPTURE_HIT: ensemble fired on top mover -> we got it")
    lines.append("- MISS_BUT_OTHER_INDICATOR_FIRED: some indicator fired on the top mover, just not a specialist (candidate for expansion)")
    lines.append("- MISS_NO_FIRE: no indicator fired on the top mover (need NEW indicator family)")
    lines.append("- DEAD_DAY: no asset moved >=5% that day (no opportunity)")

    # Headline interpretation
    lines.append("\n## F) Headline interpretation\n")
    bull_hit = summary.loc["bull", "CAPTURE_HIT"] if "bull" in summary.index else 0
    bull_total = summary.loc["bull", "TOTAL"] if "bull" in summary.index else 0
    bear_hit = summary.loc["bear", "CAPTURE_HIT"] if "bear" in summary.index else 0
    bear_total = summary.loc["bear", "TOTAL"] if "bear" in summary.index else 0
    chop_hit = summary.loc["chop", "CAPTURE_HIT"] if "chop" in summary.index else 0
    chop_total = summary.loc["chop", "TOTAL"] if "chop" in summary.index else 0
    crash_hit = summary.loc["crash", "CAPTURE_HIT"] if "crash" in summary.index else 0
    crash_total = summary.loc["crash", "TOTAL"] if "crash" in summary.index else 0
    lines.append(f"- BULL capture: {bull_hit}/{bull_total} ({bull_hit*100/max(bull_total,1):.1f}%)")
    lines.append(f"- CHOP capture: {chop_hit}/{chop_total} ({chop_hit*100/max(chop_total,1):.1f}%)")
    lines.append(f"- BEAR capture: {bear_hit}/{bear_total} ({bear_hit*100/max(bear_total,1):.1f}%)")
    lines.append(f"- CRASH capture: {crash_hit}/{crash_total} ({crash_hit*100/max(crash_total,1):.1f}%)")
    lines.append("\nGap-close priorities (high-yield):")
    lines.append("1. CHOP days with MISS_BUT_OTHER_INDICATOR_FIRED -> EXPAND specialist set to include chop-tuned variants of these")
    lines.append("2. BEAR days with MISS_BUT_OTHER_INDICATOR_FIRED -> add bear-rally specialists (counter-trend longs in oversold)")
    lines.append("3. Days with MISS_NO_FIRE -> new indicator family needed (or different feature, e.g. dollar-vol-spike)")

    (OUT_DIR / "BEAR_CHOP_SOLVE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'BEAR_CHOP_SOLVE_REPORT.md'}")

if __name__ == "__main__":
    main()
