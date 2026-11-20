"""Per-setup miss attribution + within/cross-indicator complementarity.

Per user mandate (2026-05-20): "what is being missed by each strategy when
it misses moves?" + "are there co-linear strats that can cover each other's
weaknesses: sum > individuals?"

Analysis structure:

PART A — PER-SETUP MISS PROFILE:
  For each top-N stable setup, enumerate top-mover days (any asset moves
  >=5% fwd 14d). Classify each:
    HIT:       setup fired on that asset that day
    PARTIAL:   setup fired on some asset that day, just not the top mover
    MISS_NO:   setup didn't fire at all that day
  Group misses by: regime, dna_bucket, sector, magnitude bucket, vol_regime.
  This tells us WHAT each setup misses (the structural blind spots).

PART B — WITHIN-INDICATOR COMPLEMENTARITY:
  For each indicator family (SMA, RSI, etc.), build the (asset, date) firing
  matrix per config. Compute pairwise Jaccard similarity:
    Jaccard(A,B) = |A intersect B| / |A union B|
  High Jaccard = redundant configs (overfitting risk if both selected).
  Low Jaccard = complementary configs (gap closure: sum > individuals).

PART C — CROSS-INDICATOR COMPLEMENTARITY (gap closure):
  For each pair (setup_A, setup_B), compute:
    - cover_A:  fraction of top-mover days A catches
    - cover_B:  fraction of top-mover days B catches
    - cover_AB: fraction caught by A OR B
    - lift:     cover_AB - max(cover_A, cover_B)  (marginal gain from union)
  High lift = B closes A's gap (and vice versa).

PART D — GREEDY OPTIMAL PORTFOLIO:
  Start with best-coverage single setup. Iteratively add the setup that
  maximizes MARGINAL COVERAGE on UNCAUGHT top-mover days. Stop when
  marginal lift < 1pp.

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/miss_profile_per_setup.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/within_indicator_jaccard.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/cross_indicator_lift.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/greedy_portfolio.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/MISS_ATTRIBUTION_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

TOP_N_STABLE = 50          # focus analysis on top-50 VAL-stable setups
MOVER_THRESH = 0.05        # >=5% fwd 14d = "top mover" we should catch
GREEDY_MAX_SETUPS = 20     # cap greedy portfolio at 20 picks

def main():
    print("="*78)
    print("MISS ATTRIBUTION + COMPLEMENTARITY ANALYSIS")
    print("="*78)
    events = pd.read_parquet(OUT_DIR / "per_event_enriched.parquet")
    events["date"] = pd.to_datetime(events["date"]).dt.date
    print(f"Events loaded: {len(events):,}")

    val_confirm = pd.read_csv(OUT_DIR / "val_confirmation_v2.csv")
    stable = val_confirm[val_confirm["stable"]].sort_values(
        ["val_qualifies_n_regimes", "val_n"], ascending=[False, False])
    print(f"VAL-stable setups: {len(stable)}")

    # Use top-N stable for compute tractability
    top_stable = stable.head(TOP_N_STABLE).reset_index(drop=True)
    print(f"Analysing top {TOP_N_STABLE} VAL-stable setups")

    # Define "top mover days": for each (asset, date), is the forward 14d return >=5%?
    # Use ret_E_14d from events (any setup that observed this asset on that day records the ret)
    asset_day_ret = events.groupby(["asset", "date"])["ret_E_14d"].max().reset_index()
    asset_day_ret = asset_day_ret.dropna(subset=["ret_E_14d"])
    movers = asset_day_ret[asset_day_ret["ret_E_14d"] >= MOVER_THRESH].copy()
    movers_set = set(zip(movers["asset"], movers["date"]))
    print(f"Top-mover (asset, date) pairs: {len(movers_set):,}")
    print(f"  Mean mover return: {movers['ret_E_14d'].mean()*100:+.2f}%")
    print(f"  Top deciles: {np.quantile(movers['ret_E_14d'], [0.5, 0.75, 0.9, 0.99]) * 100}")

    # Build per-setup firing set (asset, date) tuples
    print("\nBuilding per-setup firing sets...")
    setup_firings = {}
    for _, row in top_stable.iterrows():
        key = (row["indicator"], row["config"])
        sub = events[(events["indicator"] == row["indicator"]) &
                       (events["config"] == row["config"])]
        setup_firings[key] = set(zip(sub["asset"], sub["date"]))
    print(f"  Setups indexed: {len(setup_firings)}")

    # =========================================================================
    # PART A: per-setup miss profile (per regime, bucket, sector, magnitude)
    # =========================================================================
    print("\n=== PART A: PER-SETUP MISS PROFILE ===")
    # Pull regime/bucket/sector per (asset, date) from events
    meta = events.groupby(["asset", "date"]).agg(
        btc_regime_30d=("btc_regime_30d", "first"),
        dna_bucket=("dna_bucket", "first"),
        sector=("sector", "first"),
        rv30=("rv30_at_entry", "first"),
    ).reset_index()
    meta_key = meta.set_index(["asset", "date"]).to_dict("index")

    # Pre-build (asset, date) -> ret_E_14d lookup ONCE (was rebuilding index every loop)
    ret_lookup = dict(zip(zip(asset_day_ret["asset"], asset_day_ret["date"]),
                            asset_day_ret["ret_E_14d"]))
    miss_rows = []
    for key, fires in setup_firings.items():
        ind, cfg = key
        hits = movers_set & fires       # caught the top mover
        misses = movers_set - fires      # didn't fire on the top mover
        # Classify misses by metadata
        miss_by_regime = defaultdict(int)
        miss_by_bucket = defaultdict(int)
        miss_by_sector = defaultdict(int)
        miss_by_magnitude = defaultdict(int)
        for ad in misses:
            m = meta_key.get(ad)
            if m is None: continue
            miss_by_regime[str(m["btc_regime_30d"])] += 1
            miss_by_bucket[str(m["dna_bucket"])] += 1
            miss_by_sector[str(m["sector"])] += 1
            ret = ret_lookup.get(ad)
            if ret is not None:
                if ret >= 0.20:
                    miss_by_magnitude["20pct+"] += 1
                elif ret >= 0.10:
                    miss_by_magnitude["10-20pct"] += 1
                else:
                    miss_by_magnitude["5-10pct"] += 1
        miss_rows.append({
            "indicator": ind, "config": cfg,
            "n_fires": len(fires),
            "n_hits": len(hits),
            "n_misses": len(misses),
            "coverage_pct": len(hits) * 100 / max(len(movers_set), 1),
            "miss_bull": miss_by_regime.get("bull", 0),
            "miss_chop": miss_by_regime.get("chop", 0),
            "miss_bear": miss_by_regime.get("bear", 0),
            "miss_crash": miss_by_regime.get("crash", 0),
            "miss_BLUE": miss_by_bucket.get("BLUE", 0),
            "miss_STEADY": miss_by_bucket.get("STEADY", 0),
            "miss_VOLATILE": miss_by_bucket.get("VOLATILE", 0),
            "miss_DEGEN": miss_by_bucket.get("DEGEN", 0),
            "miss_5_10pct": miss_by_magnitude.get("5-10pct", 0),
            "miss_10_20pct": miss_by_magnitude.get("10-20pct", 0),
            "miss_20pct_plus": miss_by_magnitude.get("20pct+", 0),
            "top_missed_sector": max(miss_by_sector.items(), key=lambda x: x[1])[0] if miss_by_sector else None,
        })
    miss_df = pd.DataFrame(miss_rows).sort_values("coverage_pct", ascending=False)
    miss_df.to_csv(OUT_DIR / "miss_profile_per_setup.csv", index=False)
    print("Top-10 setups by coverage (% of top movers caught):")
    print(miss_df.head(10)[["indicator", "config", "n_fires", "n_hits", "coverage_pct",
                              "miss_bull", "miss_chop", "miss_bear",
                              "miss_VOLATILE", "miss_DEGEN", "miss_20pct_plus"]].to_string(index=False))

    # =========================================================================
    # PART B: within-indicator Jaccard
    # =========================================================================
    print("\n=== PART B: WITHIN-INDICATOR COMPLEMENTARITY (Jaccard) ===")
    jaccard_rows = []
    by_indicator = defaultdict(list)
    for key in setup_firings.keys():
        by_indicator[key[0]].append(key)
    for ind, keys in by_indicator.items():
        if len(keys) < 2: continue
        for k1, k2 in combinations(keys, 2):
            s1, s2 = setup_firings[k1], setup_firings[k2]
            inter = len(s1 & s2)
            uni = len(s1 | s2)
            jaccard = inter / max(uni, 1)
            jaccard_rows.append({
                "indicator": ind,
                "config_A": k1[1], "config_B": k2[1],
                "n_A": len(s1), "n_B": len(s2),
                "intersection": inter, "union": uni,
                "jaccard": round(jaccard, 4),
                "redundancy_class": "HIGH" if jaccard > 0.7 else ("MED" if jaccard > 0.4 else "LOW_COMPLEMENTARY"),
            })
    jdf = pd.DataFrame(jaccard_rows).sort_values(["indicator", "jaccard"])
    jdf.to_csv(OUT_DIR / "within_indicator_jaccard.csv", index=False)
    print(f"Total within-indicator pairs: {len(jdf)}")
    if len(jdf):
        print(f"  HIGH redundancy (J>0.7):    {(jdf['jaccard']>0.7).sum()} pairs")
        print(f"  MED  redundancy (0.4<J<=0.7): {((jdf['jaccard']<=0.7)&(jdf['jaccard']>0.4)).sum()}")
        print(f"  LOW (complementary, J<=0.4): {(jdf['jaccard']<=0.4).sum()}")
        # Per-indicator most complementary pair
        print("\nMost-complementary within-indicator pair (lowest Jaccard):")
        for ind in jdf["indicator"].unique():
            sub = jdf[jdf["indicator"] == ind].sort_values("jaccard").head(1)
            if len(sub):
                r = sub.iloc[0]
                print(f"  {ind:<22} {r['config_A']} vs {r['config_B']}: J={r['jaccard']:.3f} ({r['redundancy_class']})")

    # =========================================================================
    # PART C: cross-indicator LIFT (does B close A's gap?)
    # =========================================================================
    print("\n=== PART C: CROSS-INDICATOR LIFT ===")
    lift_rows = []
    keys = list(setup_firings.keys())
    for kA, kB in combinations(keys, 2):
        sA, sB = setup_firings[kA], setup_firings[kB]
        cov_A = len(sA & movers_set) / max(len(movers_set), 1)
        cov_B = len(sB & movers_set) / max(len(movers_set), 1)
        cov_AB = len((sA | sB) & movers_set) / max(len(movers_set), 1)
        lift = cov_AB - max(cov_A, cov_B)
        if lift < 0.01:  # filter trivial pairs
            continue
        lift_rows.append({
            "A_ind": kA[0], "A_cfg": kA[1],
            "B_ind": kB[0], "B_cfg": kB[1],
            "cov_A_pct": cov_A * 100,
            "cov_B_pct": cov_B * 100,
            "cov_AB_pct": cov_AB * 100,
            "lift_pct": lift * 100,
            "same_family": kA[0] == kB[0],
        })
    ldf = pd.DataFrame(lift_rows).sort_values("lift_pct", ascending=False)
    ldf.to_csv(OUT_DIR / "cross_indicator_lift.csv", index=False)
    print(f"Total cross-pairs with >=1pp lift: {len(ldf)}")
    print(f"  Within-family lift pairs: {ldf['same_family'].sum()}")
    print(f"  Cross-family lift pairs:  {(~ldf['same_family']).sum()}")
    print("\nTop-10 cross-family lift pairs:")
    cross = ldf[~ldf["same_family"]].head(10)
    print(cross[["A_ind", "A_cfg", "B_ind", "B_cfg",
                 "cov_A_pct", "cov_B_pct", "cov_AB_pct", "lift_pct"]].to_string(index=False))

    # =========================================================================
    # PART D: greedy optimal portfolio
    # =========================================================================
    print("\n=== PART D: GREEDY OPTIMAL PORTFOLIO ===")
    remaining_movers = set(movers_set)
    portfolio = []
    portfolio_coverage = []
    for step in range(GREEDY_MAX_SETUPS):
        best_key = None; best_gain = 0
        for key, fires in setup_firings.items():
            if key in [p[0] for p in portfolio]: continue
            gain = len(fires & remaining_movers)
            if gain > best_gain:
                best_gain = gain; best_key = key
        if best_key is None or best_gain == 0:
            break
        portfolio.append((best_key, best_gain))
        remaining_movers -= setup_firings[best_key]
        cumulative_caught = len(movers_set) - len(remaining_movers)
        cum_pct = cumulative_caught * 100 / len(movers_set)
        portfolio_coverage.append({
            "step": step + 1, "indicator": best_key[0], "config": best_key[1],
            "marginal_gain_movers": best_gain,
            "marginal_gain_pct": best_gain * 100 / len(movers_set),
            "cumulative_coverage_pct": cum_pct,
        })
        if best_gain * 100 / len(movers_set) < 1.0:
            break
    pdf = pd.DataFrame(portfolio_coverage)
    pdf.to_csv(OUT_DIR / "greedy_portfolio.csv", index=False)
    print(f"\nGreedy portfolio built (stopping when marginal gain < 1pp):")
    print(pdf.to_string(index=False))
    print(f"\nFinal coverage: {pdf['cumulative_coverage_pct'].max():.1f}% of {len(movers_set)} top movers")
    print(f"Portfolio size: {len(pdf)} setups (vs library of {len(top_stable)})")
    print(f"Best single setup coverage: {pdf.iloc[0]['cumulative_coverage_pct']:.1f}%")
    print(f"Lift from N=1 to N={len(pdf)}: +{pdf.iloc[-1]['cumulative_coverage_pct'] - pdf.iloc[0]['cumulative_coverage_pct']:.1f}pp")

    # =========================================================================
    # REPORT
    # =========================================================================
    lines = ["# Miss Attribution + Complementarity Analysis\n"]
    lines.append(f"Scope: top-{TOP_N_STABLE} VAL-stable setups, top-movers = events with fwd_14d return >= {MOVER_THRESH*100:.0f}%")
    lines.append(f"Total top-mover (asset, date) pairs in TRAIN: **{len(movers_set):,}**")
    lines.append(f"  Mean mover return: {movers['ret_E_14d'].mean()*100:+.2f}%")

    lines.append("\n## A) Per-setup miss profile (top-10 by coverage)\n")
    lines.append("Misses by regime/bucket/magnitude reveal WHAT each setup blind-spots.")
    lines.append("| indicator | config | n_fires | n_hits | coverage % | miss bull | miss chop | miss bear | miss VOLATILE | miss DEGEN | miss 20%+ moves |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in miss_df.head(10).iterrows():
        lines.append(f"| {r['indicator']} | `{r['config']}` | {int(r['n_fires'])} | {int(r['n_hits'])} | {r['coverage_pct']:.1f}% | {int(r['miss_bull'])} | {int(r['miss_chop'])} | {int(r['miss_bear'])} | {int(r['miss_VOLATILE'])} | {int(r['miss_DEGEN'])} | {int(r['miss_20pct_plus'])} |")

    lines.append("\n### Patterns in misses\n")
    avg_miss_regime = miss_df[["miss_bull", "miss_chop", "miss_bear", "miss_crash"]].mean()
    lines.append(f"Avg misses per setup by regime: bull={avg_miss_regime['miss_bull']:.0f} chop={avg_miss_regime['miss_chop']:.0f} bear={avg_miss_regime['miss_bear']:.0f} crash={avg_miss_regime['miss_crash']:.0f}")
    avg_miss_bucket = miss_df[["miss_BLUE", "miss_STEADY", "miss_VOLATILE", "miss_DEGEN"]].mean()
    lines.append(f"Avg misses per setup by bucket: BLUE={avg_miss_bucket['miss_BLUE']:.0f} STEADY={avg_miss_bucket['miss_STEADY']:.0f} VOLATILE={avg_miss_bucket['miss_VOLATILE']:.0f} DEGEN={avg_miss_bucket['miss_DEGEN']:.0f}")
    avg_miss_mag = miss_df[["miss_5_10pct", "miss_10_20pct", "miss_20pct_plus"]].mean()
    lines.append(f"Avg misses by magnitude: 5-10%={avg_miss_mag['miss_5_10pct']:.0f} 10-20%={avg_miss_mag['miss_10_20pct']:.0f} 20%+={avg_miss_mag['miss_20pct_plus']:.0f}")

    lines.append("\n## B) Within-indicator complementarity (Jaccard)\n")
    if len(jdf):
        lines.append(f"- HIGH redundancy (J>0.7): **{(jdf['jaccard']>0.7).sum()}** pairs (overfitting risk; pick one)")
        lines.append(f"- MED redundancy (0.4<J<=0.7): **{((jdf['jaccard']<=0.7)&(jdf['jaccard']>0.4)).sum()}** pairs")
        lines.append(f"- LOW J<=0.4 (COMPLEMENTARY): **{(jdf['jaccard']<=0.4).sum()}** pairs (gap-closure candidates)\n")
        lines.append("Most-complementary within-indicator pairs:")
        lines.append("| indicator | config A | config B | Jaccard | n_A | n_B |")
        lines.append("|---|---|---|--:|--:|--:|")
        for ind in jdf["indicator"].unique():
            sub = jdf[jdf["indicator"] == ind].sort_values("jaccard").head(2)
            for _, r in sub.iterrows():
                lines.append(f"| {ind} | `{r['config_A']}` | `{r['config_B']}` | {r['jaccard']:.3f} | {int(r['n_A'])} | {int(r['n_B'])} |")

    lines.append("\n## C) Cross-indicator LIFT (best gap-closure pairs)\n")
    if len(ldf):
        cross = ldf[~ldf["same_family"]].head(15)
        lines.append("Pairs where B closes A's gap most:")
        lines.append("| A indicator | A config | B indicator | B config | cov A % | cov B % | cov A+B % | lift pp |")
        lines.append("|---|---|---|---|--:|--:|--:|--:|")
        for _, r in cross.iterrows():
            lines.append(f"| {r['A_ind']} | `{r['A_cfg']}` | {r['B_ind']} | `{r['B_cfg']}` | {r['cov_A_pct']:.1f}% | {r['cov_B_pct']:.1f}% | {r['cov_AB_pct']:.1f}% | +{r['lift_pct']:.1f}pp |")

    lines.append("\n## D) Greedy optimal portfolio (incremental gap-closure)\n")
    lines.append("Starting from best-coverage single setup, each step adds the setup with highest marginal mover-capture on REMAINING uncaught movers.")
    lines.append("| step | indicator | config | marginal gain | cumulative coverage |")
    lines.append("|--:|---|---|--:|--:|")
    for _, r in pdf.iterrows():
        lines.append(f"| {int(r['step'])} | {r['indicator']} | `{r['config']}` | +{r['marginal_gain_pct']:.1f}pp | {r['cumulative_coverage_pct']:.1f}% |")
    lines.append(f"\n**Headline: {len(pdf)} complementary setups capture {pdf['cumulative_coverage_pct'].max():.1f}% of top movers. Best single setup catches {pdf.iloc[0]['cumulative_coverage_pct']:.1f}%; complementary stack adds +{pdf.iloc[-1]['cumulative_coverage_pct'] - pdf.iloc[0]['cumulative_coverage_pct']:.1f}pp.**")

    lines.append("\n## E) Interpretation\n")
    lines.append("- **WITHIN-FAMILY**: high-Jaccard pairs (redundant) should be pruned to 1; low-Jaccard pairs (complementary) keep both.")
    lines.append("- **CROSS-FAMILY**: lift pairs reveal natural composition. e.g. if SMA + RSI lift > each alone, they catch complementary move types.")
    lines.append("- **GREEDY PORTFOLIO**: stop where marginal lift falls under 1pp -- adding more risks overfitting.")
    lines.append("- The portfolio is the deploy recipe; rank co-firing setups by their PER-DAY conviction (ML/rule), not just inclusion in the library.")

    (OUT_DIR / "MISS_ATTRIBUTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'MISS_ATTRIBUTION_REPORT.md'}")

if __name__ == "__main__":
    main()
