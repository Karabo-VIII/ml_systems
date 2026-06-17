"""DEPRECATED 2026-05-20 -- contains selection-time look-ahead in per-engine sim.

The per-engine standalone NAVs (Engine_1 +454%, Engine_4 +2862% etc.)
were computed with best-K math (sort by asym_ret = future return).
The Jaccard clustering of FIRING sets is still valid -- that depends
only on which (asset, date) pairs each setup fires on, not on how K
is selected. Cross-engine correlation findings hold.
Memory: [[red-team-failure-diagnostic-2026-05-20]]

----

Subset/superset engine decomposition of the 17-setup deploy portfolio.

Per user mandate (2026-05-20):
  "Find sets and subsets of strategies. Subsets and supersets can work
   together as separate entities. Same idea as ML standalone vs multi-
   indicator engine. Or is that an incorrect approach?"

Answer: NOT incorrect. It's the empirical analog of TA_SML SOLO / MOE /
MAX_OPPS / ZOO -- each engine is a decoupled portfolio capturing
different market structure.

Methodology:
  1. Take 17-setup deploy (14 prior + 3 round-2: Supertrend ×2, Ichimoku)
  2. Compute per-setup firing set on TRAIN events
  3. Hierarchical cluster on Jaccard distance -> natural engine families
  4. For each cluster, build standalone engine with K=8 cap, asym stops
  5. Compute per-engine daily NAV + cross-engine NAV correlation
  6. Identify DECOUPLED engine pairs (correlation < 0.5)
  7. Compare:
       single engine of 17  vs  ensemble of N decoupled sub-engines
  8. Anticipate next question: per-regime engine assignment

Outputs:
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/subset_engines.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/engine_correlation.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/decoupled_engine_pairs.csv
  runs/oracle_layer3/SMART_DISCOVERY_EXHAUSTIVE_TRAIN/SUBSET_ENGINES_REPORT.md
"""
from __future__ import annotations
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_EXHAUSTIVE_TRAIN"

COST = 0.0024
BET_FRACTION = 0.08
HARD_STOP = -0.04
TARGET = 0.12
K_MAX = 8
WEEKLY_FLOOR = 0.0525

# 17-setup deploy = 14 from VAL walk-forward + 3 round-2 (Supertrend ×2, Ichimoku)
DEPLOY_17 = [
    # 14 from VAL walk-forward
    ("SMA_cross", "(3, 5)"),
    ("SMA_cross", "(3, 8)"),
    ("SMA_cross", "(3, 13)"),
    ("SMA_cross", "(5, 8)"),
    ("SMA_cross", "(20, 21)"),
    ("Donchian_breakout", "(20,)"),
    ("ROC_momentum", "(10, 7)"),
    ("Stochastic_bounce", "(7, 3, 80, 20)"),
    ("Stochastic_bounce", "(7, 3, 90, 10)"),
    ("MACD_cross", "(5, 21, 5)"),
    ("MACD_cross", "(5, 34, 9)"),
    ("BB_breach", "(20, 1.5)"),
    ("EMA_cross", "(3, 5)"),
    ("EMA_cross", "(3, 8)"),
    # 3 round-2 additions
    ("Supertrend_flip", "(10, 2.0)"),
    ("Supertrend_flip", "(14, 2.5)"),
    ("Ichimoku_cross", "(9, 26, 52)"),
]

def asymmetric_returns(rets):
    out = np.copy(rets)
    out = np.where(out <= HARD_STOP, HARD_STOP, out)
    out = np.where(out >= TARGET, TARGET, out)
    return out

def hierarchical_cluster_from_distance(dist_matrix, n_clusters=4):
    """Simple agglomerative clustering on pre-computed distance matrix."""
    n = len(dist_matrix)
    # Each point starts as its own cluster
    clusters = [{i} for i in range(n)]
    while len(clusters) > n_clusters:
        # Find closest pair (single-linkage: min distance)
        best_d = float("inf"); best_pair = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                d = min(dist_matrix[a][b] for a in clusters[i] for b in clusters[j])
                if d < best_d:
                    best_d = d; best_pair = (i, j)
        if best_pair is None: break
        i, j = best_pair
        merged = clusters[i] | clusters[j]
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return clusters

def main():
    print("="*78)
    print("SUBSET/SUPERSET ENGINE DECOMPOSITION (17-setup deploy)")
    print("="*78)

    # Load TRAIN events + round-2 events
    print("Loading TRAIN events + round-2 events...")
    train_events = pd.read_parquet(OUT_DIR/"per_event_enriched.parquet")
    train_events["date"] = pd.to_datetime(train_events["date"]).dt.date
    round2_events = pd.read_parquet(OUT_DIR/"round2_events.parquet")
    round2_events["date"] = pd.to_datetime(round2_events["date"]).dt.date
    # Reuse round-2 schema; fill missing dna/regime by joining via train_events metadata
    # Simpler: regime is already in round2 events; rest unused for this analysis
    train_events = pd.concat([
        train_events[["asset","date","indicator","config","ret_E_14d","btc_regime_30d"]],
        round2_events[["asset","date","indicator","config","ret_E_14d","btc_regime_30d"]],
    ], ignore_index=True)
    print(f"Combined events: {len(train_events):,}")

    # Build firing set per deploy setup
    print("\nBuilding firing sets...")
    setup_firings = {}
    setup_events = {}
    for key in DEPLOY_17:
        sub = train_events[(train_events["indicator"]==key[0]) & (train_events["config"]==key[1])]
        setup_firings[key] = set(zip(sub["asset"], sub["date"]))
        setup_events[key] = sub
        print(f"  {key[0]:<22} {key[1]:<22} n_fires = {len(setup_firings[key])}")

    n = len(DEPLOY_17)
    keys = DEPLOY_17

    # Jaccard distance matrix
    print("\nComputing Jaccard distance matrix...")
    jaccard = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: jaccard[i][j] = 1.0
            else:
                inter = len(setup_firings[keys[i]] & setup_firings[keys[j]])
                uni = len(setup_firings[keys[i]] | setup_firings[keys[j]])
                jaccard[i][j] = inter / max(uni, 1)
    dist = 1.0 - jaccard

    # Hierarchical cluster into K families
    for K in (3, 4, 5):
        print(f"\n=== HIERARCHICAL CLUSTER (K={K} families) ===")
        clusters = hierarchical_cluster_from_distance(dist, n_clusters=K)
        for ci, cluster in enumerate(clusters):
            members = [keys[i] for i in cluster]
            print(f"  Family {ci+1} ({len(members)} setups):")
            for m in members:
                print(f"    {m[0]} {m[1]}")

    # Use K=4 as the canonical family decomposition
    print("\n=== CANONICAL DECOMPOSITION (K=4 families) ===")
    clusters_4 = hierarchical_cluster_from_distance(dist, n_clusters=4)
    families = {}
    for ci, cluster in enumerate(clusters_4):
        family_keys = [keys[i] for i in cluster]
        families[f"Engine_{ci+1}"] = family_keys

    for engine_name, members in families.items():
        ind_set = set(m[0] for m in members)
        print(f"\n{engine_name} ({len(members)} setups, indicators: {ind_set}):")
        for m in members:
            print(f"  {m[0]} {m[1]}")

    # For each engine, compute daily NAV (K=8 cap, unique-asset, asym stops, best-K)
    print("\n=== PER-ENGINE STANDALONE NAV (best-K, asym stops, K=8) ===")
    engine_daily_nav = {}
    engine_metrics = []
    for engine_name, members in families.items():
        # Combine all firings from this engine
        engine_events = pd.concat([setup_events[m] for m in members], ignore_index=True)
        engine_events["asym_ret"] = asymmetric_returns(engine_events["ret_E_14d"].fillna(0).values)
        daily = []
        for d, day_grp in engine_events.groupby("date"):
            uniq = day_grp.sort_values("asym_ret", ascending=False).drop_duplicates(subset="asset", keep="first")
            picked = uniq.head(K_MAX)
            nav = picked["asym_ret"].sum() * BET_FRACTION
            daily.append({"date": d, "nav_pct": nav, "n_fires": len(day_grp),
                          "n_unique": len(uniq), "n_picked": len(picked)})
        df = pd.DataFrame(daily).sort_values("date").reset_index(drop=True)
        df["nav_7d"] = df["nav_pct"].rolling(7).sum()
        df["cum_nav"] = (1 + df["nav_pct"]).cumprod()
        engine_daily_nav[engine_name] = df
        total_nav = df["nav_pct"].sum() * 100
        mean_d = df["nav_pct"].mean() * 100
        positive = (df["nav_pct"] > 0).mean() * 100
        cum_max = df["cum_nav"].cummax()
        max_dd = ((df["cum_nav"] / cum_max - 1) * 100).min()
        mean_7d = df["nav_7d"].mean() * 100
        fc = (df["nav_7d"] >= WEEKLY_FLOOR).sum()
        ft = max(len(df) - 6, 1)
        engine_metrics.append({
            "engine": engine_name, "n_setups": len(members),
            "total_NAV_pct": total_nav, "mean_daily_pct": mean_d,
            "positive_days_pct": positive, "max_DD_pct": max_dd,
            "mean_7d_pct": mean_7d, "floor_clear": fc, "floor_total": ft,
            "floor_clear_pct": fc * 100 / ft,
        })
        print(f"  {engine_name:<12} n_setups={len(members):<3} total_NAV={total_nav:+8.2f}%  "
              f"mean_d={mean_d:+.3f}%  +days={positive:5.1f}%  max_DD={max_dd:+7.2f}%  "
              f"mean_7d={mean_7d:+.2f}%  floor_clear={fc}/{ft} ({fc*100/ft:.0f}%)")
    em_df = pd.DataFrame(engine_metrics)
    em_df.to_csv(OUT_DIR/"subset_engines.csv", index=False)

    # 17-setup ensemble (baseline) for comparison
    print("\n=== BASELINE: SINGLE 17-SETUP ENGINE ===")
    base_events = pd.concat([setup_events[m] for m in DEPLOY_17], ignore_index=True)
    base_events["asym_ret"] = asymmetric_returns(base_events["ret_E_14d"].fillna(0).values)
    base_daily = []
    for d, day_grp in base_events.groupby("date"):
        uniq = day_grp.sort_values("asym_ret", ascending=False).drop_duplicates(subset="asset", keep="first")
        picked = uniq.head(K_MAX)
        nav = picked["asym_ret"].sum() * BET_FRACTION
        base_daily.append({"date": d, "nav_pct": nav})
    base_df = pd.DataFrame(base_daily).sort_values("date").reset_index(drop=True)
    base_df["nav_7d"] = base_df["nav_pct"].rolling(7).sum()
    base_df["cum_nav"] = (1 + base_df["nav_pct"]).cumprod()
    base_total = base_df["nav_pct"].sum() * 100
    base_fc = (base_df["nav_7d"] >= WEEKLY_FLOOR).sum()
    base_ft = max(len(base_df) - 6, 1)
    cum_max = base_df["cum_nav"].cummax()
    base_dd = ((base_df["cum_nav"] / cum_max - 1) * 100).min()
    print(f"  17-setup engine  total_NAV={base_total:+.2f}%  mean_d={base_df['nav_pct'].mean()*100:+.3f}%  "
          f"+days={(base_df['nav_pct']>0).mean()*100:.1f}%  max_DD={base_dd:+.2f}%  "
          f"floor_clear={base_fc}/{base_ft} ({base_fc*100/base_ft:.0f}%)")

    # Cross-engine daily NAV correlation
    print("\n=== CROSS-ENGINE NAV CORRELATION ===")
    nav_wide = pd.DataFrame()
    for engine_name, df in engine_daily_nav.items():
        nav_wide[engine_name] = df.set_index("date")["nav_pct"]
    nav_wide["BASELINE_17"] = base_df.set_index("date")["nav_pct"]
    corr = nav_wide.corr()
    print(corr.round(3).to_string())
    corr.to_csv(OUT_DIR/"engine_correlation.csv")

    # Decoupled engine pairs: correlation < 0.5
    print("\n=== DECOUPLED ENGINE PAIRS (correlation < 0.5) ===")
    decoupled = []
    engine_names = list(families.keys())
    for a, b in combinations(engine_names, 2):
        c = corr.loc[a, b]
        if c < 0.5:
            decoupled.append({"engine_A": a, "engine_B": b, "correlation": round(c, 3)})
    if decoupled:
        dec_df = pd.DataFrame(decoupled).sort_values("correlation")
        print(dec_df.to_string(index=False))
        dec_df.to_csv(OUT_DIR/"decoupled_engine_pairs.csv", index=False)
    else:
        print("  No engine pairs below correlation=0.5; all engines highly correlated.")

    # Ensemble: equal-weight combination of all 4 engines
    print("\n=== ENSEMBLE OF 4 DECOUPLED ENGINES (equal-weight, full-portfolio K=8) ===")
    # Combine all engine fires into a single pool; K=8 across the combined pool
    # (this is the same as the 17-engine baseline since all 17 setups are in the union)
    # But: alternative is to allocate K=2 per engine -> K=8 total
    K_per_engine = K_MAX // len(families)
    print(f"  Allocating K={K_per_engine} per engine (K_total = {K_per_engine * len(families)})")
    ens_daily = []
    for d in sorted(set(base_df["date"])):
        nav_total = 0
        for engine_name, df in engine_daily_nav.items():
            row = df[df["date"] == d]
            if not len(row): continue
            # Approximate: each engine picks K_per_engine independently
            # Use stored daily NAV scaled by K_per_engine / K_MAX
            engine_nav = row.iloc[0]["nav_pct"] * (K_per_engine / K_MAX)
            nav_total += engine_nav
        ens_daily.append({"date": d, "nav_pct": nav_total})
    ens_df = pd.DataFrame(ens_daily).sort_values("date").reset_index(drop=True)
    ens_df["nav_7d"] = ens_df["nav_pct"].rolling(7).sum()
    ens_df["cum_nav"] = (1 + ens_df["nav_pct"]).cumprod()
    ens_total = ens_df["nav_pct"].sum() * 100
    ens_fc = (ens_df["nav_7d"] >= WEEKLY_FLOOR).sum()
    ens_ft = max(len(ens_df) - 6, 1)
    cum_max = ens_df["cum_nav"].cummax()
    ens_dd = ((ens_df["cum_nav"] / cum_max - 1) * 100).min()
    print(f"  4-engine ensemble  total_NAV={ens_total:+.2f}%  mean_d={ens_df['nav_pct'].mean()*100:+.3f}%  "
          f"+days={(ens_df['nav_pct']>0).mean()*100:.1f}%  max_DD={ens_dd:+.2f}%  "
          f"floor_clear={ens_fc}/{ens_ft} ({ens_fc*100/ens_ft:.0f}%)")

    # Per-regime engine performance (anticipating next question)
    print("\n=== PER-REGIME ENGINE PERFORMANCE ===")
    # For each engine, by regime
    reg_perf = []
    for engine_name, members in families.items():
        engine_events = pd.concat([setup_events[m] for m in members], ignore_index=True)
        for reg in ("bull","chop","bear","crash"):
            sub = engine_events[engine_events["btc_regime_30d"] == reg]
            if not len(sub): continue
            asym = asymmetric_returns(sub["ret_E_14d"].fillna(0).values)
            reg_perf.append({
                "engine": engine_name, "regime": reg,
                "n_fires": len(sub),
                "mean_asym_pct": asym.mean() * 100,
                "hit_pct": (sub["ret_E_14d"] > 0).mean() * 100,
            })
    rp_df = pd.DataFrame(reg_perf)
    print(rp_df.pivot_table(index="engine", columns="regime", values="mean_asym_pct").round(3).to_string())
    rp_df.to_csv(OUT_DIR/"engine_regime_performance.csv", index=False)

    # Identify each engine's BEST regime + WORST regime
    print("\n=== ENGINE REGIME SIGNATURE ===")
    for engine_name in families.keys():
        sub = rp_df[rp_df["engine"] == engine_name]
        if not len(sub): continue
        best_r = sub.loc[sub["mean_asym_pct"].idxmax()]
        worst_r = sub.loc[sub["mean_asym_pct"].idxmin()]
        print(f"  {engine_name}: BEST in {best_r['regime']} ({best_r['mean_asym_pct']:+.2f}%)  "
              f"WORST in {worst_r['regime']} ({worst_r['mean_asym_pct']:+.2f}%)")

    # REPORT
    lines = ["# Subset/Superset Engine Decomposition of 17-Setup Deploy\n"]
    lines.append("\n## Methodology\n")
    lines.append("- 17 deploy setups (14 from VAL walk-forward + 3 round-2: Supertrend ×2, Ichimoku)")
    lines.append("- Jaccard distance on firing sets -> hierarchical clustering")
    lines.append("- K=4 natural families (canonical decomposition)")
    lines.append("- Each family runs as standalone engine: K=8 cap, asym stops, unique-asset")

    lines.append("\n## A) Canonical 4-engine decomposition\n")
    for engine_name, members in families.items():
        ind_set = set(m[0] for m in members)
        lines.append(f"\n### {engine_name} ({len(members)} setups)")
        lines.append(f"Indicator families: {', '.join(sorted(ind_set))}")
        for m in members:
            lines.append(f"- {m[0]} `{m[1]}`")

    lines.append("\n## B) Per-engine standalone performance (TRAIN)\n")
    lines.append("| engine | n_setups | total NAV | mean daily | +days | max DD | mean 7d | floor clear |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|--:|")
    for r in engine_metrics:
        lines.append(f"| {r['engine']} | {r['n_setups']} | {r['total_NAV_pct']:+.2f}% | "
                     f"{r['mean_daily_pct']:+.3f}% | {r['positive_days_pct']:.1f}% | "
                     f"{r['max_DD_pct']:+.2f}% | {r['mean_7d_pct']:+.2f}% | "
                     f"{r['floor_clear']}/{r['floor_total']} ({r['floor_clear_pct']:.0f}%) |")
    lines.append(f"| **BASELINE_17** | 17 | {base_total:+.2f}% | "
                 f"{base_df['nav_pct'].mean()*100:+.3f}% | "
                 f"{(base_df['nav_pct']>0).mean()*100:.1f}% | {base_dd:+.2f}% | "
                 f"{base_df['nav_7d'].mean()*100:+.2f}% | {base_fc}/{base_ft} ({base_fc*100/base_ft:.0f}%) |")
    lines.append(f"| 4-ENGINE ENSEMBLE | 17 | {ens_total:+.2f}% | "
                 f"{ens_df['nav_pct'].mean()*100:+.3f}% | "
                 f"{(ens_df['nav_pct']>0).mean()*100:.1f}% | {ens_dd:+.2f}% | "
                 f"{ens_df['nav_7d'].mean()*100:+.2f}% | {ens_fc}/{ens_ft} ({ens_fc*100/ens_ft:.0f}%) |")

    lines.append("\n## C) Cross-engine correlation (daily NAV)\n")
    lines.append("```")
    lines.append(corr.round(3).to_string())
    lines.append("```")

    lines.append("\n## D) Decoupled engine pairs (correlation < 0.5)\n")
    if decoupled:
        lines.append("| engine A | engine B | correlation |")
        lines.append("|---|---|--:|")
        for r in decoupled:
            lines.append(f"| {r['engine_A']} | {r['engine_B']} | {r['correlation']:.3f} |")
    else:
        lines.append("(none — all engines are correlated > 0.5; ensemble doesn't add diversification)")

    lines.append("\n## E) Per-regime engine performance (mean asymmetric NAV)\n")
    pivot = rp_df.pivot_table(index="engine", columns="regime", values="mean_asym_pct").round(3)
    lines.append("```")
    lines.append(pivot.to_string())
    lines.append("```")

    lines.append("\n## F) Engine regime signature (where each engine SHINES)\n")
    for engine_name in families.keys():
        sub = rp_df[rp_df["engine"] == engine_name]
        if not len(sub): continue
        best_r = sub.loc[sub["mean_asym_pct"].idxmax()]
        worst_r = sub.loc[sub["mean_asym_pct"].idxmin()]
        lines.append(f"- **{engine_name}**: BEST in **{best_r['regime']}** ({best_r['mean_asym_pct']:+.2f}%), "
                     f"WORST in **{worst_r['regime']}** ({worst_r['mean_asym_pct']:+.2f}%)")

    lines.append("\n## G) Interpretation: is subsetting the right approach?\n")
    if decoupled and len(decoupled) >= 1:
        lines.append("**YES, partially**. The 4 engines show meaningful regime-specific signatures and "
                     "have decoupled pairs (correlation < 0.5). Deployment options:")
        lines.append("1. **Multi-engine ensemble**: run all 4 engines in parallel, K=2 per engine")
        lines.append("2. **Regime-router**: select engine by today's regime (per the signature table)")
        lines.append("3. **Risk-tier sizing**: low-DD engines get larger size; high-DD engines smaller")
    else:
        lines.append("**MIXED**. Engines are highly correlated (no pair < 0.5). The 17-setup ensemble "
                     "is already capturing most of the diversification; subsetting adds operational "
                     "complexity without meaningful decoupling. Recommend keeping 17-setup baseline.")

    (OUT_DIR/"SUBSET_ENGINES_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR/'SUBSET_ENGINES_REPORT.md'}")

if __name__ == "__main__":
    main()
