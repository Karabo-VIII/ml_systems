"""build_before_period_oracle.py — P4-3: BEFORE-period oracle.

For each historical +15% / +25% mover (the AFTER + DURING already mapped),
characterize the BEFORE period (5-7 days pre-trigger) feature signature.
Cluster signatures via k-means on a feature panel. Test whether any cluster
predicts forward continuation better than the unconditional trigger.

If a clean BEFORE signature emerges:
  - It becomes a new R1 detector that fires EARLIER than the +15% trigger
  - Earlier entry = better R3 timer = more of the move captured

Outputs:
  runs/audit/before_period_signatures.parquet (per-event features + cluster)
  runs/audit/P4_3_BEFORE_ORACLE_VERDICT_2026_05_19.md
"""
from __future__ import annotations
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

os.environ["PYTHONIOENCODING"] = "utf-8"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "runs" / "audit"


def main():
    panel = pl.read_parquet(str(OUT_DIR / "oracle_panel_binance_2026_05_18.parquet"))
    panel = panel.with_columns(pl.col("date").cast(pl.Date, strict=False))
    panel = panel.sort(["asset", "date"])

    # Compute ret_1d + forward returns
    panel = panel.with_columns(
        pl.col("close").pct_change(1).over("asset").alias("ret_1d")
    ).with_columns([
        pl.col("close").shift(-1).over("asset").alias("close_t1"),
        pl.col("close").shift(-4).over("asset").alias("close_t4"),
        pl.col("close").shift(-6).over("asset").alias("close_t6"),
    ]).with_columns([
        ((pl.col("close_t4") / pl.col("close_t1")) - 1).alias("fwd_3d"),
        ((pl.col("close_t6") / pl.col("close_t1")) - 1).alias("fwd_5d"),
    ])

    # BEFORE-period features: compute for every (asset, date) — what was the 7d/14d run-up?
    # then we'll select rows where ret_1d_next becomes >= +15% as "trigger events"
    # actually here ret_1d is the DAY OF the trigger; we want features at t-1
    panel = panel.with_columns([
        # Lookback rolling features (causal)
        pl.col("close").pct_change(3).over("asset").alias("ret_3d_prior"),
        pl.col("close").pct_change(7).over("asset").alias("ret_7d_prior"),
        pl.col("close").pct_change(14).over("asset").alias("ret_14d_prior"),
        pl.col("close").pct_change(30).over("asset").alias("ret_30d_prior"),
        # Volume features
        pl.col("volume").rolling_mean(7).over("asset").alias("vol_7d_mean"),
        pl.col("volume").rolling_mean(30).over("asset").alias("vol_30d_mean"),
        pl.col("trades").rolling_mean(7).over("asset").alias("trades_7d_mean"),
        # Realized volatility (7-day rolling std of ret_1d)
        pl.col("ret_1d").rolling_std(7).over("asset").alias("rv_7d"),
        pl.col("ret_1d").rolling_std(14).over("asset").alias("rv_14d"),
        # High-low range
        ((pl.col("high") - pl.col("low")) / pl.col("close")).rolling_mean(7).over("asset").alias("hl_range_7d"),
    ]).with_columns([
        # Derived: volume ratio (recent vs longer-term)
        (pl.col("vol_7d_mean") / pl.col("vol_30d_mean")).alias("vol_ratio_7v30"),
        # 1d before features (these are what's available WHEN the trigger fires; t-1 close known)
    ])

    # Filter to 8Q WF
    p8q = panel.filter((pl.col("date") >= date(2024,1,1)) & (pl.col("date") <= date(2025,12,31)))
    print(f"[before] 8Q panel: {len(p8q)} rows")

    # === TRIGGER EVENTS: ret_1d >= +15% on (asset, date) — the "WINNERS" ===
    for trig in [0.15, 0.25]:
        events = p8q.filter(pl.col("ret_1d") >= trig).drop_nulls(
            subset=["fwd_3d", "ret_3d_prior", "ret_7d_prior", "ret_14d_prior", "rv_7d", "vol_ratio_7v30", "hl_range_7d"]
        )
        n = len(events)
        if n == 0:
            print(f"  trigger >= {trig*100:.0f}%: 0 events")
            continue
        print(f"\n=== Trigger ret_1d >= {trig*100:.0f}%: n={n} events ===")

        # What were the BEFORE-period features at t-1 (one day before trigger)?
        # In this panel, the ret_3d_prior etc are computed UP TO AND INCLUDING t.
        # For BEFORE features, we want them computed at t-1 (no peek at t).
        # Shift by 1 to get "as of t-1" values
        before = p8q.sort(["asset","date"]).with_columns([
            pl.col("ret_3d_prior").shift(1).over("asset").alias("b_ret_3d"),
            pl.col("ret_7d_prior").shift(1).over("asset").alias("b_ret_7d"),
            pl.col("ret_14d_prior").shift(1).over("asset").alias("b_ret_14d"),
            pl.col("ret_30d_prior").shift(1).over("asset").alias("b_ret_30d"),
            pl.col("rv_7d").shift(1).over("asset").alias("b_rv_7d"),
            pl.col("rv_14d").shift(1).over("asset").alias("b_rv_14d"),
            pl.col("vol_ratio_7v30").shift(1).over("asset").alias("b_vol_ratio"),
            pl.col("hl_range_7d").shift(1).over("asset").alias("b_hl_range"),
        ])

        # Join the BEFORE features onto the trigger events
        events_w_before = events.select(["asset","date","ret_1d","fwd_3d","fwd_5d","close"]).join(
            before.select(["asset","date","b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_rv_14d","b_vol_ratio","b_hl_range"]),
            on=["asset","date"], how="left"
        ).drop_nulls(subset=["b_ret_3d","b_ret_7d","b_ret_14d","b_rv_7d","b_vol_ratio","b_hl_range"])
        n_eff = len(events_w_before)
        print(f"  events with BEFORE features: {n_eff}")
        if n_eff < 50:
            continue

        # Print BEFORE-feature distributions split by AFTER outcome
        winners_3d = events_w_before.filter(pl.col("fwd_3d") > 0.05)
        losers_3d = events_w_before.filter(pl.col("fwd_3d") < -0.05)
        print(f"  Continuation (fwd_3d > +5%): {len(winners_3d)} events")
        print(f"  Reversal (fwd_3d < -5%): {len(losers_3d)} events")
        print(f"\n  {'feature':<14} {'winner_mean':>14} {'loser_mean':>14} {'separation':>12}")
        for feat in ["b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_vol_ratio","b_hl_range"]:
            w = float(winners_3d[feat].mean()) if len(winners_3d) > 0 else 0
            l = float(losers_3d[feat].mean()) if len(losers_3d) > 0 else 0
            sep = w - l
            print(f"  {feat:<14} {w:>+13.4f} {l:>+13.4f} {sep:>+11.4f}")

        # === K-MEANS CLUSTERING on BEFORE features ===
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        feature_cols = ["b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_vol_ratio","b_hl_range"]
        events_w_before_clean = events_w_before.drop_nulls(subset=feature_cols)
        if len(events_w_before_clean) < len(events_w_before):
            print(f"  dropped {len(events_w_before) - len(events_w_before_clean)} rows with NaN features")
        events_w_before = events_w_before_clean
        X = events_w_before.select(feature_cols).to_numpy()
        # Replace any remaining NaN/inf with column median
        X = np.where(np.isfinite(X), X, np.nan)
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            if mask.any():
                X[mask, j] = col_medians[j]
        X = StandardScaler().fit_transform(X)
        if len(X) >= 50:
            for k in [3, 5, 7]:
                km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                labels = km.labels_
                # Per-cluster fwd_3d mean
                ev_with_cl = events_w_before.with_columns(pl.Series("cluster", labels.tolist()))
                cl_stats = ev_with_cl.group_by("cluster").agg([
                    pl.len().alias("n"),
                    pl.col("fwd_3d").mean().alias("mean_fwd_3d"),
                    pl.col("fwd_5d").mean().alias("mean_fwd_5d"),
                    (pl.col("fwd_3d") > 0).cast(pl.Float64).mean().alias("frac_pos_3d"),
                ]).sort("mean_fwd_3d", descending=True)
                print(f"\n  K-means k={k} clusters (sorted by mean_fwd_3d):")
                for r in cl_stats.iter_rows(named=True):
                    print(f"    cluster {r['cluster']}: n={r['n']:>3d} ({r['n']/n_eff*100:>4.1f}%)  "
                          f"mean_fwd_3d={r['mean_fwd_3d']*100:>+6.2f}%  mean_fwd_5d={r['mean_fwd_5d']*100:>+6.2f}%  "
                          f"frac_pos_3d={r['frac_pos_3d']*100:.1f}%")

                # Best cluster vs unconditional
                best_cl = cl_stats.head(1).row(0, named=True)
                worst_cl = cl_stats.tail(1).row(0, named=True)
                uncond_mean = float(events_w_before["fwd_3d"].mean())
                print(f"    Unconditional: mean_fwd_3d={uncond_mean*100:+.2f}%")
                print(f"    BEST cluster: {best_cl['mean_fwd_3d']*100:+.2f}% ({(best_cl['mean_fwd_3d']-uncond_mean)*100:+.2f}pp lift)")
                print(f"    WORST cluster: {worst_cl['mean_fwd_3d']*100:+.2f}% ({(worst_cl['mean_fwd_3d']-uncond_mean)*100:+.2f}pp drag)")

    # Save the +15% events with their cluster labels (best run)
    print("\n[before] saving +15% trigger events with k=5 cluster labels...")
    events = p8q.filter(pl.col("ret_1d") >= 0.15).drop_nulls(
        subset=["fwd_3d", "ret_3d_prior", "ret_7d_prior", "ret_14d_prior", "rv_7d", "vol_ratio_7v30", "hl_range_7d"]
    )
    before = p8q.sort(["asset","date"]).with_columns([
        pl.col("ret_3d_prior").shift(1).over("asset").alias("b_ret_3d"),
        pl.col("ret_7d_prior").shift(1).over("asset").alias("b_ret_7d"),
        pl.col("ret_14d_prior").shift(1).over("asset").alias("b_ret_14d"),
        pl.col("ret_30d_prior").shift(1).over("asset").alias("b_ret_30d"),
        pl.col("rv_7d").shift(1).over("asset").alias("b_rv_7d"),
        pl.col("rv_14d").shift(1).over("asset").alias("b_rv_14d"),
        pl.col("vol_ratio_7v30").shift(1).over("asset").alias("b_vol_ratio"),
        pl.col("hl_range_7d").shift(1).over("asset").alias("b_hl_range"),
    ])
    ev_full = events.select(["asset","date","ret_1d","fwd_3d","fwd_5d"]).join(
        before.select(["asset","date","b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_rv_14d","b_vol_ratio","b_hl_range"]),
        on=["asset","date"], how="left"
    ).drop_nulls()
    feature_cols = ["b_ret_3d","b_ret_7d","b_ret_14d","b_ret_30d","b_rv_7d","b_vol_ratio","b_hl_range"]
    X = ev_full.select(feature_cols).to_numpy()
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X)
    ev_with_cl = ev_full.with_columns(pl.Series("cluster", km.labels_.tolist()))
    ev_with_cl.write_parquet(str(OUT_DIR / "before_period_signatures.parquet"))
    print(f"  wrote {len(ev_with_cl)} events with cluster labels")


if __name__ == "__main__":
    main()
