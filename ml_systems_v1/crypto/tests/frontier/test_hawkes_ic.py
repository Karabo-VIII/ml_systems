"""G5 IC test: validate Hawkes branching ratio as a return-prediction feature.

Closes the IC-test followup of G5 from the gap audit 2026-04-25.

What this does:
  Loads `data/frontier/hawkes_enh/hawkes_branching_daily.parquet` (built by
  hawkes_branching_ratio.py) and tests information coefficient (Pearson and
  Spearman) of (eta_total, eta_buy, eta_sell, eta_imbalance) vs forward
  returns at horizons 1d / 3d / 5d / 7d.

Output:
  logs/g5_hawkes_ic_test/ic_per_asset.csv
  logs/g5_hawkes_ic_test/SUMMARY.md
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
HAWKES_FP = ROOT / "data" / "processed" / "panels" / "daily" / "hawkes_branching_daily.parquet"
DATA = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs" / "g5_hawkes_ic_test"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_returns_panel() -> pd.DataFrame:
    """Daily close + forward returns per asset from chimera files."""
    rows = []
    for asset in ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC"):
        fp = DATA / f"{asset.lower()}usdt_v50_chimera.parquet"
        if not fp.exists():
            continue
        df = pl.read_parquet(fp, columns=["timestamp", "close"]).to_pandas()
        df["date"] = pd.to_datetime(df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
        d = df.groupby("date").agg({"close": "last"}).reset_index()
        for h in (1, 3, 5, 7):
            d[f"fwd_{h}d"] = d["close"].shift(-h) / d["close"] - 1
        d["asset"] = asset
        d["date"] = pd.to_datetime(d["date"])
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    if not HAWKES_FP.exists():
        print(f"[g5_ic] Hawkes panel not found: {HAWKES_FP}")
        print("[g5_ic] Run src/frontier/features/hawkes_branching_ratio.py first.")
        return
    print(f"[g5_ic] loading: {HAWKES_FP}")
    h = pl.read_parquet(HAWKES_FP).to_pandas()
    h["date"] = pd.to_datetime(h["date"])
    print(f"[g5_ic] hawkes panel: {len(h)} rows, {h['asset'].nunique()} assets, "
          f"{h['date'].min()} -> {h['date'].max()}")

    rets = load_returns_panel()
    print(f"[g5_ic] returns panel: {len(rets)} rows")

    # Join on (asset, date)
    joined = h.merge(rets, on=["asset", "date"], how="inner")
    print(f"[g5_ic] joined: {len(joined)} rows")
    if len(joined) < 200:
        print("[g5_ic] insufficient data; aborting")
        return

    feat_cols = ["eta_total", "eta_buy", "eta_sell", "eta_imbalance"]
    horizons = [1, 3, 5, 7]
    rows = []
    for asset in sorted(joined["asset"].unique()):
        sub = joined[joined["asset"] == asset].copy()
        for f in feat_cols:
            for h_ in horizons:
                target = f"fwd_{h_}d"
                df_ = sub.dropna(subset=[f, target])
                if len(df_) < 15:
                    continue
                if df_[f].std() == 0 or df_[target].std() == 0:
                    continue
                ic = float(np.corrcoef(df_[f], df_[target])[0, 1])
                rho = float(stats.spearmanr(df_[f], df_[target])[0])
                # t-stat for IC under independent-sample null
                n = len(df_)
                t_ic = ic * math.sqrt((n - 2) / max(1 - ic ** 2, 1e-12))
                rows.append({
                    "asset": asset, "feature": f, "horizon": h_,
                    "n": n, "IC_pearson": ic, "IC_spearman": rho,
                    "t_stat": t_ic,
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(LOG_DIR / "ic_per_asset.csv", index=False)
    print(f"[g5_ic] saved: {LOG_DIR/'ic_per_asset.csv'}")

    # Pooled IC: stack all assets, compute single IC per (feature, horizon)
    pooled_rows = []
    for f in feat_cols:
        for h_ in horizons:
            target = f"fwd_{h_}d"
            sub = joined.dropna(subset=[f, target])
            if len(sub) < 50 or sub[f].std() == 0:
                continue
            ic = float(np.corrcoef(sub[f], sub[target])[0, 1])
            rho = float(stats.spearmanr(sub[f], sub[target])[0])
            n = len(sub)
            t_ic = ic * math.sqrt((n - 2) / max(1 - ic ** 2, 1e-12))
            pooled_rows.append({
                "feature": f, "horizon": h_, "n": n,
                "IC_pearson": ic, "IC_spearman": rho, "t_stat": t_ic,
            })
    pooled_df = pd.DataFrame(pooled_rows)
    pooled_df.to_csv(LOG_DIR / "ic_pooled.csv", index=False)

    # Summary
    summary = ["# Hawkes Branching Ratio IC Test -- 2026-04-25", ""]
    summary.append(f"Joined panel: {len(joined)} rows, {h['asset'].nunique()} assets, "
                   f"date range {joined['date'].min().date()} -> {joined['date'].max().date()}.")
    summary.append("")
    summary.append("## Pooled IC across assets")
    summary.append("")
    summary.append("| Feature | H | n | IC (Pearson) | Spearman | t-stat |")
    summary.append("|---------|---|---|--------------|----------|--------|")
    for _, r in pooled_df.sort_values(["feature", "horizon"]).iterrows():
        summary.append(
            f"| {r['feature']} | {int(r['horizon'])} | {int(r['n'])} | "
            f"{r['IC_pearson']:+.4f} | {r['IC_spearman']:+.4f} | {r['t_stat']:+.2f} |"
        )
    summary.append("")
    summary.append("## Per-asset top hits (|IC|>0.05 AND |t|>2.0)")
    summary.append("")
    hits = df_out[(df_out["IC_pearson"].abs() > 0.05) & (df_out["t_stat"].abs() > 2.0)]
    hits = hits.sort_values("t_stat", key=lambda s: s.abs(), ascending=False).head(20)
    if len(hits):
        summary.append("| Asset | Feature | H | n | IC | t-stat |")
        summary.append("|-------|---------|---|---|-----|--------|")
        for _, r in hits.iterrows():
            summary.append(f"| {r['asset']} | {r['feature']} | {int(r['horizon'])} | "
                           f"{int(r['n'])} | {r['IC_pearson']:+.4f} | {r['t_stat']:+.2f} |")
    else:
        summary.append("(none)")
    summary.append("")
    summary.append("## Interpretation")
    summary.append("")
    summary.append("- Per ml_upgrades_research_2026_04_22.md, Rambaldi 2024 claims +6-12% ShIC")
    summary.append("  for branching ratio in cascade prediction. Empirical test here shows the")
    summary.append("  pooled and per-asset IC against forward returns.")
    summary.append("- IC > +0.05 with |t|>2 = signal worth integrating into xsec ranker as a feature.")
    summary.append("- IC near zero = signal too weak to ship; do NOT integrate.")
    (LOG_DIR / "SUMMARY.md").write_text("\n".join(summary))
    print(f"[g5_ic] summary: {LOG_DIR/'SUMMARY.md'}")
    # also print pooled to stdout
    print("\n[g5_ic] pooled IC:")
    print(pooled_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
