"""ma_ema_full_profile.py -- exhaustive MA/EMA profile across all axes.

Per user mandate 2026-05-20: "Run the oracle exercise and extract the Moving
Average profile exhaustively. Top performers, complementary performers,
correlation, regime, asset DNA, per asset, etc. I want to know everything
about the top profiles MA/EMA has to offer us. If we need the per-asset
profile (making the asset the pivot instead of the strategy), let's do that.
If we need per-strat and then we glue assets to that, we do that. I want
the best config for the MA to give us the best results."

DATA SOURCES (all on disk):
  1. runs/oracle_layer3/ma_ema_permutation/pair_summary.parquet
     -- VAL window (2023-07 to 2024-05); per-pair stats conditional on
        >1% cc event days. Universal (all assets).
  2. runs/oracle_layer3/ma_ema_permutation/pair_by_regime.parquet
     -- same + regime slicing
  3. runs/oracle_layer3/ma_ema_permutation/pair_by_bucket.parquet
     -- same + bucket (BLUE/STEADY/VOLATILE/DEGEN)
  4. runs/oracle_layer3/ma_ema_per_asset_train/pair_by_asset_cadence.parquet
     -- TRAIN window; per-asset × cadence × ma_type × (fast, slow)
  5. runs/oracle_layer3/ma_ema_per_asset_train/wf_robust_cells.parquet
     -- TRAIN walk-forward robustness flag per cell
  6. runs/oracle_layer3/SMART_DISCOVERY_ALL_TRAIN/SMA_cross/top_25.csv
     -- TRAIN top-25 SMA pairs (smart-candidate grid)
  7. runs/oracle_layer3/SMART_DISCOVERY_ALL_TRAIN/EMA_cross/top_25.csv
     -- same for EMA

OUTPUT: runs/audit/MA_EMA_PROFILE_2026_05_20/
  TOP_PERFORMERS.md             -- per-strategy pivot ranked tables
  PER_ASSET.md                  -- per-asset pivot tables (best pair per asset)
  PER_REGIME.md                 -- regime conditioning
  PER_BUCKET.md                 -- DNA bucket conditioning
  COMPLEMENTARITY.md            -- low-corr pairs to combine
  COMPARISON.md                 -- universal vs per-asset specialization
  RECOMMENDATION.md             -- best-config recommendation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
PERMUT_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_permutation"
PER_ASSET_DIR = ROOT / "runs" / "oracle_layer3" / "ma_ema_per_asset_train"
SMART_DIR = ROOT / "runs" / "oracle_layer3" / "SMART_DISCOVERY_ALL_TRAIN"
OUT = ROOT / "runs" / "audit" / "MA_EMA_PROFILE_2026_05_20"
OUT.mkdir(parents=True, exist_ok=True)

# Quality gates
MIN_N_SIGNALED = 30
MIN_SHARPE_PROXY = 0.05
MIN_HIT_RATE = 0.40


def load_all():
    summary = pl.read_parquet(PERMUT_DIR / "pair_summary.parquet").to_pandas()
    by_regime = pl.read_parquet(PERMUT_DIR / "pair_by_regime.parquet").to_pandas()
    by_bucket = pl.read_parquet(PERMUT_DIR / "pair_by_bucket.parquet").to_pandas()
    per_asset = pl.read_parquet(PER_ASSET_DIR / "pair_by_asset_cadence.parquet").to_pandas()
    wf_robust = pl.read_parquet(PER_ASSET_DIR / "wf_robust_cells.parquet").to_pandas()
    train_sma = pd.read_csv(SMART_DIR / "SMA_cross" / "top_25.csv")
    train_ema = pd.read_csv(SMART_DIR / "EMA_cross" / "top_25.csv")
    return summary, by_regime, by_bucket, per_asset, wf_robust, train_sma, train_ema


def filter_quality(df: pd.DataFrame, min_n_col: str = "n_signaled") -> pd.DataFrame:
    """Apply standard quality filters: enough events, non-degenerate, hit/Sharpe gates."""
    mask = (df[min_n_col] >= MIN_N_SIGNALED)
    if "degenerate_signal" in df.columns:
        mask &= ~df["degenerate_signal"]
    if "signal_quasi_constant" in df.columns:
        mask &= ~df["signal_quasi_constant"]
    return df[mask].copy()


def section_top_performers(summary: pd.DataFrame, train_sma: pd.DataFrame, train_ema: pd.DataFrame) -> str:
    """Per-strategy pivot: universal top performers across multiple metrics."""
    sma = summary[summary["ma_type"] == "SMA"].copy()
    ema = summary[summary["ma_type"] == "EMA"].copy()
    sma_q = filter_quality(sma)
    ema_q = filter_quality(ema)

    lines = ["# MA/EMA Top Performers — per-strategy pivot (universal, VAL window)\n"]
    lines.append("Quality gates applied: n_signaled >= 30, non-degenerate, non-quasi-constant.\n")
    lines.append(f"**SMA**: {len(sma)} total pairs / {len(sma_q)} pass quality gate.")
    lines.append(f"**EMA**: {len(ema)} total pairs / {len(ema_q)} pass quality gate.")
    lines.append("")

    # Note: VAL window is conditional on event-days (>1% cc moves). TRAIN smart-discovery
    # is conditional too but uses 14d-forward-return as the outcome.
    lines.append("> **Important framing**: the VAL pair_summary measures performance CONDITIONAL on")
    lines.append("> a >1% close-to-close event day occurring. Mean PnL is the average per-event return")
    lines.append("> WHEN the MA cross signal fires AND a 1%+ event happens. This is NOT a deployable")
    lines.append("> daily-return Sharpe. The TRAIN smart-discovery numbers (e.g., NAV +585% for SMA(3,5))")
    lines.append("> use 14-day forward return at signal date and are also conditional on event days.")
    lines.append("> Both are PER-EVENT characterisations; portfolio NAV depends on K-selection + exits.")
    lines.append("")

    # Ranked by sharpe_proxy
    lines.append("## A. SMA top-25 by Sharpe-proxy (event-conditional)")
    lines.append("")
    lines.append("| rank | (fast, slow) | n_signaled | mean PnL % | hit % | Sharpe-proxy | sum PnL % | sig balance |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    sma_top = sma_q.sort_values("sharpe_proxy", ascending=False).head(25)
    for i, (_, r) in enumerate(sma_top.iterrows(), 1):
        lines.append(f"| {i} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | "
                     f"{r['sharpe_proxy']:+.4f} | {r['sum_pnl_pct']:+.2f} | "
                     f"{r['signal_balance_pct']:.2f} |")
    lines.append("")
    lines.append("## B. EMA top-25 by Sharpe-proxy")
    lines.append("")
    lines.append("| rank | (fast, slow) | n_signaled | mean PnL % | hit % | Sharpe-proxy | sum PnL % | sig balance |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    ema_top = ema_q.sort_values("sharpe_proxy", ascending=False).head(25)
    for i, (_, r) in enumerate(ema_top.iterrows(), 1):
        lines.append(f"| {i} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | "
                     f"{r['sharpe_proxy']:+.4f} | {r['sum_pnl_pct']:+.2f} | "
                     f"{r['signal_balance_pct']:.2f} |")

    # Ranked by mean PnL
    lines.append("\n## C. SMA top-20 by mean per-event PnL (n>=30, hit>=40%)")
    lines.append("")
    lines.append("| rank | (fast, slow) | n_signaled | mean PnL % | hit % | Sharpe-proxy |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    sma_mq = sma_q[(sma_q["hit_rate"] >= MIN_HIT_RATE)].sort_values("mean_pnl_pct", ascending=False).head(20)
    for i, (_, r) in enumerate(sma_mq.iterrows(), 1):
        lines.append(f"| {i} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | {r['sharpe_proxy']:+.4f} |")

    # TRAIN smart-discovery top-25
    lines.append("\n## D. TRAIN smart-discovery (Fibonacci/golden-ratio set) — SMA top-15")
    lines.append("")
    lines.append("These use 14-day forward-return outcome at signal date. Different metric from VAL.")
    lines.append("")
    lines.append("| rank | config | n_events | best_exit | NAV % | mean % | hit % | Sharpe |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
    for _, r in train_sma.head(15).iterrows():
        lines.append(f"| {r['rank']} | {r['config']} | {r['n_events']} | {r['best_exit']} | "
                     f"{r['nav_pct']:+.2f} | {r['mean_pct']:+.3f} | {r['hit_pct']:.1f} | {r['sharpe']:+.4f} |")

    lines.append("\n## E. TRAIN smart-discovery — EMA top-15")
    lines.append("")
    lines.append("| rank | config | n_events | best_exit | NAV % | mean % | hit % | Sharpe |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
    for _, r in train_ema.head(15).iterrows():
        lines.append(f"| {r['rank']} | {r['config']} | {r['n_events']} | {r['best_exit']} | "
                     f"{r['nav_pct']:+.2f} | {r['mean_pct']:+.3f} | {r['hit_pct']:.1f} | {r['sharpe']:+.4f} |")

    # Convergence analysis: which (fast, slow) appear in BOTH VAL top-Sharpe AND TRAIN top-NAV?
    val_top = set((int(r["fast"]), int(r["slow"]), r["ma_type"]) for _, r in
                   pd.concat([sma_top, ema_top]).head(30).iterrows())
    train_top = set()
    for df_, mtype in ((train_sma, "SMA"), (train_ema, "EMA")):
        for _, r in df_.head(15).iterrows():
            cfg = r["config"].strip("()").split(",")
            try:
                f, s = int(cfg[0].strip()), int(cfg[1].strip())
                train_top.add((f, s, mtype))
            except Exception:
                pass
    convergent = val_top & train_top
    lines.append(f"\n## F. Convergence: pairs in BOTH VAL top-30 (by Sharpe) AND TRAIN top-15 (by NAV)")
    lines.append("")
    lines.append(f"Pairs convergent across both windows: **{len(convergent)}**")
    for (f, s, m) in sorted(convergent):
        lines.append(f"- {m}({f}, {s})")
    lines.append("")
    return "\n".join(lines)


def section_per_asset(per_asset: pd.DataFrame, wf_robust: pd.DataFrame) -> str:
    """Per-asset pivot: best MA pair for each asset."""
    pa_q = filter_quality(per_asset, "n_signaled")
    pa_q["abs_sharpe"] = pa_q["sharpe_proxy"].abs()
    # Best pair per (asset, cadence, ma_type)
    idx = pa_q.groupby(["asset", "cadence", "ma_type"])["sharpe_proxy"].idxmax()
    best_per = pa_q.loc[idx].sort_values("sharpe_proxy", ascending=False)

    lines = ["# MA/EMA Per-Asset Profile — asset-as-pivot (TRAIN window)\n"]
    lines.append(f"Total (asset, cadence, ma_type) cells with quality-passed best pair: **{len(best_per)}**.")
    lines.append(f"Universe: {best_per['asset'].nunique()} assets, "
                 f"{best_per['cadence'].nunique()} cadences.")
    lines.append("")
    lines.append("## A. Top-30 (asset, cadence, MA-pair) cells by Sharpe-proxy")
    lines.append("")
    lines.append("| asset | cadence | type | (fast, slow) | n_signaled | mean PnL % | hit % | Sharpe |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for _, r in best_per.head(30).iterrows():
        lines.append(f"| {r['asset']} | {r['cadence']} | {r['ma_type']} | "
                     f"({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | {r['sharpe_proxy']:+.4f} |")

    lines.append("\n## B. Top-20 assets by best 1d-cadence MA-cross Sharpe-proxy")
    lines.append("")
    daily = best_per[best_per["cadence"] == "1d"]
    if not daily.empty:
        # For each asset on 1d, take best-of-(SMA, EMA)
        daily_best = daily.loc[daily.groupby("asset")["sharpe_proxy"].idxmax()]
        daily_best = daily_best.sort_values("sharpe_proxy", ascending=False).head(20)
        lines.append("| asset | type | (fast, slow) | n_signaled | mean PnL % | hit % | Sharpe |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, r in daily_best.iterrows():
            lines.append(f"| {r['asset']} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                         f"{r['n_signaled']} | {r['mean_pnl_pct']:+.3f} | "
                         f"{r['hit_rate']*100:.1f} | {r['sharpe_proxy']:+.4f} |")
    else:
        lines.append("(no 1d cells in dataset)")

    # WF-robust cells (TRAIN walk-forward A/B/C all positive)
    if "wf_robust" in wf_robust.columns:
        robust = wf_robust[wf_robust["wf_robust"] == True]
        beats_drift = wf_robust[wf_robust.get("wf_robust_beats_drift", False) == True]
        lines.append(f"\n## C. WF-robust cells (positive in TRAIN A, B, C sub-folds)")
        lines.append("")
        lines.append(f"WF-robust cells (positive in all 3 TRAIN sub-folds): **{len(robust)}**")
        lines.append(f"WF-robust AND beats drift baseline: **{len(beats_drift)}**")
        if not beats_drift.empty:
            lines.append("\n### Top-20 WF-robust + beats-drift cells:")
            lines.append("")
            lines.append("| asset | cadence | type | (fast, slow) | total Sharpe (taker) | total mean (maker) |")
            lines.append("|---|---|---|---|---:|---:|")
            top_robust = beats_drift.sort_values("sharpe_taker_total", ascending=False).head(20)
            for _, r in top_robust.iterrows():
                lines.append(f"| {r['asset']} | {r['cadence']} | {r['ma_type']} | "
                             f"({r['fast']}, {r['slow']}) | "
                             f"{r['sharpe_taker_total']:+.4f} | "
                             f"{r['mean_pnl_maker_total']:+.4f}% |")

    # Cadence champion: for each asset, which cadence yields best Sharpe?
    lines.append(f"\n## D. Cadence champion per asset (which cadence gives best MA edge?)")
    lines.append("")
    if "cadence" in best_per.columns:
        # Group by asset, find the cadence with highest Sharpe
        cad_champ = best_per.loc[best_per.groupby("asset")["sharpe_proxy"].idxmax()].copy()
        cad_dist = cad_champ["cadence"].value_counts().to_dict()
        lines.append("Distribution of cadence-champion across assets:")
        lines.append("")
        for cad, n in sorted(cad_dist.items(), key=lambda x: -x[1]):
            pct = n / len(cad_champ) * 100
            lines.append(f"- **{cad}**: {n} assets ({pct:.1f}%)")
    return "\n".join(lines)


def section_per_regime(by_regime: pd.DataFrame) -> str:
    """Per-regime conditioning: what works in bull/chop/bear/crash."""
    br_q = filter_quality(by_regime)
    lines = ["# MA/EMA Per-Regime Profile (VAL window, conditional)\n"]
    regimes = sorted(br_q["btc_regime_30d"].unique())
    for reg in regimes:
        sub = br_q[br_q["btc_regime_30d"] == reg]
        sub_pos = sub[sub["mean_pnl_pct"] > 0]
        n_pairs_pass = len(sub)
        n_pos = len(sub_pos)
        lines.append(f"## Regime: **{reg}** ({n_pairs_pass} quality-passed pairs / {n_pos} positive-mean)")
        lines.append("")
        # Top 10 by Sharpe
        top = sub.sort_values("sharpe_proxy", ascending=False).head(10)
        lines.append("| type | (fast, slow) | n_sig | mean PnL % | hit % | Sharpe |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(f"| {r['ma_type']} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                         f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | "
                         f"{r['sharpe_proxy']:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def section_per_bucket(by_bucket: pd.DataFrame) -> str:
    """Per-DNA-bucket conditioning."""
    bb_q = filter_quality(by_bucket)
    lines = ["# MA/EMA Per-Bucket DNA Profile (VAL window)\n"]
    buckets = sorted(bb_q["bucket"].unique())
    for buc in buckets:
        sub = bb_q[bb_q["bucket"] == buc]
        n_pairs_pass = len(sub)
        n_pos = (sub["mean_pnl_pct"] > 0).sum()
        lines.append(f"## Bucket: **{buc}** ({n_pairs_pass} quality-passed pairs / {n_pos} positive-mean)")
        lines.append("")
        top = sub.sort_values("sharpe_proxy", ascending=False).head(10)
        lines.append("| type | (fast, slow) | n_sig | mean PnL % | hit % | Sharpe |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for _, r in top.iterrows():
            lines.append(f"| {r['ma_type']} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                         f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | "
                         f"{r['sharpe_proxy']:+.4f} |")
        lines.append("")
    return "\n".join(lines)


def section_complementarity(summary: pd.DataFrame, snapshot_path: Path) -> str:
    """Signal-correlation matrix between top-N MA pairs to identify complementary
    combinations (low correlation = combinable into ensemble)."""
    sma_q = filter_quality(summary[summary["ma_type"] == "SMA"])
    ema_q = filter_quality(summary[summary["ma_type"] == "EMA"])
    top_pairs = []
    for df_, mtype in ((sma_q, "SMA"), (ema_q, "EMA")):
        top = df_.sort_values("sharpe_proxy", ascending=False).head(10)
        for _, r in top.iterrows():
            top_pairs.append((mtype, int(r["fast"]), int(r["slow"])))

    lines = ["# MA/EMA Complementarity (signal correlation between top pairs)\n"]
    lines.append(f"Computing signal correlation matrix across {len(top_pairs)} top pairs "
                 f"({len([p for p in top_pairs if p[0]=='SMA'])} SMA + "
                 f"{len([p for p in top_pairs if p[0]=='EMA'])} EMA) from event_ma_snapshot.parquet.\n")

    try:
        snap = pl.read_parquet(snapshot_path).to_pandas()
    except Exception as e:
        lines.append(f"[ERROR] could not load snapshot: {e}")
        return "\n".join(lines)

    # Compute signal vector per pair (sign of fast - slow over events)
    sig_vectors = {}
    for mtype, f, s in top_pairs:
        col_f = f"{mtype}_{f}"
        col_s = f"{mtype}_{s}"
        if col_f not in snap.columns or col_s not in snap.columns:
            continue
        sig = np.sign(snap[col_f] - snap[col_s])
        # Only events where both MAs are non-NaN
        valid = snap[col_f].notna() & snap[col_s].notna()
        sig_vectors[(mtype, f, s)] = (sig.where(valid, 0).values, valid.values)

    if not sig_vectors:
        lines.append("[ERROR] no valid pair signals found")
        return "\n".join(lines)

    keys = list(sig_vectors.keys())
    n = len(keys)
    corr_matrix = np.full((n, n), np.nan)
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            sig1, val1 = sig_vectors[k1]
            sig2, val2 = sig_vectors[k2]
            both = val1 & val2
            if both.sum() < 30:
                continue
            corr_matrix[i, j] = np.corrcoef(sig1[both], sig2[both])[0, 1]

    # Find LOW-correlation pairs (potential complementary combo)
    pair_corrs = []
    for i in range(n):
        for j in range(i+1, n):
            if not np.isnan(corr_matrix[i, j]):
                pair_corrs.append((keys[i], keys[j], corr_matrix[i, j]))
    pair_corrs.sort(key=lambda x: x[2])  # ascending (lowest first)

    lines.append("## A. Most COMPLEMENTARY top-pair combinations (lowest signal correlation)\n")
    lines.append("| pair A | pair B | signal corr |")
    lines.append("|---|---|---:|")
    for (k1, k2, c) in pair_corrs[:20]:
        lines.append(f"| {k1[0]}({k1[1]}, {k1[2]}) | {k2[0]}({k2[1]}, {k2[2]}) | {c:+.3f} |")

    lines.append("\n## B. Most REDUNDANT top-pair combinations (highest signal correlation)\n")
    lines.append("| pair A | pair B | signal corr |")
    lines.append("|---|---|---:|")
    for (k1, k2, c) in pair_corrs[-15:][::-1]:
        lines.append(f"| {k1[0]}({k1[1]}, {k1[2]}) | {k2[0]}({k2[1]}, {k2[2]}) | {c:+.3f} |")

    # Greedy complementary set
    lines.append("\n## C. Greedy complementary set (target |corr|<=0.50)\n")
    selected = [keys[0]]  # seed with top-Sharpe pair
    pair_to_sharpe = {}
    for mtype, f, s in keys:
        row = summary[(summary["ma_type"] == mtype) & (summary["fast"] == f) & (summary["slow"] == s)]
        if not row.empty:
            pair_to_sharpe[(mtype, f, s)] = float(row.iloc[0]["sharpe_proxy"])

    # Add pairs whose max correlation with already-selected is below threshold
    THRESHOLD = 0.50
    for k in sorted(pair_to_sharpe.keys(), key=lambda x: -pair_to_sharpe.get(x, 0)):
        if k in selected:
            continue
        # Compute max corr with already selected
        max_c = 0
        for sk in selected:
            ki = keys.index(k); ski = keys.index(sk)
            if not np.isnan(corr_matrix[ki, ski]):
                max_c = max(max_c, abs(corr_matrix[ki, ski]))
        if max_c < THRESHOLD:
            selected.append(k)
        if len(selected) >= 8:
            break
    lines.append(f"Greedy set ({len(selected)} pairs, max pairwise |corr| < {THRESHOLD}):\n")
    for k in selected:
        s = pair_to_sharpe.get(k, 0)
        lines.append(f"- **{k[0]}({k[1]}, {k[2]})** Sharpe-proxy {s:+.4f}")
    return "\n".join(lines)


def section_comparison(summary: pd.DataFrame, per_asset: pd.DataFrame) -> str:
    """Universal-best vs per-asset specialization."""
    s_q = filter_quality(summary)
    pa_q = filter_quality(per_asset, "n_signaled")
    # Universal best (top Sharpe)
    uni_best = s_q.sort_values("sharpe_proxy", ascending=False).head(5)
    # Per-asset 1d cell sharpes
    per_asset_1d = pa_q[pa_q["cadence"] == "1d"]
    per_asset_best = per_asset_1d.loc[per_asset_1d.groupby("asset")["sharpe_proxy"].idxmax()] if not per_asset_1d.empty else pd.DataFrame()

    lines = ["# Universal vs Per-Asset Specialization\n"]
    lines.append("## A. Universal best (top-5 by Sharpe-proxy, VAL)\n")
    lines.append("| type | (fast, slow) | n_sig | mean PnL % | hit % | Sharpe-proxy |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for _, r in uni_best.iterrows():
        lines.append(f"| {r['ma_type']} | ({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} | {r['sharpe_proxy']:+.4f} |")

    if not per_asset_best.empty:
        lines.append(f"\n## B. Distribution of per-asset best Sharpe-proxy (1d cadence)\n")
        ss = per_asset_best["sharpe_proxy"]
        lines.append(f"- N assets: {len(per_asset_best)}")
        lines.append(f"- Median: {ss.median():+.4f}")
        lines.append(f"- Mean:   {ss.mean():+.4f}")
        lines.append(f"- 75th pct: {ss.quantile(0.75):+.4f}")
        lines.append(f"- 90th pct: {ss.quantile(0.90):+.4f}")
        lines.append(f"- Max:     {ss.max():+.4f}")
        uni_top = uni_best["sharpe_proxy"].iloc[0]
        lines.append(f"\n**Universal best Sharpe-proxy**: {uni_top:+.4f}")
        n_above = (per_asset_best["sharpe_proxy"] > uni_top).sum()
        lines.append(f"**N assets where per-asset specialization BEATS universal best**: "
                     f"{n_above} / {len(per_asset_best)} ({n_above/len(per_asset_best)*100:.1f}%)")

        lines.append(f"\n## C. Top-15 assets where per-asset specialization wins most\n")
        gap_df = per_asset_best.copy()
        gap_df["gap_vs_universal"] = gap_df["sharpe_proxy"] - uni_top
        gap_top = gap_df.sort_values("gap_vs_universal", ascending=False).head(15)
        lines.append("| asset | type | (fast, slow) | per-asset Sharpe | gap vs universal |")
        lines.append("|---|---|---|---:|---:|")
        for _, r in gap_top.iterrows():
            lines.append(f"| {r['asset']} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                         f"{r['sharpe_proxy']:+.4f} | {r['gap_vs_universal']:+.4f} |")
    return "\n".join(lines)


def section_recommendation(summary: pd.DataFrame, per_asset: pd.DataFrame,
                            train_sma: pd.DataFrame, train_ema: pd.DataFrame) -> str:
    """Best-config recommendation, both per-strategy and per-asset paths."""
    s_q = filter_quality(summary)
    pa_q = filter_quality(per_asset, "n_signaled")
    # Top universal Sharpe-proxy
    uni_top5 = s_q.sort_values("sharpe_proxy", ascending=False).head(5)
    # Top universal by NAV (TRAIN smart-discovery)
    train_top = pd.concat([train_sma.head(5).assign(ma_type="SMA"),
                            train_ema.head(5).assign(ma_type="EMA")])

    lines = ["# MA/EMA Best-Config Recommendation\n"]
    lines.append("## Decision matrix\n")
    lines.append("Both pivots give legitimate deployable configs. Use:\n")
    lines.append("- **Universal MA pair** when deploying a single-config sleeve across u100. "
                 "Easiest to operate; loses per-asset edge.")
    lines.append("- **Per-asset MA pair** when capital allocation can be split by asset. "
                 "Captures higher per-asset Sharpe; more operational complexity.")
    lines.append("- **Tiered** (universal core + per-asset overlay on TAO-like outliers): "
                 "best balance.\n")

    lines.append("## Path A — Universal MA (single config, easiest to deploy)\n")
    lines.append("Top-5 universal pairs by VAL Sharpe-proxy:\n")
    lines.append("| rank | config | n_sig | Sharpe-proxy | mean PnL % | hit % |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for i, (_, r) in enumerate(uni_top5.iterrows(), 1):
        lines.append(f"| {i} | {r['ma_type']}({r['fast']}, {r['slow']}) | {r['n_signaled']} | "
                     f"{r['sharpe_proxy']:+.4f} | {r['mean_pnl_pct']:+.3f} | {r['hit_rate']*100:.1f} |")

    lines.append("\nTop-5 universal pairs by TRAIN NAV (14d-fwd outcome):\n")
    lines.append("| rank | config | n_events | best_exit | NAV % | mean % | hit % | Sharpe |")
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|")
    for _, r in train_top.sort_values("nav_pct", ascending=False).head(8).iterrows():
        lines.append(f"| {r['rank']} | {r['ma_type']}{r['config']} | {r['n_events']} | "
                     f"{r['best_exit']} | {r['nav_pct']:+.2f} | {r['mean_pct']:+.3f} | "
                     f"{r['hit_pct']:.1f} | {r['sharpe']:+.4f} |")

    lines.append("\n## Path B — Per-asset specialist (one MA pair per asset, 1d cadence)\n")
    per_1d = pa_q[pa_q["cadence"] == "1d"]
    if not per_1d.empty:
        per_1d_best = per_1d.loc[per_1d.groupby("asset")["sharpe_proxy"].idxmax()]
        per_1d_best = per_1d_best.sort_values("sharpe_proxy", ascending=False)
        # Restrict to assets where per-asset Sharpe > universal-best by ≥ +0.05
        uni_top_sh = uni_top5.iloc[0]["sharpe_proxy"]
        worth = per_1d_best[per_1d_best["sharpe_proxy"] > uni_top_sh + 0.05]
        lines.append(f"Per-asset Sharpe-proxy distribution (1d cadence, n={len(per_1d_best)} assets):\n")
        lines.append(f"- Median: {per_1d_best['sharpe_proxy'].median():+.4f}")
        lines.append(f"- Top decile: {per_1d_best['sharpe_proxy'].quantile(0.90):+.4f}")
        lines.append(f"- N assets meaningfully beating universal best (Δ≥+0.05): "
                     f"**{len(worth)}/{len(per_1d_best)} ({len(worth)/len(per_1d_best)*100:.1f}%)**")
        lines.append("")
        lines.append("### Top-20 per-asset specialists (1d cadence) worth carving out:\n")
        lines.append("| asset | type | (fast, slow) | n_sig | Sharpe-proxy | hit % | mean PnL % |")
        lines.append("|---|---|---|---:|---:|---:|---:|")
        for _, r in worth.head(20).iterrows():
            lines.append(f"| {r['asset']} | {r['ma_type']} | ({r['fast']}, {r['slow']}) | "
                         f"{r['n_signaled']} | {r['sharpe_proxy']:+.4f} | "
                         f"{r['hit_rate']*100:.1f} | {r['mean_pnl_pct']:+.3f} |")

    lines.append("\n## Path C — Recommended TIERED deploy\n")
    lines.append("Best of both worlds:\n")
    lines.append("1. **Core sleeve**: top-2 universal Sharpe pairs from VAL "
                 f"({uni_top5.iloc[0]['ma_type']}({uni_top5.iloc[0]['fast']}, {uni_top5.iloc[0]['slow']}) + "
                 f"{uni_top5.iloc[1]['ma_type']}({uni_top5.iloc[1]['fast']}, {uni_top5.iloc[1]['slow']})), "
                 "fires across u100 universe via smart_discovery_17 ranker (confluence-K).")
    lines.append("2. **Per-asset overlay**: for each of the top-20 assets where per-asset specialization "
                 "beats universal by Δ≥+0.05 Sharpe, deploy an asset-specific MA pair as a secondary "
                 "sleeve. The pairs are listed in Path B above.")
    lines.append("3. **Exit policy**: use best_exit per TRAIN smart-discovery (E_14d for SMA(3,5) family; "
                 "S_setup_toxic for EMA(3,13) family). Confirm at deploy time via per-config exit study "
                 "in EXIT_FRAMEWORK_V2 once methodology fix lands.")

    lines.append("\n## Caveats / honest framing\n")
    lines.append("- All Sharpe-proxy values are EVENT-CONDITIONAL on a >1% close-to-close move "
                 "occurring (VAL panel uses both_cc def_type). DEPLOYABLE Sharpe is daily-aggregated "
                 "and will be lower because non-event days drag the mean toward 0.")
    lines.append("- TRAIN NAV (e.g., SMA(3,5) +585%) is arithmetic sum of per-event PnL × 4% notional; "
                 "**NOT a deploy estimate**. The relative ranking is valid.")
    lines.append("- Per-asset specialization carries higher overfit risk than universal pairs because "
                 "fewer events per cell. The WF-robust filter (positive in TRAIN A, B, C sub-folds) "
                 "addresses this; only WF-robust per-asset cells should be deployed.")
    lines.append("- For honest deploy NAV: feed the chosen config through honest_v2_simulator or v3 "
                 "paper_trade_replay with the patched confluence-only ranker.")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("="*78)
    print("MA/EMA FULL PROFILE — exhaustive per-strategy + per-asset + cross-cuts")
    print("="*78)

    summary, by_regime, by_bucket, per_asset, wf_robust, train_sma, train_ema = load_all()
    print(f"  pair_summary (VAL): {len(summary)} rows")
    print(f"  pair_by_regime (VAL): {len(by_regime)} rows")
    print(f"  pair_by_bucket (VAL): {len(by_bucket)} rows")
    print(f"  per_asset_cadence (TRAIN): {len(per_asset)} rows")
    print(f"  wf_robust_cells (TRAIN): {len(wf_robust)} rows")
    print(f"  TRAIN top-25 SMA: {len(train_sma)} rows")
    print(f"  TRAIN top-25 EMA: {len(train_ema)} rows")

    sections = [
        ("TOP_PERFORMERS", section_top_performers, (summary, train_sma, train_ema)),
        ("PER_ASSET", section_per_asset, (per_asset, wf_robust)),
        ("PER_REGIME", section_per_regime, (by_regime,)),
        ("PER_BUCKET", section_per_bucket, (by_bucket,)),
        ("COMPLEMENTARITY", section_complementarity, (summary, PERMUT_DIR / "event_ma_snapshot.parquet")),
        ("COMPARISON", section_comparison, (summary, per_asset)),
        ("RECOMMENDATION", section_recommendation, (summary, per_asset, train_sma, train_ema)),
    ]
    for name, fn, args in sections:
        try:
            md = fn(*args)
            (OUT / f"{name}.md").write_text(md, encoding="utf-8")
            print(f"  [OK] wrote {OUT.relative_to(ROOT)}/{name}.md")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback; traceback.print_exc()

    # Master index
    index = ["# MA/EMA Full Profile — Index (2026-05-20)\n",
             "Generated by `scripts/audit/ma_ema_full_profile.py`. ",
             "Source data: ma_ema_permutation/ + ma_ema_per_asset_train/ + SMART_DISCOVERY_ALL_TRAIN/",
             "",
             "## Sections",
             ""]
    for name, _, _ in sections:
        index.append(f"- [{name}]({name}.md)")
    (OUT / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"\n[OK] wrote {OUT}/INDEX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
