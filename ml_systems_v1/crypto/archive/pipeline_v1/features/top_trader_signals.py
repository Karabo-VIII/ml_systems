"""Top-trader long/short ratio features (from data.binance.vision S3 metrics).

Inputs:
    data/frontier/metrics/s3_metrics_panel.parquet
        columns: date, asset, oi, oi_usd, top_acct_lsr, top_pos_lsr,
                 global_lsr, taker_lsr

Features (per asset per day):
    top_pos_lsr_z  (z-score vs rolling 30d baseline, shifted-1 for no-leak)
    top_pos_lsr_xsec_z  (cross-sectional z-score per day)
    top_pos_lsr_rank_pct  (rank in universe 0..1)
    top_pos_lsr_delta_1d  (d/dt; direction of flow change)
    top_pos_lsr_delta_3d

    top_acct_lsr_z, top_acct_lsr_delta_1d

    taker_lsr_z  (execution-bias z-score)
    taker_lsr_delta_1d

    # Smart-money vs retail divergence — key signal
    smart_vs_retail = top_pos_lsr - global_lsr  (when >0, smart are more long than retail)
    smart_vs_retail_z  (z-score vs own rolling 30d baseline)

    # Bucket signals
    smart_bullish = 1 if top_pos_lsr_z > 1.5 OR top_pos_lsr_xsec_z > 1.5
    smart_bearish = 1 if top_pos_lsr_z < -1.5 OR top_pos_lsr_xsec_z < -1.5
    smart_flipping_long = 1 if top_pos_lsr_delta_1d > +0.3 (big jump long)
    smart_extreme_long = 1 if top_pos_lsr > 3.0 (position ratio 3:1 long:short)
    smart_extreme_short = 1 if top_pos_lsr < 0.5 (position ratio 0.5:1, i.e., shorts dominate)
    smart_retail_divergence_bull = 1 if (top_pos_lsr_z > 1) & (global_lsr_z < -1)
       (smart long, retail short = potential squeeze setup)

Output:
    data/frontier/metrics/s3_features_long.parquet  (same long-format, + features)
"""
from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PANEL = ROOT / "data" / "processed" / "panels" / "daily" / "s3_metrics_panel.parquet"
OUT = ROOT / "data" / "processed" / "panels" / "daily" / "s3_features_long.parquet"


# 2026-05-22 oracle pipeline-progress closure: lazy phase_log helper with
# dual-import fallback (works whether src/ or src/pipeline/ is on sys.path).
def _pl(phase, message, **kw):
    try:
        from progress import phase_log
    except ImportError:
        from pipeline.progress import phase_log
    phase_log("s3_feat", phase, message, **kw)


def build() -> pd.DataFrame:
    if not PANEL.exists():
        # @browser B1: explicit dependency-missing error so build_panels
        # surfaces a useful message instead of FileNotFoundError stack.
        raise FileNotFoundError(
            f"top_trader_signals depends on {PANEL.name} (built by "
            f"binance_s3_metrics.py). Missing at {PANEL}. "
            f"Either: (a) build s3 panel first "
            f"(python src/frontier/ingest/binance_s3_metrics.py --universe u50), "
            f"or (b) run build_panels.py with the s3 stage NOT skipped "
            f"(remove --skip-existing or use --force).")
    df = pd.read_parquet(PANEL)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    print(f"[feat] input: {df.shape}, assets: {df['asset'].nunique()}")

    # Per-asset time-series features
    for col in ["top_pos_lsr", "top_acct_lsr", "global_lsr", "taker_lsr"]:
        if col not in df.columns:
            continue
        g = df.groupby("asset")[col]
        df[f"{col}_delta_1d"] = g.diff()
        df[f"{col}_delta_3d"] = g.diff(3)
        df[f"{col}_roll_mean"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
        df[f"{col}_roll_std"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).std())
        df[f"{col}_z"] = (df[col] - df[f"{col}_roll_mean"]) / df[f"{col}_roll_std"].replace(0, np.nan)
        df = df.drop(columns=[f"{col}_roll_mean", f"{col}_roll_std"])

    # Cross-sectional rank + z per day (use transform to avoid apply/include_groups mess)
    xsec_cols_new = []
    for col in ["top_pos_lsr", "top_acct_lsr", "global_lsr", "taker_lsr"]:
        if col not in df.columns:
            continue
        df[f"{col}_xsec_z"] = df.groupby("date")[col].transform(lambda s: (s - s.mean()) / (s.std() if s.std() > 0 else 1.0))
        df[f"{col}_rank_pct"] = df.groupby("date")[col].rank(pct=True)

    # Derived signals
    df["smart_vs_retail"] = df["top_pos_lsr"] - df["global_lsr"]
    g = df.groupby("asset")["smart_vs_retail"]
    df["smart_vs_retail_mean"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).mean())
    df["smart_vs_retail_std"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=10).std())
    df["smart_vs_retail_z"] = (df["smart_vs_retail"] - df["smart_vs_retail_mean"]) / df["smart_vs_retail_std"].replace(0, np.nan)
    df = df.drop(columns=["smart_vs_retail_mean", "smart_vs_retail_std"])

    # Bucket/event flags
    df["smart_bullish"] = (
        (df["top_pos_lsr_z"] > 1.5) | (df["top_pos_lsr_xsec_z"] > 1.5)
    ).astype(int)
    df["smart_bearish"] = (
        (df["top_pos_lsr_z"] < -1.5) | (df["top_pos_lsr_xsec_z"] < -1.5)
    ).astype(int)
    df["smart_flipping_long"] = (df["top_pos_lsr_delta_1d"] > 0.3).astype(int)
    df["smart_flipping_short"] = (df["top_pos_lsr_delta_1d"] < -0.3).astype(int)
    df["smart_extreme_long"] = (df["top_pos_lsr"] > 3.0).astype(int)
    df["smart_extreme_short"] = (df["top_pos_lsr"] < 0.5).astype(int)
    df["smart_retail_diverge_bull"] = (
        (df["top_pos_lsr_z"] > 1.0) & (df["global_lsr_z"] < -1.0)
    ).astype(int)
    df["smart_retail_diverge_bear"] = (
        (df["top_pos_lsr_z"] < -1.0) & (df["global_lsr_z"] > 1.0)
    ).astype(int)

    return df


def main():
    # G-AUDIT-024: argparse stub. Previously had NO argparse, so any flag
    # passed by refresh.py (--force, --universe, --asset) was silently
    # swallowed by Python. The aggregator always rebuilds (which is what
    # --force semantically means), so --force is a no-op.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="No-op stub. Aggregator always rebuilds the panel.")
    ap.add_argument("--universe", default=None,
                    help="No-op stub (panel is multi-asset).")
    ap.add_argument("--asset", default=None,
                    help="No-op stub (panel is multi-asset).")
    # parse_args (not parse_known_args): the no-op flags are declared above, so a
    # real flag typo now surfaces instead of being silently swallowed.
    ap.parse_args()
    df = build()
    # G-AUDIT-020: atomic-tmp-rename + column-name verify (RED TEAM contract)
    _tmp = OUT.with_suffix(".parquet.tmp")
    df.to_parquet(_tmp, index=False)
    import pyarrow.parquet as _pq
    _written = set(_pq.read_schema(_tmp).names)
    if "date" not in _written or "asset" not in _written:
        _tmp.unlink(missing_ok=True)
        raise ValueError(f"top_trader: missing date/asset cols (got {sorted(_written)[:5]}...)")
    os.replace(str(_tmp), str(OUT))  # atomic overwrite (Windows-safe)
    print(f"[feat] saved: {OUT} ({len(df)} rows, {len(df.columns)} cols)")
    print("\n[signal incidence]:")
    for s in ["smart_bullish", "smart_bearish", "smart_flipping_long", "smart_flipping_short",
              "smart_extreme_long", "smart_extreme_short", "smart_retail_diverge_bull",
              "smart_retail_diverge_bear"]:
        if s in df.columns:
            n = int(df[s].sum())
            print(f"  {s}: {n} asset-days ({100*n/len(df):.2f}%)")


if __name__ == "__main__":
    main()
