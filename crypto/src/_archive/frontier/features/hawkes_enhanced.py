"""Hawkes-enhanced features — lightweight derivatives of existing chimera features.

Rambaldi 2024 proposes full tick-level MLE of Hawkes branching ratio η. That
requires fitting millions of processes (~1-2 day sprint).

Lightweight alternative: derive features from existing `norm_hawkes_*` features
that approximate the branching-ratio effect:

    hawkes_intensity_accel_1d     d/dt change in norm_hawkes_intensity (1-day)
    hawkes_intensity_accel_7d     7-day momentum of intensity
    hawkes_intensity_z30          z-score of intensity vs 30d baseline
    hawkes_imbalance_z30          z-score of imbalance vs own history
    hawkes_imbalance_vol_7d       realized-vol of imbalance (regime indicator)
    hawkes_persistence_7d         7-day autocorrelation of imbalance (self-excitation proxy)
    hawkes_buy_over_sell_7d       7-day mean ratio of buy/sell intensity
    hawkes_regime_shift           1 if imbalance crosses zero with |magnitude| > 1

These are DAILY aggregates computed from chimera bar-level features. If IC
vs forward returns is material (>0.005), feed to xsec ranker as new features.

Input:
    data/processed/<asset>usdt_v50_chimera.parquet  (has norm_hawkes_* at bar level)

Output:
    data/frontier/hawkes_enh/hawkes_enh_daily.parquet
        columns: date, asset, + enhanced features above
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "processed" / "panels" / "daily"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "hawkes_enh_daily.parquet"


def build_asset_daily(fp: Path) -> pd.DataFrame | None:
    try:
        df = pl.read_parquet(fp, columns=[
            "timestamp", "close",
            "norm_hawkes_intensity", "norm_hawkes_buy_intensity",
            "norm_hawkes_sell_intensity", "norm_hawkes_imbalance",
        ]).to_pandas()
    except Exception:
        return None
    if len(df) < 500:
        return None
    df["date"] = pd.to_datetime(df["timestamp"].apply(lambda _t: _t // 1000 if _t >= 1e15 else _t), unit="ms").dt.normalize()
    # Daily aggregates
    d = df.groupby("date").agg({
        "close": "last",
        "norm_hawkes_intensity": "mean",
        "norm_hawkes_buy_intensity": "mean",
        "norm_hawkes_sell_intensity": "mean",
        "norm_hawkes_imbalance": "mean",
    }).reset_index()
    d["asset"] = fp.stem.replace("usdt_v50_chimera", "").upper()

    # Compute enhancements
    d["hawkes_intensity_accel_1d"] = d["norm_hawkes_intensity"].diff()
    d["hawkes_intensity_accel_7d"] = d["norm_hawkes_intensity"].diff(7)

    # 30-day rolling z-scores (shift-1 no leak)
    rm = d["norm_hawkes_intensity"].shift(1).rolling(30, min_periods=10).mean()
    rs = d["norm_hawkes_intensity"].shift(1).rolling(30, min_periods=10).std()
    d["hawkes_intensity_z30"] = (d["norm_hawkes_intensity"] - rm) / rs.replace(0, np.nan)

    rm = d["norm_hawkes_imbalance"].shift(1).rolling(30, min_periods=10).mean()
    rs = d["norm_hawkes_imbalance"].shift(1).rolling(30, min_periods=10).std()
    d["hawkes_imbalance_z30"] = (d["norm_hawkes_imbalance"] - rm) / rs.replace(0, np.nan)

    # Realized vol + persistence
    d["hawkes_imbalance_vol_7d"] = d["norm_hawkes_imbalance"].rolling(7, min_periods=3).std()
    d["hawkes_persistence_7d"] = (
        d["norm_hawkes_imbalance"].rolling(7, min_periods=3)
        .apply(lambda s: s.autocorr(lag=1) if len(s) > 1 else 0, raw=False)
    )

    # Buy/sell ratio (tame with eps)
    eps = 1e-6
    d["hawkes_buy_over_sell_7d"] = (
        d["norm_hawkes_buy_intensity"].rolling(7, min_periods=3).mean() /
        (d["norm_hawkes_sell_intensity"].rolling(7, min_periods=3).mean().abs() + eps)
    )

    # Regime shift: imbalance crosses zero with meaningful magnitude
    crossed = (np.sign(d["norm_hawkes_imbalance"]) != np.sign(d["norm_hawkes_imbalance"].shift(1))).astype(int)
    magnitude = d["norm_hawkes_imbalance"].abs() > 1
    d["hawkes_regime_shift"] = (crossed & magnitude).astype(int)

    # Forward returns for IC test ONLY (dropped before saving to avoid leakage)
    d["_fwd_ret_1d"] = d["close"].pct_change().shift(-1)
    d["_fwd_ret_3d"] = d["close"].pct_change(3).shift(-3)
    d["_fwd_ret_5d"] = d["close"].pct_change(5).shift(-5)

    return d


def main():
    fps = sorted(glob.glob(str(DATA / "*_chimera.parquet")))
    rows = []
    for fp in fps:
        d = build_asset_daily(Path(fp))
        if d is not None:
            rows.append(d)
    if not rows:
        print("[err] no data")
        return

    panel = pd.concat(rows, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["asset", "date"]).reset_index(drop=True)

    # IC test uses internal _fwd_ret_* columns then drops them
    new_feats = [
        "hawkes_intensity_accel_1d", "hawkes_intensity_accel_7d",
        "hawkes_intensity_z30", "hawkes_imbalance_z30",
        "hawkes_imbalance_vol_7d", "hawkes_persistence_7d",
        "hawkes_buy_over_sell_7d", "hawkes_regime_shift",
    ]
    print("\n[IC of enhanced features vs forward returns, ALIGNED 2025-01-01 -> 2026-04-16]:")
    aligned = panel[(panel["date"] >= "2025-01-01") & (panel["date"] < "2026-04-16")].copy()
    print(f"  aligned sample size: {len(aligned)}")
    for feat in new_feats:
        for h in ["_fwd_ret_1d", "_fwd_ret_3d", "_fwd_ret_5d"]:
            ic_per_day = aligned.groupby("date").apply(
                lambda g: g[feat].corr(g[h], method="spearman"), include_groups=False
            )
            ic_mean = ic_per_day.mean()
            ic_std = ic_per_day.std()
            t_stat = ic_mean / (ic_std / np.sqrt(len(ic_per_day))) if ic_std > 0 else 0
            print(f"  {feat:<35} vs {h}: IC {ic_mean:+.4f} +/- {ic_std:.4f} (t={t_stat:+.2f})")

    # LEAKAGE FIX: drop forward-return columns before saving parquet
    # (prior versions shipped ret_1d/ret_3d/ret_5d which caused massive
    # look-ahead bias when downstream scripts merged this panel as features)
    leakage_cols = [c for c in panel.columns if c.startswith("_fwd_")]
    if leakage_cols:
        panel = panel.drop(columns=leakage_cols)
        print(f"[hawkes_enh] dropped forward-return columns from parquet (leakage prevention): {leakage_cols}")
    panel.to_parquet(OUT_PATH, index=False)
    print(f"[hawkes_enh] saved: {OUT_PATH} ({len(panel)} rows, {panel['asset'].nunique()} assets)")


if __name__ == "__main__":
    main()
